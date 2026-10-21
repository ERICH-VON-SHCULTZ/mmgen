#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=36:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=matched_rerank
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/matched_rerank.out

# ============================================================================================
# MATCHED HEAD-TO-HEAD: the SAME listwise reranker, trained the SAME way, on the MFD pool and the
# chain pool. Both generative models stay FROZEN; reranker training is offline on dumps.
#
# Matching (the comparison is worthless if these slip):
#   * equal decoder compute: 808 decoder-state evals/item on both arms
#       MFD   K_f=200 K_t=200 K_i=50 -> nb*(2+K_f) = 4*202  = 808
#       chain beam=100               -> 4 + nb*(1+2*Bw) = 4+4*201 = 808   (printed per arm)
#   * identical reranker: same arch/hyperparams/features (views + hard + soft history match),
#     same listwise CE, same optimizer/schedule/early-stopping, same seed.
#   * identical protocol: fit on train split, select on VAL, report on TEST.
#   * equal training data: chain covers less, so MFD is ALSO run capped to chain's covered count.
#     If natural and subsampled disagree, the subsampled one is the fair number.
#   * candidate representation: FIXED RANDOM VECTORS on both arms (--geom random), sized to each
#     arm's own SID widths READ from its semantic_ids.pt (chain is 1024/1024/1024 -- never assumed;
#     a prior chain analysis was invalidated by hard-coding MD's 2048/1024/1024). Widths + a
#     per-level "max code < width" assert are printed in the run header. Prior ablation found random
#     statistically identical to MD three-view (0.2151 vs 0.2148), so this favours neither side.
#   * K_pool=500 on both (recall@10 is K_pool-invariant).
#
# The chain pool dumps must carry what the reranker consumes (encoder memory, history SIDs, and the
# per-candidate component log-probs logP(l0|b)/logP(l1|b,l0)/logP(l2|b,l0,l1)) -- the chain audit was
# extended to dump exactly the MFD schema, so they are regenerated here.
#
# NOTE on chain model construction: the reference chain checkpoint was trained with the config
# default codebook_sizes=[2048,1024,1024] (over-allocated level 0), so the model MUST be built that
# way to load; the SID DATA widths are 1024/1024/1024 and are what the reranker geometry uses.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
CH_CKPT=/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt
CH_LINK=/scratch/yw8866/rec-tmall/_mr_chain.ckpt; ln -sf "$CH_CKPT" "$CH_LINK"

OUT=/scratch/yw8866/rec-tmall/matched_rerank; mkdir -p $OUT
# MFD arm: the existing 808-state dumps (K_f=200,K_t=200,K_i=50) with memory + history
MFD_TR=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
MFD_VA=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
MFD_TE=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
# chain arm: regenerated at beam=100 WITH memory/history/components
CH_TR=$OUT/chain808_train_dump.pt
CH_VA=$OUT/chain808_val_dump.pt
CH_TE=$OUT/chain808_test_dump.pt

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

CH_COMMON=\"experiment=tiger_mb_train_flat train=false test=true ckpt_path=$CH_LINK data_dir=$DATA \
semantic_id_path=$TG_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
+model.chain_pool_audit=true +model.chain_pool_beam=100 +model.chain_pool_size=500 \
+model.chain_dump_memory=true +model.chain_pool_pv_cap=300000 \
data_loading.test_dataloader_config.dataloader.batch_size_per_device=16 +trainer.limit_test_batches=6000\"

# ---------- chain pool dumps at beam=100 (808 states), same splits as MFD ----------
if [ ! -f $CH_TE ]; then
  echo '=== chain@808 TEST dump ==='
  python /scratch/yw8866/GRID/src/train.py \$CH_COMMON \
      +model.chain_pool_max_events=200000 +model.chain_pool_path=$CH_TE
else echo '=== chain TEST dump exists ==='; fi

if [ ! -f $CH_VA ]; then
  echo '=== chain@808 VAL dump ==='
  python /scratch/yw8866/GRID/src/train.py \$CH_COMMON \
      +model.chain_pool_max_events=200000 +model.chain_pool_path=$CH_VA \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation
else echo '=== chain VAL dump exists ==='; fi

if [ ! -f $CH_TR ]; then
  echo '=== chain@808 TRAIN dump ==='
  python /scratch/yw8866/GRID/src/train.py \$CH_COMMON \
      +model.chain_pool_max_events=250000 +model.chain_pool_path=$CH_TR \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training
else echo '=== chain TRAIN dump exists ==='; fi

# ---------- identical reranker hyperparameters on BOTH arms ----------
RR=\"--cand_repr md3 --geom random --use_hard --use_soft --layers 2 --heads 4 --dropout 0.3 \
--wd 1e-3 --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 --batch 32 --eval_batch 64 --seed 0\"

echo '################ ARM: MFD (natural) ################'
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \$RR \
    --train_dump $MFD_TR --val_dump $MFD_VA --test_dump $MFD_TE --sid_path $MD_SID \
    --save $OUT/rr_mfd_natural.pt --save_ranks $OUT/ranks_mfd_natural.pt

echo '################ ARM: chain (natural) ################'
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \$RR \
    --train_dump $CH_TR --val_dump $CH_VA --test_dump $CH_TE --sid_path $TG_SID \
    --save $OUT/rr_chain_natural.pt --save_ranks $OUT/ranks_chain_natural.pt

# ---------- equal-training-data setting: cap MFD to chain's covered train count ----------
NSUB=\$(python - <<'PY'
import torch
d = torch.load('$CH_TR', map_location='cpu', weights_only=False); c = d['cols']
N, nb, Kp = c['pool_f'].shape
fl = lambda x: x.reshape(N, nb*Kp)
br = torch.arange(nb).repeat_interleave(Kp).view(1,-1).expand(N,-1)
g = ((br == c['gt_b'].long()[:,None]) & (fl(c['pool_f']).long() == c['gt_f'].long()[:,None])
     & (fl(c['pool_t']).long() == c['gt_t'].long()[:,None]) & (fl(c['pool_i']).long() == c['gt_i'].long()[:,None])
     & (fl(c['pool_sidscore']).float() > -1e30))
print(int(g.any(1).sum()))
PY
)
echo \"chain covered TRAIN events = \$NSUB  -> capping MFD to the same count\"

echo '################ ARM: MFD (subsampled to chain train size) ################'
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \$RR \
    --train_dump $MFD_TR --val_dump $MFD_VA --test_dump $MFD_TE --sid_path $MD_SID \
    --max_train_events \$NSUB \
    --save $OUT/rr_mfd_sub.pt --save_ranks $OUT/ranks_mfd_sub.pt

echo '################ MATCHED COMPARISON ################'
python /scratch/yw8866/rec-tmall/matched_rerank_compare.py \
    --mfd_test $MFD_TE --chain_test $CH_TE \
    --mfd_ckpt $OUT/rr_mfd_natural.pt --chain_ckpt $OUT/rr_chain_natural.pt \
    --mfd_ckpt_sub $OUT/rr_mfd_sub.pt --chain_ckpt_sub $OUT/rr_chain_natural.pt \
    --wb 1.0 --out_dir $OUT
"
