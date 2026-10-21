#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=14:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_chain_rescore
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_chain_rescore.out

# ============================================================================================
# CHAIN-CONDITIONAL RESCORING on top of the frozen MFD + PMQ + Q_f pool. The main model is NOT
# retrained; MFD's factorized pool is left untouched (coverage is invariant) and only RE-RANKED
# using logP(i|f,t) -- the chain edge MFD drops for parallel generation but which is one batched
# forward once the pool is fixed. Mirrors mfd_rerank_best.sbatch for the dump stage.
#
#   STAGE A  feasibility on the existing MFD pool dump (mfd_pool_cmp_dump.pt) -- prints GO/NO-GO.
#   STAGE B  a small cross-attention head P(i|f,t) over the encoder memory + lifted (f,t) codes,
#            trained TEACHER-FORCED on GT (f,t)->i (everything else frozen; scheduled sampling off).
#   STAGE C  learned fusion over logP(b),logP(f),logP(t|f),logP(i|f),logP(i|f,t) fit on the eval
#            split, vs the hand-set baseline (w_b=1, tau=1) and a chain-ablated fusion.
#
# Reuses the SAME train/val/test pool dumps as the best reranker (created here if missing, guarded;
# each carries +hist_ids + enc_mem via mfd_dump_memory). TRAIN on training split, SELECT on eval,
# REPORT on testing. Test split counts asserted (83,005; pv/fav/cart/buy 77,833/2,042/1,554/1,576).
# Coverage@pool asserted score-invariant (ref 0.2615). P(rank<10|covered): baseline 0.7805,
# chain reference ceiling 0.8581 -- the mechanism metric to move.
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt   # '='-free symlink for Hydra (shared)
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

POOL_CMP=/scratch/yw8866/rec-tmall/mfd_pool_cmp_dump.pt                  # existing -> STAGE A
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt   # training split -> TRAIN
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt            # evaluation split -> VAL/eval
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt          # testing split -> TEST

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f   # noglob: keep codebook_sizes=[..] literal
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

MFD_DUMP=\"experiment=tiger_mb_mfd_flat train=false test=true ckpt_path=$LINK data_dir=$DATA \
semantic_id_path=$SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
+model.mfd_dump_memory=true mfd_audit_dump=true +trainer.limit_test_batches=6000\"

# ---- STAGE 1a: EVALUATION split -> VAL (guarded) ----
if [ ! -f $VAL_DUMP ]; then
  echo '=== dumping EVALUATION split (VAL, +hist_ids+enc_mem) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_DUMP \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation \
      mfd_audit_path=$VAL_DUMP
else echo '=== VAL_DUMP exists, skipping ==='; fi

# ---- STAGE 1b: TESTING split -> TEST (guarded) ----
if [ ! -f $TEST_DUMP ]; then
  echo '=== dumping TESTING split (TEST, +hist_ids+enc_mem) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_DUMP \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=200000 mfd_audit_path=$TEST_DUMP
else echo '=== TEST_DUMP exists, skipping ==='; fi

# ---- STAGE 1c: TRAINING split -> TRAIN (guarded) ----
if [ ! -f $TRAIN_DUMP ]; then
  echo '=== dumping TRAINING split (TRAIN, +hist_ids+enc_mem) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_DUMP \
      mfd_audit_max_events=250000 mfd_audit_pv_cap=300000 \
      data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training \
      mfd_audit_path=$TRAIN_DUMP
else echo '=== TRAIN_DUMP exists, skipping ==='; fi

# ---- STAGES A/B/C + eval ----
echo '=== chain-conditional rescoring: Stage A feasibility, Stage B head, Stage C fusion, eval ==='
python /scratch/yw8866/rec-tmall/chain_rescore.py \
    --pool_cmp_dump $POOL_CMP \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --wb 1.0 --head_layers 1 --heads 4 --dropout 0.1 \
    --head_lr 1e-3 --wd 1e-3 --head_epochs 30 --head_patience 4 --ss_prob 0.0 \
    --batch 64 --eval_batch 16 --seed 0 \
    --out_dir /scratch/yw8866/rec-tmall/chain_rescore
"
