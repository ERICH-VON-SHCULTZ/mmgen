#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=10:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=reeval_t2fix
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/reeval_baselines_t2fix.out

# ============================================================================================
# RE-EVALUATE the 3 baselines after the t2 beam-collapse FIX in generate_conditioned.
#
# The fix (tiger_generation_model.py generate_conditioned): the conditioned t2 beam was seeded
# with all top_k beams IDENTICAL + uniform marginal_log_prob, so _beam_search_one_step collapsed
# them to top_k copies of the single greedy SID -> the t2 pool had one distinct candidate and
# recall@5 == recall@10 == ndcg@5 == ndcg@10 (a degenerate recall@1). Now only beam 0 is seeded
# "alive" (prob 1), so the FIRST SID step expands it into top_k DISTINCT codes (like
# generate_multibehavior), giving a genuine ranked t2 pool. The fix lives in the shared base class
# SemanticIDMultiBehaviorEncoderDecoder, so it reaches BOTH the plain chain baseline AND the PBA
# baselines (PBA inherits generate_conditioned; only MFD overrides it and was never collapsed).
#
# This is an INFERENCE-time fix: no retraining. Each baseline's EXISTING checkpoint is re-tested
# with the fixed code. Architecture-affecting flags are replicated exactly so the checkpoint loads
# cleanly; codebook_init_path is nulled (the trained ckpt already carries the codebook weights, so
# re-loading the 50GB MD-RQ-VAE init source is unnecessary and would only be overwritten).
#
# Baselines (checkpoint -> OLD collapsed t2_recall@10, all four t2 metrics were ~equal):
#   1  MDRQVAE + MB + Focal      2026-07-04/18-55-09     step=026250  OLD t2@10 = 0.15565
#   2  MBGen(PBA) + MBGen SID    2026-07-14/23-35-46     step=011250  OLD t2@10 = 0.15688
#   3  MBGen(PBA) + MDRQVAE SID  2026-07-16/10-25-02     step=018750  OLD t2@10 = 0.15744
#   4  MB (chain) + RQ-VAE SID   2026-08-15/mb_rqvae_baseline step=07500  OLD t2@10 = 0.15729
#      (the plain multi-behavior chain TIGER on the tiger_1024 chain RQ-VAE SID, trained fresh at
#       the CORRECT codebook_sizes=[1024,1024,1024]; no focal -> plain state_dict load.)
# After the fix expect: t2_recall@10 >= t2_recall@5, and t2_ndcg < t2_recall (proper rank spread);
# t2_recall@10 should RISE vs the collapsed recall@1. t3 metrics are unchanged (t3 never collapsed).
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MDSID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MBGENSID=/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt
TGSID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt

CK1=/scratch/yw8866/logs/train/runs/2026-07-04/18-55-09/checkpoints/checkpoint_epoch=000_step=026250.ckpt
CK2=/scratch/yw8866/logs/train/runs/2026-07-14/23-35-46/checkpoints/checkpoint_epoch=000_step=011250.ckpt
CK3=/scratch/yw8866/logs/train/runs/2026-07-16/10-25-02/checkpoints/checkpoint_epoch=000_step=018750.ckpt
CK4=/scratch/yw8866/logs/train/runs/2026-08-15/mb_rqvae_baseline/checkpoints/checkpoint_epoch=000_step=007500.ckpt
# symlink so hydra doesn't choke on the '=' in the checkpoint filename
L1=/scratch/yw8866/rec-tmall/_reeval_ck1.ckpt; ln -sf "$CK1" "$L1"
L2=/scratch/yw8866/rec-tmall/_reeval_ck2.ckpt; ln -sf "$CK2" "$L2"
L3=/scratch/yw8866/rec-tmall/_reeval_ck3.ckpt; ln -sf "$CK3" "$L3"
L4=/scratch/yw8866/rec-tmall/_reeval_ck4.ckpt; ln -sf "$CK4" "$L4"

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

echo '################################################################################'
echo '# BASELINE 1/3 : MDRQVAE + MB + Focal (chain)  [OLD collapsed t2@10 = 0.15565]  #'
echo '################################################################################'
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat train=false test=true \
    ckpt_path=$L1 data_dir=$DATA semantic_id_path=$MDSID num_hierarchies=3 \
    codebook_init_path=null codebook_sizes=[2048,1024,1024] \
    focal_behavior=True focal_gamma=1.0 focal_alpha=[1.0,0.5,0.5,2.0]
    # focal flags are REQUIRED at eval: the checkpoint stores the FocalLoss buffer
    # behavior_loss_function.alpha, so the model must be built with focal (else strict
    # state_dict load fails with 'Unexpected key behavior_loss_function.alpha').

echo '################################################################################'
echo '# BASELINE 2/3 : MBGen(PBA) + MBGen SID  [OLD collapsed t2@10 = 0.15688]        #'
echo '################################################################################'
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_pba_flat train=false test=true \
    ckpt_path=$L2 data_dir=$DATA semantic_id_path=$MBGENSID num_hierarchies=3 \
    codebook_init_path=null \
    use_bmtv=False use_item_pe=False focal_behavior=False behavior_weighted_sid=False \
    use_semantic_align=False use_semantic_rerank=False probe_sid_distance=False

echo '################################################################################'
echo '# BASELINE 3/3 : MBGen(PBA) + MDRQVAE SID  [OLD collapsed t2@10 = 0.15744]      #'
echo '################################################################################'
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_pba_flat train=false test=true \
    ckpt_path=$L3 data_dir=$DATA semantic_id_path=$MDSID num_hierarchies=3 \
    codebook_init_path=null codebook_sizes=[2048,1024,1024] \
    use_bmtv=False use_item_pe=False focal_behavior=False behavior_weighted_sid=False \
    use_semantic_align=False use_semantic_rerank=False probe_sid_distance=False

echo '################################################################################'
echo '# BASELINE 4/4 : MB (chain) + RQ-VAE SID  [OLD collapsed t2@10 = 0.15729]       #'
echo '################################################################################'
python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat train=false test=true \
    ckpt_path=$L4 data_dir=$DATA semantic_id_path=$TGSID num_hierarchies=3 \
    codebook_init_path=null codebook_sizes=[1024,1024,1024]

echo '################################################################################'
echo '# READ-OUT: for each run compare test/t2_recall_5, test/t2_recall_10,           #'
echo '#   test/t2_ndcg_5, test/t2_ndcg_10. They should now DIFFER (proper ranking):    #'
echo '#   recall@10 >= recall@5 and ndcg < recall. If they are still all equal, the    #'
echo '#   fix did not take -- check that this src tree is the one on the PYTHONPATH.    #'
echo '#   t3_* metrics should be unchanged from the original runs (t3 never collapsed).#'
echo '################################################################################'
"