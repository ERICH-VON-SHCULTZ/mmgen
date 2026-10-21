#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train_tiger_mb
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tiger_mb_baseline.out

# ============================================================================================
# TRAIN-FROM-SCRATCH the CHAIN RQ-VAE baseline row: multi-behavior TIGER (standard sequential
# decoder, one token per SID level, level_k | all previous levels) on the tiger_1024 chain
# RQ-VAE semantic IDs. This is the "clean" recipe that produced the reference checkpoint
# 2026-07-10/09-06-58 -- NO focal loss, NO item-PE/modality-TE, NO behavior weighting, NO
# semantic rerank. All of those already default to OFF in tiger_mb_train_flat, so the clean
# recipe is just the required (semantic_id_path, num_hierarchies) plus the codebook geometry.
#
# codebook_init_path=null  -> random-init SID-token codebooks (chain RQ-VAE, NOT MD-RQ-VAE init).
# codebook_sizes=[1024,1024,1024] -> the TRUE tiger_1024 per-level widths. The SID was generated
#   with codebook_width=1024 applied to every RQ level (.hydra/overrides.yaml of the tiger_1024
#   inference run), so all three levels are 1024 wide. NOTE: the config default is [2048,1024,1024]
#   (the MD-RQ-VAE level-0 width) and the earlier reference checkpoint 2026-07-10 inherited that
#   default -> it over-allocated level 0 to 2048. That runs and gives ~equivalent metrics (level-0
#   codes only ever occupy 0..1023, so rows 1024..2047 are dead), but the CORRECT geometry is 1024.
#   This override MUST be set (it differs from the default). Train and auto-test use the same value,
#   so the run is self-consistent.
#
# train=True + test=True (defaults) -> after training, the SAME evaluator
# (MultiBehaviorSIDRetrievalEvaluator) auto-runs on tfrecords_mb_userdrop/testing (83,005 events)
# and logs the caliber-aligned row: t2/t3 recall@{5,10} + ndcg@{5,10}, t3_behavior_accuracy, and
# per-behavior behavior-head precision/recall/f1 -- t3 = union/behavior-marginalized (w_b=1),
# t2 = routed to GT behavior. top_k_for_generation=10 (decoder-states/item = 1+3*10 = 31), the
# same free-beam budget the other baseline rows (MDRQVAE+MB, MBGen) were measured at.
# (recall@20 / per-behavior recall / the 808-4008 pool points come from chain_baseline_row.sbatch.)
# ============================================================================================

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py310
set -f   # noglob: keep codebook_sizes=[..] literal
export HF_HOME=/scratch/yw8866/hf_cache
export WANDB_API_KEY='REDACTED_ROTATE_ME'
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat \
    train=true test=true \
    data_dir=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop \
    semantic_id_path=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt \
    num_hierarchies=3 \
    codebook_init_path=null codebook_sizes=[1024,1024,1024]
"
