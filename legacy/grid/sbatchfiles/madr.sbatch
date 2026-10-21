#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=madr
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/madr.out

# ============================================================================================
# MADR: Modality-Aware Dual-channel Ranker. Rescores the frozen MFD pool with TWO structurally
# separated channels (text, image), each with its own cross-attention over the FULL encoder memory
# and its own MLP, fused by a head that can express cross-modal agreement. Offline on existing dumps;
# the main model is NOT retrained. Select on VAL, report on TEST.
#
# Trains 4 variants in one job: PRIMARY (both channels), text-only, image-only (channel ablation),
# and random-views (view-substitution ablation). Channel diagnostics on the primary model:
# corr(s_t,s_i) [>0.9 => collapse], q_t-vs-q_i attention JS divergence, and SigLIP2 GT-vs-displacer
# similarity. Asserts test split (83,005; pv/fav/cart/buy 77,833/2,042/1,554/1,576) and coverage
# invariance (0.2615). Mechanism metrics: P(rank<10|cov) base 0.7897 / chain 0.8581; rank10-11 gap
# base 0.1485 / chain 0.3213; #within +/-0.5 of GT base 32.59 / chain 22.09.
# ============================================================================================

MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
TXT_NPY=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384.npy
TXT_IDS=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384_ids.pkl
IMG_NPY=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384.npy
IMG_IDS=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384_ids.pkl

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

python /scratch/yw8866/rec-tmall/train_madr.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --md_sid $MD_SID --text_npy $TXT_NPY --text_ids $TXT_IDS --image_npy $IMG_NPY --image_ids $IMG_IDS \
    --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 \
    --epochs 25 --abl_epochs 15 --patience 4 --batch 32 --eval_batch 16 --seed 0 \
    --out_dir /scratch/yw8866/rec-tmall/madr
"
