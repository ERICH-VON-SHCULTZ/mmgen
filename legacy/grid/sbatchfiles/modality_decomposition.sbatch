#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=modality_decomp
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/modality_decomposition.out

# ============================================================================================
# Empirically test whether MD-RQ-VAE codebooks are modality-decomposed and a chain RQ-VAE's are not.
# Pure offline analysis: no training, no generation, no downstream model. Loads the MD checkpoint
# (codebooks C_f/C_t/C_i + decoders D_t/D_i) with weights_only=False, the two SID files (column ==
# item_id layout), and the SigLIP2-so400m 1152-d text/image features (24.9GB npy, memmapped).
#
#   TEST 1a  MD decoder intervention: 3x2 cosine-drop table (subbed code x damaged modality) + CI.
#            (the text/image cross-drop is 0 BY CONSTRUCTION -- the report flags this; the honest
#             signal is the own-modality drops and the shared sub, plus TEST 1b/2/3.)
#   TEST 1b  representation-level analogue on BOTH tokenizers (fair, non-architectural).
#   TEST 2   additivity: D_t(e_f+e_t) vs D_t(e_f) vs D_t(e_t); shared-code modality balance.
#   TEST 3   codebook semantic purity: per-code SigLIP text-spread vs image-spread + scatter plots
#            (saved to the out dir -- intended for the paper), MD C_f/C_t/C_i and chain level0/1/2.
#
# Fixed random sample (200k items with both modalities), same sample for every measurement; the
# sample size + seed are printed. Codebook shapes are read from the checkpoint and reported (not
# assumed); chain widths are read from its SID file (do NOT assume the MD [2048,1024,1024] layout).
# Report + figures -> /scratch/yw8866/rec-tmall/modality_decomp/ (and the .out).
# ============================================================================================

MD_CKPT=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
CHAIN_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
TXT_NPY=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384.npy
TXT_IDS=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384_ids.pkl
IMG_NPY=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384.npy
IMG_IDS=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384_ids.pkl
OUT=/scratch/yw8866/rec-tmall/modality_decomp

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

python /scratch/yw8866/rec-tmall/analyze_modality_decomposition.py \
    --md_ckpt $MD_CKPT --md_sid $MD_SID --chain_sid $CHAIN_SID \
    --text_npy $TXT_NPY --text_ids $TXT_IDS \
    --image_npy $IMG_NPY --image_ids $IMG_IDS \
    --sample 200000 --seed 0 --min_items 50 --out_dir $OUT --device cuda
"
