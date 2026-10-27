#!/bin/bash
# MD-RQ-VAE tokenizer attempt: train -> SIDs -> diagnostics (one GPU, short).
# Submit: sbatch --export=ALL,D=Instruments,NAME=a01_122,TAG=md122_a01,ARGS="--n_shared 1 --n_modal 2" sbatch/md/tok_attempt.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon,a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_md_tok
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
OUT=/scratch/yw8866/mmgen_runs/md/$D/tok/$NAME; mkdir -p $OUT
echo "job $SLURM_JOB_ID on $(hostname) D=$D NAME=$NAME TAG=$TAG ARGS=$ARGS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $REPO/macrec/cross_index
python -u main_mdrqvae.py --dataset $D --out_dir $OUT $ARGS > $OUT/train.out 2>&1
python -u generate_indices_md.py --ckpt $OUT/best.pth --tag $TAG
python -u md_diagnostics.py --ckpt $OUT/best.pth
"
