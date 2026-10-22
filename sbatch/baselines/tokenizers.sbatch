#!/bin/bash
# Baselines: tokenizers. Submit: sbatch --export=ALL,D=Instruments sbatch/baselines/tokenizers.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=mmgen_tokenizers
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-baselines WANDB_DIR=/scratch/yw8866/mmgen_runs/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
echo "job $SLURM_JOB_ID on $(hostname)  D=${D:-} B=${B:-} SEED=${SEED:-}  commit $(cd $REPO && git rev-parse --short HEAD)"
START=$(date +%s)
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "D=$D bash $REPO/sbatch/baselines/tokenizers.sh"
echo "job wall time: $(( $(date +%s) - START )) s"
