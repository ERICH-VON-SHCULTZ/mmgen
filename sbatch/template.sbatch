#!/bin/bash
# Template for mmgen GPU jobs. Copy, rename, edit the job name/output/time and the payload.
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mmgen_job
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge

REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
# Overlay is ALWAYS mounted :ro in jobs. Never mount it :rw while any job is using it.
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3

# Secrets live OUTSIDE the repo. ~/.config/mmgen/secrets.sh (chmod 600) should contain:
#   export WANDB_API_KEY=<key>
# If it is absent, W&B runs in offline mode (logged locally under the run dir; `wandb sync` later).
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=${WANDB_PROJECT:-mmgen}

export HF_HOME=/scratch/yw8866/hf_cache
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
# The overlay also has TensorFlow + Keras 3 (GRID TFRecord tooling); transformers must not import TF.
export USE_TF=0
# torch>=2.6 defaults torch.load(weights_only=True); MACRec's RQ-VAE checkpoints pickle an
# argparse.Namespace and are loaded without weights_only (cross_index/generate_indices_distance.py:71).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

singularity exec --nv \
    --bind /scratch/yw8866 \
    --overlay "$OVERLAY":ro \
    "$SIF" \
    /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $REPO/macrec
nvidia-smi --query-gpu=name,memory.total --format=csv
# ---- payload ----
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"
