#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=03:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=capforce
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/cap_forced.out
# What fills MFD's top-10 when it misses (MFD-only repeats stuck at 0.51, covered new items at 0.001)?
module purge
singularity exec --nv --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT='MMMBGRec' WANDB_INSECURE_DISABLE_SSL=true WANDB_API_KEY='REDACTED_ROTATE_ME'
cd /scratch/yw8866/rec-tmall && python cap_eval_forced.py --spec /scratch/yw8866/rec-tmall/force_exp/spec.json --out /scratch/yw8866/rec-tmall/force_exp/cap_forced.txt"
