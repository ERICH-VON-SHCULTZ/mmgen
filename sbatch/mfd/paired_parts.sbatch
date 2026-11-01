#!/bin/bash
# Paired bootstrap for MFD Parts A-C from saved per-user ranks (sbatch/tools/mfd_paired_parts.py), CPU only.
# Submit: sbatch sbatch/mfd/paired_parts.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_mfd_paired_parts
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
cp $REPO/sbatch/tools/mfd_paired_parts.py $TMPDIR/
echo "job $SLURM_JOB_ID on $(hostname) commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && python -u $TMPDIR/mfd_paired_parts.py"
echo done
