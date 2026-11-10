#!/bin/bash
# §20 collection (CPU): sbatch/tools/round20_collect.py <D> [VAR ...] -> mmgen_runs/geo/round20_<D>.json
# sbatch --export=ALL,D=Arts,VARS="P2a P2b" sbatch/tools/round20_collect.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_r20_collect
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
echo "job $SLURM_JOB_ID D=$D VARS=${VARS:-} commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u round20_collect.py $D ${VARS:-}"
