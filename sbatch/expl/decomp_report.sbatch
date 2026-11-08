#!/bin/bash
# §20 Task 0 analysis (CPU): sbatch/tools/expl_decomp_report.py <D> <decomp tag> <topk tag> -> mmgen_runs/expl/<D>/decomp.json
# Submit after the GPU jobs: sbatch --dependency=afterok:<decomp>:<topk> --export=ALL,D=Arts,TAG=P1a,LTAG=P1a_ii sbatch/expl/decomp_report.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_decomp_report
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/; M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/
echo "job $SLURM_JOB_ID D=$D TAG=$TAG LTAG=$LTAG commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && MACREC=$M TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u expl_decomp_report.py $D $TAG $LTAG"
