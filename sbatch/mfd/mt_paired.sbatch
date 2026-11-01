#!/bin/bash
# MFD with C's task mix: paired bootstrap vs mfd122_mv and vs the plug-in 1+2+2 merged list (docs/MFD_REPORT.md §13),
# from saved ranks / beams. CPU only. Frozen snapshot of the tools and MACRec's evaluate.py + ensemble.py.
# Submit: sbatch [--export=ALL,D=Instruments] sbatch/mfd/mt_paired.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_mfd_mt_paired
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
D=${D:-Instruments}
F=$TMPDIR/frozen; mkdir -p $F
cp $REPO/macrec/evaluate.py $REPO/macrec/ensemble.py $REPO/sbatch/tools/plugin_paired.py $REPO/sbatch/tools/mfd_mt_paired.py $F/
echo "job $SLURM_JOB_ID on $(hostname) D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $F
MACREC=$F python -u mfd_mt_paired.py $D
"
echo done
