#!/bin/bash
# Shared-f vs main plug-in runs, paired on the same users (docs/PLUGIN_REPORT.md §6). CPU only, no model.
# Frozen snapshot of plugin_paired.py and MACRec's evaluate.py + ensemble.py. Submit: sbatch sbatch/plugin/paired_sharedf.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_plugin_sharedf
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
F=$TMPDIR/frozen; mkdir -p $F
cp $REPO/macrec/evaluate.py $REPO/macrec/ensemble.py $REPO/sbatch/tools/plugin_paired.py $F/
echo "job $SLURM_JOB_ID on $(hostname) commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $F
MACREC=$F REF=plug122 python -u plugin_paired.py Instruments:plug122s
MACREC=$F REF=plug133 python -u plugin_paired.py Instruments:plug133s
"
echo done
