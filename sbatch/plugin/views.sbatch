#!/bin/bash
# Plug-in views (docs/PLUGIN_REPORT.md): split each locked v1 MD SID into a text view [f t..] and an image view
# [f i..], make each view unique with MQL4GRec's distance reassignment, write main + shared-f index files.
# CPU only. Frozen snapshot of macrec/cross_index. Submit: sbatch sbatch/plugin/views.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_plugin_views
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
CI=$TMPDIR/cross_index; cp -r $REPO/macrec/cross_index $CI
echo "job $SLURM_JOB_ID on $(hostname) commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $CI
for D in ${DATASETS:-Instruments Arts}; do for C in 122 133; do
  O=$RUNS/plugin/\$D/\$C; mkdir -p \$O
  python -u plugin_views.py --v1_dir $RUNS/md/sids/v1/\$D/1+\${C:1:1}+\${C:2:1} --tag plug\$C --out_dir \$O > \$O/views.log 2>&1
  tail -40 \$O/views.log
done; done
"
echo done
