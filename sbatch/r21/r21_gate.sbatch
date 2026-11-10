#!/bin/bash
# Gate of the round-21 runs (docs/MFD_REPORT.md §21): macrec/r21_check.py (layout, loss identities, stop-gradient, view
# rates, matched steps; on P1a seed 42's checkpoint) -> geo/<D>/r21_check.json, then dry runs of P1aref and every VAR
# (sbatch/r21/run_r21_train.sh DRYRUN=1) and tools/r21_config_diff.py -> geo/gate/<D>/r21_config_diff.json. No training.
# Writes only r21_* files (never geo_check.json / geo_config_diff.json, which round-20 jobs read).
# Submit: sbatch --export=ALL,D=Arts sbatch/r21/r21_gate.sbatch   (VARS default: all seven)
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r21_gate
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export WANDB_MODE=offline WANDB_DIR=$RUNS/wandb             # dry runs never train
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
D=${D:-Arts}; VARS=${VARS:-"G0 A01 A05 B01 B05 C033 C060"}
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RR=$TMPDIR/run_r21_train.sh; cp $REPO/sbatch/r21/run_r21_train.sh $RR
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/r21_config_diff.py $REPO/sbatch/tools/plugin_config_diff.py $T/
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D VARS=$VARS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u r21_check.py --data_path $RUNS/data --dataset $D --p1a_ckpt $RUNS/mfd/$D/mfd122_mv_geoP1a/seed42 \
  --out_file $RUNS/geo/$D/r21_check.json > $RUNS/geo/$D/r21_check.log 2>&1 || { tail -20 $RUNS/geo/$D/r21_check.log; echo 'R21 CHECK FAILED'; exit 1; }
tail -1 $RUNS/geo/$D/r21_check.log
for V in P1aref $VARS; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RR; done
cd $T && python r21_config_diff.py $D $VARS
"
echo done
