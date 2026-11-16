#!/bin/bash
# Round-22 gate (docs/MFD_REPORT.md §22): macrec/r22_check.py (aux heads on A01 seed 42 + 1+3+3 encoder / decoder layout) ->
# geo/<D>/r22_check.json (replaced only when every check passes), then dry runs of every VAR (run_r22_train.sh DRYRUN=1) and
# tools/r22_config_diff.py vs round 21's A01 dry run -> geo/gate/<D>/r22_config_diff.json (merged). No training.
# Submit: sbatch --export=ALL,D=Arts sbatch/r22/r22_gate.sbatch   (VARS default "E2g01 E2g05 E1")
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --job-name=mmgen_r22_gate
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
export WANDB_MODE=offline WANDB_DIR=$RUNS/wandb             # dry runs never train
D=${D:-Arts}; VARS=${VARS:-"E2g01 E2g05 E1"}
RR=$TMPDIR/run_r22_train.sh; cp $REPO/sbatch/r22/run_r22_train.sh $RR
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D VARS=$VARS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u r22_check.py --data_path $RUNS/data --dataset $D --a01_ckpt $RUNS/mfd/$D/r21_A01/seed42 --v1_dir $RUNS/md/sids/v1/$D/1+2+2 \
  --out_file $RUNS/geo/$D/r22_check.new.json > $RUNS/geo/$D/r22_check.log 2>&1 || { tail -20 $RUNS/geo/$D/r22_check.log; echo 'R22 CHECK FAILED'; exit 1; }
mv $RUNS/geo/$D/r22_check.new.json $RUNS/geo/$D/r22_check.json
tail -1 $RUNS/geo/$D/r22_check.log
for V in $VARS; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RR; done
cd $T && python r22_config_diff.py $D $VARS
"
echo done
