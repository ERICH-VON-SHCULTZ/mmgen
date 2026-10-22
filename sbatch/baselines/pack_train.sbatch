#!/bin/bash
# Train several baseline runs CONCURRENTLY on one GPU (each run alone uses ~30% of an H100, below the
# cluster's utilisation threshold). Training math is unchanged; only wall-clock is shared, so training
# time from these jobs is "contended". Evaluation / inference cost runs separately (baseline.sbatch STAGE=eval).
# Submit: sbatch --export=ALL,RUNS="Instruments:mql:42 Instruments:macrec:42" sbatch/baselines/pack_train.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --mem=160GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=mmgen_pack_train
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-baselines WANDB_DIR=/scratch/yw8866/mmgen_runs/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
LOGD=/scratch/yw8866/mmgen_runs/logs/pack_$SLURM_JOB_ID; mkdir -p $LOGD
echo "job $SLURM_JOB_ID on $(hostname)  RUNS=$RUNS  commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > $LOGD/gpu_util.csv &
START=$(date +%s)
PIDS=()
for R in $RUNS; do
  IFS=: read D B SEED <<< "$R"
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
    "D=$D B=$B SEED=$SEED STAGE=train NW=2 bash $REPO/sbatch/baselines/run_baseline.sh" > $LOGD/$D-$B-$SEED.log 2>&1 &
  PIDS+=($!); sleep 20
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
awk -F', ' '{u+=$1; n++} END {printf "mean GPU util over job: %.1f%% (%d samples)\n", u/n, n}' $LOGD/gpu_util.csv
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
