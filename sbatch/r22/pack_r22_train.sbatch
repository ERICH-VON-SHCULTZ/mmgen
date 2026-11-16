#!/bin/bash
# Round-22 trainings CONCURRENTLY on one GPU (GPU-utilisation policy, see sbatch/mfd/pack_mfd_train.sbatch); one frozen
# copy of macrec/ and run_r22_train.sh made at job start. 12 CPUs / 80 GB / 16 h (2026-11-10: the 24 / 160 GB / 48 h request sat
# pending for over a day on a full cluster; 4 runs x 2 loader workers fit, and P1a trains in under 3 h alone). RUNS entries: <D>:<VAR>[:<SEED>] (VAR as run_r22_train.sh).
# Submit: sbatch --dependency=afterok:<gate> --export=ALL,RUNS="Arts:E2g01 Arts:E2g05 Arts:E1" sbatch/r22/pack_r22_train.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --mem=80GB
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r22_pack
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS_DIR=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-mfd WANDB_DIR=$RUNS_DIR/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RR=$TMPDIR/run_r22_train.sh; cp $REPO/sbatch/r22/run_r22_train.sh $RR
LOGD=$RUNS_DIR/logs/pack_$SLURM_JOB_ID; mkdir -p $LOGD
echo "job $SLURM_JOB_ID on $(hostname)  RUNS=$RUNS  commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > $LOGD/gpu_util.csv &
START=$(date +%s); PIDS=()
for R in $RUNS; do
  IFS=: read D V S <<< "$R"
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "M=$M D=$D VAR=$V SEED=${S:-42} bash $RR" \
    > $LOGD/${R//:/_}.log 2>&1 &
  PIDS+=($!); sleep 20
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
awk -F', ' '{u+=$2; n++} END {printf "mean GPU util over job: %.1f%% (%d samples)\n", u/n, n}' $LOGD/gpu_util.csv
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
