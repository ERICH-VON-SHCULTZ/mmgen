#!/bin/bash
# Round-21 trainings CONCURRENTLY on one GPU (GPU-utilisation policy, see sbatch/mfd/pack_mfd_train.sbatch); one frozen
# copy of macrec/ and run_r21_train.sh made at job start. RUNS entries: <D>:<VAR>[:<SEED>] (VAR as run_r21_train.sh).
# Submit: sbatch --dependency=afterok:<gate> --export=ALL,RUNS="Arts:G0 Arts:A01 Arts:A05 Arts:B01" sbatch/r21/pack_r21_train.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --mem=160GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r21_pack
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
RR=$TMPDIR/run_r21_train.sh; cp $REPO/sbatch/r21/run_r21_train.sh $RR
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
