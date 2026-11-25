#!/bin/bash
# §24.4 baseline tuning (docs/MFD_REPORT.md): trainings packed on one GPU; each run refuses to train unless its gate passed
# RUNS = "D:B:TAG ..." (sbatch/r24/r24_base.sh). Submit: sbatch --export=ALL,RUNS="Arts:mql:lr1e3" sbatch/r24/r24_base_pack.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=100GB
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r24_bpack
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen; RUNS_DIR=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh; if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-baselines WANDB_DIR=$RUNS_DIR/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $RUNS_DIR/wandb; export TRITON_CACHE_DIR=$TMPDIR/triton
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB; BB=$TMPDIR/r24_base.sh; cp $REPO/sbatch/r24/r24_base.sh $BB
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
: "${RUNS:?}"
echo "job $SLURM_JOB_ID on $(hostname) RUNS=$RUNS commit $(cd $REPO && git rev-parse --short HEAD)"
SING="singularity exec --nv --bind /scratch/yw8866 --overlay $OVERLAY:ro $SIF /bin/bash -c"
run() { $SING "source /ext3/env.sh && conda activate py310 && D=$1 B=$2 TAG=$3 SEED=42 MODE=$4 RB=$RB MACREC_DIR=$M bash $BB"; }
LOGD=$RUNS_DIR/logs/pack_$SLURM_JOB_ID; mkdir -p $LOGD
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > $LOGD/gpu_util.csv &
START=$(date +%s); PIDS=()
for R in $RUNS; do IFS=: read D B TAG <<< "$R"
  run $D $B $TAG train > $LOGD/${R//:/_}.log 2>&1 &
  PIDS+=($!); sleep 20
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
awk -F', ' '{u+=$2; n++} END {printf "mean GPU util over job: %.1f%% (%d samples)\n", u/n, n}' $LOGD/gpu_util.csv
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
