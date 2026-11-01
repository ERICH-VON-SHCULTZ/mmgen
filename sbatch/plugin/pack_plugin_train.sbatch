#!/bin/bash
# Plug-in trainings (docs/PLUGIN_REPORT.md) CONCURRENTLY on one GPU, exactly C's recipe (run_baseline.sh B=plug*).
# Gates, all before any training: plugin/config_diff.json must be all_ok and list the run as ok (config_check.sbatch);
# the v1 SID set must match its locked hash; each view index file must match the hash recorded in views.json.
# Frozen snapshot of macrec/ and run_baseline.sh. Wall-clock is contended (packed).
# Submit: sbatch --export=ALL,RUNS="Instruments:plug122 Arts:plug122" sbatch/plugin/pack_plugin_train.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --mem=160GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_plugin_pack
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RD=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-plugin WANDB_DIR=$RD/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB
LOGD=$RD/logs/pack_$SLURM_JOB_ID; mkdir -p $LOGD
echo "job $SLURM_JOB_ID on $(hostname)  RUNS=$RUNS  commit $(cd $REPO && git rev-parse --short HEAD)"
cp $REPO/sbatch/tools/plugin_gate.py $TMPDIR/
for R in $RUNS; do
  singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
    "source /ext3/env.sh && conda activate py310 && python $TMPDIR/plugin_gate.py $R" || { echo "GATE FAILED for $R: not training"; exit 1; }
done
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > $LOGD/gpu_util.csv &
START=$(date +%s); PIDS=()
for R in $RUNS; do
  D=${R%%:*}; B=${R##*:}
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
    "MACREC_DIR=$M D=$D B=$B SEED=42 STAGE=train NW=2 bash $RB" > $LOGD/$D-$B.log 2>&1 &
  PIDS+=($!); sleep 20
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
awk -F', ' '{u+=$2; n++} END {printf "mean GPU util over job: %.1f%% (%d samples)\n", u/n, n}' $LOGD/gpu_util.csv
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
