#!/bin/bash
# Evaluate several trained runs in ONE GPU job (the cluster is GPU-quota-bound, so 1 job beats 5).
# Phase 1: full valid+test evaluation of every run CONCURRENTLY (metrics; wall-clock contended).
# Phase 2: UNCONTENDED inference cost: each beam search run alone on a 2,000-user test subsample.
# Accepts any of the GPU partitions; the GPU type is recorded in every cost file.
# Submit: sbatch --export=ALL,RUNS="Instruments:mql:42 Instruments:macrec:42" sbatch/baselines/pack_eval.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon,a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_pack_eval
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
LOGD=/scratch/yw8866/mmgen_runs/logs/pack_eval_$SLURM_JOB_ID; mkdir -p $LOGD
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION  RUNS=$RUNS  commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader -l 10 > $LOGD/gpu_util.csv &
START=$(date +%s)
SING="singularity exec --nv --bind /scratch/yw8866 --overlay $OVERLAY:ro $SIF /bin/bash -c"

# ---- phase 1: full evaluation, concurrent ----
PIDS=()
for R in $RUNS; do
  IFS=: read D B SEED <<< "$R"
  $SING "D=$D B=$B SEED=$SEED STAGE=eval bash $REPO/sbatch/baselines/run_baseline.sh" > $LOGD/$D-$B-$SEED.log 2>&1 &
  PIDS+=($!); sleep 10
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
echo "phase 1 done after $(( $(date +%s) - START )) s  fail=$FAIL"

# ---- phase 2: uncontended inference cost on a fixed 2,000-user test subsample, one search at a time ----
for R in $RUNS; do
  IFS=: read D B SEED <<< "$R"
  $SING "source /ext3/env.sh && conda activate py310
    OUT=/scratch/yw8866/mmgen_runs/baselines/$D/$B/seed$SEED; C=\$OUT/eval/cost2000; mkdir -p \$C
    IDX=\$(python -c \"import json; print(json.load(open('\$OUT/config.json'))['index_file'])\")
    IIDX=\$(python -c \"import json; print(json.load(open('\$OUT/config.json'))['image_index_file'])\")
    TESTS=\$(python -c \"import json; print(json.load(open('\$OUT/config.json'))['test_tasks'])\")
    cd $REPO/macrec
    for T in \$TESTS; do
      torchrun --nproc_per_node=1 --master_port=\$((20000 + RANDOM % 20000)) eval_split.py --split test --sample_num 2000 \
        --ckpt_path \$OUT --data_path /scratch/yw8866/mmgen_runs/data/ --dataset $D --test_batch_size 64 --num_beams 20 \
        --index_file \$IDX --image_index_file \$IIDX --test_task \$T --results_file \$C/results_\${T}_20.json \
        --save_file \$C/save_\${T}_20.json --stats_file \$C/cost_\${T}.json --filter_items > \$C/eval_\${T}.log 2>&1
    done" || FAIL=1
done
for R in $RUNS; do IFS=: read D B SEED <<< "$R"; python3 $REPO/sbatch/tools/summarize_run.py /scratch/yw8866/mmgen_runs/baselines/$D/$B/seed$SEED || true; done
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
