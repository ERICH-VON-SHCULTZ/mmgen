#!/bin/bash
# Round 21 stage 2 (CPU; docs/MFD_REPORT.md §21): collect Arts (tools/r21_collect.py -> geo/r21_Arts.json), pick the best run
# per family (G0, A, B, C) by Arts VALIDATION HR@10 with plain inference (tie NDCG@10) -> geo/r21_family_choice.json,
# then submit those (up to 4) on Instruments seed 42: one pack (after the Instruments gate), an H200 eval per run, and the
# Instruments collection. Job IDs appended to mmgen_runs/logs/round21_jobs.txt.
# Submit: sbatch --dependency=afterok:<all Arts evals>:<Arts + Instruments P1a reference evals>:<Instruments gate> sbatch/r21/r21_stage2.sbatch
# (the Instruments pack is submitted without a dependency: the gate has passed by then, and run_r21_train.sh re-checks its files)
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_r21_stage2
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
echo "job $SLURM_JOB_ID commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u r21_collect.py Arts && python -u r21_family_choice.py"
VARS=$(cat $RUNS/geo/r21_family_choice.txt)
L=$RUNS/logs/round21_jobs.txt
ENTRIES=""; for V in $VARS; do ENTRIES="$ENTRIES Instruments:$V"; done
P=$(sbatch --parsable --export=ALL,RUNS="$ENTRIES" $REPO/sbatch/r21/pack_r21_train.sbatch)
EV=""
for V in $VARS; do
  E=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=Instruments,RUN=r21_$V $REPO/sbatch/r21/r21_eval.sbatch); EV="$EV:$E"
done
CO=$(sbatch --parsable --dependency=afterok$EV --export=ALL,D=Instruments $REPO/sbatch/r21/r21_collect.sbatch)
echo "stage2 Instruments VARS=$VARS pack $P evals ${EV#:} collect $CO" | tee -a $L
