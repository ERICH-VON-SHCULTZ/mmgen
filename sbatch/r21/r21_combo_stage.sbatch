#!/bin/bash
# §21.4 combo stage (CPU): after the Arts AC1 / AC0 evals; tools/r21_combo_choice.py; every combo with Arts valid HR@10 above
# BOTH C033 and A01 is chained to Instruments seed 42 (pack, H200 eval, collection). Job IDs -> logs/round21_jobs.txt.
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --job-name=mmgen_r21_combo
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round21_jobs.txt
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && python -u $REPO/sbatch/tools/r21_combo_choice.py"
VARS=$(cat $RUNS/geo/r21_combo_choice.txt)
if [ -z "$VARS" ]; then echo "combo stage: no combo beats both C033 and A01 on Arts valid; nothing chained" | tee -a $L; exit 0; fi
ENTRIES=""; for V in $VARS; do ENTRIES="$ENTRIES Instruments:$V"; done
P=$(sbatch --parsable --nice=200 --export=ALL,RUNS="$ENTRIES" $REPO/sbatch/r21/pack_r21_train.sbatch)
EV=""; for V in $VARS; do E=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=Instruments,RUN=r21_$V $REPO/sbatch/r21/r21_eval.sbatch); EV="$EV:$E"; done
CO=$(sbatch --parsable --dependency=afterok$EV --export=ALL,D=Instruments $REPO/sbatch/r21/r21_collect.sbatch)
echo "combo stage: Instruments VARS=$VARS pack $P evals ${EV#:} collect $CO" | tee -a $L
