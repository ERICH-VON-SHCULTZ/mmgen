#!/bin/bash
# §22.10 E10 stage (CPU), STAGE=e10c|e10b: tools/r22_e10_report.py --stage $STAGE (report of every E10 run + POOLED seed-42 validation
# HR@10 over Arts + Instruments + Games vs P1a and A01). If the candidate (E10c, or A01 with the per-dataset beta of E10b) beats BOTH,
# its runs are chained on seeds 43 / 44 for all three datasets (A01 = beta 0.1 already exists): packs of <= 3 runs, H200 eval + fdump,
# then the 3 x 3 table (r22_e10_report.py --final $STAGE). Job IDs -> logs/round22_jobs.txt.
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_r22_e10stage
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round22_jobs.txt
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u r22_e10_report.py --stage $STAGE"
ENTRIES=$(cat $RUNS/geo/r22_e10_${STAGE}_stage.txt)
if [ -z "$ENTRIES" ]; then echo "E10 stage $STAGE: does not beat both P1a and A01 on pooled seed-42 validation HR@10; nothing chained" | tee -a $L; exit 0; fi
ALL=""; for E in $ENTRIES; do for S in 43 44; do ALL="$ALL $E:$S"; done; done
set -- $ALL; PACKS=""; EV=""
while [ $# -gt 0 ]; do
  G="$1 ${2:-} ${3:-}"; shift $(( $# < 3 ? $# : 3 ))
  P=$(sbatch --parsable --export=ALL,RUNS="$G" $REPO/sbatch/r22/pack_r22_train.sbatch); PACKS="$PACKS $P"
  for R in $G; do IFS=: read D V S <<< "$R"
    e=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_$V,SEED=$S,STEP=eval $REPO/sbatch/r22/r22_run.sbatch)
    f=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_$V,SEED=$S,STEP=fdump $REPO/sbatch/r22/r22_run.sbatch)
    EV="$EV:$e:$f"
  done
done
F=$(sbatch --parsable --dependency=afterok$EV --export=ALL,TOOL="r22_e10_report.py --final $STAGE" $REPO/sbatch/r21/r21_tool.sbatch)
echo "E10 stage $STAGE: chained $ALL; packs$PACKS evals+fdumps ${EV#:}; 3x3 table $F" | tee -a $L
