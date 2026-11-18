#!/bin/bash
# §22.8 / §22.9 E8 + E9 stage (CPU): after the Arts seed-42 E8 / E9 evals; tools/r22_runs_report.py --stage e89 E8 E9. Every E8 / E9
# variant whose Arts seed-42 valid HR@10 beats A01's is chained on Arts seeds 43 / 44 (pack, H200 eval + fdump, report).
# Job IDs -> logs/round22_jobs.txt.
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --job-name=mmgen_r22_e89stage
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round22_jobs.txt
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $REPO/sbatch/tools && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u r22_runs_report.py --stage e89 E8 E9"
VARS=$(cat $RUNS/geo/r22_e89_stage.txt)
if [ -z "$VARS" ]; then echo "E8/E9 stage: no E8/E9 run beats A01 on Arts seed-42 validation; nothing chained" | tee -a $L; exit 0; fi
ENTRIES=""; for V in $VARS; do ENTRIES="$ENTRIES Arts:$V:43 Arts:$V:44"; done
P=$(sbatch --parsable --nice=50 --export=ALL,RUNS="$ENTRIES" $REPO/sbatch/r22/pack_r22_train.sbatch)
EV=""
for V in $VARS; do for S in 43 44; do
  e=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=Arts,RUN=r22_$V,SEED=$S,STEP=eval $REPO/sbatch/r22/r22_run.sbatch)
  f=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=Arts,RUN=r22_$V,SEED=$S,STEP=fdump $REPO/sbatch/r22/r22_run.sbatch)
  EV="$EV:$e:$f"
done; done
R=$(sbatch --parsable --dependency=afterok$EV --export=ALL,TOOL="r22_runs_report.py E8 E9" $REPO/sbatch/r21/r21_tool.sbatch)
echo "E8/E9 stage: Arts seeds 43/44 VARS=$VARS pack $P evals+fdumps ${EV#:} report $R" | tee -a $L
