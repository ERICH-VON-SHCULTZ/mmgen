#!/bin/bash
# Round 24 submission (docs/MFD_REPORT.md §24; job IDs -> mmgen_runs/logs/round24_jobs.txt). Two independent chains:
#  Task 0  E9k4 seeds 43 / 44 on Instruments + Games: E8/E9 gates (VARS=E9k4) -> one pack of 4 -> H200 evals + fdumps ->
#          tools/r22_runs_report.py E9 (E6 split) + tools/r22_e10_report.py --final E9k4 (3 x 3 vs P1a, A01, C, pooled bootstrap)
#  A1 + B  Arts + Instruments seed 42: r24 gates (VARS = A1 + B1 B2) -> packs of 4 -> evals + fdumps -> STAGE a1 (after the A1
#          evals; carries the B eval IDs so STAGE ab waits for both tracks; see sbatch/r24/r24_stage.sbatch)
set -euo pipefail
PART=${PART:-all}                                           # all | task0 | ab (resubmit one chain)
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round24_jobs.txt
cd $REPO
[ -z "$(git status --porcelain -- macrec sbatch)" ] || { echo "commit macrec/ and sbatch/ first"; exit 1; }
echo "=== $(date +%F) round 24 submit, commit $(git rev-parse --short HEAD)" | tee -a $L
# Task 0
if [ $PART != ab ]; then
G1=$(sbatch --parsable --export=ALL,D=Instruments,VARS=E9k4 sbatch/r22/r22_e89_gate.sbatch)
G2=$(sbatch --parsable --export=ALL,D=Games,VARS=E9k4 sbatch/r22/r22_e89_gate.sbatch)
P=$(sbatch --parsable --dependency=afterok:$G1:$G2 --export=ALL,RUNS="Instruments:E9k4:43 Instruments:E9k4:44 Games:E9k4:43 Games:E9k4:44" sbatch/r22/pack_r22_train.sbatch)
EV=""
for D in Instruments Games; do for S in 43 44; do
  e=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_E9k4,SEED=$S,STEP=eval sbatch/r22/r22_run.sbatch)
  f=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_E9k4,SEED=$S,STEP=fdump sbatch/r22/r22_run.sbatch)
  EV="$EV:$e:$f"
done; done
R1=$(sbatch --parsable --dependency=afterok$EV --export=ALL,TOOL="r22_runs_report.py E9" sbatch/r21/r21_tool.sbatch)
R2=$(sbatch --parsable --dependency=afterok$EV --export=ALL,TOOL="r22_e10_report.py --final E9k4" sbatch/r21/r21_tool.sbatch)
echo "Task 0 (E9k4 seeds 43/44, Instruments + Games): gates $G1 $G2 pack $P evals+fdumps ${EV#:} runs report $R1 3x3 $R2" | tee -a $L
fi
[ $PART = task0 ] && exit 0
# Track A1 + Track B
VARS="lr3e4 lr1e3 do02 do03 wd01 gn5 B1 B2"
GA=$(sbatch --parsable --export=ALL,D=Arts,VARS="$VARS" sbatch/r24/r24_gate.sbatch)
GI=$(sbatch --parsable --export=ALL,D=Instruments,VARS="$VARS" sbatch/r24/r24_gate.sbatch)
A1EV=""; BEV=""; PACKS=""
for GRP in "Arts:lr3e4 Arts:lr1e3 Arts:do02 Arts:do03" "Arts:wd01 Arts:gn5 Arts:B1 Arts:B2" \
           "Instruments:lr3e4 Instruments:lr1e3 Instruments:do02 Instruments:do03" "Instruments:wd01 Instruments:gn5 Instruments:B1 Instruments:B2"; do
  P=$(sbatch --parsable --dependency=afterok:$GA:$GI --export=ALL,RUNS="$GRP" sbatch/r24/pack_r24_train.sbatch); PACKS="$PACKS $P"
  for x in $GRP; do D=${x%%:*}; V=${x#*:}
    e=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r24_$V,SEED=42,STEP=eval sbatch/r22/r22_run.sbatch)
    f=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r24_$V,SEED=42,STEP=fdump sbatch/r22/r22_run.sbatch)
    case $V in B*) BEV="$BEV:$e:$f" ;; *) A1EV="$A1EV:$e:$f" ;; esac
  done
done
ST=$(sbatch --parsable --dependency=afterany$A1EV --export=ALL,STAGE=a1,B_EVALS="${BEV#:}" sbatch/r24/r24_stage.sbatch)
echo "A1 + B (Arts + Instruments seed 42): gates $GA $GI packs$PACKS A1 evals+fdumps ${A1EV#:} B evals+fdumps ${BEV#:} STAGE a1 $ST" | tee -a $L
