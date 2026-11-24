#!/bin/bash
# Round 24, wave 2 (2026-11-24; docs/MFD_REPORT.md §24.1). Resubmits every missing eval / fdump of Task 0 and A1 + B with short time
# limits (eval 2 h, fdump 30 min instead of 6 h) and everything chained behind them, and adds four A1 runs (lr2e3, lr1e3_do02 on
# Arts + Instruments): gates -> one pack -> evals + fdumps -> the same STAGE a1. Cancels the dependents FIRST: STAGE a1 depends
# afterany on the evals, so cancelling the evals first would start it on missing results.
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round24_jobs.txt
cd $REPO
[ -z "$(git status --porcelain -- macrec sbatch)" ] || { echo "commit macrec/ and sbatch/ first"; exit 1; }
quiet() { "$@" 2> >(grep -v _add_delay >&2); }
echo "=== $(date +%F) round 24 wave 2, commit $(git rev-parse --short HEAD)" | tee -a $L
quiet scancel 24199198 24196142 24196143                                       # STAGE a1, Task 0 runs report, Task 0 3x3
for i in $(seq 60); do [ -z "$(squeue -h -j 24199198,24196142,24196143 2>/dev/null)" ] && break; sleep 2; done
[ -z "$(squeue -h -j 24199198,24196142,24196143 2>/dev/null)" ] || { echo "dependents still queued, stop"; exit 1; }
quiet scancel --state=PENDING --name=mmgen_r22_run -u $USER
for i in $(seq 120); do [ -z "$(squeue -h -u $USER -n mmgen_r22_run -t PD 2>/dev/null)" ] && break; sleep 2; done
[ -z "$(squeue -h -u $USER -n mmgen_r22_run -t PD)" ] || { echo "pending evals still queued, stop"; exit 1; }
declare -A LIVE                                                                # evals that were already running: keep them
for j in $(squeue -h -u $USER -n mmgen_r22_run -t R,CG -o %i); do
  k=$(head -1 $RUNS/logs/mmgen_r22_run_$j.out | sed -E 's#.*STEP=([a-z]+) .*/mfd/([A-Za-z]+)/([^/]+)/seed([0-9]+) .*#\1:\2:\3:\4#'); LIVE[$k]=$j; echo "  running, kept: $k $j" | tee -a $L
done
one() {  # $1 D, $2 RUN, $3 SEED, $4 extra dependency; prints the job IDs of the missing eval / fdump (":"-prefixed)
  local out="" step f t
  for step in eval fdump; do
    f=$RUNS/mfd/$1/$2/seed$3/r21_eval/ranks.pt; t=02:00:00
    [ $step = fdump ] && { f=$RUNS/geo/r22/$1/fdump_$2_s$3.pt; t=00:30:00; }
    if [ -n "${LIVE[$step:$1:$2:$3]:-}" ]; then out="$out:${LIVE[$step:$1:$2:$3]}"; continue; fi
    [ -f $f ] && continue
    out="$out:$(sbatch --parsable --time=$t --kill-on-invalid-dep=yes ${4:+--dependency=afterok:$4} --export=ALL,D=$1,RUN=$2,SEED=$3,STEP=$step sbatch/r22/r22_run.sbatch)"
  done
  echo "$out"
}
# Task 0
T0=""; for D in Instruments Games; do for S in 43 44; do T0="$T0$(one $D r22_E9k4 $S)"; done; done
R1=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok$T0 --export=ALL,TOOL="r22_runs_report.py E9" sbatch/r21/r21_tool.sbatch)
R2=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok$T0 --export=ALL,TOOL="r22_e10_report.py --final E9k4" sbatch/r21/r21_tool.sbatch)
echo "Task 0 evals+fdumps ${T0#:} runs report $R1 3x3 $R2" | tee -a $L
# A1 + B, existing runs
A1EV=""; BEV=""
for D in Arts Instruments; do
  for V in lr3e4 lr1e3 do02 do03 wd01 gn5; do A1EV="$A1EV$(one $D r24_$V 42)"; done
  for V in B1 B2; do BEV="$BEV$(one $D r24_$V 42)"; done
done
# A1 additions
GA=$(sbatch --parsable --export=ALL,D=Arts,VARS="lr2e3 lr1e3_do02" sbatch/r24/r24_gate.sbatch)
GI=$(sbatch --parsable --export=ALL,D=Instruments,VARS="lr2e3 lr1e3_do02" sbatch/r24/r24_gate.sbatch)
P=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$GA:$GI --export=ALL,RUNS="Arts:lr2e3 Arts:lr1e3_do02 Instruments:lr2e3 Instruments:lr1e3_do02" sbatch/r24/pack_r24_train.sbatch)
NEW=""; for D in Arts Instruments; do for V in lr2e3 lr1e3_do02; do NEW="$NEW$(one $D r24_$V 42 $P)"; done; done
ST=$(sbatch --parsable --dependency=afterany$A1EV$NEW --export=ALL,STAGE=a1,B_EVALS="${BEV#:}" sbatch/r24/r24_stage.sbatch)
echo "A1 evals+fdumps ${A1EV#:} B evals+fdumps ${BEV#:}" | tee -a $L
echo "A1 additions lr2e3, lr1e3_do02: gates $GA $GI pack $P evals+fdumps ${NEW#:}; STAGE a1 $ST (after all A1 evals; stops if any A1 result is missing)" | tee -a $L
