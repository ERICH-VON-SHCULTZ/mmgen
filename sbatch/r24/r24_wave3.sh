#!/bin/bash
# Round 24, wave 3 (2026-11-25; docs/MFD_REPORT.md §24.1 / §24.4), in priority order:
#  1 baseline fairness: C at lr 1e-3 / 2e-3, D at lr 2e-3 (own batch 1024 x 4) and lr 1e-3 at P1a's batch (256 x 4), Arts + Instruments seed 42;
#    gate (dry-run diff vs own settings + 5 real steps) -> pack -> H200 evals -> interim comparison. TIGER lr 1e-3 / 2e-3 after the C / D packs.
#  2 P1a lr 3e-3, 4e-3 and lr 2e-3 + dropout 0.2 (P1a's size), Arts + Instruments seed 42 -> evals + fdumps -> STAGE lrpeak (baselines get the
#    peak lr too if it is above 2e-3; final baseline comparison)
#  3 STAGE ab (24545165) replaced by one that also waits for the runs of 2 (it stops if any result is missing)
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round24_jobs.txt; cd $REPO
[ -z "$(git status --porcelain -- macrec sbatch)" ] || { echo "commit macrec/ and sbatch/ first"; exit 1; }
K=--kill-on-invalid-dep=yes
echo "=== $(date +%F) round 24 wave 3, commit $(git rev-parse --short HEAD)" | tee -a $L
base() {  # $1 = runs, $2 = extra dependency, $3 = nice; sets G P EV
  G=$(sbatch --parsable ${2:+--dependency=afterany:$2} ${3:+--nice=$3} --export=ALL,RUNS="$1" sbatch/r24/r24_base_gate.sbatch)
  P=$(sbatch --parsable $K ${3:+--nice=$3} --dependency=afterok:$G --export=ALL,RUNS="$1" sbatch/r24/r24_base_pack.sbatch); EV=""
  for x in $1; do EV="$EV:$(sbatch --parsable $K ${3:+--nice=$3} --dependency=afterok:$P --export=ALL,RUNS="$x" sbatch/r24/r24_base_eval.sbatch)"; done
}
base "Arts:mql:lr1e3 Arts:mql:lr2e3 Instruments:mql:lr1e3 Instruments:mql:lr2e3";  GC=$G; PC=$P; EC=$EV
base "Arts:macrecpaper:lr2e3 Arts:macrecpaper:lr1e3_b256 Instruments:macrecpaper:lr2e3 Instruments:macrecpaper:lr1e3_b256"; GD=$G; PD=$P; ED=$EV
IR=$(sbatch --parsable --dependency=afterany$EC$ED --export=ALL,TOOL="r24_baselines.py" sbatch/r21/r21_tool.sbatch)
echo "1 C: gate $GC pack $PC evals ${EC#:} | D: gate $GD pack $PD evals ${ED#:} | interim comparison $IR" | tee -a $L
# 2 P1a
V="lr3e3 lr4e3 lr2e3_do02"
GA=$(sbatch --parsable --export=ALL,D=Arts,VARS="$V" sbatch/r24/r24_gate.sbatch)
GI=$(sbatch --parsable --export=ALL,D=Instruments,VARS="$V" sbatch/r24/r24_gate.sbatch)
PE=""; PP=""
for D in Arts Instruments; do
  R=""; for v in $V; do R="$R $D:$v"; done
  Q=$(sbatch --parsable $K --dependency=afterok:$GA:$GI --export=ALL,RUNS="${R# }" sbatch/r24/pack_r24_train.sbatch); PP="$PP $Q"
  for v in $V; do
    PE="$PE:$(sbatch --parsable --time=02:00:00 $K --dependency=afterok:$Q --export=ALL,D=$D,RUN=r24_$v,SEED=42,STEP=eval sbatch/r22/r22_run.sbatch)"
    PE="$PE:$(sbatch --parsable --time=00:30:00 $K --dependency=afterok:$Q --export=ALL,D=$D,RUN=r24_$v,SEED=42,STEP=fdump sbatch/r22/r22_run.sbatch)"
  done
done
echo "2 P1a lr3e3 lr4e3 lr2e3_do02: gates $GA $GI packs$PP evals+fdumps ${PE#:}" | tee -a $L
# TIGER after the C / D packs
base "Arts:tiger_text:lr1e3 Arts:tiger_text:lr2e3 Instruments:tiger_text:lr1e3 Instruments:tiger_text:lr2e3" $PC:$PD 200; GT=$G; PT=$P; ET=$EV
echo "1b TIGER: gate $GT pack $PT evals ${ET#:} (after the C / D packs, nice 200)" | tee -a $L
BE="${EC#:}$ED$ET"
LP=$(sbatch --parsable --dependency=afterany$PE --export=ALL,STAGE=lrpeak,BASE_EVALS="$BE" sbatch/r24/r24_stage.sbatch)
# 3 STAGE ab must wait: the new one is submitted first, then the old one is cancelled
A2EV=$(scontrol show job 24545165 | grep -oE "Dependency=[^ ]*" | grep -oE "[0-9]{8}" | sort -u | tr '\n' ':' | sed 's/:$//')
AB=$(sbatch --parsable --dependency=afterany${A2EV:+:$A2EV}$PE --export=ALL,STAGE=ab sbatch/r24/r24_stage.sbatch)
scancel 24545165 2> >(grep -v _add_delay >&2)
echo "3 STAGE lrpeak $LP (after the P1a runs; baselines extended if the peak is above 2e-3; final comparison); STAGE ab $AB replaces 24545165 (after A2 evals ${A2EV} and the P1a runs; stops if a result is missing)" | tee -a $L
