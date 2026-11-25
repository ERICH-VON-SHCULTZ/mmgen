#!/bin/bash
# Round-24 chaining stage (CPU; docs/MFD_REPORT.md §24). Runs tools/r24_report.py (selection on POOLED Arts + Instruments seed-42
# VALIDATION only; a config qualifies when its pooled valid HR@10 > P1a's) and submits the next step as Slurm dependencies:
#   STAGE=a1    best A1 lr and regularisation (carried together) -> A2 = dm256[_lr][_reg], l6[_lr][_reg] on Arts + Instruments seed 42: gates -> pack -> evals + fdumps;
#               P1a-opt check + timing (round-23 harness, batch 1 / 64, H200) per dataset; then STAGE=ab after A2 + B evals (B_EVALS)
#   STAGE=ab    A-best, B-best -> combo (A-best + B tokens) on Arts + Instruments seed 42 -> STAGE=best after its evals
#   STAGE=best  best of {A-best, B-best, combo} -> Games seed 42 (gate, pack, eval + fdump) -> STAGE=seeds
#   STAGE=seeds best on seeds 43 / 44 for all three datasets (gates, packs of 3, evals + fdumps) -> final report:
#               tools/r24_report.py + 3 x 3 table vs P1a, A01, C (tools/r22_e10_report.py --final r24_<best>)
# Job IDs -> logs/round24_jobs.txt. No promotion call anywhere.
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --job-name=mmgen_r24_stage
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
shopt -s inherit_errexit                                     # a failed sbatch inside $(train_eval ...) stops the stage
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round24_jobs.txt
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
: "${STAGE:?}"
report() { singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u r24_report.py $*"; }
jget() { python3 -c "import json; v = json.load(open('$1'))['$2']; print(v if v else '')"; }
# train (gate per dataset -> one pack) + eval / fdump per run; prints the ':'-joined eval + fdump job IDs
train_eval() {   # $1 = "D:VAR:SEED ..." entries
  local G="" EV="" D V S P e f
  for D in $(for x in $1; do echo ${x%%:*}; done | sort -u); do
    VS=$(for x in $1; do if [ ${x%%:*} = $D ]; then y=${x#*:}; echo ${y%%:*}; fi; done | sort -u | tr '\n' ' ')
    G="$G:$(sbatch --parsable --export=ALL,D=$D,VARS="$VS" $REPO/sbatch/r24/r24_gate.sbatch)"
  done
  set -- $1; local PACKS=""
  while [ $# -gt 0 ]; do
    local GRP="$1 ${2:-} ${3:-} ${4:-}"; shift $(( $# < 4 ? $# : 4 ))
    P=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok$G --export=ALL,RUNS="$GRP" $REPO/sbatch/r24/pack_r24_train.sbatch); PACKS="$PACKS $P"
    for x in $GRP; do IFS=: read D V S <<< "$x"
      e=$(sbatch --parsable --time=02:00:00 --kill-on-invalid-dep=yes --dependency=afterok:$P --export=ALL,D=$D,RUN=r24_$V,SEED=$S,STEP=eval $REPO/sbatch/r22/r22_run.sbatch)
      f=$(sbatch --parsable --time=00:30:00 --kill-on-invalid-dep=yes --dependency=afterok:$P --export=ALL,D=$D,RUN=r24_$V,SEED=$S,STEP=fdump $REPO/sbatch/r22/r22_run.sbatch)
      EV="$EV:$e:$f"
    done
  done
  echo "  gates ${G#:} packs$PACKS evals+fdumps ${EV#:}" >> $L
  echo "${EV#:}"
}
case $STAGE in
  a1)
    report --stage a1
    CARRY=$(jget $RUNS/geo/r24_stage_a1.json carry)          # best qualifying lr and / or regularisation, e.g. lr1e3_do02
    S1=${CARRY:+_$CARRY}
    A2="dm256$S1 l6$S1"
    echo "--- $(date +%F) STAGE a1: carried A1 settings '${CARRY:-none (no A1 setting beats P1a on pooled validation)}' -> A2 VARS $A2" | tee -a $L
    python3 - <<PY | tee -a $L
import json
r = json.load(open("$RUNS/geo/r24_stage_a1.json"))
print("  rule", r["rule"])
for x in r["reasons"]:
    print("  " + x)
if r["lr_and_reg_untested_together"]:
    print("  NOTE: the carried lr and regularisation come from different A1 runs; A2 uses a combination A1 never tested")
if r["best_lr_at_grid_edge"]:
    print("  NOTE: the best lr is at the edge of the tested range")
PY
    EV=$(train_eval "Arts:dm256$S1:42 Arts:l6$S1:42 Instruments:dm256$S1:42 Instruments:l6$S1:42")
    COMMIT=$(cd $REPO && git rev-parse --short HEAD); CODE=$RUNS/timing/r23/code_$COMMIT
    mkdir -p $CODE; cp $REPO/macrec/*.py $CODE/; cp -r $REPO/macrec/config $CODE/; echo $COMMIT > $CODE/COMMIT
    TJ=""
    for D in Arts Instruments; do
      PREV=""
      for V in $A2; do                                         # P1a-opt == the A2 model (4,096 test users, K_f 50), then timing
        PREV=$(D=$D STEP=p1aoptverify VRUN=r24_$V CODE=$CODE sbatch --parsable --dependency=afterok:$EV${PREV:+,afterany:$PREV} --export=ALL $REPO/sbatch/r23/r23_job.sbatch)
      done
      TJ="$TJ:$(D=$D STEP=time TAG=r24_a2 METHODS=P1aopt:50,opt@r24_dm256$S1:50,opt@r24_l6$S1:50 BATCHES=1,64 REPEATS=3 CODE=$CODE \
        sbatch --parsable --dependency=afterok:$PREV --export=ALL $REPO/sbatch/r23/r23_job.sbatch)"
    done
    AB=$(sbatch --parsable --dependency=afterany:$EV${B_EVALS:+:$B_EVALS} --export=ALL,STAGE=ab $REPO/sbatch/r24/r24_stage.sbatch)
    echo "  A2 P1a-opt check + timing jobs ${TJ#:} (code $COMMIT); STAGE ab $AB (after A2 + B evals)" | tee -a $L ;;
  lrpeak)                                                       # §24.4: the baselines get P1a's peak lr too when it is above 2e-3
    report --stage lrpeak
    X=$(jget $RUNS/geo/r24_stage_lrpeak.json extend_baselines)
    echo "--- $(date +%F) STAGE lrpeak: $(cat $RUNS/geo/r24_stage_lrpeak.json | tr -d '\n ')" | tee -a $L
    NEW=""
    if [ -n "$X" ]; then
      RL=""; for B in mql macrecpaper tiger_text; do for D in Arts Instruments; do RL="$RL $D:$B:$X"; done; done
      G=$(sbatch --parsable --export=ALL,RUNS="${RL# }" $REPO/sbatch/r24/r24_base_gate.sbatch)
      P=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$G --export=ALL,RUNS="${RL# }" $REPO/sbatch/r24/r24_base_pack.sbatch)
      for x in $RL; do NEW="$NEW:$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$P --export=ALL,RUNS="$x" $REPO/sbatch/r24/r24_base_eval.sbatch)"; done
      echo "  baselines at $X: gate $G pack $P evals ${NEW#:}" | tee -a $L
    fi
    DEP="${BASE_EVALS:+:$BASE_EVALS}$NEW"
    F=$(sbatch --parsable ${DEP:+--dependency=afterany$DEP} --export=ALL,TOOL="r24_baselines.py" $REPO/sbatch/r21/r21_tool.sbatch)
    echo "  final baseline comparison $F" | tee -a $L ;;
  ab)
    report --stage ab
    C=$(jget $RUNS/geo/r24_stage_ab.json combo)
    echo "--- $(date +%F) STAGE ab: $(cat $RUNS/geo/r24_stage_ab.json | tr -d '\n ')" | tee -a $L
    if [ -n "$C" ]; then
      EV=$(train_eval "Arts:$C:42 Instruments:$C:42")
      B=$(sbatch --parsable --dependency=afterany:$EV --export=ALL,STAGE=best $REPO/sbatch/r24/r24_stage.sbatch)
    else
      B=$(sbatch --parsable --export=ALL,STAGE=best $REPO/sbatch/r24/r24_stage.sbatch)
    fi
    echo "  combo '${C:-none}'; STAGE best $B" | tee -a $L ;;
  best)
    report --stage best
    BEST=$(jget $RUNS/geo/r24_stage_best.json best)
    echo "--- $(date +%F) STAGE best: $(cat $RUNS/geo/r24_stage_best.json | tr -d '\n ')" | tee -a $L
    if [ -z "$BEST" ]; then echo "  no round-24 config beats P1a on pooled Arts + Instruments validation: nothing chained" | tee -a $L; exit 0; fi
    EV=$(train_eval "Games:$BEST:42")
    S=$(sbatch --parsable --dependency=afterany:$EV --export=ALL,STAGE=seeds,BEST=$BEST $REPO/sbatch/r24/r24_stage.sbatch)
    echo "  $BEST -> Games seed 42; STAGE seeds $S" | tee -a $L ;;
  seeds)
    : "${BEST:?}"
    EV=$(train_eval "Arts:$BEST:43 Instruments:$BEST:43 Games:$BEST:43 Arts:$BEST:44 Instruments:$BEST:44 Games:$BEST:44")
    F1=$(sbatch --parsable --dependency=afterany:$EV --export=ALL,TOOL="r24_report.py" $REPO/sbatch/r21/r21_tool.sbatch)
    F2=$(sbatch --parsable --dependency=afterany:$EV --export=ALL,TOOL="r22_e10_report.py --final r24_$BEST" $REPO/sbatch/r21/r21_tool.sbatch)
    echo "--- $(date +%F) STAGE seeds: $BEST seeds 43/44 on all datasets; final report $F1, 3x3 table $F2" | tee -a $L ;;
  *) echo "STAGE?"; exit 1 ;;
esac
