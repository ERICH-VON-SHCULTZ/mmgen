#!/bin/bash
# §24.4 baseline tuning: one run of a baseline with another learning rate (inside the container; wraps run_baseline.sh).
# Env: D, B (mql = C | macrecpaper = D | tiger_text), TAG, SEED, MODE (train|eval|dry|smoke), RB (frozen run_baseline.sh), MACREC_DIR.
# TAG = ref (the baseline's own settings, gate reference only) or tokens joined by "_": lr1e3 lr2e3 lr3e3 lr4e3 = learning rate;
# b256 = per-device batch 256 x accumulation 4 (P1a's batch; D's own is 1024 x 4). Everything else = the baseline's own settings.
# train / eval -> mmgen_runs/baselines_tuned/<D>/<B>_<TAG>/seed<S>; dry / smoke -> mmgen_runs/geo/gate/r24_base/<D>/<B>_<TAG>/
set -euo pipefail
RUNS=/scratch/yw8866/mmgen_runs; G=$RUNS/geo/gate/r24_base/$D/${B}_$TAG
unset LR_OVERRIDE BS_OVERRIDE SMOKE_STEPS DRYRUN GEOARGS OUTROOT
for t in ${TAG//_/ }; do
  case $t in
    ref) ;;
    lr1e3) export LR_OVERRIDE=1e-3 ;;  lr2e3) export LR_OVERRIDE=2e-3 ;;  lr3e3) export LR_OVERRIDE=3e-3 ;;  lr4e3) export LR_OVERRIDE=4e-3 ;;
    b256) export BS_OVERRIDE=256 ;;
    *) echo "unknown TAG token $t"; exit 1 ;;
  esac
done
case $MODE in
  dry)   export OUTROOT=$G DRYRUN=1 STAGE=train ;;
  smoke) export OUTROOT=$G/smoke SMOKE_STEPS=5 STAGE=train; rm -rf $OUTROOT ;;
  train) export OUTROOT=$RUNS/baselines_tuned/$D/${B}_$TAG STAGE=train
         python3 -c "import json, sys; d = json.load(open('$RUNS/geo/gate/r24_base/$D/diff_${B}_$TAG.json')); sys.exit(0 if d['ok'] else 1)" || { echo "BASELINE GATE NOT PASSED ($D $B $TAG)"; exit 3; } ;;
  eval)  export OUTROOT=$RUNS/baselines_tuned/$D/${B}_$TAG STAGE=eval ;;
  *) echo "MODE?"; exit 1 ;;
esac
export D B SEED MACREC_DIR NW=2
bash $RB
