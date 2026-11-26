#!/bin/bash
# Round 24, wave 4 (2026-11-26; docs/MFD_REPORT.md §24.5 / §24.6), Arts + Instruments seed 42, all at --nice=500 (below the seed 43/44 chain):
#  §24.5 symmetric baseline grid: C (mql) at lr 3e-4 (P1a was tested at 3e-4; C so far only upward); same selection rule (tools/r24_baselines.py)
#  §24.6 is the lr gain specific to full P1a? lr 2e-3, every other setting the model's own:
#        MD chain decoder (md122_a05_ep500_reassign; baseline path, gate = dry-run diff vs its own settings + 5 real steps)
#        V2-MFD (run_r24_train.sh VAR v2_lr2e3) and P1a without codebook init (noinit at P1a's lr 5e-4 = its default reference, noinit_lr2e3);
#        gate = r24_gate (dry runs, config diff, 5-step smoke, built-model check) -> one pack per dataset -> H200 evals (plain K_f = 50)
#        + the existing V2-MFD default run evaluated with the same K_f = 50 harness (it only had K_f 20 / 100 evals)
#  reports: tools/r24_baselines.py (after the baseline evals), tools/r24_lrspec.py (after every eval)
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round24_jobs.txt; cd $REPO
[ -z "$(git status --porcelain -- macrec sbatch)" ] || { echo "commit macrec/ and sbatch/ first"; exit 1; }
K=--kill-on-invalid-dep=yes; N=--nice=500
echo "=== $(date +%F) round 24 wave 4, commit $(git rev-parse --short HEAD)" | tee -a $L
# baselines: C lr 3e-4, MD chain lr 2e-3
BR="Arts:mql:lr3e4 Instruments:mql:lr3e4 Arts:md122_a05_ep500_reassign:lr2e3 Instruments:md122_a05_ep500_reassign:lr2e3"
G=$(sbatch --parsable $N --export=ALL,RUNS="$BR" sbatch/r24/r24_base_gate.sbatch)
P=$(sbatch --parsable $K $N --dependency=afterok:$G --export=ALL,RUNS="$BR" sbatch/r24/r24_base_pack.sbatch); BE=""
for x in $BR; do BE="$BE:$(sbatch --parsable $K $N --dependency=afterok:$P --export=ALL,RUNS="$x" sbatch/r24/r24_base_eval.sbatch)"; done
F1=$(sbatch --parsable --dependency=afterany$BE --export=ALL,TOOL="r24_baselines.py" sbatch/r21/r21_tool.sbatch)
echo "§24.5/§24.6 baselines (C lr3e4, chain lr2e3): gate $G pack $P evals ${BE#:} | baseline comparison $F1" | tee -a $L
# MFD models
V="v2_lr2e3 noinit noinit_lr2e3"
GA=$(sbatch --parsable $N --export=ALL,D=Arts,VARS="$V" sbatch/r24/r24_gate.sbatch)
GI=$(sbatch --parsable $N --export=ALL,D=Instruments,VARS="$V" sbatch/r24/r24_gate.sbatch)
ME=""; PP=""
for D in Arts Instruments; do
  R=""; for v in $V; do R="$R $D:$v"; done
  Q=$(sbatch --parsable $K $N --dependency=afterok:$GA:$GI --export=ALL,RUNS="${R# }" sbatch/r24/pack_r24_train.sbatch); PP="$PP $Q"
  for v in $V; do
    ME="$ME:$(sbatch --parsable $N --time=02:00:00 $K --dependency=afterok:$Q --export=ALL,D=$D,RUN=r24_$v,SEED=42,STEP=eval sbatch/r22/r22_run.sbatch)"
  done
  ME="$ME:$(sbatch --parsable $N --time=02:00:00 --export=ALL,D=$D,RUN=mfd122_mv_geoV2,SEED=42,STEP=eval sbatch/r22/r22_run.sbatch)"
done
F2=$(sbatch --parsable --dependency=afterany$ME$BE --export=ALL,TOOL="r24_lrspec.py" sbatch/r21/r21_tool.sbatch)
echo "§24.6 MFD models ($V; V2-MFD default eval): gates $GA $GI packs$PP evals ${ME#:} | lr-specificity report $F2" | tee -a $L
