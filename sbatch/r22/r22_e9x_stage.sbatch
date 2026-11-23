#!/bin/bash
# §22.9 E9k4 cross-dataset stage (CPU), after the Arts seed-43/44 E9k4 report. Decision on VALIDATION only (no test numbers):
#   E9k4 "holds" on Arts  <=>  valid HR@10 (plain, K_f = 50) > A01's of the same seed on BOTH seeds 43 and 44.
# If it holds: Instruments + Games seed 42 before anything else (no promotion on Arts alone): E8/E9 gate per dataset with
# VARS=E9k4 (checks + dry run + 5-step smoke) -> one pack (Instruments:E9k4:42 Games:E9k4:42) -> H200 eval + fdump -> common
# report (tools/r22_runs_report.py E9: valid/test vs A01 + E6 repeat-f / new-f split). Decision -> geo/r22_e9x_stage.json,
# job IDs -> logs/round22_jobs.txt.
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=mmgen_r22_e9xstage
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs; L=$RUNS/logs/round22_jobs.txt
HOLDS=$(python3 - <<'E'
import json
R = "/scratch/yw8866/mmgen_runs/mfd/Arts"
out, ok = {}, True
for S in (43, 44):
    v = json.load(open(f"{R}/r22_E9k4/seed{S}/r21_eval/eval.json"))["valid"]["plain"]["hr10"]
    a = json.load(open(f"{R}/r21_A01/seed{S}/r21_eval/eval.json"))["valid"]["plain"]["hr10"]
    out[f"seed{S}"] = {"E9k4_valid_hr10": v, "A01_valid_hr10": a, "diff": v - a}
    ok &= v > a
out["rule"] = "E9k4 valid HR@10 > A01 valid HR@10 on both Arts seeds 43 and 44 (validation only)"
out["holds"] = bool(ok)
json.dump(out, open("/scratch/yw8866/mmgen_runs/geo/r22_e9x_stage.json", "w"), indent=1)
print(json.dumps(out))
E
)
echo "$HOLDS"
if ! echo "$HOLDS" | grep -q '"holds": true'; then
  echo "E9k4 x-dataset stage: E9k4 does not hold on Arts seeds 43/44 validation ($HOLDS); Instruments/Games not run" | tee -a $L; exit 0
fi
# run_r22_train.sh needs BOTH gate files per dataset: geo/<D>/r22_check.json (round-22 gate, r22_gate.sbatch) and
# geo/<D>/r22_e89_check.json (E8/E9 gate); only Arts had the first (2026-11-22 pack 24090105 failed on it at start)
G0=""; for D in Instruments Games; do [ -f $RUNS/geo/$D/r22_check.json ] || G0="$G0:$(sbatch --parsable --export=ALL,D=$D,VARS=E9k4 $REPO/sbatch/r22/r22_gate.sbatch)"; done
G1=$(sbatch --parsable --export=ALL,D=Instruments,VARS=E9k4 $REPO/sbatch/r22/r22_e89_gate.sbatch)
G2=$(sbatch --parsable --export=ALL,D=Games,VARS=E9k4 $REPO/sbatch/r22/r22_e89_gate.sbatch)
P=$(sbatch --parsable --dependency=afterok:$G1:$G2$G0 --export=ALL,RUNS="Instruments:E9k4:42 Games:E9k4:42" $REPO/sbatch/r22/pack_r22_train.sbatch)
EV=""
for D in Instruments Games; do
  e=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_E9k4,SEED=42,STEP=eval $REPO/sbatch/r22/r22_run.sbatch)
  f=$(sbatch --parsable --dependency=afterok:$P --export=ALL,D=$D,RUN=r22_E9k4,SEED=42,STEP=fdump $REPO/sbatch/r22/r22_run.sbatch)
  EV="$EV:$e:$f"
done
R=$(sbatch --parsable --dependency=afterok$EV --export=ALL,TOOL="r22_runs_report.py E9" $REPO/sbatch/r21/r21_tool.sbatch)
echo "E9k4 x-dataset stage: holds on Arts 43/44 valid; r22 gates ${G0#:} e89 gates $G1 $G2 pack $P (Instruments:E9k4:42 Games:E9k4:42) evals+fdumps ${EV#:} report $R" | tee -a $L
