#!/bin/bash
# §20 Task 2, stage 2 (CPU): pick lambda on ARTS VALIDATION (P2a = 0.1 vs P2b = 0.5: the higher valid HR@10 of each run's best
# fused config in dense_eval/chosen.json "overall", tie NDCG@10), then submit Instruments + Games: one pack with the winning
# lambda AND the other one (the other is only there to keep the GPU above the cluster's utilisation threshold; it is not
# evaluated or used for any choice), the standard eval + dense sweep (valid) + dense apply (test, H200) of the winner.
# Writes mmgen_runs/geo/p2_lambda_choice.json and appends the job ids to logs/round20_jobs_v2.txt.
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=mmgen_p2_stage2
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen; R=/scratch/yw8866/mmgen_runs
W=$(python3 - <<PY
import json
c = {v: json.load(open(f"$R/mfd/Arts/mfd122_mv_geo{v}/seed42/dense_eval/chosen.json"))["overall"] for v in ("P2a", "P2b")}
win = max(c, key=lambda v: (c[v]["hr10"], c[v]["ndcg10"]))
json.dump({"rule": "max Arts valid HR@10 of the best fused config, tie NDCG@10", "candidates": c, "winner": win},
          open("$R/geo/p2_lambda_choice.json", "w"), indent=1)
print(win)
PY
)
O=$([ $W = P2a ] && echo P2b || echo P2a)
echo "winner $W (filler $O)"
cd $REPO
P=$(sbatch --parsable --export=ALL,RUNS="geo:Instruments:mfd:$W geo:Games:mfd:$W geo:Instruments:mfd:$O geo:Games:mfd:$O" sbatch/mfd/pack_mfd_train.sbatch)
L=$R/logs/round20_jobs_v3.txt; echo "stage2 winner $W pack $P" >> $L
for D in Instruments Games; do
  E=$(sbatch --parsable --dependency=afterok:$P --partition=a100_tandon,h100_tandon,h200_tandon --time=12:00:00 --export=ALL,D=$D,KIND=mfd,RUN=mfd122_mv_geo$W,CK=$R/mfd/$D/mfd122_mv_geo$W/seed42 sbatch/geo/eval_geo.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$P --time=16:00:00 --export=ALL,D=$D,RUN=mfd122_mv_geo$W,STAGE=sweep sbatch/mfd/dense_eval.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S --partition=h200_tandon --time=6:00:00 --export=ALL,D=$D,RUN=mfd122_mv_geo$W,STAGE=apply sbatch/mfd/dense_eval.sbatch)
  echo "stage2 $D $W eval $E dense sweep $S apply $A" >> $L
done
echo done
