#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=cf_rerank
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/cf_rerank.out

# ============================================================================================
# Collaborative-filtering features for the reranker: generic gain or MFD-specific?  Offline only,
# both generative models frozen. No new dumps: MFD = existing P3 top-5000 dumps truncated to
# K_pool=2000 (current best, t3@10 0.2155 / cov 0.2968); chain = existing 808-state dumps (K_pool=500,
# t3@10 0.2140 / cov 0.2471). Reranker R2_rw1, capped 36,864, identical protocol in every cell.
#
# Feature groups (statistics from the TRAIN dump's HISTORY items only; LOO on train; z-scored with
# TRAIN-candidate stats; joins on packed SID keys verified by brute-force recount -- job halts if any
# recount disagrees):
#   CF-A  L0 (shared-code) co-occurrence, PPMI, per history behavior max/mean        (8 features)
#   CF-B  L0 match count / recency-weighted sum / max recency per behavior            (12)
#   CF-C  item(tuple)-item co-occurrence, count>=2, cosine, top-100/item; max/mean/top-3 (12)
# Chain uses its own L0 codes (1024-wide) and its own SID tuples.
#
# Stage 1 PRE-CHECK: per feature, univariate AUC(candidate == GT) + coverage on TRAIN pools, both
#   arms; a group whose best |AUC-0.5| < 0.01 is reported and its cells skipped.
# Stage 2 GRID (both arms): baseline, +A, +B, +A+B, +C, +all. Fit train / select VAL / report TEST.
# Report: full metrics, per-behavior, four diagnostic cells, VERDICT 1 ABSOLUTE (paired bootstrap gain
# over the arm's own baseline), VERDICT 2 DIFFERENTIAL (MFD gain / chain gain, paired CI of the gain
# difference on user_id-joined events, lead with the feature), shared-code expectation check.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/cf_exp; mkdir -p $OUT
CACHE=$OUT/cache
CF=/scratch/yw8866/rec-tmall/cf_rerank.py
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'

for f in $POOLS/p3k_train.pt $POOLS/p3k_val.pt $POOLS/p3k_test.pt \
         $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done
rm -f $OUT/cf_report.txt $OUT/precheck_*.json

cat > $OUT/spec.json <<JSON
{
 "MFD": {"tr": "$POOLS/p3k_train.pt", "va": "$POOLS/p3k_val.pt", "te": "$POOLS/p3k_test.pt", "sid": "$MD_SID", "kpool": 2000},
 "CH":  {"tr": "$CHD/chain808_train_dump.pt", "va": "$CHD/chain808_val_dump.pt", "te": "$CHD/chain808_test_dump.pt", "sid": "$TG_SID", "kpool": null}
}
JSON

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

echo '################ assert the two TEST dumps (83,005 events, per-behavior counts, 808 states) ################'
python $VAL --dump $POOLS/p3k_test.pt          --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1
python $VAL --dump $CHD/chain808_test_dump.pt  --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1

COMMON=\"--spec $OUT/spec.json --cache $CACHE --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 \
--epochs 15 --patience 3 --min_delta 0.0 --batch 8 --eval_batch 16 --boot 2000 --seed 0 --min_cnt 2 --topn 100 --out_dir $OUT\"

echo '################ STAGE 1: CF statistics + join verification + univariate PRE-CHECK, both arms ################'
python $CF --stage precheck \$COMMON || { echo '** pre-check / join verification failed -- halting before any training **'; exit 1; }

echo '################ STAGE 2: ablation grid, both arms (baseline, +A, +B, +A+B, +C, +all) ################'
python $CF --stage all \$COMMON
"
