#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=bpf2
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/bpf_round2.out

# ============================================================================================
# GOAL (see rec-tmall/GOAL.md): MFD must beat chain on t3@10 by >= +0.01 ABSOLUTE, stable across
# seeds, by a mechanism chain cannot replicate. Settled 2026-09-21 over 5 seeds: lead +0.00173 +-
# 0.00047. The bar needs +0.00827 more = 687 additional MFD top-10 hits out of 83,005 events.
#
# STEP 0 -- HEADROOM AUDIT (no training, no dumps, seconds of CPU; runs off the cached per-event
# results). Splits the test set by POOL MEMBERSHIP: BOTH / MFD-ONLY / CHAIN-ONLY / NEITHER, and
# reports each population's size, per-arm recovery, contribution to the current lead, the rank
# distribution of MFD's misses, and the recovery rate each population would need to carry the whole
# +0.00827 alone. This decides, before any GPU time, whether the bar is reachable by reranking and
# from where. MFD-ONLY is the only population where a gain moves the LEAD one-for-one -- chain's pool
# does not contain those GTs at all, so it cannot follow.
#
# THE LEVER. recall@10 = coverage x P(rank<10|cov). MFD 0.2968 x 0.7252, chain 0.2471 x 0.8642.
# Coverage is dead (newly covered GTs reach top-10 at 2.3%), conversions are worth at most +0.0031
# in total, and widening does not evict (the 500->2000 pcov drop is composition, ~42 events of
# eviction). So the target is ORDERING, specifically on the MFD-ONLY population.
#
# THIS JOB -- BPF, branch-posterior position features. MFD reads the complete 1024-way P(t|f,b) and
# P(i|f,b) from ONE forward pass per (b,f) root, so a candidate's position INSIDE its branch's
# posterior is free at 808 decoder states; chain would need ~1024 continuations per root for the same
# information. 14 features per candidate (margins to the branch's best t/i, share of the branch's
# pooled mass, branch log-mass / distinct-code count / top1-top2 gap, root margin within the behavior,
# branch size, base-score margin within the branch and within the event), z-scored on train candidates
# and clipped. Computed from the EXISTING dumps -- no regeneration.
# This is not the failed calibration: that REPLACED the absolute log-probs with within-branch z-scores
# and collapsed P(rank<10|cov) 0.78 -> 0.60. BPF ADDS relative position beside the untouched base score.
#
# ROUND 2 -- stack the mechanisms that round 1 isolated, and confirm at 5 seeds.
# Round 1 screened four variants separately: features (+BPF), score form (+BPF_gate), deep weighting
# (+BPF_gate_rw2) and objective (+BPF_gate_hn64). Round 2 combines them and adds the ablation that
# says whether the branch-posterior FEATURES are needed once the score form and objective are fixed:
#   +stack          bpf + gate + hn64, rw1
#   +stack_rw2      bpf + gate + hn64, rw2 (deep bands 9x)
#   +gatehn_nobpf   gate + hn64 WITHOUT the features -> isolates the features' contribution
# Variants come from --variants, so no code edit is needed to change them. Cache is cache2 so round-1
# results are never silently reused under the same key. Seeds 0-4 (5 seeds, matching the seed-variance
# study) because the bar requires the across-seed t-interval to clear +0.01, not just the mean.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/bpf_exp; mkdir -p $OUT
CACHE=$OUT/cache2
BASE=/scratch/yw8866/rec-tmall/seedvar/cache      # baselines, seeds 0-4, same code path
BPF=/scratch/yw8866/rec-tmall/bpf_rerank.py
HR=/scratch/yw8866/rec-tmall/headroom.py
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'
SEEDS=0,1,2,3,4

for f in $POOLS/p3k_train.pt $POOLS/p3k_val.pt $POOLS/p3k_test.pt \
         $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt \
         $BASE/MFD__seed0.pt $BASE/CH__seed0.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done
rm -f $OUT/bpf_round2_report.txt

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

echo '################ assert both TEST dumps (83,005 events, per-behavior counts, 808 states) ################'
python $VAL --dump $POOLS/p3k_test.pt         --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1
python $VAL --dump $CHD/chain808_test_dump.pt --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1

echo '################ ROUND 2: stacked variants, 5 seeds x 2 arms ################'
python $BPF --spec $OUT/spec.json --cache $CACHE --base_cache $BASE --seeds $SEEDS --stage all \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --bar 0.01 --out_dir $OUT \
    --variants '+stack:1,rw1,64,1;+stack_rw2:1,rw2,64,1;+gatehn_nobpf:0,rw1,64,1'
"
