#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=rerank_sweep2
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/rerank_sweep2.out

# ============================================================================================
# THREE TASKS IN ONE JOB. Offline reranker work only; both generative models stay frozen.
#
# TASK A -- SETTLE SIGNIFICANCE (blocking, runs first in the report)
#   Two runs gave contradictory CIs for MFD-minus-chain: aligned/baseline [+0.0004,+0.0018] vs
#   sweep/VAL-best [-0.0027,+0.0046]. DIAGNOSIS (verified in code): BOTH bootstraps resampled EVENTS
#   at B=2000, so neither the resampling unit nor the replicate count is the cause. The cause is that
#   sweep_rerank.py never ALIGNED the arms -- the same bootstrap index hit DIFFERENT events in each
#   arm, converting a paired test into an effectively independent one (Var(m-c)=Var(m)+Var(c) rather
#   than Var(m)+Var(c)-2Cov). Comparing each arm's own VAL-best config (A2 vs A1) also decorrelates
#   the hit vectors -- real but secondary. The job reports the CI computed BOTH ways so the cause is
#   demonstrated, not asserted, and then fixes the protocol:
#     same config on both arms; events joined on user_id; PAIRED bootstrap; B=2000; diff + CI +
#     P(MFD>chain). This protocol is stated in the output for future runs to match.
#
# TASK B -- FIX POPULARITY (it collapsed t3@10 0.2144 -> 0.1129, P(r<10|cov) 0.82 -> 0.43)
#   Three defects, each reported with evidence: (1) SELF-INCLUSION -- counts came from the TRAIN GT
#   tuples, so a training event's own label inflated its own feature (circular; vanishes at test).
#   Fixed with leave-one-out counts on train. (2) NO STANDARDISATION -- raw log1p counts are
#   long-tailed and on a different scale from the other 28 features. Fixed by z-scoring with TRAIN
#   statistics. (3) JOIN -- verified (sorted-unique keys + exact-match); the lookup hit-rate is
#   printed so a misaligned join would be visible. Scope was already TRAIN-only and is re-asserted.
#   B3/B4 stack on B2, so fixing B2 should restore all three. Re-run B2/B3/B4 on BOTH arms.
#   B5 = branch entropy + its z-score within the pool: MFD-ONLY and reported as such (chain has no
#   per-branch conditional), rather than pretending the arms are symmetric on that axis.
#
# TASK C -- TARGET DEEP / MFD-ONLY GTs (generic capacity is saturated: 466k -> 3.46M params moved
#   t3@10 by 0.0002)
#   C1 rank-weighted loss (3 schedules), C2 candidate self-attention (top-M relative modelling; every
#   candidate keeps its cross-attention delta, only the CONTEXT is truncated, since full-pool
#   self-attention is O(C^2)), C3 hard-negative windows (50, 200), C4 best combination.
#
# EVERY configuration reports the FOUR diagnostic cells (BOTH/10-49, BOTH/50+, own-only/10-49,
# own-only/50+) -- aggregate t3@10 has moved <0.001 across many configs, so the cells are the
# sensitive read. Collapsed runs are FLAGGED and EXCLUDED from trend analysis rather than treated as
# data points (the previous sweep fitted a slope through collapsed runs and produced two
# contradictory conclusions from the same numbers).
#
# Protocol: 808 decoder states, K_pool 500, same optimizer/schedule/early-stopping; fit train, select
# VAL, report TEST; every config in BOTH natural and capped(36,864) settings. Asserts: TEST 83,005
# (pv 77,833 / fav 2,042 / cart 1,554 / buy 1,576); coverage unchanged by reranking (MFD 0.2615,
# chain 0.2471); SID widths READ from the files (MFD 2048/1024/1024, chain 1024/1024/1024).
# RESUMABLE: every (arm, config, setting) is cached under $OUT/cache -- just resubmit if it runs out
# of wall clock.
# ============================================================================================

MR=/scratch/yw8866/rec-tmall/matched_rerank
MFD_TR=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
MFD_VA=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
MFD_TE=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
CH_TR=$MR/chain808_train_dump.pt
CH_VA=$MR/chain808_val_dump.pt
CH_TE=$MR/chain808_test_dump.pt
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
OUT=/scratch/yw8866/rec-tmall/sweep2

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

python /scratch/yw8866/rec-tmall/sweep2_rerank.py \
    --mfd_tr $MFD_TR --mfd_va $MFD_VA --mfd_te $MFD_TE \
    --ch_tr $CH_TR --ch_va $CH_VA --ch_te $CH_TE \
    --md_sid $MD_SID --tg_sid $TG_SID \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 \
    --epochs 15 --patience 3 --min_delta 0.0 --batch 32 --eval_batch 64 \
    --boot 2000 --seed 0 --out_dir $OUT
"
