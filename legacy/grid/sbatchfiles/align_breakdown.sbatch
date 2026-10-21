#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=4:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=align_breakdown
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/align_breakdown.out

# ============================================================================================
# Repair the cross-arm event alignment that blocked the matched head-to-head, then produce the
# grouped breakdown. Pure offline analysis on the saved rank files + pool dumps -- no training,
# no regeneration.
#
# DIAGNOSED CAUSE: both arms hold the same 83,005 test events (gt_b MULTISETS are equal, per-behavior
# counts identical) but in different ROW ORDER -- gt_b agreement in row order is only 0.893, first
# mismatch at row 82. The two arms were dumped at different test batch sizes and tf.data interleaving
# makes row order batch-size dependent. The previous compare used torch.equal(user_id_mfd,
# user_id_chain), an ORDER-SENSITIVE test, so it (correctly) refused to proceed. The repair is to
# JOIN on the stored per-event identifier (user_id; fingerprint fallback) instead of assuming row
# alignment, and to PROVE the join with post-join gt_behavior agreement.
#
# Reports: what each dump stores as its event id, whether orders differ, whether any event is missing
# from either side, which key was used, and how many events joined. Then the partition (BOTH /
# MFD-only / chain-only), the per-group decomposition of each arm's t3@10 gain (parts sum to the
# whole), the pre-rerank rank-band split (10-49 / 50+), per behavior, the MFD entropy-z bucket lift,
# and bootstrap CIs over EVENTS for each gain and for the MFD-minus-chain post-rerank difference.
#
# SID widths are READ from each semantic_ids.pt (chain 1024/1024/1024, MFD 2048/1024/1024) and every
# dumped code id is asserted below its level width -- an earlier analysis was invalidated by assuming
# the wrong layout.
# ============================================================================================

MR=/scratch/yw8866/rec-tmall/matched_rerank
MFD_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
CH_DUMP=$MR/chain808_test_dump.pt          # produced by matched_rerank.sbatch (not matched_compute/)
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt

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

python /scratch/yw8866/rec-tmall/align_and_breakdown.py \
    --mfd_dump $MFD_DUMP --chain_dump $CH_DUMP \
    --mfd_ranks $MR/ranks_mfd_natural.pt --chain_ranks $MR/ranks_chain_natural.pt \
    --mfd_ranks_sub $MR/ranks_mfd_sub.pt \
    --md_sid $MD_SID --tg_sid $TG_SID \
    --boot 2000 --seed 0 --out_dir $MR
"
