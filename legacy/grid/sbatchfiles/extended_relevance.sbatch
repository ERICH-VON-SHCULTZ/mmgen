#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=ext_relevance
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/extended_relevance.out

# ============================================================================================
# EXTENDED-RELEVANCE re-evaluation of the chain vs MFD candidate pools. Pure offline analysis on
# the EXISTING pool-comparison dumps (no regeneration, no training).
#
# The test set is last-item leave-one-out, so NO interactions exist after the target -> the
# extended relevance set is the user's LAST N interactions (target + N-1 preceding), dedup on the
# (behavior, l0, l1, l2) 4-tuple. N=1 == the single next item -> reproduces the single-GT numbers
# (asserted). This is a "recent-interest cluster" probe, not literal future (the N-1 extra items
# are recent past); fair for MFD-vs-chain since both arms can re-surface recent interests.
#
# Reads the testing tfrecords (per user_id: full time-ordered item sequence + behaviors), maps
# each item id -> that arm's SID (MD-RQ-VAE for MFD, tiger_1024 for chain), joins to the dumps by
# user_id, and reports per arm / per N in {1,3,5,10,20} / per behavior: mean set size (collapse),
# coverage@pool, hit@10, hit@20, recall@10 (set denominator), P(hit@10|covered), and the out-of-
# top10 rank distribution of in-pool relevant items.
# ============================================================================================

MFD_DUMP=/scratch/yw8866/rec-tmall/mfd_pool_cmp_dump.pt
CH_DUMP=/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt
TEST_DIR=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop/testing
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

python /scratch/yw8866/rec-tmall/analyze_extended_relevance.py \
    --mfd_dump $MFD_DUMP --chain_dump $CH_DUMP \
    --test_dir $TEST_DIR --md_sid $MD_SID --tiger_sid $TG_SID \
    --ref_mfd_r10 0.2041 --ref_chain_r10 0.2127
"