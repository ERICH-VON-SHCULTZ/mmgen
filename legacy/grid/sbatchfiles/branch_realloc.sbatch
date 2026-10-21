#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=branch_realloc
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/branch_realloc.out

# ============================================================================================
# Offline simulation: would reallocating the per-branch candidate budget by branch entropy improve
# recall? Feasibility CEILING estimate -- no regeneration, no training, score UNCHANGED.
#
#   STAGE 1  coverage ceiling: GT survives reallocated branch pruning iff tf_t_rank < K_t'(branch)
#            and tf_i_rank < K_i'(branch) and f_star_rank < K_f (exact from the dumped GT ranks).
#            Budget preserved per event; rules = proportional-to-H / exp(H) / entropy-rank, and a
#            two-tier flat/peaked multiplier sweep. Reports survival + newly recovered/lost.
#   STAGE 2  the decisive test: rank each newly-covered GT with the CURRENT score against the pool
#            (OPTIMISTIC -- the real reallocated pool would add unseen flat-branch competitors, so
#            these top-k fractions are an upper bound). Net recall@10. DECISION: < ~20% of newly-
#            covered reaching top-10 => coverage-without-recall, not worth implementing.
#
# Branch entropy is the truncated (pooled-candidate) entropy; the GT-branch must be present in the
# pool (excluded fraction reported). Asserts test split 83,005 (pv/fav/cart/buy 77,833/2,042/1,554/
# 1,576). Refs (do NOT tune toward): recall@10 0.2041 hand-set / 0.2065 fusion; coverage 0.2615.
# ============================================================================================

TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt   # has tf ranks/logps + pool + 83,005 split

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

python /scratch/yw8866/rec-tmall/branch_realloc.py \
    --test_dump $TEST_DUMP --wb 1.0 --seed 0 --out_dir /scratch/yw8866/rec-tmall/branch_realloc
"
