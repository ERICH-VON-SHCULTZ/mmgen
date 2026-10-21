#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=fusion_ordering
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_fusion_and_ordering.out

# ============================================================================================
# TWO tasks in one job, both offline on existing dumps (no retraining).
#
# TASK 1  promote_fusion.py: fit the 4-term score fusion on the eval(VAL) split, PERSIST the weights
#         to mfd_fusion_weights.pt (consumed by the model switch +model.mfd_learned_fusion=true and by
#         the listwise reranker), and re-report the FULL metric set (hand-set vs learned, paired) with
#         honest per-behavior breakdowns + behavior-head P/R/F1. Coverage@pool asserted invariant.
#         A behavior-weighted fit is reported as a SECONDARY variant (not the headline).
#         The learned fusion is now a config switch on the MFD model (mfd_learned_fusion /
#         mfd_fusion_weights), applied as a coverage-invariant re-rank of the FIXED pool.
#
# TASK 2  attribute_ordering_gap.py: attribute the in-pool ordering gap (MFD P(rank<10|cov) 0.7806 vs
#         chain 0.8581). Margins, crowding sweep, spread, rank bands, SigLIP2 near-miss similarity of
#         displacers, and the DIVERSITY-cap vs SIZE-cap subsampling sweeps that test the "cost of
#         diversity" hypothesis against the "pool-size artefact" one. Uses the pool-comparison dumps
#         and the matched-compute arms. Sweeps reported in full; nothing tuned toward a conclusion.
# ============================================================================================

SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt          # MD-RQ-VAE
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt       # chain
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
MFD_POOL=/scratch/yw8866/rec-tmall/mfd_pool_cmp_dump.pt
CHAIN_POOL=/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt
MC=/scratch/yw8866/rec-tmall/matched_compute
TXT_NPY=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384.npy
TXT_IDS=/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384_ids.pkl
IMG_NPY=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384.npy
IMG_IDS=/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384_ids.pkl

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

echo '################################## TASK 1: adopt the learned score fusion ##################################'
python /scratch/yw8866/rec-tmall/promote_fusion.py \
    --val_dump $VAL_DUMP --test_dump $TEST_DUMP --wb 1.0 \
    --weights_out /scratch/yw8866/rec-tmall/mfd_fusion_weights.pt \
    --out_dir /scratch/yw8866/rec-tmall/promote_fusion --seed 0

echo '################################## TASK 2: attribute the in-pool ordering gap ##############################'
python /scratch/yw8866/rec-tmall/attribute_ordering_gap.py \
    --mfd_dump $MFD_POOL --chain_dump $CHAIN_POOL \
    --mfd_sid $SID --chain_sid $TG_SID \
    --text_npy $TXT_NPY --text_ids $TXT_IDS --image_npy $IMG_NPY --image_ids $IMG_IDS \
    --extra_arm 'MFD@808' $MC/mc_mfd_808.pt $SID \
    --extra_arm 'chain@808' $MC/mc_chain_808.pt $TG_SID \
    --extra_arm 'MFD@4008' $MC/mc_mfd_4008.pt $SID \
    --extra_arm 'chain@4008' $MC/mc_chain_4008.pt $TG_SID \
    --wb 1.0 --siglip_sample 4000 --out_dir /scratch/yw8866/rec-tmall/ordering_gap --seed 0
"
