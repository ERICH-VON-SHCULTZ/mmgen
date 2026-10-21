#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=branch_calib
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/branch_calibrate.out

# ============================================================================================
# Test whether MFD's in-pool ordering deficit is a SCORE COMPARABILITY problem across f-branches,
# and fix it by calibrating branch scores before merging candidates. Pure offline rescoring on the
# existing dumps -- no retraining, no regeneration.
#
#   STAGE 1  diagnostic (go/no-go): per f-branch entropy / top-1 / eff-count / size of P(t|f),P(i|f);
#            within-event branch-entropy spread; and P(rank<10|cov) bucketed by the GT branch's
#            within-event entropy z. Monotone decline (GT in a flatter-than-peers branch ranks worse)
#            confirms the scale hypothesis. Chain analog is a best-effort proxy (its dump lacks the
#            per-level conditional logps -- stated in the report).
#   STAGE 2  three calibration variants (coverage invariant): (a) within-branch z-standardization
#            (+ rank-based robust variant), (b) entropy-driven temperature tau_f (fit on VAL),
#            (c) learned branch MLP -> (a_f,b_f) (fit on TRAIN, selected on VAL). Winner is also
#            combined with the adopted learned fusion weights (mfd_fusion_weights.pt).
#
# Branch statistics are computed from the POOLED candidates' component logps already in the dumps
# (pool_lpf/lpt/lpi) -- no regeneration. Asserts test split (83,005; pv/fav/cart/buy
# 77,833/2,042/1,554/1,576) and coverage@pool 0.2615 invariance. Metrics: P(rank<10|cov) cur 0.7945
# / chain 0.8581 / break-even 0.81; rank10-11 gap cur 0.2063 / chain 0.3213; #within +/-0.5 cur 27.50
# / chain 22.09. Do NOT tune toward the reference points.
# ============================================================================================

CHAIN_POOL=/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt
TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt

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

python /scratch/yw8866/rec-tmall/branch_calibrate.py \
    --chain_pool $CHAIN_POOL \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --fusion_weights /scratch/yw8866/rec-tmall/mfd_fusion_weights.pt \
    --wb 1.0 --seed 0 --out_dir /scratch/yw8866/rec-tmall/branch_calib
"
