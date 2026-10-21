#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --time=03:00:00
#SBATCH --mem=250GB
#SBATCH --job-name=tau_sel
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/tau_select.out

# ============================================================================================
# GOAL (rec-tmall/GOAL.md): MFD must beat chain on t3@10 by >= +0.01 absolute. Lead today +0.00196.
#
# ROUND 1 (job 18472175) CLOSED THE RERANKING AVENUE. Four mechanisms -- branch-posterior features,
# a learned per-candidate base anchor, 9x deep-band weighting, and a GT-vs-top-64 objective -- all
# came back null on VAL (best +0.0001, inside seed noise), and MFD-ONLY promotions stayed at 202-235
# against a required 790-890. The gate had complete freedom to discount the generator's score and did
# nothing, so the score form was not the constraint; hn64 made deep promotion the literal objective and
# barely moved it, so the objective was not either.
#
# WHAT IS LEFT, AND WHY IT IS DIFFERENT. Every experiment in the ledger -- learned fusion, branch
# calibration, MADR, BPF -- rescored a FIXED pool; coverage@pool was an asserted invariant. The pool
# SELECTION score has never been varied:
#     score = w_b*logP(b) + logP(f|b) + (1/tau)*[ logP(t|f,b) + logP(i|f,b) ],  tau frozen at 1.0
# Lowering tau amplifies the two modality terms against the shared-code term -- exactly the MFD-ONLY
# signature (weak f-root, sharp t and i). Because it changes WHICH candidates are selected, the ground
# truths it admits arrive at GOOD ranks, unlike K_pool widening which appended them deep and converted
# at 2.3%. It is also MFD-specific: chain has no tau, its levels are sequential.
#
# THIS JOB is the cheap screen, not the experiment: the p3k dump stores the top-5,000 per behavior with
# their component log-probs, so every tau can be re-scored and re-selected offline, exactly, in one
# pass. Reports per tau: coverage@2000, base t3@10 (no reranker), GT rank distribution, and the
# MFD-ONLY population joined against chain's pool membership by user_id.
# CAVEAT the job prints itself: the stored 5,000 were chosen by the tau=1 score, so these are LOWER
# BOUNDS on full-catalog selection at tau!=1. A tau that wins here justifies a re-dump; one that loses
# here cannot be rescued by one.
# ============================================================================================

POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
OUT=/scratch/yw8866/rec-tmall/tau_exp; mkdir -p $OUT
TS=/scratch/yw8866/rec-tmall/tau_select.py
CHR=/scratch/yw8866/rec-tmall/seedvar/cache/CH__seed0.pt

for f in $POOLS/p3k_test.pt $CHR; do [ -f $f ] || { echo "** missing $f **"; exit 1; }; done

module purge

singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f
export TF_CPP_MIN_LOG_LEVEL=2

python $TS --dump $POOLS/p3k_test.pt --chain_result $CHR \
    --taus 0.25,0.4,0.5,0.7,0.85,1.0,1.25,1.5,2.0 --wb 1.0 --kpool 2000 --chunk 1000 --out_dir $OUT
"
