#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=inject
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/inject_rerank.out

# ============================================================================================
# VALIDITY CHECK for the goal's criterion (d), "a mechanism chain cannot replicate".
# Almost every hit either system gets is a page-view REPEAT or a conversion, and MFD's whole lead is
# repeat coverage (its pool holds 13,662 of the 14,140 pv repeats, chain's 13,120). Adding the user's own
# history items to the candidate set is standard repeat-candidate generation that costs no decoder states.
# This job gives BOTH arms exactly that -- history tuples injected into every behavior branch, deduplicated,
# with an UNINFORMATIVE generator score identical in construction for both arms -- trains the same
# reranker (R2_rw1 capped, 3 seeds) and compares against the non-injected baselines (seedvar/cache).
# If chain+injection matches MFD, the current lead is replicable by a heuristic; the lead that survives
# with both arms injected is the part attributable to MFD.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/inject_exp; mkdir -p $OUT
for f in $POOLS/p3k_train.pt $POOLS/p3k_val.pt $POOLS/p3k_test.pt \
         $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt \
         /scratch/yw8866/rec-tmall/seedvar/cache/MFD__seed2.pt /scratch/yw8866/rec-tmall/seedvar/cache/CH__seed2.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done
rm -f $OUT/inject_report.txt
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
python /scratch/yw8866/rec-tmall/inject_rerank.py --spec $OUT/spec.json --cache $OUT/cache --seeds 0,1,2 \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --out_dir $OUT
"
