#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=seedvar
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/seed_variance.out

# ============================================================================================
# Seed variance of the current best setting: MFD (P3 wide pool, K_f=200, full t/i support, 808 states,
# K_pool=2000) vs chain (808 states, K_pool=500); reranker R2_rw1, capped 36,864; same code path as the
# cf_rerank baseline cells (seed 0: 0.2161 vs 0.2139). Seeds 0-4; seed 0 is reused from the cf_rerank
# cache, seeds 1-4 are trained (~35 min MFD + ~11 min chain each). Per-seed leads with paired event
# bootstrap CIs, then mean / std / min / max across seeds, the t-interval of the mean lead, and whether
# seed noise or event-sampling noise dominates. No new dumps.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/seedvar; mkdir -p $OUT
CACHE=$OUT/cache
SV=/scratch/yw8866/rec-tmall/seed_variance.py
SEED0=/scratch/yw8866/rec-tmall/cf_exp/cache
SEEDS=0,1,2,3,4

for f in $POOLS/p3k_train.pt $POOLS/p3k_val.pt $POOLS/p3k_test.pt \
         $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done

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

python $SV --spec $OUT/spec.json --cache $CACHE --seeds $SEEDS --seed0_cache $SEED0 \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --out_dir $OUT
"
