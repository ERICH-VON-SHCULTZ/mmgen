#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=kpool_k5000
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/kpool_sweep_k5000.out

# ============================================================================================
# Re-run of the K_pool sweep, K_pool=5000 ONLY. Run 1 (kpool_sweep.out, copied to
# coverage_exp/kpool_sweep_run1.out) generated + validated the top-5000 dumps and finished K=500 and
# K=2000 (cached in coverage_exp/cache/P3K500__ / P3K2000__R2_rw1__capped.pt), then died AFTER the
# K=5000 reranker had trained 9 epochs: Pool.branch_ent (a post-training DIAGNOSTIC, the entropy-z
# buckets) moved the whole N x C = 83,005 x 20,000 candidate table to the GPU at once -> CUDA OOM
# (74 GB). Fix: branch_ent now runs in 2,048-event chunks (exact: every entropy group lives inside one
# event). Nothing else changed: same dumps, same reranker, same protocol, same seed.
# This job: NO regeneration, NO cache wipe; re-validates the three existing dumps, prints the K=500 /
# K=2000 results of run 1 as reference, then the sweep script loads K=500/2000 from cache and trains
# only K=5000, and prints the full three-K report (deciding metric = NET t3@10 vs K=500).
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
OUT=/scratch/yw8866/rec-tmall/coverage_exp
POOLS=$OUT/pools
CACHE=$OUT/cache
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
KS=/scratch/yw8866/rec-tmall/kpool_sweep.py
PB_TEST='77833,2042,1554,1576'
COV_TEST='500:0.2618:0.004,2000:0.2991:0.02,5000:0.3346:0.02'
COV_VAL='500:0.2586:0.004'
COV_TRAIN='500:0.3933:0.015'

for f in $CACHE/P3K500__R2_rw1__capped.pt $CACHE/P3K2000__R2_rw1__capped.pt $CACHE/CH__R2_rw1__capped.pt \
         $POOLS/p3k_test.pt $POOLS/p3k_val.pt $POOLS/p3k_train.pt; do
  [ -f $f ] || { echo "** missing $f -- this job only re-runs K=5000 and needs run 1's artefacts **"; exit 1; }
done
rm -f $CACHE/P3K5000__R2_rw1__capped.pt      # only the K=5000 result is recomputed

echo '################ REFERENCE (run 1, kpool_sweep.out): K_pool=500 and K_pool=2000 ################'
cat $OUT/kpool_sweep_run1_k500_k2000_reference.txt
echo '################ end of reference ################'

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

echo '################ re-validate the existing top-5000 dumps (no regeneration) ################'
python $VAL --dump $POOLS/p3k_test.pt  --expect_n 83005  --expect_perbeh '$PB_TEST' --expect_states 808 --expect_cov '$COV_TEST'  || exit 1
python $VAL --dump $POOLS/p3k_val.pt   --expect_n 83005  --expect_perbeh ''         --expect_states 808 --expect_cov '$COV_VAL'   || exit 1
python $VAL --dump $POOLS/p3k_train.pt --expect_n 100000 --expect_perbeh ''         --expect_states 808 --expect_cov '$COV_TRAIN' || exit 1

echo '################ K_pool sweep: K=500 / K=2000 from cache, K=5000 trained; chain reused ################'
python $KS --tr $POOLS/p3k_train.pt --va $POOLS/p3k_val.pt --te $POOLS/p3k_test.pt --sid $MD_SID \
    --ks 500,2000,5000 --cache $CACHE --chain_key CH__R2_rw1__capped.pt \
    --ladder 500:0.2634,2000:0.2991,5000:0.3346 \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --out_dir $OUT
"
