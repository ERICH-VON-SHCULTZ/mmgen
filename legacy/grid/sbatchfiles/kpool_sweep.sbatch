#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=kpool_sweep
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/kpool_sweep.out

# ============================================================================================
# Does enlarging K_pool convert MFD's extra coverage into recall?
#
# Step 1 (convert_coverage.out) settled the mechanism from inside the composition stage: on the
# wide_catalog path K_t/K_i prune nothing (composed-at-all = f-root survival = 0.7519), 0 real bugs,
# and 66.3% of missing GTs were composed+scored but ranked beyond K_pool=500. In-model ladder:
# cov@500=0.2634, @2000=0.2991, @5000=0.3346 (0.50 would need K~100,000 -- out of reach).
#
# This job: regenerate P3 (wide_catalog, K_f=200, full t/i support, 808 states) train/val/test ONCE
# with the top-5000 per behavior stored (C = 4*5000 = 20,000 candidates/event), validate each dump at
# generation time (event counts, 808 states, coverage@top-K vs the ladder within sampling error --
# halt on mismatch), then sweep K_pool in {500, 2000, 5000} by truncating the stored score-sorted
# slots, training R2_rw1 (capped 36,864) per K and evaluating on TEST. Chain R2_rw1 capped is REUSED
# from the reranker cache (t3@10 0.2140, coverage 0.2471) -- never retrained.
# Deciding metric = NET t3@10 vs K=500 (newly-covered gain + regained - previously-covered LOST),
# paired CI; verdict fixed in advance; all three K_pool values reported.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MFD_CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
MFD_LINK=/scratch/yw8866/rec-tmall/_cov_mfd.ckpt; ln -sf "$MFD_CKPT" "$MFD_LINK"

OUT=/scratch/yw8866/rec-tmall/coverage_exp
POOLS=$OUT/pools; mkdir -p $POOLS
CACHE=$OUT/cache
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
KS=/scratch/yw8866/rec-tmall/kpool_sweep.py
PB_TEST='77833,2042,1554,1576'
KSTORE=5000
# ladder (in-model, 8,000-event sample, SE~0.006) and the K=500 values of the previous full dumps
COV_TEST='500:0.2618:0.004,2000:0.2991:0.02,5000:0.3346:0.02'
COV_VAL='500:0.2586:0.004'
COV_TRAIN='500:0.3933:0.015'
rm -f $CACHE/P3K*__R2_rw1__capped.pt      # fresh sweep results; P1/CH caches untouched

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

MFD_CORE=\"experiment=tiger_mb_mfd_flat train=false test=true ckpt_path=$MFD_LINK data_dir=$DATA \
semantic_id_path=$MD_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
mfd_behavior_marginalized=true mfd_behavior_weight=1.0 mfd_top_k_shared=200 \
mfd_top_k_text=1024 mfd_top_k_image=1024 +model.mfd_wide_catalog_pool=true \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
mfd_audit_dump=true mfd_audit_pv_cap=300000 +model.mfd_dump_memory=true mfd_pool_size=$KSTORE\"

# gen SPLIT BATCH COVSPEC -> FORCE-generate + validate one split (top-$KSTORE stored, with memory)
gen () {
  SP=\$1; BATCH=\$2; COVSPEC=\$3
  case \$SP in
    test)  FOLD=''; NEED=83005;  MAXE=200000; PB='$PB_TEST' ;;
    val)   FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation\"; NEED=83005;  MAXE=200000; PB='' ;;
    train) FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training\";   NEED=100000; MAXE=100000; PB='' ;;
  esac
  LIMIT=\$(( (NEED + BATCH - 1) / BATCH + 50 ))
  OUTF=$POOLS/p3k_\${SP}.pt
  rm -f \$OUTF
  echo \"=== p3k \$SP: K_pool stored=$KSTORE batch=\$BATCH limit=\$LIMIT (cap \$(( LIMIT*BATCH )) >= \$NEED) [regenerating] ===\"
  python /scratch/yw8866/GRID/src/train.py \$MFD_CORE mfd_audit_max_events=\$MAXE +trainer.limit_test_batches=\$LIMIT \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=\$BATCH \$FOLD mfd_audit_path=\$OUTF
  [ -f \$OUTF ] || { echo \"** ABORT: \$OUTF not produced **\"; exit 1; }
  python $VAL --dump \$OUTF --expect_n \$NEED --expect_perbeh \"\$PB\" --expect_states 808 --expect_cov \"\$COVSPEC\" \
    || { echo \"** HALT: \$OUTF failed validation (event count / per-behavior / 808 states / coverage ladder) **\"; exit 1; }
}

echo '################ generate P3 (wide, top-5000 stored) test / val / train, validated at generation time ################'
gen test  8 '$COV_TEST'
gen val   8 '$COV_VAL'
gen train 8 '$COV_TRAIN'

echo '################ K_pool sweep {500, 2000, 5000}: R2_rw1 capped per K, chain reused ################'
python $KS --tr $POOLS/p3k_train.pt --va $POOLS/p3k_val.pt --te $POOLS/p3k_test.pt --sid $MD_SID \
    --ks 500,2000,5000 --cache $CACHE --chain_key CH__R2_rw1__capped.pt \
    --ladder 500:0.2634,2000:0.2991,5000:0.3346 \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --out_dir $OUT
"
