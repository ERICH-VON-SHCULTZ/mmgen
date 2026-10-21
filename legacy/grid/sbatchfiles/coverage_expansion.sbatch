#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=coverage_exp
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/coverage_expansion.out

# ============================================================================================
# RE-TEST COVERAGE EXPANSION UNDER THE CURRENT STRONG RERANKER.  Both generative models FROZEN:
# pool regeneration + offline reranker training only.
#
# ---- BUG FIXED IN THIS REVISION (why the previous attempt aborted) -------------------------
# P2's dumps held 24,000 events instead of 83,005 and the split assertion (correctly) stopped the
# job. Cause: `+trainer.limit_test_batches=6000` MULTIPLIES with `batch_size_per_device`. P1 used
# the config default batch 64 -> 6000*64 = 384,000, so the cap never bound. To avoid OOM on P2's
# dense compose the batch was lowered to 4 -> 6000*4 = 24,000, silently truncating every P2 split.
# P3 at batch 8 was truncated the same way to 48,000 (its files were exactly 2x P2's size).
# FIX: limit_test_batches is now COMPUTED per split as ceil(NEED/BATCH)+50 so it can never bind,
# and both truncated pools were deleted for regeneration.
# HARDENING: every dump is validated at the END OF GENERATION (event count, per-behavior counts,
# decoder-states/item, coverage) by validate_dump.py, which exits non-zero -- so a bad dump fails
# immediately instead of after paying for three generation runs and failing at load time.
#
# ---- POOLS (MFD, K_f=200, K_pool=500) ------------------------------------------------------
#   P1 K_t=200  K_i=50    control (coverage 0.2615) -- cached dumps REUSED, never regenerated
#   P2 K_t=200  K_i=200   widen the weak image branch (branch_i loss 0.1385 vs branch_t 0.0243)
#   P3 K_t=1024 K_i=1024  full support (ceiling 0.7561) -- wide-catalog scoring: branch-by-branch,
#                         catalog-constrained, truncated to K_pool as it goes, so the K_f*K_t*K_i
#                         cross product is NEVER materialised. Peak GPU memory is recorded in the
#                         dump meta and printed by the validator.
# Widening K_t/K_i costs NO extra decoder forwards (both heads already emit full 1024-way softmaxes;
# top-K is a composition-stage choice), so all three sit at nb*(2+K_f) = 808 states -- ASSERTED per
# dump, not assumed. If it drifts, the equal-compute match to chain is broken.
#
# ---- CHAIN ---------------------------------------------------------------------------------
# The beam-allocation sweep is DONE and is not repeated: at 808 states with B1+B2=200 the VAL
# coverage was 180x20 0.2373 / 150x50 0.2443 / 100x100 0.2462 / 50x150 0.2446 / 20x180 0.2390 --
# uniform wins and the whole curve spans only 0.0089. B1=B2=100 is chain's arm and the sweep is
# reproduced in the final report, establishing chain had its best configuration at equal compute.
#
# ---- GRID / PROTOCOL -----------------------------------------------------------------------
# R1 = baseline reranker (466,690 params, 2 layers, d=128, views+hard+soft); R2 = R1 + rw1
# rank-weighted objective. Capacity NOT swept (466k -> 3.46M moved t3@10 by 0.0002). Both natural
# and capped(36,864) settings. Fit train / select VAL / report TEST. Significance = the fixed PAIRED
# bootstrap (same config both arms, joined on user_id, B=2000); the unpaired version inflated the
# width 0.0015 -> 0.0073 and must not be used. SID widths READ from the files.
# RESUMABLE: dumps are [ ! -f ]-guarded (and re-validated even when cached); every
# (pool,config,setting) run is cached under $OUT/cache.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
MFD_CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
MFD_LINK=/scratch/yw8866/rec-tmall/_cov_mfd.ckpt; ln -sf "$MFD_CKPT" "$MFD_LINK"
CH_CKPT=/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt
CH_LINK=/scratch/yw8866/rec-tmall/_cov_chain.ckpt; ln -sf "$CH_CKPT" "$CH_LINK"

OUT=/scratch/yw8866/rec-tmall/coverage_exp; mkdir -p $OUT
POOLS=$OUT/pools; mkdir -p $POOLS
MR=/scratch/yw8866/rec-tmall/matched_rerank
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'

P1_TR=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
P1_VA=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
P1_TE=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt

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

# NOTE: limit_test_batches is deliberately NOT in the base string -- it is computed per split below,
# because limit_test_batches * batch_size_per_device is what truncated the previous run.
MFD_BASE=\"experiment=tiger_mb_mfd_flat train=false test=true ckpt_path=$MFD_LINK data_dir=$DATA \
semantic_id_path=$MD_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
mfd_behavior_marginalized=true mfd_behavior_weight=1.0 mfd_top_k_shared=200 mfd_pool_size=500 \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
+model.mfd_dump_memory=true mfd_audit_dump=true mfd_audit_pv_cap=300000\"

CH_BASE=\"experiment=tiger_mb_train_flat train=false test=true ckpt_path=$CH_LINK data_dir=$DATA \
semantic_id_path=$TG_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
+model.chain_pool_audit=true +model.chain_pool_size=500 +model.chain_pool_pv_cap=300000 \
data_loading.test_dataloader_config.dataloader.batch_size_per_device=16 +trainer.limit_test_batches=6000\"

# ---- generate one MFD pool split, then VALIDATE it immediately -------------------------------
gen_mfd () {   # \$1=tag \$2=split \$3=batch \$4=extra hydra args
  TAG=\$1; SP=\$2; BATCH=\$3; EXTRA=\$4
  case \$SP in
    test)  FOLD=''; NEED=83005;  MAXE=200000; PB='$PB_TEST' ;;
    val)   FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation\"; NEED=83005;  MAXE=200000; PB='' ;;
    train) FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training\";   NEED=250000; MAXE=250000; PB='' ;;
  esac
  # THE FIX: limit_test_batches * batch must cover NEED, else the dump silently truncates.
  LIMIT=\$(( (NEED + BATCH - 1) / BATCH + 50 ))
  OUTF=$POOLS/\${TAG}_\${SP}.pt
  if [ ! -f \$OUTF ]; then
    echo \"=== MFD \$TAG \$SP dump (batch=\$BATCH limit_test_batches=\$LIMIT -> capacity \$(( LIMIT*BATCH )) >= \$NEED) ===\"
    python /scratch/yw8866/GRID/src/train.py \$MFD_BASE \$EXTRA \
        mfd_audit_max_events=\$MAXE +trainer.limit_test_batches=\$LIMIT \
        data_loading.test_dataloader_config.dataloader.batch_size_per_device=\$BATCH \
        \$FOLD mfd_audit_path=\$OUTF
    [ -f \$OUTF ] || { echo \"** ABORT: \$OUTF was not produced -- see the traceback above **\"; exit 1; }
  else echo \"=== \$TAG \$SP exists ===\"; fi
  python $VAL --dump \$OUTF --expect_n \$NEED --expect_perbeh \"\$PB\" --expect_states 808 \
    || { echo \"** ABORT: \$OUTF failed validation -- delete it and re-run **\"; exit 1; }
}

for SP in test val train; do
  gen_mfd p2 \$SP 4 \"mfd_top_k_text=200 mfd_top_k_image=200\"
done
for SP in test val train; do
  gen_mfd p3 \$SP 8 \"mfd_top_k_text=1024 mfd_top_k_image=1024 +model.mfd_wide_catalog_pool=true\"
done

# ---- chain: sweep already done; reuse the uniform (winning) 100x100 dumps ---------------------
for SP in test val train; do
  case \$SP in
    test)  SRC=$MR/chain808_test_dump.pt ;;
    val)   SRC=$MR/chain808_val_dump.pt ;;
    train) SRC=$MR/chain808_train_dump.pt ;;
  esac
  OUTF=$POOLS/ch_\${SP}.pt
  [ -f \$OUTF ] || ln -sf \$SRC \$OUTF
done
python $VAL --dump $POOLS/ch_test.pt --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 \
  || { echo '** ABORT: chain test dump failed validation **'; exit 1; }

# ---- P1 control: validate the cached dumps too (never regenerate) -----------------------------
python $VAL --dump $P1_TE --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 \
  || { echo '** ABORT: P1 test dump failed validation **'; exit 1; }

python - > $OUT/spec.json <<PY
import json
print(json.dumps({
 'P1': {'arm':'MFD','tr':'$P1_TR','va':'$P1_VA','te':'$P1_TE','sid':'$MD_SID'},
 'P2': {'arm':'MFD','tr':'$POOLS/p2_train.pt','va':'$POOLS/p2_val.pt','te':'$POOLS/p2_test.pt','sid':'$MD_SID'},
 'P3': {'arm':'MFD','tr':'$POOLS/p3_train.pt','va':'$POOLS/p3_val.pt','te':'$POOLS/p3_test.pt','sid':'$MD_SID'},
 'CH': {'arm':'chain','tr':'$POOLS/ch_train.pt','va':'$POOLS/ch_val.pt','te':'$POOLS/ch_test.pt','sid':'$TG_SID'},
}, indent=1))
PY

echo '################ FULL GRID: 4 pools x {R1,R2} x {natural,capped} ################'
python /scratch/yw8866/rec-tmall/coverage_expansion.py \
    --spec $OUT/spec.json --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 \
    --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 32 --eval_batch 64 --boot 2000 --seed 0 --out_dir $OUT \
    --chain_sweep 180x20=0.2373 --chain_sweep 150x50=0.2443 --chain_sweep 100x100=0.2462 \
    --chain_sweep 50x150=0.2446 --chain_sweep 20x180=0.2390 --chain_sweep_winner 100x100
"
