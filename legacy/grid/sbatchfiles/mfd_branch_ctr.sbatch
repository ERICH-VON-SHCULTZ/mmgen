#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=30:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_bctr
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_branch_ctr.out

# ============================================================================================
# GOAL (rec-tmall/GOAL.md): MFD beats chain on t3@10 by >= +0.01 absolute. Lead today +0.00196.
#
# WHERE THE PREVIOUS ROUNDS LEFT IT. Re-ordering the frozen pool (round 1: four reranker mechanisms)
# and re-selecting it (tau screen) are both closed. The COMPONENT AUDIT then localised the failure:
# for the 4,963 MFD-ONLY ground truths the reranker misses, the shared code is already right (f*
# median rank 1) but the item heads are diffuse (t* rank 6, i* rank 3, logP ~ -3.9 / -3.3); the 362
# it rescues have t* and i* at rank 0 even under a weak root. The model knows the CATEGORY and cannot
# pin the ITEM inside it -- and every reranker feature is derived from those same heads, which is why
# no reranker could fix it.
#
# THE CHANGE. A train/inference mismatch that only MFD has: its t and i heads are trained by
# independent cross-entropy over all 1,024 codes, but inference only ever ranks CATALOG-VALID (t,i)
# pairs inside the GT's branch. +model.mfd_branch_contrastive adds cross-entropy over EVERY catalog
# tuple under f*, target = the GT tuple (verified against a brute-force per-item softmax). It is free
# for MFD -- both full heads come from one forward pass, so a branch's ~3,000 tuples are scored by a
# gather -- while chain would need a sequential decode per negative. Fine-tuned from Run-C, identical
# architecture and generation settings, 10k steps.
#
# THEN, with nothing else changed: re-dump P3 (wide catalog, K_f=200, full t/i, 808 states, K_pool=2000)
# train/val/test, validated at generation time; rerank WITH NEW-ITEM PROTECTION (nip_rerank.py -- the
# current reranker evicts new items however sharp their heads are, so sharper heads can only convert
# under protection) on 3 seeds against NIP's CACHED chain results (never retrained); decompose the lead by pool membership; and check the mechanism on
# a FIXED event set: did t*/i* ranks fall for the exact ground truths that were being missed?
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
RUNC=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
RUNC_LINK=/scratch/yw8866/rec-tmall/_bctr_warm.ckpt; ln -sf "$RUNC" "$RUNC_LINK"   # Hydra cannot parse '=' in override values
LAMBDA=1.0
STEPS=10000
FT=/scratch/yw8866/logs/train/mfd_bctr_l${LAMBDA}
OUT=/scratch/yw8866/rec-tmall/bctr_exp; mkdir -p $OUT/pools $OUT/cache
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OLD_TEST=/scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt
SEEDV=/scratch/yw8866/rec-tmall/seedvar/cache
NIPC=/scratch/yw8866/rec-tmall/nip_exp/cache          # chain results from the NIP run (chain never retrained)
WNEW=1                                                  # NIP: w_new=3 gave nothing and VAL always chose P0
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'

for f in $RUNC $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt $OLD_TEST \
         $SEEDV/MFD__seed0.pt $SEEDV/CH__seed0.pt $SEEDV/CH__seed1.pt $SEEDV/CH__seed2.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done
# chain is REUSED, never retrained: seed the new cache with NIP's chain results (same policy family)
for f in $NIPC/CH__wnew*__seed*.pt; do [ -f "$f" ] && cp -n "$f" $OUT/cache/; done

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

GEN=\"experiment=tiger_mb_mfd_flat data_dir=$DATA semantic_id_path=$MD_SID num_hierarchies=3 \
codebook_init_path=null codebook_sizes=[2048,1024,1024] mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true\"

# ======================= STAGE 1: fine-tune Run-C with the within-branch contrastive loss ==========
CKPT=\$(ls -t $FT/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -z \"\$CKPT\" ]; then
  echo '################ STAGE 1: fine-tune (lambda=$LAMBDA, $STEPS steps, warm-start from Run-C) ################'
  python /scratch/yw8866/GRID/src/train.py \$GEN train=true test=false \
      mfd_top_k_shared=50 mfd_top_k_text=10 mfd_top_k_image=10 mfd_pool_size=50 \
      +model.mfd_warmstart_path=$RUNC_LINK +model.mfd_branch_contrastive=$LAMBDA \
      trainer.max_steps=$STEPS trainer.val_check_interval=5000 hydra.run.dir=$FT
  CKPT=\$(ls -t $FT/checkpoints/*.ckpt 2>/dev/null | head -1)
fi
[ -n \"\$CKPT\" ] || { echo '** ABORT: fine-tune produced no checkpoint **'; exit 1; }
echo \"fine-tuned checkpoint: \$CKPT\"
LINK=$OUT/_bctr.ckpt; ln -sf \"\$CKPT\" \$LINK

# ======================= STAGE 2: re-dump P3 at K_pool=2000 with memory, validated ==================
gen () {
  SP=\$1
  case \$SP in
    test)  FOLD=''; NEED=83005;  MAXE=200000; PB='$PB_TEST' ;;
    val)   FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation\"; NEED=83005; MAXE=200000; PB='' ;;
    train) FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training\";   NEED=100000; MAXE=100000; PB='' ;;
  esac
  LIMIT=\$(( (NEED + 8 - 1) / 8 + 50 ))
  OUTF=$OUT/pools/bctr_\${SP}.pt
  if [ ! -f \$OUTF ]; then
    python /scratch/yw8866/GRID/src/train.py \$GEN train=false test=true ckpt_path=\$LINK \
        mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 +model.mfd_wide_catalog_pool=true \
        mfd_pool_size=2000 +model.mfd_dump_memory=true mfd_audit_dump=true mfd_audit_pv_cap=300000 \
        mfd_audit_max_events=\$MAXE +trainer.limit_test_batches=\$LIMIT \
        data_loading.test_dataloader_config.dataloader.batch_size_per_device=8 \$FOLD mfd_audit_path=\$OUTF
  fi
  [ -f \$OUTF ] || { echo \"** ABORT: \$OUTF not produced **\"; exit 1; }
  python $VAL --dump \$OUTF --expect_n \$NEED --expect_perbeh \"\$PB\" --expect_states 808 \
    || { echo \"** ABORT: \$OUTF failed validation **\"; exit 1; }
}
echo '################ STAGE 2: re-dump P3 (wide, K_pool=2000) test / val / train ################'
gen test; gen val; gen train

# ======================= STAGE 3: mechanism check BEFORE spending reranker time =====================
echo '################ STAGE 3: did the item heads sharpen on the ground truths that were being missed? ################'
python /scratch/yw8866/rec-tmall/head_sharpening.py --old_dump $OLD_TEST --new_dump $OUT/pools/bctr_test.pt \
    --old_mfd $SEEDV/MFD__seed0.pt --ch $SEEDV/CH__seed0.pt --out $OUT/head_sharpening.txt

# ======================= STAGE 4: rerank R2_rw1 capped, 3 seeds; chain cached =======================
cat > $OUT/spec.json <<JSON
{
 \"MFD\": {\"tr\": \"$OUT/pools/bctr_train.pt\", \"va\": \"$OUT/pools/bctr_val.pt\", \"te\": \"$OUT/pools/bctr_test.pt\", \"sid\": \"$MD_SID\", \"kpool\": null},
 \"CH\":  {\"tr\": \"$CHD/chain808_train_dump.pt\", \"va\": \"$CHD/chain808_val_dump.pt\", \"te\": \"$CHD/chain808_test_dump.pt\", \"sid\": \"$TG_SID\", \"kpool\": null}
}
JSON
echo '################ STAGE 4: rerank with new-item protection (NIP policies, chosen on VAL), 3 seeds ################'
python /scratch/yw8866/rec-tmall/nip_rerank.py --spec $OUT/spec.json --cache $OUT/cache --seeds 0,1,2 --w_new $WNEW \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --bar 0.01 --out_dir $OUT

# ======================= STAGE 5: decomposition of the new lead + component audit ===================
echo '################ STAGE 5: component audit on the fine-tuned generator ################'
python /scratch/yw8866/rec-tmall/component_audit.py --dump $OUT/pools/bctr_test.pt \
    --mfd $OUT/cache/MFD__wnew1__seed0.pt --ch $OUT/cache/CH__wnew1__seed0.pt --out $OUT/component_audit.txt
"
