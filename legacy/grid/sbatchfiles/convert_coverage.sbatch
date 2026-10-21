#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=convert_cov
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/convert_coverage.out

# ============================================================================================
# ONE QUESTION: under the strong reranker, can MFD's coverage advantage be converted into recall?
# Both generative models FROZEN. Steps 1-3 in order; NOTHING is taken from the pool cache
# (every P3 dump is regenerated; the `exists` branch is bypassed with FORCE=1); the job HALTS
# at step 1 or step 2 if the widened pool does not actually widen coverage.
#
# STEP 1a (offline, on the EXISTING P1/P2/P3 test dumps): for every GT with rank(t*)<K_t and
#   rank(i*)<K_i yet absent from the pool, is its own hand-set score below the 500th pool score
#   (truncated by K_pool) or above it (a real composition/filter bug)?
# STEP 1b (IN-MODEL, instrumented): a short audit pass with +model.mfd_compose_debug=3 on the wide
#   path (K_t=K_i=1024) prints, FROM INSIDE _audit_forward: the K_t/K_i/K_f/K_pool locals actually in
#   effect, the head tensor widths the t/i log-probs are gathered from, candidates composed per event
#   and per f-branch (and whether that tracks K_t x K_i), survivors after the catalog filter and after
#   the top-K_pool cut, the list of every truncation op on the path, and per event the GT's f*/t*/i*
#   ranks, its rank among ALL composed candidates (new column gt_comp_rank), its score vs the cutoff
#   and a verdict (in pool / pruned by K_f / pruned by K_t,K_i / TRUNCATED BY K_pool / BUG).
#   diag_sample then classifies all missing GTs of the sample, reads coverage@K off gt_comp_rank,
#   states the root cause, and fixes K_pool (K_fix = smallest ladder K with sample coverage >= 0.52,
#   capped at MAX_KPOOL) -- or HALTS: exit 3 = real bug found (fix code), exit 2 = no K <= MAX_KPOOL
#   reaches the target (the ladder shows what it would take).
# STEP 2: FORCE-regenerate the P3 TEST dump (K_t=K_i=1024, K_pool=K_fix, with memory) and assert
#   at generation time: 83,005 events, pv 77,833 / fav 2,042 / cart 1,554 / buy 1,576, 808 decoder
#   states/item, AND coverage@pool >= 0.50 (the validator now FAILS on coverage, it no longer just
#   prints it). If it fails: stop and report; no train/val, no training.
# STEP 3 (only if step 2 passes): FORCE-regenerate P3 val/train at K_fix, then the minimal grid:
#   P1 x {R1,R2_rw1} and CH x {R1,R2_rw1} REUSED from the reranker cache (capped: chain R1 0.2142,
#   R2_rw1 0.2140, cov 0.2471); only P3W x {R1,R2_rw1} capped is trained (fresh cache key P3W -- the
#   old K_pool=500 P3 results are NOT reused). Deciding metric = NET t3@10 (P3W - P1) decomposed into
#   newly-covered gain / previously-covered regained / previously-covered LOST, with paired CIs;
#   newly-covered recovery @10/20/50 + CI, rank distribution, entropy-z buckets, per behavior; the
#   four cells vs chain; P3W-vs-chain paired significance. Verdict on the NET, absolute threshold.
#
# PROTOCOL NOTE (deliberate deviation forced by the step-1 finding): P1 and chain stay at K_pool=500;
# P3W uses K_fix. Decoder compute is unchanged (808 states, asserted on every dump). The extra cost
# is the reranker's candidate count (nb*K_fix vs 2,000/event), which IS the "wider pool" under test.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MFD_CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
MFD_LINK=/scratch/yw8866/rec-tmall/_cov_mfd.ckpt; ln -sf "$MFD_CKPT" "$MFD_LINK"

OUT=/scratch/yw8866/rec-tmall/coverage_exp
POOLS=$OUT/pools; mkdir -p $POOLS
CACHE=$OUT/cache
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
CC=/scratch/yw8866/rec-tmall/convert_coverage.py
PB_TEST='77833,2042,1554,1576'
P1_TR=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
P1_VA=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
P1_TE=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
TARGET_COV=0.50      # the gate
MAX_KPOOL=5000       # largest K_pool the reranker stage can hold in 250GB RAM (dumps scale ~linearly in K_pool)
rm -f $OUT/convert_report.txt
# stale results of the K_pool=500 "P3" (superseded): never read again by this job
rm -f $CACHE/P3W__*.pt

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

# common MFD args: wide path, K_t=K_i=1024; K_pool / memory / limit / debug are set per dump
MFD_CORE=\"experiment=tiger_mb_mfd_flat train=false test=true ckpt_path=$MFD_LINK data_dir=$DATA \
semantic_id_path=$MD_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
mfd_behavior_marginalized=true mfd_behavior_weight=1.0 mfd_top_k_shared=200 \
mfd_top_k_text=1024 mfd_top_k_image=1024 +model.mfd_wide_catalog_pool=true \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
mfd_audit_dump=true mfd_audit_pv_cap=300000\"

# gen_wide TAG SPLIT KPOOL BATCH MEM(true|false) MINCOV(''|float) -> FORCE-generate + validate one split
gen_wide () {
  TAG=\$1; SP=\$2; KPOOL=\$3; BATCH=\$4; MEM=\$5; MINCOV=\$6
  case \$SP in
    test)  FOLD=''; NEED=83005;  MAXE=200000; PB='$PB_TEST' ;;
    val)   FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation\"; NEED=83005;  MAXE=200000; PB='' ;;
    train) FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training\";   NEED=100000; MAXE=100000; PB='' ;;
  esac
  LIMIT=\$(( (NEED + BATCH - 1) / BATCH + 50 ))      # limit_test_batches*batch must cover NEED
  OUTF=$POOLS/\${TAG}_\${SP}.pt
  rm -f \$OUTF                                        # FORCE: never take an exists branch
  echo \"=== \$TAG \$SP: K_pool=\$KPOOL mem=\$MEM batch=\$BATCH limit=\$LIMIT (cap \$(( LIMIT*BATCH )) >= \$NEED) [regenerating] ===\"
  python /scratch/yw8866/GRID/src/train.py \$MFD_CORE mfd_pool_size=\$KPOOL +model.mfd_dump_memory=\$MEM \
      mfd_audit_max_events=\$MAXE +trainer.limit_test_batches=\$LIMIT \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=\$BATCH \
      \$FOLD mfd_audit_path=\$OUTF
  [ -f \$OUTF ] || { echo \"** ABORT: \$OUTF not produced **\"; exit 1; }
  MC=''; [ -n \"\$MINCOV\" ] && MC=\"--min_cov \$MINCOV\"
  python $VAL --dump \$OUTF --expect_n \$NEED --expect_perbeh \"\$PB\" --expect_states 808 \$MC \
    || { echo \"** ABORT: \$OUTF failed validation **\"; exit 1; }
}

# ============================ STEP 1a: offline diagnosis on the existing dumps ================
echo '################ STEP 1a: offline -- where did the missing GTs of the existing P1/P2/P3 dumps go? ################'
python $CC --stage diagnose --p1_test $P1_TE --p2_test $POOLS/p2_test.pt --p3_test $POOLS/p3_test.pt --out_dir $OUT \
  || { echo '** STEP 1a failed **'; exit 1; }

# ============================ STEP 1b: IN-MODEL instrumented sample ==========================
echo '################ STEP 1b: instrumented composition stage (prints come from INSIDE _audit_forward) ################'
DIAG=$POOLS/diag_wide_sample.pt; rm -f \$DIAG
python /scratch/yw8866/GRID/src/train.py \$MFD_CORE mfd_pool_size=500 +model.mfd_dump_memory=false \
    +model.mfd_compose_debug=3 mfd_audit_max_events=8000 +trainer.limit_test_batches=1000 \
    data_loading.test_dataloader_config.dataloader.batch_size_per_device=8 mfd_audit_path=\$DIAG
[ -f \$DIAG ] || { echo '** ABORT: diagnostic sample not produced **'; exit 1; }
python $CC --stage diag_sample --diag_dump \$DIAG --target_cov $TARGET_COV --margin 0.02 --max_kpool $MAX_KPOOL \
    --kfix_json $POOLS/kfix.json --out_dir $OUT
RC=\$?
if [ \$RC -ne 0 ]; then
  echo \"################ HALT at STEP 1 (rc=\$RC: 3 = real composition bug, 2 = no K_pool <= $MAX_KPOOL reaches $TARGET_COV). Not regenerating. ################\"
  exit \$RC
fi
KFIX=\$(python -c \"import json;print(json.load(open('$POOLS/kfix.json'))['Kpool'])\")
echo \"STEP 1 DONE: root cause = K_pool truncation; regenerating P3 at K_t=K_i=1024, K_pool=\$KFIX\"

# ============================ STEP 2: FORCE-regenerate P3 TEST and gate ======================
echo \"################ STEP 2: P3W test dump at K_pool=\$KFIX (with memory), gated on coverage >= $TARGET_COV ################\"
gen_wide p3w test \$KFIX 8 true $TARGET_COV
python $CC --stage curve --wide_dump $POOLS/p3w_test.pt --ks 500,1000,2000,3000,4000,5000 \
    --target_cov $TARGET_COV --kstar_json $POOLS/kstar.json --out_dir $OUT
RC=\$?
if [ \$RC -ne 0 ]; then
  echo '################ HALT at STEP 2: the regenerated P3 test dump did not reach the coverage gate. No train/val, no training. ################'
  exit \$RC
fi
echo 'STEP 2 PASSED'

# ============================ STEP 3: P3 val/train at K_fix, then the minimal grid ===========
echo \"################ STEP 3: P3W val/train at K_pool=\$KFIX, then P1 x {R1,R2_rw1} (cached) vs P3W x {R1,R2_rw1} (train), chain reused ################\"
gen_wide p3w val   \$KFIX 8 true ''
gen_wide p3w train \$KFIX 8 true ''

python - > $OUT/spec_pw.json <<PY
import json
print(json.dumps({
 'P1':  {'arm':'MFD','tr':'$P1_TR','va':'$P1_VA','te':'$P1_TE','sid':'$MD_SID'},
 'P3W': {'arm':'MFD','tr':'$POOLS/p3w_train.pt','va':'$POOLS/p3w_val.pt','te':'$POOLS/p3w_test.pt','sid':'$MD_SID'},
}, indent=1))
PY

python $CC --stage grid --spec $OUT/spec_pw.json --cache $CACHE --cap 36864 --wb 1.0 \
    --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --out_dir $OUT
"
