#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=30:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=fh_spec
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_force_spec.out
# ============================================================================================
# GOAL: MFD beats chain on t3@10 by >= +0.01. recall = coverage x P(rank<10|cov).
# Ceiling measurement (cap_eval2.txt): even with 10 distinct items and the correct behavior, the lead is
# +0.0028 -- because predictable events already convert at P~0.99 there, the lead is capped by HOW MANY MORE
# predictable events MFD's pool holds than chain's (18,796 vs 18,233 -> <= +0.0067). +0.01 needs ~840 more.
# MFD misses 478 pv repeats from its pool, chain 1,020.
# THIS JOB: history forcing, decoder states unchanged (808): the user's history shared codes are kept among
# the K_f=200 roots (1 root each; the t/i heads then score every tuple under them for free) and the history
# tuples enter K_pool=2000 with their TRUE generator scores. Chain would need beam slots at two sequential
# levels per forced item. Then: rerank (R2_rw1 capped, 3 seeds) vs chain's cached results; report coverage
# of repeats and the lead decomposed by pool membership x {repeat, new}.
# ============================================================================================
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
RUNC=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_forcehist.ckpt; ln -sf "$RUNC" "$LINK"
OUT=/scratch/yw8866/rec-tmall/force_exp; mkdir -p $OUT/pools $OUT/cache $OUT/cache2 $OUT/spec2
CHD=/scratch/yw8866/rec-tmall/matched_rerank
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'

cat > $OUT/spec.json <<JSON
{
 "MFD": {"tr": "$OUT/pools/fh_train.pt", "va": "$OUT/pools/fh_val.pt", "te": "$OUT/pools/fh_test.pt", "sid": "$MD_SID", "kpool": null},
 "CH":  {"tr": "$CHD/chain808_train_dump.pt", "va": "$CHD/chain808_val_dump.pt", "te": "$CHD/chain808_test_dump.pt", "sid": "$TG_SID", "kpool": null}
}
JSON
module purge
singularity exec --nv --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310; set -f
export HF_HOME=/scratch/yw8866/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT='MMMBGRec' WANDB_INSECURE_DISABLE_SSL=true WANDB_API_KEY='REDACTED_ROTATE_ME'
GEN=\"experiment=tiger_mb_mfd_flat data_dir=$DATA semantic_id_path=$MD_SID num_hierarchies=3 \
codebook_init_path=null codebook_sizes=[2048,1024,1024] mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true\"
gen () {
  SP=\$1
  case \$SP in
    test)  FOLD=''; NEED=83005;  MAXE=200000; PB='$PB_TEST' ;;
    val)   FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/evaluation\"; NEED=83005; MAXE=200000; PB='' ;;
    train) FOLD=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/training\";   NEED=100000; MAXE=100000; PB='' ;;
  esac
  LIMIT=\$(( (NEED + 8 - 1) / 8 + 50 ))
  OUTF=$OUT/pools/fh_\${SP}.pt
  if [ ! -f \$OUTF ]; then
    python /scratch/yw8866/GRID/src/train.py \$GEN train=false test=true ckpt_path=$LINK \
        mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 +model.mfd_wide_catalog_pool=true \
        +model.mfd_force_history=true mfd_pool_size=2000 +model.mfd_dump_memory=true mfd_audit_dump=true mfd_audit_pv_cap=300000 \
        mfd_audit_max_events=\$MAXE +trainer.limit_test_batches=\$LIMIT \
        data_loading.test_dataloader_config.dataloader.batch_size_per_device=8 \$FOLD mfd_audit_path=\$OUTF
  fi
  [ -f \$OUTF ] || { echo \"** ABORT: \$OUTF not produced **\"; exit 1; }
  python $VAL --dump \$OUTF --expect_n \$NEED --expect_perbeh \"\$PB\" --expect_states 808 || { echo \"** ABORT: \$OUTF failed validation **\"; exit 1; }
}
echo '######## dumps (cached) ########'
gen test; gen val; gen train
echo '######## rerank with BOTH specialists (new-item + repeat), 3 seeds, both arms ########'
python /scratch/yw8866/rec-tmall/nip_rerank.py --spec $OUT/spec.json --cache $OUT/cache2 --seeds 0,1,2 --w_new 1 \
    --specialist --repeat_spec \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --bar 0.01 --out_dir $OUT/spec2
"
