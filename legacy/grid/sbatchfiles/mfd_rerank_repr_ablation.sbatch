#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=16:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=mfd_repr_abl
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/mfd_rerank_repr_ablation.out

# ============================================================================================
# CANDIDATE-REPRESENTATION ablation inside the winning reranker. Everything is held identical to
# the shipped +both run (same frozen MFD+PMQ+Q_f model, same G4 candidate dumps, same reranker
# architecture, same training loop, same val-select/test-report protocol, same engagement features
# = +both). The ONLY thing that changes is how each candidate is ENCODED (the view fed to view_proj):
#   md3    : MD-RQ-VAE 3 modality views [u_f, u_f+u_t, u_f+u_i]   (current; 0 added params)
#   shared : MD shared level only  u_f                            (0 added params)
#   mbgen1 : MBGen level-1 QAE code embedding (the ONLY geometry a QAE+kmeans+dedup chain exposes)
#   tiger1 : TIGER chain RQ-VAE level-1 centroid                  (chain, level-1 only)
#   tiger3 : TIGER chain RQ-VAE additive residual sum c1+c2+c3    (chain, full reconstruction)
#   learned: a freshly LEARNED item(=unique-tuple) embedding table, trained jointly (report its size)
#   random : fixed random vector per candidate, frozen           (lower-bound control)
# Headline: md3 (0 params) vs learned (millions of params) tests the paper claim; shared/mbgen1/
# tiger1/tiger3 quantify how much comes from the modality residuals vs the shared/chain code.
#
# STAGE 0 builds a small cross-tokenizer cache (MD tuple -> representative item -> MBGen id1 /
# TIGER l1,l2,l3 + their codebooks); STAGE 1 reuses the existing engagement dumps (guarded);
# STAGE 2 runs the representation sweep (--ablation_repr; engagement fixed at +both).
# ============================================================================================

CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
LINK=/scratch/yw8866/rec-tmall/_mfd_rerank_runc.ckpt
ln -sf "$CKPT" "$LINK"
DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
MDRQVAE=/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt

TRAIN_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
VAL_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
TEST_DUMP=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
REPR_CACHE=/scratch/yw8866/rec-tmall/cand_repr_tables.pt

MBGEN_SID=/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt
MBGEN_ART=/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/mbgen_sid_artifacts.pt
TIGER_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
TIGER_CKPT=/scratch/yw8866/logs/train/runs/2026-04-29/tiger_1024/checkpoints/checkpoint_000_010000.ckpt

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

# ---- STAGE 0: cross-tokenizer representation cache (loud diagnostics; validates alignment) ----
if [ ! -f $REPR_CACHE ]; then
  echo '=== STAGE 0: building candidate-representation cache (MD/MBGen/TIGER) ==='
  python /scratch/yw8866/rec-tmall/build_cand_repr_tables.py \
      --md_sid $SID --mbgen_sid $MBGEN_SID --mbgen_artifacts $MBGEN_ART \
      --tiger_sid $TIGER_SID --tiger_ckpt $TIGER_CKPT --tiger_k 1024 --out $REPR_CACHE
else echo '=== STAGE 0: REPR_CACHE exists, skipping ==='; fi

# ---- STAGE 1: engagement dumps (guarded; already exist from the best run) ----
for split in val test trainsplit; do
  case \$split in
    val) DUMP=$VAL_DUMP; FOLDER=evaluation; MAXE=200000;;
    test) DUMP=$TEST_DUMP; FOLDER=testing; MAXE=200000;;
    trainsplit) DUMP=$TRAIN_DUMP; FOLDER=training; MAXE=250000;;
  esac
  if [ ! -f \$DUMP ]; then
    echo \"=== STAGE 1: dumping \$FOLDER split -> \$DUMP ===\"
    EXTRA=\"\"; [ \$split != test ] && EXTRA=\"data_loading.test_dataloader_config.dataloader.data_folder=$DATA/\$FOLDER\"
    python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
        ckpt_path=$LINK data_dir=$DATA semantic_id_path=$SID num_hierarchies=3 \
        codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
        mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
        mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
        +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
        +model.mfd_dump_memory=true mfd_audit_dump=true \
        mfd_audit_max_events=\$MAXE mfd_audit_pv_cap=300000 +trainer.limit_test_batches=6000 \
        \$EXTRA mfd_audit_path=\$DUMP
  else echo \"=== STAGE 1: \$DUMP exists, skipping ===\"; fi
done

# ---- STAGE 2: representation sweep (engagement fixed at +both) ----
echo '=== STAGE 2: candidate-representation ablation sweep ==='
python /scratch/yw8866/rec-tmall/train_mfd_reranker.py \
    --train_dump $TRAIN_DUMP --val_dump $VAL_DUMP --test_dump $TEST_DUMP \
    --mdrqvae_ckpt $MDRQVAE \
    --centroid_cache /scratch/yw8866/rec-tmall/mdrqvae_centroids.pt \
    --repr_cache $REPR_CACHE \
    --wb 1.0 --layers 2 --heads 4 --dropout 0.3 --wd 1e-3 \
    --lr 1e-3 --epochs 40 --patience 5 --min_delta 0.0 \
    --batch 32 --eval_batch 64 --seed 0 --ablation_repr \
    --save /scratch/yw8866/rec-tmall/mfd_reranker_repr.pt
"