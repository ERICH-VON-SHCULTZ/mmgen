#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=14:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=pool_cmp
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/pool_comparison.out

# ============================================================================================
# POOL COMPARISON: chain sequential decoding vs MFD factorized decoding, matched candidate budget.
# Pure analysis + dumping; NO model is retrained. Both arms on the SAME 83,005 test events.
#
#   ARM 1  MFD + PMQ + Q_f (MD-RQ-VAE SIDs), ref config K_f=200,K_t=200,K_i=50,K_pool=500.
#   ARM 2  Chain multi-behavior TIGER on the residual RQ-VAE (tiger_1024) SIDs, sequential
#          decoding l0 -> l1|l0 -> l2|l0,l1. Two budget schemes are compared in the analysis:
#            scheme A (beam=500/level): the ACTUAL sequential-beam pool dumped here (500 scored);
#            scheme B (mirror-MFD grid): coverage@grid read OFFLINE from the per-level TF ranks
#                     (running the 2M-composition beam would need ~40k decoder states/event --
#                     infeasible; its grid coverage is the GENEROUS chain ceiling, which only
#                     strengthens the comparison).
#   ARM 3 (MBGen chain) is skipped: only a PBA (not plain-chain) model exists for MBGen SIDs.
#
# The chain checkpoint is 2026-07-10/09-06-58 (clean: no focal/PE/weighting; codebook_sizes
# [2048,1024,1024] -- actual widths printed in the analysis header, not assumed).
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt

MFD_CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
MFD_LINK=/scratch/yw8866/rec-tmall/_pool_mfd.ckpt; ln -sf "$MFD_CKPT" "$MFD_LINK"
CH_CKPT=/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt
CH_LINK=/scratch/yw8866/rec-tmall/_pool_chain.ckpt; ln -sf "$CH_CKPT" "$CH_LINK"

MFD_DUMP=/scratch/yw8866/rec-tmall/mfd_pool_cmp_dump.pt
CH_DUMP=/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt

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

# ---- ARM 1: MFD factorized pool dump (K_f=200,K_t=200,K_i=50,K_pool=500) ----
if [ ! -f $MFD_DUMP ]; then
  echo '=== ARM 1: MFD factorized pool audit dump ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_mfd_flat train=false test=true \
      ckpt_path=$MFD_LINK data_dir=$DATA semantic_id_path=$MD_SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      mfd_behavior_marginalized=true mfd_behavior_weight=1.0 \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_pool_size=500 \
      +model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
      mfd_audit_dump=true mfd_audit_path=$MFD_DUMP \
      mfd_audit_max_events=200000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=6000
else echo '=== ARM 1: MFD_DUMP exists, skipping ==='; fi

# ---- ARM 2: chain sequential-beam pool dump (beam=500) on the tiger_1024 residual SIDs ----
if [ ! -f $CH_DUMP ]; then
  echo '=== ARM 2: chain sequential-beam pool audit dump ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat train=false test=true \
      ckpt_path=$CH_LINK data_dir=$DATA semantic_id_path=$TG_SID num_hierarchies=3 \
      codebook_init_path=null 'codebook_sizes=[2048,1024,1024]' \
      +model.chain_pool_audit=true +model.chain_pool_path=$CH_DUMP \
      +model.chain_pool_beam=500 +model.chain_pool_size=500 \
      +model.chain_pool_max_events=200000 +model.chain_pool_pv_cap=300000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=16 \
      +trainer.limit_test_batches=6000
else echo '=== ARM 2: CH_DUMP exists, skipping ==='; fi

# ---- analysis ----
echo '=== ANALYSIS: pool comparison ==='
python /scratch/yw8866/rec-tmall/analyze_pool_comparison.py \
    --mfd_dump $MFD_DUMP --chain_dump $CH_DUMP \
    --chain_tag 'chain RQ-VAE (tiger_1024)' --wb 1.0
"