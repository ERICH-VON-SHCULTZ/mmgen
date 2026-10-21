#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=matched_compute
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/matched_compute.out

# ============================================================================================
# MATCHED-COMPUTE pool comparison: chain sequential vs MFD factorized decoding at EQUAL decoder-
# state budget (not equal scored-pool size). Pure analysis + dumping; NO model is retrained.
# Both arms on the SAME test events; K_pool=500 on every arm (scoring identical; recall@10 is
# K_pool-invariant so this does not bias the comparison).
#
# decoder-state evaluations / item:
#   chain : 1 + 3 + nb*(1 + 2*Bw)            -> grows with beam Bw
#   MFD   : nb*(2 + Kf)                       -> grows ONLY with Kf; K_t/K_i are free head-reads
#                                               off the shared (b,f) root (that is the factorization)
# nb=4, so the four matched points are:
#   chain@808  : Bw=100    -> 4 + 4*(1+200)  = 808
#   chain@4008 : Bw=500    -> 4 + 4*(1+1000) = 4008
#   MFD@808    : Kf=200    -> 4*(2+200)      = 808     (reference config: K_t=200,K_i=50)
#   MFD@4008   : Kf=1000   -> 4*(2+1000)     = 4008    (K_t,K_i widened to full for FREE)
# plus one auxiliary point to split the MFD 808->4008 move into FREE (K_t,K_i) vs PAID (Kf):
#   MFD@808+free : Kf=200, K_t=K_i=full  -> still 808 decoder states (wide-catalog scoring)
#
# Allocation rule (reproducible): the entire decoder-state budget goes to Kf (the only beam that
# costs decoder states, and also the largest branch loss f_pruned=0.244); K_t/K_i are then set to
# codebook-full at zero decoder-state cost. This is why "widen by branch loss" collapses to
# "widen Kf" for MFD.
#
# All FIVE dumps are re-generated fresh with the new GPU-synced wall-clock timer so sec/1000-events
# is measured on the SAME node for every point (timing is only comparable within one run). Each dump
# is guarded by `[ ! -f ]`; if the job hits the wall-clock limit, just resubmit and it resumes with
# the heavy points (chain@4008, MFD@4008) last.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt

MFD_CKPT=/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt
MFD_LINK=/scratch/yw8866/rec-tmall/_mc_mfd.ckpt; ln -sf "$MFD_CKPT" "$MFD_LINK"
CH_CKPT=/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt
CH_LINK=/scratch/yw8866/rec-tmall/_mc_chain.ckpt; ln -sf "$CH_CKPT" "$CH_LINK"

OUT=/scratch/yw8866/rec-tmall/matched_compute
mkdir -p $OUT
CH808=$OUT/mc_chain_808.pt
CH4008=$OUT/mc_chain_4008.pt
MFD808=$OUT/mc_mfd_808.pt
MFD808F=$OUT/mc_mfd_808_free.pt
MFD4008=$OUT/mc_mfd_4008.pt

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f   # noglob: the unquoted \$MFD_COMMON/\$CH_COMMON expansion must keep codebook_sizes=[..] literal
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

MFD_COMMON=\"experiment=tiger_mb_mfd_flat train=false test=true ckpt_path=$MFD_LINK data_dir=$DATA \
semantic_id_path=$MD_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
mfd_behavior_marginalized=true mfd_behavior_weight=1.0 mfd_pool_size=500 \
+model.use_pmq=true +model.use_query_f=true +model.decoder_slot_type_emb=true \
mfd_audit_dump=true mfd_audit_max_events=200000 mfd_audit_pv_cap=300000 +trainer.limit_test_batches=6000\"

CH_COMMON=\"experiment=tiger_mb_train_flat train=false test=true ckpt_path=$CH_LINK data_dir=$DATA \
semantic_id_path=$TG_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024] \
+model.chain_pool_audit=true +model.chain_pool_size=500 +model.chain_pool_max_events=200000 \
+model.chain_pool_pv_cap=300000 data_loading.test_dataloader_config.dataloader.batch_size_per_device=16 \
+trainer.limit_test_batches=6000\"

# ---- LIGHT points first (808) ----

# MFD@808 : reference config, standard compose (K_t=200,K_i=50). fwd = 4*(2+200)=808.
if [ ! -f $MFD808 ]; then
  echo '=== MFD@808  (Kf=200,Kt=200,Ki=50) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_COMMON \
      mfd_top_k_shared=200 mfd_top_k_text=200 mfd_top_k_image=50 mfd_audit_path=$MFD808
else echo '=== MFD@808 exists, skip ==='; fi

# MFD@808+free : Kf=200 but K_t=K_i=full via wide-catalog scoring -> still 808 decoder states.
if [ ! -f $MFD808F ]; then
  echo '=== MFD@808+free  (Kf=200,Kt=Ki=1024, wide-catalog) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_COMMON \
      mfd_top_k_shared=200 mfd_top_k_text=1024 mfd_top_k_image=1024 \
      +model.mfd_wide_catalog_pool=true mfd_audit_path=$MFD808F
else echo '=== MFD@808+free exists, skip ==='; fi

# chain@808 : Bw=100. fwd = 4 + 4*(1+200) = 808.
if [ ! -f $CH808 ]; then
  echo '=== chain@808  (beam=100) ==='
  python /scratch/yw8866/GRID/src/train.py \$CH_COMMON \
      +model.chain_pool_beam=100 +model.chain_pool_path=$CH808
else echo '=== chain@808 exists, skip ==='; fi

# ---- HEAVY points last (4008) ----

# MFD@4008 : Kf=1000, K_t=K_i=full. fwd = 4*(2+1000) = 4008. Smaller batch (5x the modality tensors).
if [ ! -f $MFD4008 ]; then
  echo '=== MFD@4008  (Kf=1000,Kt=Ki=1024, wide-catalog) ==='
  python /scratch/yw8866/GRID/src/train.py \$MFD_COMMON \
      mfd_top_k_shared=1000 mfd_top_k_text=1024 mfd_top_k_image=1024 \
      +model.mfd_wide_catalog_pool=true mfd_audit_path=$MFD4008 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=8
else echo '=== MFD@4008 exists, skip ==='; fi

# chain@4008 : Bw=500. fwd = 4 + 4*(1+1000) = 4008.
if [ ! -f $CH4008 ]; then
  echo '=== chain@4008  (beam=500) ==='
  python /scratch/yw8866/GRID/src/train.py \$CH_COMMON \
      +model.chain_pool_beam=500 +model.chain_pool_path=$CH4008
else echo '=== chain@4008 exists, skip ==='; fi

# ---- analysis (only when all five dumps exist) ----
if [ -f $CH808 ] && [ -f $CH4008 ] && [ -f $MFD808 ] && [ -f $MFD808F ] && [ -f $MFD4008 ]; then
  echo '=== ANALYSIS: matched-compute comparison ==='
  python /scratch/yw8866/rec-tmall/analyze_matched_compute.py \
      --point 'chain@808'    chain $CH808 \
      --point 'chain@4008'   chain $CH4008 \
      --point 'MFD@808'      MFD   $MFD808 \
      --point 'MFD@808+free' MFD   $MFD808F \
      --point 'MFD@4008'     MFD   $MFD4008 \
      --order 'chain@808,chain@4008,MFD@808,MFD@4008' --wb 1.0
else
  echo '=== one or more dumps missing; resubmit to finish dumping, analysis skipped ==='
fi
"