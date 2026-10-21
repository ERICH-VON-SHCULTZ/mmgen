#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=chain_base_row
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/chain_baseline_row.out

# ============================================================================================
# CHAIN RQ-VAE baseline row for the results table: multi-behavior TIGER (standard sequential
# decoder, one token per SID level) on the tiger_1024 chain RQ-VAE semantic IDs. NO training --
# reuses the clean checkpoint 2026-07-10/09-06-58 (same recipe as the other baselines); eval only.
#
# Two measurement modes, reported SIDE BY SIDE (they answer different comparability questions;
# the two must agree in spirit and any gap is surfaced, per the caliber requirement):
#
#   (A) FREE-BEAM eval  == AUTHORITATIVE, caliber-aligned with the MDRQVAE+MB / MBGen rows: the
#       SAME evaluator (MultiBehaviorSIDRetrievalEvaluator) and the SAME free-beam generation
#       family. t3 = union/behavior-marginalized (w_b=1); t2 = routed to GT behavior. Decoder
#       states/item = 1 + H*top_k_for_generation (top_k=10 -> 31; this is the budget the baselines
#       ran at). A second pass at top_k=50 adds recall@20 in the free-beam family.
#
#   (B) DUMP fast-path == CATALOG-CONSTRAINED sequential-beam pools, comparable to the MFD POOL
#       (matched_compute): reuses chain_pool_cmp_dump.pt (beam 500 / 4008 states) and a beam-100
#       (808-state) dump, and computes the full metric set offline incl per-behavior recall and
#       recall@20. Reproduces the pool-comparison t3@10=0.2127 as a cross-check.
#
# SPLIT UNIT TEST: the dump analyzer HARD-asserts total==83,005 and pv/fav/cart/buy ==
# 77,833/2,042/1,554/1,576 (guards the 45k-pv subsample bug); if a dump mismatches its numbers are
# skipped rather than emitted.
# ============================================================================================

DATA=/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
CH_CKPT=/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt
CH_LINK=/scratch/yw8866/rec-tmall/_cbr_chain.ckpt; ln -sf "$CH_CKPT" "$CH_LINK"

OUT=/scratch/yw8866/rec-tmall/matched_compute; mkdir -p $OUT
DUMP_4008=/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt   # existing beam-500 dump (4008 states)
DUMP_808=$OUT/mc_chain_808.pt                                # matched_compute beam-100 dump if present
CBR_808=$OUT/cbr_chain_808.pt                                # else produce our own beam-100 dump

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f   # noglob: keep codebook_sizes=[..] / top_k_list=[..] literal
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

echo '=========================================================================================='
echo 'RUN HEADER: chain RQ-VAE (tiger_1024) codebook widths (read from the file, not assumed)'
python /scratch/yw8866/rec-tmall/print_sid_widths.py $TG_SID
echo 'expected test split: 83,005 events (pv 77,833 / fav 2,042 / cart 1,554 / buy 1,576)'
echo '=========================================================================================='

CH_EVAL=\"experiment=tiger_mb_train_flat train=false test=true ckpt_path=$CH_LINK data_dir=$DATA \
semantic_id_path=$TG_SID num_hierarchies=3 codebook_init_path=null codebook_sizes=[2048,1024,1024]\"

# ---- (A) AUTHORITATIVE free-beam eval: top_k=10, top_k_list=[5,10] (caliber = the other rows) ----
echo '=== (A1) FREE-BEAM eval  top_k_for_generation=10  top_k_list=[5,10]  (decoder-states/item = 1+3*10 = 31) ==='
python /scratch/yw8866/GRID/src/train.py \$CH_EVAL \
    top_k_for_generation=10 eval.evaluator.top_k_list=[5,10]

# ---- (A2) free-beam eval widened to top_k=50 for recall@20 (free-beam family) ----
echo '=== (A2) FREE-BEAM eval  top_k_for_generation=50  top_k_list=[5,10,20]  (decoder-states/item = 1+3*50 = 151) ==='
python /scratch/yw8866/GRID/src/train.py \$CH_EVAL \
    top_k_for_generation=50 eval.evaluator.top_k_list=[5,10,20]

# ---- (B) DUMP fast-path: ensure a beam-100 (808-state) dump exists, then analyze all chain dumps ----
if [ ! -f $DUMP_808 ] && [ ! -f $CBR_808 ]; then
  echo '=== (B0) producing beam-100 (808-state) chain pool dump ==='
  python /scratch/yw8866/GRID/src/train.py experiment=tiger_mb_train_flat train=false test=true \
      ckpt_path=$CH_LINK data_dir=$DATA semantic_id_path=$TG_SID num_hierarchies=3 \
      codebook_init_path=null codebook_sizes=[2048,1024,1024] \
      +model.chain_pool_audit=true +model.chain_pool_path=$CBR_808 \
      +model.chain_pool_beam=100 +model.chain_pool_size=500 \
      +model.chain_pool_max_events=200000 +model.chain_pool_pv_cap=300000 \
      data_loading.test_dataloader_config.dataloader.batch_size_per_device=16 \
      +trainer.limit_test_batches=6000
fi
D808=$DUMP_808; [ -f $D808 ] || D808=$CBR_808

echo '=== (B1) ANALYSIS: chain baseline row from catalog-constrained dumps ==='
POINTS=\"--point chain@4008_beam500 $DUMP_4008\"
[ -f \$D808 ] && POINTS=\"\$POINTS --point chain@808_beam100 \$D808\"
python /scratch/yw8866/rec-tmall/analyze_chain_baseline_row.py \$POINTS --wb 1.0 --ref_t3r10 0.2127

echo '=========================================================================================='
echo 'READ-OUT: the AUTHORITATIVE table row is (A1) free-beam top_k=10 (matches the other rows).'
echo 'The (B) dump numbers are catalog-constrained (comparable to the MFD pool). Compare t3@10:'
echo '  free-beam (A1) vs dump beam-500 (B, ~0.2127). A gap is the catalog-constraint effect --'
echo '  report it, do not average. The t2@10 free-beam number (A1) is the headline the paper needs.'
echo '=========================================================================================='
"