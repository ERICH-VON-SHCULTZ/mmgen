#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=47:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=rerank_sweep
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/rerank_sweep.out

# ============================================================================================
# RERANKER SWEEP on BOTH arms: does MFD's advantage over chain GROW with a stronger ranker, or does
# the gain ratio collapse toward 1:1 as signal is added? Offline training only; both generative
# models stay frozen. One job, one consolidated report.
#
# Reference at matched 808-decoder-state compute (baseline ranker, 466,690 params):
#   MFD   t3@10 0.2042 -> 0.2151 (+0.0110)      chain 0.2120 -> 0.2140 (+0.0020)     ratio 5.3x
#
# Configurations (all on both arms, in BOTH the natural and the capped=36,864 training setting):
#   baseline        2 layers, d=128, views + hard + soft
#   A1/A2/A3        capacity: 4L/128, 2L/256, 4L/256   (d_model != d_mem -> encoder memory projected)
#   B1..B4          cumulative features: decay, +popularity, +session, +user
#   C1              two-stage: cheap scalar-only stage-1 shortlists top-50, cross-attn refines it
#   BESTOF          winning capacity + winning feature set (+ two-stage if it helped), selected on
#                   EACH ARM'S OWN VAL split, never on TEST
#
# !! B1 CAVEAT: the TFRecords store only user_id / sequence_data / behavior_data. Sequences are
#    SORTED by timestamp but timestamps are NOT stored, so elapsed TIME is not recoverable from any
#    artefact. B1 is implemented as an exponential POSITION-decay proxy (two rates) and reported as
#    such -- it is a different signal from the existing linear recency, but it is not elapsed time.
#
# Popularity is computed from the TRAINING split ONLY (never val/test); its "recent window" is the
# last 20% of training rows (a row-order proxy, again because timestamps are unavailable). Its
# marginal contribution (B2 minus B1) is reported separately.
#
# Asserts per arm: TEST 83,005 (pv 77,833 / fav 2,042 / cart 1,554 / buy 1,576); coverage@pool
# unchanged by reranking (MFD 0.2615, chain 0.2471); SID widths READ from each semantic_ids.pt with
# per-level in-range asserts (chain 1024/1024/1024, MFD 2048/1024/1024) -- never hard-coded.
#
# RESUMABLE: every (arm, config, setting) result is cached under $OUT/cache, so if the job hits the
# wall clock just resubmit and it continues. ~40 training runs total.
# ============================================================================================

MR=/scratch/yw8866/rec-tmall/matched_rerank
MFD_TR=/scratch/yw8866/rec-tmall/mfd_rerank_eng_trainsplit_dump.pt
MFD_VA=/scratch/yw8866/rec-tmall/mfd_rerank_eng_val_dump.pt
MFD_TE=/scratch/yw8866/rec-tmall/mfd_rerank_eng_test_dump.pt
CH_TR=$MR/chain808_train_dump.pt
CH_VA=$MR/chain808_val_dump.pt
CH_TE=$MR/chain808_test_dump.pt
MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
OUT=/scratch/yw8866/rec-tmall/sweep

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

python /scratch/yw8866/rec-tmall/sweep_rerank.py \
    --mfd_tr $MFD_TR --mfd_va $MFD_VA --mfd_te $MFD_TE \
    --ch_tr $CH_TR --ch_va $CH_VA --ch_te $CH_TE \
    --md_sid $MD_SID --tg_sid $TG_SID \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 \
    --epochs 15 --patience 3 --min_delta 0.0 --batch 32 --eval_batch 64 \
    --boot 2000 --seed 0 --out_dir $OUT
"
