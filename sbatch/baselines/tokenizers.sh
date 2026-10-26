#!/bin/bash
# Tokenizers for all baselines on one dataset (runs inside the container). Usage: D=Instruments bash tokenizers.sh
# Optional ONLY="mql macrecpaper" restricts to those tokenizers (tags: tiger tiger_fnorm tiger_fraw mql macrec macrecpaper).
# Outputs: $DATA/$D/$D.index_<tag>.json + $RUN/tok/<tag>/ (ckpts, logs, stats). Skips finished steps.
set -euo pipefail
source /ext3/env.sh && conda activate py310
REPO=/scratch/yw8866/mmgen; M=$REPO/macrec; CI=$M/cross_index
DATA=/scratch/yw8866/mmgen_runs/data
TOK=/scratch/yw8866/mmgen_runs/baselines/$D/tok; mkdir -p $TOK
TXT=$DATA/$D/$D.emb-llama-td.npy; IMGE=$DATA/$D/$D.emb-ViT-L-14.npy
# numpy>=2 breaks the str()-based de-dup in the reassignment generators -> numpy-1 reprs
LEGACY="import sys, runpy, numpy as np; np.set_printoptions(legacy='1.25'); s=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(s, run_name='__main__')"
cd $CI
want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }

# ---- A/B: TIGER (3 RQ levels x 256 + appended collision index), single-modality RQ-VAE (MQL4GRec main.py) ----
for MOD in text image; do
  want tiger || continue
  EMB=$TXT; [ $MOD = image ] && EMB=$IMGE
  O=$TOK/tiger_$MOD; TAG=$([ $MOD = text ] && echo lemb || echo vitemb)_tiger
  if [ ! -f $DATA/$D/$D.index_$TAG.json ]; then
    mkdir -p $O; echo "== TIGER $MOD"; date
    python -u main_single.py --num_emb_list 256 256 256 --sk_epsilons 0.0 0.0 0.0 --device cuda:0 \
      --data_path $EMB --ckpt_dir $O --eval_step 2 --batch_size 2048 --epochs 500 > $O/train.log 2>&1
    python -u generate_indices_tiger.py --ckpt_path $O/best_collision_model.pth --content $MOD \
      --output_file $DATA/$D/$D.index_$TAG.json --stats_file $O/sid_stats.json
  fi
done

# ---- E: early-fusion TIGER. Same RQ-VAE command and SID rule as A/B; only the input vector differs
# (fuse_features.py: concat of L2-normalised (fnorm) or raw (fraw) text + image). The RQ-VAE input
# layer follows the input width (EmbDataset.dim), the only architectural difference to A/B.
for F in fnorm fraw; do
  want tiger_$F || continue
  EMB=$DATA/$D/$D.emb-fused-${F#f}.npy
  [ -f $EMB ] || python $M/data_process/fuse_features.py --dataset $D --data_root $DATA
  O=$TOK/tiger_$F; TAG=${F}_tiger
  if [ ! -f $DATA/$D/$D.index_$TAG.json ]; then
    mkdir -p $O; echo "== TIGER $F"; date
    python -u main_single.py --num_emb_list 256 256 256 --sk_epsilons 0.0 0.0 0.0 --device cuda:0 \
      --data_path $EMB --ckpt_dir $O --eval_step 2 --batch_size 2048 --epochs 500 > $O/train.log 2>&1
    python -u generate_indices_tiger.py --ckpt_path $O/best_collision_model.pth --content text \
      --output_file $DATA/$D/$D.index_$TAG.json --stats_file $O/sid_stats.json
  fi
done

# ---- C: MQL4GRec (4 x 256, Sinkhorn eps 0.003 on the last level as in MQL4GRec run.sh) ----
for MOD in text image; do
  want mql || continue
  EMB=$TXT; [ $MOD = image ] && EMB=$IMGE
  O=$TOK/mql_$MOD; TAG=$([ $MOD = text ] && echo lemb || echo vitemb)_mql
  if [ ! -f $DATA/$D/$D.index_$TAG.json ]; then
    mkdir -p $O; echo "== MQL4GRec $MOD"; date
    python -u main_single.py --num_emb_list 256 256 256 256 --sk_epsilons 0.0 0.0 0.0 0.003 --device cuda:0 \
      --data_path $EMB --ckpt_dir $O --eval_step 2 --batch_size 2048 --epochs 500 > $O/train.log 2>&1
    python -u -c "$LEGACY" generate_indices_single.py --dataset $D --device cuda:0 --content $MOD \
      --ckpt_path $O/best_collision_model.pth --output_dir $DATA/$D --output_file $D.index_$TAG.json > $O/gen.log 2>&1
    python $REPO/sbatch/tools/sid_stats.py --index $DATA/$D/$D.index_$TAG.json --gen_log $O/gen.log --out $O/sid_stats.json
  fi
done

# ---- D: MACRec. k-means pseudo-labels (K=512 per modality), then cross RQ-VAE ----
{ want macrec || want macrecpaper; } && [ ! -f $DATA/$D/$D.index_vitemb_kmeans512.json ] && python $M/data/kmeans_dataset.py --dataset $D --data_root $DATA
# variant "macrec":      repo as shipped: run_rqvae.sh -> run_cross_rqvae.sh 0 <name> <D> 0.1 0.1 0.001
#                        (begin_cross_layer 0; the weight flags are parsed but ignored by main.py)
# variant "macrecpaper": paper setting: cross layers 2-3 (begin_cross_layer 2), weights 0.1/0.1/0.001 applied
for V in macrec macrecpaper; do
  want $V || continue
  O=$TOK/$V
  if [ ! -f $DATA/$D/$D.index_vitemb_$V.json ]; then
    mkdir -p $O; echo "== $V"; date
    if [ $V = macrec ]; then ENTRY=main.py; BEGIN=0; else ENTRY=main_cross_weights.py; BEGIN=2; fi
    python -u $ENTRY --num_emb_list 256 256 256 256 --sk_epsilons 0.0 0.0 0.0 0.0 --device cuda:0 \
      --text_data_path $TXT --image_data_path $IMGE --ckpt_dir $O --eval_step 2 --batch_size 2048 \
      --begin_cross_layer $BEGIN --use_cross_rq True \
      --text_class_info $DATA/$D/$D.index_lemb_kmeans512.json --image_class_info $DATA/$D/$D.index_vitemb_kmeans512.json \
      --text_contrast_weight 0.1 --image_contrast_weight 0.1 --recon_contrast_weight 0.001 \
      --epochs 1000 > $O/train.log 2>&1
    for MOD in text image; do
      TAG=$([ $MOD = text ] && echo lemb || echo vitemb)_$V
      CK=$O/best_$([ $MOD = text ] && echo text || echo image)_collision_model.pth
      python -u -c "$LEGACY" generate_indices_distance.py --dataset $D --text_data_path $TXT --image_data_path $IMGE \
        --device cuda:0 --ckpt_path $CK --output_dir $DATA/$D --output_file $D.index_$TAG.json --content $MOD > $O/gen_$MOD.log 2>&1
      python $REPO/sbatch/tools/sid_stats.py --index $DATA/$D/$D.index_$TAG.json --gen_log $O/gen_$MOD.log --out $O/sid_stats_$MOD.json
    done
  fi
done
date; echo "== SID summary"; cat $TOK/*/sid_stats*.json | python -c "import sys; print(sys.stdin.read())" | grep -E 'index|collision_rate_before|unique|utilization' | head -80
