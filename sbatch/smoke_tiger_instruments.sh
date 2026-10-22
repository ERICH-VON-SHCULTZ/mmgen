#!/bin/bash
# Smoke test payload (runs INSIDE the container; launched by smoke_tiger_instruments.sbatch).
# TIGER-style pipeline with MACRec's own code, unmodified, on Instruments:
#   1) LLaMA-7B text embeddings          data_process/amazon_text_emb.py
#   2) single-modality RQ-VAE (4x256)    cross_index/main.py  (cross-RQ OFF: flag omitted)
#   3) SID assignment + collision fix    cross_index/generate_indices_distance.py --content text
#   4) T5 seqrec training (short)        finetune_contrastive.py --tasks seqrec
#   5) constrained-beam eval HR/NDCG     test_ddp_save.py --test_task seqrec --filter_items
# All generated files go to $RUN (outside the repo). Step outputs are skipped if present.
set -euo pipefail
source /ext3/env.sh && conda activate py310

REPO=/scratch/yw8866/mmgen
M=$REPO/macrec
D=Instruments
RUN=${RUN:-/scratch/yw8866/mmgen_runs/smoke_tiger_$D}
DATA=$RUN/data
NAME=tiger_smoke
EMB=$DATA/$D/$D.emb-llama-td.npy
IDX=.index_lemb_$NAME.json
PLM_CACHE=/scratch/yw8866/hf_cache/macrec_plm
T5_EPOCHS=${T5_EPOCHS:-5}
RQ_EPOCHS=${RQ_EPOCHS:-200}
mkdir -p "$DATA"
[ -d "$DATA/$D" ] || cp -r "$M/data/$D" "$DATA/$D"
t() { date +%s; }
T0=$(t)

echo "== [1] text embeddings (LLaMA-7B, title+description, mean-pooled) =="
if [ ! -f "$EMB" ]; then
  python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("huggyllama/llama-7b", cache_dir="$PLM_CACHE",
                      allow_patterns=["*.json", "tokenizer.model", "*.safetensors"])
print("llama-7b at", p)
PY
  (cd $M/data_process && python amazon_text_emb.py --dataset $D --root $DATA \
      --model_cache_dir $PLM_CACHE --gpu_id 0)
fi
python -c "import numpy as np; e=np.load('$EMB'); print('emb', e.shape, e.dtype)"
T1=$(t)

echo "== [2] RQ-VAE (TIGER tokenizer: text only, 4 levels x 256, no cross-RQ) =="
RQ=$RUN/rqvae
if [ ! -f "$RQ/best_text_collision_model.pth" ]; then
  # DualEmbDataset needs an image array; the text array is passed for both; only the text branch
  # (--content text) is used. NOTE --use_cross_rq True --begin_cross_layer 4: with 4 levels the cross
  # branch (i >= 4, rqvae.py:172) never fires, so this is a plain per-level RQ. It is required because
  # generate_indices_distance.py only handles the per-level distance lists of the cross path
  # (rqvae.py:246-266); the non-cross path returns a stacked CUDA tensor (rq.py:56) and crashes l.147.
  (cd $M/cross_index && python -u main.py \
      --num_emb_list 256 256 256 256 --sk_epsilons 0.0 0.0 0.0 0.0 \
      --use_cross_rq True --begin_cross_layer 4 \
      --device cuda:0 --text_data_path $EMB --image_data_path $EMB \
      --ckpt_dir $RQ --eval_step 10 --batch_size 2048 --epochs $RQ_EPOCHS)
fi
T2=$(t)

echo "== [3] SID assignment with MACRec collision re-assignment =="
if [ ! -f "$DATA/$D/$D$IDX" ]; then
  # numpy>=2 reprs re-assigned codes as 'np.int64(5)', so the str()-based de-dup (l.208-229) misses
  # clashes and writes duplicate SIDs. legacy='1.25' restores numpy-1 reprs; MACRec code is unchanged.
  (cd $M/cross_index && python -u -c "import sys, runpy, numpy as np; np.set_printoptions(legacy='1.25'); sys.argv=['generate_indices_distance.py']+sys.argv[1:]; runpy.run_path('generate_indices_distance.py', run_name='__main__')" --dataset $D \
      --text_data_path $EMB --image_data_path $EMB --device cuda:0 \
      --ckpt_path $RQ/best_text_collision_model.pth \
      --output_dir $DATA/$D --output_file $D$IDX --content text) > $RUN/gen_indices.log 2>&1
  grep -E 'Collision Rate|collision items|Max number|check' $RUN/gen_indices.log | cut -c1-200
fi
python -c "import json,collections; d=json.load(open('$DATA/$D/$D$IDX')); c=collections.Counter(''.join(v) for v in d.values()); print('SID file: items', len(d), 'unique', len(c))"

T3=$(t)

echo "== [4] T5 seqrec training ($T5_EPOCHS epochs) =="
OUT=$RUN/t5_seqrec_${T5_EPOCHS}ep
cd $M
# data.py:47-50 always opens image_index_file too, so it points at the same text index.
torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) finetune_contrastive.py \
    --data_path $DATA/ --dataset $D --output_dir $OUT --base_model ./config/ckpt \
    --per_device_batch_size 1024 --learning_rate 1e-3 --epochs $T5_EPOCHS --weight_decay 0.01 \
    --save_and_eval_strategy epoch --logging_step 10 --max_his_len 20 --prompt_num 4 --patient 10 \
    --index_file $IDX --image_index_file $IDX --tasks seqrec --valid_task seqrec
T4=$(t)

echo "== [5] test: trie-constrained beam search (20 beams), HR/NDCG =="
torchrun --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) test_ddp_save.py \
    --ckpt_path $OUT --data_path $DATA/ --dataset $D --test_batch_size 64 --num_beams 20 \
    --index_file $IDX --image_index_file $IDX --test_task seqrec \
    --results_file $OUT/results_seqrec_20.json --save_file $OUT/save_seqrec_20.json --filter_items
T5=$(t)

echo "== RESULT =="
python -c "import json; r=json.load(open('$OUT/results_seqrec_20.json')); print(json.dumps(r['mean_results'], indent=1))"
echo "wall seconds: emb=$((T1-T0)) rqvae=$((T2-T1)) sid=$((T3-T2)) t5_train=$((T4-T3)) test=$((T5-T4)) total=$((T5-T0))"
