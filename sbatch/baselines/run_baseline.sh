#!/bin/bash
# One baseline run (inside the container). Env: D (Instruments|Arts|Games), B (tiger_text|tiger_image|tiger_fnorm|tiger_fnormj|tiger_fnormj_ep500|tiger_fraw|mql|
# macrec|macrecpaper), SEED, STAGE (train|eval|all; default all), NW (dataloader workers; default 2). Fine-tune with MACRec's trainer, then evaluate valid+test with trie beam
# search (20 beams), single lists and (C/D) the MQL4GRec/MACRec merge (ensemble.py).
set -euo pipefail
source /ext3/env.sh && conda activate py310
REPO=/scratch/yw8866/mmgen; M=$REPO/macrec
DATA=/scratch/yw8866/mmgen_runs/data/
OUT=/scratch/yw8866/mmgen_runs/baselines/$D/$B/seed$SEED; mkdir -p $OUT/eval/valid $OUT/eval/test
PORT=$((20000 + RANDOM % 20000))
STAGE=${STAGE:-all}; NW=${NW:-2}

# Effective batch = per_device x 2 GPUs x grad_accum 2 in both repos' launchers (torchrun --nproc_per_node=2,
# utils.py default gradient_accumulation_steps=2). On 1 GPU we keep it with grad_accum 4.
case $B in
  tiger_text)  IDX=.index_lemb_tiger.json;   IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256;  LR=5e-4 ;;
  tiger_image) IDX=.index_vitemb_tiger.json; IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256;  LR=5e-4 ;;
  # E early-fusion TIGER: identical to A/B except the tokenizer input (concat of L2-normed / raw text+image)
  tiger_fnorm) IDX=.index_fnorm_tiger.json;  IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256;  LR=5e-4 ;;
  # MD-RQ-VAE + chain decoder (Part 3): SID [shared, text chain, image chain] as one sequence, TIGER settings
  md*)         IDX=.index_$B.json;          IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256;  LR=5e-4 ;;
  tiger_fnormj) IDX=.index_fnormj_tiger.json; IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256; LR=5e-4 ;;
  tiger_fnormj_ep500) IDX=.index_fnormj_ep500_tiger.json; IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256; LR=5e-4 ;;
  tiger_fraw)  IDX=.index_fraw_tiger.json;   IIDX=$IDX; TASKS=seqrec; TESTS=seqrec; MC=plain; BS=256;  LR=5e-4 ;;
  mql)         IDX=.index_lemb_mql.json; IIDX=.index_vitemb_mql.json
               TASKS=seqrec,seqimage,item2image,image2item,fusionseqrec; TESTS="seqrec seqimage"; MC=plain; BS=256; LR=5e-4 ;;
  macrec|macrecpaper) IDX=.index_lemb_$B.json; IIDX=.index_vitemb_$B.json
               TASKS=seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image; TESTS="seqrec seqimage"
               MC=contrastive; BS=1024; LR=1e-3 ;;
esac
# run_config.json: written BEFORE training; finetune_ext adds every resolved argument + the T5 config.
# (config.json in $OUT is the HF model config, overwritten on every save.) Eval stage: config_eval.json.
CFG=$OUT/run_config.json; [ "$STAGE" = eval ] && CFG=$OUT/config_eval.json
[ "$STAGE" != eval ] && [ -f $OUT/train_stats.json ] && CFG=$OUT/config_rerun.json   # never clobber a finished run's config
cat > $CFG <<EOF
{"dataset":"$D","baseline":"$B","seed":$SEED,"index_file":"$IDX","image_index_file":"$IIDX","tasks":"$TASKS",
 "test_tasks":"$TESTS","model_class":"$MC","per_device_batch_size":$BS,"gradient_accumulation_steps":4,
 "effective_batch":$((BS*4)),"learning_rate":"$LR","weight_decay":0.01,"epochs":200,"patient":10,
 "lr_scheduler":"cosine","warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,"num_beams":20,
 "dataloader_num_workers":$NW,"tokenizer_dir":"/scratch/yw8866/mmgen_runs/baselines/$D/tok","git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
EOF
cd $M
if [ "$STAGE" != eval ] && [ ! -f $OUT/train_stats.json ]; then
  echo "== train $D $B seed $SEED"; date
  torchrun --nproc_per_node=1 --master_port=$PORT finetune_ext.py --model_class $MC --stats_file $OUT/train_stats.json --dataloader_num_workers $NW \
    --run_config_file $CFG \
    --seed $SEED --data_path $DATA --dataset $D --output_dir $OUT --base_model ./config/ckpt \
    --per_device_batch_size $BS --gradient_accumulation_steps 4 --learning_rate $LR --epochs 200 --weight_decay 0.01 \
    --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
    --index_file $IDX --image_index_file $IIDX --tasks $TASKS --valid_task seqrec > $OUT/train.log 2>&1
fi
[ "$STAGE" = train ] && exit 0
for SPLIT in valid test; do
  for T in $TESTS; do
    if [ ! -f $OUT/eval/$SPLIT/results_${T}_20.json ]; then
      echo "== eval $SPLIT $T"; date
      torchrun --nproc_per_node=1 --master_port=$((PORT+1)) eval_split.py --split $SPLIT --ckpt_path $OUT \
        --data_path $DATA --dataset $D --test_batch_size 64 --num_beams 20 --index_file $IDX --image_index_file $IIDX \
        --test_task $T --results_file $OUT/eval/$SPLIT/results_${T}_20.json --save_file $OUT/eval/$SPLIT/save_${T}_20.json \
        --stats_file $OUT/eval/$SPLIT/cost_${T}.json --filter_items > $OUT/eval/$SPLIT/eval_${T}.log 2>&1
    fi
  done
  if [ "$TESTS" = "seqrec seqimage" ]; then
    S=$(python -c "import time; print(time.time())")
    python ensemble.py --output_dir $OUT/eval/$SPLIT --dataset $D --data_path $DATA --index_file $IDX \
      --image_index_file $IIDX --num_beams 20 > $OUT/eval/$SPLIT/ensemble.log 2>&1
    python -c "import time, json; json.dump({'merge_seconds': time.time() - $S}, open('$OUT/eval/$SPLIT/cost_merge.json', 'w'))"
  fi
done
python $REPO/sbatch/tools/summarize_run.py $OUT
