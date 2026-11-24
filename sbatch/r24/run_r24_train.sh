#!/bin/bash
# One round-24 training run (inside the container; docs/MFD_REPORT.md §24). Base = P1a exactly (§19: finetune_geo.py --mfd --geo
# codebook --item_enc, seqrec_mv, 256 x 4, lr 5e-4, wd 0.01, cosine, warmup 0.01, 400 epochs, patience 10 on validation loss),
# run through macrec/finetune_r24.py with the settings named by VAR = tokens joined by "_" (one factor per token):
#   P1aref     P1a's exact command (dry-run reference of the config diff only)
#   lr3e4 lr1e3 lr2e3            learning rate 3e-4 / 1e-3 / 2e-3 (P1a 5e-4)
#   do02 do03                    T5 dropout_rate 0.2 / 0.3 (P1a 0.1)               (edited copy of config/ckpt)
#   wd01                         weight decay 0.1 = 10 x P1a's 0.01
#   gn5                          max_grad_norm 5 (P1a 1.0)                          (--r24_max_grad_norm)
#   dm256                        d_model 256, d_ff 2048, 12 heads, d_kv 64 (P1a 128 / 1024 / 6 / 64)
#   l6                           6 encoder + 6 decoder layers (P1a 4 + 4)
#   B1 B2                        Track B item examples, loss weight 1.0 / 0.3      (--r24_item; macrec/r24_items.py)
#   e.g. VAR=do02, VAR=dm256_do02, VAR=B2, VAR=do02_B2
# Env: D, VAR, SEED, M (frozen macrec/ copy), DRYRUN (0|1), SMOKE (0|1).
# -> mmgen_runs/mfd/<D>/r24_<VAR>/seed<S>. Gate: geo/<D>/r24_check.json all_pass (B runs) + geo/gate/<D>/r24_config_diff.json
#    runs.r24_<VAR>.ok + geo/gate/r24_smoke/<D>/r24_<VAR>/smoke.json ok.
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen; DRYRUN=${DRYRUN:-0}; SMOKE=${SMOKE:-0}; EPOCHS=400
SID=1+2+2; IDX=.index_md122_a05_ep500_reassign.json
LR=5e-4; WD=0.01; DROP=0.1; DM=128; DFF=1024; NH=6; NL=4; EXTRA=""
for t in ${VAR//_/ }; do
  case $t in
    P1aref) ;;
    lr3e4) LR=3e-4 ;;  lr1e3) LR=1e-3 ;;  lr2e3) LR=2e-3 ;;
    do02)  DROP=0.2 ;; do03)  DROP=0.3 ;;
    wd01)  WD=0.1 ;;
    gn5)   EXTRA="$EXTRA --r24_max_grad_norm 5" ;;
    dm256) DM=256; DFF=2048; NH=12 ;;
    l6)    NL=6 ;;
    B1)    EXTRA="$EXTRA --r24_item 1.0" ;;
    B2)    EXTRA="$EXTRA --r24_item 0.3" ;;
    *) echo "unknown VAR token $t (VAR=$VAR)"; exit 1 ;;
  esac
done
V1=$RUNS/md/sids/v1/$D/$SID; TOKCK=$V1/tokenizer.pth
NAME=r24_$VAR; OUT=$RUNS/mfd/$D/$NAME/seed$SEED
[ "$DRYRUN" = 1 ] && { OUT=$RUNS/geo/gate/r24_dryrun/$D/$NAME; rm -rf $OUT; }
[ "$SMOKE" = 1 ] && { OUT=$RUNS/geo/gate/r24_smoke/$D/$NAME; rm -rf $OUT; }
mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1 ($SID)"; exit 1; }
# base model dir: config/ckpt with the edited T5 config (tokenizer files unchanged)
BASE=$OUT/base_model; rm -rf $BASE; mkdir -p $BASE; cp $M/config/ckpt/* $BASE/
python - <<PY
import json
p = "$BASE/config.json"; c = json.load(open(p))
c.update(dropout_rate=$DROP, d_model=$DM, d_ff=$DFF, num_heads=$NH, d_kv=64, num_layers=$NL, num_decoder_layers=$NL)
json.dump(c, open(p, "w"), indent=2)
PY
PORT=$((20000 + RANDOM % 20000))
echo "r24 train D=$D VAR=$VAR SEED=$SEED lr=$LR wd=$WD dropout=$DROP d_model=$DM d_ff=$DFF heads=$NH layers=$NL+$NL extra='$EXTRA' -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ "$DRYRUN" != 1 ] && [ "$SMOKE" != 1 ]; then
  if [[ $VAR == *B1* || $VAR == *B2* ]]; then
    python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/r24_check.json'))['all_pass'] else 1)" || { echo "R24 CHECK NOT PASSED"; exit 3; }
  fi
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/r24_smoke/$D/$NAME/smoke.json'))['ok'] else 1)" || { echo "R24 SMOKE NOT PASSED ($NAME)"; exit 3; }
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/$D/r24_config_diff.json'))['runs']['$NAME']['ok'] else 1)" \
    || { echo "R24 CONFIG DIFF NOT OK ($NAME)"; exit 3; }
fi
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","variant":"$VAR","base":"P1a (§19)","r24_args":"$EXTRA","learning_rate":"$LR","weight_decay":$WD,
 "dropout_rate":$DROP,"d_model":$DM,"d_ff":$DFF,"num_heads":$NH,"num_layers":$NL,"sid":"$SID","geo_args":"--geo codebook --item_enc",
 "seed":$SEED,"index_file":"$IDX","v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)","tokenizer_ckpt":"$TOKCK",
 "entry":"finetune_r24.py","tasks":"seqrec_mv","per_device_batch_size":256,"gradient_accumulation_steps":4,"epochs":$EPOCHS,"patient":10,
 "lr_scheduler":"cosine","warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,"control":"mfd/$D/mfd122_mv_geoP1a/seed$SEED",
 "git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
ENTRY="finetune_r24.py $EXTRA"; [ "$DRYRUN" = 1 ] && ENTRY="$ENTRY --dump_file $OUT/dryrun.json"
[ "$SMOKE" = 1 ] && ENTRY="$ENTRY --smoke_steps 5"
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT $ENTRY --geo codebook --item_enc --mfd --geo_ckpt $TOKCK \
  --geo_report $OUT/geo_init.json --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data --dataset $D --output_dir $OUT --base_model $BASE \
  --per_device_batch_size 256 --gradient_accumulation_steps 4 --learning_rate $LR --epochs $EPOCHS --weight_decay $WD \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks seqrec_mv --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
