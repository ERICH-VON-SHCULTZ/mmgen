#!/bin/bash
# One round-22 training run (inside the container; docs/MFD_REPORT.md §22). Base = A01 exactly as round 21
# (finetune_r21.py --r21_mode A --r21_beta 0.1; grouped views, 341 pairs x 3 rows, accumulation 1; P1a's other settings).
# Env: D, VAR, SEED, M (frozen macrec/ copy), DRYRUN (0|1).
#   VAR  E2g01 / E2g05   A01 + aux t1 / i1 heads at the f position, gamma 0.1 / 0.5 (--r22_aux_gamma; macrec/r22_aux.py)
#        E1              A01 on the locked 1+3+3 SIDs (index md133_a06_ep500_reassign, its v1 tokenizer for codebook init;
#                        encoder sums all 7 codes, decoder branches 3 steps each)
#        E7a / E7b       §22.7: A01 + zero-initialised W on the item's [f; text; image] code sums (a) / on its f embedding (b)
#                        (--r22_e7 a|b; macrec/r22_e7.py). Also needs geo/<D>/r22_e7_check.json all_pass + the 5-step smoke run.
#        E8a / E8b       §22.8: f embedding E(f) + W c(f), c = mean child-code sum (a: t1 t2 i1 i2, b: t1 i1), W zero-init (--r22_e8)
#        E9k2 / E9k4     §22.9: mixture of K = 2 / 4 softmaxes at the f step (--r22_e9_k). E8 / E9 need geo/<D>/r22_e89_check.json
#                        all_pass + the 5-step smoke run.
# -> mmgen_runs/mfd/<D>/r22_<VAR>/seed<S>. Gate: geo/<D>/r22_check.json all_pass + geo/gate/<D>/r22_config_diff.json runs.r22_<VAR>.ok
# DRYRUN=1: dumps the resolved config into mmgen_runs/geo/gate/r22_dryrun/<D>/r22_<VAR> (no training).
# SMOKE=1: the exact command for 5 real optimizer steps (--smoke_steps 5) -> geo/gate/r22_smoke/<D>/r22_<VAR>/smoke.json, then exit.
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen; DRYRUN=${DRYRUN:-0}; SMOKE=${SMOKE:-0}; EPOCHS=400
SID=1+2+2; IDX=.index_md122_a05_ep500_reassign.json; EXTRA=""
case $VAR in
  E2g01) EXTRA="--r22_aux_gamma 0.1" ;;
  E2g05) EXTRA="--r22_aux_gamma 0.5" ;;
  E1)    SID=1+3+3; IDX=.index_md133_a06_ep500_reassign.json ;;
  E7a)   EXTRA="--r22_e7 a" ;;
  E7b)   EXTRA="--r22_e7 b" ;;
  E8a)   EXTRA="--r22_e8 a" ;;
  E8b)   EXTRA="--r22_e8 b" ;;
  E9k2)  EXTRA="--r22_e9_k 2" ;;
  E9k4)  EXTRA="--r22_e9_k 4" ;;
  *) echo "VAR must be E2g01, E2g05, E1, E7a, E7b, E8a, E8b, E9k2 or E9k4"; exit 1 ;;
esac
V1=$RUNS/md/sids/v1/$D/$SID; TOKCK=$V1/tokenizer.pth
NAME=r22_$VAR; OUT=$RUNS/mfd/$D/$NAME/seed$SEED
[ "$DRYRUN" = 1 ] && { OUT=$RUNS/geo/gate/r22_dryrun/$D/$NAME; rm -rf $OUT; }
[ "$SMOKE" = 1 ] && { OUT=$RUNS/geo/gate/r22_smoke/$D/$NAME; rm -rf $OUT; }
mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1 ($SID)"; exit 1; }
PORT=$((20000 + RANDOM % 20000))
echo "r22 train D=$D VAR=$VAR SEED=$SEED SID=$SID -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ "$DRYRUN" != 1 ] && [ "$SMOKE" != 1 ]; then
  if [[ $VAR == E7* ]]; then
    python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/r22_e7_check.json'))['all_pass'] else 1)" || { echo "R22 E7 CHECK NOT PASSED"; exit 3; }
  fi
  if [[ $VAR == E8* || $VAR == E9* ]]; then
    python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/r22_e89_check.json'))['all_pass'] else 1)" || { echo "R22 E8/E9 CHECK NOT PASSED"; exit 3; }
  fi
  if [[ $VAR == E7* || $VAR == E8* || $VAR == E9* ]]; then
    python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/r22_smoke/$D/$NAME/smoke.json'))['ok'] else 1)" || { echo "R22 SMOKE NOT PASSED ($NAME)"; exit 3; }
  fi
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/r22_check.json'))['all_pass'] else 1)" || { echo "R22 CHECK NOT PASSED"; exit 3; }
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/$D/r22_config_diff.json'))['runs']['$NAME']['ok'] else 1)" \
    || { echo "R22 CONFIG DIFF NOT OK ($NAME)"; exit 3; }
fi
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","variant":"$VAR","base":"A01 (round 21)","r21_args":"--r21_mode A --r21_beta 0.1","r22_args":"$EXTRA",
 "sid":"$SID","geo_args":"--geo codebook --item_enc","seed":$SEED,"index_file":"$IDX","v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)",
 "tokenizer_ckpt":"$TOKCK","tokenizer_sha256":"$(sha256sum < $TOKCK | cut -d' ' -f1)","entry":"finetune_r21.py","tasks":"seqrec_mv",
 "per_device_batch_size":341,"gradient_accumulation_steps":1,"sequences_per_step":1023,"learning_rate":"5e-4","weight_decay":0.01,
 "epochs":$EPOCHS,"patient":10,"lr_scheduler":"cosine","warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,
 "control":"mfd/$D/r21_A01/seed$SEED","git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
ENTRY="finetune_r21.py --r21_mode A --r21_beta 0.1 $EXTRA"; [ "$DRYRUN" = 1 ] && ENTRY="$ENTRY --dump_file $OUT/dryrun.json"
[ "$SMOKE" = 1 ] && ENTRY="$ENTRY --smoke_steps 5"
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT $ENTRY --geo codebook --item_enc --mfd --geo_ckpt $TOKCK \
  --geo_report $OUT/geo_init.json --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size 341 --gradient_accumulation_steps 1 --learning_rate 5e-4 --epochs $EPOCHS --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks seqrec_mv --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
