#!/bin/bash
# One round-21 training run (inside the container; docs/MFD_REPORT.md §21, macrec/r21_views.py, finetune_r21.py).
# Env: D, VAR, SEED, M (frozen macrec/ copy), DRYRUN (0|1).
#   VAR  G0          grouped views, no new loss                  (--r21_mode G0;  341 pairs x 3 views per step, accumulation 1)
#        A01 / A05   G0 + ensemble-teacher KL, beta 0.1 / 0.5    (--r21_mode A --r21_beta 0.1|0.5; same batch as G0)
#        B01 / B05   G0 + mutual view KL, beta 0.1 / 0.5         (--r21_mode B --r21_beta 0.1|0.5; same batch as G0)
#        C033 / C060 item-level random views, p_full 1/3 / 0.6   (--r21_mode C --r21_p_full ..; P1a's batch 256 x 4)
#        P1aref      DRYRUN only: the exact P1a command (finetune_geo.py), reference of the config gate
# Every other argument = P1a (sbatch/geo/run_geo_train.sh KIND=mfd VAR=P1a): codebook init, item-level encoder, MFD, seqrec_mv,
# LR 5e-4 cosine, warmup 0.01, wd 0.01, 400 epochs, early stopping patience 10 on validation loss.
# -> mmgen_runs/mfd/<D>/r21_<VAR>/seed<S>. Gate: geo/<D>/r21_check.json all_pass + geo/gate/<D>/r21_config_diff.json runs.r21_<VAR>.ok
# DRYRUN=1: through config_dryrun into mmgen_runs/geo/gate/r21_dryrun/<D>/<run> (no training).
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen; DRYRUN=${DRYRUN:-0}; EPOCHS=400
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
BS=341; GA=1
case $VAR in
  G0)   R21="--r21_mode G0" ;;
  A01)  R21="--r21_mode A --r21_beta 0.1" ;;
  A05)  R21="--r21_mode A --r21_beta 0.5" ;;
  B01)  R21="--r21_mode B --r21_beta 0.1" ;;
  B05)  R21="--r21_mode B --r21_beta 0.5" ;;
  C033) R21="--r21_mode C --r21_p_full 0.3333333333333333"; BS=256; GA=4 ;;
  C060) R21="--r21_mode C --r21_p_full 0.6"; BS=256; GA=4 ;;
  P1aref) R21=""; BS=256; GA=4; [ "$DRYRUN" = 1 ] || { echo "P1aref is dry-run only"; exit 1; } ;;
  *) echo "VAR must be G0, A01, A05, B01, B05, C033, C060 or P1aref"; exit 1 ;;
esac
NAME=r21_$VAR; [ $VAR = P1aref ] && NAME=P1aref
OUT=$RUNS/mfd/$D/$NAME/seed$SEED
[ "$DRYRUN" = 1 ] && { OUT=$RUNS/geo/gate/r21_dryrun/$D/$NAME; rm -rf $OUT; }
mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
TOKCK=$V1/tokenizer.pth
PORT=$((20000 + RANDOM % 20000))
echo "r21 train D=$D VAR=$VAR SEED=$SEED -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ "$DRYRUN" != 1 ]; then
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/r21_check.json'))['all_pass'] else 1)" || { echo "R21 CHECK NOT PASSED"; exit 3; }
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/$D/r21_config_diff.json'))['runs']['$NAME']['ok'] else 1)" \
    || { echo "R21 CONFIG DIFF NOT OK ($NAME)"; exit 3; }
fi
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","variant":"$VAR","r21_args":"$R21","geo_args":"--geo codebook --item_enc","seed":$SEED,"index_file":"$IDX",
 "v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)","tokenizer_ckpt":"$TOKCK","tokenizer_sha256":"$(sha256sum < $TOKCK | cut -d' ' -f1)",
 "entry":"$([ $VAR = P1aref ] && echo finetune_geo.py || echo finetune_r21.py)","tasks":"seqrec_mv","model_class":"plain",
 "per_device_batch_size":$BS,"gradient_accumulation_steps":$GA,"sequences_per_step":$([ $BS = 341 ] && echo 1023 || echo 1024),
 "learning_rate":"5e-4","weight_decay":0.01,"epochs":$EPOCHS,"patient":10,"lr_scheduler":"cosine",
 "warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,"dataloader_num_workers":2,"control":"mfd/$D/mfd122_mv_geoP1a/seed$SEED",
 "git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
if [ $VAR = P1aref ]; then ENTRY="geo_dryrun.py --dump_file $OUT/dryrun.json"
else ENTRY="finetune_r21.py $R21"; [ "$DRYRUN" = 1 ] && ENTRY="$ENTRY --dump_file $OUT/dryrun.json"; fi
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT $ENTRY --geo codebook --item_enc --mfd --geo_ckpt $TOKCK \
  --geo_report $OUT/geo_init.json --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size $BS --gradient_accumulation_steps $GA --learning_rate 5e-4 --epochs $EPOCHS --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks seqrec_mv --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
