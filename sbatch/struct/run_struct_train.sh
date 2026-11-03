#!/bin/bash
# One chain-decoder run with item-level encoder structure (inside the container; docs/MFD_REPORT.md §16,
# macrec/enc_struct.py, finetune_struct.py). Env: D, VAR (E1|E2), SEED, M (frozen macrec/ copy), DRYRUN (0|1).
# Every other argument = the MD chain-decoder control (run_baseline.sh B=md122_a05_ep500_reassign: seqrec, finetune_ext
# path, data_path spelled with its trailing slash) -> mmgen_runs/struct/<D>/chain122_<VAR>/seed<S>.
# Gate before training: struct_check.json all_pass (sanity 4-5) and struct_config_diff.json runs.<run>.ok (sanity 1).
# DRYRUN=1: the exact command through struct_dryrun.py into mmgen_runs/struct/gate/dryrun/<D>/<run> (no training).
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen; DRYRUN=${DRYRUN:-0}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
case $VAR in E1|E2) ;; *) echo "VAR must be E1 or E2"; exit 1 ;; esac
NAME=chain122_$VAR; OUT=$RUNS/struct/$D/$NAME/seed$SEED
[ "$DRYRUN" = 1 ] && { OUT=$RUNS/struct/gate/dryrun/$D/$NAME; rm -rf $OUT; }
mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
PORT=$((20000 + RANDOM % 20000))
echo "struct train D=$D VAR=$VAR SEED=$SEED -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ "$DRYRUN" != 1 ]; then
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/struct/gate/$D/struct_check.json'))['all_pass'] else 1)" || { echo "STRUCT CHECK NOT PASSED"; exit 3; }
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/struct/gate/$D/struct_config_diff.json'))['runs']['$NAME']['ok'] else 1)" \
    || { echo "CONFIG DIFF NOT OK ($NAME)"; exit 3; }
fi
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","variant":"$VAR","seed":$SEED,"index_file":"$IDX","v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)",
 "entry":"finetune_struct.py","tasks":"seqrec","model_class":"plain","per_device_batch_size":256,"gradient_accumulation_steps":4,
 "effective_batch":1024,"learning_rate":"5e-4","weight_decay":0.01,"epochs":200,"patient":10,"lr_scheduler":"cosine",
 "warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,"dataloader_num_workers":2,
 "control":"baselines/$D/md122_a05_ep500_reassign/seed$SEED","git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
ENTRY=finetune_struct.py; [ "$DRYRUN" = 1 ] && ENTRY="struct_dryrun.py --dump_file $OUT/dryrun.json"
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT $ENTRY --struct $VAR --struct_report $OUT/struct_init.json \
  --model_class plain --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data/ --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size 256 --gradient_accumulation_steps 4 --learning_rate 5e-4 --epochs 200 --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks seqrec --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
