#!/bin/bash
# One MFD training run (inside the container), identical arguments to the MD chain-decoder control
# (run_baseline.sh case md*), for any v1 MD SID. Env: D, CFG (122|133), AUG (0|1), SEED, M (frozen macrec/ copy).
# AUG=1: multi-view history augmentation (--tasks seqrec_mv, mfd_data.MVMFDDataset). Steps: v1 hash check ->
# sanity checks on an UNTRAINED model (stop on failure) -> run_config.json -> training.
# Run dirs: mmgen_runs/mfd/<D>/mfd<CFG>_stage1/seed<S> (AUG=0) | mfd<CFG>_mv/seed<S> (AUG=1).
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen
case $CFG in
  122) IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2 ;;
  133) IDX=.index_md133_a06_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+3+3 ;;
  *) echo "CFG must be 122 or 133"; exit 1 ;;
esac
if [ "$AUG" = 1 ]; then NAME=mfd${CFG}_mv; TASKS=seqrec_mv; else NAME=mfd${CFG}_stage1; TASKS=seqrec; fi
OUT=$RUNS/mfd/$D/$NAME/seed$SEED; mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
PORT=$((20000 + RANDOM % 20000))
echo "MFD train D=$D CFG=$CFG AUG=$AUG SEED=$SEED -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","seed":$SEED,"index_file":"$IDX","v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)",
 "entry":"finetune_mfd.py","model":"MFDT5 two-branch decoder","tasks":"$TASKS","augmentation":$AUG,"model_class":"plain",
 "per_device_batch_size":256,"gradient_accumulation_steps":4,"effective_batch":1024,"learning_rate":"5e-4",
 "weight_decay":0.01,"epochs":200,"patient":10,"lr_scheduler":"cosine","warmup_ratio":0.01,"max_his_len":20,
 "prompt_num":4,"dataloader_num_workers":2,"control":"baselines/$D/md${CFG}_a0$([ $CFG = 122 ] && echo 5 || echo 6)_ep500_reassign/seed$SEED",
 "git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
python -u mfd_eval.py --random_init --sanity_only --base_model ./config/ckpt --data_path $RUNS/data --dataset $D \
  --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --k_f 20 \
  --out_dir $OUT/sanity_untrained > $OUT/sanity_untrained.log 2>&1 || { echo "SANITY FAILED ($OUT)"; exit 3; }
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT finetune_mfd.py --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size 256 --gradient_accumulation_steps 4 --learning_rate 5e-4 --epochs 200 --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks $TASKS --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
