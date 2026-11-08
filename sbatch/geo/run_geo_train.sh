#!/bin/bash
# One codebook-geometry run (inside the container; docs/MFD_REPORT.md §14, macrec/geo_embed.py, finetune_geo.py).
# Env: D, KIND (chain|mfd), VAR, SEED, M (frozen macrec/ copy), DRYRUN (0|1).
#   VAR  A    codebook-initialised code embeddings          (--geo codebook)
#        C    shuffled-codebook control                     (--geo shuffled --geo_perm_seed 0)
#        AB1  A + soft labels for f, eps = 0.1              (--geo codebook --soft_f_eps 0.1)
#        AB2  A + soft labels for f, eps = 0.2              (--geo codebook --soft_f_eps 0.2)
#   §17 (on top of V2-MFD = KIND=mfd VAR=V2):
#        P1a  one encoder position per history item, up to 400 epochs   (--geo codebook --item_enc)
#        P1b  / P1c  f loss weight 2 / 3                                (--geo codebook --f_loss_weight 2|3)
#        P1d  / P1e  offset penalty lambda 1e-3 / 1e-2                  (--geo codebook --offset_lambda 1e-3|1e-2)
#   §18 (schedule check on Arts):
#        V2s  V2-MFD with P1a's schedule (max 400 epochs, cosine over 400)  (--geo codebook; EPOCHS=400)
#        P1a200  P1a with V2-MFD's schedule (max 200 epochs), the other corner of the 2 x 2  (--geo codebook --item_enc)
#   §20 Task 2 (on top of P1a, same 400-epoch cap):
#        P2a / P2b  P1a + dense item-level head, lambda 0.1 / 0.5  (--geo codebook --item_enc --dense_lambda 0.1|0.5)
#   §15 (disentangling AB2):
#        V1   shuffled init + AB2's soft labels             (--geo shuffled --geo_perm_seed 0 --soft_f_eps 0.2)
#        V2   codebook init only (= A), one-hot f           (--geo codebook)
#        V3   standard embeddings + uniform label smoothing on f, eps 0.2   (--geo none --soft_f_eps 0.2 --soft_f_mode uniform)
#        V4   codebook init + geometric soft labels, T putting 70% of the eps mass on the 10 nearest f codes
#             (--geo codebook --soft_f_eps 0.2 --soft_f_nn_mass 0.7)
# KIND=chain: every other argument = the MD chain-decoder control (run_baseline.sh B=md122_a05_ep500_reassign: seqrec,
#   finetune_ext path) -> mmgen_runs/geo/<D>/chain122_<VAR>/seed<S>.
# KIND=mfd: every other argument = mfd122_mv (run_mfd_train.sh CFG=122 AUG=1: seqrec_mv, finetune_mfd path)
#   -> mmgen_runs/mfd/<D>/mfd122_mv_geo<VAR>/seed<S> (so sbatch/mfd/eval_mt.sbatch evaluates it unchanged).
# Gate before training: geo_check.json all_pass (sanity 1-2) and geo_config_diff.json runs.<run>.ok (sanity 3).
# DRYRUN=1: the exact command through geo_dryrun.py into mmgen_runs/geo/gate/dryrun/<D>/<run> (no training).
set -euo pipefail
source /ext3/env.sh && conda activate py310
RUNS=/scratch/yw8866/mmgen_runs; REPO=/scratch/yw8866/mmgen; DRYRUN=${DRYRUN:-0}; EPOCHS=200
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
case $VAR in
  A)   GEOARGS="--geo codebook" ;;
  C)   GEOARGS="--geo shuffled --geo_perm_seed 0" ;;
  AB1) GEOARGS="--geo codebook --soft_f_eps 0.1" ;;
  AB2) GEOARGS="--geo codebook --soft_f_eps 0.2" ;;
  V1)  GEOARGS="--geo shuffled --geo_perm_seed 0 --soft_f_eps 0.2" ;;
  V2)  GEOARGS="--geo codebook" ;;
  V3)  GEOARGS="--geo none --soft_f_eps 0.2 --soft_f_mode uniform" ;;
  V4)  GEOARGS="--geo codebook --soft_f_eps 0.2 --soft_f_nn_mass 0.7" ;;
  P1a) GEOARGS="--geo codebook --item_enc"; EPOCHS=400 ;;
  V2s) GEOARGS="--geo codebook"; EPOCHS=400 ;;
  P1a200) GEOARGS="--geo codebook --item_enc" ;;
  P2a) GEOARGS="--geo codebook --item_enc --dense_lambda 0.1"; EPOCHS=400 ;;
  P2b) GEOARGS="--geo codebook --item_enc --dense_lambda 0.5"; EPOCHS=400 ;;
  P1b) GEOARGS="--geo codebook --f_loss_weight 2" ;;
  P1c) GEOARGS="--geo codebook --f_loss_weight 3" ;;
  P1d) GEOARGS="--geo codebook --offset_lambda 1e-3" ;;
  P1e) GEOARGS="--geo codebook --offset_lambda 1e-2" ;;
  *) echo "VAR must be A, C, AB1, AB2, V1..V4 or P1a..P1e"; exit 1 ;;
esac
case $KIND in
  chain) NAME=chain122_$VAR; OUT=$RUNS/geo/$D/$NAME/seed$SEED; TASKS=seqrec; MFDARG=""; DATAP=$RUNS/data/ ;;   # run_baseline.sh's spelling
  mfd)   NAME=mfd122_mv_geo$VAR; OUT=$RUNS/mfd/$D/$NAME/seed$SEED; TASKS=seqrec_mv; MFDARG="--mfd"; DATAP=$RUNS/data ;;
  *) echo "KIND must be chain or mfd"; exit 1 ;;
esac
[ "$DRYRUN" = 1 ] && { OUT=$RUNS/geo/gate/dryrun/$D/$NAME; rm -rf $OUT; }
mkdir -p $OUT
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
TOKCK=$V1/tokenizer.pth
PORT=$((20000 + RANDOM % 20000))
echo "geo train D=$D KIND=$KIND VAR=$VAR SEED=$SEED -> $OUT (commit $(cd $REPO && git rev-parse --short HEAD))"
if [ "$DRYRUN" != 1 ]; then
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/$D/geo_check.json'))['all_pass'] else 1)" || { echo "GEO CHECK NOT PASSED"; exit 3; }
  python -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/geo/gate/$D/geo_config_diff.json'))['runs']['$NAME']['ok'] else 1)" \
    || { echo "CONFIG DIFF NOT OK ($NAME)"; exit 3; }
fi
if [ ! -f $OUT/train_stats.json ]; then
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"$NAME","kind":"$KIND","variant":"$VAR","geo_args":"$GEOARGS","seed":$SEED,"index_file":"$IDX",
 "v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)","tokenizer_ckpt":"$TOKCK","tokenizer_sha256":"$(sha256sum < $TOKCK | cut -d' ' -f1)",
 "entry":"finetune_geo.py","tasks":"$TASKS","model_class":"plain","per_device_batch_size":256,"gradient_accumulation_steps":4,
 "effective_batch":1024,"learning_rate":"5e-4","weight_decay":0.01,"epochs":$EPOCHS,"patient":10,"lr_scheduler":"cosine",
 "warmup_ratio":0.01,"max_his_len":20,"prompt_num":4,"dataloader_num_workers":2,
 "control":"$([ $KIND = chain ] && echo baselines/$D/md122_a05_ep500_reassign/seed$SEED || echo mfd/$D/mfd122_mv/seed$SEED)",
 "git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"${SLURM_JOB_ID:-}"}
JSON
fi
cd $M
ENTRY=finetune_geo.py; [ "$DRYRUN" = 1 ] && ENTRY="geo_dryrun.py --dump_file $OUT/dryrun.json"
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT $ENTRY $GEOARGS $MFDARG --geo_ckpt $TOKCK \
  --geo_report $OUT/geo_init.json --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $DATAP --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size 256 --gradient_accumulation_steps 4 --learning_rate 5e-4 --epochs $EPOCHS --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks $TASKS --valid_task seqrec > $OUT/train.log 2>&1
echo "done $OUT"
