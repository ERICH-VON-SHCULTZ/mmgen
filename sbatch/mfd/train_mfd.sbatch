#!/bin/bash
# MFD stage 1 training (macrec/finetune_mfd.py), identical arguments to the MD chain-decoder control
# (sbatch/baselines/run_baseline.sh, case md*). Steps: sanity checks on an UNTRAINED model (stop on
# failure) -> run_config.json -> training. Runs a frozen snapshot of macrec/*.py.
# Submit: sbatch --export=ALL,D=Instruments,CFG=122,SEED=42 sbatch/mfd/train_mfd.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_train
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-mfd WANDB_DIR=$RUNS/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
D=${D:-Instruments}; CFG=${CFG:-122}; SEED=${SEED:-42}
case $CFG in 122) IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2 ;; *) echo "stage 1 is 1+2+2 only"; exit 1 ;; esac
OUT=$RUNS/mfd/$D/mfd${CFG}_stage1/seed$SEED; mkdir -p $OUT
# frozen code snapshot (later edits cannot affect this job)
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
# the data-dir index must be the locked v1 set
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
PORT=$((20000 + RANDOM % 20000))
echo "job $SLURM_JOB_ID on $(hostname) D=$D CFG=$CFG SEED=$SEED commit $(cd $REPO && git rev-parse --short HEAD)"
cat > $OUT/run_config.json <<JSON
{"dataset":"$D","run":"mfd${CFG}_stage1","seed":$SEED,"index_file":"$IDX","v1_index_sha256":"$(cut -d' ' -f1 $V1/index.sha256)",
 "entry":"finetune_mfd.py","model":"MFDT5 two-branch decoder","tasks":"seqrec","model_class":"plain",
 "per_device_batch_size":256,"gradient_accumulation_steps":4,"effective_batch":1024,"learning_rate":"5e-4",
 "weight_decay":0.01,"epochs":200,"patient":10,"lr_scheduler":"cosine","warmup_ratio":0.01,"max_his_len":20,
 "prompt_num":4,"dataloader_num_workers":2,"control":"baselines/$D/md${CFG}_a05_ep500_reassign/seed$SEED",
 "git_commit":"$(cd $REPO && git rev-parse HEAD)","job_id":"$SLURM_JOB_ID"}
JSON
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u mfd_eval.py --random_init --sanity_only --base_model ./config/ckpt --data_path $RUNS/data --dataset $D \
  --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --k_f 20 \
  --out_dir $OUT/sanity_untrained > $OUT/sanity_untrained.log 2>&1
[ -f $OUT/train_stats.json ] || torchrun --nproc_per_node=1 --master_port=$PORT finetune_mfd.py --model_class plain \
  --stats_file $OUT/train_stats.json --dataloader_num_workers 2 --run_config_file $OUT/run_config.json \
  --seed $SEED --data_path $RUNS/data --dataset $D --output_dir $OUT --base_model ./config/ckpt \
  --per_device_batch_size 256 --gradient_accumulation_steps 4 --learning_rate 5e-4 --epochs 200 --weight_decay 0.01 \
  --save_and_eval_strategy epoch --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file $IDX --image_index_file $IDX --tasks seqrec --valid_task seqrec > $OUT/train.log 2>&1
"
echo done
