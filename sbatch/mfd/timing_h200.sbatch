#!/bin/bash
# ms/user on an H200 with the procedure of the reported runs, written to a separate folder (reported results untouched).
#   MODE=mfd CK=<MFD run dir> [KF=100 WF=1.0 WT=0.5 WI=0.5]  mfd_eval.py at one setting, batch 64 -> <CK>/timing_h200/
#   MODE=c   D=<dataset>                                      C seed 42 test beam search (eval_split.py, 20 beams, batch 64,
#                                                             seqrec + seqimage) -> mmgen_runs/timing/<D>/mql/
# Frozen snapshot of macrec/*.py. Submit: sbatch --export=ALL,MODE=mfd,CK=... sbatch/mfd/timing_h200.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_timing
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
MODE=${MODE:-mfd}; KF=${KF:-100}; WF=${WF:-1.0}; WT=${WT:-0.5}; WI=${WI:-0.5}
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) MODE=$MODE ${CK:-} ${D:-} commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
if [ $MODE = mfd ]; then
  D=$(python3 -c "import json; print(json.load(open('$CK/run_config.json'))['dataset'])")
  IDX=.index_md122_a05_ep500_reassign.json
  OUT=$CK/timing_h200; mkdir -p $OUT
  CMD="python -u mfd_eval.py --ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX \
    --max_his_len 20 --test_batch_size 64 --tasks seqrec_mv --prompt_num 4 --k_f $KF --w_f $WF --w_t $WT --w_i $WI --out_dir $OUT"
  LOG=$OUT/eval.log
else
  CK=$RUNS/baselines/$D/mql/seed42; OUT=$RUNS/timing/$D/mql; mkdir -p $OUT
  CMD=""
  for T in seqrec seqimage; do
    CMD="$CMD torchrun --nproc_per_node=1 --master_port=\$((20000 + RANDOM % 20000)) eval_split.py --split test --ckpt_path $CK \
      --data_path $RUNS/data/ --dataset $D --test_batch_size 64 --num_beams 20 --index_file .index_lemb_mql.json \
      --image_index_file .index_vitemb_mql.json --test_task $T --results_file $OUT/results_${T}_20.json \
      --save_file $OUT/save_${T}_20.json --stats_file $OUT/cost_${T}.json --filter_items > $OUT/eval_${T}.log 2>&1;"
  done
  LOG=/dev/null
fi
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
$CMD ${LOG:+> $LOG 2>&1}
"
echo done
