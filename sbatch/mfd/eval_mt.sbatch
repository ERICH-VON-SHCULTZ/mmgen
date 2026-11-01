#!/bin/bash
# Evaluation of MFD with C's task mix (docs/MFD_REPORT.md §13) on an H200, one run per job. Validation and test use
# the full-view history with the fusionseqrec prompt (mfd_eval.load_split, from the run's --tasks).
#   (i)   K_f=20,  w=1/1/1      mfd_eval.py --sanity        -> <run>/eval_kf20/   (modes, coverage, f accuracy, cost)
#   (ii)  K_f=100, w=1/0.5/0.5  mfd_eval.py                 -> <run>/eval_ii/     (the 4 modes at these weights)
#   (iii) Part B grid on VALIDATION -> chosen.json -> test  mfd_sweep.py        -> <run>/sweep/
# RUN=mfd122_mv: only (ii) (its (i) and (iii) exist). Frozen snapshot of macrec/*.py.
# Submit: sbatch --export=ALL,D=Instruments,RUN=mfd122_mtA sbatch/mfd/eval_mt.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_mt_eval
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
D=${D:-Instruments}; RUN=${RUN:-mfd122_mtA}; SEED=42
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
CK=$RUNS/mfd/$D/$RUN/seed$SEED
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
TASKS=$(python3 -c "import json; print(json.load(open('$CK/run_config.json'))['tasks'])")
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) $CK tasks=$TASKS commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
A='--ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --tasks $TASKS --prompt_num 4'
python -u mfd_eval.py \$A --k_f 100 --w_f 1.0 --w_t 0.5 --w_i 0.5 --out_dir $CK/eval_ii > $CK/eval_ii.log 2>&1
if [ $RUN != mfd122_mv ]; then
  python -u mfd_eval.py \$A --sanity --k_f 20 --out_dir $CK/eval_kf20 > $CK/eval_kf20.log 2>&1
  [ -f $CK/sweep/chosen.json ] || python -u mfd_sweep.py --mode sweep \$A --out_dir $CK/sweep > $CK/sweep_valid.log 2>&1
  python -u mfd_sweep.py --mode apply --split test --config $CK/sweep/chosen.json \$A --out_dir $CK/sweep > $CK/sweep_apply_test.log 2>&1
fi
"
echo done
