#!/bin/bash
# MFD wider search (Part B), inference only, on an H200: (1) --mode sweep on VALIDATION (K_f x weight grid,
# real-path timing per K_f, ONE setting chosen by valid HR@10 / NDCG@10 -> chosen.json); (2) --mode apply on
# TEST with that single setting (+ the stage-1 reference K_f=20, w=1). Frozen snapshot of macrec/*.py.
# Submit: sbatch --export=ALL,D=Instruments,RUN=mfd122_stage1 sbatch/mfd/sweep_mfd.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_sweep
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
D=${D:-Instruments}; RUN=${RUN:-mfd122_stage1}; SEED=${SEED:-42}
case $RUN in mfd122*) IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2 ;;
             mfd133*) IDX=.index_md133_a06_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+3+3 ;; esac
CK=$RUNS/mfd/$D/$RUN/seed$SEED; OUT=$CK/sweep
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) $CK commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
A='--ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --out_dir $OUT'
[ -f $OUT/chosen.json ] || python -u mfd_sweep.py --mode sweep \$A > $CK/sweep_valid.log 2>&1
python -u mfd_sweep.py --mode apply --split test --config $OUT/chosen.json \$A > $CK/sweep_apply_test.log 2>&1
"
echo done
