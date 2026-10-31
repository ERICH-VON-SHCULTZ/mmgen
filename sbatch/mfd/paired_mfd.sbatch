#!/bin/bash
# MFD stage 1 vs chain decoder, per user: paired bootstrap, ranking given coverage on the same users, HR@10 by
# target-f rank (sbatch/tools/mfd_paired.py). Needs eval_mfd.sbatch outputs. Frozen snapshot of macrec/*.py.
# Submit: sbatch --export=ALL,D=Instruments,CFG=122,SEED=42 sbatch/mfd/paired_mfd.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_paired
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
D=${D:-Instruments}; CFG=${CFG:-122}; SEED=${SEED:-42}; K_F=${K_F:-20}
IDX=.index_md122_a05_ep500_reassign.json
OUT=$RUNS/mfd/$D/mfd${CFG}_stage1/seed$SEED
CHAIN=$RUNS/baselines/$D/md${CFG}_a05_ep500_reassign/seed$SEED
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp $REPO/sbatch/tools/mfd_paired.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) D=$D K_F=$K_F commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
MACREC=$M python -u mfd_paired.py --mfd_run $OUT --chain_run $CHAIN --data_path $RUNS/data \
  --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --k_f $K_F \
  > $OUT/eval_kf$K_F/paired.log 2>&1
"
echo done
