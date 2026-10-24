#!/bin/bash
# MACRec k-means pseudo-labels (K=512 per modality) on CPU. sklearn KMeans leaves a GPU idle, which the
# cluster cancels after 2 h (job 20806460), so this runs on the CPU account before tokenizers.sbatch.
# Submit: sbatch --export=ALL,D="Arts Games" sbatch/baselines/kmeans.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=mmgen_kmeans
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
DATA=/scratch/yw8866/mmgen_runs/data
echo "job $SLURM_JOB_ID on $(hostname)  D=$D  commit $(cd $REPO && git rev-parse --short HEAD)"
START=$(date +%s)
for DD in $D; do   # one dataset at a time, all cores each
  [ -f $DATA/$DD/$DD.index_vitemb_kmeans512.json ] && continue
  singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c \
    "source /ext3/env.sh && conda activate py310 && OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python $REPO/macrec/data/kmeans_dataset.py --dataset $DD --data_root $DATA"
  echo "$DD done after $(( $(date +%s) - START )) s"
done
