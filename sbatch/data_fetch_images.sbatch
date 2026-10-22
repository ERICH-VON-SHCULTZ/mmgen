#!/bin/bash
# STEP 1 (baselines): raw Amazon-2018 meta + main product image per item, for Instruments/Arts/Games.
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=mmgen_fetch_images
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
RAW=/scratch/yw8866/mmgen_runs/raw/amazon18
DATA=/scratch/yw8866/mmgen_runs/data
mkdir -p $RAW/Metadata $RAW/Images $DATA
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"

BASE=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2
for F in Musical_Instruments Arts_Crafts_and_Sewing Video_Games; do
  [ -s $RAW/Metadata/meta_$F.json.gz ] || curl -sSfL -o $RAW/Metadata/meta_$F.json.gz $BASE/meta_$F.json.gz
done
ls -la $RAW/Metadata; md5sum $RAW/Metadata/*.gz

for D in Instruments Arts Games; do
  [ -d $DATA/$D ] || cp -r $REPO/macrec/data/$D $DATA/$D
done

singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -e; source /ext3/env.sh && conda activate py310
cd $REPO/macrec/data_process
for D in Instruments Arts Games; do
  date; python fetch_item_images.py --dataset \$D --data_root $DATA --meta_data_path $RAW/Metadata \
      --image_root $RAW/Images --workers 16
done
date"
du -sh $RAW/Images/*
