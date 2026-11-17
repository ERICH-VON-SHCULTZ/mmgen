#!/bin/bash
# Round-22 E5 (docs/MFD_REPORT.md §22.5): coarse item partitions per dataset (macrec/r22_e5_regions.py; CPU, no training)
# -> geo/r22/<D>/e5_regions.{pt,json}. Submit: sbatch --export=ALL,DS="Arts Instruments" sbatch/r22/r22_e5_regions.sbatch
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=cs,cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --job-name=mmgen_r22_e5reg
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
M=$TMPDIR/macrec; mkdir -p $M; cp -r $REPO/macrec/*.py $REPO/macrec/cross_index $M/
DS=${DS:-"Arts Instruments"}; IDX=.index_md122_a05_ep500_reassign.json
echo "job $SLURM_JOB_ID DS=$DS commit $(cd $REPO && git rev-parse --short HEAD)"
for D in $DS; do
  V1=$RUNS/md/sids/v1/$D/1+2+2
  [ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
  singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $M && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 \
python -u r22_e5_regions.py --data_path $RUNS/data --dataset $D --tokenizer_ckpt $V1/tokenizer.pth --out_dir $RUNS/geo/r22/$D"
done
echo done
