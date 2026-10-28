#!/bin/bash
# MD-RQ-VAE: distance-based reassignment (no extra token) + diagnostics on the reassigned SIDs.
# Submit: sbatch --export=ALL,D=Instruments,NAME=a05_122,TAG=md122_a05_ep500_reassign,CKPT=final sbatch/md/reassign.sbatch
# CKPT=final (official from 2026-10-27: last epoch, no selection) or best (lowest collision; superseded)
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon,a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_md_reassign
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
# frozen copy of the code this job runs (later edits cannot affect it)
SNAP=$TMPDIR/cross_index; cp -r $REPO/macrec/cross_index $SNAP
OUT=/scratch/yw8866/mmgen_runs/md/$D/tok/$NAME
CKPT=${CKPT:-final}
echo "job $SLURM_JOB_ID on $(hostname) D=$D NAME=$NAME TAG=$TAG CKPT=$CKPT commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $SNAP
python -u reassign_md.py --ckpt $OUT/$CKPT.pth --tag $TAG
python -u md_diagnostics.py --ckpt $OUT/$CKPT.pth --codes $OUT/codes_$TAG.npy --out_name diagnostics_$TAG.json
"
