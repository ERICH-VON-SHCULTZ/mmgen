#!/bin/bash
# STEP 1 (baselines): item features with the repo's own extractors.
#   text  = data_process/amazon_text_emb.py  (huggyllama/llama-7b, title+description, mean pooling)
#   image = data_process/clip_feature.py     (CLIP ViT-L/14 image branch, main product image)
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=mmgen_features
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
DATA=/scratch/yw8866/mmgen_runs/data
IMG=/scratch/yw8866/mmgen_runs/raw/amazon18/Images
PLM=/scratch/yw8866/hf_cache/macrec_plm          # llama-7b already downloaded here (smoke test)
SMOKE_EMB=/scratch/yw8866/mmgen_runs/smoke_tiger_Instruments/data/Instruments/Instruments.emb-llama-td.npy

singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -e; source /ext3/env.sh && conda activate py310
cd $REPO/macrec/data_process
nvidia-smi --query-gpu=name --format=csv,noheader
for D in Instruments Arts Games; do
  echo \"== \$D CLIP ViT-L/14\"; date
  REF=''; [ \$D = Instruments ] && REF='--ref_npy /scratch/yw8866/tmp/ref/MQL4GRec/data/Instruments/Instruments.emb-ViT-L-14.npy'
  # clip_feature.py logic; tolerates the few items whose image is no longer downloadable (see clip_feature_ext.py)
  [ -f $DATA/\$D/\$D.emb-ViT-L-14.npy ] || python clip_feature_ext.py --dataset \$D --image_root $IMG \
      --save_root $DATA --backbone ViT-L/14 --model_cache_dir /scratch/yw8866/hf_cache/clip \$REF
done
[ -f $DATA/Instruments/Instruments.emb-llama-td.npy ] || cp $SMOKE_EMB $DATA/Instruments/
for D in Arts Games; do
  echo \"== \$D LLaMA-7B\"; date
  [ -f $DATA/\$D/\$D.emb-llama-td.npy ] || python amazon_text_emb.py --dataset \$D --root $DATA \
      --model_cache_dir $PLM --gpu_id 0
done
date
python $REPO/sbatch/tools/check_features.py
"
