#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=biaseval
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/bias_eval.out
# Per-behavior score bias tuned on VAL: MFD forced-pool models (3 seeds) and chain (seed-0 model), same grid.
module purge
singularity exec --nv --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_CPP_MIN_LOG_LEVEL=2
cd /scratch/yw8866/rec-tmall
F=/scratch/yw8866/rec-tmall/force_exp/cache
python bias_eval.py --spec /scratch/yw8866/rec-tmall/force_exp/spec.json --arm MFD \
  --models \$F/MFD__wnew1__seed0.model.pt,\$F/MFD__wnew1__seed1.model.pt,\$F/MFD__wnew1__seed2.model.pt --out /scratch/yw8866/rec-tmall/force_exp/bias_mfd.txt
python bias_eval.py --spec /scratch/yw8866/rec-tmall/force_exp/spec.json --arm CH \
  --models /scratch/yw8866/rec-tmall/nip_mix/topk_ch_seed0.model.pt --out /scratch/yw8866/rec-tmall/force_exp/bias_ch.txt
"
