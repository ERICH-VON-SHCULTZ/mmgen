#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=revisit
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/revisit.out
# Revisit model (which past item comes back), trained on the full training split; go/no-go against the
# reranker's item-level ranking of repeats (BOTH 0.983, forced-only 0.587).
module purge
singularity exec --nv --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310; export TF_CPP_MIN_LOG_LEVEL=2
cd /scratch/yw8866/rec-tmall && python revisit.py --max_users 400000 --max_train 4000000 --epochs 3 \
  --out /scratch/yw8866/rec-tmall/force_exp/revisit.txt \
  --test_dump /scratch/yw8866/rec-tmall/force_exp/pools/fh_test.pt \
  --mfd_res /scratch/yw8866/rec-tmall/force_exp/cache/MFD__wnew1__seed0.pt \
  --ch_res /scratch/yw8866/rec-tmall/nip_exp/cache/CH__wnew1__seed0.pt"
