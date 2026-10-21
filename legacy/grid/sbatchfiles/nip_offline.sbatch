#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=03:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=nip_off
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/nip_offline.out

# CPU-only estimate of NIP's reserved-slot policy (see rec-tmall/nip_offline.py), from the cached
# 5-seed reranker results + the dumps' base scores and history masks. Runs without waiting for the GPU
# quota, and gives the go/no-go on "reserve slots for the generator's new-item picks" before the
# GPU job (18512956) finishes. Both arms, identical computation.

module purge
singularity exec \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f
export OMP_NUM_THREADS=16
python /scratch/yw8866/rec-tmall/nip_offline.py \
    --mfd_dump /scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt \
    --ch_dump /scratch/yw8866/rec-tmall/matched_rerank/chain808_test_dump.pt \
    --cache /scratch/yw8866/rec-tmall/seedvar/cache --seeds 0,1,2,3,4 --mfd_kpool 2000 --chunk 400
"
