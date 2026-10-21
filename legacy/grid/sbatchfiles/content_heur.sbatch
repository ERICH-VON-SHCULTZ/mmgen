#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=content
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/content_heur.out
# CPU: recover per-event timestamps (history + target) for the forced-pool MFD dumps and the chain dumps.
module purge
singularity exec --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310
P=/scratch/yw8866/rec-tmall/force_exp/pools; C=/scratch/yw8866/rec-tmall/matched_rerank
cd /scratch/yw8866/rec-tmall && python ts_sidecar.py --dumps \$P/fh_test.pt && python content_heuristics.py"
