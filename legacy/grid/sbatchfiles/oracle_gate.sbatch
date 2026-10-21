#!/bin/bash
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=ogate
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/oracle_gate.out
# CPU-only: ceiling of a per-event repeat/new mixture for both arms (rec-tmall/oracle_gate.py).
module purge
singularity exec --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh; conda activate py310; export OMP_NUM_THREADS=16
cd /scratch/yw8866/rec-tmall && python oracle_gate.py"
