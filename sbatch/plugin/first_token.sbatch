#!/bin/bash
# First-token accuracy (macrec/first_token_acc.py) for C (mql) and plug-in runs, valid + test. Not a timing job.
# Frozen snapshot of macrec/. Submit: sbatch --export=ALL,RUNS="Instruments:mql Arts:mql" sbatch/plugin/first_token.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_plugin_ftok
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RD=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) RUNS=$RUNS commit $(cd $REPO && git rev-parse --short HEAD)"
for R in $RUNS; do
  D=${R%%:*}; B=${R##*:}
  case $B in mql) IDX=.index_lemb_mql.json; IIDX=.index_vitemb_mql.json ;;
             plug*) IDX=.index_${B}_text.json; IIDX=.index_${B}_image.json ;; *) echo "unknown $B"; exit 1 ;; esac
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $M && python -u first_token_acc.py --ckpt_path $RD/baselines/$D/$B/seed42 \
  --data_path $RD/data/ --dataset $D --index_file $IDX --image_index_file $IIDX --max_his_len 20 --test_batch_size 256"
done
echo done
