#!/bin/bash
# Exploration (docs/EXPLORATION.md), inference only, no training. One job per STEP:
#   STEP=first D=<D>   per-user first-decoding-step rank of the target's first token (macrec/expl_first_step.py, test):
#                      C seed 42 (mql: seqrec + seqimage), plug-in 1+2+2 (plug122), MD chain (f), mfd122_mv (f)
#                      -> mmgen_runs/expl/<D>/first_<run>.pt
#   STEP=infer D=<D>   Part 3 re-scoring checks on mfd122_mv, VALIDATION only (macrec/expl_infer.py)
#                      -> mmgen_runs/expl/<D>/infer_valid.json
# Frozen snapshot of macrec/*.py. Submit: sbatch --export=ALL,STEP=first,D=Instruments sbatch/expl/expl.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_expl
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
D=${D:-Instruments}; STEP=${STEP:-first}
MD=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$MD | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
O=$RUNS/expl/$D; mkdir -p $O
echo "job $SLURM_JOB_ID on $(hostname) STEP=$STEP D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
C='--data_path $RUNS/data --dataset $D --max_his_len 20 --test_batch_size 64'
if [ $STEP = first ]; then
  python -u expl_first_step.py \$C --ckpt_path $RUNS/baselines/$D/mql/seed42 --index_file .index_lemb_mql.json --image_index_file .index_vitemb_mql.json --out_file $O/first_C.pt
  python -u expl_first_step.py \$C --ckpt_path $RUNS/baselines/$D/plug122/seed42 --index_file .index_plug122_text.json --image_index_file .index_plug122_image.json --out_file $O/first_plug122.pt
  python -u expl_first_step.py \$C --ckpt_path $RUNS/baselines/$D/md122_a05_ep500_reassign/seed42 --index_file $MD --image_index_file $MD --first_tasks seqrec --out_file $O/first_chain.pt
  python -u expl_first_step.py \$C --ckpt_path $RUNS/mfd/$D/mfd122_mv/seed42 --index_file $MD --image_index_file $MD --first_tasks seqrec --out_file $O/first_mv.pt
else
  python -u expl_infer.py \$C --ckpt_path $RUNS/mfd/$D/mfd122_mv/seed42 --index_file $MD --image_index_file $MD --out_file $O/infer_valid.json
fi
"
echo done
