#!/bin/bash
# E-norm (fnormj) TIGER tokenizer retrained with MQL4GRec main_single.py (same command as tokenizers.sh)
# and indexed from the FINAL epoch (no selection): $DATA/$D/$D.index_fnormj_ep500_tiger.json.
# The rerun's best_collision index is also written (..._fnormj_rerun_tiger.json) to check that the
# rerun reproduces the original tokenizer. Submit: sbatch --export=ALL,D=Instruments sbatch/baselines/e_final_tok.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_e_final_tok
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
SNAP=$TMPDIR/cross_index; cp -r $REPO/macrec/cross_index $SNAP   # frozen copy of the code
DATA=/scratch/yw8866/mmgen_runs/data
O=/scratch/yw8866/mmgen_runs/baselines/$D/tok/tiger_fnormj_ep500; mkdir -p $O
echo "job $SLURM_JOB_ID on $(hostname) D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $SNAP
python -u main_single_final.py --num_emb_list 256 256 256 --sk_epsilons 0.0 0.0 0.0 --device cuda:0 \
  --data_path $DATA/$D/$D.emb-fused-normj.npy --ckpt_dir $O --eval_step 2 --batch_size 2048 --epochs 500 > $O/train.log 2>&1
python -u generate_indices_tiger.py --ckpt_path $O/final_model.pth --content text \
  --output_file $DATA/$D/$D.index_fnormj_ep500_tiger.json --stats_file $O/sid_stats.json
python -u generate_indices_tiger.py --ckpt_path $O/best_collision_model.pth --content text \
  --output_file $DATA/$D/$D.index_fnormj_rerun_tiger.json --stats_file $O/sid_stats_rerun_best.json
"
cmp -s $DATA/$D/$D.index_fnormj_rerun_tiger.json $DATA/$D/$D.index_fnormj_tiger.json \
  && echo "rerun best_collision index IDENTICAL to the original" || echo "rerun best_collision index DIFFERS from the original"
