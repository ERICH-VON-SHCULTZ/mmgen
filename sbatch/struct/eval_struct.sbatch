#!/bin/bash
# Evaluation of an encoder-structure run or the chain control (docs/MFD_REPORT.md §16) on an H200, one run per job:
#   valid + test trie beam search exactly as the chain control (20 beams, eval_split / eval_struct), then the
#   history-truncation diagnostic on VALIDATION: the same model with the history cut at inference to the last 1 and 5
#   items (20 = the normal valid result) -> <run>/eval/valid_his{1,5}/; then geo_diag (f accuracy, near-miss, HR@10 by f rank).
#   RUN=chain122_E1|chain122_E2 (mmgen_runs/struct/<D>/<RUN>/seed42) or RUN=control (baselines/<D>/md122_a05_ep500_reassign;
#   its valid/test evals exist, only the truncation runs).
# Submit: sbatch --export=ALL,D=Instruments,RUN=chain122_E1 sbatch/struct/eval_struct.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_struct_eval
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
D=${D:-Instruments}; RUN=${RUN:-chain122_E1}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
if [ $RUN = control ]; then CK=$RUNS/baselines/$D/md122_a05_ep500_reassign/seed42; ENTRY=eval_split.py
else CK=$RUNS/struct/$D/$RUN/seed42; ENTRY=eval_struct.py; fi
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) $CK entry=$ENTRY commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
ev() {  # split, max_his_len, out dir
  mkdir -p \$3
  [ -f \$3/results_seqrec_20.json ] || torchrun --nproc_per_node=1 --master_port=\$((20000 + RANDOM % 20000)) $ENTRY \
    --split \$1 --ckpt_path $CK --data_path $RUNS/data/ --dataset $D --test_batch_size 64 --num_beams 20 --index_file $IDX \
    --image_index_file $IDX --test_task seqrec --max_his_len \$2 --results_file \$3/results_seqrec_20.json \
    --save_file \$3/save_seqrec_20.json --stats_file \$3/cost_seqrec.json --filter_items > \$3/eval_seqrec.log 2>&1
}
ev valid 20 $CK/eval/valid
ev test 20 $CK/eval/test
ev valid 1 $CK/eval/valid_his1
ev valid 5 $CK/eval/valid_his5
python -u geo_diag.py --kind chain --ckpt_path $CK --tokenizer_ckpt $V1/tokenizer.pth --data_path $RUNS/data --dataset $D \
  --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 > $CK/geo_diag.log 2>&1
tail -1 $CK/geo_diag.log
"
echo done
