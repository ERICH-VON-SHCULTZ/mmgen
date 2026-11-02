#!/bin/bash
# Evaluation + diagnostics of a codebook-geometry run or its control (docs/MFD_REPORT.md §14) on an H200, one run per job.
#   KIND=chain RUN=chain122_<VAR>        valid/test trie beam search exactly as the chain control (run_baseline.sh eval:
#                                        eval_split.py, 20 beams) -> mmgen_runs/geo/<D>/<RUN>/seed42/eval/, then geo_diag
#   KIND=mfd   RUN=mfd122_mv_geo<VAR>    settings (i) K_f=20 w=1, (ii) K_f=100 w=1/.5/.5, (iii) grid chosen on VALID
#                                        then test (as sbatch/mfd/eval_mt.sbatch), then geo_diag
#   DIAG_ONLY=1 CK=<run dir>             only geo_diag (controls: baselines/<D>/md122_a05_ep500_reassign/seed42, mfd122_mv)
# Frozen snapshot of macrec/*.py. Submit: sbatch --export=ALL,KIND=chain,RUN=chain122_A sbatch/geo/eval_geo.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_geo_eval
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
D=${D:-Instruments}; KIND=${KIND:-chain}; RUN=${RUN:-chain122_A}; SEED=42; DIAG_ONLY=${DIAG_ONLY:-0}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
if [ -z "${CK:-}" ]; then [ $KIND = chain ] && CK=$RUNS/geo/$D/$RUN/seed$SEED || CK=$RUNS/mfd/$D/$RUN/seed$SEED; fi
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) KIND=$KIND $CK diag_only=$DIAG_ONLY commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
if [ $DIAG_ONLY != 1 ] && [ $KIND = chain ]; then
  for SPLIT in valid test; do
    mkdir -p $CK/eval/\$SPLIT
    [ -f $CK/eval/\$SPLIT/results_seqrec_20.json ] || torchrun --nproc_per_node=1 --master_port=\$((20000 + RANDOM % 20000)) eval_split.py \
      --split \$SPLIT --ckpt_path $CK --data_path $RUNS/data/ --dataset $D --test_batch_size 64 --num_beams 20 --index_file $IDX \
      --image_index_file $IDX --test_task seqrec --results_file $CK/eval/\$SPLIT/results_seqrec_20.json \
      --save_file $CK/eval/\$SPLIT/save_seqrec_20.json --stats_file $CK/eval/\$SPLIT/cost_seqrec.json --filter_items \
      > $CK/eval/\$SPLIT/eval_seqrec.log 2>&1
  done
fi
if [ $DIAG_ONLY != 1 ] && [ $KIND = mfd ]; then
  A='--ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --tasks seqrec_mv --prompt_num 4'
  python -u mfd_eval.py \$A --k_f 100 --w_f 1.0 --w_t 0.5 --w_i 0.5 --out_dir $CK/eval_ii > $CK/eval_ii.log 2>&1
  python -u mfd_eval.py \$A --sanity --k_f 20 --out_dir $CK/eval_kf20 > $CK/eval_kf20.log 2>&1
  [ -f $CK/sweep/chosen.json ] || python -u mfd_sweep.py --mode sweep \$A --out_dir $CK/sweep > $CK/sweep_valid.log 2>&1
  python -u mfd_sweep.py --mode apply --split test --config $CK/sweep/chosen.json \$A --out_dir $CK/sweep > $CK/sweep_apply_test.log 2>&1
fi
python -u geo_diag.py --kind $KIND --ckpt_path $CK --tokenizer_ckpt $V1/tokenizer.pth --data_path $RUNS/data --dataset $D \
  --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 > $CK/geo_diag.log 2>&1
tail -1 $CK/geo_diag.log
"
echo done
