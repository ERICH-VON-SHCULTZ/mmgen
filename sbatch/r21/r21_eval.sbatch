#!/bin/bash
# Round-21 evaluation (docs/MFD_REPORT.md §21, macrec/r21_eval.py) on an H200: plain K_f = 50, 1c chosen on validation at
# K_f = 50, f diagnostics; valid + test. Env: D, RUN (r21_<VAR> or mfd122_mv_geoP1a = the reference row), SEED (42), LIMIT.
# Out: mmgen_runs/mfd/<D>/<RUN>/seed<S>/r21_eval/. Submit: sbatch --dependency=afterok:<pack> --export=ALL,D=Arts,RUN=r21_G0 sbatch/r21/r21_eval.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r21_eval
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
SEED=${SEED:-42}; LIMIT=${LIMIT:-0}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
CK=$RUNS/mfd/$D/$RUN/seed$SEED; OUT=$CK/r21_eval; [ $LIMIT != 0 ] && OUT=$RUNS/tmp_smoke/r21_eval_${D}_$RUN; mkdir -p $OUT
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) $CK commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u r21_eval.py --ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX \
  --max_his_len 20 --test_batch_size 64 --tasks seqrec_mv --prompt_num 4 --out_dir $OUT --limit $LIMIT > $OUT/eval.log 2>&1
tail -c 600 $OUT/eval.log
"
echo done
