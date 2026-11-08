#!/bin/bash
# §20 Task 2 (docs/MFD_REPORT.md): fused P1a + dense-head evaluation (macrec/dense_eval.py), inference only.
#   STAGE=sweep  VALIDATION grid w_d x f-selection rule A/B, dense-only retrieval (any GPU)
#   LIMIT=N      smoke test on the first N users into mmgen_runs/tmp_smoke/ (no reported output touched)
#   STAGE=apply  TEST once for the chosen w_d of each rule, timed (submit on h200_tandon)
# Env: D, RUN (default mfd122_mv_geoP2a), SEED (42). Out: mmgen_runs/mfd/<D>/<RUN>/seed<S>/dense_eval/
# Frozen snapshot of macrec/*.py. Submit: sbatch --export=ALL,D=Arts,STAGE=sweep sbatch/mfd/dense_eval.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_dense_eval
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
RUN=${RUN:-mfd122_mv_geoP2a}; SEED=${SEED:-42}; STAGE=${STAGE:-sweep}; LIMIT=${LIMIT:-0}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
CK=$RUNS/mfd/$D/$RUN/seed$SEED; OUT=$CK/dense_eval; [ $LIMIT != 0 ] && OUT=$RUNS/tmp_smoke/dense_eval_$D; mkdir -p $OUT
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) STAGE=$STAGE $CK commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u dense_eval.py --mode $STAGE --ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX \
  --max_his_len 20 --test_batch_size 64 --out_dir $OUT --limit $LIMIT > $OUT/$STAGE.log 2>&1
tail -3 $OUT/$STAGE.log
"
echo done
