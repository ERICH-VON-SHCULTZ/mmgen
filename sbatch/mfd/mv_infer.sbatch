#!/bin/bash
# §20 Task 1 (docs/MFD_REPORT.md): test-time multi-view ensemble of an MFD run (macrec/mfd_mv_infer.py), inference only.
#   STAGE=sweep  VALIDATION grid of variants 1a / 1b / 1c, chosen config per variant (any GPU)
#   LIMIT=N      smoke test on the first N users into mmgen_runs/tmp_smoke/ (no reported output touched)
#   STAGE=apply  TEST once for the chosen configs, timed with views in separate calls and stacked (submit on h200_tandon)
# Env: D, RUN (default mfd122_mv_geoP1a), SEED (42). Out: mmgen_runs/mfd/<D>/<RUN>/seed<S>/mv_infer/
# Frozen snapshot of macrec/*.py. Submit: sbatch --export=ALL,D=Arts,STAGE=sweep sbatch/mfd/mv_infer.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mv_infer
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
RUN=${RUN:-mfd122_mv_geoP1a}; SEED=${SEED:-42}; STAGE=${STAGE:-sweep}; LIMIT=${LIMIT:-0}
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
CK=$RUNS/mfd/$D/$RUN/seed$SEED; OUT=$CK/mv_infer; [ $LIMIT != 0 ] && OUT=$RUNS/tmp_smoke/mv_infer_$D; mkdir -p $OUT
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
echo "job $SLURM_JOB_ID on $(hostname) STAGE=$STAGE $CK commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u mfd_mv_infer.py --mode $STAGE --ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX \
  --max_his_len 20 --test_batch_size 64 --out_dir $OUT --limit $LIMIT --check_ranks $CK/eval_ii/ranked_valid.pt > $OUT/$STAGE.log 2>&1
tail -3 $OUT/$STAGE.log
"
echo done
