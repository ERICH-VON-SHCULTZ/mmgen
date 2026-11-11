#!/bin/bash
# §21.6 gain diagnostic (GPU, inference only; macrec/r21_gain_diag.py): per test user log P(true f), margin to the top f,
# rank split into other-bucket / own-bucket candidates above the target, plain K_f = 50.
# Env: D, RUNS="run:seed ..." (e.g. "mfd122_mv_geoP1a:42 r21_A01:42"). Out: mmgen_runs/geo/r21_gain/<D>/<run>_s<seed>.pt
# then tools/r21_gain_report.py (CPU) -> geo/r21_gain_<D>.json. Any GPU (no timing).
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r21_gain
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS_DIR=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS_DIR/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS_DIR/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
echo "job $SLURM_JOB_ID on $(hostname) D=$D RUNS=$RUNS commit $(cd $REPO && git rev-parse --short HEAD)"
for R in $RUNS; do
  IFS=: read RUN S <<< "$R"
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
  source /ext3/env.sh && conda activate py310 && cd $M && python -u r21_gain_diag.py --ckpt_path $RUNS_DIR/mfd/$D/$RUN/seed$S \
    --data_path $RUNS_DIR/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 \
    --tasks seqrec_mv --prompt_num 4 --out_file $RUNS_DIR/geo/r21_gain/$D/${RUN}_s$S.pt"
done
singularity exec --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
source /ext3/env.sh && conda activate py310 && cd $T && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u r21_gain_report.py $D"
echo done
