#!/bin/bash
# §22.10 E10a (docs/MFD_REPORT.md): per-view f distributions (macrec/r22_e10_views.py) of every existing P1a / G0 / G0acc / A01
# checkpoint of one dataset (seeds 42/43/44 where present) -> geo/r22/<D>/views_<RUN>_s<S>.pt. Existing outputs are kept.
# Submit: sbatch --export=ALL,D=Arts sbatch/r22/r22_e10a_views.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_tandon,h100_tandon,a100_tandon
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --job-name=mmgen_r22_e10a
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
O=$RUNS/geo/r22/$D; mkdir -p $O
echo "job $SLURM_JOB_ID on $(hostname) D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
CMDS=""
for R in mfd122_mv_geoP1a r21_G0 r21_G0acc r21_A01 ${EXTRA_RUNS:-}; do for S in 42 43 44; do
  CK=$RUNS/mfd/$D/$R/seed$S; [ -f $CK/config.json ] || continue; [ -f $O/views_${R}_s$S.pt ] && continue
  CMDS="$CMDS echo $R s$S; python -u r22_e10_views.py --ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --tasks seqrec_mv --prompt_num 4 --out_file $O/views_${R}_s$S.pt;"
done; done
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "source /ext3/env.sh && conda activate py310 && cd $M && $CMDS true"
echo done
