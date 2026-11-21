#!/bin/bash
# Round 23 (docs/MFD_REPORT.md §23): inference-speed measurement on one H200, nothing else on the GPU (exclusive gres).
#   STEP=prep     macrec/r23_prep.py      (batch-1 subset, C-fast == C on the full test set, reimplementation / instrumentation
#                                          checks, P1a reference ranks) -> mmgen_runs/timing/r23/<D>/prep.json
#   STEP=time     macrec/r23_timing.py --mode time    METHODS=.. BATCHES=.. TAG=.. [REPEATS=3]  -> timing/r23/<D>/time_<TAG>.json (+ anchor)
#   STEP=profile  macrec/r23_timing.py --mode profile METHODS=.. STAGEB=64,1        -> timing/r23/<D>/profile_<TAG>.json
# CODE = frozen code snapshot made by sbatch/r23/r23_submit.sh (every job of the round runs the same code).
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=96GB
#SBATCH --time=01:58:00
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=mmgen_r23
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
: "${D:?}" "${STEP:?}" "${CODE:?}"
IDX=.index_md122_a05_ep500_reassign.json; V1=$RUNS/md/sids/v1/$D/1+2+2
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
OUT=$RUNS/timing/r23/$D; mkdir -p $OUT
M=$TMPDIR/macrec; mkdir -p $M; cp $CODE/*.py $M/; cp -r $CODE/config $M/
echo "job $SLURM_JOB_ID on $(hostname) D=$D STEP=$STEP METHODS=${METHODS:-} BATCHES=${BATCHES:-} TAG=${TAG:-} code $(cat $CODE/COMMIT)"
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
case $STEP in
  prep)    CMD="python -u r23_prep.py --dataset $D" ;;
  time)    CMD="python -u r23_timing.py --mode time --dataset $D --methods $METHODS --batches ${BATCHES:-64,256} --repeats ${REPEATS:-3} \
                --warmup 200 --anchor --out_file $OUT/time_${TAG}.json" ;;
  profile) CMD="python -u r23_timing.py --mode profile --dataset $D --methods $METHODS --stage_batches ${STAGEB:-} \
                --out_file $OUT/profile_${TAG}.json" ;;
  smoke)   # every code path on a few users, outputs under tmp_smoke (never the real r23 folder)
           export R23_OUT=$RUNS/tmp_smoke/r23; OUT=$R23_OUT/$D; mkdir -p $OUT; ALL=P1a:20,P1a:50,A01:50,C,Cfast,D,TIGER,chain
           CMD="python -u r23_prep.py --dataset $D --limit 256 && \
                python -u r23_timing.py --mode time --dataset $D --methods $ALL --batches 1,64,256 --repeats 2 --warmup 200 --anchor \
                  --limit 512 --out_file $OUT/time_smoke.json && \
                python -u r23_timing.py --mode profile --dataset $D --methods $ALL --stage_batches 64,1 --limit 64 \
                  --gpu_prof_users 128 --out_file $OUT/profile_smoke.json" ;;
  smokeprof) # profile path only (counts, hook-split FLOPs, stages, profiler) on a few users, under tmp_smoke
           export R23_OUT=$RUNS/tmp_smoke/r23; OUT=$R23_OUT/$D; mkdir -p $OUT
           CMD="python -u r23_timing.py --mode profile --dataset $D --methods ${METHODS:-P1a:50,C,chain} --stage_batches 64,1 --limit 64 \
                  --gpu_prof_users 128 --out_file $OUT/profile_smokeprof.json" ;;
esac
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
$CMD
"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo done
