#!/bin/bash
# Round-22 inference jobs (GPU; H200 for anything timed). Env: D, RUN (run dir name), SEED (42), STEP:
#   e3      macrec/r22_e3_smooth.py   codebook-smoothed P(f): (tau, lambda) grid on valid, test once timed -> <run>/seed<S>/r22_e3/
#   e2diag  macrec/r22_e2_diag.py     t1 / i1 ranks given the target f (test) -> geo/r22/<D>/e2diag_<RUN>_s<S>.pt
#   e2eval  macrec/r22_e2_eval.py     A01 + aux heads: w grid on valid, test once timed -> <run>/seed<S>/r22_e2eval/
#   e5      macrec/r22_e5_eval.py     region-mass score (needs geo/r22/<D>/e5_regions.pt): beta x K_f on valid, test once -> <run>/seed<S>/r22_e5/
#   fdump   macrec/r22_e4_first.py    A01-style full-view log P(f) per TEST user (renormalised over f) -> geo/r22/<D>/fdump_<RUN>_s<S>.pt (E6, E7)
#   eval    macrec/r21_eval.py        plain K_f = 50 + 1c + f diagnostics, on the run's own SID set (SID=1+2+2|1+3+3) -> <run>/seed<S>/r21_eval/
# Submit: sbatch --export=ALL,D=Arts,RUN=r21_A01,SEED=42,STEP=e3 sbatch/r22/r22_run.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_tandon
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --job-name=mmgen_r22_run
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
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
SEED=${SEED:-42}; SID=${SID:-1+2+2}
[ $SID = 1+3+3 ] && IDX=.index_md133_a06_ep500_reassign.json || IDX=.index_md122_a05_ep500_reassign.json
V1=$RUNS/md/sids/v1/$D/$SID
[ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
CK=$RUNS/mfd/$D/$RUN/seed$SEED
A="--ckpt_path $CK --data_path $RUNS/data --dataset $D --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --tasks seqrec_mv --prompt_num 4"
echo "job $SLURM_JOB_ID on $(hostname) STEP=$STEP $CK SID=$SID commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
case $STEP in
  e3)     OUT=$CK/r22_e3; mkdir -p $OUT; CMD="python -u r22_e3_smooth.py $A --tokenizer_ckpt $V1/tokenizer.pth --out_dir $OUT > $OUT/e3.log 2>&1; tail -c 800 $OUT/e3.log" ;;
  e2diag) mkdir -p $RUNS/geo/r22/$D; CMD="python -u r22_e2_diag.py $A --out_file $RUNS/geo/r22/$D/e2diag_${RUN}_s$SEED.pt" ;;
  e2eval) OUT=$CK/r22_e2eval; mkdir -p $OUT; CMD="python -u r22_e2_eval.py $A --out_dir $OUT > $OUT/e2eval.log 2>&1; tail -c 800 $OUT/e2eval.log" ;;
  e5)     OUT=$CK/r22_e5; mkdir -p $OUT; CMD="python -u r22_e5_eval.py $A --regions $RUNS/geo/r22/$D/e5_regions.pt --out_dir $OUT > $OUT/e5.log 2>&1; tail -c 800 $OUT/e5.log" ;;
  fdump)  mkdir -p $RUNS/geo/r22/$D; CMD="python -u r22_e4_first.py $A --kind mfd --out_file $RUNS/geo/r22/$D/fdump_${RUN}_s$SEED.pt" ;;
  eval)   OUT=$CK/r21_eval; mkdir -p $OUT; CMD="python -u r21_eval.py $A --out_dir $OUT > $OUT/eval.log 2>&1; tail -c 600 $OUT/eval.log" ;;
  *) echo "STEP?"; exit 1 ;;
esac
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "source /ext3/env.sh && conda activate py310 && cd $M && $CMD"
echo done
