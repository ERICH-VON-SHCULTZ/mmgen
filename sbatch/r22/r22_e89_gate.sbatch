#!/bin/bash
# Gate of the §22.8 E8 / §22.9 E9 runs (docs/MFD_REPORT.md): macrec/r22_e89_check.py (E8: step 0 = A01, W gradient, c(f) finite +
# bucket sizes, bake / save / reload; E9: initial loss vs A01, pi / W_k gradients, inference = training P(f), bake / save / reload;
# on A01 seed 42) -> geo/<D>/r22_e89_check.json (replaced only when every check passes); dry runs of VARS + tools/r22_config_diff.py
# (merged into geo/gate/<D>/r22_config_diff.json); then every VAR through the exact training command for 5 real optimizer steps
# (run_r22_train.sh SMOKE=1 -> geo/gate/r22_smoke/<D>/r22_<VAR>/smoke.json: losses, grad norms finite, new parameters moved).
# Submit: sbatch --export=ALL,D=Arts sbatch/r22/r22_e89_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r22_e89gate
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out
set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export WANDB_MODE=offline WANDB_DIR=$RUNS/wandb             # gate runs: offline
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/*.py $T/
D=${D:-Arts}; VARS=${VARS:-"E8a E8b E9k2 E9k4"}; SMOKE_VARS=${SMOKE_VARS:-$VARS}
RR=$TMPDIR/run_r22_train.sh; cp $REPO/sbatch/r22/run_r22_train.sh $RR
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D VARS=$VARS SMOKE_VARS=$SMOKE_VARS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u r22_e89_check.py --data_path $RUNS/data --dataset $D --a01_ckpt $RUNS/mfd/$D/r21_A01/seed42 \
  --tok_ckpt $RUNS/md/sids/v1/$D/1+2+2/tokenizer.pth \
  --out_file $RUNS/geo/$D/r22_e89_check.new.json > $RUNS/geo/$D/r22_e89_check.log 2>&1 || { tail -30 $RUNS/geo/$D/r22_e89_check.log; echo 'R22 E8/E9 CHECK FAILED'; exit 1; }
mv $RUNS/geo/$D/r22_e89_check.new.json $RUNS/geo/$D/r22_e89_check.json
grep -E 'PASS|FAIL' $RUNS/geo/$D/r22_e89_check.log
for V in $VARS; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RR; done
cd $T && python r22_config_diff.py $D $VARS && cd $M
for V in $SMOKE_VARS; do
  M=$M D=$D VAR=\$V SEED=42 SMOKE=1 bash $RR
  python -c \"import json; r = json.load(open('$RUNS/geo/gate/r22_smoke/$D/r22_\$V/smoke.json')); print('smoke r22_\$V', 'OK' if r['ok'] else 'NOT OK', r['steps'], [round(x, 4) for x in r['losses']], r['e8_weight_norm'], r['e9'])\"
done
"
echo done
