#!/bin/bash
# Gate of the round-24 runs (docs/MFD_REPORT.md §24), one dataset: macrec/r24_check.py on P1a seed 42 (Track B item examples:
# P1a's structure unchanged, item input without E(f), full-SID targets, weighted loss, example counts) -> geo/<D>/r24_check.json
# (only when every check passes); dry runs of P1aref + VARS and tools/r24_config_diff.py (vs r24 P1aref; r24 P1aref vs r21 P1aref);
# then every VAR through the exact training command for 5 real optimizer steps (run_r24_train.sh SMOKE=1 -> smoke.json).
# Submit: sbatch --export=ALL,D=Arts,VARS="lr3e4 lr1e3 do02 do03 wd01 gn5 B1 B2" sbatch/r24/r24_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r24_gate
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
D=${D:-Arts}; VARS=${VARS:?}
RR=$TMPDIR/run_r24_train.sh; cp $REPO/sbatch/r24/run_r24_train.sh $RR
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D VARS=$VARS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u r24_check.py --dataset $D --out_file $RUNS/geo/$D/r24_check.json > $RUNS/geo/$D/r24_check.log 2>&1 \
  || { tail -40 $RUNS/geo/$D/r24_check.log; echo 'R24 CHECK FAILED'; exit 1; }
grep -E '\"ok\"|all_pass' $RUNS/geo/$D/r24_check.log
for V in P1aref $VARS; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RR; done
cd $T
python r24_config_diff.py $D $VARS || { echo 'R24 CONFIG DIFF NOT OK'; exit 1; }
cd $M
for V in $VARS; do
  M=$M D=$D VAR=\$V SEED=42 SMOKE=1 bash $RR
  python -c \"import json; r = json.load(open('$RUNS/geo/gate/r24_smoke/$D/r24_\$V/smoke.json')); print('smoke r24_\$V', 'OK' if r['ok'] else 'NOT OK', r['steps'], [round(x, 4) for x in r['losses']], [round(x, 2) for x in r['grad_norms']], r['info']['t5'], r['max_grad_norm'], r['learning_rate'], r['weight_decay'])\"
  # §24.6: the model the smoke run actually built (codebook init installed unless noinit; item-level encoder unless v2)
  python -c \"import json, sys; i = json.load(open('$RUNS/geo/gate/r24_smoke/$D/r24_\$V/smoke.json'))['info']; t = '\$V'.split('_'); \
exp = dict(geo='none' if 'noinit' in t else 'codebook', geo_installed='noinit' not in t, item_enc='v2' not in t); \
got = {k: i.get(k) for k in exp}; print('model r24_\$V', 'OK' if got == exp else 'NOT OK', got); sys.exit(0 if got == exp else 1)\"
done
"
echo done
