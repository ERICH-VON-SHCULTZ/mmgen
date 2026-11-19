#!/bin/bash
# Gate of the §22.10 E10b / E10c runs (docs/MFD_REPORT.md), one dataset. If VARS has E10c: macrec/r22_e10c_check.py on P1a seed 42
# (data layout = P1a, beta 0 = P1a loss + gradients, no gradient through the extra forwards, KL = independent recomputation, Geo path +
# bake / reload, cost) -> geo/<D>/r22_e10c_check.json (replaced only when every check passes). Then dry runs of VARS +
# tools/r22_config_diff.py (E10c vs P1aref, E10b vs A01) and every VAR through the exact training command for 5 real optimizer steps
# (run_r22_train.sh SMOKE=1 -> geo/gate/r22_smoke/<D>/r22_<VAR>/smoke.json); the smoke's r21_beta / batch / accumulation must be
# the VAR's, its clipping is printed (P1a's regime: every step clipped, pre-clip median ~2.7 over a full run).
# Submit: sbatch --export=ALL,D=Instruments,VARS="E10c E10b005 E10b02 E10b05" sbatch/r22/r22_e10_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_r22_e10gate
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
D=${D:-Arts}; VARS=${VARS:-E10c}
RR=$TMPDIR/run_r22_train.sh; cp $REPO/sbatch/r22/run_r22_train.sh $RR
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D VARS=$VARS commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
if [[ ' $VARS ' == *' E10c '* ]]; then
  python -u r22_e10c_check.py --data_path $RUNS/data --dataset $D --p1a_ckpt $RUNS/mfd/$D/mfd122_mv_geoP1a/seed42 \
    --tok_ckpt $RUNS/md/sids/v1/$D/1+2+2/tokenizer.pth \
    --out_file $RUNS/geo/$D/r22_e10c_check.new.json > $RUNS/geo/$D/r22_e10c_check.log 2>&1 || { tail -30 $RUNS/geo/$D/r22_e10c_check.log; echo 'R22 E10c CHECK FAILED'; exit 1; }
  mv $RUNS/geo/$D/r22_e10c_check.new.json $RUNS/geo/$D/r22_e10c_check.json
  grep -E 'PASS|FAIL' $RUNS/geo/$D/r22_e10c_check.log
fi
for V in $VARS; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RR; done
cd $T && python r22_config_diff.py $D $VARS && cd $M
for V in $VARS; do
  M=$M D=$D VAR=\$V SEED=42 SMOKE=1 bash $RR
  python - <<PY
import json, sys
V = '\$V'
r = json.load(open('$RUNS/geo/gate/r22_smoke/$D/r22_' + V + '/smoke.json'))
want = {'E10c': (0.1, 256, 4, 'E10c'), 'E10b005': (0.05, 341, 1, 'A'), 'E10b02': (0.2, 341, 1, 'A'), 'E10b05': (0.5, 341, 1, 'A')}[V]
got = (r['r21_beta'], r['per_device_train_batch_size'], r['gradient_accumulation_steps'], r['r21_mode'])
n = r['grad_norms']
print('smoke r22_' + V, 'OK' if r['ok'] else 'NOT OK', r['steps'], [round(x, 4) for x in r['losses']], 'beta/bs/accum/mode', got,
      'pre-clip norms', [round(x, 3) for x in n], 'clipped', sum(x > 1.0 for x in n), '/', len(n), 'e10c', r.get('e10c'))
if not r['ok'] or tuple(got) != want:
    print('SMOKE NOT OK', V, got, want); sys.exit(1)
PY
done
"
echo done
