#!/bin/bash
# Gate of the codebook-geometry runs (docs/MFD_REPORT.md §14): sanity checks 1-2 (macrec/geo_check.py) and the dry-run
# config diff against the controls (sanity 3, tools/geo_config_diff.py). No training. Frozen copy of the code.
#   KINDS="chain mfd" VARS="A C AB1 AB2" (defaults): dry runs of every <kind>:<var>.
# Outputs: mmgen_runs/geo/<D>/geo_check.json, mmgen_runs/geo/gate/<D>/geo_config_diff.json, dry runs in geo/gate/dryrun/.
# Submit: sbatch sbatch/geo/geo_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_geo_gate
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export WANDB_MODE=offline WANDB_DIR=$RUNS/wandb             # dry runs never train
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"; export TRITON_CACHE_DIR=$TMPDIR/triton
D=${D:-Instruments}; KINDS=${KINDS:-"chain mfd"}; VARS=${VARS:-"A C AB1 AB2"}
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/; cp -r $REPO/macrec/cross_index $M/
RG=$TMPDIR/run_geo_train.sh; cp $REPO/sbatch/geo/run_geo_train.sh $RG
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/geo_config_diff.py $REPO/sbatch/tools/plugin_config_diff.py $T/
mkdir -p $RUNS/geo/$D $RUNS/geo/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D KINDS=$KINDS VARS=$VARS commit $(cd $REPO && git rev-parse --short HEAD)"
NAMES=""
for K in $KINDS; do for V in $VARS; do
  [ $K = chain ] && NAMES="$NAMES chain122_$V" || NAMES="$NAMES mfd122_mv_geo$V"
done; done
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u geo_check.py --index $RUNS/data/$D/$D.index_md122_a05_ep500_reassign.json --v1_dir $RUNS/md/sids/v1/$D/1+2+2 \
  --out_file $RUNS/geo/$D/geo_check.json > $RUNS/geo/$D/geo_check.log 2>&1 || { tail -5 $RUNS/geo/$D/geo_check.log; echo 'GEO CHECK FAILED'; exit 1; }
tail -1 $RUNS/geo/$D/geo_check.log
[ -f $RUNS/plugin/dryrun/$D/md122_a05_ep500_reassign/seed42/dryrun.json ] || MACREC_DIR=$M D=$D B=md122_a05_ep500_reassign SEED=42 STAGE=train NW=2 DRYRUN=1 bash $RB
for K in $KINDS; do for V in $VARS; do
  M=$M D=$D KIND=\$K VAR=\$V SEED=42 DRYRUN=1 bash $RG
done; done
cd $T && python geo_config_diff.py $D $NAMES
"
echo done
