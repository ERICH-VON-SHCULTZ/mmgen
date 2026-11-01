#!/bin/bash
# Plug-in config gate (docs/PLUGIN_REPORT.md §2): dry runs (run_baseline.sh DRYRUN=1: exact training command, stops
# before trainer.train()) of C (mql) and every plug-in run, then sbatch/tools/plugin_config_diff.py. Exit 1 = an
# unintended difference from C: do not train. Frozen snapshot of macrec/ and run_baseline.sh.
# Submit: sbatch [--export=ALL,RUNS="Instruments:plug122 ..."] sbatch/plugin/config_check.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_plugin_cfg
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export WANDB_MODE=offline                                    # dry runs stop before training; W&B stays enabled (AGENTS.md §9)
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB
cp $REPO/sbatch/tools/plugin_config_diff.py $TMPDIR/
RUNS=${RUNS:-"Instruments:plug122 Instruments:plug122s Instruments:plug133 Instruments:plug133s Arts:plug122 Arts:plug133"}
echo "job $SLURM_JOB_ID on $(hostname) RUNS=$RUNS commit $(cd $REPO && git rev-parse --short HEAD)"
DS=$(for R in $RUNS; do echo ${R%%:*}; done | sort -u | tr "\n" " ")
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
for D in $DS; do MACREC_DIR=$M D=\$D B=mql SEED=42 DRYRUN=1 NW=2 bash $RB; done
for R in $RUNS; do MACREC_DIR=$M D=\${R%%:*} B=\${R##*:} SEED=42 DRYRUN=1 NW=2 bash $RB; done
source /ext3/env.sh && conda activate py310
python -u $TMPDIR/plugin_config_diff.py $RUNS
"
