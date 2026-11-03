#!/bin/bash
# Gate of the encoder-structure runs (docs/MFD_REPORT.md §16): sanity checks 4-5 (macrec/struct_check.py) and the
# dry-run config diff against the chain control (sanity 1, tools/struct_config_diff.py). No training. Frozen code copy.
# Outputs: mmgen_runs/struct/gate/<D>/{struct_check.json,struct_config_diff.json}, dry runs in struct/gate/dryrun/<D>/.
# Submit: sbatch --export=ALL,D=Instruments sbatch/struct/struct_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_struct_gate
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
D=${D:-Instruments}
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RS=$TMPDIR/run_struct_train.sh; cp $REPO/sbatch/struct/run_struct_train.sh $RS
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB
T=$TMPDIR/tools; mkdir -p $T; cp $REPO/sbatch/tools/{struct_config_diff,geo_config_diff,plugin_config_diff}.py $T/
mkdir -p $RUNS/struct/gate/$D
echo "job $SLURM_JOB_ID on $(hostname) D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
python -u struct_check.py --base_model ./config/ckpt --data_path $RUNS/data/ --dataset $D --index_file .index_md122_a05_ep500_reassign.json \
  --image_index_file .index_md122_a05_ep500_reassign.json --max_his_len 20 --out_file $RUNS/struct/gate/$D/struct_check.json \
  > $RUNS/struct/gate/$D/struct_check.log 2>&1 || { tail -20 $RUNS/struct/gate/$D/struct_check.log; echo 'STRUCT CHECK FAILED'; exit 1; }
tail -40 $RUNS/struct/gate/$D/struct_check.log
[ -f $RUNS/plugin/dryrun/$D/md122_a05_ep500_reassign/seed42/dryrun.json ] || MACREC_DIR=$M D=$D B=md122_a05_ep500_reassign SEED=42 STAGE=train NW=2 DRYRUN=1 bash $RB
for V in E1 E2; do M=$M D=$D VAR=\$V SEED=42 DRYRUN=1 bash $RS; done
cd $T && python struct_config_diff.py $D chain122_E1 chain122_E2
"
echo done
