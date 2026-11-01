#!/bin/bash
# Gate for MFD with C's task mix (docs/MFD_REPORT.md §13), before training. Sanity checks:
#   1 branch independence (+ exact scoring, f loss once) on an UNTRAINED model with the task-mix inputs (mfd_eval.py),
#   2-3 task inputs/targets on printed samples + examples per epoch per task (mfd_mt_check.py),
#   4 config diff against mfd122_mv: dry runs (run_mfd_train.sh DRYRUN=1) + sbatch/tools/mfd_mt_config_diff.py.
# Exit 1 on any failure: do not train. Frozen snapshot of macrec/ and the run scripts.
# Submit: sbatch [--export=ALL,D=Instruments] sbatch/mfd/mt_gate.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_mt_gate
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export WANDB_MODE=offline                                    # dry runs stop before training; W&B stays enabled (AGENTS.md §9)
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
D=${D:-Instruments}
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RM=$TMPDIR/run_mfd_train.sh; cp $REPO/sbatch/mfd/run_mfd_train.sh $RM
cp $REPO/sbatch/tools/mfd_mt_config_diff.py $REPO/sbatch/tools/plugin_config_diff.py $TMPDIR/
G=$RUNS/mfd/mt_gate/$D; mkdir -p $G
IDX=.index_md122_a05_ep500_reassign.json
echo "job $SLURM_JOB_ID on $(hostname) D=$D commit $(cd $REPO && git rev-parse --short HEAD)"
singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
M=$M D=$D CFG=122 AUG=1 SEED=42 DRYRUN=1 bash $RM
for X in A B; do M=$M D=$D CFG=122 AUG=0 MT=\$X SEED=42 DRYRUN=1 bash $RM; done
source /ext3/env.sh && conda activate py310
cd $M
for X in A B; do
  T=\$(python -c \"import json; print(json.load(open('$RUNS/mfd/mt_gate/dryrun/$D/mfd122_mt'+'\$X'+'/run_config.json'))['tasks'])\")
  echo \"== task check mfd122_mt\$X (\$T)\"
  python -u mfd_mt_check.py --base_model ./config/ckpt --data_path $RUNS/data --dataset $D --index_file $IDX \
    --image_index_file $IDX --max_his_len 20 --prompt_num 4 --tasks \$T --out_file $G/task_check_mfd122_mt\$X.json
  echo \"== untrained sanity mfd122_mt\$X\"
  python -u mfd_eval.py --random_init --sanity_only --base_model ./config/ckpt --data_path $RUNS/data --dataset $D \
    --index_file $IDX --image_index_file $IDX --max_his_len 20 --test_batch_size 64 --k_f 20 --tasks \$T \
    --out_dir $G/sanity_mfd122_mt\$X | tail -3
done
cd $TMPDIR && python -u mfd_mt_config_diff.py $D mfd122_mtA mfd122_mtB
"
echo done
