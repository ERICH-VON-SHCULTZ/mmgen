#!/bin/bash
# Round-22 E4 (docs/MFD_REPORT.md §22.4): level-1 (first-step) distributions per TEST user, no training.
# Per dataset in DS (default "Arts Instruments"), seed 42: A01 f (macrec/r22_e4_first.py --kind mfd), C text + image level 1
# (mql), TIGER A text level 1 (tiger_text), TIGER B image level 1 (tiger_image) -> geo/r22/<D>/e4_<model>.pt.
# Analysis (CPU): sbatch --export=ALL,TOOL="r22_e4_report.py" sbatch/r21/r21_tool.sbatch -> geo/r22_e4.json
# Submit: sbatch sbatch/r22/r22_e4.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_tandon
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=mmgen_r22_e4
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
DS=${DS:-"Arts Instruments"}; IDX=.index_md122_a05_ep500_reassign.json
echo "job $SLURM_JOB_ID on $(hostname) DS=$DS commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
for D in $DS; do
  V1=$RUNS/md/sids/v1/$D/1+2+2
  [ "$(sha256sum < $RUNS/data/$D/$D$IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V1/index.sha256)" ] || { echo "index != v1"; exit 1; }
  O=$RUNS/geo/r22/$D; mkdir -p $O
  B=$RUNS/baselines/$D
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh && conda activate py310
cd $M
C='--data_path $RUNS/data --dataset $D --max_his_len 20 --test_batch_size 64'
python -u r22_e4_first.py \$C --kind mfd --ckpt_path $RUNS/mfd/$D/r21_A01/seed42 --index_file $IDX --image_index_file $IDX --tasks seqrec_mv --prompt_num 4 --out_file $O/e4_A01.pt
python -u r22_e4_first.py \$C --kind chain --ckpt_path $B/mql/seed42 --index_file .index_lemb_mql.json --image_index_file .index_vitemb_mql.json --out_file $O/e4_C.pt
python -u r22_e4_first.py \$C --kind chain --ckpt_path $B/tiger_text/seed42 --index_file .index_lemb_tiger.json --image_index_file .index_lemb_tiger.json --first_tasks seqrec --out_file $O/e4_A.pt
python -u r22_e4_first.py \$C --kind chain --ckpt_path $B/tiger_image/seed42 --index_file .index_vitemb_tiger.json --image_index_file .index_vitemb_tiger.json --first_tasks seqrec --out_file $O/e4_B.pt
"
done
echo done
