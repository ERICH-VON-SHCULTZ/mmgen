#!/bin/bash
# Reproduce the locked v1 MD-RQ-VAE SID sets from scratch and compare them with v1 (never touches v1).
# Same steps as the official sets: main_mdrqvae.py (a05 / a06 settings) -> final.pth -> reassign_md.py.
# Output: /scratch/yw8866/mmgen_runs/md/repro/<MODE>/<D>/<name>/ (+ index.json there), summary repro_v1.tsv.
# Submit: sbatch [--export=ALL,SETS="Instruments:122 Arts:133"] sbatch/md/repro_v1.sbatch
# MODE=full (default): retrain from scratch, then reassign. MODE=reassign: copy the LOCKED v1
# tokenizer.pth and rerun only the reassignment (stage 2), which must reproduce v1 exactly.
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h200_tandon,a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_md_repro
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
export USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
SNAP=$TMPDIR/cross_index; cp -r $REPO/macrec/cross_index $SNAP
MODE=${MODE:-full}
SETS=${SETS:-"Instruments:122 Instruments:133 Arts:122 Arts:133 Games:122 Games:133"}
echo "job $SLURM_JOB_ID on $(hostname) MODE=$MODE SETS=$SETS commit $(cd $REPO && git rev-parse --short HEAD)"
SUM=$RUNS/md/repro/repro_v1_${MODE}_$SLURM_JOB_ID.tsv; mkdir -p $RUNS/md/repro
echo -e "dataset\tcfg\tgpu\tv1_sha256\trepro_sha256\tmatch\titems_differing" > $SUM
for s in $SETS; do
  D=${s%%:*}; C=${s##*:}
  case $C in
    122) NAME=a05_122; NM=2; TAG=md122_a05_ep500_reassign; CFG=1+2+2 ;;
    133) NAME=a06_133; NM=3; TAG=md133_a06_ep500_reassign; CFG=1+3+3 ;;
  esac
  OUT=$RUNS/md/repro/$MODE/$D/$NAME; rm -rf $OUT; mkdir -p $OUT
  V1DIR=$RUNS/md/sids/v1/$D/$CFG
  [ $MODE = reassign ] && cp $V1DIR/tokenizer.pth $OUT/final.pth && chmod u+w $OUT/final.pth
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "
  set -euo pipefail
  source /ext3/env.sh && conda activate py310
  cd $SNAP
  [ $MODE = reassign ] || python -u main_mdrqvae.py --dataset $D --out_dir $OUT --n_shared 1 --n_modal $NM \
      --dead_code_reset 10 --dead_code_until 400 --fusion xattn > $OUT/train.out 2>&1
  python -u reassign_md.py --ckpt $OUT/final.pth --tag repro_$TAG > $OUT/reassign.out 2>&1
  "
  # reassign_md.py writes the index into the data dir; move it next to the run
  mv $RUNS/data/$D/$D.index_repro_$TAG.json $OUT/index.json
  V1=$RUNS/md/sids/v1/$D/$CFG/index.json
  A=$(sha256sum < $V1 | cut -d' ' -f1); B=$(sha256sum < $OUT/index.json | cut -d' ' -f1)
  NDIFF=$(python3 -c "import json;a=json.load(open('$V1'));b=json.load(open('$OUT/index.json'));print(sum(a[k]!=b.get(k) for k in a))")
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  M=$([ "$A" = "$B" ] && echo MATCH || echo DIFF)
  echo -e "$D\t$CFG\t$GPU\t$A\t$B\t$M\t$NDIFF" | tee -a $SUM
done
echo "summary: $SUM"
