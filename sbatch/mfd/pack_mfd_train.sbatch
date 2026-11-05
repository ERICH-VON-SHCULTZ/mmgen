#!/bin/bash
# Several MFD / chain trainings CONCURRENTLY on one GPU (one run alone keeps a GPU ~30% busy, below the cluster's
# utilisation threshold; see pack_train.sbatch). Training math is unchanged; wall-clock is contended.
# Every run uses ONE frozen copy of macrec/ and of the run scripts, made at job start.
# RUNS entries:  mfd:<D>:<CFG 122|133>:<AUG 0|1>[:<SEED>]   (sbatch/mfd/run_mfd_train.sh; SEED default 42)
#                chain:<D>:<baseline, e.g. mdmv122> (sbatch/baselines/run_baseline.sh STAGE=train)
#                mfdmt:<D>:<A|B|C>[:<SEED>]   MFD 1+2+2 with C's task mix (run_mfd_train.sh MT=; needs a passed sbatch/mfd/mt_gate.sbatch)
#                struct:<D>:<E1|E2>[:<SEED>]   chain decoder with item-level encoder structure (sbatch/struct/run_struct_train.sh; needs sbatch/struct/struct_gate.sbatch)
#                cgeo:<D>   baseline C + codebook-initialised code embeddings (run_baseline.sh B=mql GEOARGS, §17 Part 3) -> geo/<D>/c_geo/
#                geo:<D>:<chain|mfd>:<A|C|AB1|AB2|V1..V4>[:<SEED>]   codebook-geometry run (sbatch/geo/run_geo_train.sh; needs a passed sbatch/geo/geo_gate.sbatch)
# Submit: sbatch --export=ALL,RUNS="mfd:Instruments:133:0 chain:Arts:mdmv122" sbatch/mfd/pack_mfd_train.sbatch
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 24
#SBATCH --mem=160GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mmgen_mfd_pack
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/%x_%j.out

set -euo pipefail
module purge
REPO=/scratch/yw8866/mmgen
RUNS_DIR=/scratch/yw8866/mmgen_runs
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3   # always :ro in jobs
SECRETS=$HOME/.config/mmgen/secrets.sh
if [ -f "$SECRETS" ]; then source "$SECRETS"; fi
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then export WANDB_MODE=offline; fi
export WANDB_PROJECT=mmgen-mfd WANDB_DIR=$RUNS_DIR/wandb
export HF_HOME=/scratch/yw8866/hf_cache USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/scratch/yw8866/tmp/slurm_$SLURM_JOB_ID; mkdir -p "$TMPDIR" $WANDB_DIR
M=$TMPDIR/macrec; mkdir -p $M; cp $REPO/macrec/*.py $M/; cp -r $REPO/macrec/config $M/
RB=$TMPDIR/run_baseline.sh; cp $REPO/sbatch/baselines/run_baseline.sh $RB
RM=$TMPDIR/run_mfd_train.sh; cp $REPO/sbatch/mfd/run_mfd_train.sh $RM
RG=$TMPDIR/run_geo_train.sh; cp $REPO/sbatch/geo/run_geo_train.sh $RG
RS=$TMPDIR/run_struct_train.sh; cp $REPO/sbatch/struct/run_struct_train.sh $RS
LOGD=$RUNS_DIR/logs/pack_$SLURM_JOB_ID; mkdir -p $LOGD
echo "job $SLURM_JOB_ID on $(hostname)  RUNS=$RUNS  commit $(cd $REPO && git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > $LOGD/gpu_util.csv &
# chain runs: check their index against v1 here (run_baseline.sh does not)
for R in $RUNS; do
  IFS=: read K D X Y <<< "$R"
  if [ $K = chain ]; then
    case $X in mdmv122) I=$RUNS_DIR/data/$D/$D.index_md122_a05_ep500_reassign.json; V=$RUNS_DIR/md/sids/v1/$D/1+2+2 ;; *) echo "unknown chain run $X"; exit 1 ;; esac
    [ "$(sha256sum < $I | cut -d' ' -f1)" = "$(cut -d' ' -f1 $V/index.sha256)" ] || { echo "$I != v1"; exit 1; }
  fi
done
START=$(date +%s); PIDS=()
for R in $RUNS; do
  IFS=: read K D X Y S <<< "$R"
  if [ $K = mfd ]; then
    CMD="M=$M D=$D CFG=$X AUG=$Y SEED=${S:-42} bash $RM"
  elif [ $K = mfdmt ]; then
    CMD="M=$M D=$D CFG=122 AUG=0 MT=$X SEED=${Y:-42} bash $RM"
  elif [ $K = cgeo ]; then
    T=$RUNS_DIR/baselines/$D/tok
    CMD="MACREC_DIR=$M D=$D B=mql SEED=42 STAGE=train NW=2 OUTROOT=$RUNS_DIR/geo/$D/c_geo GEOARGS='--geo codebook --geo_sid c --geo_ckpt $T/mql_text/best_collision_model.pth --geo_ckpt_image $T/mql_image/best_collision_model.pth' bash $RB"
  elif [ $K = struct ]; then
    CMD="M=$M D=$D VAR=$X SEED=${Y:-42} bash $RS"
  elif [ $K = geo ]; then
    CMD="M=$M D=$D KIND=$X VAR=$Y SEED=${S:-42} bash $RG"
  else
    CMD="MACREC_DIR=$M D=$D B=$X SEED=42 STAGE=train NW=2 bash $RB"
  fi
  singularity exec --nv --bind /scratch/yw8866 --overlay "$OVERLAY":ro "$SIF" /bin/bash -c "$CMD" > $LOGD/${R//:/_}.log 2>&1 &
  PIDS+=($!); sleep 20
done
FAIL=0; for P in "${PIDS[@]}"; do wait $P || FAIL=1; done
awk -F', ' '{u+=$2; n++} END {printf "mean GPU util over job: %.1f%% (%d samples)\n", u/n, n}' $LOGD/gpu_util.csv
echo "job wall time: $(( $(date +%s) - START )) s  fail=$FAIL"
exit $FAIL
