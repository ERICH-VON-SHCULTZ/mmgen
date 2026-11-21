#!/bin/bash
# Round 23 (docs/MFD_REPORT.md §23) submission: one frozen code snapshot, then per dataset a serial Slurm chain on H200s
#   prep (afterok gate: C-fast == C, checks) -> timing jobs one after another (each < 2 h: the cluster cancels jobs whose GPU
#   utilisation stays < ~60 % for 2 h, and the Python-trie beam searches keep the GPU ~12 % busy) -> profile (stage breakdown,
#   counts, FLOPs); the three dataset chains run in parallel; CPU report after all of them.
# Why several jobs instead of one: the protocol's single job would run ~35 h at ~12 % GPU utilisation (cancelled after 2 h);
# every job asks for the same GPU type (gpu:h200 on h200_tandon), records host / GPU / driver, and repeats a fixed anchor
# measurement so jobs can be compared. Usage: [REPORT_AFTER=j1:j2] bash sbatch/r23/r23_submit.sh [datasets]   (log: mmgen_runs/logs/round23_jobs.txt)
set -euo pipefail
REPO=/scratch/yw8866/mmgen; RUNS=/scratch/yw8866/mmgen_runs
cd $REPO
[ -z "$(git status --porcelain -- macrec sbatch/r23)" ] || { echo "commit macrec/ and sbatch/r23 first"; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
CODE=$RUNS/timing/r23/code_$COMMIT; mkdir -p $CODE; cp macrec/*.py $CODE/; cp -r macrec/config $CODE/; echo $COMMIT > $CODE/COMMIT
LOG=$RUNS/logs/round23_jobs.txt
J=sbatch/r23/r23_job.sbatch
DSETS=${1:-"Arts Instruments Games"}
LAST=()
for D in $DSETS; do
  export D CODE                                 # values with commas go through the environment (--export=ALL), not --export=
  if [ $D = Games ]; then        # 42k users: C-fast == C (two full beam searches, ~1.5 h) is its own job after the other checks
    # PREP1_DONE=1: part 1 already in timing/r23/Games/prep.json (all its checks passed), run part 2 only
    if [ -n "${PREP1_DONE:-}" ]; then P0=done; D1=""; else P0=$(STEP=prep PREPARGS="--skip cfast" sbatch --parsable --export=ALL $J); D1="--dependency=afterok:$P0"; fi
    PREP=$(STEP=prep PREPARGS="--resume --skip p1a_ref,reimpl,instrumented" sbatch --parsable $D1 --export=ALL $J)
    echo "$(date +%F) $D prep $P0 (checks 0/2/3/4) -> $PREP (C-fast == C, 1/1b) code $COMMIT" | tee -a $LOG
  else
    PREP=$(STEP=prep sbatch --parsable --export=ALL $J)
    echo "$(date +%F) $D prep $PREP code $COMMIT" | tee -a $LOG
  fi
  PREV=""
  # (TAG:METHODS:BATCHES[@REPEATS]) one entry per job (method names contain ":", so REPEATS follows "@"), each < 2 h at the smoke-test rates (Arts: P1a:50 10.5, C 65, C-fast 53,
  # TIGER 30, chain 37 ms/user at batch 64). Games' C / D / C-fast passes take ~45 min each (42k users), so there each repeat is
  # its own job (REPEATS=1, tags _r1.._r3; the report pools the three passes). C-fast 64 and 256 are separate jobs everywhere.
  if [ $D = Games ]; then
    JOBS="P1a64:P1a:20,P1a:50,P1a:100:64 P1a256:P1a:20,P1a:50,P1a:100:256 b1:P1a:20,P1a:50,P1a:100,Cfast,C,D,TIGER,chain:1
          TIGER64:TIGER:64 TIGER256:TIGER:256 chain64:chain:64 chain256:chain:256"
    for M in Cfast C D; do for B in 64 256; do for r in 1 2 3; do JOBS="$JOBS $M${B}_r$r:$M:$B@1"; done; done; done
  elif [ $D = Arts ]; then
    JOBS="P1a64:P1a:20,P1a:50,P1a:100,A01:50:64 P1a256:P1a:20,P1a:50,P1a:100,A01:50:256 Cfast64:Cfast:64 Cfast256:Cfast:256
          b1:P1a:20,P1a:50,P1a:100,A01:50,Cfast,C,D,TIGER,chain:1 C64:C:64 C256:C:256 D64:D:64 D256:D:256 TIGER:TIGER:64,256
          chain64:chain:64 chain256:chain:256"
  else
    JOBS="P1a:P1a:20,P1a:50,P1a:100:64,256 Cfast64:Cfast:64 Cfast256:Cfast:256 b1:P1a:20,P1a:50,P1a:100,Cfast,C,D,TIGER,chain:1
          C64:C:64 C256:C:256 D64:D:64 D256:D:256 TIGER:TIGER:64,256 chain:chain:64,256"
  fi
  for spec in $JOBS; do
    REPEATS=3; case $spec in *@*) REPEATS=${spec##*@}; spec=${spec%@*} ;; esac
    TAG=${spec%%:*}; rest=${spec#*:}; BATCHES=${rest##*:}; METHODS=${rest%:*}
    DEP="afterok:$PREP"; [ -n "$PREV" ] && DEP="$DEP,afterany:$PREV"
    PREV=$(STEP=time TAG=$TAG METHODS=$METHODS BATCHES=$BATCHES REPEATS=$REPEATS sbatch --parsable --dependency=$DEP --export=ALL $J)
    echo "$(date +%F) $D time $TAG [$METHODS @ $BATCHES x$REPEATS] $PREV" | tee -a $LOG
  done
  # profile: counts + FLOPs for every method; on Arts also the stage breakdown (batch 64 and 1), one job per method (< 2 h)
  ALLM="P1a:20 P1a:50 P1a:100 Cfast C D TIGER chain"; [ $D = Arts ] && ALLM="$ALLM A01:50"
  if [ $D = Arts ]; then
    for M in $ALLM; do
      PREV=$(STEP=profile TAG=${M/:/K} METHODS=$M STAGEB=64,1 sbatch --parsable --dependency=afterok:$PREP,afterany:$PREV --export=ALL $J)
      echo "$(date +%F) $D profile $M $PREV" | tee -a $LOG
    done
  else
    PREV=$(STEP=profile TAG=all METHODS=${ALLM// /,} STAGEB= sbatch --parsable --dependency=afterok:$PREP,afterany:$PREV --export=ALL $J)
    echo "$(date +%F) $D profile $PREV" | tee -a $LOG
  fi
  LAST+=($PREV)
done
DEPS=$(IFS=:; echo "${LAST[*]}${REPORT_AFTER:+:$REPORT_AFTER}")   # REPORT_AFTER: profile jobs of chains submitted earlier
REP=$(sbatch --parsable --dependency=afterany:$DEPS --account=torch_pr_69_general --partition=cs,cpu_short --time=00:30:00 -c 2 --mem=16GB \
  --job-name=mmgen_r23_report --output=$RUNS/logs/%x_%j.out --export=NONE \
  --wrap="singularity exec --bind /scratch/yw8866 --overlay /scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif bash -c 'source /ext3/env.sh && conda activate py310 && \
  python $REPO/sbatch/tools/r23_report.py'")
echo "$(date +%F) report $REP (after $DEPS)" | tee -a $LOG
