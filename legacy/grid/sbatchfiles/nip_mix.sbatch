#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=nipmix
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/nip_mix.out

# ============================================================================================
# GOAL: MFD beats chain on t3@10 by >= +0.01 absolute. recall = coverage x P(rank<10 | covered).
# BREAKDOWN (5 seeds): on covered pv-NEW items chain's reranker converts 5.9% (own-def 1.7%), MFD's 0.1%,
# although MFD's generator had 2.8% before reranking; MFD covers 3,643 MORE new items than chain (5,839 vs
# 2,196) -- items chain's beam cannot reach. Oracle-gate ceiling with the BASE score as new-item ranker:
# lead +0.0033 (MFD new P 0.056 vs chain 0.1025). With equal new-item P for both arms, MFD's extra new-item
# coverage is worth P x 0.0439 of t3@10 -- so the bar needs a LEARNED new-item ranker near P ~ 0.15.
# THIS JOB: calibrated per-event mixture, both arms identically, chosen on VAL:
#   P(c) = G' P_main(c|history) for history candidates, (1-G') P_new(c) for the rest,
#   G = main reranker's mass on history candidates, logit(G') = logit(G) - beta, beta in -1..6,
#   P_new = new-item SPECIALIST (trained only on non-repeat events, history masked).
# Reports each arm's specialist VAL new-item hit@10 (its P_new) and the lead at each arm's VAL choice.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/nip_mix; mkdir -p $OUT
NIP=/scratch/yw8866/rec-tmall/nip_rerank.py
VAL=/scratch/yw8866/rec-tmall/validate_dump.py
PB_TEST='77833,2042,1554,1576'

for f in $POOLS/p3k_train.pt $POOLS/p3k_val.pt $POOLS/p3k_test.pt \
         $CHD/chain808_train_dump.pt $CHD/chain808_val_dump.pt $CHD/chain808_test_dump.pt; do
  [ -f $f ] || { echo "** missing $f **"; exit 1; }
done
rm -f $OUT/nip_report.txt
cat > $OUT/spec.json <<JSON
{
 "MFD": {"tr": "$POOLS/p3k_train.pt", "va": "$POOLS/p3k_val.pt", "te": "$POOLS/p3k_test.pt", "sid": "$MD_SID", "kpool": 2000},
 "CH":  {"tr": "$CHD/chain808_train_dump.pt", "va": "$CHD/chain808_val_dump.pt", "te": "$CHD/chain808_test_dump.pt", "sid": "$TG_SID", "kpool": null}
}
JSON

module purge

singularity exec --nv \
    --overlay /scratch/yw8866/grid_env/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate py310
set -f
export HF_HOME=/scratch/yw8866/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT='MMMBGRec'
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_API_KEY='REDACTED_ROTATE_ME'

python $VAL --dump $POOLS/p3k_test.pt         --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1
python $VAL --dump $CHD/chain808_test_dump.pt --expect_n 83005 --expect_perbeh '$PB_TEST' --expect_states 808 || exit 1

python $NIP --spec $OUT/spec.json --cache $OUT/cache --seeds 0,1,2 --w_new 1 --specialist \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --bar 0.01 --out_dir $OUT
"
