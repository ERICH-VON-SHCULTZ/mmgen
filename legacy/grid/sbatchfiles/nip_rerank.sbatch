#!/bin/bash

#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=16:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=nip
#SBATCH --output=/scratch/yw8866/GRID/expoutfile/nip_rerank.out

# ============================================================================================
# GOAL (rec-tmall/GOAL.md): MFD beats chain on t3@10 by >= +0.01 absolute. Lead today +0.0017-0.0020.
#
# WHAT THE LAST ANALYSES FOUND (cached per-event results, 5 seeds):
#  * For page-views both systems succeed almost only on REPEATS: MFD 12,951 repeat hits vs 6 new-item
#    hits; chain 12,669 vs 134. MFD's whole lead is repeat coverage (+0.0034), partly given back on
#    new items (-0.0016).
#  * On new items MFD's RERANKER evicts what its generator got right:
#        BOTH     pv new items (1,176):  MFD base 0.0570 -> rerank 0.0009   chain 0.0842 -> 0.0847
#        MFD-ONLY pv new items (4,663):  MFD base 0.0212 -> rerank 0.0010   (not in chain's pool)
#    MFD's pool is 4x chain's and carries far more history-matched candidates; a listwise model trained
#    mostly on repeat-GT events fills the top 10 with them.
#
# THIS JOB -- NIP, new-item protection, both arms treated identically, everything chosen on VAL:
#  * final-list policy P_k: top-(10-k) by the reranker + top-k by the GENERATOR's base score among
#    candidates NOT in the user's history (k in 0..3; P0 = today's list; k chosen on each arm's VAL).
#  * training reweight w_new: new-item (non-repeat) training events weighted x w_new (1 = today, 3).
#  * policy D: one top-10 slot per ITEM (keep the best behavior of each (f,t,i)); t3 ranks the union of
#    four behavior pools, so the same item can waste slots. Duplicate-slot rate is reported per run.
# 3 seeds x 2 arms x 2 w_new. No new dumps. The deciding cell of the report is MFD-ONLY x new item:
# ground truths chain's pool does not contain, so gains there move the lead one-for-one.
# ============================================================================================

MD_SID=/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt
TG_SID=/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt
POOLS=/scratch/yw8866/rec-tmall/coverage_exp/pools
CHD=/scratch/yw8866/rec-tmall/matched_rerank
OUT=/scratch/yw8866/rec-tmall/nip_exp; mkdir -p $OUT
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

python $NIP --spec $OUT/spec.json --cache $OUT/cache --seeds 0,1,2 --w_new 1,3 \
    --cap 36864 --wb 1.0 --heads 4 --dropout 0.3 --lr 1e-3 --wd 1e-3 --epochs 15 --patience 3 --min_delta 0.0 \
    --batch 8 --eval_batch 16 --boot 2000 --seed 0 --bar 0.01 --out_dir $OUT
"
