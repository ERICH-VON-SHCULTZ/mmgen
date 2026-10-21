#!/bin/bash
# Build the MACRec env as a COPY of the GRID overlay (the original is never modified),
# add MACRec's missing packages without upgrading anything else, then verify read-only.
#SBATCH --account=torch_pr_69_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --job-name=mmgen_build_env
#SBATCH --output=/scratch/yw8866/mmgen_runs/logs/build_env_%j.out

set -euo pipefail
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
SRC=/scratch/yw8866/grid_env/overlay-25GB-500K.ext3
DST_DIR=/scratch/yw8866/envs/macrec
DST=$DST_DIR/overlay-25GB-500K.ext3
REPO=/scratch/yw8866/mmgen
mkdir -p "$DST_DIR"

# ---- 1) copy (skip if a complete copy already exists) ----
if [ ! -f "$DST" ] || [ "$(stat -c %s "$DST")" != "$(stat -c %s "$SRC")" ]; then
  echo "[copy] $SRC -> $DST"; date
  cp --sparse=always "$SRC" "$DST.partial" && mv "$DST.partial" "$DST"
  date
fi

# ---- 2) modify ONLY the copy (rw; this job is the only one touching it) ----
singularity exec --fakeroot --overlay "$DST":rw "$SIF" /bin/bash -c '
set -e
# env.sh: the GRID one exports PYTHONPATH=/ext3/miniforge3/bin:$PATH (a bin dir on PYTHONPATH).
cat > /ext3/env.sh <<EOF
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:\$PATH
# TensorFlow + Keras 3 are installed (GRID TFRecord tooling); keep transformers torch-only.
export USE_TF=0
EOF
source /ext3/env.sh
conda activate py310
# constraints = everything already installed, so the new packages cannot upgrade torch/transformers/...
pip freeze | grep -v " @ " > /tmp/constraints.txt
pip install --no-cache-dir -c /tmp/constraints.txt evaluate==0.4.3 ftfy==6.3.1 ipdb==0.13.13
'

# ---- 3) verify (ro) + lock file ----
singularity exec --overlay "$DST":ro "$SIF" /bin/bash -c "
set -e
source /ext3/env.sh && conda activate py310
echo PYTHONPATH=\${PYTHONPATH:-<unset>}
python -V
pip freeze > $REPO/envs/requirements.lock
cd $REPO/macrec
python - <<'PY'
import torch, transformers, sys
print('torch', torch.__version__, 'built for CUDA', torch.version.cuda, '| transformers', transformers.__version__)
a = transformers.TrainingArguments(output_dir='/tmp/ta_check', evaluation_strategy='epoch',
                                   save_strategy='epoch', report_to=None)
print('TrainingArguments(evaluation_strategy=...) OK ->', a.eval_strategy, '| report_to resolved to', a.report_to)
import utils, data, collator, modeling, generation_trie, evaluate as macrec_eval
assert hasattr(macrec_eval, 'get_topk_results'), 'pip evaluate shadowed MACRec evaluate.py'
print('MACRec top-level modules import OK (local evaluate.py wins over pip evaluate)')
PY
# cross_index/ and data_process/ each have their own utils.py: separate interpreters, run from their dir
(cd cross_index && python -c 'import models.rqvae, trainer, datasets; print(\"cross_index models/trainer/datasets import OK\")')
(cd data_process && python -c 'from clip import clip; import utils; print(\"data_process clip/utils import OK\")')
python finetune_contrastive.py --help > /dev/null && echo 'finetune_contrastive.py --help OK'
python test_ddp_save.py --help > /dev/null && echo 'test_ddp_save.py --help OK'
(cd cross_index && python main.py --help > /dev/null) && echo 'cross_index/main.py --help OK'
(cd cross_index && python generate_indices_distance.py --help > /dev/null) && echo 'generate_indices_distance.py --help OK'
(cd data_process && python amazon_text_emb.py --help > /dev/null) && echo 'amazon_text_emb.py --help OK'
(cd data_process && python amazon18_data_process.py --help > /dev/null) && echo 'amazon18_data_process.py --help OK'
(cd data_process && python clip_feature.py --help > /dev/null) && echo 'clip_feature.py --help OK'
"
echo "[done] env at $DST"
