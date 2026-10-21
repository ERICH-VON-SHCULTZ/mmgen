# Environment setup (MACRec working base)

**Route taken: copy of the GRID overlay, modified in the copy only.**
- Original, never written to: `/scratch/yw8866/grid_env/overlay-25GB-500K.ext3`
- Copy used by all mmgen jobs: `/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3`
- Image: `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Conda env: `py310` (Python 3.10.20)
- Builder: `sbatch/build_env_macrec.sbatch` (CPU account `torch_pr_69_general`). It copies the
  overlay, edits it, installs, and verifies. Full freeze: `envs/requirements.lock` (282 packages).

## 1. MACRec requirements vs the GRID overlay

MACRec's `requirements.txt` is unpinned, so the "required" column lists the package names only.
The rows marked *implicit* are packages the code imports but `requirements.txt` does not list.

| package | MACRec requires | GRID overlay | status |
|---|---|---|---|
| python | (3.8+ implied) | 3.10.20 | match |
| torch | implicit | 2.6.0+cu124 | compatible (see §2) |
| transformers | yes | 4.47.0 | **compatible with a caveat**: `evaluation_strategy=` (`finetune_contrastive.py:132`) is deprecated but still accepted in 4.47 (verified). It is removed in later versions, so **pin ≤ 4.47** |
| accelerate | yes | 1.8.1 | match |
| bitsandbytes | yes | 0.46.0 | match (unused by the seqrec path) |
| deepspeed | yes | 0.16.3 | match (unused; `args.deepspeed=None`, `finetune_contrastive.py:48`) |
| peft | yes | 0.15.2 | match (unused) |
| sentencepiece | yes | 0.2.0 | match (T5Tokenizer) |
| tqdm | yes | 4.67.1 | match |
| evaluate | yes | **missing** | installed `evaluate==0.4.3`. MACRec's own `evaluate.py` shadows it when run from `macrec/` (verified) |
| ipdb | implicit (`modeling.py:16`, top-level import) | **missing** | installed `ipdb==0.13.13`. Without it `modeling.py` fails to import |
| ftfy | implicit (`data_process/clip/simple_tokenizer.py`) | **missing** | installed `ftfy==6.3.1` |
| regex, requests, pillow, torchvision, scikit-learn, numpy | implicit | 2024.11.6, 2.32.4, 11.3.0, 0.21.0+cu124, 1.7.2, 2.0.2 | match |
| setuptools (`pkg_resources`, `data_process/clip/clip.py:6`) | implicit | 80.9.0 | compatible: prints a deprecation warning; keep **< 81** |
| huggingface-hub / tokenizers / safetensors | via transformers | 0.33.2 / 0.21.0 / 0.5.3 | match |
| wandb | (HF Trainer `report_to=None` = all) | 0.19.0 | present, so the Trainer logs to W&B and TensorBoard |
| tensorflow + keras 3.10 | not needed | **present** (GRID TFRecord tooling) | **conflict**: transformers imports TF and crashes on Keras 3. Fixed with `export USE_TF=0` (in the copy's `/ext3/env.sh` and in every sbatch) |

New packages were installed with `pip install -c <freeze of the existing env>`, so nothing that
was already installed could be up- or downgraded.

## 2. torch build vs CUDA 12.1
- The container image ships CUDA 12.1. The torch wheel is `2.6.0+cu124`, which **bundles its own
  CUDA 12.4 runtime** (`nvidia-cuda-runtime-cu12 12.4.127`).
- What matters is the host driver, not the image toolkit. The same overlay trained all GRID
  models on these H100 nodes, and the smoke test checks `torch.cuda.is_available()`.
- The cu121 toolkit in the image would only matter when compiling CUDA extensions. MACRec
  compiles none.
- **torch ≥ 2.6 `torch.load` default.** It changed to `weights_only=True`, and MACRec's RQ-VAE
  checkpoints pickle an `argparse.Namespace` (`cross_index/trainer.py:362-371`, loaded at
  `cross_index/generate_indices_distance.py:71`). Jobs export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`;
  MACRec code is not patched.

## 3. Changes made to the copy (and only the copy)
1. `/ext3/env.sh` rewritten. The GRID version exported `PYTHONPATH=/ext3/miniforge3/bin:$PATH`,
   which puts bin directories on `PYTHONPATH`. The new file:
   ```bash
   unset -f which
   source /ext3/miniforge3/etc/profile.d/conda.sh
   export PATH=/ext3/miniforge3/bin:$PATH
   export USE_TF=0
   ```
2. `pip install evaluate==0.4.3 ftfy==6.3.1 ipdb==0.13.13`. The dependencies pulled in were
   ipython 8.39.0, jedi, parso, traitlets, pexpect, stack_data and others; all are listed in
   `requirements.lock`.

Writing to an overlay on this cluster requires `--fakeroot`. Without it, the mount fails with
`overlay upper dir ... permission denied`.

## 4. Verified (job 20477972, log `/scratch/yw8866/mmgen_runs/logs/build_env_20477972.out`)
- `PYTHONPATH` unset.
- `import utils, data, collator, modeling, generation_trie, evaluate` from `macrec/`.
- `cross_index` models, trainer and datasets import.
- `data_process` clip and utils import.
- `--help` exits cleanly for `finetune_contrastive.py`, `test_ddp_save.py`, `cross_index/main.py`,
  `generate_indices_distance.py`, `amazon_text_emb.py`, `amazon18_data_process.py` and
  `clip_feature.py`.

## 5. Re-creating it by hand

```bash
# rebuild: copy + modify + verify (CPU job, about 10 min). Never run while a job has the copy mounted.
sbatch /scratch/yw8866/mmgen/sbatch/build_env_macrec.sbatch

# or interactively, step by step:
mkdir -p /scratch/yw8866/envs/macrec
cp --sparse=always /scratch/yw8866/grid_env/overlay-25GB-500K.ext3 /scratch/yw8866/envs/macrec/
singularity exec --fakeroot --overlay /scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
#   (inside) write /ext3/env.sh as in §3, then:
#   source /ext3/env.sh && conda activate py310
#   pip freeze | grep -v " @ " > /tmp/c.txt && pip install -c /tmp/c.txt evaluate==0.4.3 ftfy==6.3.1 ipdb==0.13.13

# use (always :ro in jobs):
singularity exec --nv --bind /scratch/yw8866 \
  --overlay /scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh && conda activate py310 && cd /scratch/yw8866/mmgen/macrec && python ..."
```

**Fresh-environment fallback** (if the copy is ever lost): Miniforge into a new
`overlay-25GB-500K.ext3` (from `/share/apps/overlay-fs-ext3/`), `conda create -n py310
python=3.10`, then `pip install torch==2.6.0 torchvision==0.21.0 --index-url
https://download.pytorch.org/whl/cu124` and `pip install transformers==4.47.0 accelerate==1.8.1
peft==0.15.2 sentencepiece==0.2.0 evaluate==0.4.3 ftfy==6.3.1 ipdb==0.13.13 regex requests
scikit-learn "setuptools<81" tqdm wandb==0.19.0`. bitsandbytes and deepspeed are optional;
the seqrec path does not use them.

## 6. Secrets / W&B
- Job scripts source `~/.config/mmgen/secrets.sh` (chmod 600, outside the repo). It currently
  holds only a commented placeholder. Put `export WANDB_API_KEY=<new key>` there **after
  rotating** the old key.
- With no key and no `~/.netrc`, jobs set `WANDB_MODE=offline`: W&B still records locally under
  `/scratch/yw8866/mmgen_runs/wandb`, and `wandb sync` uploads it later. Jobs never put the key
  on a command line and never use `set -x`.
