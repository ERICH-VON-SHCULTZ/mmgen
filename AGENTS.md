# AGENTS.md: mmgen

Read this first. It is the single entry point for any agent or new contributor working in this repo.
**Keep it current: every change that adds or moves a file, changes a setup step, a path, a convention,
or a result must update this file in the same commit** (see §12).

Last updated: 2026-10-24 (baseline reproduction in progress; Instruments seed 42 done).

---

## 1. What this project is

Research on **multi-modal generative recommendation with semantic IDs (SIDs)** on the Amazon
benchmarks **Instruments, Arts, Games** (about 6k–14k items each). The main line is:

- **MD-RQ-VAE**: a tokenizer that turns each item's text and image embeddings into a SID with a
  **shared** code `f` plus **text-specific** and **image-specific** codes. Text is reconstructed
  from `e_f + e_t` and image from `e_f + e_i`. See `docs/MDRQVAE.md`.
- **MFD (Modality-Factorized Decoding)**: a decoder that predicts `f`, then the text and image
  branches **in parallel**, conditionally independent given `f`. It drops the chain's `t → i`
  edge and builds candidates compositionally with a catalog constraint. See `docs/MFD.md`.

Baselines to compare against: **TIGER**, **MQL4GRec**, **MACRec**. The working codebase is
**MACRec** (`macrec/`). The old GRID/Tmall codebase is kept **read-only** in `legacy/grid/` as
reference. Behavior modeling, rerankers, history forcing and Tmall pipelines from the old project
are reference-only, not part of the main line.

**Current status (2026-10-24):**
- Repo, docs and environment are done.
- The TIGER smoke test on Instruments passes.
- **Baseline reproduction in progress** (TIGER-text, TIGER-image, MQL4GRec, MACRec on
  Instruments → Arts → Games). Workflow in §4b; report in `docs/BASELINE_REPORT.md`.
  Instruments seed 42: MQL4GRec and MACRec (paper setting) match; TIGER-text and MACRec as shipped
  miss (§7). Only matched baselines are extended to more seeds and datasets.
- **MD-RQ-VAE and MFD are NOT yet ported into MACRec.** The plan and the open decisions are in
  `docs/PORTING.md`.

---

## 2. Repository layout

```
mmgen/
├── AGENTS.md              this file (keep current)
├── CLAUDE.md              points Claude Code at this file
├── PROVENANCE.md          sources, commits, licenses, where large artifacts live
├── .gitignore             artifacts never enter git (see §9)
├── macrec/                ★ WORKING BASE: MACRec, upstream commit 7a48e74 (first git commit = pristine)
│   ├── data/{Instruments,Arts,Games}/   processed Amazon-18 data shipped by upstream (*.inter.json, *.item.json, ...)
│   ├── data_process/      raw→processed, LLaMA-7B text emb (amazon_text_emb.py), CLIP image emb (clip_feature.py)
│   ├── cross_index/       tokenizer: models/rqvae.py (CrossRQVAE), main.py (train), generate_indices_distance.py (SIDs)
│   ├── data.py            SeqRecDataset & multi-task datasets (leave-one-out splits)
│   ├── modeling.py        CrossModalContrastive (T5 + small encoder contrast)
│   ├── finetune_contrastive.py   HF-Trainer training entry
│   ├── test_ddp_save.py   constrained beam-search eval (trie) → hit@k / ndcg@k
│   ├── evaluate.py, generation_trie.py, collator.py, ensemble.py, utils.py
│   ├── config/ckpt/       T5 config + tokenizer (d_model 128, 4+4 layers, 6 heads, d_ff 1024)
│   ├── scripts/           upstream launchers (finetune_full.sh, run_finetune.sh)
│   │   ── mmgen additions (new files only; upstream files untouched, see git diff f3c9707 -- macrec) ──
│   ├── finetune_ext.py    finetune_contrastive.train + --model_class plain|contrastive + train stats
│   ├── eval_split.py      test_ddp_save on --split valid|test + inference cost (decoder steps, ms/user)
│   ├── cross_index/main_single.py, generate_indices_single.py   MQL4GRec single-modality tokenizer (verbatim @5ad9c27)
│   ├── cross_index/generate_indices_tiger.py   TIGER SIDs: 3 RQ levels + appended collision index
│   ├── cross_index/main_cross_weights.py       main.py with the contrast weights actually applied
│   ├── data/kmeans_dataset.py                  data/kmeans.py parameterised (K=512 pseudo-labels)
│   └── data_process/fetch_item_images.py, clip_feature_ext.py   main image download; CLIP tolerant to missing images
├── legacy/grid/           READ-ONLY reference: old GRID fork (Tmall MD-RQ-VAE + MFD). Never import.
│   ├── src/               md_rqvae.py, mfd_generation_model.py, tiger_generation_model.py, eval, data
│   ├── configs/experiment/   Hydra configs of the key runs
│   ├── sbatchfiles/       launchers of every old experiment (W&B key REDACTED)
│   ├── rec-tmall/         analysis scripts (+ GOAL.md research log, results/ small measured outputs)
│   ├── AGENTS.md          the OLD project's handbook (not this one)
│   └── README_LEGACY.md
├── docs/
│   ├── MDRQVAE.md         old tokenizer: architecture, loss, hyperparams, verified properties (cited file:line)
│   ├── MFD.md             old decoder: block layout, scoring, candidate construction, results, obsolete parts
│   ├── PORTING.md         ★ MACRec component map + how/where to port MD-RQ-VAE + MFD + OPEN DECISIONS
│   └── LEGACY_MAP.md      every legacy path with status KEEP-AS-REFERENCE / SUPPORT / OBSOLETE
├── envs/
│   ├── SETUP.md           environment: how it was built, conflicts, fixes, smoke-test results
│   ├── requirements.lock  pip freeze of the working env
│   └── grid_overlay_piplist.txt   package list of the ORIGINAL GRID overlay (for comparison)
└── sbatch/
    ├── template.sbatch            ★ start every new GPU job from this
    ├── build_env_macrec.sbatch    (re)build the env overlay copy (CPU job)
    ├── smoke_tiger_instruments.sbatch / .sh   STEP-4 smoke test (TIGER-style, Instruments)
    ├── data_fetch_images.sbatch   raw Amazon-18 meta + main images (CPU)
    ├── features.sbatch            LLaMA-7B text + CLIP ViT-L/14 image features (GPU)
    ├── baselines/tokenizers.{sbatch,sh}   all baseline tokenizers for one dataset
    ├── baselines/baseline.sbatch + run_baseline.sh   one (dataset, baseline, seed) train + valid/test eval
    └── tools/                     sid_stats.py, summarize_run.py, check_features.py
```

Where to look for what:

| task | look at |
|---|---|
| implementing the port | `docs/PORTING.md`, then `docs/MDRQVAE.md` / `docs/MFD.md` |
| understanding an old file | `docs/LEGACY_MAP.md` |
| running anything | §4 and `sbatch/template.sbatch` |
| environment problems | `envs/SETUP.md` |

---

## 3. Key paths (outside the repo)

| what | path |
|---|---|
| this repo | `/scratch/yw8866/mmgen` |
| env overlay (copy, used by all jobs) | `/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3` |
| original GRID overlay (**never modify**) | `/scratch/yw8866/grid_env/overlay-25GB-500K.ext3` |
| Singularity image | `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
| run outputs (all generated data, checkpoints, results) | `/scratch/yw8866/mmgen_runs/` |
| job logs | `/scratch/yw8866/mmgen_runs/logs/` |
| W&B offline runs | `/scratch/yw8866/mmgen_runs/wandb/` |
| HF cache | `/scratch/yw8866/hf_cache` (LLaMA-7B in `hf_cache/macrec_plm`) |
| secrets (W&B key), **outside git** | `~/.config/mmgen/secrets.sh` (chmod 600) |
| smoke-test run dir | `/scratch/yw8866/mmgen_runs/smoke_tiger_Instruments/` |
| baseline data (processed + features + SID index files) | `/scratch/yw8866/mmgen_runs/data/<D>/` |
| raw Amazon-18 meta + downloaded images | `/scratch/yw8866/mmgen_runs/raw/amazon18/{Metadata,Images}` |
| baseline runs | `/scratch/yw8866/mmgen_runs/baselines/<D>/<baseline>/seed<S>/` (`run_summary.json`) and `.../<D>/tok/` |
| MQL4GRec reference clone (ships Instruments LLaMA/CLIP features + index files) | `/scratch/yw8866/tmp/ref/MQL4GRec` (@5ad9c27) |
| old GRID repo (source of `legacy/`) | `/scratch/yw8866/GRID` (contains an un-rotated W&B key) |
| old Tmall data, embeddings, dumps | `/scratch/yw8866/rec-tmall/` |
| old checkpoints / SIDs | `/scratch/yw8866/logs/{train,inference}/...` (see `PROVENANCE.md`) |
| upstream MACRec clone | `/scratch/yw8866/tmp/macrec_upstream` |

---

## 4. Setup and running

**Cluster:**
- Slurm + Singularity.
- GPU account `torch_pr_69_tandon_advanced`; CPU account `torch_pr_69_general`.
- **Default GPU is A100** (`--gres=gpu:a100:1`), which queues faster than H100. Override at submit
  time with `sbatch --gres=gpu:h100:1 ...`.
- **GPU-utilization policy:** HPC cancels jobs whose mean GPU utilization stays below about 60%
  for more than 2 h. One 13M-parameter T5 training uses only about 30% of a GPU (job 20617403 was
  cancelled at 32%). So **train with `sbatch/baselines/pack_train.sbatch`**, which runs several
  runs concurrently (5 runs: 97% utilization, job 20637308), and run evaluation as separate short
  jobs (`baseline.sbatch STAGE=eval`, under 1 h each).
- Do not run heavy Python on the login node: submit an sbatch.

**Environment (already built).** Python 3.10.20, conda env `py310`, torch 2.6.0+cu124,
transformers 4.47.0. Details and the rebuild procedure are in `envs/SETUP.md`.

```bash
# rebuild the env copy (only if lost/broken; never while a job has it mounted)
sbatch /scratch/yw8866/mmgen/sbatch/build_env_macrec.sbatch

# interactive use (read-only)
singularity exec --nv --bind /scratch/yw8866 \
  --overlay /scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh && conda activate py310 && cd /scratch/yw8866/mmgen/macrec && python ..."
```

**Required environment variables.** All are set in `sbatch/template.sbatch`:
- `USE_TF=0`: the overlay has TensorFlow + Keras 3, which crashes transformers.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`: torch 2.6 `torch.load` default vs MACRec's checkpoints.
- `HF_HOME=/scratch/yw8866/hf_cache`.
- `TMPDIR` under `/scratch/yw8866/tmp/`.
- The W&B key, sourced from `~/.config/mmgen/secrets.sh`. If absent, `WANDB_MODE=offline`.

**New job:**
1. `cp sbatch/template.sbatch sbatch/<name>.sbatch`.
2. Edit the job name, time, and the payload. For long payloads, use a separate `.sh` file as the
   smoke test does.
3. `sbatch sbatch/<name>.sbatch`.
4. The log goes to `/scratch/yw8866/mmgen_runs/logs/<name>_<jobid>.out`.

**Smoke test / TIGER-style baseline:**
```bash
sbatch sbatch/smoke_tiger_instruments.sbatch                                     # 5 epochs (default)
sbatch --export=ALL,T5_EPOCHS=50 sbatch/smoke_tiger_instruments.sbatch           # 50 epochs
```
The pipeline is: LLaMA-7B text embedding → text-only RQ-VAE (4×256) → MACRec SID assignment and
collision fix → T5 `seqrec` → trie beam search (20 beams). Steps whose outputs already exist in
the run dir are skipped. Set `RUN=` for a fresh run dir.

**Baselines (§4b):**
```bash
sbatch sbatch/data_fetch_images.sbatch                 # meta + images (CPU)
sbatch sbatch/features.sbatch                          # LLaMA + CLIP features
sbatch --export=ALL,D=Instruments sbatch/baselines/tokenizers.sbatch
sbatch --export=ALL,D="Arts Games" sbatch/baselines/kmeans.sbatch     # MACRec k-means on CPU FIRST (GPU would sit idle)
sbatch --dependency=afterok:<kmeans job> --export=ALL,D="Arts Games",ONLY="mql macrecpaper" sbatch/baselines/tokenizers.sbatch   # datasets run concurrently
sbatch --export=ALL,RUNS="Instruments:mql:43 Instruments:macrecpaper:43" sbatch/baselines/pack_train.sbatch   # packed training
sbatch --export=ALL,RUNS="Instruments:mql:43 Instruments:macrecpaper:43" sbatch/baselines/pack_eval.sbatch    # sequential eval
sbatch --export=ALL,D=Instruments,B=mql,SEED=42 sbatch/baselines/baseline.sbatch   # B in tiger_text|tiger_image|mql|macrec|macrecpaper
```
- Fine-tuning settings live in `sbatch/baselines/run_baseline.sh`; each run writes `config.json`
  with the git commit and job id.
- Effective batch = per-device × 2 GPUs × grad-accum 2, as in both repos' launchers, emulated on
  1 GPU with grad-accum 4.

---

## 5. MACRec pipeline in one paragraph

Text embeddings come from `data_process/amazon_text_emb.py` (LLaMA-7B, title and description,
mean-pooled → `<D>.emb-llama-td.npy`). Image embeddings come from `clip_feature.py` (CLIP
ViT-L/14 → `<D>.emb-ViT-L-14.npy`), which **needs raw Amazon-18 meta with image URLs, not in the
repo**. Next, `data/kmeans.py` makes pseudo-classes, `cross_index/main.py` trains the tokenizer
(two RQ stacks, text and image, 4×256, optional cross-modal contrast), and
`cross_index/generate_indices_distance.py` assigns SIDs. That script resolves collisions by
re-assigning the last levels to the next-nearest codes, and writes
`<D>.index_{lemb,vitemb}_<name>.json` with token strings `<a_k><b_k>…` (text) or `<A_k>…`
(image). Then `finetune_contrastive.py` trains T5 on tokenized SID strings (tasks `seqrec`,
`seqimage`, cross-modal tasks), and `test_ddp_save.py` runs trie-constrained beam search with
hit@{1,5,10} and ndcg@{5,10}. Finally `ensemble.py` merges the text-SID and image-SID runs into
the MACRec number. Splits are leave-one-out: train = prefixes of `items[:-2]`, valid
`items[-2]`, test `items[-1]`, history ≤ 20. Full component map with line numbers:
`docs/PORTING.md` §1.

---

## 6. MACRec gotchas (all verified; work around without editing `macrec/`)

| issue | workaround |
|---|---|
| `--use_cross_rq` is `type=bool`: `--use_cross_rq False` **enables** it (`cross_index/main.py:41`) | omit the flag to disable |
| `generate_indices_distance.py` crashes for checkpoints trained without cross-RQ (expects per-level distance lists) | text-only tokenizer: `--use_cross_rq True --begin_cross_layer 4` (with 4 levels the cross branch never fires, so the model is a plain RQ) |
| under numpy ≥ 2 the collision fix silently leaves duplicate SIDs (`np.int64` repr in str-based de-dup); symptom `check: False` | run it with `np.set_printoptions(legacy='1.25')` via `runpy` (see `sbatch/smoke_tiger_instruments.sh`) |
| `main.py` parses `--{text,image,recon}_contrast_weight` but never passes them to `CrossRQVAE` | defaults 1.0 / 1.0 / 0.001 always apply |
| `cross_index/main.py` always loads both modalities; `data.py:47-50` always opens both index files | text-only: pass the text `.npy` as `--image_data_path`, and the text index as `--image_index_file` |
| `cross_index/scripts/gen.sh` calls a non-existent script | use `gen_code.sh` |
| `data/kmeans.py` hard-codes `dataset="Games"` | edit a copy, not the original |
| k-means (sklearn, CPU) inside a GPU job leaves the GPU idle; Arts+Games took >2 h and the cluster cancelled job 20806460 | run `sbatch/baselines/kmeans.sbatch` on the CPU account first; `tokenizers.sh` skips existing k-means files |
| `amazon_text_emb.py` sets `local_files_only=True` whenever `--model_cache_dir` exists | pre-download (`snapshot_download`) into that dir first |
| HF Trainer `report_to=None` means *all*, so it logs to W&B and TensorBoard; upstream `finetune_full.sh` sets `WANDB_MODE=disabled` | our jobs keep W&B on (offline when there is no key) |
| `evaluation_strategy=` is deprecated | works in transformers 4.47; **do not upgrade transformers past 4.47** |
| "missing keys … lm_head.weight" when the best model reloads | benign (T5 tied embeddings) |
| `OSError: Device or resource busy: .nfs…` at the end of DataLoader runs | benign NFS temp cleanup |

---

## 7. Results so far (Amazon)

| run | dataset | hit@1 | hit@5 | hit@10 | ndcg@5 | ndcg@10 | job / where |
|---|---|---|---|---|---|---|---|
| TIGER-style smoke, 5 epochs | Instruments | 0.0001 | 0.0011 | 0.0029 | 0.0006 | 0.0012 | 20532665 (6,232/6,250 unique SIDs, numpy-2 bug) |
| TIGER-style smoke, 50 epochs | Instruments | 0.0127 | 0.0396 | 0.0615 | 0.0268 | 0.0338 | 20539125, `mmgen_runs/smoke_tiger_Instruments/t5_seqrec_50ep/` |

These are pipeline checks, not tuned baselines: 50 epochs vs MACRec's 200 with early stopping,
on 1 GPU instead of 2. Every future result goes here and in `docs/` as appropriate.

**Baselines, Instruments, seed 42, TEST** (HR@10 / NDCG@10; C/D = merged list; pack_train 20637308,
pack_eval 20769295 + 20802837; runs in `mmgen_runs/baselines/Instruments/<B>/seed42/run_summary.json`):

| baseline | test HR@10 | test NDCG@10 | target | gap | status |
|---|---|---|---|---|---|
| A TIGER-text | 0.1131 | 0.0859 | 0.1221 / 0.0950 | −7.4% / −9.6% | miss |
| B TIGER-image | 0.1120 | 0.0867 | none published | n/a | n/a |
| C MQL4GRec | 0.1301 | 0.0998 | 0.1291 / 0.0997 | +0.8% / +0.1% | match |
| D MACRec, paper setting (`macrecpaper`) | 0.1321 | 0.1021 | 0.1363 / 0.1046 | −3.1% / −2.4% | match |
| D MACRec, repo as shipped (`macrec`) | 0.1221 | 0.0949 | 0.1363 / 0.1046 | −10.4% / −9.3% | miss |

Test beam search: 5 decoder steps × 20 beams per user, ≈21.7 ms/user on an H200 (uncontended).
The valid seqrec cost files come from a contended concurrent job and are not usable for cost.

---

## 8. Legacy facts worth knowing (Tmall; details in `docs/`)

- **MD-RQ-VAE (Tmall):** SigLIP2 1152-d, latent 512, codebooks 2048/1024/1024, EMA VQ.
  - Verified properties: additivity 0.9918 / 0.9885; normalized separation ratios text 1.105,
    image 0.745; shared balance +0.0798; CMI(t;i|f) ≈ 1.7% of H(t|f).
  - Collisions (36.2%) were left in.
- **MFD (Tmall):**
  - The best decoder was Run-C (PMQ + Query-f).
  - The "+27% t2" claim in `legacy/grid/AGENTS.md` is **invalid** (chain beam-collapse bug).
    The true t2@10 lead is about +0.003–0.006.
  - End to end, matched at 808 decoder states over 5 seeds: MFD 0.2152 vs chain 0.2135.
  - MFD gains coverage cheaply but orders worse inside the pool.
- **Licensing:** legacy code is Snap Inc. non-commercial research-only. MACRec ships no license.
  **Keep any git remote private** (§11).

---

## 9. Conventions and rules

- **`macrec/` upstream files stay unmodified.** Put new code in new files next to MACRec's
  (planned names in `docs/PORTING.md` §2.3), and subclass when behavior must change. The first
  git commit is pristine upstream, so `git diff f3c9707 -- macrec/` shows every local change.
- **`legacy/grid/` is read-only and never imported.** Cite it as `legacy/grid/<path>:<line>`.
- **No artifacts in git.** `.gitignore` blocks `*.pt *.pth *.ckpt *.pkl *.npy *.safetensors
  *.bin *.tfrecord* *.ext3 *.out wandb/ logs/ outputs/` plus generated MACRec embeddings and
  index files. Generated data goes to `/scratch/yw8866/mmgen_runs/`.
- **No secrets in git.** Before every commit:
  `git diff --cached | grep -nE 'wandb_v1_|WANDB_API_KEY=[0-9a-f]{40}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_'`
  must print nothing. Keys live only in `~/.config/mmgen/secrets.sh`. Never use `set -x` in jobs.
- **W&B stays enabled.** Do not set `WANDB_MODE=disabled`. Offline mode is acceptable when there
  is no key.
- **Overlays.** Jobs always mount the overlay `:ro`. Writes need `--fakeroot`, and only when no
  job is using it. Never touch the original GRID overlay.
- **Comparisons** (lessons from Tmall):
  - match the decoder-state budget between MFD and chain;
  - use paired bootstrap over the same test users;
  - use ≥ 3 seeds;
  - report coverage@pool and `P(rank<10 | covered)` alongside recall.
- Read SID widths and sizes from the files; never assume them.

---

## 10. Git history

| commit | content |
|---|---|
| `f3c9707` | pristine MACRec upstream `7a48e74` |
| `232a611` | legacy GRID reference, docs, env spec, job scripts |
| `7bef831` | smoke-test fixes and results; MACRec bugs documented |
| `f5144fc` | AGENTS.md, CLAUDE.md |
| `747c6c0`…`6202cc8` | baseline entry points, PlainT5 fix, packed train / sequential eval jobs |
| (next) | Instruments seed-42 results; tokenizer ONLY filter and multi-dataset jobs |

---

## 11. Remote

None configured yet. The repo must be **private**, for licensing reasons (§8).
Once configured, record the URL here.

---

## 12. Keeping this file current

Update AGENTS.md in the **same commit** as any change that:
- adds, moves or removes a top-level file, a module, or an sbatch;
- changes a setup step, an env variable, a package version, or a path;
- discovers a new gotcha (§6);
- produces a result worth comparing (§7);
- resolves an open decision from `docs/PORTING.md` §6 (record the choice here and there).

Also update the "Last updated" date and the status line in §1.
