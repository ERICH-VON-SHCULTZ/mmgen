# AGENTS.md: mmgen

Read this first. It is the single entry point for any agent or new contributor working in this repo.
**Keep it current: every change that adds or moves a file, changes a setup step, a path, a convention,
or a result must update this file in the same commit** (see §12).

Last updated: 2026-11-02 (MFD with MQL4GRec's task mix, versions A/B/C, `docs/MFD_REPORT.md` §13; codebook-geometry-aware training in progress, §14).

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

**Current status (2026-10-31):**
- Repo, docs and environment are done.
- The TIGER smoke test on Instruments passes.
- **Baseline reproduction done** (report: `docs/BASELINE_REPORT.md`, numbers in §7).
  MQL4GRec matches on all three datasets (3 seeds). MACRec (paper setting) matches on Instruments,
  misses Arts/Games by 4.5–6.3% and ties MQL4GRec everywhere. TIGER-text and MACRec-as-shipped miss
  on Instruments and were not extended. Open questions for the user: report §9.
- **Baseline decisions (2026-10-26, report §0):** our reproduced numbers are the reference
  (published ones listed alongside); `macrecpaper` is the official MACRec; TIGER is accepted as our
  reading of the paper; data and features are frozen (Games banner images kept).
- **Done:** TIGER A/B on Arts/Games (seeds 42–44) and Instruments seeds 43/44 (in `docs/BASELINE_REPORT.md`).
- **MD-RQ-VAE built (2026-10-27)**: `macrec/cross_index/models/mdrqvae.py`; final configs a05
  (1+2+2) and a06 (1+3+3), attempt log `docs/MDRQVAE_LOG.md`.
- **Official MD-RQ-VAE SID (2026-10-27):** the distance-reassigned index with no trailing token
  (`<D>.index_md122_a05_ep500_reassign.json`, 5 tokens; `<D>.index_md133_a06_ep500_reassign.json`,
  7 tokens). Tokenizer checkpoint = final epoch (500), no selection (`final.pth`). The
  `*_a0X_reassign.json` files (no `ep500`) are the superseded best-collision sets, kept as controls.
  Rule and the six frozen v1 sets (path + SHA-256) in `docs/MDRQVAE.md` §A2; all six are
  final-epoch sets (Games 1+3+3 accepted at 2.02% excl. CMI). **v1 is LOCKED (2026-10-28)**:
  never overwrite; changes become v2. Check the data-dir index against `index.sha256` before runs.
  Recipe to train the SIDs yourself: `docs/MDRQVAE.md` A3. A retrained set has the same quality but
  not identical codes (expected and acceptable; acceptance list in A3.7). Reported numbers used the
  locked v1 files. Chain and MFD runs must use the same set. The `index_md122_a05.json` /
  `index_md133_a06.json` files (extra `<x_k>` token) are kept only as the comparison point.
- **Instruments Step 3 done (2026-10-28), report `docs/MDRQVAE_REPORT.md`:** chain decoder on the
  official SID, seed 42, test HR@10 / NDCG@10: MD 1+2+2 0.1181 / 0.0932 (better config; 6 decoder
  passes), 1+3+3 0.1161 / 0.0916; E-norm 0.1055 / 0.0822; E-raw 0.1132 / 0.0870. Token vs
  reassignment: within seed noise, one decoder pass fewer.
- **Arts/Games chain decoder done (2026-10-30), report §5.1:** seed 42, test HR@10 / NDCG@10:
  Arts MD 1+2+2 0.1059 / 0.0739 (1+3+3 0.1027 / 0.0726) vs A 3-seed 0.0986 / 0.0661, E-norm
  0.0825 / 0.0520; Games MD 1+3+3 0.0899 / 0.0475 (1+2+2 0.0885 / 0.0468) vs A 0.0885 / 0.0471,
  E-norm 0.0843 / 0.0454. MD > E-norm everywhere; MD ≥ A (Arts within A's seed spread, Games tie);
  C/D single lists stay above MD. Evals 21382066–21382069 all ran on H200 (cost comparable).
- **E-raw done on Arts/Games (2026-10-30):** 0.0902 / 0.0565 and 0.0859 / 0.0448; MD beats E-raw and
  E-norm on every dataset (report §5.1).
- **C/D-gap controls (done 2026-10-31, seed 42, Instruments + Arts; report §5.2):** (1) C's MQL4GRec SIDs
  trained seqrec-only with TIGER settings, one run per modality (`mqltext`, `mqlimage`; train
  21422951, evals 21422953–56); (2) MD 1+2+2 v1 SID with C-style multi-task training, one SID
  sequence (`mdmt122`, `macrec/md_mt_data.py`: seqrec + mdshared/mdtext/mdimage + mdtrans; train
  21422952, evals 21422957–58). Report optimizer steps and training examples next to metrics
  (`sbatch/tools/train_examples.py`). Why: C/D's single lists come from multi-task models, so they
  are not single-modality controls; decoding is identical for all runs. **Result:** C's SIDs alone
  are 0.003–0.010 HR@10 below TIGER's; C's multi-task recipe adds +0.019 to +0.031 (the whole C/D
  lead); copied onto MD's single sequence it does not help (−0.005 HR@10 on both datasets).
- **MFD stage 1 done (2026-10-31), report `docs/MFD_REPORT.md`:** two-branch decoder on Instruments 1+2+2 v1,
  seed 42, same recipe as the MD chain-decoder control (train 21515330, eval 21515331, paired 21528838 →
  `mmgen_runs/mfd/Instruments/mfd122_stage1/seed42/`). All sanity checks pass. Test HR@10 / NDCG@10:
  joint 0.1177 / 0.0923, C-style merge 0.1183 / 0.0926, chain 0.1181 / 0.0932; paired 95% CI of
  joint − chain HR@10 [−0.0036, +0.0026]: a tie, with 3 passes and 6.7 ms/user vs 6 passes and 25.3 ms/user
  (H200). Limit = coverage via f prediction: the target f is outside the top 20 for 65% of users in both
  models (f top-20 0.346 vs 0.352); on the same covered users MFD ranks 0.013 HR@10 below the chain but
  covers more users.
- **Seed policy:** seed 42 only while iterating. The final paper comparisons (MD-RQ-VAE vs E, MFD
  vs chain decoding) need 3 seeds (42/43/44), at least on Arts: TIGER-text's Arts seed spread
  (HR@10 0.0931–0.1061) is larger than the differences we expect to measure.
- **MFD stage 1** lives in `macrec/mfd_*.py` (new files); report `docs/MFD_REPORT.md`.
- **MFD Parts A–C done (2026-10-31, seed 42), `docs/MFD_REPORT.md` §7–§12** (paired CIs: `mmgen_runs/mfd/paired_parts.json`).
  Test HR@10 / NDCG@10:
  - A: MFD 1+3+3 0.1196 / 0.0931 (4 passes, 21.4 ms/user) vs 1+3+3 chain 0.1161 / 0.0916 (8, 32.0).
  - B (chosen on valid): 1+2+2 K_f=100, w=1/0.5/0.5 → 0.1210 / 0.0936 (29 ms/user); 1+3+3 K_f=256, w=1/1/0.5 →
    0.1221 / 0.0938 (169 ms/user). The gain is mostly the weights (trust f more), not K_f.
  - C: multi-view history augmentation helps MFD (+0.003 HR@10) not the chain (+0.0007 / +0.0001). Best MFD (aug +
    Part B setting) vs chain: Instruments 0.1237 vs 0.1181, +0.0056 [+0.0025, +0.0088]; Arts 0.1066 vs 0.1059, tie.
    Plain Arts MFD (K_f=20) is below the chain: 0.0990 vs 0.1059.
  - Limit: f prediction (top-1 ≈ 0.10, top-20 0.35–0.41 for every model, ±0.01 with augmentation); still −0.007
    (Instruments) / −0.017 (Arts) HR@10 below D's best single list. Next: seeds 43/44, cost-capped K_f rule.
- **MFD with C's task mix done (2026-11-02, Instruments, seed 42), `docs/MFD_REPORT.md` §13:** A = seqrec + seqimage +
  fullseqrec (full-view → both branches; code id `mt_fusionseqrec`, NOT C's fusionseqrec), B = A + item2image/image2item,
  C = A + crossseqrec (C's fusionseqrec, the cross-modal coin flip). Test HR@10 at K_f=100, w=1/.5/.5: A 0.1227, B 0.1202
  (CI < 0 vs mv), C 0.1245 vs mfd122_mv 0.1237 (ties). f prediction unchanged. Arts rule (valid > 0.1560) not triggered.
- **Codebook-geometry-aware T5 (started 2026-11-01), `docs/MFD_REPORT.md` §14:** SID code embeddings = P_level · codebook
  vector + delta (tied head), shuffled-codebook control, soft labels for f. Chain runs done (`mmgen_runs/geo/<D>/chain122_<VAR>/`);
  MFD + chosen variant (AB2) and Arts in progress.
- **MFD paused (2026-10-31).** **Plug-in done (2026-11-01, seed 42), report `docs/PLUGIN_REPORT.md`:** the v1 MD SID split
  into a text view [f t..] and an image view [f i..] used as C's two modality SIDs, trained with C's exact recipe
  (`run_baseline.sh` B=plug122|plug122s|plug133|plug133s; `s` = shared-f token). Views made unique with MQL4GRec's
  distance reassignment (`macrec/cross_index/plugin_views.py`; provenance + v1 hash in `mmgen_runs/plugin/<D>/<cfg>/views.json`).
  Config gate before training: `sbatch/plugin/config_check.sbatch` (dry runs, `tools/plugin_config_diff.py`) → only
  index files + vocabulary differ from C seed 42 (`mmgen_runs/plugin/config_diff.json`).
  - **Result: MD loses to C under C's recipe.** Merged test HR@10 vs C seed 42 (paired, all CIs < 0):
    Instruments 1+2+2 0.1246 vs 0.1301 (−0.0055 [−0.0088, −0.0023]), 1+3+3 0.1217; Arts 1+2+2 0.1172 vs 0.1267
    (−0.0095), 1+3+3 0.1162. All below C's worst seed. Best single list also loses (−0.003 to −0.007).
  - C's recipe still lifts MD well above its own chain decoder (Instruments text 0.1227 vs 0.1181, Arts 0.1143 vs 0.1059),
    but the merge gains MD little (two views share f → correlated lists). Shared-f hurts at 1+2+2 (merged −0.0045 vs main,
    CI < 0), no effect at 1+3+3. f first-token top-20 0.36–0.43 over 256 codes vs C's first level 0.59–0.83 over 55–97 codes.

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
│   ├── finetune_ext.py    finetune_contrastive.train + --model_class plain|contrastive + train stats (+ examples per task)
│   ├── md_mt_data.py      C-style multi-task datasets on one MD SID (mdshared, mdtext, mdimage, mdtrans)
│   ├── mfd_model.py, mfd_data.py, finetune_mfd.py, mfd_eval.py   MFD: two-branch decoder, data/catalog (any 1+L+L SID), training entry, exact-scoring eval (1+L passes) + sanity checks
│   ├── mfd_sweep.py       MFD wider search: K_f x score-weight sweep on VALID, chosen setting applied to test
│   ├── mfd_mt_data.py, mfd_mt_check.py, mfd_dryrun.py   MFD with C's task mix (§13): tasks mt_* (mt_fusionseqrec = fullseqrec, mt_crossseqrec = C's fusionseqrec), input/target checks, dry run
│   ├── geo_embed.py, finetune_geo.py, geo_check.py, geo_dryrun.py, geo_diag.py   codebook-geometry-aware T5 (§14): embeddings P·c+delta / shuffled control / soft f labels, bake to plain T5; entry; sanity checks; dry run; diagnostics
│   ├── mv_aug_data.py     multi-view history augmentation for the chain decoder (task seqrec_mv; MFD: mfd_data.MVMFDDataset)
│   ├── config_dryrun.py   finetune_ext.py with the exact command, stopped before trainer.train(): dumps resolved args, TrainingArguments, model config
│   ├── first_token_acc.py first-decoding-step accuracy (target's first token top-1 / top-20) for seqrec + seqimage lists
│   ├── cross_index/plugin_views.py   plug-in: MD v1 SID -> text/image views, MQL4GRec reassignment per view, main + shared-f index files
│   ├── eval_split.py      test_ddp_save on --split valid|test + inference cost (decoder steps, ms/user)
│   ├── cross_index/main_single.py, generate_indices_single.py   MQL4GRec single-modality tokenizer (verbatim @5ad9c27)
│   ├── cross_index/generate_indices_tiger.py   TIGER SIDs: 3 RQ levels + appended collision index
│   ├── cross_index/main_cross_weights.py       main.py with the contrast weights actually applied
│   ├── cross_index/models/mdrqvae.py, main_mdrqvae.py   MD-RQ-VAE model + trainer
│   ├── cross_index/generate_indices_md.py      MD SIDs (old: + <x_k> token on collision)
│   ├── cross_index/reassign_md.py              ★ official MD SIDs: distance reassignment, no extra token
│   ├── cross_index/md_diagnostics.py           diagnostics 1-7 (recon, util, collision, CMI+null, AMI, separation, additivity)
│   ├── cross_index/main_single_final.py        main_single.py unchanged + also saves final_model.pth (E final-epoch rule)
│   ├── data/kmeans_dataset.py                  data/kmeans.py parameterised (K=512 pseudo-labels)
│   ├── data_process/fuse_features.py           baseline E input: concat(L2-normed | raw) text + image features
│   └── data_process/fetch_item_images.py, clip_feature_ext.py   main image download; CLIP tolerant to missing images
├── legacy/grid/           READ-ONLY reference: old GRID fork (Tmall MD-RQ-VAE + MFD). Never import.
│   ├── src/               md_rqvae.py, mfd_generation_model.py, tiger_generation_model.py, eval, data
│   ├── configs/experiment/   Hydra configs of the key runs
│   ├── sbatchfiles/       launchers of every old experiment (W&B key REDACTED)
│   ├── rec-tmall/         analysis scripts (+ GOAL.md research log, results/ small measured outputs)
│   ├── AGENTS.md          the OLD project's handbook (not this one)
│   └── README_LEGACY.md
├── docs/
│   ├── MDRQVAE.md         ★ tokenizer: Part A current Amazon implementation + official SID (v1); Part B Tmall reference
│   ├── MFD.md             old decoder: block layout, scoring, candidate construction, results, obsolete parts
│   ├── BASELINE_REPORT.md ★ baseline reproduction: results, diagnostics, deviations, open questions
│   ├── MDRQVAE_REPORT.md  ★ MD-RQ-VAE on Amazon: implementation, diagnostics, chain decoder vs baselines, official SID
│   ├── MDRQVAE_LOG.md     every MD-RQ-VAE attempt and control, in order (Part 2 log)
│   ├── MFD_REPORT.md      ★ MFD stage 1 (two-branch decoder) vs the chain decoder on Instruments
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
    ├── baselines/tokenizers.{sbatch,sh}   baseline tokenizers (D="Arts Games" concurrent, ONLY= filter)
    ├── baselines/kmeans.sbatch    MACRec k-means pseudo-labels on CPU (run before tokenizers)
    ├── baselines/baseline.sbatch + run_baseline.sh   one (dataset, baseline, seed) train + valid/test eval
    ├── baselines/pack_train.sbatch   several trainings concurrently on one GPU (keeps utilisation >95%)
    ├── baselines/pack_eval.sbatch    sequential eval; ONE run per job (<2 h); use --partition=h200_tandon for cost comparable to the baselines
    ├── baselines/e_final_tok.sbatch  E-norm tokenizer rerun, indexed from the final epoch
    └── tools/                     sid_stats.py, summarize_run.py, check_features.py,
                                   collect_baselines.py (all runs -> baselines/baselines_all.json),
                                   ami_first_level.py (E vs A/B first-level AMI),
                                   rank_metrics.py (independent metric/merge check),
                                   freeze_md_sids.py (versioned SID sets; --lock makes them read-only),
                                   cmi_shared_image.py (first-level CMI without items sharing an image file),
                                   train_examples.py (examples per epoch/task, optimizer steps, examples seen)
    ├── md/tok_attempt.sbatch      MD-RQ-VAE tokenizer attempt: train -> SIDs -> diagnostics
    ├── md/reassign.sbatch         reassignment + diagnostics on the new index
    ├── mfd/train_mfd.sbatch, mfd/eval_mfd.sbatch   MFD stage 1 train (sanity on untrained model first) / eval on H200
    ├── mfd/pack_mfd_train.sbatch + run_mfd_train.sh   MFD (any CFG, AUG) / chain (mdmv122) trainings packed on one GPU, frozen code
    ├── mfd/sweep_mfd.sbatch       Part B: sweep on valid -> chosen.json -> that one setting on test (H200)
    ├── mfd/paired_mfd.sbatch      MFD vs chain per user (tools/mfd_paired.py: paired bootstrap, same-user coverage/ranking); tools/mfd_table.py = comparison table
    ├── mfd/paired_parts.sbatch    Parts A–C paired bootstrap from saved per-user ranks, CPU (tools/mfd_paired_parts.py); tools/mfd_parts_table.py = Parts A–C tables
    ├── plugin/views.sbatch        plug-in view index files (CPU); plugin/config_check.sbatch = dry runs + config diff vs C (gate)
    ├── plugin/pack_plugin_train.sbatch   plug-in trainings packed on one GPU; refuses to train unless tools/plugin_gate.py passes
    ├── plugin/first_token.sbatch  first-token accuracy for C and plug-in runs
    ├── plugin/paired.sbatch       plug-in vs C paired bootstrap (tools/plugin_paired.py, REF= picks another reference run) + tables (tools/plugin_table.py); CPU
    ├── plugin/paired_sharedf.sbatch   shared-f vs main plug-in, paired (CPU)
    ├── mfd/mt_gate.sbatch, eval_mt.sbatch, mt_paired.sbatch   task-mix gate (dry runs + task checks + config diff), eval (i)/(ii)/(iii), paired CIs (tools/mfd_mt_*.py)
    ├── geo/run_geo_train.sh, geo_gate.sbatch, eval_geo.sbatch   geometry runs (KIND chain|mfd, VAR A|C|AB1|AB2; pack entry geo:<D>:<kind>:<var>), gate (sanity 1-3), eval + diagnostics; tools/geo_config_diff.py, geo_chain_paired.py
    ├── md/repro_v1.sbatch         reproduction check of the v1 sets (MODE=full retrain | reassign = stage 2 on the locked tokenizer)
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
| MD-RQ-VAE tokenizer runs | `/scratch/yw8866/mmgen_runs/md/<D>/tok/<attempt>/` |
| frozen MD SID sets (official) | `/scratch/yw8866/mmgen_runs/md/sids/v1/<D>/1+X+X/` (see `docs/MDRQVAE.md` §A2) |
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
sbatch --export=ALL,RUNS="Instruments:mql:43" sbatch/baselines/pack_eval.sbatch    # ONE run per eval job (<2 h, see §6)
sbatch --export=ALL,D=Instruments,B=mql,SEED=42 sbatch/baselines/baseline.sbatch   # B in tiger_text|tiger_image|mql|macrec|macrecpaper
```
- Fine-tuning settings live in `sbatch/baselines/run_baseline.sh`. Before training, each run writes
  `run_config.json` (shell settings + every resolved argument + T5 config + GPU, via
  `finetune_ext.py --run_config_file`). **`config.json` in a run dir is the HF model config** and is
  overwritten on every save; runs before 2026-10-26 have their settings only in `config_eval.json`
  and `training_args.bin`.
- Baselines: `tiger_text` (A), `tiger_image` (B), `tiger_fnorm` / `tiger_fraw` (E early fusion),
  `mql` (C), `macrecpaper` (D, official), `macrec` (D as shipped, deviation). Controls: `mqltext` /
  `mqlimage` (C's SIDs, seqrec only), `mdmt122` (MD 1+2+2 + C-style multi-task), `mdmv122` (MD 1+2+2 +
  multi-view history augmentation), `plug122|plug122s|plug133|plug133s` (plug-in, C's recipe). `DRYRUN=1` runs the exact
  training command through `config_dryrun.py` into `mmgen_runs/plugin/dryrun/` (no training). `run_baseline.sh` uses `MACREC_DIR` (frozen code copy) when set;
  `pack_eval.sbatch` and `pack_mfd_train.sbatch` set it.
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
| sequential beam-search eval keeps the GPU ~12% busy; job 20819917 (4 runs) was cancelled at 2 h 09 | `pack_eval.sbatch` with ONE C/D run per job (~40 min); submit several jobs |
| bash reads a script while it runs: editing `run_baseline.sh` during a job broke job 21047067 (syntax error after its last eval) | `pack_*.sbatch` / `baseline.sbatch` now run a snapshot copied to `$TMPDIR` at job start |
| k-means (sklearn, CPU) inside a GPU job leaves the GPU idle; Arts+Games took >2 h and the cluster cancelled job 20806460 | run `sbatch/baselines/kmeans.sbatch` on the CPU account first; `tokenizers.sh` skips existing k-means files |
| `scontrol update` / `hold` / `nice` on our pending jobs fails with "Unspecified error" | job order can only be set at submit time (`--dependency=after:<job>`); otherwise cancel and resubmit |
| `tokenizers.sbatch` asks for `gpu:a100` only; job 21251957 waited 17 h for an A100 | submit with `--partition=a100_tandon,h100_tandon,h200_tandon --gres=gpu:1` |
| HF schedules `epochs × floor(batches / grad_accum)` steps, so runs can end at a fractional epoch (Arts 1+3+3: 21,000 steps = epoch 198.1) | normal; the cosine schedule has finished |
| MD-RQ-VAE tokenizer training is not bit-deterministic (GPU kernels, sklearn k-means init): a rerun gives SIDs of the same quality but different codes | expected; check it with `docs/MDRQVAE.md` A3.7; use your own NAME/TAG so the official files are not overwritten; for exact v1 use the locked files |
| HF `resize_token_embeddings` (mean-resizing) gives every added token (all SID codes) the SAME embedding row, so an untrained model cannot distinguish codes | sanity checks on untrained models must re-init the added rows randomly (`mfd_eval.py --random_init`); trained models are unaffected |
| C's seed-42 train.log prints `deepspeed='./config/ds_z3_bf16.json'`, but `finetune_contrastive.train()` sets `args.deepspeed = None` right after that print | not a config difference; compare printed args with printed args (`plugin_config_diff.py`) |
| the chain eval's `save_seqrec_20.json` stores users in sampler order, not dataset order | align by `all_users` (= dataset index), as `sbatch/tools/mfd_paired.py` does |
| `pack_train.sbatch` (24 CPUs, long time limit) is rejected on h200_tandon ("partition not valid for this job") | train with `--partition=a100_tandon,h100_tandon --gres=gpu:1`; H200 only for evals |
| ms/user is GPU-dependent; every baseline cost file is from an H200 | evaluate with `--partition=h200_tandon` (queue reason `QOSGrpGRES` = account H200s busy) |
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

**Final baseline table (TEST, mean over seeds 42/43/44, HR@10 / NDCG@10; C/D merged list).**
Full tables, std, single-list results and cost: `docs/BASELINE_REPORT.md`;
raw: `mmgen_runs/baselines/baselines_all.json` (`python3 sbatch/tools/collect_baselines.py`).

| baseline | Instruments | Arts | Games |
|---|---|---|---|
| A TIGER-text | 0.1154 / 0.0876 (−5.5% / −7.8%, miss) | 0.0986 / 0.0661 (−15.5% / −18.0%, miss) | 0.0885 / 0.0471 (+3.3% / +4.1%) |
| B TIGER-image (no target) | 0.1139 / 0.0883 | 0.0957 / 0.0657 | 0.0782 / 0.0415 |
| C MQL4GRec | 0.1310 / 0.1006 (+1.4% / +0.9%) | 0.1262 / 0.0898 (+0.6% / 0.0%) | 0.1007 / 0.0537 (0.0% / −0.2%) |
| D MACRec paper setting | 0.1310 / 0.1011 (−3.9% / −3.4%) | 0.1255 / 0.0893 (−5.6% / −6.3%, miss) | 0.1020 / 0.0539 (−5.4% / −4.5%, miss) |
| D MACRec as shipped (1 seed) | 0.1221 / 0.0949 (−10.4% / −9.3%, miss) | — | — |

**MD-RQ-VAE chain decoder and E (seed 42, TEST, one list; HR@10 / NDCG@10; decoder passes /
user).** Full tables, controls and cost: `docs/MDRQVAE_REPORT.md` §5, §5.1, §7.3.

| method | passes | Instruments | Arts | Games |
|---|---|---|---|---|
| MD 1+2+2 (official SID) | 6 | 0.1181 / 0.0932 | 0.1059 / 0.0739 | 0.0885 / 0.0468 |
| MD 1+3+3 (official SID) | 8 | 0.1161 / 0.0916 | 0.1027 / 0.0726 | 0.0899 / 0.0475 |
| E-norm (rescaled) | 5 | 0.1055 / 0.0822 | 0.0825 / 0.0520 | 0.0843 / 0.0454 |
| E-raw | 5 | 0.1132 / 0.0870 | 0.0902 / 0.0565 | 0.0859 / 0.0448 |
| C's SIDs, seqrec only (text / image) | 5 | 0.1036 / 0.0832, 0.1081 / 0.0853 | 0.0880 / 0.0603, 0.0921 / 0.0625 | — |
| MD 1+2+2, C-style multi-task | 6 | 0.1126 / 0.0894 | 0.1007 / 0.0726 | — |
| MFD stage 1, joint / C-style merge (3 passes, 6.7 ms/user) | 3 | 0.1177 / 0.0923, 0.1183 / 0.0926 | 0.0990 / 0.0685, 0.0980 / 0.0678 | — |
| MFD 1+2+2, Part B setting (K_f=100, w=1/0.5/0.5) | 3 | 0.1210 / 0.0936 | 0.1044 / 0.0705 | — |
| MD 1+2+2 chain + multi-view aug | 6 | 0.1188 / 0.0940 | 0.1061 / 0.0736 | — |
| MFD 1+2+2 + multi-view aug, K_f=20 / Part B setting | 3 | 0.1208 / 0.0948, 0.1237 / 0.0959 | 0.1023 / 0.0710, 0.1066 / 0.0729 | — |
| MFD 1+3+3, K_f=20 / own Part B setting (K_f=256) | 4 | 0.1196 / 0.0931, 0.1221 / 0.0938 | — | — |
| Plug-in 1+2+2 main in C's recipe: text / image / merged (2 × 4 passes) | 4 | 0.1227 / 0.0960, 0.1238 / 0.0966, 0.1246 / 0.0975 | 0.1143 / 0.0805, 0.1144 / 0.0807, 0.1172 / 0.0826 | — |
| Plug-in 1+3+3 main in C's recipe: text / image / merged (2 × 5 passes) | 5 | 0.1207 / 0.0938, 0.1208 / 0.0942, 0.1217 / 0.0948 | 0.1119 / 0.0822, 0.1143 / 0.0829, 0.1162 / 0.0845 | — |
| MFD 1+2+2 + C's task mix A / B / C (K_f=100, w=1/.5/.5) | 3 | 0.1227 / 0.0949, 0.1202 / 0.0933, 0.1245 / 0.0962 | — | — |
| Plug-in shared-f merged, 1+2+2 / 1+3+3 | 4 / 5 | 0.1201 / 0.0946, 0.1199 / 0.0934 | — | — |
| C seed 42 merged (paired reference for the plug-in; `docs/PLUGIN_REPORT.md`) | 5 | 0.1301 / 0.0998 | 0.1267 / 0.0897 | — |
| C / D, best single list (3 seeds) | 5 | 0.1287 / 0.0988, 0.1302 / 0.0997 | 0.1217 / 0.0871, 0.1218 / 0.0861 | 0.0970 / 0.0514, 0.0976 / 0.0513 |

Earlier per-seed notes:

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

Seeds 43/44 (test merged HR@10 / NDCG@10): MQL4GRec 0.1324 / 0.1013, 0.1304 / 0.1008;
MACRec paper setting seed 43 0.1304 / 0.1005, seed 44 0.1306 / 0.1007.

Data note: in Games, 2,287 items' first image URL is the same generic banner (`31BDm2VqflL.jpg`);
3,475 Games items (25.1%) share an identical image file with another item, which is why the
Games image tokenizers collide 25.8% before reassignment (Instruments 164 items, Arts 425).

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
| `84fc4d2`…`24ef25a` | Instruments results; tokenizer ONLY filter; CPU k-means job; one run per eval job |
| `bd4bb27`…`3e026d1` | MD-RQ-VAE model, attempts a01–a06, diagnostics, distance reassignment |
| `f6e5b9d`…`714968b` | official SID (reassigned), final-epoch checkpoint rule, v1 sets, CMI without shared images, E final-epoch tokenizer, TIGER A/B 3-seed report |
| `69b23fd`…`a7e6801` | Instruments chain decoder results, controls, `docs/MDRQVAE_REPORT.md`, v1 lock, docs restructure |
| `a6f8637` | Arts/Games chain decoder results (report §5.1) |
| `5f782e1`…`856de37` | C/D-gap controls code, E-raw Arts/Games results |
| `0ec4749`…`b0b54b4` | C/D-gap control results (report §5.2, BASELINE_REPORT §11), stale-status cleanup |
| `d4cf286`…`6171446` | v1 reproduction recipe (`docs/MDRQVAE.md` A3), `repro_v1.sbatch`, reproducibility check, newcomer wording |
| `15b6002`, `4c30cb2` | MFD stage 1 code (`macrec/mfd_*.py`, `sbatch/mfd/`); untrained-sanity embedding fix, `mfd_table.py` |
| `93b32d9` | MFD stage 1 results: `docs/MFD_REPORT.md`, `mfd_paired.py`, `paired_mfd.sbatch` |
| `4a6adca`…`5e29eca` | MFD Parts A–C code: L-level MFD, sweep, multi-view augmentation, pack training, eval jobs |
| `57a585f`, `df0a5f8` | MFD Parts A–C results: `docs/MFD_REPORT.md` §7–§12, `mfd_parts_table.py`, `mfd_paired_parts.py` |
| `15a2e1a`…`20db1ca` | Plug-in (MD SIDs in C's framework) code: views, config gate, training/eval jobs, first-token, paired + table tools |
| `e87ae13` | Plug-in results: `docs/PLUGIN_REPORT.md` (MD loses to C under C's recipe) |
| `91ff52e`…`bc8c0bc` | MFD with C's task mix: datasets, checks, gate, eval, paired tools; pack seed field |
| `84ab720`…`29e1744` | codebook-geometry-aware T5: model, entry, checks, gate, diagnostics, run scripts |
| (next) | MFD task-mix results (`docs/MFD_REPORT.md` §13) |

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
