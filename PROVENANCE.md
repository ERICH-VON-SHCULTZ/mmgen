# Provenance

Created 2026-10-20. Git history from the source repos was **not** carried over.

## macrec/  (working base)

| | |
|---|---|
| Source | https://github.com/zhangfw123/MACRec |
| Commit | `7a48e7429a8806e2290a1fc6ac5661c1ff934b51` (2025-12-09, "Update README.md") |
| Local clone used | `/scratch/yw8866/tmp/macrec_upstream` |
| Modifications in first commit | none (`.git/` and `.DS_Store` omitted); verified with `diff -rq` |
| License | **No LICENSE file upstream**, so default copyright applies. Research use only until clarified with the authors. |

Upstream ships processed Amazon-2018 interaction and item files for Instruments, Arts and Games
(`macrec/data/<D>/`). It does **not** ship embeddings (`*.emb-llama-td.npy`,
`*.emb-ViT-L-14.npy`), code-index files (`*.index_*.json`), k-means class info, raw
Amazon-2018 metadata or images. All of these are generated locally and git-ignored.

## legacy/grid/  (read-only reference)

| | |
|---|---|
| Source | `/scratch/yw8866/GRID` (local, no remote used) |
| HEAD | `5b3c3ee2c698475f85984d80f88d20ad87891d3f` (2026-08-08, "add mfd and rerank result") |
| Copied from | the **working tree**, not HEAD. At copy time it had 5 modified tracked files (`src/models/modules/semantic_id/{mfd,tiger}_generation_model.py`, `configs/experiment/tiger_mb_mfd_flat.yaml`, `sbatchfiles/train_tiger_baseline.sbatch`, one log) and 41 untracked sbatch files. The copy therefore includes the latest MFD code (history forcing, wide-catalog pool), which HEAD does not. |
| Excluded | `.git/`, `expoutfile/` (711 MB of job logs), `logs/`, `outputs/`, `notices.txt` (26 MB third-party notices, still at `/scratch/yw8866/GRID/notices.txt`), all `*.pt/*.ckpt/*.pkl/*.npy/*.tfrecord` |
| License | `legacy/grid/LICENSE`: Snap Inc. 2025, **non-commercial research only** |
| Secrets | the W&B API key (49 occurrences in `sbatchfiles/`) was replaced with `REDACTED_ROTATE_ME`. **The original key is still in `/scratch/yw8866/GRID` (files + 7 git commits) and must be rotated.** |

`legacy/grid/rec-tmall/` holds a copy of every `*.py`/`*.md` in `/scratch/yw8866/rec-tmall/`:
the analysis, reranker and diagnostic scripts, plus `GOAL.md` (the full research log).
None of its data files were copied.

## Where the artifacts of the key runs live (not in this repo)

| Artifact | Path |
|---|---|
| Tmall TFRecords (multi-behavior, user-drop split) | `/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop/` |
| SigLIP2 text / image item embeddings (1152-d) | `/scratch/yw8866/rec-tmall/{text,image}_embedding/siglip2-so400m-patch14-384.npy` (+ `_ids.pkl`) |
| MD-RQ-VAE checkpoint (2048/1024/1024) | `/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt` |
| MD-RQ-VAE semantic IDs | `/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt` |
| Chain (TIGER) SIDs, tiger_1024 | `/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt` |
| Chain T5 reference checkpoint | `/scratch/yw8866/logs/train/runs/2026-07-10/09-06-58/checkpoints/checkpoint_epoch=000_step=007500.ckpt` |
| MFD (PMQ + Query-f) checkpoint | `/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt` |
| MFD reranker dumps / caches | `/scratch/yw8866/rec-tmall/*dump.pt`, `rec-tmall/seedvar/`, `rec-tmall/coverage_exp/` |
| Job logs (stdout of every sbatch) | `/scratch/yw8866/GRID/expoutfile/` |
| Modality-decomposition report | `/scratch/yw8866/rec-tmall/modality_decomp/report.txt` |

The configs of these runs live in `legacy/grid/configs/experiment/` (`md_rqvae_train_flat.yaml`,
`md_rqvae_inference_flat.yaml`, `tiger_mb_train_flat.yaml`, `tiger_mb_mfd_flat.yaml`, ...)
and in their launch scripts, `legacy/grid/sbatchfiles/` (`train_mdrqvae.sbatch`, `train_mfd_pmq_qf.sbatch`,
`seed_variance.sbatch`, ...).
