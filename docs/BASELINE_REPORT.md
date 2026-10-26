# Baseline reproduction report: TIGER, MQL4GRec, MACRec on Amazon 2018

Date: 2026-10-26. Repo commit of the last eval jobs: `24ef25a`.
All numbers below are read from `/scratch/yw8866/mmgen_runs/baselines/baselines_all.json`
(written by `sbatch/tools/collect_baselines.py`). Each run directory also holds `run_summary.json`.

Protocol for every number:
- full ranking over all items, trie-constrained beam search, 20 beams;
- checkpoint selected on validation loss (early stopping, patience 10), metrics reported on test;
- "matched" means within 5% relative of the published HR@10 **and** NDCG@10;
- C and D are judged on the merged list (MQL4GRec / MACRec `ensemble.py`).

No cross-domain pre-training was used. No hyperparameter was tuned after seeing a result.

---

## 1. Status

| baseline | Instruments | Arts | Games |
|---|---|---|---|
| A TIGER-text | **miss**, 1 seed (−7.4% / −9.6%) | not run (Instruments miss) | not run (Instruments miss) |
| B TIGER-image | 1 seed, no published target | not run | not run |
| C MQL4GRec (no pre-training) | **match**, 3 seeds | **match**, 3 seeds | **match**, 3 seeds |
| D MACRec, spec / paper setting (`macrecpaper`) | **match**, 3 seeds | **miss**, 3 seeds (−5.6% / −6.3%) | **miss**, 3 seeds (−5.4% / −4.5%) |
| D MACRec, repo as shipped (`macrec`) | **miss**, 1 seed (−10.4% / −9.3%) | not run | not run |

`macrecpaper` follows the task spec and the MACRec paper (cross-modal contrast on layers 2–3,
weights 0.1 / 0.1 / 0.001). `macrec` is the repo's `run_rqvae.sh` exactly as shipped, which uses
contrast on all layers and silently ignores the weight flags (see §6). `macrecpaper` is the
baseline that was extended; `macrec` is kept as a recorded variant.

---

## 2. Results per dataset (TEST, mean ± std over seeds 42/43/44)

C and D: merged list. A and B: their single list. Gap = mean / target − 1.

### Instruments

| baseline | n | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | target HR@10 / NDCG@10 | gap |
|---|---|---|---|---|---|---|---|---|
| A TIGER-text | 1 | 0.0655 | 0.0929 | 0.1131 | 0.0794 | 0.0859 | 0.1221 / 0.0950 | −7.4% / −9.6% |
| B TIGER-image | 1 | 0.0689 | 0.0915 | 0.1120 | 0.0801 | 0.0867 | — | — |
| C MQL4GRec | 3 | 0.0787 ± 0.0006 | 0.1069 ± 0.0005 | 0.1310 ± 0.0012 | 0.0929 ± 0.0007 | 0.1006 ± 0.0008 | 0.1291 / 0.0997 | +1.4% / +0.9% |
| D MACRec (spec) | 3 | 0.0785 ± 0.0012 | 0.1082 ± 0.0009 | 0.1310 ± 0.0009 | 0.0938 ± 0.0008 | 0.1011 ± 0.0009 | 0.1363 / 0.1046 | −3.9% / −3.4% |
| D MACRec (repo) | 1 | 0.0747 | 0.1020 | 0.1221 | 0.0885 | 0.0949 | 0.1363 / 0.1046 | −10.4% / −9.3% |

Extra TIGER references (MQL4GRec Table 2, Instruments): without user token 0.1216 / 0.0958
(A: −7.0% / −10.3%); with reassignment 0.1277 / 0.0987 (A: −11.4% / −13.0%).

Per seed (HR@10 / NDCG@10):
- C: 0.1301 / 0.0998, 0.1324 / 0.1013, 0.1304 / 0.1008.
- D (spec): 0.1321 / 0.1021, 0.1304 / 0.1005, 0.1306 / 0.1007.

### Arts

| baseline | n | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | target | gap |
|---|---|---|---|---|---|---|---|---|
| C MQL4GRec | 3 | 0.0624 ± 0.0011 | 0.0996 ± 0.0011 | 0.1262 ± 0.0012 | 0.0813 ± 0.0011 | 0.0898 ± 0.0011 | 0.1254 / 0.0898 | +0.6% / +0.0% |
| D MACRec (spec) | 3 | 0.0621 ± 0.0015 | 0.0987 ± 0.0026 | 0.1255 ± 0.0027 | 0.0807 ± 0.0020 | 0.0893 ± 0.0021 | 0.1329 / 0.0953 | −5.6% / −6.3% |

Per seed HR@10: C 0.1267, 0.1270, 0.1248; D 0.1277, 0.1224, 0.1264.

### Games

| baseline | n | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | target | gap |
|---|---|---|---|---|---|---|---|---|
| C MQL4GRec | 3 | 0.0205 ± 0.0003 | 0.0627 ± 0.0014 | 0.1007 ± 0.0008 | 0.0415 ± 0.0009 | 0.0537 ± 0.0007 | 0.1007 / 0.0538 | −0.0% / −0.2% |
| D MACRec (spec) | 3 | 0.0202 ± 0.0007 | 0.0623 ± 0.0007 | 0.1020 ± 0.0019 | 0.0412 ± 0.0006 | 0.0539 ± 0.0010 | 0.1078 / 0.0565 | −5.4% / −4.5% |

Validation metrics for every run and list are in `baselines_all.json` (`runs.<D>/<B>/<seed>.valid`).
Validation is consistently higher than test (e.g. Instruments C merged valid HR@10 0.1594 vs test 0.1310).

---

## 3. Single-modality lists vs merged (TEST, 3-seed mean, HR@10 / NDCG@10)

| dataset | baseline | text list | image list | merged |
|---|---|---|---|---|
| Instruments | C MQL4GRec | 0.1275 / 0.0984 | 0.1287 / 0.0988 | 0.1310 / 0.1006 |
| Instruments | D MACRec (spec) | 0.1302 / 0.0997 | 0.1285 / 0.0992 | 0.1310 / 0.1011 |
| Instruments | D MACRec (repo), 1 seed | 0.1206 / 0.0941 | 0.1206 / 0.0937 | 0.1221 / 0.0949 |
| Arts | C MQL4GRec | 0.1212 / 0.0860 | 0.1217 / 0.0871 | 0.1262 / 0.0898 |
| Arts | D MACRec (spec) | 0.1218 / 0.0861 | 0.1211 / 0.0866 | 0.1255 / 0.0893 |
| Games | C MQL4GRec | 0.0965 / 0.0513 | 0.0970 / 0.0514 | 0.1007 / 0.0537 |
| Games | D MACRec (spec) | 0.0976 / 0.0513 | 0.0975 / 0.0516 | 0.1020 / 0.0539 |

The merge adds +0.0008 to +0.0045 HR@10 over the better single list. The text and image lists are
within 0.002 of each other in every row.

---

## 4. Tokenizer diagnostics

Collision rate = share of items whose code tuple is not unique **before** reassignment (or, for
TIGER, before the appended index). After reassignment / the appended index **every SID is unique
in every tokenizer** (`all_unique = true`).

| dataset | tokenizer | text collision | image collision | text utilization per level | image utilization per level |
|---|---|---|---|---|---|
| Instruments | A/B TIGER (3×256 + index) | 6.08% (max group 15) | 6.13% (max group 10) | [84, 239, 254] | [104, 256, 256] |
| Instruments | C MQL4GRec (4×256) | 2.42% | 3.31% | [58, 225, 243, 256] | [97, 256, 256, 256] |
| Instruments | D MACRec spec | 2.54% | 3.17% | [98, 252, 255, 255] | [120, 256, 256, 256] |
| Instruments | D MACRec repo | 0.58% | 2.64% | [204, 256, 256, 256] | [227, 256, 256, 256] |
| Arts | C MQL4GRec | 3.75% | 5.10% | [55, 238, 245, 256] | [84, 256, 256, 256] |
| Arts | D MACRec spec | 4.43% | 5.07% | [70, 245, 255, 256] | [99, 256, 256, 256] |
| Games | C MQL4GRec | 2.34% | 25.75% | [103, 235, 244, 256] | [101, 256, 256, 256] |
| Games | D MACRec spec | 2.86% | 25.76% | [73, 240, 253, 256] | [69, 252, 256, 256] |

Published Instruments references: MACRec 2.38% / 3.23%, MQL4GRec 2.76% / 3.71%.
D spec (2.54 / 3.17) is the closest to MACRec's figures; D repo (0.58 / 2.64) is far from them.
The Games image collision rate is a data property (§8), not a tokenizer property.

---

## 5. Inference cost

Each list is one trie-constrained beam search: **5 decoder forward passes per user** (4 SID tokens +
EOS), 20 beams, so 100 beam-hypothesis evaluations per user. Test batch size 64. C and D run two
searches (text, image) plus a CPU merge.

| dataset (test users) | per search, ms/user | per search, wall | merge (C/D) | GPU |
|---|---|---|---|---|
| Instruments (17,112) | 21.5–22.3 | 368–382 s | 1.7–1.8 s | H200 |
| Arts (22,171) | 21.3–21.8 / 17.2–17.3 / 26.3–26.9 | 473–483 / 382–384 / 584–596 s | 1.8–2.5 s | H200 / H100 / A100 |
| Games (42,259) | 20.9–21.3 / 17.5–17.7 | 885–898 / 738–748 s | 3.3–4.1 s | H200 / H100 |

So a C or D prediction costs 2 × 5 = 10 decoder passes per user (≈ 2 × 21.7 ms on H200) versus
5 for TIGER. The same GPU type gives the same ms/user across baselines; the differences between
rows come from the GPU type (H100 fastest here, A100 slowest). The search keeps the GPU only
about 12% busy at batch 64.

The valid **text-list** cost files of the five Instruments seed-42 runs came from a job that ran five
searches concurrently (≈ 470 ms/user, H100). They are contended and must not be used as cost.

Training (all packed several runs per GPU, so wall time is **contended**, not solo):

| dataset | baseline | epochs run | selected epoch | train wall (h) | peak GPU mem (GB) | pack job |
|---|---|---|---|---|---|---|
| Instruments | A TIGER-text | 117 | 107 | 1.17 | 3.12 | 20637308 (5 runs) |
| Instruments | B TIGER-image | 123 | 113 | 1.21 | 3.12 | 20637308 |
| Instruments | C seeds 42/43/44 | 102 / 114 / 109 | 92 / 104 / 99 | 2.20 / 2.58 / 2.57 | 3.27 | 20637308, 20806461 (4 runs) |
| Instruments | D spec seeds 42/43/44 | 106 / 123 / 112 | 96 / 113 / 102 | 3.40 / 3.97 / 3.85 | 12.75 | 20637308, 20806461 |
| Instruments | D repo | 144 | 134 | 3.77 | 12.76 | 20637308 |
| Arts | C seeds 42/43/44 | 120 / 128 / 138 | 110 / 118 / 128 | 5.00 / 5.43 / 5.53 | 3.27 | 20839400 (6 runs) |
| Arts | D spec seeds 42/43/44 | 111 / 120 / 112 | 101 / 110 / 102 | 7.96 / 8.07 / 7.96 | 12.74 | 20839400 |
| Games | C seeds 42/43/44 | 149 / 132 / 139 | 139 / 122 / 129 | 14.89 / 13.87 / 13.79 | 3.28 | 20839401 (6 runs) |
| Games | D spec seeds 42/43/44 | 119 / 141 / 119 | 109 / 131 / 109 | 18.83 / 19.42 / 18.83 | 12.73 | 20839401 |

All training ran on H100 80GB; packed jobs held the GPU at 97–99.8% utilization.
Peak memory is the PyTorch peak per process. D needs about 4× the memory of C (per-device batch
1024 vs 256 plus the contrastive head).

Tokenizer jobs: Instruments all tokenizers 20617400 (58 min); Arts/Games C 20806460 (C finished,
job then cancelled during k-means, §6); k-means 20813948 (CPU, 17 min); Arts/Games D 20813949 (41 min).

Eval jobs: Instruments 20802837, 20819917, 20839355; Arts 20890965–20890970; Games 20917360–20917371.
Each run's `config_eval.json` records its eval job id and git commit.

---

## 6. Deviations from the published setup

The repos' configs were treated as the source of truth. Every known difference:

1. **One GPU instead of two.** Both repos launch `torchrun --nproc_per_node=2` with
   gradient-accumulation 2. We use grad-accum 4 on one GPU, keeping the effective batch:
   A/B/C per-device 256 → effective 1024 (lr 5e-4); D per-device 1024 → effective 4096 (lr 1e-3).
   The spec says fine-tuning batch 512; the repos' launchers give the numbers above.
2. **Learning rate per repo.** C and A/B use MQL4GRec's 5e-4; D uses MACRec's 1e-3.
3. **A/B hyperparameters.** Neither repo ships a TIGER fine-tuning setup. A/B use MQL4GRec's
   fine-tuning settings with `seqrec` only. The batch 256 comes from the default output directory
   name in MQL4GRec's `ensemble.py` (`ckpt_b256_lr0.0005...`).
4. **TIGER tokenizer is our implementation.** `macrec/cross_index/generate_indices_tiger.py`:
   MQL4GRec's single-modality RQ-VAE with 3×256 levels, argmin codes (no Sinkhorn), then a 4th
   token = index of the item within its code-tuple group. Neither repo contains this code.
5. **RQ-VAE batch 2048, not 1024.** Both repos' tokenizer scripts pass `--batch_size 2048`
   (MQL4GRec `index/scripts/run.sh`, MACRec `cross_index/run_rqvae.sh`); we follow them.
   Other tokenizer settings: AdamW lr 1e-3, weight decay 1e-4, 32-dim codes, 500 epochs
   (C, TIGER) or 1000 epochs (D), Sinkhorn ε 0.003 on C's last level only.
6. **MACRec repo bugs** (no repo file edited):
   - `run_rqvae.sh` uses `begin_cross_layer 0` (contrast on all layers), not layers 2–3;
   - `cross_index/main.py` parses `--{text,image,recon}_contrast_weight` but never passes them to
     `CrossRQVAE`, so the defaults 1.0 / 1.0 / 0.001 apply;
   - `--use_cross_rq` is `type=bool`.
   The `macrecpaper` variant sets `begin_cross_layer 2` and applies 0.1 / 0.1 / 0.001 through
   `cross_index/main_cross_weights.py` (monkeypatches the constructor defaults, then runs `main.py`).
7. **numpy 2 / torch 2.6 workarounds.** The reassignment generators run under
   `np.set_printoptions(legacy='1.25')` (their str-based dedup otherwise leaves duplicates);
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; `USE_TF=0`.
8. **Plain-T5 entry point.** C uses `finetune_ext.py --model_class plain` (T5ForConditionalGeneration
   with an explicit `forward` signature, needed for HF Trainer's column filtering). MQL4GRec's own
   trainer uses the same class. D uses MACRec's `CrossModalContrastive` unchanged.
9. **Dataloader workers** = 2 per run (packing several runs per node).
10. **Images.** Downloaded by us from the Amazon-2018 meta files (§8). For imageless items, the
    Instruments CLIP rows were filled from MQL4GRec's shipped file, Arts rows with the dataset mean.
11. **Training wall time is contended** (several runs per GPU). Peak memory and epochs are unaffected.
12. **Eval GPU types differ** (H200 / H100 / A100, recorded per cost file). Metrics are unaffected.
13. **`config.json` in the run directory is overwritten** by the HF model config when the model is
    saved (same output dir). The run settings survive in `config_eval.json`, `training_args.bin`
    and the pack-job logs.

---

## 7. Diagnoses of the misses

The required order was followed: data statistics → split → features → collision handling →
beam / full ranking → schedule. Nothing was tuned.

### A TIGER-text, Instruments (−7.4% / −9.6%)

1. Data statistics: match exactly (§8).
2. Split: same data files and the same `SeqRecDataset` leave-one-out split as C, which matched.
3. Features: our LLaMA text embedding equals MQL4GRec's shipped one (max abs diff 4e-5, cosine 1.0000).
4. Collision handling: 6.08% of items collide before the appended index (max group 15), against
   2.42% for C's 4-level codes. The gap is **not** concentrated on those items. Test HR@10 split by
   whether the target item is in a TIGER collision group (673 items, 2,560 test users):
   - colliding targets: A 0.3215, C text list 0.3461, D text list 0.3531;
   - all other targets (14,552 users): A 0.0764, C 0.0886, D 0.0937.
   A is lower on both subsets, so the collision index does not explain the miss.
5. Beam / full ranking: same `eval_split.py`, trie, 20 beams as C.
6. Schedule: early stopping selected epoch 107 of 117; effective batch 1024, lr 5e-4, as C.

**Unresolved.** Neither repo contains the TIGER configuration behind the published number
(tokenizer depth, collision handling, fine-tuning settings), so our A is our reading of the TIGER
paper (deviation 3–4). TIGER-image also scores below C, consistent with A.

### D MACRec as shipped (`macrec`), Instruments (−10.4% / −9.3%)

Cause identified at step 4 (collision handling / tokenizer config): the shipped script differs from
the paper setting (deviation 6). Its collision rates (0.58% / 2.64%) are far from the published
2.38% / 3.23%, while the paper setting gives 2.54% / 3.17% and matches on Instruments.

### D MACRec spec, Arts (−5.6% / −6.3%) and Games (−5.4% / −4.5%)

1. Data statistics: match exactly.
2. Split: identical code path to C, which matches on both datasets.
3. Features: the same LLaMA / CLIP files are used by C. We cannot compare Arts / Games features to
   the authors' (MQL4GRec ships Instruments features only). Games has the duplicated-image issue (§8),
   which affects C and D alike.
4. Collision handling: all SIDs unique; collision rates are close to C's (Arts 4.43 / 5.07 vs 3.75 / 5.10).
   No published Arts / Games collision figures to compare with.
5. Beam / full ranking: same code and settings as C.
6. Schedule: early stopping at epochs 101–131, well below the 200 cap; same as on Instruments.

**Unresolved, with one consistent pattern:** on all three datasets our D equals our C within noise
(Instruments 0.1310 vs 0.1310, Arts 0.1255 vs 0.1262, Games 0.1020 vs 0.1007), while the paper
reports D above C by 5.6%, 6.0% and 7.0%. D passes on Instruments only because the paper's D–C
margin is smallest there and our C runs 1.4% above its own target. What we reproduce is MQL4GRec-level accuracy
from MACRec's pipeline; the paper's MACRec margin over MQL4GRec is not reproduced by the shipped
code with the paper's settings.

---

## 8. Data notes

- **Statistics match exactly** (users / items / interactions): Instruments 17,112 / 6,250 / 136,226;
  Arts 22,171 / 9,416 / 174,079; Games 42,259 / 13,839 / 373,514. Repeat ratio 0.0 on all three.
  MACRec's and MQL4GRec's Instruments data files are byte-identical.
- Text features: LLaMA-7B (`huggyllama/llama-7b`), mean-pooled title + description, via the repo's
  `amazon_text_emb.py`.
- Image features: CLIP ViT-L/14 on the first valid URL in `imageURLHighRes`
  (meta from `mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/`).
  - Items with no image obtainable: Instruments 2, Arts 3, Games 0.
  - First URL failed, later one used: Instruments 1, Arts 18, Games 19.
  - Instruments CLIP vs MQL4GRec's shipped file: median cosine 0.99865, but 457 / 6,250 rows have
    cosine < 0.99 (min 0.645); the images behind some URLs have likely changed since the papers.
- **Duplicated images.** Items sharing the exact same image file with another item: Instruments 164,
  Arts 425, **Games 3,475 (25.1%)**. In Games, 2,287 items' first image is the same generic banner
  (`31BDm2VqflL.jpg`, an orange abstract strip, no product). This is why Games image tokenizers
  collide 25.8% before reassignment, and the reassignment then spreads those items over arbitrary
  nearby codes. We kept the first-valid-URL rule; we do not know what the authors' download gave.
- MQL4GRec's shipped Instruments text index (`.index_lemb.json`) has only 5,988 unique SIDs for
  6,250 items; we did not use it (all our SIDs are regenerated and unique).

---

## 9. What is needed from you

1. **TIGER (A/B):** decide whether to accept A as "not reproduced" with the diagnosis above, or to
   specify a TIGER setup to follow (e.g. a public TIGER implementation's tokenizer and fine-tuning
   settings). A/B were not extended to more seeds or datasets.
2. **MACRec variant:** confirm that `macrecpaper` (spec / paper setting) is the MACRec baseline and
   `macrec` (as shipped) stays a recorded deviation.
3. **MACRec on Arts / Games:** decide whether to accept the −4.5% to −6.3% misses as the
   reproducible MACRec level (it ties MQL4GRec), or to contact the authors for their exact
   tokenizer checkpoints / index files, which would separate tokenizer from fine-tuning causes.
4. **Games images:** decide whether to keep the generic-banner images (current, follows the
   download rule) or to treat them as missing; either choice should be applied to all baselines.
5. **Security, still open:** rotate the W&B key `wandb_v1_VF5U…` found in the old GRID repo.
