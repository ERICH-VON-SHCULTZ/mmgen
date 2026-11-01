# Plug-in: MD-RQ-VAE SIDs inside C's (MQL4GRec) training framework

Question: if the locked v1 MD-RQ-VAE SIDs are given to MQL4GRec (baseline C) as its two modality SIDs,
and trained with exactly C's recipe, do they beat C's own SIDs?

**Answer: no. MD SIDs lose to C under C's recipe on both datasets, in every configuration.**
The merged list is 0.0055–0.0105 HR@10 below C seed 42. Every paired 95% CI on the test split excludes 0.
Every plug-in merged list also falls below C's worst seed. C's recipe does help MD SIDs a lot compared with
MD's own chain decoder (+0.005 HR@10 on Instruments, +0.011 on Arts). It does not close the gap to C.
A shared f token (one f embedding for both views) does not help. It costs 0.0045 merged HR@10 at 1+2+2 and
makes no detectable difference at 1+3+3.

Everything is seed 42, TEST split unless noted. Valid numbers are in `mmgen_runs/plugin/tables.json` and
`mmgen_runs/plugin/paired.json`. The MFD line of work stays paused, and nothing in it was changed.

## 1. How the views were built

**Source.** The locked v1 MD SIDs (`docs/MDRQVAE.md` §0) and the matching v1 tokenizer, final epoch 500.
The hash of every v1 index file is checked against the locked `index.sha256` before use. v1 is not modified.

| dataset | SID | v1 index (`mmgen_runs/md/sids/v1/<D>/<cfg>/index.json`) sha256 | items |
|---|---|---|---|
| Instruments | 1+2+2 | `512221a10e72602185bb0bae1127daef72f6f4f54e67ca0b1a704de331166564` | 6,250 |
| Instruments | 1+3+3 | `1fe4df38cd83c69ec0bb18c94293c441b9bb57e61a43c8bc1270f35b8955c3af` | 6,250 |
| Arts | 1+2+2 | `cbc70c4ed3da57fe3172bc280a2feccee5ab92bec1689691c9cb3eb88a7e0960` | 9,416 |
| Arts | 1+3+3 | `8612e3cbfd92fa1929df7134aa0898c842c2515b98a3889aa93ccb81f6a14961` | 9,416 |

**Token format.** Each v1 SID `[f, t1.., i1..]` is split into two views:
- text view `[f, t1, t2(, t3)]`;
- image view `[f, i1, i2(, i3)]`.

The views are written in MQL4GRec's own token format:
- **Main (pure drop-in):** text view `<a_f><b_t1><c_t2>(<d_t3>)`, image view `<A_f><B_i1><C_i2>(<D_i3>)`. So f is a
  text-prefix token in the text view and an image-prefix token in the image view, exactly like C's first level.
- **Shared-f:** the same, except that f is one token `<s_f>` in both views. Only the vocabulary changes.

Example (Instruments 1+2+2): the v1 SID `<s0_60><t0_53><t1_45><v0_46><v1_172>` gives
- main: text `<a_60><b_53><c_45>`, image `<A_60><B_46><C_172>`;
- shared-f: text `<s_60><b_53><c_45>`, image `<s_60><B_46><C_172>`.

**Uniqueness per view.** A view can collide even though the full v1 SID is unique. For example, two items can
share `[f, t1, t2]` and differ only in their image codes. Each view is made unique with MQL4GRec's own rule,
transcribed in `macrec/cross_index/plugin_views.py`:
- colliding items are grouped;
- inside a group, items are sorted by their own distance to their last-level code;
- each item takes the nearest last-level code of that view that is not yet used, using the v1 tokenizer's
  residual distances along the item's actual codes;
- the second-to-last level is the fallback, and it was never needed;
- f is never changed;
- only colliding items are touched, and the first item of each group keeps its code;
- the procedure is deterministic.

All views were unique after one pass. Every file's sha256 is recorded in `mmgen_runs/plugin/<D>/<cfg>/views.json`.

| dataset | SID | view | items colliding before | groups (max size) | items reassigned | 2nd-to-last changed | main / shared-f file (in `mmgen_runs/data/<D>/`) |
|---|---|---|---|---|---|---|---|
| Instruments | 1+2+2 | text | 85 (1.36%) | 42 (3) | 43 | 0 | `Instruments.index_plug122_text.json` / `…plug122s_text.json` |
| Instruments | 1+2+2 | image | 238 (3.81%) | 108 (4) | 130 | 0 | `Instruments.index_plug122_image.json` / `…plug122s_image.json` |
| Instruments | 1+3+3 | text | 10 (0.16%) | 5 (2) | 5 | 0 | `Instruments.index_plug133_text.json` / `…plug133s_text.json` |
| Instruments | 1+3+3 | image | 102 (1.63%) | 48 (4) | 54 | 0 | `Instruments.index_plug133_image.json` / `…plug133s_image.json` |
| Arts | 1+2+2 | text | 128 (1.36%) | 63 (3) | 65 | 0 | `Arts.index_plug122_text.json` / `…plug122s_text.json` |
| Arts | 1+2+2 | image | 679 (7.21%) | 262 (32) | 417 | 0 | `Arts.index_plug122_image.json` / `…plug122s_image.json` |
| Arts | 1+3+3 | text | 36 (0.38%) | 18 (2) | 18 | 0 | `Arts.index_plug133_text.json` / `…plug133s_text.json` |
| Arts | 1+3+3 | image | 318 (3.38%) | 135 (8) | 183 | 0 | `Arts.index_plug133_image.json` / `…plug133s_image.json` |

The Arts shared-f files were built, but only the main variant was trained on Arts, as specified.

Job: `sbatch/plugin/views.sbatch` (CPU).

## 2. Config diff against C seed 42

Gate: `sbatch/plugin/config_check.sbatch` runs `sbatch/tools/plugin_config_diff.py`, and its output is
`mmgen_runs/plugin/config_diff.json`. Every run goes through `run_baseline.sh` with `DRYRUN=1`. That executes
C's exact training command up to the point where `train()` starts, and dumps:
- the resolved arguments;
- TrainingArguments;
- the model config and parameter count;
- the training examples per task.

Two checks were made:
1. **C's dry run reproduces C's real seed-42 run.** It was compared with the printed args in C's real
   train.log, its `training_args.bin` and its `config.json`. The only differences are run-directory paths
   (output_dir, logging_dir, run_name, stats_file, run_config_file). Result: OK on Instruments and Arts.
2. **Each plug-in dry run against C's dry run.** The only differences are listed below. Run-directory paths
   also differ (6 fields). Nothing else differs: model, batch 256, LR 5e-4, schedule, early stopping, tasks,
   and the training examples per task and per epoch are all identical (Instruments 267,170 per epoch, Arts 341,530).

| run | index_file (C → plug-in) | image_index_file (C → plug-in) | vocab size (C → plug-in) | parameters (C → plug-in) |
|---|---|---|---|---|
| Instruments 1+2+2 main | `.index_lemb_mql.json` → `.index_plug122_text.json` | `.index_vitemb_mql.json` → `.index_plug122_image.json` | 33,747 → 33,634 | 8,779,264 → 8,764,800 |
| Instruments 1+2+2 shared-f | → `.index_plug122s_text.json` | → `.index_plug122s_image.json` | 33,747 → 33,378 | → 8,732,032 |
| Instruments 1+3+3 main | → `.index_plug133_text.json` | → `.index_plug133_image.json` | 33,747 → 34,144 | → 8,830,080 |
| Instruments 1+3+3 shared-f | → `.index_plug133s_text.json` | → `.index_plug133s_image.json` | 33,747 → 33,888 | → 8,797,312 |
| Arts 1+2+2 main | → `.index_plug122_text.json` | → `.index_plug122_image.json` | 33,746 → 33,632 | 8,779,136 → 8,764,544 |
| Arts 1+3+3 main | → `.index_plug133_text.json` | → `.index_plug133_image.json` | 33,746 → 34,145 | → 8,830,208 |

Verdict: **all OK**, with no unintended difference. Training then ran with frozen code copies. A gate
(`sbatch/tools/plugin_gate.py`) re-checked the config diff, the v1 hashes and the view-file hashes inside the
job before each training. The resolved config of each run was saved before training
(`mmgen_runs/plugin/dryrun/<D>/<B>/seed42/dryrun.json`).

Gotcha: C's train.log prints `deepspeed='./config/ds_z3_bf16.json'` before `train()` sets it to None, so the
printed args are compared with printed args (AGENTS.md §6).

**Training.** All six runs were packed on one H100 (job 21556214, 4 h 21 min, about 99% GPU utilization).
C was trained alone. The steps are not a recipe difference: early stopping chose them under the same rule.

| run | steps | epochs (best) | C: steps, epochs (best) |
|---|---|---|---|
| Instruments 1+2+2 main | 22,185 | 85 (75) | 26,622, 102 (92) |
| Instruments 1+2+2 shared-f | 21,924 | 84 (74) | |
| Instruments 1+3+3 main | 26,361 | 101 (91) | |
| Instruments 1+3+3 shared-f | 30,537 | 117 (107) | |
| Arts 1+2+2 main | 29,970 | 90 (80) | 39,960, 120 (–) |
| Arts 1+3+3 main | 31,635 | 95 (85) | |

**Evaluation.** C's evaluation code was used unchanged:
- `eval_split.py` (test_ddp_save with `--filter_items`), 20 beams with trie-constrained search, then `ensemble.py` for the merge.
- One run per job on an H200 (jobs 21584371–76).

## 3. Results

The table shows test results, with valid HR@10 / NDCG@10 next to them. Notes:
- "Passes / list" counts decoder forward passes per user per list. The merged list costs 2 × that.
- ms/user was measured with batch decoding on an H200 that was not shared, except D Arts (A100).
- The MD chain decoder and MFD rows are seed-42 runs from `docs/MDRQVAE_REPORT.md` and `docs/MFD_REPORT.md`.
  Best MFD is chosen on validation HR@10 among the evaluated MFD settings.

### Instruments

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | valid HR@10 / NDCG@10 | passes / list | ms / user |
|---|---|---|---|---|---|---|---|---|
| C MQL4GRec seed 42, text | 0.0771 | 0.1029 | 0.1271 | 0.0901 | 0.0978 | 0.1542 / 0.1178 | 5 | 21.6 |
| C MQL4GRec seed 42, image | 0.0781 | 0.1036 | 0.1271 | 0.0908 | 0.0984 | 0.1581 / 0.1203 | 5 | 21.6 |
| **C MQL4GRec seed 42, merged** | 0.0780 | 0.1064 | **0.1301** | 0.0921 | **0.0998** | 0.1598 / 0.1206 | 2 × 5 | 43.3 |
| C, text (3-seed mean ± std) | 0.0773 ± 0.0006 | 0.1039 ± 0.0009 | 0.1275 ± 0.0009 | 0.0908 ± 0.0008 | 0.0984 ± 0.0008 | – | 5 | |
| C, image (3-seed mean ± std) | 0.0777 ± 0.0007 | 0.1044 ± 0.0009 | 0.1287 ± 0.0016 | 0.0910 ± 0.0002 | 0.0988 ± 0.0006 | – | 5 | |
| C, merged (3-seed mean ± std) | 0.0787 ± 0.0006 | 0.1069 ± 0.0005 | 0.1310 ± 0.0012 | 0.0929 ± 0.0007 | 0.1006 ± 0.0008 | – | 2 × 5 | |
| D MACRec seed 42, merged | 0.0799 | 0.1090 | 0.1321 | 0.0947 | 0.1021 | 0.1589 / 0.1207 | 2 × 5 | 43.2 |
| D MACRec, merged (3-seed mean ± std) | 0.0785 ± 0.0012 | 0.1082 ± 0.0009 | 0.1310 ± 0.0009 | 0.0938 ± 0.0008 | 0.1011 ± 0.0009 | – | 2 × 5 | |
| plug-in 1+2+2 main, text | 0.0760 | 0.1028 | 0.1227 | 0.0896 | 0.0960 | 0.1498 / 0.1159 | 4 | 18.0 |
| plug-in 1+2+2 main, image | 0.0768 | 0.1026 | 0.1238 | 0.0898 | 0.0966 | 0.1532 / 0.1170 | 4 | 18.2 |
| **plug-in 1+2+2 main, merged** | 0.0772 | 0.1037 | **0.1246** | 0.0908 | **0.0975** | 0.1539 / 0.1175 | 2 × 4 | 36.2 |
| plug-in 1+2+2 shared-f, text | 0.0747 | 0.1004 | 0.1204 | 0.0878 | 0.0942 | 0.1477 / 0.1136 | 4 | 18.2 |
| plug-in 1+2+2 shared-f, image | 0.0745 | 0.1012 | 0.1192 | 0.0880 | 0.0938 | 0.1473 / 0.1143 | 4 | 18.5 |
| plug-in 1+2+2 shared-f, merged | 0.0751 | 0.1012 | 0.1201 | 0.0885 | 0.0946 | 0.1491 / 0.1152 | 2 × 4 | 36.7 |
| plug-in 1+3+3 main, text | 0.0739 | 0.1006 | 0.1207 | 0.0873 | 0.0938 | 0.1493 / 0.1142 | 5 | 22.0 |
| plug-in 1+3+3 main, image | 0.0745 | 0.1006 | 0.1208 | 0.0877 | 0.0942 | 0.1472 / 0.1134 | 5 | 21.6 |
| plug-in 1+3+3 main, merged | 0.0750 | 0.1016 | 0.1217 | 0.0884 | 0.0948 | 0.1508 / 0.1154 | 2 × 5 | 43.6 |
| plug-in 1+3+3 shared-f, text | 0.0726 | 0.0992 | 0.1180 | 0.0861 | 0.0921 | 0.1469 / 0.1125 | 5 | 22.1 |
| plug-in 1+3+3 shared-f, image | 0.0742 | 0.1012 | 0.1215 | 0.0879 | 0.0944 | 0.1483 / 0.1133 | 5 | 21.8 |
| plug-in 1+3+3 shared-f, merged | 0.0729 | 0.1002 | 0.1199 | 0.0871 | 0.0934 | 0.1492 / 0.1136 | 2 × 5 | 43.9 |
| MD 1+2+2 chain decoder (seqrec only) | 0.0751 | 0.0984 | 0.1181 | 0.0869 | 0.0932 | 0.1476 / 0.1132 | 6 | 25.3 |
| MD 1+3+3 chain decoder (seqrec only) | 0.0732 | 0.0975 | 0.1161 | 0.0856 | 0.0916 | 0.1435 / 0.1105 | 8 | 32.0 |
| best MFD: 1+2+2 + multi-view aug, Part B setting (K_f=100, w=1/0.5/0.5) | 0.0756 | 0.1024 | 0.1237 | 0.0891 | 0.0959 | 0.1540 / 0.1160 | 3 | 29.2 |

### Arts

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | valid HR@10 / NDCG@10 | passes / list | ms / user |
|---|---|---|---|---|---|---|---|---|
| C MQL4GRec seed 42, text | 0.0589 | 0.0935 | 0.1189 | 0.0764 | 0.0846 | 0.1420 / 0.0991 | 5 | 21.5 |
| C MQL4GRec seed 42, image | 0.0617 | 0.0971 | 0.1235 | 0.0798 | 0.0882 | 0.1410 / 0.0992 | 5 | 21.5 |
| **C MQL4GRec seed 42, merged** | 0.0619 | 0.0993 | **0.1267** | 0.0809 | **0.0897** | 0.1457 / 0.1026 | 2 × 5 | 43.0 |
| C, text (3-seed mean ± std) | 0.0594 ± 0.0009 | 0.0953 ± 0.0017 | 0.1212 ± 0.0021 | 0.0777 ± 0.0014 | 0.0860 ± 0.0015 | – | 5 | |
| C, image (3-seed mean ± std) | 0.0609 ± 0.0013 | 0.0965 ± 0.0013 | 0.1217 ± 0.0018 | 0.0791 ± 0.0013 | 0.0871 ± 0.0014 | – | 5 | |
| C, merged (3-seed mean ± std) | 0.0624 ± 0.0011 | 0.0996 ± 0.0011 | 0.1262 ± 0.0012 | 0.0813 ± 0.0011 | 0.0898 ± 0.0011 | – | 2 × 5 | |
| D MACRec seed 42, merged | 0.0637 | 0.1009 | 0.1277 | 0.0827 | 0.0912 | 0.1470 / 0.1041 | 2 × 5 | 53.6 (A100) |
| D MACRec, merged (3-seed mean ± std) | 0.0621 ± 0.0015 | 0.0987 ± 0.0026 | 0.1255 ± 0.0027 | 0.0807 ± 0.0020 | 0.0893 ± 0.0021 | – | 2 × 5 | |
| plug-in 1+2+2 main, text | 0.0550 | 0.0901 | 0.1143 | 0.0728 | 0.0805 | 0.1323 / 0.0928 | 4 | 18.2 |
| plug-in 1+2+2 main, image | 0.0555 | 0.0883 | 0.1144 | 0.0722 | 0.0807 | 0.1319 / 0.0925 | 4 | 18.1 |
| **plug-in 1+2+2 main, merged** | 0.0561 | 0.0914 | **0.1172** | 0.0743 | **0.0826** | 0.1363 / 0.0954 | 2 × 4 | 36.3 |
| plug-in 1+3+3 main, text | 0.0592 | 0.0900 | 0.1119 | 0.0751 | 0.0822 | 0.1317 / 0.0957 | 5 | 21.3 |
| plug-in 1+3+3 main, image | 0.0592 | 0.0908 | 0.1143 | 0.0753 | 0.0829 | 0.1328 / 0.0949 | 5 | 21.2 |
| plug-in 1+3+3 main, merged | 0.0605 | 0.0920 | 0.1162 | 0.0767 | 0.0845 | 0.1378 / 0.0984 | 2 × 5 | 42.5 |
| MD 1+2+2 chain decoder (seqrec only) | 0.0488 | 0.0833 | 0.1059 | 0.0665 | 0.0739 | 0.1244 / 0.0867 | 6 | 24.9 |
| MD 1+3+3 chain decoder (seqrec only) | 0.0489 | 0.0814 | 0.1027 | 0.0657 | 0.0726 | 0.1262 / 0.0875 | 8 | 31.5 |
| best MFD: 1+2+2 stage 1, Part B setting (K_f=100, w=1/0.5/0.5) | 0.0442 | 0.0794 | 0.1044 | 0.0625 | 0.0705 | 0.1269 / 0.0839 | 3 | 37.3 |

## 4. Paired comparison against C seed 42

Method:
- Per-user ranks are rebuilt from the saved beams with MACRec's own `evaluate.get_topk_results` and
  `ensemble.py` functions. They reproduce every reported metric exactly (asserted).
- Users are aligned by dataset index, so the test users are the same for both runs.
- 2,000 paired bootstrap resamples of users, seed 0, 95% percentile CI.
- Best single list: each run's own best list (text or image), chosen on validation HR@10.
- Tool: `sbatch/tools/plugin_paired.py`, job `sbatch/plugin/paired.sbatch`, output `mmgen_runs/plugin/paired.json`.

Δ = plug-in − C. Test split:

| dataset | run | list (plug-in vs C) | HR@10 plug-in vs C | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] | users won / lost (hit@10) |
|---|---|---|---|---|---|---|
| Instruments | 1+2+2 main | merged | 0.1246 vs 0.1301 | −0.0055 [−0.0088, −0.0023] | −0.0023 [−0.0041, −0.0005] | 349 / 443 |
| Instruments | 1+2+2 main | best single (image vs image) | 0.1238 vs 0.1271 | −0.0033 [−0.0067, −0.0002] | −0.0018 [−0.0036, −0.0000] | 381 / 438 |
| Instruments | 1+2+2 shared-f | merged | 0.1201 vs 0.1301 | −0.0100 [−0.0133, −0.0068] | −0.0052 [−0.0071, −0.0033] | 319 / 490 |
| Instruments | 1+2+2 shared-f | best single (text vs image) | 0.1204 vs 0.1271 | −0.0067 [−0.0101, −0.0036] | −0.0042 [−0.0063, −0.0023] | 359 / 473 |
| Instruments | 1+3+3 main | merged | 0.1217 vs 0.1301 | −0.0084 [−0.0115, −0.0052] | −0.0049 [−0.0067, −0.0031] | 326 / 470 |
| Instruments | 1+3+3 main | best single (text vs image) | 0.1207 vs 0.1271 | −0.0064 [−0.0098, −0.0030] | −0.0046 [−0.0064, −0.0027] | 354 / 464 |
| Instruments | 1+3+3 shared-f | merged | 0.1199 vs 0.1301 | −0.0103 [−0.0134, −0.0071] | −0.0064 [−0.0082, −0.0046] | 306 / 482 |
| Instruments | 1+3+3 shared-f | best single (image vs image) | 0.1215 vs 0.1271 | −0.0056 [−0.0089, −0.0023] | −0.0040 [−0.0059, −0.0021] | 356 / 452 |
| Arts | 1+2+2 main | merged | 0.1172 vs 0.1267 | −0.0095 [−0.0122, −0.0064] | −0.0071 [−0.0088, −0.0054] | 476 / 686 |
| Arts | 1+2+2 main | best single (text vs text) | 0.1143 vs 0.1189 | −0.0046 [−0.0075, −0.0015] | −0.0041 [−0.0059, −0.0023] | 555 / 656 |
| Arts | 1+3+3 main | merged | 0.1162 vs 0.1267 | −0.0105 [−0.0134, −0.0072] | −0.0052 [−0.0068, −0.0034] | 447 / 679 |
| Arts | 1+3+3 main | best single (image vs text) | 0.1143 vs 0.1189 | −0.0046 [−0.0076, −0.0014] | −0.0017 [−0.0035, −0.0000] | 533 / 635 |

**Validation.** The same picture holds: every ΔHR@10 is between −0.0050 and −0.0107, and every CI excludes 0.

**C's seed spread** (test HR@10, seeds 42/43/44, min–max):

| C list | Instruments | Arts |
|---|---|---|
| merged | 0.1301–0.1324 (std 0.0012) | 0.1248–0.1270 (std 0.0012) |
| text | 0.1269–0.1285 (std 0.0009) | 0.1189–0.1231 (std 0.0021) |
| image | 0.1271–0.1303 (std 0.0016) | 0.1198–0.1235 (std 0.0018) |

Every plug-in merged list is below C's worst seed: Instruments ≤ 0.1246 vs 0.1301, Arts ≤ 0.1172 vs 0.1248.
The C seed-42 merged result used as the reference is C's lowest seed on Instruments, so the paired gaps are,
if anything, conservative.

**Shared-f vs main**, paired the same way (job `sbatch/plugin/paired_sharedf.sbatch`, `REF=plug122` / `REF=plug133`,
outputs `mmgen_runs/plugin/paired_vs_plug122.json` and `paired_vs_plug133.json`). Test split, Instruments:

| SID | list | ΔHR@10 shared-f − main [95% CI] | ΔNDCG@10 [95% CI] |
|---|---|---|---|
| 1+2+2 | merged | −0.0045 [−0.0073, −0.0017] | −0.0029 [−0.0045, −0.0014] |
| 1+2+2 | text | −0.0023 [−0.0051, +0.0007] | −0.0018 [−0.0035, −0.0001] |
| 1+2+2 | image | −0.0046 [−0.0075, −0.0018] | −0.0029 [−0.0045, −0.0013] |
| 1+3+3 | merged | −0.0019 [−0.0046, +0.0009] | −0.0015 [−0.0030, +0.0001] |
| 1+3+3 | text | −0.0026 [−0.0054, +0.0002] | −0.0017 [−0.0033, −0.0000] |
| 1+3+3 | image | +0.0007 [−0.0023, +0.0037] | +0.0002 [−0.0015, +0.0019] |

## 5. First-token accuracy against C

Method:
- Measured at the first decoding step on exactly the test inputs (history only).
- The logits are restricted to the first tokens the trie allows (the first token of every catalog SID).
- Each list is scored against the target item's first token: f for the plug-ins, C's first code level for C.
- The "first tokens" column gives the size of that choice: C uses only 55–97 distinct first-level codes,
  while MD's f uses all 256.
- Uniform guessing would give top-20 = 20/58 = 0.34 (C text, Instruments) but 20/256 = 0.08 for f.
- Tool: `macrec/first_token_acc.py`, job `sbatch/plugin/first_token.sbatch` (A100; accuracy only, no timing).

Test split, valid in brackets:

| dataset | run | text: first tokens | text top-1 | text top-20 | image: first tokens | image top-1 | image top-20 |
|---|---|---|---|---|---|---|---|
| Instruments | C MQL4GRec | 58 | 0.1690 (0.1867) | 0.7638 (0.7741) | 97 | 0.1476 (0.1714) | 0.5892 (0.6145) |
| Instruments | plug-in 1+2+2 main | 256 | 0.1002 (0.1192) | 0.3651 (0.3938) | 256 | 0.1043 (0.1232) | 0.3644 (0.3943) |
| Instruments | plug-in 1+2+2 shared-f | 256 | 0.1001 (0.1174) | 0.3651 (0.3922) | 256 | 0.1013 (0.1187) | 0.3616 (0.3953) |
| Instruments | plug-in 1+3+3 main | 256 | 0.0957 (0.1124) | 0.3607 (0.3898) | 256 | 0.0985 (0.1119) | 0.3559 (0.3863) |
| Instruments | plug-in 1+3+3 shared-f | 256 | 0.0955 (0.1104) | 0.3597 (0.3928) | 256 | 0.0947 (0.1092) | 0.3604 (0.3914) |
| Arts | C MQL4GRec | 55 | 0.2164 (0.2366) | 0.8332 (0.8460) | 84 | 0.1528 (0.1670) | 0.6412 (0.6602) |
| Arts | plug-in 1+2+2 main | 256 | 0.1089 (0.1240) | 0.4291 (0.4549) | 256 | 0.1087 (0.1240) | 0.4241 (0.4537) |
| Arts | plug-in 1+3+3 main | 256 | 0.1005 (0.1161) | 0.3999 (0.4315) | 256 | 0.1017 (0.1126) | 0.3991 (0.4328) |

Under C's recipe, f is predicted about as well from the text view as from the image view (the top-1 values
differ by at most 0.004). The same model sees f under two different tokens and learns both equally.

## 6. Reading

**MD loses to C under C's recipe, on both datasets and in every variant.**
- **Merged list.** The paired ΔHR@10 is −0.0055 to −0.0105 (−4% to −8% relative), with every CI below 0.
  Every plug-in merged list is below C's worst seed.
- **Best single list.** The gap is smaller (−0.0033 to −0.0067) but still significant everywhere. MD's single
  lists are closer to C's than its merged list, because merging gains C much more than it gains MD (next point).
- **1+3+3 is worse than 1+2+2** on merged in both datasets, as in the chain decoder results.

**What C's recipe buys MD SIDs.** Compared with MD's own chain decoder (seqrec only), the plug-in single lists
are clearly better:
- Instruments 1+2+2 text: 0.1227 vs 0.1181.
- Arts 1+2+2 text: 0.1143 vs 0.1059.

So a large part of the earlier MD-vs-C gap was the training framework, not the SIDs: the multi-task recipe
(seqrec, seqimage, item2image, image2item, fusionseqrec) and the merge. The remaining gap of about 0.004–0.005
on single lists, and 0.006–0.010 on merged lists, is attributable to the SIDs themselves.

The plug-in merged list is also at least as good as the best MFD setting:
- Instruments 0.1246 vs 0.1237 (a tie);
- Arts 0.1172 vs 0.1044.

These two comparisons are unpaired.

**Why the merge gains less for MD.** Gain of merged over the best single list (test HR@10):

| | C | plug-in 1+2+2 | plug-in 1+3+3 |
|---|---|---|---|
| Instruments | +0.0030 | +0.0008 | +0.0009 |
| Arts | +0.0032 | +0.0028 | +0.0019 |

With shared-f the gain is −0.0003 and −0.0016 (Instruments). C's two lists come from two independent
tokenizers, so their errors are partly independent. MD's two views share f, so both lists commit to the same
coarse partition first and fail on the same users. MD's factorized design takes away the diversity that
C's ensemble relies on.

**First token.** At its first decoding step, C chooses among only 55–97 coarse codes and gets top-20
accuracy of 0.59–0.83. MD's f has 256 codes, and its top-20 accuracy is 0.36–0.43. f is a finer first decision
than C's first level, and a first-level error there removes the target from every
beam. These numbers are not directly comparable across runs, because the choice sets differ in size, so they
explain the gap only qualitatively.

**Shared-f.** One shared f embedding for both views does not help and partly hurts:
- At 1+2+2 it costs 0.0045 merged HR@10 against main (CI excludes 0), mostly through the image list (−0.0046).
- At 1+3+3 it makes no detectable difference: merged −0.0019, CI includes 0.
- First-token accuracy is unchanged (≤ 0.004).

So sharing the token does not make f easier to predict. It makes the two lists more alike, which leaves the
merge even less to gain: the merge gain is negative with shared-f.

**Bottom line.** Under C's own strong recipe, MD-RQ-VAE SIDs lose to C's SIDs, by a small but significant margin.
C's recipe closes most of the gap that MD's own decoder left. The rest is due to the SIDs: a first level that is
harder to predict, and two views that are too correlated to merge well.

**Cost.** A 1+2+2 plug-in list costs 4 decoder passes (about 18 ms/user on an H200) against C's 5
(about 21.6 ms). 1+3+3 costs the same as C.

## 7. Files and jobs

| item | where |
|---|---|
| view builder | `macrec/cross_index/plugin_views.py`, `sbatch/plugin/views.sbatch` |
| config gate | `macrec/config_dryrun.py`, `sbatch/plugin/config_check.sbatch`, `sbatch/tools/plugin_config_diff.py`, `sbatch/tools/plugin_gate.py` |
| training | `sbatch/plugin/pack_plugin_train.sbatch` (job 21556214), `run_baseline.sh` B=plug122/plug122s/plug133/plug133s |
| evaluation | `run_baseline.sh` STAGE=eval (jobs 21584371–76, H200) |
| first token | `macrec/first_token_acc.py`, `sbatch/plugin/first_token.sbatch` (job 21584377) |
| paired CIs | `sbatch/tools/plugin_paired.py`, `sbatch/plugin/paired.sbatch` (21586430), `sbatch/plugin/paired_sharedf.sbatch` (21586563) |
| tables | `sbatch/tools/plugin_table.py` → `mmgen_runs/plugin/tables.json` |
| runs | `mmgen_runs/baselines/<D>/plug*/seed42/` (`run_summary.json`, `eval/`) |
