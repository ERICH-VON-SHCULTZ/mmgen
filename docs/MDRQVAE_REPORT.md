# MD-RQ-VAE on Amazon: report

Date: 2026-10-28, Arts/Games chain decoder added 2026-10-30, C/D-gap controls 2026-10-31. Seed 42 for every new run. All test numbers are full ranking, trie-constrained
beam search with 20 beams, and a checkpoint chosen on validation loss. Raw numbers:
`mmgen_runs/baselines/baselines_all.json` (`sbatch/tools/collect_baselines.py`). Attempt-level
detail: `docs/MDRQVAE_LOG.md`. Official SID and the current implementation: `docs/MDRQVAE.md` Part A.

Status: v1 locked (2026-10-28). Chain decoder done on all three datasets for 1+2+2 and 1+3+3 (§5,
§5.1). E-raw done on all three datasets. C/D-gap controls done (§5.2).

---

## 1. Baselines (TEST, HR@10 / NDCG@10)

A, B and E train T5 on seqrec only and decode one list. C and D train on extra cross-modal tasks.
Their papers decode two lists (text, image) and merge them; here they are shown with their best
single list, with the merged list in its own column.

| baseline | Instruments | Arts | Games |
|---|---|---|---|
| A TIGER-text, 3 seeds | 0.1154 / 0.0876 | 0.0986 / 0.0661 | 0.0885 / 0.0471 |
| B TIGER-image, 3 seeds | 0.1139 / 0.0883 | 0.0957 / 0.0657 | 0.0782 / 0.0415 |
| E-norm (rescaled, c = mean train-item text norm / √2), seed 42 | 0.1055 / 0.0822 | 0.0825 / 0.0520 | 0.0843 / 0.0454 |
| E-raw, seed 42 | 0.1132 / 0.0870 | 0.0902 / 0.0565 | 0.0859 / 0.0448 |
| C MQL4GRec, best single list, 3 seeds | 0.1287 / 0.0988 | 0.1217 / 0.0871 | 0.0970 / 0.0514 |
| D MACRec (paper setting), best single list, 3 seeds | 0.1302 / 0.0997 | 0.1218 / 0.0861 | 0.0976 / 0.0513 |
| C / D merged (2 beam searches), 3 seeds | 0.1310 / 0.1006, 0.1310 / 0.1011 | 0.1262 / 0.0898, 0.1255 / 0.0893 | 0.1007 / 0.0537, 0.1020 / 0.0539 |

The E-norm constant c was computed once from training items and not tuned: Instruments 37.868,
Arts 37.359, Games 36.744. The E-norm tokenizer uses few first-level codes (Instruments 67, Arts
42, Games 38 of 256). Its full-SID collision is high before the index token: 5.33%, 7.33% and 12.59%.
E-raw beats E-norm on Instruments by more than the seed spread (+0.0077 HR@10), so E-raw was
extended to Arts and Games (tokenizer 21335767, T5 21335768, evals 21450640/41): it beats E-norm on
Arts (+0.0077) and ties it on Games. Under the final-epoch checkpoint
rule, the E-norm index on Instruments is byte-identical to the best-collision index. Its lowest
collision falls at the last evaluation, so MD and E use the same rule there.

---

## 2. Implementation and differences from Tmall

The design principles are kept: SID = [shared, text chain, image chain]; the shared code is
quantized first from a fused representation; each branch = Linear([shared residual (hard code);
modality latent]); text is reconstructed from shared + text codes, image from shared + image codes;
conditional independence is measured (CMI with a permutation null).

| aspect | Tmall (legacy GRID) | Amazon (mmgen) |
|---|---|---|
| features | SigLIP2 text + image, one space, 1152-d | LLaMA-7B text (4096-d) and CLIP ViT-L/14 image (768-d), as MACRec / MQL4GRec |
| input | LayerNorm(L2-normalized); target = raw embedding | L2-normalized × √dim (also the reconstruction target) |
| encoders / decoders | MLP 1152→768→256→512 | the A/B RQ-VAE MLP [2048,1024,512,256,128,64] → 32 |
| latent / codes | 512-d; codebooks 2048 / 1024 / 1024; 1 level each | 32-d; 256 per level; 1+2+2 or 1+3+3 levels (residual chains) |
| codebooks | EMA, dead codes re-seeded every step | MACRec VectorQuantizer (gradient, k-means init, β 0.25); dead-code reset every 10 epochs until epoch 400 |
| fusion | cross-attention, 4 heads | the same cross-attention (a05/a06); MLP fusion was a01–a04 |
| loss | MSE_t + MSE_i + VQ + InfoNCE(z_f,z_t) + InfoNCE(z_f,z_i), one-directional | same |
| optimizer | Adagrad 1e-3, 10k steps | AdamW 1e-3, wd 1e-4, batch 2048, 500 epochs |
| checkpoint | last step | **final epoch (500), no selection** (decision 2026-10-27) |
| collisions | kept (36% of 5.4M items) | removed by distance-based reassignment, no extra token (§7) |

---

## 3. Attempt log summary (Instruments, one change per attempt)

| attempt | change | outcome |
|---|---|---|
| a01 | Tmall design at Amazon scale, MLP fusion | utilization ~110 codes on first levels; corrected CMI 4.5% |
| a02 | + dead-code reset | all levels ≥ 254 codes; CMI 1.56%. Kept |
| a03 | InfoNCE weight 3 | worse on every diagnostic. Rejected |
| a04 | a02 at 1+3+3 | passes (CMI 1.83%) |
| **a05** | a02 + cross-attention fusion | reconstruction above A/B, CMI 1.14%. **Final 1+2+2** |
| **a06** | a05 at 1+3+3 | CMI 0.93%. **Final 1+3+3** |

The bar (no collapse, low collision, CMI near its null) was met after 6 of the allowed ~10 attempts.
Arts and Games use exactly the a05/a06 settings, with no tuning.

---

## 4. Diagnostics of the six official sets (final-epoch checkpoint, reassigned)

CMI = first-level CMI(t₁; i₁ | f) minus its permutation-null mean, as % of H(t₁|f). "excl." =
the same after removing items whose image file is byte-identical to another item's.
Balance = cos(D_t(q_f), x_t) − cos(D_i(q_f), x_i).

| set | rec cos text / image (A / B) | util per level | collision before | CMI all / excl. | AMI f~A0 / f~B0 | sep. shared / text / image | additivity t / i | balance |
|---|---|---|---|---|---|---|---|---|
| Instruments 1+2+2 | 0.9600 / 0.9150 (0.9527 / 0.9023) | 253–256 | 0.38% | 1.05% / 0.68% | 0.321 / 0.296 | 0.749 / 1.279 / 0.354 | 0.980 / 0.990 | +0.082 |
| Instruments 1+3+3 | 0.9659 / 0.9265 | 254–256 | 0.26% | 0.93% / 0.51% | 0.299 / 0.269 | 0.765 / 1.330 / 0.370 | 0.995 / 0.996 | +0.080 |
| Arts 1+2+2 | 0.9677 / 0.9136 (0.9600 / 0.8947) | 254–256 | 0.57% | 1.42% / 0.72% | 0.354 / 0.304 | 0.784 / 1.599 / 0.266 | 0.991 / 0.995 | +0.115 |
| Arts 1+3+3 | 0.9676 / 0.9160 | 255–256 | 0.47% | 1.27% / 0.55% | 0.319 / 0.268 | 0.825 / 1.362 / 0.307 | 0.994 / 0.997 | +0.122 |
| Games 1+2+2 | 0.9560 / 0.9176 (0.9462 / 0.9049) | 253–256 | 1.02% | 3.10% / 1.82% | 0.373 / 0.373 | 1.046 / 2.011 / 0.367 | 0.984 / 0.987 | +0.078 |
| Games 1+3+3 | 0.9601 / 0.9223 | 254–256 | 0.58% | 3.61% / 2.02% | 0.316 / 0.358 | 0.839 / 2.436 / 0.425 | 0.985 / 0.975 | +0.072 |

Shared-code balance, both measures:
- By AMI, the shared code leans to text on Instruments and Arts, and to image or neither on Games.
- By shared-only reconstruction, it reconstructs text better than image everywhere; the margin is
  smallest on Instruments and Games.

Reconstruction was still rising at epoch 500 in all six runs. The rise is roughly monotonic, with at
most 12 dips in 50 evaluations and each dip ≤ 0.0023 cos.

---

## 5. Chain decoder vs the baselines (Instruments, seed 42, TEST)

TIGER settings for the decoder: plain T5, batch 256, LR 5e-4, seqrec, 200 epochs, patience 10.
Cost is measured on an H200 with no other job on the GPU.

| method | SID length | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | decoder passes / user | ms / user |
|---|---|---|---|---|---|---|---|---|
| **MD 1+2+2 (official)** | 5 | **0.0751** | **0.0984** | **0.1181** | **0.0869** | **0.0932** | 6 | 25.3 |
| MD 1+3+3 (official) | 7 | 0.0732 | 0.0975 | 0.1161 | 0.0856 | 0.0916 | 8 | 32.0 |
| MD 1+2+2, old index + token | 6 | 0.0720 | 0.0981 | 0.1186 | 0.0853 | 0.0919 | 7 | 30.0 |
| MD 1+3+3, old index + token | 8 | 0.0747 | 0.0969 | 0.1165 | 0.0859 | 0.0922 | 9 | 37.4 |
| A TIGER-text | 4 | 0.0655 | 0.0929 | 0.1131 | 0.0794 | 0.0859 | 5 | 21.8 |
| B TIGER-image | 4 | 0.0689 | 0.0915 | 0.1120 | 0.0801 | 0.0867 | 5 | 22.3 |
| E-raw | 4 | 0.0676 | 0.0928 | 0.1132 | 0.0805 | 0.0870 | 5 | 20.7 |
| E-norm | 4 | 0.0651 | 0.0877 | 0.1055 | 0.0765 | 0.0822 | 5 | 21.5 |
| C MQL4GRec, best single list (3 seeds) | 4 | — | — | 0.1287 | — | 0.0988 | 5 | 21.6 |
| D MACRec, best single list (3 seeds) | 4 | — | — | 0.1302 | — | 0.0997 | 5 | 21.6 |
| C / D merged (two lists) | 4 per list | — | — | 0.1301 / 0.1321 | — | 0.0998 / 0.1021 | 2 × 5 | 2 × 21.6 |

- **MD vs E (the key comparison):** MD 1+2+2 beats E-norm on all five metrics: HR@10 +12%,
  NDCG@10 +13%, HR@1 +15%. It also beats E-raw, A and B on all five. Its HR@10 gain over A's 3-seed
  mean (+0.0027) is close to A's seed std (0.0022). The NDCG@10 gain (+0.0056) is about 4× that std.
- **1+2+2 vs 1+3+3:** 1+2+2 is better and cheaper. 1+3+3's best checkpoint is at epoch 197 of the
  200-epoch cap, so it may be undertrained under the shared TIGER budget.
- **C and D are full-system references, not controls.** Their extra training tasks and their
  tokenizers keep even their single lists above MD. The merge itself adds only +0.001 to +0.004 HR@10.

### 5.1 Arts and Games (seed 42, TEST; train 21258335 / 21260038, eval 21382066–21382069 on H200)

Same decoder settings as Instruments. A and B are 3-seed means ± std; C/D are the better single
list by 3-seed mean (C = image list, D = text list on both datasets). E-raw was evaluated on an H100
(eval 21450640/41), so its ms/user is not comparable.

**Arts**

| method | SID length | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | passes | ms / user |
|---|---|---|---|---|---|---|---|---|
| **MD 1+2+2 (official)** | 5 | 0.0488 | 0.0833 | **0.1059** | 0.0665 | **0.0739** | 6 | 24.9 |
| MD 1+3+3 (official) | 7 | 0.0489 | 0.0814 | 0.1027 | 0.0657 | 0.0726 | 8 | 31.5 |
| A TIGER-text, seed 42 | 4 | 0.0382 | 0.0719 | 0.0931 | 0.0554 | 0.0622 | 5 | 21.1 |
| A TIGER-text, 3 seeds | 4 | — | — | 0.0986 ± 0.0068 | — | 0.0661 ± 0.0052 | 5 | — |
| B TIGER-image, 3 seeds | 4 | — | — | 0.0957 ± 0.0015 | — | 0.0657 ± 0.0019 | 5 | — |
| E-norm | 4 | 0.0275 | 0.0628 | 0.0825 | 0.0457 | 0.0520 | 5 | 21.4 |
| E-raw | 4 | 0.0283 | 0.0692 | 0.0902 | 0.0498 | 0.0565 | 5 | (17.5, H100) |
| C MQL4GRec, best single list | 4 | — | — | 0.1217 | — | 0.0871 | 5 | — |
| D MACRec, best single list | 4 | — | — | 0.1218 | — | 0.0861 | 5 | — |

**Games**

| method | SID length | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | passes | ms / user |
|---|---|---|---|---|---|---|---|---|
| MD 1+2+2 (official) | 5 | 0.0169 | 0.0548 | 0.0885 | 0.0360 | 0.0468 | 6 | 24.5 |
| **MD 1+3+3 (official)** | 7 | 0.0174 | 0.0562 | **0.0899** | 0.0367 | **0.0475** | 8 | 32.3 |
| A TIGER-text, seed 42 | 4 | 0.0173 | 0.0546 | 0.0873 | 0.0359 | 0.0464 | 5 | 21.1 |
| A TIGER-text, 3 seeds | 4 | — | — | 0.0885 ± 0.0018 | — | 0.0471 ± 0.0011 | 5 | — |
| B TIGER-image, 3 seeds | 4 | — | — | 0.0782 ± 0.0030 | — | 0.0415 ± 0.0018 | 5 | — |
| E-norm | 4 | 0.0180 | 0.0530 | 0.0843 | 0.0354 | 0.0454 | 5 | 21.7 |
| E-raw | 4 | 0.0153 | 0.0529 | 0.0859 | 0.0342 | 0.0448 | 5 | (17.5, H100) |
| C MQL4GRec, best single list | 4 | — | — | 0.0970 | — | 0.0514 | 5 | — |
| D MACRec, best single list | 4 | — | — | 0.0976 | — | 0.0513 | 5 | — |

Best T5 epochs: Arts 1+2+2 176 of 186, 1+3+3 198 of 198.1 (cap; 21,000 steps); Games 1+2+2 129 of
139, 1+3+3 173 of 183.

- **MD vs E-norm:** MD wins on both datasets. Arts 1+2+2 HR@10 +28% (0.1059 vs 0.0825), NDCG@10 +42%.
  Games: 1+2+2 +5% / +3%, 1+3+3 +7% / +5%. On Games E-norm's HR@1 (0.0180) is above both MD configs.
- **MD vs E-raw:** MD wins on both datasets. Arts 1+2+2 +0.0157 HR@10 (+17%) / +0.0174 NDCG@10 (+31%);
  Games 1+2+2 +0.0026 / +0.0020 (+3% / +4%), 1+3+3 +0.0040 / +0.0027. E-raw beats E-norm on Arts
  (+0.0077 HR@10, as on Instruments) and ties it on Games; it stays below A on both.
- **MD vs A/B:** Arts: MD 1+2+2 is above A's 3-seed mean by +0.0073 HR@10 (≈1.1 std) and +0.0078
  NDCG@10 (≈1.5 std), and above B by more. Against A at the same seed (42) the gap is +0.0128 /
  +0.0117. Games: MD ties A (1+2+2 equals A's mean on HR@10; 1+3+3 is +0.0014 ≈ 0.8 std) and clearly
  beats B (+0.010 to +0.012 HR@10).
- **1+2+2 vs 1+3+3:** 1+2+2 is better on Instruments and Arts; 1+3+3 is slightly better on Games
  (+0.0014 HR@10, within one A std). 1+3+3 again stopped at the epoch cap on Arts.
- **C and D stay above MD** on both datasets, as on Instruments (Arts −0.016 HR@10, Games −0.007 to
  −0.009 vs the best MD config).
- **One seed.** The Arts gap to A is the size of A's own seed spread (0.0931–0.1061); MD 1+2+2's
  0.1059 sits at A's best seed. 3 seeds are needed before claiming MD > A on Arts.

### 5.2 Controls for the C/D gap (seed 42, TEST; Instruments and Arts; 2026-10-31)

**Question.** C/D's single lists beat MD although decoding is identical (same `eval_split.py`, trie
beam search, 20 beams, one list). Is that their tokenizer or their training recipe (both SIDs, 5–6
tasks)? Two controls:

1. **C's SIDs, seqrec only** (`mqltext`, `mqlimage`): MQL4GRec's text or image SID, trained exactly
   like A/B/E/MD (plain T5, batch 256, LR 5e-4, seqrec). Train 21422951, evals 21422953–56.
2. **MD 1+2+2 + C-style multi-task** (`mdmt122`, `macrec/md_mt_data.py`): v1 SID as one sequence;
   seqrec + shared (history → `[s]`) + text view (→ `[s;t]`) + image view (→ `[s;v]`), each with its
   own soft prompt, + translation (`[s;t]` → `[v]` and `[s;v]` → `[t]`, per item). C's batch / LR,
   plain T5; validation and test on seqrec only, so decoding (6 passes) is unchanged. Train 21422952,
   evals 21422957–58.

Examples seen = optimizer steps × effective batch (1024; D 4096). Counts: `train_stats.json`
(`train_examples_per_task`) or `sbatch/tools/train_examples.py`.

| dataset | run | SIDs in training | HR@1 | HR@10 | NDCG@10 | ex/epoch | steps | ex seen | best / run epochs |
|---|---|---|---|---|---|---|---|---|---|
| Instruments | A TIGER-text | text | 0.0655 | 0.1131 | 0.0859 | 84,890 | 9,711 | 9.9M | 107 / 117 |
| Instruments | B TIGER-image | image | 0.0689 | 0.1120 | 0.0867 | 84,890 | 10,209 | 10.5M | 113 / 123 |
| Instruments | **C's text SID, seqrec only** | text | 0.0677 | 0.1036 | 0.0832 | 84,890 | 10,375 | 10.6M | 115 / 125 |
| Instruments | **C's image SID, seqrec only** | image | 0.0688 | 0.1081 | 0.0853 | 84,890 | 10,790 | 11.0M | 120 / 130 |
| Instruments | C multi-task, text / image list | both | 0.0771 / 0.0781 | 0.1271 / 0.1271 | 0.0978 / 0.0984 | 267,170 | 26,622 | 27.3M | 92 / 102 |
| Instruments | MD 1+2+2, seqrec | one MD SID | 0.0751 | 0.1181 | 0.0932 | 84,890 | 16,019 | 16.4M | 183 / 193 |
| Instruments | **MD 1+2+2, C-style multi-task** | one MD SID | 0.0719 | 0.1126 | 0.0894 | 352,060 | 19,264 | 19.7M | 46 / 56 |
| Arts | A TIGER-text | text | 0.0382 | 0.0931 | 0.0622 | 107,566 | 11,342 | 11.6M | 97 / 107 |
| Arts | B TIGER-image | image | 0.0404 | 0.0949 | 0.0643 | 107,566 | 12,720 | 13.0M | 110 / 120 |
| Arts | **C's text SID, seqrec only** | text | 0.0398 | 0.0880 | 0.0603 | 107,566 | 12,720 | 13.0M | 110 / 120 |
| Arts | **C's image SID, seqrec only** | image | 0.0380 | 0.0921 | 0.0625 | 107,566 | 12,508 | 12.8M | 108 / 118 |
| Arts | C multi-task, text / image list | both | 0.0589 / 0.0617 | 0.1189 / 0.1235 | 0.0846 / 0.0882 | 341,530 | 39,960 | 40.9M | 110 / 120 |
| Arts | MD 1+2+2, seqrec | one MD SID | 0.0488 | 0.1059 | 0.0739 | 107,566 | 19,716 | 20.2M | 176 / 186 |
| Arts | **MD 1+2+2, C-style multi-task** | one MD SID | 0.0515 | 0.1007 | 0.0726 | 449,096 | 24,584 | 25.2M | 46 / 56 |

All rows seed 42 (A/B/C seed 42, not the 3-seed means). MD multi-task per-epoch tasks: seqrec,
shared, text, image 84,890 each + translation 12,500 (Instruments); 107,566 each + 18,832 (Arts).
ms/user: the MD multi-task Instruments eval ran on an H200 (25.1, same as MD seqrec); the other
control evals ran on H100/A100 and their ms/user is not comparable. Decoder passes: C SIDs 5, MD 6.

- **C's tokenizer alone gives no gain.** Trained like A/B, C's SIDs are 0.003–0.010 HR@10 *below*
  TIGER's on both datasets and both modalities.
- **C's lead is the training recipe.** Same C SIDs, seqrec-only → C multi-task: HR@10 +0.019 to
  +0.024 on Instruments, +0.031 on Arts (+18% to +35%); NDCG@10 +0.013 to +0.026. C trains on 2.5–3.2×
  the examples, and every task pairs two *different* SIDs of the item (text history vs image
  history, text SID ↔ image SID).
- **The same recipe does not transfer to MD's single sequence.** MD multi-task is below MD seqrec:
  HR@10 −0.0055 (Instruments, ≈2.5× A's seed std) and −0.0052 (Arts, within A's Arts seed spread);
  NDCG@10 −0.0038 / −0.0013. Arts HR@1 is higher (+0.0027). It early-stopped at epoch 56 (best 46) on
  seqrec validation loss, after 1.2× the steps of MD seqrec. Valid HR@10 is also lower (Instruments
  0.1426 vs 0.1476, Arts 0.1225 vs 0.1244).
- **Likely reason (not tested):** our MD tasks only vary the *target* (sub-sequences of the full SID,
  which seqrec already trains). The input history is always the full SID. C's tasks also vary the
  *input* (text-only vs image-only histories), which acts as two views of the user. A test is listed
  in §6.

---

## 6. Blockers and open items

- **Next step: MFD** (`docs/MFD.md`, design not final), on the same v1 sets as the chain decoder.
- **Tokenizer variance is unmeasured.** Retraining a v1 tokenizer with identical settings and seed
  gives an equally good but different SID set (shared-level AMI to v1 only 0.20–0.38;
  `docs/MDRQVAE.md` A3.7). All results so far vary only the T5 seed. Decide whether the 3-seed paper
  runs also retrain the tokenizer.
- **One seed:** several gaps are near seed noise. Final paper comparisons need seeds 42/43/44, at
  least on Arts, where TIGER-text's HR@10 ranges from 0.0931 to 0.1061 across seeds.
- **Done (§5.2):** C's SIDs with seqrec only (no tokenizer gain) and MD with C-style multi-task (no gain).
- **Optional, not queued:** MD multi-task with view-only histories (`[s;t]` history → full SID,
  `[s;v]` history → full SID), the part of C's recipe that §5.2's control did not copy.
- **Optional:** 1+3+3 with a larger epoch cap (best epoch at the cap on Instruments and Arts).

---

## 7. Official SID: reassignment instead of the trailing token (decision 2026-10-27)

### 7.1 Rule as implemented (`macrec/cross_index/reassign_md.py`)

1. The shared code never changes.
2. Only full-SID collision groups are touched. The item with the lowest total quantization distance
   to its own codes keeps its SID; ties go to the lower item id.
3. Every other item changes exactly one code, on the last level of the text or image chain. It
   takes the code with the smallest increase in that level's quantization distance, searched over
   both chains, among codes whose new SID is unused.
4. Only if no such code exists, the second-to-last level is used. This never happened.
5. Deterministic, with no randomness: groups are ordered by their first item id, items by id.

The SID length is 5 for 1+2+2 and 7 for 1+3+3, with no extra token. The tokenizers were not
retrained; checkpoint = final epoch.

| set | items changed | text / image (all on the last level) | distance increase mean / max |
|---|---|---|---|
| Instruments 1+2+2 | 24 | 15 / 9 | 0.448 / 1.236 |
| Instruments 1+3+3 | 16 | 11 / 5 | 0.220 / 0.566 |
| Arts 1+2+2 | 54 | 40 / 14 | 0.398 / 1.677 |
| Arts 1+3+3 | 44 | 31 / 13 | 0.207 / 0.996 |
| Games 1+2+2 | 141 | 83 / 58 | 0.586 / 3.906 |
| Games 1+3+3 | 80 | 41 / 39 | 0.352 / 1.500 |

For scale, an item's mean total own distance is 50–86. Every SID is unique after reassignment
(asserted).

### 7.2 Diagnostics before and after reassignment

At a fixed checkpoint, reassignment leaves utilization, CMI, AMI and separation exactly unchanged,
because the first levels are untouched. Additivity is also unchanged, and reconstruction changes
only in the 5th decimal. The only change is full-SID collision → 0. The "before" values are the
token index at the same checkpoint (§4, column "collision before").

The final-epoch checkpoint, decided at the same time, does change the diagnostics. The best-collision
→ final-epoch comparison is in `docs/MDRQVAE_LOG.md` ("Final-epoch checkpoint"):
- Instruments and Arts improve or stay the same.
- On Games, reconstruction improves a lot, but CMI rises: excl. 0.79% → 1.82% and 1.64% → 2.02%.

### 7.3 Chain decoder: token vs reassignment (Instruments)

| config | index | SID length | decoder passes | ms / user | valid HR@10 / NDCG@10 | test HR@1 / HR@10 / NDCG@10 |
|---|---|---|---|---|---|---|
| 1+2+2 | token (best-collision checkpoint, epoch 440) | 6 | 7 | 30.0 | 0.1456 / 0.1111 | 0.0720 / 0.1186 / 0.0919 |
| 1+2+2 | reassigned (epoch 440) | 5 | 6 | 28.0 | 0.1457 / 0.1120 | 0.0708 / 0.1148 / 0.0895 |
| 1+2+2 | **reassigned, final epoch (official)** | 5 | 6 | 25.3 | 0.1476 / 0.1132 | 0.0751 / 0.1181 / 0.0932 |
| 1+3+3 | token (epoch 500) | 8 | 9 | 37.4 | 0.1454 / 0.1118 | 0.0747 / 0.1165 / 0.0922 |
| 1+3+3 | reassigned (epoch 500, official) | 7 | 8 | 32.0 | 0.1435 / 0.1105 | 0.0732 / 0.1161 / 0.0916 |

As expected:
- There is one decoder pass fewer, and time per user drops 7–14%.
- Metrics stay within seed noise: test HR@10 changes by at most 0.0038, and 1+2+2 validation
  HR@10 is the same, 0.1456 vs 0.1457.
- At 1+2+2, the final-epoch checkpoint adds +0.0033 test HR@10 over the best-collision one, at the
  same SID length.
- For 1+3+3 both checkpoint rules choose epoch 500, and the two T5 runs on the same index give
  bit-identical metrics, so T5 training is deterministic at seed 42.

### 7.4 Unexpected findings

- **CMI on Games rises with tokenizer training** (all items 1.86% → 3.10% and 3.09% → 3.61%),
  while on Instruments and Arts it does not. Games 1+3+3 ends at 2.02% excl., just above the ~2%
  bar; you accepted it so that all six sets follow one rule.
- **Byte-identical images explain about half of the excess CMI in every set**, not only on Games.
  Games has 4,213 such items; the largest group, 2,287 items, is the banner image.
- **The checkpoint rule matters on Games:** with the old lowest-collision rule, Games 1+2+2 used
  epoch 200, with reconstruction below A/B. The rule comes from MQL4GRec's trainer, where the
  strictly lower collision wins and the earliest epoch wins ties. Our first MD trainer let the later
  epoch win ties instead; this is now documented.
- **E-norm is the weakest single-list method on Instruments,** and its first level uses only 38–67
  of 256 codes on every dataset.
- **Scheduling:** the old-token job could not be reprioritized, because the cluster rejects `scontrol
  update/hold` on these jobs. It was cancelled after it had run 69 s and resubmitted as two jobs.
