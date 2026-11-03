# MFD stage 1 on Instruments: report

> **Parts A–C (1+3+3, wider search, multi-view augmentation, Arts) are §7–§12; summary in §7. MQL4GRec task mix on MFD: §13. Codebook-geometry-aware training: §14.**

Two-branch decoder (MFD) vs the MD chain-decoder control, same SID and training recipe.
Instruments, MD-RQ-VAE 1+2+2 v1 SID, seed 42. Written 2026-10-31.

Runs:
- MFD: `mmgen_runs/mfd/Instruments/mfd122_stage1/seed42/`
  - training job 21515330, which runs the sanity checks on an untrained model first;
  - evaluation job 21515331 → `eval_kf20/`;
  - paired analysis job 21528838 → `eval_kf20/paired_{valid,test}.json`.
- Control: `mmgen_runs/baselines/Instruments/md122_a05_ep500_reassign/seed42/`.
- Code: `macrec/mfd_model.py`, `mfd_data.py`, `finetune_mfd.py`, `mfd_eval.py`, `sbatch/mfd/*.sbatch`,
  `sbatch/tools/mfd_table.py`, `mfd_paired.py`.
- Commits: 15b6002, 4c30cb2, and the commit adding this report.

**Short answer.** MFD matches the chain decoder; it does not beat it.

| | test HR@10 / NDCG@10 | valid HR@10 / NDCG@10 | decode ms/user (H200) |
|---|---|---|---|
| MFD (c) merge | 0.1183 / 0.0926 | 0.1483 / 0.1138 | 6.7 |
| MFD (d) joint | 0.1177 / 0.0923 | 0.1460 / 0.1132 | 6.7 |
| MD chain decoder | 0.1181 / 0.0932 | 0.1476 / 0.1132 | 25.3 |

- MFD needs 3 decoder passes per user instead of 6, and decodes about 3.8× faster.
- Every MFD-vs-chain difference is inside the paired 95% CI (§3).
- The two models predict the shared code f equally well. In both, the target's f is outside the
  top 20 for about 65% of test users.
- **Coverage is the limiting factor (§5).**

---

## 1. Implementation as built

### 1.1 Model and training
- **SID.** `Instruments.index_md122_a05_ep500_reassign.json`, the locked v1 set. Each item is
  `[f; t1 t2; i1 i2]`, 5 tokens.
  - `train_mfd.sbatch` checks the data-dir copy against the v1 sha256 before training.
  - Nothing was regenerated.
- **Encoder input.** The same as the control: MACRec's `SeqRecDataset` (task `seqrec`, the same
  leave-one-out splits). The history is up to 20 items, as full SID token strings, tokenized by the
  same T5 tokenizer. The encoder runs once per example.
- **Decoder.** One T5 decoder (the control's config and checkpoint: d_model 128, 4+4 layers,
  6 heads) runs two sequences per example as two rows of one batched call:

  | branch | decoder input | targets |
  |---|---|---|
  | text | `[start, f, <T>, t1]` | `[f, –, t1, t2]` |
  | image | `[start, f, <I>, i1]` | `[–, –, i1, i2]` |

  - `start` = T5's `decoder_start_token_id`; `–` = ignored (−100).
  - `<T>` and `<I>` are two new vocabulary tokens. The hidden state at `<T>` predicts t1, and the
    one at `<I>` predicts i1.
  - The encoder output is repeated for the two rows. Each row uses T5's standard causal mask, so
    neither branch can see the other's codes.
- **Loss.** Mean token cross-entropy over the 5 targets per example (f, t1, t2, i1, i2).
  - f is a target only in the text row, so it is counted once.
  - The image row gets the same ground-truth f as input.
  - Output head: T5's tied LM head with a full-vocabulary softmax, as in the chain decoder.
- **Training path.** `finetune_mfd.py` calls the control's exact entry point:
  `finetune_ext.main()` → MACRec `finetune_contrastive.train()` → HF Trainer, with the control's
  arguments. Only three things are swapped in: the model class (`MFDT5`), the dataset (the target
  is the item id) and the collator (it builds the two sequences).
  - Identical to the control: AdamW, LR 5e-4, cosine schedule, warmup 0.01, weight decay 0.01,
    batch 256 × grad-accum 4, 200 epochs, early stopping with patience 10 on validation loss,
    best checkpoint by validation loss, seqrec only.
  - The resolved config is in `run_config.json`, written before training (git commit 4c30cb2,
    v1 sha256).
  - The job ran a frozen copy of `macrec/*.py`.
- **One evaluation-only setting.** `prediction_loss_only=True`, so validation keeps the loss
  without accumulating (B, 2, 4, V) logits. Training and checkpoint selection are unaffected.

### 1.2 Inference: three batched decoder passes per user, K_f = 20
- **Pass 1** `[start]` → log P(f). Only the 256 shared codes used in the catalog are considered;
  keep the top 20.
- **Pass 2** `[start, f, <T>]` and `[start, f, <I>]` for each kept f (40 sequences)
  → log P(t1 | f) and log P(i1 | f).
- **Pass 3** `[start, f, <T>, t1]` for every (f, t1) prefix in the catalog under a kept f, and
  likewise for (f, i1) → log P(t2 | f, t1) and log P(i2 | f, i1).
  - On test that is 585 text and 569 image prefixes per user on average.
- **Exact scoring.** Every catalog item whose f is kept is scored exactly; nothing is truncated
  before ranking:
  ```
  score = log P(f) + w_t·[log P(t1|f) + log P(t2|f,t1)] + w_i·[log P(i1|f) + log P(i2|f,i1)],   w_t = w_i = 1
  ```
  - All log-probabilities are full-vocabulary log-softmax values, as in the chain decoder's beam
    scores.
  - The scoring is vectorized with CSR prefix tables (`mfd_data.Catalog`).
- **Modes:**
  - (a) text only: `w_i = 0`.
  - (b) image only: `w_t = 0`.
  - (c) the top-20 lists of (a) and (b), merged with MQL4GRec's rule. The function is MACRec's own
    `ensemble.get_topk_results_ensemble`: an item in both lists scores (s_a + s_b)/2 + 1.
  - (d) joint: `w_t = w_i = 1`.
- **Ties** are broken by item id: sort by id, then a stable descending sort by score.
- **Metrics** come from MACRec's `evaluate.get_metrics_results`, the same function the baselines
  use.

### 1.3 Differences from `docs/MFD.md` (the Tmall MFD)

| # | Tmall MFD (`docs/MFD.md`) | stage 1 as built | why |
|---|---|---|---|
| 1 | behavior token `b`, `behavior_mlp`, CE(b), `w_b·logP(b)` | none | GOAL; Amazon has one behavior |
| 2 | PMQ / Query-f: learnable `q_f`, `q_t`, `q_i` slots plus a zero-init `decoder_slot_type_emb`; f read from `h(q_f)` | branch tokens `<T>` / `<I>` are ordinary vocabulary tokens; f is read from `h(start)`; no slot-type embedding | GOAL block layout `[BOS, f, <T>, t1, t2]` |
| 3 | one code per branch (3-level SID f / t / i), so one modality forward per root | **two codes per branch** (t1 t2, i1 i2): a 2-step chain *inside* each branch, so a third pass is needed | v1 SID is 1+2+2 |
| 4 | per-level `Linear` heads over that level's codebook | T5 tied LM head, full-vocabulary log-softmax; catalog restriction only at selection | same head and scoring convention as the chain control |
| 5 | encoder fed `inputs_embeds` of a stride-3/4 code stream | MACRec text-token history (SID token strings), identical to the control | GOAL; same encoder input as the control |
| 6 | dense `K_f × K_t × K_i` grid, or the G5 wide pool (audit path only), then `topk(K_pool)` | the G5 idea in the main path: every catalog item under the kept f is scored exactly; **no K_t, K_i or K_pool**; the ranked list is cut only at 20 | `MFD.md` §4b–4c: the wide pool was recommended, and K_pool was the binding cut on Tmall |
| 7 | τ (inference sweep), learned fusion | τ = 1, w = 1 fixed; no fusion | GOAL: nothing tuned, nothing chosen on test |
| 8 | Run-C warm-started from the plain MFD checkpoint | trained from scratch with the control's recipe | GOAL: same recipe as the control |
| 9 | loss CE(b) + CE(f) + CE(t\|f) + CE(i\|f) (sum of per-level means) | mean over the 5 target tokens, f counted once (f weight 1/5 of the loss) | GOAL: CE on f once + t1, t2 + i1, i2 |
| 10 | no history forcing on Amazon (Tmall-only) | none | GOAL |
| 11 | state accounting `(1 + K_f)`, or `(1 + 2K_f)` short passes under PMQ | 3 passes; 1 + 2·K_f + #prefixes sequences per user (1,195 on test); no KV cache (each pass re-runs its ≤4-token sequences) | follows from #3 and #6 |
| 12 | single-branch ablation `w_t, w_i ∈ {0, 1}` | the same, as modes (a) / (b); plus the new (c) C-style merge | GOAL |

**Note on (c).** MQL4GRec merges HF beam scores, which are length-normalized (sum / length). The
MFD (a)/(b) scores are raw sums over 3 tokens (f + two branch codes).
- So the "+1 if in both lists" bonus is about 3× smaller relative to the score spread than in C.
- The rule is applied as specified, untuned.

---

## 2. Sanity checks

All checks passed, both before training (on an untrained model) and after training (on the
trained model). Files: `sanity_untrained/sanity.json` and `eval_kf20/sanity.json`.

| check | untrained | trained |
|---|---|---|
| **1. Branch independence, text → image.** 8 test users; replace the text branch's t1 input with a random catalog t1 code; compare every image-branch logit (all positions, full vocabulary) | identical (max abs diff 0.0); text-branch logits changed | identical (0.0); changed |
| **1. Branch independence, image → text.** The same, replacing i1 | identical (0.0); changed | identical (0.0); changed |
| **2. Vectorized score = manual.** For 4 users, 5 random candidates plus the target if scored: joint score from `score_batch` (3-pass inference) vs a manual sum of 5 log-softmax values from one training-style forward of the item's two sequences | 20 pairs, max abs diff 7.6e-6 (tol 1e-4) | 22 pairs, 7.6e-6 |
| **3. f loss counted once.** 5 targets per example; the image row's f label is ignored; text f target = image f input; model loss = manual mean over the 5 targets | 10.871969 vs 10.871970 (diff 9.5e-7) | 1.6439203 vs 1.6439205 (diff 2.4e-7) |

**First attempt failed check 1; reported, not worked around.**
- In the first submission (jobs 21513722 / 21513777, commit 15b6002), the untrained-model
  independence check returned:
  - `other_branch_identical: true`;
  - but also `modified_branch_changed: false`, so the check itself had no power.
- **Cause.** HF `resize_token_embeddings` (mean-resizing) gives every added token (all 1,280 SID
  codes plus `<T>` and `<I>`) the same embedding row. An untrained model therefore cannot tell
  one code from another, and replacing a code changes nothing.
  - A CPU debug run confirmed the decoder is not at fault: with distinct embeddings, changing the
    text t1 input changes the text logits at that position. Earlier positions and every
    image-branch logit stay bit-identical.
- **Fix (commit 4c30cb2).** Only the untrained *sanity copy* (`mfd_eval.py --random_init`) gets
  distinct random rows for the added tokens.
  - Training and the model are unchanged.
  - The chain control starts from the same identical new-token rows, so both models start alike.
  - `modified_branch_changed` remains a required condition of the check.
- Chain after the fix: dry run 21515329 → training 21515330 → evaluation 21515331, all
  successful.

---

## 3. Results

Instruments, seed 42, test with validation next to it. Generated by `sbatch/tools/mfd_table.py`;
the table is also saved as `eval_kf20/mfd_table.json`.
- Every row is a seed-42 run.
- C/D "best single list" is chosen by **validation** HR@10 (image for both).
- ms/user is GPU time inside decoding, measured on an H200.

| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | valid HR@10 / NDCG@10 | decoder passes / user | ms / user |
|---|---|---|---|---|---|---|---|---|
| MFD (a) text only | 0.0729 | 0.0991 | 0.1170 | 0.0862 | 0.0919 | 0.1456 / 0.1130 | 3 | 6.7 |
| MFD (b) image only | 0.0732 | 0.0981 | 0.1169 | 0.0858 | 0.0919 | 0.1448 / 0.1119 | 3 | 6.7 |
| MFD (c) C-style merge of (a)+(b) | 0.0732 | 0.0994 | 0.1183 | 0.0865 | 0.0926 | 0.1483 / 0.1138 | 3 | 6.7 (+0.3 CPU merge) |
| **MFD (d) joint** | 0.0730 | 0.0986 | 0.1177 | 0.0862 | 0.0923 | 0.1460 / 0.1132 | 3 | 6.7 |
| MD 1+2+2 chain decoder | 0.0751 | 0.0984 | 0.1181 | 0.0869 | 0.0932 | 0.1476 / 0.1132 | 6 | 25.3 |
| A TIGER-text | 0.0655 | 0.0929 | 0.1131 | 0.0794 | 0.0859 | 0.1419 / 0.1080 | 5 | 21.7 |
| B TIGER-image | 0.0689 | 0.0915 | 0.1120 | 0.0801 | 0.0867 | 0.1383 / 0.1065 | 5 | 22.3 |
| E-raw | 0.0676 | 0.0928 | 0.1132 | 0.0805 | 0.0870 | 0.1401 / 0.1074 | 5 | 20.7 |
| E-norm | 0.0651 | 0.0877 | 0.1055 | 0.0765 | 0.0822 | 0.1352 / 0.1035 | 5 | 21.5 |
| C MQL4GRec, best single list (image) | 0.0781 | 0.1036 | 0.1271 | 0.0908 | 0.0984 | 0.1581 / 0.1203 | 5 | 21.6 |
| C MQL4GRec, merged | 0.0780 | 0.1064 | 0.1301 | 0.0921 | 0.0998 | 0.1598 / 0.1206 | 2 × 5 | 43.3 |
| D MACRec, best single list (image) | 0.0788 | 0.1068 | 0.1302 | 0.0930 | 0.1006 | 0.1591 / 0.1202 | 5 | 21.7 |
| D MACRec, merged | 0.0799 | 0.1090 | 0.1321 | 0.0947 | 0.1021 | 0.1589 / 0.1207 | 2 × 5 | 43.2 |

**How to read the rows.**
- All MFD modes come from the same three passes. The ms/user is measured for (d); (a) and (b) are
  sub-sums of the same scores.
- The chain and the baselines decode with 20 beams. For the chain, 6 passes means 6 steps
  × 20 beams = 120 hypothesis evaluations per user.
- MFD's 3 passes evaluate 1 + 40 + about 1,154 short sequences per user (one batched call per
  pass). Peak GPU memory is 10.9 GB at 64 users per batch.
- E-norm on Instruments is the byte-identical final-epoch rerun (`tiger_fnormj_ep500`).
- C/D rows are multi-task models. Per the C/D-gap controls (`docs/MDRQVAE_REPORT.md` §5.2), their
  lead comes from the training recipe, not from decoding.

**Paired comparison with the chain decoder** (same 17,112 test users; 2,000 bootstrap resamples;
`eval_kf20/paired_test.json`):

| MFD mode − chain | test ΔHR@10 [95% CI] | test ΔNDCG@10 [95% CI] | test users won / lost at HR@10 | valid ΔHR@10 [95% CI] | valid ΔNDCG@10 [95% CI] |
|---|---|---|---|---|---|
| (a) text | −0.0011 [−0.0043, +0.0018] | −0.0013 [−0.0032, +0.0004] | 354 / 373 | −0.0020 [−0.0054, +0.0011] | −0.0002 [−0.0020, +0.0016] |
| (b) image | −0.0012 [−0.0043, +0.0019] | −0.0013 [−0.0032, +0.0004] | 366 / 386 | −0.0028 [−0.0060, +0.0004] | −0.0013 [−0.0031, +0.0005] |
| (d) joint | −0.0004 [−0.0036, +0.0026] | −0.0009 [−0.0028, +0.0008] | 368 / 375 | −0.0016 [−0.0049, +0.0016] | −0.0001 [−0.0020, +0.0019] |

- Every interval contains 0. On this seed MFD and the chain decoder are statistically
  indistinguishable.
- The two models also hit largely different users: about 370 users each way of roughly 2,000
  hits.
- Mode (c) is not in the paired table: its merged lists are rebuilt inside `mfd_eval.py` and not
  saved per user. Its test HR@10 is +0.0002 versus the chain, so it falls inside the same range.

**Coverage and ranking given coverage**, per MFD mode. Coverage = the target item's f is among the
20 kept shared codes; it is the same for all modes because they share pass 1.

| mode | test coverage | test HR@10 given coverage | valid coverage | valid HR@10 given coverage |
|---|---|---|---|---|
| (a) text | 0.3461 | 0.3380 | 0.3815 | 0.3816 |
| (b) image | 0.3461 | 0.3378 | 0.3815 | 0.3794 |
| (c) merge | 0.3461 | 0.3419 | 0.3815 | 0.3886 |
| (d) joint | 0.3461 | 0.3400 | 0.3815 | 0.3827 |

- For every mode, HR@10 = coverage × HR@10 given coverage exactly, since a user whose target f is
  not kept can never hit.
- **Ties in the single-branch modes.** Test users with equal scores among ranks 1–11:
  - (a) text: 3,399 of 17,112 (19.9%);
  - (b) image: 6,360 (37.2%).
  - Validation: 3,338 and 5,992.
  - Cause: items that share (f, t1, t2) in mode (a), or (f, i1, i2) in mode (b), get identical
    scores, and ranking falls back to item id.
  - Those items are distinguished only by the other branch's codes, which the single-branch mode
    ignores.
  - Mode (c) inherits these ties; mode (d) uses all five codes, which are unique per item.

**f prediction, MFD vs chain decoder** (the chain's f = its first decoding step, restricted to
catalog f codes; same users):

| | test f top-1 | test f top-20 | valid f top-1 | valid f top-20 |
|---|---|---|---|---|
| MFD pass 1 | 0.0952 | 0.3461 | 0.1121 | 0.3815 |
| chain decoder, step 1 | 0.0992 | 0.3515 | 0.1143 | 0.3874 |
| chain decoder: target f among the f's of its 20 final beams | — | 0.2795 | — | 0.3184 |

- The chain's beams keep fewer distinct f than 20, so its effective coverage is 0.2795.
- The chain's HR@10 given that coverage is 0.4226 on test and 0.4635 on validation.
- The two "given coverage" numbers are on **different user sets**. The same-user comparison is
  §5.
- **Bug found and fixed in the paired script, no published number affected.**
  - The chain's `save_seqrec_20.json` stores users in sampler order, so `all_users[k]` = dataset
    index. The first `mfd_paired.py` run (job 21521214) stopped on its own order assertion.
  - The fix aligns users by `all_users` and checks every target. A second run (21523062) had a
    shadowed loop variable in the beam-f coverage check; it reported 0.026 instead of 0.2795.
  - The final run is 21528838.
  - `mfd_eval.py`'s own chain diagnostic reads each user's beams with that user's target, so it
    never depended on order.

---

## 4. Training cost vs the chain-decoder control

| | MFD stage 1 | MD chain control |
|---|---|---|
| training examples per epoch | 84,890 | 84,890 |
| decoder sequences per epoch | 169,780 (2 per example, 4 positions each = 8 decoder positions) | 84,890 (1 per example, 6 positions: 5 SID tokens + EOS) |
| optimizer steps | 16,102 (194 epochs; early stop, patience 10) | 16,019 (193 epochs; early stop) |
| best checkpoint | epoch 184 (step 15,272) | epoch 183 (step 15,189) |
| training wall-clock (`train_runtime`) | 5,264 s on an H200 (job 21515330, alone on the GPU) | 4,256 s on an H100, sharing the GPU with packed runs |
| steps / s | 3.15 | 3.90 |

- Both runs used the same number of epochs and examples, and early-stopped within one epoch of
  each other.
- MFD processes 2× the decoder sequences, and about 1.33× the decoder positions (8 vs 6), per
  example. The encoder runs once in both.
- The wall-clock times come from different GPUs and contention levels, so they are **not
  comparable**. The per-step cost is roughly 1.2–1.3× the chain's: steps/s is 19% lower even on
  the faster GPU.
- The decoder is a small part of a 13M-parameter T5 step, so the training cost is of the same
  order.
- The job's total of 1 h 30 min includes the untrained sanity checks (about 1 min) and loading.

---

## 5. What limits the result: coverage

**Verdict.** The binding constraint is **coverage, set by how well the shared code f is
predicted**, and it binds equally hard for both decoders.
- Ranking given coverage is at the chain's level: slightly behind on the same users, but too
  little to decide the outcome.
- All numbers below are test; validation shows the same pattern (`paired_valid.json`).
- Source: `eval_kf20/paired_test.json`. In this file the target-f rank comes from each model's own
  first step (MFD pass 1; chain decoding step 1).

**1. Most users are lost at f, and both models lose them.**

| target f rank (own model) | MFD users | MFD (d) HR@10 | chain users | chain HR@10 |
|---|---|---|---|---|
| 1 | 1,629 | 0.749 | 1,697 | 0.768 |
| 2–5 | 1,381 | 0.226 | 1,366 | 0.240 |
| 6–20 | 2,913 | 0.165 | 2,952 | 0.132 |
| > 20 | **11,189 (65.4%)** | **0** | **11,097 (64.9%)** | **0** |

- For 65% of test users the target's shared code is not among either model's top 20, and neither
  model ever recovers those users. The chain gets only 1.6% HR@10 on users MFD does not cover.
- MFD's coverage of 0.346 caps its HR@10 at 0.346. It reaches 0.118.
- The f head is equally good in both models:

  | | f top-1 | f top-20 |
  |---|---|---|
  | MFD | 0.095 | 0.346 |
  | chain | 0.099 | 0.352 |

  The target f is in both models' top 20 for 27.9% of users.

**2. Raising K_f is not the fix.**
- HR@10 falls steeply with the target f's rank: 0.75 at rank 1, 0.23 at ranks 2–5, 0.17 at
  ranks 6–20.
- Users whose f would only enter at ranks beyond 20 carry an even lower log P(f), so they would
  rarely reach the top 10. More K_f mostly buys pool size, not hits.
- The constraint is the f prediction itself: top-1 accuracy is only 0.095 over 256 codes.

**3. Ranking given coverage, on the same users.** Users whose target f is in MFD's top 20
**and** among the f's of the chain's 20 final beams: 3,989 test users (4,682 valid).

| | chain | MFD (a) text | MFD (b) image | MFD (d) joint |
|---|---|---|---|---|
| test HR@10, same users | 0.463 | 0.455 | 0.453 | 0.450 |
| valid HR@10, same users | 0.505 | 0.494 | 0.492 | 0.488 |

- On identical users, MFD's ranking is 0.008–0.017 behind the chain.
- MFD makes this up by **covering more users**: 5,923 users vs the chain's 4,782 (target f among
  its beams). The chain's beam search drops f codes that MFD keeps and scores exactly.
- On the users MFD covers, MFD's HR@10 is 0.340 against the chain's 0.312.
- The two effects cancel to the null result in §3.

**What this implies for the next stage.**
- Better f prediction or a better ranker (fusion, calibration) moves both models alike.
- What separates MFD from the chain is a small within-coverage ranking deficit, against a cost
  that is about 3.8× lower at decode time.

---

## 6. Open items
- One seed. MFD vs chain needs 3 seeds (42/43/44) and the paired CI per seed, per the seed
  policy (`AGENTS.md` §1).
- K_f = 20 was fixed by the GOAL. Done: the K_f × weight sweep (chosen on validation) is Part B, §9.
- Arts: done in Part C (§10). Games is not run yet.

---

# Parts A–C (2026-10-31): 1+3+3, wider search, multi-view augmentation

Seed 42 throughout, locked v1 SID sets (hash checked before every training and evaluation run).
Everything on test was evaluated once, with settings chosen on validation.

Runs (all under `mmgen_runs/`):
- Part A: `mfd/Instruments/mfd133_stage1/seed42/` (pack-train job 21531817; eval 21544744 → `eval_kf20/`;
  sweep 21544747 → `sweep/`).
- Part B: `mfd/Instruments/mfd122_stage1/seed42/sweep/` (job 21531816) and the 1+3+3 sweep above.
- Part C: `mfd/{Instruments,Arts}/mfd122_mv/seed42/`, `mfd/Arts/mfd122_stage1/seed42/` and
  `baselines/{Instruments,Arts}/mdmv122/seed42/` (pack-train jobs 21531817, 21531818; evals 21544748,
  21545630–33, 21548914–17, 21548934/36; Part B setting applied in `sweep_applied/`).
- Paired bootstrap of all comparisons: `mfd/paired_parts.json` (`sbatch/tools/mfd_paired_parts.py`, CPU job
  `sbatch/mfd/paired_parts.sbatch`; 2,000 resamples over the same users, joint mode).
- Tables: `sbatch/tools/mfd_parts_table.py` → `mfd/parts_tables.json`.
- Code: `macrec/mfd_data.py` / `mfd_eval.py` (any 1+L+L SID), `macrec/mfd_sweep.py`, `macrec/mv_aug_data.py`,
  `sbatch/mfd/{pack_mfd_train.sbatch,run_mfd_train.sh,sweep_mfd.sbatch}`.

## 7. Summary

| | Instruments test HR@10 / NDCG@10 | Arts test HR@10 / NDCG@10 |
|---|---|---|
| MD 1+2+2 chain (6 passes, ~25 ms/user) | 0.1181 / 0.0932 | 0.1059 / 0.0739 |
| MD 1+2+2 chain + multi-view aug | 0.1188 / 0.0940 | 0.1061 / 0.0736 |
| MFD 1+2+2, K_f = 20, w = 1 (stage 1 setting; 3 passes, 7–8 ms/user) | 0.1177 / 0.0923 | 0.0990 / 0.0685 |
| MFD 1+2+2, Part B setting (K_f = 100, w = 1 / 0.5 / 0.5; 29–37 ms/user) | 0.1210 / 0.0936 | 0.1044 / 0.0705 |
| MFD 1+2+2 + multi-view aug, K_f = 20 | 0.1208 / 0.0948 | 0.1023 / 0.0710 |
| **MFD 1+2+2 + multi-view aug, Part B setting** | **0.1237 / 0.0959** | **0.1066 / 0.0729** |
| MFD 1+3+3, K_f = 20 (4 passes, 21.4 ms/user) | 0.1196 / 0.0931 | — |
| MFD 1+3+3, its own Part B setting (K_f = 256, w = 1 / 1 / 0.5; 169 ms/user) | 0.1221 / 0.0938 | — |
| MD 1+3+3 chain (8 passes, 32.0 ms/user) | 0.1161 / 0.0916 | — |
| D MACRec best single list / merged | 0.1302 / 0.1006, 0.1321 / 0.1021 | 0.1236 / 0.0883, 0.1277 / 0.0912 |

- **Instruments:** MFD + augmentation at the Part B setting beats the chain decoder by +0.0056 HR@10
  [95% CI +0.0025, +0.0088] and +0.0027 NDCG@10 [+0.0010, +0.0043]. The gap holds on validation
  (+0.0065 [+0.0033, +0.0098]). It also beats the augmented chain (+0.0049 [+0.0020, +0.0081]).
- **Arts:** MFD does not beat the chain. Plain MFD at K_f = 20 is clearly worse (−0.0069 [−0.0101, −0.0040]).
  The Part B setting plus augmentation only closes the gap: +0.0006 [−0.0024, +0.0036], a tie.
- **1+3+3 (Part A):** MFD beats the 1+3+3 chain at K_f = 20 (+0.0035 [+0.0007, +0.0062]) with half the
  decoder passes (4 vs 8) and 0.67× the time (21.4 vs 32.0 ms/user). On validation the gap is +0.0012
  [−0.0016, +0.0040], so treat it as weak. 1+3+3 does not beat MFD 1+2+2 at its own K_f.
- **Wider search (Part B)** gains +0.003 to +0.005 HR@10 over K_f = 20, w = 1 for every MFD model (paired CIs
  exclude 0 on test for the three non-augmented models). Most of that comes from the weights, not from K_f (§9).
- **Augmentation (Part C)** helps MFD a little: +0.003 HR@10 on both datasets at K_f = 20 (test CIs just exclude 0;
  validation +0.0025 and −0.0003, not significant) and +0.0025 NDCG@10 on both (significant). It does not help the
  chain (+0.0007 Instruments, +0.0001 Arts). It barely moves f prediction (§10).
- **The C/D gap is still open:** best MFD vs D's best single list is −0.0065 HR@10 on Instruments and −0.0170
  on Arts.
- **Caveat:** one seed. The Instruments MFD-vs-chain result needs seeds 43/44 before it is a claim.

## 8. Part A: MFD on the 1+3+3 SID (Instruments)

**Setup.** Exactly the stage-1 recipe (§1.1) on the locked 1+3+3 v1 set; only the SID changes.
- Each branch is a 3-code chain: text [start, f, <T>, t1, t2] → [f, −, t1, t2, t3]; image likewise with f
  not scored. Decoder sequence length 5 instead of 4.
- Inference: 1 + L = **4 decoder passes**. Pass 1 scores f, pass 2 runs [start, f, <B>] for the kept K_f
  codes, passes 3 and 4 score every **catalog prefix** (f, c1) and (f, c1, c2) of each branch in one batch.
  Scoring stays exact over the catalog. At K_f = 20 that is 2,916 decoder sequences per user (passes 3–4: 1,460
  text and 1,415 image prefixes), against 1,195 for 1+2+2 (585 + 569 prefixes in pass 3).
- Training: 16,600 optimizer steps (all 200 epochs, best checkpoint 16,268), 84,890 examples and 169,780
  decoder sequences per epoch, 8,693 s on a shared H100 (packed with the Arts run; wall-clock contended).
  The 1+3+3 chain control: 16,600 steps, 5,107 s.

**Sanity checks** (`sanity_untrained/sanity.json` before training, `eval_kf20/sanity.json` on the trained model):
all pass.
- Branch independence: changing the text branch's code leaves the image branch's scores identical
  (max |diff| 0.0) and vice versa.
- f is scored once: 7 targets per example (1 + 3 + 3), the image branch's f position ignored.
- Exact scoring: 22 (user, item) pairs re-scored by teacher forcing, max |diff| 2.3e-5 (tolerance 1e-4).

**Results** (Instruments, test with validation next to it; ms/user = GPU time on an H200):

| method | test HR@10 / NDCG@10 | valid HR@10 / NDCG@10 | decoder passes / user | ms / user |
|---|---|---|---|---|
| MFD 1+3+3 (text) | 0.1171 / 0.0915 | 0.1435 / 0.1103 | 4 | 21.4 (H200) |
| MFD 1+3+3 (image) | 0.1185 / 0.0925 | 0.1443 / 0.1103 | 4 | 21.4 (H200) |
| MFD 1+3+3 (merge) | 0.1194 / 0.0926 | 0.1445 / 0.1106 | 4 | 21.4 (H200) |
| MFD 1+3+3 (joint) | 0.1196 / 0.0931 | 0.1448 / 0.1110 | 4 | 21.4 (H200) |
| MD 1+3+3 chain | 0.1161 / 0.0916 | 0.1435 / 0.1105 | 8 | 32.0 (H200) |
| MFD 1+2+2 (text) | 0.1170 / 0.0919 | 0.1456 / 0.1130 | 3 | 6.7 (H200) |
| MFD 1+2+2 (image) | 0.1169 / 0.0919 | 0.1448 / 0.1119 | 3 | 6.7 (H200) |
| MFD 1+2+2 (merge) | 0.1183 / 0.0926 | 0.1483 / 0.1138 | 3 | 6.7 (H200) |
| MFD 1+2+2 (joint) | 0.1177 / 0.0923 | 0.1460 / 0.1132 | 3 | 6.7 (H200) |
| MD 1+2+2 chain | 0.1181 / 0.0932 | 0.1476 / 0.1132 | 6 | 25.3 (H200) |

- MFD 1+3+3 joint vs the 1+3+3 chain, paired: test +0.0035 HR@10 [+0.0007, +0.0062], NDCG +0.0014
  [−0.0002, +0.0030]; valid +0.0012 [−0.0016, +0.0040].
- f prediction is the same as the chain's: test f top-1 0.093 vs 0.094, top-20 0.359 vs 0.357.
- Cost: 4 passes and 21.4 ms/user vs 8 passes and 32.0 ms/user for the chain, peak 14.1 GB. MFD 1+3+3 is 3.2×
  slower than MFD 1+2+2 (6.7 ms) because passes 3–4 run over all catalog prefixes of depth 1 and 2.
- 1+3+3 vs 1+2+2 MFD: +0.0019 HR@10 on test, −0.0012 on validation. No evidence the deeper SID helps MFD.

## 9. Part B: wider search, inference only

**Method.**
- score = w_f · log P(f) + w_t · (text-branch log-probs) + w_i · (image-branch log-probs), over every catalog
  item whose f is among the kept K_f.
- Grid: K_f ∈ {20, 50, 100, all 256} × w_f ∈ {0.5, 0.75, 1.0} × w_t, w_i ∈ {0.5, 1.0, 1.5} (108 joint settings,
  plus text-only and image-only at w = 1).
- One exact pass at K_f = 256 on **validation** gives every item's three terms and its f's rank; each setting
  is then ranked by masking items whose f rank ≥ K_f. This is identical to running inference at that K_f:
  the real K_f path was also run for each K_f, and its ranks matched the sweep's for every user (0
  mismatches at every K_f, both models).
- ONE setting chosen: max valid HR@10, tie-break NDCG@10. Test was run once, with that setting and the
  stage-1 reference. Timing and peak memory come from the real K_f path on an H200 (batch 64).

### 9.1 Instruments 1+2+2 (stage-1 model)

| K_f | w_f / w_t / w_i | valid HR@10 / NDCG@10 | coverage | HR@10 among covered | ms / user (H200) | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| 20 | 1 / 1 / 1 | 0.1460 / 0.1132 | 0.381 | 0.383 | 6.6 | 10.9 |
| 20 | 1.0 / 0.5 / 0.5 (best at this K_f) | 0.1487 / 0.1140 | 0.381 | 0.390 | 6.6 | 10.9 |
| 50 | 1 / 1 / 1 | 0.1469 / 0.1135 | 0.586 | 0.251 | 15.3 | 11.0 |
| 50 | 1.0 / 0.5 / 0.5 (best at this K_f) | 0.1497 / 0.1145 | 0.586 | 0.255 | 15.3 | 11.0 |
| 100 | 1 / 1 / 1 | 0.1472 / 0.1135 | 0.785 | 0.188 | 28.6 | 11.2 |
| 100 | 1.0 / 0.5 / 0.5 (best at this K_f) | 0.1498 / 0.1145 | 0.785 | 0.191 | 28.6 | 11.2 |
| 256 | 1 / 1 / 1 | 0.1471 / 0.1135 | 1.000 | 0.147 | 58.7 | 11.6 |
| 256 | 1.0 / 0.5 / 0.5 (best at this K_f) | 0.1498 / 0.1145 | 1.000 | 0.150 | 58.7 | 11.6 |

Text-only / image-only (w = 1) per K_f, valid HR@10: K_f=20: text 0.1456, image 0.1448; K_f=50: text 0.1477, image 0.1459; K_f=100: text 0.1477, image 0.1462; K_f=256: text 0.1477, image 0.1462

Chosen on validation: **K100_wf1.0_wt0.5_wi0.5** (valid 0.1498 / 0.1145).

HR@10 by the target f's rank (users in brackets):

| setting | 1 | 2-5 | 6-20 | 21-50 | 51-100 | 101-256 |
|---|---|---|---|---|---|---|
| valid, K_f=20, w=1 | 0.764 (1918) | 0.313 (1550) | 0.179 (3060) | 0.000 (3504) | 0.000 (3395) | 0.000 (3685) |
| valid, chosen | 0.804 (1918) | 0.315 (1550) | 0.132 (3060) | 0.034 (3504) | 0.002 (3395) | 0.000 (3685) |
| test, K_f=20, w=1 | 0.749 (1629) | 0.226 (1381) | 0.165 (2913) | 0.000 (3505) | 0.000 (3577) | 0.000 (4107) |
| test, chosen | 0.786 (1629) | 0.232 (1381) | 0.120 (2913) | 0.033 (3505) | 0.001 (3577) | 0.000 (4107) |

Test (the chosen setting evaluated once):

| MFD setting | test HR@10 / NDCG@10 | coverage | HR@10 among covered | passes | ms / user (H200) | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| K_f=20, w=1 | 0.1177 / 0.0923 | 0.346 | 0.340 | 3 | 6.8 | 10.9 |
| chosen K100_wf1.0_wt0.5_wi0.5 | 0.1210 / 0.0936 | 0.760 | 0.159 | 3 | 29.2 | 11.2 |

- Paired, test: chosen vs K_f = 20, w = 1: +0.0033 HR@10 [+0.0011, +0.0054]; chosen vs the chain: +0.0029
  [−0.0004, +0.0060] (not significant).
- **The gain is from the weights, not from K_f.** At w = 1, K_f 20 → 256 moves valid HR@10 only 0.1460 → 0.1471
  while coverage goes 0.38 → 1.00. At K_f = 20, the chosen weights alone reach 0.1487 (of the 0.1498 total).
  Halving the branch weights means trusting f more: the biggest change is for users whose f is ranked 1
  (valid HR@10 0.764 → 0.804).
- Newly covered users are almost never hit: f rank 21–50 → 0.034, 51–100 → 0.002, 101–256 → 0.
- Cost: K_f = 100 is 4.3× the time of K_f = 20 (28.6 vs 6.6 ms/user, valid) for the K_f part of the gain;
  the weights are free.

### 9.2 Instruments 1+3+3 (Part A model)

| K_f | w_f / w_t / w_i | valid HR@10 / NDCG@10 | coverage | HR@10 among covered | ms / user (H200) | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| 20 | 1 / 1 / 1 | 0.1448 / 0.1110 | 0.386 | 0.375 | 20.9 | 14.1 |
| 20 | 1.0 / 0.5 / 1.5 (best at this K_f) | 0.1457 / 0.1112 | 0.386 | 0.377 | 20.9 | 14.1 |
| 50 | 1 / 1 / 1 | 0.1477 / 0.1117 | 0.593 | 0.249 | 48.2 | 14.4 |
| 50 | 0.75 / 0.5 / 0.5 (best at this K_f) | 0.1481 / 0.1120 | 0.593 | 0.250 | 48.2 | 14.4 |
| 100 | 1 / 1 / 1 | 0.1470 / 0.1115 | 0.796 | 0.185 | 88.0 | 14.7 |
| 100 | 0.75 / 0.5 / 0.5 (best at this K_f) | 0.1486 / 0.1122 | 0.796 | 0.187 | 88.0 | 14.7 |
| 256 | 1 / 1 / 1 | 0.1471 / 0.1116 | 1.000 | 0.147 | 165.2 | 15.5 |
| 256 | 1.0 / 1.0 / 0.5 (best at this K_f) | 0.1488 / 0.1124 | 1.000 | 0.149 | 165.2 | 15.5 |

Text-only / image-only (w = 1) per K_f, valid HR@10: K_f=20: text 0.1435, image 0.1443; K_f=50: text 0.1456, image 0.1453; K_f=100: text 0.1460, image 0.1453; K_f=256: text 0.1462, image 0.1455

Chosen on validation: **K256_wf1.0_wt1.0_wi0.5** (valid 0.1488 / 0.1124).

HR@10 by the target f's rank (users in brackets):

| setting | 1 | 2-5 | 6-20 | 21-50 | 51-100 | 101-256 |
|---|---|---|---|---|---|---|
| valid, K_f=20, w=1 | 0.745 (1865) | 0.297 (1592) | 0.195 (3150) | 0.000 (3545) | 0.000 (3474) | 0.000 (3486) |
| valid, chosen | 0.747 (1865) | 0.276 (1592) | 0.144 (3150) | 0.053 (3545) | 0.019 (3474) | 0.002 (3486) |
| test, K_f=20, w=1 | 0.746 (1597) | 0.223 (1385) | 0.173 (3159) | 0.000 (3612) | 0.000 (3652) | 0.000 (3707) |
| test, chosen | 0.746 (1597) | 0.196 (1385) | 0.124 (3159) | 0.042 (3612) | 0.019 (3652) | 0.003 (3707) |

Test (the chosen setting evaluated once):

| MFD setting | test HR@10 / NDCG@10 | coverage | HR@10 among covered | passes | ms / user (H200) | peak GPU mem (GB) |
|---|---|---|---|---|---|---|
| K_f=20, w=1 | 0.1196 / 0.0931 | 0.359 | 0.333 | 4 | 21.4 | 14.1 |
| chosen K256_wf1.0_wt1.0_wi0.5 | 0.1221 / 0.0938 | 1.000 | 0.122 | 4 | 168.6 | 15.5 |

- Paired, test: chosen vs K_f = 20, w = 1: +0.0026 HR@10 [+0.0003, +0.0052]; vs the 1+3+3 chain: +0.0061
  [+0.0029, +0.0091].
- Here K_f matters more than the weights: at w = 1, K_f 20 → 50 gives 0.1448 → 0.1477 on validation; the
  best weights at K_f = 20 give only 0.1457. Users with f rank 21–50 reach 0.053 (vs 0.034 for 1+2+2).
- **The chosen setting is expensive:** K_f = 256 costs 165–169 ms/user (8× K_f = 20) and 15.5 GB. On validation
  K_f = 50 (w = 0.75 / 0.5 / 0.5) is within 0.0007 HR@10 of it at 48 ms/user. The rule picks the maximum, so
  0.1221 is the reported number; a cost-constrained rule would pick K_f = 50.

### 9.3 Chosen settings next to stage 1 and the chain (test)

| model | stage 1 (K_f = 20, w = 1) | Part B chosen (valid) | test with chosen setting | ms/user | chain (passes, ms/user) |
|---|---|---|---|---|---|
| MFD 1+2+2 | 0.1177 / 0.0923 | K_f = 100, w = 1 / 0.5 / 0.5 | **0.1210 / 0.0936** | 29.2 | 0.1181 / 0.0932 (6, 25.3) |
| MFD 1+3+3 | 0.1196 / 0.0931 | K_f = 256, w = 1 / 1 / 0.5 | **0.1221 / 0.0938** | 168.6 | 0.1161 / 0.0916 (8, 32.0) |

## 10. Part C: multi-view history augmentation

**Setup.**
- Every training example appears three times per epoch: history as the full view [f t1 t2 i1 i2], as the
  text view [f t1 t2], and as the image view [f i1 i2]. The view is per sequence (the whole history). The
  target is always the full SID. Validation and test use the full view. No item-level translation tasks.
- Chain: `mv_aug_data.MVSeqRecDataset` (baseline `mdmv122`); MFD: `mfd_data.MVMFDDataset` (task `seqrec_mv`).
- Everything else identical to the controls (same recipe, early stopping on validation loss with patience 10,
  best checkpoint by validation loss). The untrained-model sanity checks pass for both augmented MFD runs.
- MFD is evaluated at K_f = 20 (w = 1) and at the setting chosen in Part B on Instruments 1+2+2
  (K_f = 100, w = 1 / 0.5 / 0.5), applied unchanged; Arts was **not** re-tuned.

**Training cost** (seed 42; wall-clock on shared H100s, packed with other runs, so only roughly comparable):

| run | examples / epoch | decoder sequences / epoch | optimizer steps | epochs (best) | train s |
|---|---|---|---|---|---|
| Instruments chain | 84,890 | — | 16,019 | 193 (183) | 4,256 |
| Instruments chain + aug | 254,670 | — | 14,632 | 59 (49) | 6,345 |
| Instruments MFD | 84,890 | 169,780 | 16,102 | 194 (184) | 5,264 (H200) |
| Instruments MFD + aug | 254,670 | 509,340 | 15,128 | 61 (51) | 8,319 |
| Arts chain | 107,566 | — | 19,716 | 186 (176) | 5,150 |
| Arts chain + aug | 322,698 | — | 16,748 | 53 (43) | 6,899 |
| Arts MFD | 107,566 | 215,132 | 18,656 | 176 (166) | 6,836 |
| Arts MFD + aug | 322,698 | 645,396 | 17,696 | 56 (46) | 8,862 |

Augmentation triples the examples per epoch; early stopping then fires after about a third of the epochs, so
the optimizer steps (and examples seen) end up 5–15% **below** the controls.

### 10.1 Instruments (1+2+2)

| method | test HR@10 / NDCG@10 | valid HR@10 / NDCG@10 | decoder passes / user | ms / user |
|---|---|---|---|---|
| MD chain | 0.1181 / 0.0932 | 0.1476 / 0.1132 | 6 | 25.3 (H200) |
| MD chain + multi-view aug | 0.1188 / 0.0940 | 0.1479 / 0.1145 | 6 | 25.0 (H200) |
| MFD joint, K_f=20, w=1 | 0.1177 / 0.0923 | 0.1460 / 0.1132 | 3 | 6.7 (H200) |
| MFD merge, K_f=20 | 0.1183 / 0.0926 | 0.1483 / 0.1138 | 3 | 6.7 (H200) |
| MFD joint, Part B setting (K100_wf1.0_wt0.5_wi0.5) | 0.1210 / 0.0936 | 0.1498 / 0.1145 | 3 | 29.2 (H200) |
| MFD + multi-view aug joint, K_f=20, w=1 | 0.1208 / 0.0948 | 0.1484 / 0.1140 | 3 | 6.8 (H200) |
| MFD + multi-view aug merge, K_f=20 | 0.1223 / 0.0953 | 0.1515 / 0.1151 | 3 | 6.8 (H200) |
| MFD + multi-view aug joint, Part B setting (K100_wf1.0_wt0.5_wi0.5) | 0.1237 / 0.0959 | 0.1540 / 0.1160 | 3 | 29.2 (H200) |
| C MQL4GRec, best single list (image) | 0.1271 / 0.0984 | 0.1581 / 0.1203 | 5 | 21.6 (H200) |
| C MQL4GRec, merged | 0.1301 / 0.0998 | 0.1598 / 0.1206 | 2 × 5 | 43.3 (H200) |
| D MACRec, best single list (image) | 0.1302 / 0.1006 | 0.1591 / 0.1202 | 5 | 21.7 (H200) |
| D MACRec, merged | 0.1321 / 0.1021 | 0.1589 / 0.1207 | 2 × 5 | 43.2 (H200) |

f prediction ± augmentation (Instruments):

| model | test f top-1 | test f top-20 | valid f top-1 | valid f top-20 |
|---|---|---|---|---|
| MFD | 0.0952 | 0.3461 | 0.1121 | 0.3815 |
| MFD + aug | 0.0996 | 0.3567 | 0.1172 | 0.3948 |
| chain | 0.0992 | 0.3515 | 0.1143 | 0.3874 |
| chain + aug | 0.1020 | 0.3569 | 0.1176 | 0.3936 |

### 10.2 Arts (1+2+2)

| method | test HR@10 / NDCG@10 | valid HR@10 / NDCG@10 | decoder passes / user | ms / user |
|---|---|---|---|---|
| MD chain | 0.1059 / 0.0739 | 0.1244 / 0.0867 | 6 | 24.9 (H200) |
| MD chain + multi-view aug | 0.1061 / 0.0736 | 0.1252 / 0.0860 | 6 | 25.7 (H200) |
| MFD joint, K_f=20, w=1 | 0.0990 / 0.0685 | 0.1206 / 0.0816 | 3 | 8.4 (H200) |
| MFD merge, K_f=20 | 0.0980 / 0.0678 | 0.1206 / 0.0813 | 3 | 8.4 (H200) |
| MFD joint, Part B setting (K100_wf1.0_wt0.5_wi0.5) | 0.1044 / 0.0705 | 0.1269 / 0.0839 | 3 | 37.3 (H200) |
| MFD + multi-view aug joint, K_f=20, w=1 | 0.1023 / 0.0710 | 0.1202 / 0.0834 | 3 | 8.4 (H200) |
| MFD + multi-view aug merge, K_f=20 | 0.1009 / 0.0706 | 0.1192 / 0.0831 | 3 | 8.4 (H200) |
| MFD + multi-view aug joint, Part B setting (K100_wf1.0_wt0.5_wi0.5) | 0.1066 / 0.0729 | 0.1253 / 0.0858 | 3 | 37.3 (H200) |
| C MQL4GRec, best single list (text) | 0.1189 / 0.0846 | 0.1420 / 0.0991 | 5 | 21.5 (H200) |
| C MQL4GRec, merged | 0.1267 / 0.0897 | 0.1457 / 0.1026 | 2 × 5 | 43.0 (H200) |
| D MACRec, best single list (text) | 0.1236 / 0.0883 | 0.1419 / 0.0997 | 5 | 26.6 (A100-SXM4-80GB) |
| D MACRec, merged | 0.1277 / 0.0912 | 0.1470 / 0.1041 | 2 × 5 | 53.6 (A100-SXM4-80GB) |

- The Arts D runs were evaluated on an A100, so their ms/user is not comparable with the H200 rows.
- MFD rows at the Part B setting use the setting chosen on Instruments 1+2+2 validation, not re-tuned on Arts.

f prediction ± augmentation (Arts):

| model | test f top-1 | test f top-20 | valid f top-1 | valid f top-20 |
|---|---|---|---|---|
| MFD | 0.0975 | 0.4000 | 0.1141 | 0.4306 |
| MFD + aug | 0.0981 | 0.4102 | 0.1152 | 0.4413 |
| chain | 0.0975 | 0.3984 | 0.1160 | 0.4316 |
| chain + aug | 0.0989 | 0.4115 | 0.1154 | 0.4390 |

**Paired differences (test; validation in brackets where the sign differs).**

| comparison | Instruments ΔHR@10 [95% CI] | Instruments ΔNDCG@10 | Arts ΔHR@10 [95% CI] | Arts ΔNDCG@10 |
|---|---|---|---|---|
| chain + aug − chain | +0.0007 [−0.0022, +0.0035] | +0.0008 [−0.0008, +0.0023] | +0.0001 [−0.0027, +0.0030] | −0.0002 [−0.0019, +0.0014] |
| MFD + aug − MFD, K_f = 20 | +0.0031 [+0.0001, +0.0061] | +0.0025 [+0.0008, +0.0041] | +0.0032 [+0.0006, +0.0060] (valid −0.0003) | +0.0025 [+0.0009, +0.0042] |
| MFD + aug − MFD, Part B setting | +0.0027 [−0.0001, +0.0057] | +0.0023 [+0.0006, +0.0040] | +0.0022 [−0.0007, +0.0051] (valid −0.0016) | +0.0024 [+0.0007, +0.0042] |
| MFD − chain, K_f = 20 | −0.0004 [−0.0036, +0.0026] | −0.0009 [−0.0028, +0.0008] | **−0.0069 [−0.0101, −0.0040]** | **−0.0053 [−0.0072, −0.0035]** |
| MFD − chain, Part B setting | +0.0029 [−0.0004, +0.0060] | +0.0004 [−0.0014, +0.0022] | −0.0015 [−0.0045, +0.0013] | **−0.0034 [−0.0051, −0.0016]** |
| MFD + aug − chain, K_f = 20 | +0.0027 [−0.0004, +0.0057] | +0.0015 [−0.0002, +0.0033] | **−0.0037 [−0.0069, −0.0007]** | **−0.0028 [−0.0048, −0.0010]** |
| MFD + aug − chain, Part B setting | **+0.0056 [+0.0025, +0.0088]** | **+0.0027 [+0.0010, +0.0043]** | +0.0006 [−0.0024, +0.0036] | −0.0010 [−0.0028, +0.0007] |
| MFD + aug − chain + aug, Part B setting | **+0.0049 [+0.0020, +0.0081]** | **+0.0019 [+0.0005, +0.0036]** | +0.0005 [−0.0024, +0.0034] | −0.0007 [−0.0025, +0.0009] |

Bold = CI excludes 0. Full list, including validation: `mfd/paired_parts.json`.

**f prediction ± augmentation.** Augmentation was meant to make the history encoder use each modality alone
and so predict f better. It does so only marginally:
- f top-20 rises by +0.005 to +0.013 (Instruments MFD 0.346 → 0.357, chain 0.352 → 0.357; Arts MFD 0.400 → 0.410,
  chain 0.398 → 0.412). Validation shows the same.
- f top-1 rises by at most +0.004 (MFD Instruments 0.095 → 0.100, chain 0.099 → 0.102); Arts +0.001 for both.
- So the MFD gain from augmentation is mostly in ranking within the pool (NDCG +0.0025 on both datasets),
  not in reaching more targets.

## 11. What limits the result now

1. **f prediction, unchanged by everything tried.** Across all ten models (both datasets, ± augmentation, 1+3+3) the target f is ranked first for only
   9.3–10.2% of test users and is in the top 20 for 35–41%. Augmentation adds ≤ 0.013 to top-20 and ≤ 0.004 to
   top-1. Hits concentrate where f is right: with the chosen 1+2+2 setting on Instruments test, the 9.5% of users
   whose f is ranked 1 give 62% of all hits (1,280 of ~2,070; HR@10 0.786 in that bucket vs 0.232 at ranks 2–5,
   0.120 at 6–20).
2. **Widening K_f does not convert coverage into hits.** At K_f = 256 coverage is 1.0, yet the 60–65% of users
   whose f ranks beyond 20 reach HR@10 0.000–0.053 (1+2+2: ≤ 0.034). For 1+2+2 the whole K_f effect is +0.001
   on validation at w = 1; for 1+3+3 it is +0.003 but costs 8× decode time (169 vs 21 ms/user). The items under
   a low-ranked f are outscored by items under high-ranked f's.
3. **On Arts, ranking inside the pool is the extra problem.** Plain MFD and the chain predict f equally (top-20
   0.400 vs 0.398, top-1 0.0975 both), yet MFD is −0.0069 HR@10 and −0.0053 NDCG@10 behind. Even users whose f
   is ranked first reach only HR@10 0.522 on Arts (Instruments 0.749): Arts has 9,416 items over 256 f codes
   (37 per code vs 24). A plausible reading is that MFD's two branch scores, added without the chain's t → i
   edge, separate those items less well; the f-rank-1 bucket is where this shows.
   Down-weighting the branches (Part B) lifts that bucket to 0.602 and recovers HR@10, but NDCG stays −0.0034
   [−0.0051, −0.0016] behind without augmentation.
4. **The remaining gap to C/D is training, not decoding.** Best MFD vs D's best single list: −0.0065 HR@10
   (Instruments), −0.0170 (Arts). `docs/MDRQVAE_REPORT.md` §5.2 attributes the whole C/D lead to C's multi-task
   recipe (+0.019 to +0.031); multi-view history augmentation, which only re-views the input, recovers +0.003.
5. **One seed.** Every MFD-vs-chain difference here is 0.003–0.007 HR@10 with CIs ±0.003; the Instruments win
   needs seeds 43/44 before it is reported as a result.

## 12. Open items after Parts A–C
- Seeds 43/44 for: MFD + aug (Part B setting) vs chain on Instruments and Arts.
- The 1+3+3 sweep chose K_f = 256 (169 ms/user); a selection rule with a cost cap (e.g. ≤ chain time) should be
  fixed before the multi-seed runs.
- Games is not run.
- f prediction is the binding limit: the next lever has to change it (e.g. a dedicated f objective or the C-style
  multi-task recipe on top of MFD), not the search.

## 13. MFD with MQL4GRec's task mix (versions A, B, C; Instruments)

Written 2026-11-02. Instruments, MD-RQ-VAE 1+2+2 v1 SID (hash verified at every job), seed 42. The question: does C's
multi-task recipe (which carries the whole C/D lead over single-task decoders, `docs/MDRQVAE_REPORT.md` §5.2) help MFD?
Model, T5 config, optimizer, batch, LR, schedule and early stopping are mfd122_mv's; only the task mix and the prompts change.

Runs: `mmgen_runs/mfd/Instruments/mfd122_mt{A,B,C}/seed42/`. Code: `macrec/mfd_mt_data.py`, `mfd_mt_check.py`,
`mfd_dryrun.py`, `sbatch/mfd/mt_gate.sbatch`, `eval_mt.sbatch`, `mt_paired.sbatch`, `sbatch/tools/mfd_mt_config_diff.py`,
`mfd_mt_real_diff.py`, `mfd_mt_paired.py`. Training: A/B pack 21588791, C pack 21598855 (with mfd122_mv seeds 43/44; C alone
kept the GPU 27% busy, so the first C job 21591694 was stopped at epoch 1 and repacked). Evals 21591177/8, 21598856 (H200).
Paired CIs: `mmgen_runs/mfd/Instruments/mt_paired.json`.

### 13.1 Task mapping and naming

**Naming.** Our full-view task is called **fullseqrec** in this report (code identifier `mt_fusionseqrec`, kept because
runs were trained with it). It is **not** C's fusionseqrec: C's fusionseqrec is the cross-modal coin-flip task, implemented
here as **`mt_crossseqrec`** (version C only).

| C task (MQL4GRec `data.py`) | C definition | MFD task (code id) | MFD version | in |
|---|---|---|---|---|
| seqrec | bare text-SID history → next text SID, no prompt | mt_seqrec | text-view history [f t1 t2]… → text branch | A, B, C |
| seqimage | bare image-SID history → next image SID, no prompt | mt_seqimage | image-view history [f i1 i2]… → image branch | A, B, C |
| (none) | – | **fullseqrec** (mt_fusionseqrec) | soft prompt + full-view history → both branches (the inference task) | A, B, C |
| item2image | prompt + item text SID → its image SID | mt_item2image | prompt + item text view → image branch | B |
| image2item | prompt + item image SID → its text SID | mt_image2item | prompt + item image view → text branch | B |
| fusionseqrec | prompt + text history → image SID, or (coin flip per example) image history → text SID | **mt_crossseqrec** | the same on MFD's branches | C |

Deviations from C's recipe:
- fullseqrec takes fusionseqrec's slot (prompt, example share) in A and B (decided with the user, 2026-11-01): MFD
  decodes both branches from the full view at inference, which C's fusionseqrec never trains. C's cross-modal direction
  is therefore absent from A and B; version C adds it as an extra task with its own prompt.
- f is a target once per example (text row, position 0); single-branch tasks mask the other branch's codes, not f.
- Prompts mirror C: none on the two sequence tasks, C's soft prompts (`finetune_contrastive.train`, task i → token ids
  100(i+1)…) on the others, rebuilt identically at evaluation.
- Validation / early stopping use only the inference task (fullseqrec on the valid split, eval_loss, patience 10), as
  mfd122_mv; mfd122_mv's validation input has no prompt.

### 13.2 Sanity checks and epoch size

All passed before training (`mmgen_runs/mfd/mt_gate/Instruments/`):
1. Branch independence: untrained-model check (`mfd_eval.py --sanity_only --random_init`) per run.
2. Inputs / targets (`mfd_mt_check.py`): printed sample per task; on 5,000 random examples per sequence task (all of
   each item task) the input is [prompt +] the right view and the labels carry loss on exactly the targeted branch plus
   f once. Version C's coin flip: text-history → image share 0.5032 (n = 5,000).
3. Examples per epoch per task = C's: 84,890 per sequence task, 6,250 per item task.
4. Config diff vs mfd122_mv (dry run vs dry run): only the task mix (tasks, soft prompts, examples) and run paths differ;
   soft prompts equal C's construction. The dry run does not reproduce real finetune_mfd runs in two TrainingArguments
   (`dataloader_num_workers` 0 vs 2, `prediction_loss_only`), so each real run was also compared with mfd122_mv's real
   run after its first checkpoint (`mfd_mt_real_diff.py`): only paths and `tasks` differ, for A, B and C.

| run | tasks | examples / epoch | optimizer steps / epoch | steps run | epochs run | best checkpoint (valid loss) |
|---|---|---|---|---|---|---|
| mfd122_mv | seqrec_mv (3 views) | 254,670 | 248 | 15,128 | 61 | 12,648 (2.1418) |
| A | seqrec, seqimage, fullseqrec | 254,670 | 248 | 17,360 | 70 | 14,880 (2.1636) |
| B | A + item2image, image2item | 267,170 | 261 | 15,399 | 59 | 12,789 (2.1967) |
| C | A + crossseqrec | 339,560 | 331 | 18,924 | 57 | 15,604 (2.1342) |

(Validation losses are on the same users and targets; mfd122_mv's input has no prompt, so its loss is only roughly comparable.)

### 13.3 Results (test; validation HR@10 in brackets)

Settings: (i) K_f = 20, w = 1; (ii) K_f = 100, w_f = 1, w_t = w_i = 0.5 (mfd122_mv's best); (iii) the 108-setting grid
chosen on VALIDATION, applied once to test. Joint scoring. H200, ms/user = decode time.

| model | setting | HR@10 | NDCG@10 | HR@1 / HR@5 / NDCG@5 | decoder passes / user | ms / user |
|---|---|---|---|---|---|---|
| A | (i) | 0.1196 [0.1483] | 0.0940 | 0.0751 / 0.0996 / 0.0875 | 3 | 6.8 |
| A | (ii) | 0.1227 [0.1511] | 0.0949 | 0.0745 / 0.1017 / 0.0882 | 3 | 29.8 |
| A | (iii) K100, w=.75/.5/.5 | 0.1224 [0.1517] | 0.0949 | 0.0742 / 0.1016 / 0.0882 | 3 | 29.8 |
| B | (i) | 0.1174 [0.1464] | 0.0926 | 0.0738 / 0.0992 / 0.0867 | 3 | 6.8 |
| B | (ii) | 0.1202 [0.1501] | 0.0933 | 0.0741 / 0.0982 / 0.0862 | 3 | 29.4 |
| B | (iii) K256, w=1/1/.5 | 0.1209 [0.1502] | 0.0936 | 0.0740 / 0.0986 / 0.0865 | 3 | 61.4 |
| C | (i) | 0.1221 [0.1491] | 0.0947 | 0.0747 / 0.1013 / 0.0880 | 3 | 6.8 |
| C | (ii) | **0.1245** [0.1541] | **0.0962** | 0.0750 / 0.1030 / 0.0893 | 3 | 29.8 |
| C | (iii) = (ii) | 0.1245 [0.1541] | 0.0962 | | 3 | 29.8 |
| mfd122_mv | (i) | 0.1208 [0.1484] | 0.0948 | 0.0757 / 0.1006 / 0.0883 | 3 | 6.8 |
| mfd122_mv | (ii) (its Part B choice) | 0.1237 [0.1540] | 0.0959 | 0.0756 / 0.1024 / 0.0891 | 3 | 29.4 |
| MD chain decoder 1+2+2 | beam 20 | 0.1181 [0.1476] | 0.0932 | | 6 | 25.3 |
| Plug-in 1+2+2 merged (MD SID in C's recipe) | | 0.1246 [0.1539] | 0.0975 | | 2 × 4 | 36.2 |
| C (MQL4GRec) seed 42 merged | | 0.1301 | 0.0998 | | 2 × 5 | 43.3 |

**Four scoring modes at (ii)** (test HR@10 / NDCG@10; text / image = one branch's score with f, merge = C's ensemble rule
over the two single-branch lists, joint = w_f s_f + w_t s_t + w_i s_i):

| model | text | image | merge | joint |
|---|---|---|---|---|
| A | 0.1124 / 0.0899 | 0.1128 / 0.0899 | 0.1143 / 0.0907 | 0.1227 / 0.0949 |
| B | 0.1104 / 0.0881 | 0.1090 / 0.0877 | 0.1115 / 0.0890 | 0.1202 / 0.0933 |
| C | 0.1158 / 0.0916 | 0.1143 / 0.0911 | 0.1180 / 0.0927 | 0.1245 / 0.0962 |
| mfd122_mv | 0.1135 / 0.0909 | 0.1135 / 0.0905 | 0.1161 / 0.0920 | 0.1237 / 0.0959 |

Joint scoring stays the best mode for every version (+0.007 to +0.010 HR@10 over merge).

**f accuracy and coverage (test; valid in brackets).**

| model | f top-1 | f top-20 = coverage at K_f=20 | coverage at K_f=100 | HR@10 among covered, (ii) |
|---|---|---|---|---|
| mfd122_mv | 0.0996 [0.1172] | 0.3567 [0.3948] | 0.7713 [0.7934] | 0.1604 |
| A | 0.0983 [0.1155] | 0.3469 [0.3835] | 0.7662 [0.7872] | 0.1601 |
| B | 0.0969 [0.1135] | 0.3400 [0.3780] | 0.7494 [0.7779] | 0.1604 |
| C | 0.0989 [0.1165] | 0.3544 [0.3908] | 0.7699 [0.7935] | 0.1617 |

No task mix improves f prediction; B's item tasks lower it (−0.003 top-1, −0.017 top-20).
Peak decode memory 10.9 GB at (i), 11.2 GB at (ii), every run (H200, batch 64).

### 13.4 Paired differences (2,000 paired resamples of the same users, 95% CI)

| comparison | test ΔHR@10 | test ΔNDCG@10 | valid ΔHR@10 |
|---|---|---|---|
| A (i) − mv (i) | −0.0012 [−0.0041, +0.0015] | −0.0008 [−0.0024, +0.0007] | −0.0001 [−0.0031, +0.0029] |
| A (ii) − mv (ii) | −0.0011 [−0.0040, +0.0017] | −0.0009 [−0.0027, +0.0006] | −0.0030 [−0.0060, +0.0001] |
| A (iii) − mv (ii) | −0.0013 [−0.0043, +0.0015] | −0.0010 [−0.0027, +0.0006] | −0.0023 [−0.0055, +0.0008] |
| B (i) − mv (i) | **−0.0034 [−0.0065, −0.0002]** | **−0.0022 [−0.0039, −0.0005]** | −0.0020 [−0.0053, +0.0011] |
| B (ii) − mv (ii) | **−0.0035 [−0.0065, −0.0006]** | **−0.0026 [−0.0044, −0.0010]** | **−0.0039 [−0.0072, −0.0006]** |
| B (iii) − mv (ii) | −0.0029 [−0.0060, +0.0002] | **−0.0023 [−0.0041, −0.0006]** | **−0.0039 [−0.0072, −0.0006]** |
| C (i) − mv (i) | +0.0013 [−0.0016, +0.0042] | −0.0000 [−0.0016, +0.0015] | +0.0006 [−0.0025, +0.0037] |
| C (ii) − mv (ii) | +0.0008 [−0.0021, +0.0038] | +0.0003 [−0.0013, +0.0019] | +0.0001 [−0.0029, +0.0032] |
| A (ii) − plug-in 1+2+2 merged | −0.0020 [−0.0050, +0.0010] | **−0.0026 [−0.0042, −0.0010]** | −0.0028 [−0.0060, +0.0002] |
| B (ii) − plug-in 1+2+2 merged | **−0.0044 [−0.0076, −0.0014]** | **−0.0042 [−0.0060, −0.0025]** | **−0.0037 [−0.0071, −0.0004]** |
| C (ii) − plug-in 1+2+2 merged | −0.0001 [−0.0031, +0.0029] | −0.0013 [−0.0030, +0.0003] | +0.0002 [−0.0029, +0.0035] |
| B (ii) − A (ii) | −0.0025 [−0.0054, +0.0006] | **−0.0017 [−0.0034, −0.0000]** | −0.0009 [−0.0039, +0.0022] |
| C (ii) − A (ii) | +0.0019 [−0.0008, +0.0046] | +0.0013 [−0.0003, +0.0028] | **+0.0030 [+0.0001, +0.0061]** |
| mv (ii) − plug-in 1+2+2 merged | −0.0009 [−0.0039, +0.0022] | −0.0016 [−0.0032, +0.0000] | +0.0002 [−0.0030, +0.0033] |

Bold = CI excludes 0. (i)/(iii) vs plug-in rows: `mt_paired.json`.

### 13.5 Decision and reading

**Arts rule** (a version must beat mfd122_mv's validation HR@10 at (ii), 0.1540, by more than 0.002): A 0.1511, B 0.1501,
C 0.1541. **Not triggered for any version; no Arts runs.**

- C's multi-task recipe does not transfer to MFD. A is a tie with mfd122_mv; B's item-level translation tasks hurt
  (−0.0035 HR@10, CI < 0); C (+ the cross-modal coin-flip task) is the best of the three and ties mfd122_mv (+0.0008)
  and the plug-in merged list (−0.0001 HR@10, NDCG −0.0013).
- On MFD the cross-modal sequence task is the only one with a positive sign (C − A: +0.0030 valid, CI > 0; +0.0019 test),
  consistent with it being C's most useful task. It adds ≈ nothing over multi-view augmentation, which already shows
  the decoder each single-modality view of the history (mfd122_mv).
- None of the versions changes f prediction (top-1 0.097–0.100 vs 0.0996); the binding limit of §11 stays.
- C seed-42 merged (0.1301) remains 0.0056 HR@10 above the best MFD.

## 14. Codebook-geometry-aware training (chain first, then MFD; Instruments and Arts)

Written 2026-11-03. Seed 42, locked v1 1+2+2 SID and the tokenizer checkpoint that produced it (hashes verified: index =
v1, `md/sids/v1/<D>/1+2+2/tokenizer.pth` = the attempt's `final.pth`, and re-encoding every item reproduces the index
except the 24 (Instruments) items moved by the distance reassignment). Code: `macrec/geo_embed.py`, `finetune_geo.py`,
`geo_check.py`, `geo_dryrun.py`, `geo_diag.py`; jobs `sbatch/geo/{run_geo_train.sh,geo_gate.sbatch,eval_geo.sbatch}`;
tools `sbatch/tools/{geo_config_diff,geo_chain_paired,geo_data_check}.py`. Runs: `mmgen_runs/geo/<D>/chain122_<VAR>/`,
`mmgen_runs/mfd/<D>/mfd122_mv_geo<VAR>/`.

### 14.1 Implementation as built

- **A. Codebook-initialised code embeddings.** For every code token (levels s0 = f, t0, t1, v0, v1 = the user's f, t1,
  t2, i1, i2) emb = P_level c(code) + delta(code). c(code) = row of `<chain>_rq.vq_layers.<l>.embedding.weight` in the
  tokenizer checkpoint: MACRec's VectorQuantizer is a plain nn.Embedding trained by the VQ loss after k-means init, there
  is no EMA / centroid buffer, and the SIDs are argmins over exactly these rows. P_level = alpha · Q (Q with orthonormal
  columns, so cosines are preserved exactly at init); alpha matches the mean norm of T5's default embeddings (11.29);
  delta = 0 at init. Other tokens keep their rows. **The T5 ties input embeddings and output head** (checked at
  install), so the geometry enters the logits too. After training the model is baked to a plain T5 (logits unchanged,
  max diff 0.0), so evaluation is the control's.
- **C. Shuffled control:** codebook rows permuted among the codes of each level (seed 0); same norms, no geometry
  (cosine correlation with the true geometry 0.00–0.05).
- **B. Soft labels for f:** q = (1 − eps) one-hot + eps softmax_{f'≠f*}(−d²/T), T = median squared distance to the
  10th-nearest f code (Instruments 17.2, Arts 24.2). At this T the soft target is nearly flat: the 5 / 10 nearest f codes
  get only 6.4% / 11.7% of the eps mass (Arts 5.3% / 9.8%); see §14.5.
- Sanity checks (all passed, `mmgen_runs/geo/<D>/geo_check.json`): init cosines = projected codebook cosines (max diff
  < 1e-4) for A and the permuted vectors for C; norms; tied head; gradients reach P and delta; save/reload and bake
  equivalence; soft-label rows sum to 1 with 1 − eps on the true f; soft-label loss reaches the Trainer as
  `outputs["loss"]` (a first AB1/AB2 attempt failed on this and was fixed). Config diff (dry run vs the control's dry run):
  only the model class and the added P / delta parameters differ (`mmgen_runs/geo/gate/<D>/geo_config_diff.json`).

### 14.2 Runs and results

Chain variants were chosen on Instruments VALIDATION HR@10: AB2 (0.1497 vs control 0.1476, +0.0022 > the 0.002 rule,
shuffled control −0.0018), so the Arts rule triggered and AB2 ran on MFD (Instruments, Arts) and the chain (Arts).

| run | eval | valid HR@10 / NDCG@10 | test HR@1 | test HR@10 / NDCG@10 | Δ test HR@10 vs control [95% CI] | epochs / best / steps |
|---|---|---|---|---|---|---|
| Instr. chain control | beam 20 | 0.1476 / 0.1132 | 0.0751 | 0.1181 / 0.0932 | — | 193 / 183 / 16,019 |
| chain + A | beam 20 | 0.1476 / 0.1156 | 0.0767 | 0.1179 / 0.0943 | −0.0002 [−0.0036, +0.0032] | 101 / 91 / 8,383 |
| chain + C (shuffled) | beam 20 | 0.1457 / 0.1137 | 0.0770 | 0.1197 / 0.0951 | +0.0016 [−0.0016, +0.0047] | 129 / 119 / 10,707 |
| chain + AB1 (eps 0.1) | beam 20 | 0.1491 / 0.1154 | 0.0754 | 0.1172 / 0.0932 | −0.0009 [−0.0041, +0.0022] | 102 / 92 / 8,466 |
| chain + AB2 (eps 0.2) | beam 20 | 0.1497 / 0.1158 | 0.0759 | 0.1189 / 0.0941 | +0.0008 [−0.0025, +0.0042] | 105 / 95 / 8,715 |
| Instr. MFD mv | (ii) | 0.1540 / 0.1160 | 0.0756 | 0.1237 / 0.0959 | — | 61 / ≈51 / 15,128 |
| MFD mv + AB2 | (ii) = (iii) | 0.1557 / 0.1181 | 0.0765 | 0.1239 / 0.0962 | +0.0002 [−0.0029, +0.0032] | 47 / 37 / 11,656 |
| MFD mv + C (shuffled) | (ii) | 0.1499 / 0.1156 | 0.0770 | 0.1244 / 0.0970 | +0.0007 [−0.0024, +0.0037] | 52 / 42 / 12,896 |
| Arts chain control | beam 20 | 0.1244 / 0.0867 | 0.0488 | 0.1059 / 0.0739 | — | 186 / 176 / 19,716 |
| **Arts chain + AB2** | beam 20 | **0.1304** / 0.0935 | 0.0577 | **0.1106 / 0.0807** | **+0.0046 [+0.0016, +0.0077]** | 116 / 106 / 12,296 |
| Arts MFD mv | (ii) | 0.1253 / 0.0858 | 0.0474 | 0.1066 / 0.0729 | — | 56 / ≈46 / 17,696 |
| **Arts MFD mv + AB2** | (ii) | **0.1373** / 0.0959 | 0.0568 | **0.1180 / 0.0827** | **+0.0114 [+0.0084, +0.0143]** | 51 / 41 / 16,116 |

Setting (i) (K_f = 20): Instruments MFD + AB2 0.1201 vs 0.1208; Arts 0.1110 vs 0.1023 (+0.0088 [+0.0059, +0.0116]).
Arts MFD + AB2 (iii) chose K_f = 256 (same HR@10 as (ii), 84.6 ms/user). Paired NDCG@10 on Arts: chain +0.0069, MFD
+0.0098 (CIs exclude 0). Decoder passes / ms per user are unchanged by the embeddings (chain 6 / ≈25, MFD 3 / 29–37).

### 14.3 Diagnostics (test)

| run | f top-1 / top-20 | wrong top-1 f: distance to true f / mean pair distance | wrong top-1 within the true f's 10 NN | learned-embedding vs codebook cosine correlation (s0 / t0 / t1 / v0 / v1) |
|---|---|---|---|---|
| Instr. chain control | 0.0992 / 0.3515 | 0.855 | 0.061 | 0.14 / 0.29 / 0.10 / 0.37 / 0.22 |
| chain + A / AB1 / AB2 | 0.100–0.101 / 0.351–0.362 | 0.841–0.848 | 0.064–0.068 | 0.94 / 0.83 / 0.97 / 0.94 / 0.97 |
| chain + C | 0.0995 / 0.3418 | 0.854 | 0.065 | ≈ 0.0 |
| Instr. MFD mv / + AB2 / + C | 0.0996 / 0.1024 / 0.1027 (top-20 0.357 / 0.370 / 0.357) | 0.853 / 0.848 / 0.849 | 0.068 / 0.065 / 0.062 | mv 0.07–0.25; AB2 0.76–0.93; C ≈ 0 |
| Arts chain control / + AB2 | 0.0975 / **0.1094** (top-20 0.398 / 0.420) | 0.861 / 0.845 | 0.120 / 0.130 | 0.17–0.41 / 0.84–0.96 |
| Arts MFD mv / + AB2 | 0.0981 / **0.1096** (top-20 0.410 / 0.425) | 0.860 / 0.847 | 0.121 / 0.127 | 0.08–0.31 / 0.77–0.94 |

HR@10 by the rank of the true f (MFD at K_f = 100; the chain's "0 beyond f rank 20" is a consequence of beam 20, not
evidence; `geo_diag.json` of each run): ranks 21–50 give 0.029–0.039, 51–100 ≤ 0.007, for every Instruments MFD model.

### 14.4 Reading

- The model keeps the codebook geometry (correlation 0.77–0.97 after training vs 0.07–0.41 for the controls), but
  wrong top-1 f guesses do not land closer to the true f on either dataset (within-10-NN 0.065 vs 0.061–0.068).
- **Instruments: no effect** (all test differences within ±0.002, shuffled control ties AB2 on test).
- **Arts: a large gain** on both decoders (+0.005 chain, +0.011 MFD test HR@10, CIs > 0), with f top-1 up 0.098 → 0.110
  and training stopping ~70 epochs (chain) / ~5 epochs (MFD) earlier. Without an Arts shuffled control the cause is
  open: geometry, the near-uniform soft labels acting as label smoothing, or the re-initialisation. §15 disentangles it.
- Earlier statements that more geometry work is not warranted (`docs/EXPLORATION.md`) were written before the Arts
  results and are corrected there.

### 14.5 Data check and soft-label temperature (no training; `sbatch/tools/geo_data_check.py`)

Next-f signal from the history's f codes, test top-1 / top-20 (users whose target f is not in the history in brackets):
geometry (mean distance) Instruments 0.034 / 0.248 (0.004 / 0.169), Arts 0.072 / 0.324 (0.004 / 0.202); co-occurrence of
f transitions in training sequences 0.050 / 0.328 (0.017 / 0.260), Arts 0.089 / 0.393 (0.010 / 0.272); popularity
0.032 / 0.230, Arts 0.030 / 0.188; trained models 0.100 / 0.357 and 0.098 / 0.410. Behavioural co-occurrence is the
stronger signal; content geometry is weak (below popularity on Instruments' non-repeat users).

Temperatures that put a given share of the eps mass on the 10 nearest f codes (eps mass on 5-NN / 10-NN):

| | Instruments T | 5-NN / 10-NN | Arts T | 5-NN / 10-NN |
|---|---|---|---|---|
| used (median 10th-NN) | 17.22 | 0.064 / 0.117 | 24.23 | 0.053 / 0.098 |
| 50% | 4.60 | 0.345 / 0.500 | 5.46 | 0.347 / 0.500 |
| 70% | 3.14 | 0.534 / 0.700 | 3.80 | 0.536 / 0.700 |
| 90% | 1.92 | 0.779 / 0.900 | 2.37 | 0.780 / 0.900 |
