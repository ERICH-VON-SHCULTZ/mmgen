# MFD stage 1 on Instruments: report

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
- K_f = 20 was fixed by the GOAL. Coverage rises with K_f at no ranking-model cost; a K_f sweep
  must be chosen on **validation**.
- Arts and Games are not run yet.
