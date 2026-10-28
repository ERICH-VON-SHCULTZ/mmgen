# MD-RQ-VAE attempt log (Amazon Instruments)

Every tokenizer attempt on Instruments, in order, including failed ones. One change per attempt
relative to the attempt named as its parent. Tokenizers train on the item catalog only (no
interactions), so no validation or test interaction data is involved in these choices.

Code: `macrec/cross_index/models/mdrqvae.py`, `main_mdrqvae.py`, `generate_indices_md.py`,
`md_diagnostics.py`; job `sbatch/md/tok_attempt.sbatch`. Run dirs:
`/scratch/yw8866/mmgen_runs/md/Instruments/tok/<name>/` (`run_config.json`, `train_log.jsonl`,
`best.pth`, `sid_stats.json`, `diagnostics.json`).

Diagnostics (Part-2 bar):
1. reconstruction cos per modality vs A (text TIGER) and B (image TIGER) on the same items;
2. codebook utilisation per level (curve logged every 10 epochs);
3. full-SID collision rate before any disambiguation code;
4. CMI(t_1; i_1 | f) plug-in vs a permutation null (image codes shuffled within each shared code,
   100 reps); "corrected" = plug-in − null mean, given as a fraction of H(t_1 | f);
5. AMI(f, A level 0) and AMI(f, B level 0);
6. separation (Tmall TEST 1b adapted): normalized ratio (text/base) ÷ (image/base) for pairs that
   differ only in that code; > 1 = the code moves text more;
7. additivity: share of items where D(q_f + q_m) beats both D(q_f) and D(q_m).

References on the same items: A text recon cos 0.9527, B image recon cos 0.9023; A level-0
utilisation 84, B 104; AMI(A0, B0) = 0.326. Early-fusion chain E-norm separation ratios by
level: 0.936 / 0.678 / 0.539.

---

## a01: Tmall design at Amazon scale (parent: none)

Settings: input L2-normalised × √dim per modality (unit per-dimension RMS; also the
reconstruction target); encoders/decoders = the A/B RQ-VAE MLP ([2048,1024,512,256,128,64] → 32);
fusion = MLP on [z_t ; z_i] (64→64→32); shared RQ 1 × 256; branches = single Linear on [r_f ; z_m]
(r_f against the hard shared code); modality RQ L × 256; codebooks = MACRec VectorQuantizer
(gradient-trained, k-means init on the first batch, L2 lookup, β = 0.25); loss = MSE_t + MSE_i +
VQ + 1.0 × (InfoNCE(z_f, z_t) + InfoNCE(z_f, z_i)), one-directional, learnable temperature
(Tmall); AdamW lr 1e-3, wd 1e-4, batch 2048, 500 epochs; checkpoint = lowest full-SID collision
(as A/B). Seed 42. Training ≈ 1 min.

| config | ckpt epoch | rec cos text / image (A / B) | util per level | full-SID coll. | CMI₁ plug-in / null (±sd) | corrected, % H(t₁\|f) | AMI f~A0 / f~B0 | sep. ratio shared / text / image | additivity text / image |
|---|---|---|---|---|---|---|---|---|---|
| 1+2+2 | 420 | 0.9475 / 0.8790 (0.9527 / 0.9023) | s 113; t 112, 253; v 108, 240 | 0.40% | 3.215 / 3.007 (0.005) | 0.208 bits, **4.5%** | 0.371 / 0.318 | 0.728 / **1.092** / **0.406** | 0.929 / 0.908 |
| 1+3+3 | 310 | 0.9463 / 0.8718 | s 110; t 168, 252, 254; v 112, 248, 250 | 0.27% | 3.380 / 3.194 (0.006) | 0.186 bits, **3.8%** | 0.342 / 0.305 | 0.667 / **1.097** / **0.443** | 0.913 / 0.877 |

Notes:
- Utilisation dips early (epoch 10: 19–60 codes per level) and recovers by epoch 30–40; final
  levels use 108–254 codes. No level collapsed.
- The full-chain CMI is pure estimator bias at this item count (1+2+2: plug-in 5.843 bits vs
  null 5.835, both ≈ H(t|f) = 5.97), so only the first-level test is informative.
- The first-level CMI is far above its null in z-score (p = 0 over 100 shuffles), but the excess
  is small: 0.19–0.21 bits, 3.8–4.5% of H(t₁|f) (Tmall: 1.7%). Separation is strong: the text code
  moves text more than image (1.09), the image code is very image-selective (0.41), more than any
  level of the early-fusion chain E (≥ 0.54).
- Verdict: passes utilisation, collisions, separation, additivity; CMI excess above null is
  the open point. Next: try to reduce it with one change at a time.

---

## All attempts (one row per attempt; columns as in a01)

| attempt | parent → change | ckpt ep. | rec cos text / image | util per level (s; t…; v…) | full-SID coll. | CMI₁ plug-in / null (sd) | corrected, % H(t₁\|f) | AMI f~A0 / f~B0 | sep. ratio shared / text / image | additivity t / i | balance cos(D_t(q_f)) − cos(D_i(q_f)) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a01 1+2+2 | — (Tmall design, MLP fusion) | 420 | 0.9475 / 0.8790 | 113 112 253 108 240 | 0.40% | 3.215 / 3.007 (0.005) | 0.208, 4.52% | 0.371 / 0.318 | 0.728 / 1.092 / 0.406 | 0.929 / 0.908 | +0.094 |
| a01 1+3+3 | — | 310 | 0.9463 / 0.8718 | 110 168 252 254 112 248 250 | 0.27% | 3.380 / 3.194 (0.006) | 0.186, 3.79% | 0.342 / 0.305 | 0.667 / 1.097 / 0.443 | 0.913 / 0.877 | +0.105 |
| a02 1+2+2 | a01 → dead-code reset every 10 epochs until epoch 400 | 500 | 0.9532 / 0.8978 | 256 254 256 255 256 | 0.32% | 3.548 / 3.482 (0.003) | 0.066, 1.56% | 0.360 / 0.331 | 0.697 / 1.073 / 0.367 | 0.954 / 0.954 | +0.092 |
| a03 1+2+2 | a01 → InfoNCE weight 1.0 → 3.0 | 420 | 0.9398 / 0.8659 | 105 161 253 151 248 | 0.32% | 3.718 / 3.558 (0.004) | 0.160, 3.20% | 0.342 / 0.297 | 0.737 / 0.948 / 0.471 | 0.888 / 0.871 | +0.113 |
| a04 1+3+3 | a02 settings at 1+3+3 | 490 | 0.9577 / 0.9010 | 256 256 256 255 256 256 256 | 0.26% | 3.500 / 3.425 (0.003) | 0.076, 1.83% | 0.339 / 0.309 | 0.636 / 1.127 / 0.378 | 0.975 / 0.963 | +0.103 |
| **a05 1+2+2** | a02 → cross-attention fusion (Tmall CrossAttentionFusion, 2 tokens, 4 heads) | 440 | **0.9565 / 0.9081** | 256 255 256 256 256 | 0.30% | 4.214 / 4.162 (0.001) | **0.052, 1.14%** | 0.320 / 0.292 | 0.769 / **1.271** / 0.360 | 0.975 / 0.990 | +0.078 |
| **a06 1+3+3** | a05 settings at 1+3+3 | 500 | **0.9659 / 0.9265** | 256 256 256 254 255 256 255 | 0.26% | 4.261 / 4.219 (0.001) | **0.043, 0.93%** | 0.299 / 0.269 | 0.765 / **1.330** / 0.370 | 0.995 / 0.996 | +0.080 |

References (same items): A text rec cos 0.9527, B image 0.9023.

### a02: dead-code reset (parent a01)
Why: a01 used only 108–119 of 256 codes on the shared and first modality levels; an under-used
shared code leaves common information to both chains, which couples them. Change: every 10
epochs, codes unused by the full catalog are re-seeded from random encoder outputs of the same
quantizer; stopped after epoch 400 so codes settle. Result: every level 254–256 codes; recon
matches A/B; corrected CMI 4.5% → 1.6%. **Kept.**

### a03: InfoNCE weight 3.0 (parent a01)
Why: a stronger pull of z_f toward both modality latents could make f carry more shared
information. Result: worse on every diagnostic (recon, CMI 3.2%, text separation 0.95 < 1).
**Rejected.**

### a04: a02 at 1+3+3
The second configuration with a02's settings: passes (CMI 1.8%, all levels used).

### a05: cross-attention fusion (parent a02)
Why: Tmall's fusion; with one vector per modality it mixes two tokens, but with residual
connections and LayerNorm the shared latent keeps both modality latents visible. Result: better
than a02 on reconstruction (text 0.9565, image 0.9081, both above A/B), CMI (1.14%), text
separation (1.27) and additivity (0.975 / 0.990). **Kept: final 1+2+2.**

### a06: a05 at 1+3+3
Final 1+3+3: reconstruction 0.9659 / 0.9265, CMI 0.93%, text separation 1.33, additivity ≈ 0.995.
**Final 1+3+3.**

### Stop
The bar is met by a05 and a06 after 6 attempts: no level collapsed (all ≥ 254 codes at the end;
the early dip at epoch 1–10 recovers by epoch 20–30), full-SID collision 0.26–0.30%, and the CMI
excess over the permutation null is 0.04–0.05 bits (0.9–1.1% of H(t₁|f)), the same order as
Tmall's bias-corrected 1.7%. The excess is still many null standard deviations above the null
(p = 0 / 100 shuffles): the dependence is real but small. The full-chain CMI is not informative at
6,250 items (plug-in ≈ null ≈ H).

Collision handling (superseded 2026-10-27): the 0.26–0.30% of items with a duplicate full SID
(groups of at most 2) first got a trailing disambiguation token `<x_k>`, as A/B/E do
(`Instruments.index_md122_a05.json`, 6 tokens; `index_md133_a06.json`, 8). **Official from
2026-10-27:** distance-based reassignment with no extra token (`reassign_md.py`, rule in
`docs/MDRQVAE.md` §0): `Instruments.index_md122_a05_reassign.json` (5 tokens, 19 items changed,
all on the last text/image level) and `index_md133_a06_reassign.json` (7 tokens, 16 items changed).
Diagnostics are unchanged except collision → 0 (first levels untouched).

---

## Follow-ups on Arts/Games (2026-10-27; observations, no tokenizer change)

**Convergence check (all six; checkpoint rule = lowest full-SID collision, later epoch wins ties,
500 epochs).** Best epochs: Instruments 440 / 500, Arts 500 / 400, Games 200 / 450 (1+2+2 / 1+3+3).
At the epoch limit: Instruments 1+3+3 (collision 0.256% at 290 and again at 500, a tie) and Arts
1+2+2 (0.573% at 450 and 500, a tie). Games 1+2+2 is **not** at the limit: it trained 500 epochs,
its collision minimum is at epoch 200 (0.87%) and later epochs are 0.87–1.14%. Its reconstruction
keeps improving to epoch 500 (cos text / image 0.9366 / 0.8841 at 200 → 0.9560 / 0.9176 at 500,
above A / B 0.9462 / 0.9049), as it does in every run: the Games 1+2+2 reconstruction gap
comes from the collision-based checkpoint choice, not from too few epochs. Pending a user decision.

**CMI without shared images** (`sbatch/tools/cmi_shared_image.py` →
`mmgen_runs/md/cmi_shared_image.json`). First-level corrected CMI (plug-in − permutation-null mean,
% of H(t₁|f)) on the official reassigned sets, all items vs after excluding every item whose image
file is byte-identical (SHA-256) to another item's, the Games banner included:

| dataset | items excluded (groups, largest) | 1+2+2 all → excl. | 1+3+3 all → excl. |
|---|---|---|---|
| Instruments | 270 (106, 8) | 0.052 bits 1.14% → 0.033 bits 0.73% | 0.043 bits 0.93% → 0.023 bits 0.51% |
| Arts | 634 (208, 63) | 0.071 bits 1.42% → 0.036 bits 0.72% | 0.070 bits 1.40% → 0.030 bits 0.61% |
| Games | 4,213 (738, 2,287) | 0.099 bits 1.86% → 0.040 bits 0.79% | 0.158 bits 3.09% → 0.079 bits 1.64% |

Items sharing an image account for about half of the excess in every set, not only in Games.
Without them Games 1+3+3 (1.64%) is still the highest, about twice the other sets (0.51–0.79%).
The byte-identical count for Games (4,213) is larger than the 3,475 noted earlier; that figure
counted a different notion of "identical image file", so the two should not be mixed.

**Shared-code balance** (observation): AMI(f, A0) / AMI(f, B0); balance = cos(D_t(q_f), x_t) −
cos(D_i(q_f), x_i); separation ratio of the shared code (>1 moves text more, relative to baseline).

| dataset | 1+2+2 AMI A0 / B0 | 1+3+3 AMI A0 / B0 | balance 1+2+2 / 1+3+3 | sep. shared 1+2+2 / 1+3+3 | AMI(A0, B0) |
|---|---|---|---|---|---|
| Instruments | 0.320 / 0.292 (text) | 0.299 / 0.269 (text) | +0.078 / +0.080 | 0.769 / 0.765 | 0.326 |
| Arts | 0.354 / 0.304 (text) | 0.307 / 0.256 (text) | +0.115 / +0.121 | 0.784 / 0.807 | 0.270 |
| Games | 0.322 / 0.341 (image) | 0.315 / 0.356 (image) | +0.055 / +0.061 | 1.012 / 0.864 | 0.283 |

By AMI the shared code leans to text on Instruments and Arts and to image on Games. By
reconstruction, q_f alone reconstructs text better than image everywhere, with the smallest margin
on Games. The measures agree that Games is the most image-leaning dataset.

**Final-epoch checkpoint (decision 2026-10-27).** All six tokenizers re-indexed from `final.pth`
(epoch 500), reassigned (tags `md1X2_a0X_ep500_reassign`), diagnostics rerun. Before → after
(best-collision → final epoch); excl. CMI = corrected CMI without items with byte-identical images.

| set | ckpt epoch | rec cos text / image | corrected CMI all | excl. CMI | AMI f~A0 / f~B0 | sep. shared / text / image | additivity t / i | balance | items changed |
|---|---|---|---|---|---|---|---|---|---|
| Instruments 1+2+2 | 440 → 500 | 0.9565 / 0.9081 → 0.9600 / 0.9150 | 1.14% → 1.05% | 0.73% → 0.68% | 0.320 / 0.292 → 0.321 / 0.296 | 0.769 / 1.271 / 0.360 → 0.749 / 1.279 / 0.354 | 0.975 / 0.990 → 0.980 / 0.990 | +0.078 → +0.082 | 19 → 24 |
| Instruments 1+3+3 | 500 → 500 | unchanged 0.9659 / 0.9265 | 0.93% | 0.51% | 0.299 / 0.269 | 0.765 / 1.330 / 0.370 | 0.995 / 0.996 | +0.080 | 16 |
| Arts 1+2+2 | 500 → 500 | unchanged 0.9677 / 0.9136 | 1.42% | 0.72% | 0.354 / 0.304 | 0.784 / 1.599 / 0.266 | 0.991 / 0.995 | +0.115 | 54 |
| Arts 1+3+3 | 400 → 500 | 0.9644 / 0.9074 → 0.9676 / 0.9160 | 1.40% → 1.27% | 0.61% → 0.55% | 0.307 / 0.256 → 0.319 / 0.268 | 0.807 / 1.419 / 0.321 → 0.825 / 1.362 / 0.307 | 0.994 / 0.996 → 0.994 / 0.997 | +0.121 → +0.122 | 41 → 44 |
| Games 1+2+2 | 200 → 500 | 0.9366 / 0.8841 → 0.9560 / 0.9176 | **1.86% → 3.10%** | **0.79% → 1.82%** | 0.322 / 0.341 → 0.373 / 0.373 | 1.012 / 2.209 / 0.442 → 1.046 / 2.011 / 0.367 | 0.970 / 0.970 → 0.984 / 0.987 | +0.055 → +0.078 | 120 → 141 |
| Games 1+3+3 | 450 → 500 | 0.9584 / 0.9182 → 0.9601 / 0.9223 | **3.09% → 3.61%** | **1.64% → 2.02%** | 0.315 / 0.356 → 0.316 / 0.358 | 0.864 / 2.602 / 0.439 → 0.839 / 2.436 / 0.425 | 0.982 / 0.976 → 0.985 / 0.975 | +0.061 → +0.072 | 69 → 80 |

Utilisation stays 253–256 per level everywhere; full-SID collision before reassignment rises
slightly with the final epoch on Instruments 1+2+2 (0.30 → 0.38%), Arts 1+3+3 (0.44 → 0.47%), Games
(0.87 → 1.02%, 0.50 → 0.58%), and is 0 after reassignment everywhere. All reassignment changes stay
on the last level. Instruments and Arts improve or stay equal on every diagnostic. Games 1+2+2
gains a lot in reconstruction (now above A / B) but its CMI more than doubles (excl. 0.79 → 1.82%,
under the bar); Games 1+3+3 rises to 2.02% excl., marginally above the ~2% bar. It was first held
back; the user then accepted it (2.02% is within "about 2%") so that all six sets follow one rule.
**All six v1 sets are now final-epoch sets.**

Observation: on Games the corrected CMI rises with training (excl. shared images: 1+2+2 0.79% →
1.82%, 1+3+3 1.64% → 2.02%; all items 1.86% → 3.10%, 3.09% → 3.61%), while on Instruments and Arts
it does not (it falls or stays equal: Instruments 0.73 → 0.68%, 0.51% = 0.51%; Arts 0.72% = 0.72%,
0.61 → 0.55%). In both Games sets the text-chain separation drops
(2.21 → 2.01, 2.60 → 2.44): the later epochs trade some modality specificity for reconstruction.

**E-norm under the final-epoch rule (Instruments; job 21165054).** The E-norm (fnormj) tokenizer
was retrained with the unchanged MQL4GRec command through `main_single_final.py` (seed 2023 as in
main_single.py). The rerun's best-collision index is byte-identical to the original
`Instruments.index_fnormj_tiger.json` (training is reproducible), and the final-epoch index
`Instruments.index_fnormj_ep500_tiger.json` is also byte-identical to it: E-norm's lowest collision
(5.328%) is reached at the last evaluation (epoch 499), so for E on Instruments both rules give the
same SIDs. The T5 run `tiger_fnormj_ep500` (21165055) therefore uses the same index and seed as
`tiger_fnormj` (21103991) and doubles as a T5 reproducibility check.

---

## Chain decoder on the official SID (Instruments, seed 42; train 21164568, eval 21178172 on H200)

TIGER settings (plain T5, BS 256, LR 5e-4, seqrec, 200 epochs, patience 10), single list.

| run | SID length | best epoch / run | valid HR@10 / NDCG@10 | test HR@1 / HR@5 / HR@10 / NDCG@5 / NDCG@10 | decoder passes/user | ms/user (H200, alone) |
|---|---|---|---|---|---|---|
| MD 1+2+2 (a05, final epoch, reassigned) | 5 | 183 / 193 | 0.1476 / 0.1132 | 0.0751 / 0.0984 / 0.1181 / 0.0869 / 0.0932 | 6 | 25.3 |
| MD 1+3+3 (a06, final epoch, reassigned) | 7 | 197 / 200 | 0.1435 / 0.1105 | 0.0732 / 0.0975 / 0.1161 / 0.0856 / 0.0916 | 8 | 32.0 |
| A TIGER-text, seed 42 (3-seed mean) | 4 | 107 | — | 0.0655 / 0.0929 / 0.1131 (0.1154) / 0.0794 / 0.0859 (0.0876) | 5 | 21.8 |
| B TIGER-image, seed 42 (3-seed mean) | 4 | 113 | — | 0.0689 / 0.0915 / 0.1120 (0.1139) / 0.0801 / 0.0867 (0.0883) | 5 | 22.3 |
| E-raw, seed 42 | 4 | — | — | 0.0676 / 0.0928 / 0.1132 / 0.0805 / 0.0870 | 5 | 20.7 |
| **E-norm (fnormj, final-epoch rule = same index), seed 42** | 4 | 103 / 113 | 0.1352 / 0.1035 | 0.0651 / 0.0877 / 0.1055 / 0.0765 / 0.0822 | 5 | 21.5 |
| control: MD 1+2+2, best-collision ckpt (ep 440), reassigned | 5 | 189 / 199 | 0.1457 / 0.1120 | 0.0708 / 0.0951 / 0.1148 / 0.0832 / 0.0895 | 6 | 28.0 |
| control: MD 1+3+3, best-collision ckpt (= ep 500, same index) | 7 | 197 / 200 | 0.1435 / 0.1105 | identical to the official 1+3+3 run | 8 | 32.4 |
| C MQL4GRec, one list (text / image), 3-seed mean | 4 | — | — | HR@10 0.1275 / 0.1287, NDCG@10 0.0984 / 0.0988 | 5 | 21.6 |
| D MACRec, one list (text / image), 3-seed mean | 4 | — | — | HR@10 0.1302 / 0.1285, NDCG@10 0.0997 / 0.0992 | 5 | 21.6 |
| C / D merged (two beam searches + union), seed 42 | 4 per list | — | — | HR@10 0.1301 / 0.1321, NDCG@10 0.0998 / 0.1021 | 5 × 2 | 2 × 21.6 |

Fair comparison: MD, A, B and E all train T5 on seqrec only and decode one list; C/D train on extra
cross-modal tasks, and their published protocol decodes two lists (text, image) and merges them.
C/D's lead does not come from the merge (+0.001–0.004 HR@10 over their best single list): their
single lists, at 5 passes, are already above MD. C/D are full-system references, not controls.

1+2+2 is the better config (all five test metrics, and cheaper). The 1+3+3 run's best checkpoint is
at epoch 197 of the 200-epoch cap, so it may be under-trained under the shared TIGER budget.

- **MD vs E (key comparison):** MD 1+2+2 beats E-norm on every test metric: HR@10 0.1181 vs 0.1055
  (+12%), NDCG@10 0.0932 vs 0.0822 (+13%), HR@1 +15%. E-norm is also below E-raw (0.1132) and A/B.
- **Checkpoint effect (1+2+2):** final epoch 0.1181 / 0.0932 vs best-collision 0.1148 / 0.0895 test
  (valid 0.1476 vs 0.1457 HR@10); +0.0033 HR@10, ~1.5× A's seed std (0.0022). Same SID length, same
  decoder passes. 1+3+3 has no checkpoint contrast (both rules pick epoch 500); its two T5 runs are
  bit-identical, so T5 training is deterministic at seed 42.
- **E-raw extension:** E-raw − E-norm on Instruments = +0.0077 HR@10, more than the seed spread, so
  E-raw is extended to Arts and Games (tokenizer 21251957, T5 21251971).
- **Token effect** (old index with a trailing `<x_k>` token, best-collision checkpoint; train 21168751,
  eval 21220669 on H200). Same checkpoint, token vs reassignment:

  | config | index | SID len | passes/user | ms/user | valid HR@10 / NDCG@10 | test HR@1 / HR@10 / NDCG@10 |
  |---|---|---|---|---|---|---|
  | 1+2+2 | old, with token (ep 440) | 6 | 7 | 30.0 | 0.1456 / 0.1111 | 0.0720 / 0.1186 / 0.0919 |
  | 1+2+2 | reassigned (ep 440) | 5 | 6 | 28.0 | 0.1457 / 0.1120 | 0.0708 / 0.1148 / 0.0895 |
  | 1+2+2 | **official**: reassigned, final epoch | 5 | 6 | 25.3 | 0.1476 / 0.1132 | 0.0751 / 0.1181 / 0.0932 |
  | 1+3+3 | old, with token (ep 500) | 8 | 9 | 37.4 | 0.1454 / 0.1118 | 0.0747 / 0.1165 / 0.0922 |
  | 1+3+3 | **official**: reassigned (ep 500) | 7 | 8 | 32.0 | 0.1435 / 0.1105 | 0.0732 / 0.1161 / 0.0916 |

  As expected: one decoder pass fewer and 7–14% less time per user, metrics within seed noise
  (|Δ test HR@10| ≤ 0.0038 ≈ 1.7× A's seed std; 1+2+2 valid HR@10 is identical, 0.1456 vs 0.1457).
  The official 1+2+2 set is at least as good as the old token index on every metric except test
  HR@10 (−0.0005).

E-norm on Arts / Games (seed 42, test HR@10 / NDCG@10): 0.0825 / 0.0520 and 0.0843 / 0.0454
(A 3-seed 0.0986 / 0.0661 and 0.0885 / 0.0471; B 0.0957 / 0.0657 and 0.0782 / 0.0415).
