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
