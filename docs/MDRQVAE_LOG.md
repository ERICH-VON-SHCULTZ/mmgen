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
