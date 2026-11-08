# Exploration: making better use of the MD-RQ-VAE SID

Written 2026-11-03. Analysis and inference-only checks; **no training runs**. Locked v1 1+2+2 SIDs (hash checked in every
job), seed-42 models, Instruments and Arts. Nothing below was chosen on test: Part 1 is descriptive (test users, as asked),
Part 3 uses validation only.

Code: `macrec/expl_first_step.py` (per-user first-step ranks), `macrec/expl_infer.py` (Part 3 re-scoring),
`macrec/expl_group.py` (f regions), `sbatch/tools/expl_part1.py` (error analysis), `sbatch/expl/expl.sbatch`.
Outputs: `mmgen_runs/expl/<D>/` (`part1.json`, `first_*.pt`, `infer_valid.json`, `group.json`).
Jobs: 21699582–21699586, 21712892–21712893 (GPU), CPU analyses via srun.

**Short answer.**
- C beats us on users where **its first decoding step is right and ours is not**. C's first code is coarse (55–58 text
  codes) and content-coherent: an item's nearest text neighbour shares it 77% of the time. Our first code f is a fine,
  near-uniform 256-way split (7.7 of 8 bits) that separates near-duplicates: the nearest text neighbour shares f only
  32% (Instruments) / 38% (Arts). On the users only C hits, C's first code is top-1 for 46% (Arts 63%); our f is top-1
  for 6% (Arts 11%).
- Popularity, history length and "target f in history" do not separate C's wins from ours.
- No inference-only re-scoring moves validation HR@10 by more than +0.0015 (Part 3). Two checks show where the limit is
  NOT: coverage (keeping f by item mass gives the same HR@10 at K_f = 20 as at 100) and coarse regions of the codebook
  (k-means groups of the f codebook are barely more content-coherent than random groups).
- Ranked proposals (Part 4): (1) a content-coherent coarse-to-fine factorisation of the first decision, (2) more
  training weight on f, (3) item-level structure in the encoder, (4) cheaper f selection by item mass (cost, not
  accuracy), (5) the small inference tweaks.

**Outcome of the proposals (update 2026-11-07, `docs/MFD_REPORT.md` §16–§18; all on top of codebook init, seed 42).**
- (2) More training weight on f: **tested, hurts.** f loss × 2 / × 3 on V2-MFD (Arts, §17 P1b/P1c): valid HR@10 −0.0032 /
  −0.0079, and f top-1 does not rise (valid 0.128 → 0.125 / 0.121; test 0.110 → 0.112 / 0.106).
- (3) Item-level encoder structure: chain decoder ties (§16); **on MFD it is the one change that worked** (§17 P1a, one
  summed vector per history item): Arts +0.0035 valid / +0.0044 test HR@10, Games +0.0039 / +0.0071, Instruments −0.0013
  valid / +0.0030 test; about 2× faster. §18 Task 2 rules out the longer schedule as the cause.
- (1) Coarse-to-fine f factorisation and (4) f selection by item mass: not tested yet.
- Part 1's finding holds after these changes: for users only C hits, the target's f is outside our top-10 f codes for
  72% (Arts) / 82% (Instruments) of them while being scored, and our top 10 is not more concentrated under few f than
  C's (§18 Task 1a), so the remaining gap is the ordering across f codes.

---

## Part 1. Where does C win?

Per TEST user: hit@10 of **C seed 42 merged** vs three of ours, rebuilt from the saved lists exactly as reported
(asserted against the results files): **mfd122_mv at setting (ii)** (K_f = 100, w = 1/0.5/0.5), **plug-in 1+2+2
merged**, **MD chain decoder** (20 beams). First-step ranks from one forward pass of each model on the evaluation inputs,
restricted to the first tokens that exist in the catalog (as the trie).

Test HR@10: Instruments C 0.1301, mfd122_mv 0.1237, plug-in 0.1246, chain 0.1181; Arts C 0.1267, mfd122_mv 0.1066,
plug-in 0.1172, chain 0.1059.

### 1.1 Groups (C merged vs mfd122_mv (ii); the other two comparisons look the same)

| | both | only C | only ours | neither |
|---|---|---|---|---|
| **Instruments** users | 1,774 | 453 | 343 | 14,542 |
| history length (mean / median) | 5.8 / 5 | 7.0 / 6 | 6.4 / 5 | 7.1 / 5 |
| target training interactions (median; catalog median 7) | 762 | 124 | 151 | 18 |
| target f in the history | 0.41 | 0.19 | 0.17 | 0.09 |
| **our** f rank (median) / top-1 / top-20 | 0 / 0.71 / 0.97 | **18 / 0.06 / 0.52** | 6 / 0.14 / 0.85 | 50 / 0.03 / 0.27 |
| our coverage (f in the kept 100) | 1.00 | **0.95** | 1.00 | 0.73 |
| **C's** first-code rank (median) / top-1 (min over its text / image lists) | 0 / 0.78 | **1 / 0.46** | 3 / 0.24 | 6 / 0.15 |
| text NN cosine distance of the target (median; catalog 0.050) | 0.012 | **0.035** | 0.049 | 0.047 |
| image NN cosine distance (median; catalog 0.126) | 0.159 | 0.117 | 0.121 | 0.125 |
| **Arts** users | 1,929 | 880 | 434 | 18,928 |
| target training interactions (median) | 140 | 61 | 76 | 14 |
| target f in the history | 0.63 | 0.37 | 0.28 | 0.13 |
| our f rank (median) / top-1 / top-20 | 0 / 0.61 / 0.98 | **11 / 0.11 / 0.62** | 3 / 0.26 / 0.91 | 42 / 0.04 / 0.33 |
| our coverage | 1.00 | 0.95 | 1.00 | 0.74 |
| C's first-code rank (median) / top-1 | 0 / 0.84 | **0 / 0.63** | 1 / 0.37 | 4 / 0.21 |
| text NN cosine distance (median; catalog 0.040) | 0.021 | **0.025** | 0.041 | 0.043 |

(The target item itself is never in the history. Full per-group statistics, popularity buckets and the plug-in / chain
comparisons: `part1.json`.)

### 1.2 Findings

1. **Each model wins where its own first decision is right.** On "only C" users our f is top-1 for 6% (Arts 11%) and
   C's first code for 46% (63%); on "only ours" users it flips (our f top-1 14–18% / 26–31%, C's first code median rank
   1–3). Hits are almost impossible without a correct first step: on "both", f top-1 is 0.61–0.75.
2. **Coverage is not the problem; ranking across f is.** For mfd122_mv the target f is among the 100 kept codes for 95% of
   the "only C" users, yet the item misses the top 10: its f ranks ~11–18, and items under higher-ranked f codes outscore
   it. (The plug-in and the chain lose these users earlier: coverage 0.59 / 0.34 on Instruments.)
3. **Popularity, history length and repeats do not separate C's wins from ours.** Both are head-item recommenders: "both"
   targets have a median of 762 (Arts 140) training interactions vs a catalog median of 7; "only C" and "only ours" have
   similar popularity (124 vs 151; 61 vs 76) and history length. The target's f is in the history slightly more often
   for C's wins on Arts (0.37 vs 0.28), not on Instruments (0.19 vs 0.17).
4. **C wins on items with close content neighbours.** "Only C" targets sit closer to their nearest text neighbour than
   "only ours" targets (median cosine distance 0.035 vs 0.049; Arts 0.025 vs 0.041); image distances barely differ.
5. **Why: our first code splits near-duplicates; C's does not.** Share of items whose nearest neighbour has the same
   first code (random-pair baseline in brackets):

   | | MD f (256 codes) | MD t1 | MD i1 | C text L1 (55–58) | C text L1+L2 | C image L1 (84–97) |
   |---|---|---|---|---|---|---|
   | Instruments, text NN | 0.317 (0.005) | 0.140 | 0.099 | **0.768** (0.026) | 0.305 | 0.412 |
   | Instruments, image NN | 0.241 | 0.065 | 0.189 | 0.357 | 0.112 | **0.734** (0.016) |
   | Arts, text NN | 0.380 (0.005) | 0.141 | 0.128 | **0.761** (0.030) | 0.346 | 0.349 |
   | Arts, image NN | 0.334 | 0.070 | 0.254 | 0.375 | 0.123 | **0.686** (0.017) |

   f is as fine as C's first *two* text levels (0.317 vs 0.305). Entropy: H(f) = 7.73 / 7.84 bits (near-uniform over 256)
   vs H(C text L1) = 5.47 / 5.36. f is not a refinement of C's first level (purity of f w.r.t. C text L1: 0.40 / 0.41;
   NMI 0.44–0.48). So our first decision is a 256-way choice among content-mixed groups; C's is a ~57-way choice among
   content-coherent groups, refined later. Items with a close neighbour (finding 4) are exactly the ones whose f is
   ambiguous.

---

## Part 2. How the pipeline uses the SID today

| place (code) | what the model sees / produces | item boundaries | code level | modality branch | shared vs modality split | additivity (q_f + q_m ≈ modality content) | conditional independence given f |
|---|---|---|---|---|---|---|---|
| SID → tokens (`data.py _remap_items`, `"".join`) | 5 tokens `<s0_k><t0_k><t1_k><v0_k><v1_k>` per item, no separator | implicit (every 5th token is `<s0_…>`) | only via token namespace | only via namespace | only via namespace | **ignored** | n/a |
| encoder input (`SeqRecDataset`, `MFDDataset`; mv training adds text-only / image-only views) | up to 20 items × 5 tokens, oldest first; no prompt | **not marked**; no item index | namespace only | full view; views only in mv training | views drop one branch's codes | **ignored** | n/a |
| positions (T5 relative bias, 32 buckets) | token distance; exact only within ±8 tokens (< 2 items), log-spaced to 128 | **ignored** (token distance, not item distance; recency of item k is coarse) | ignored | ignored | ignored | ignored | n/a |
| embeddings (`shared`, tied to head) | one free row per code token; HF mean-resizing init (all rows equal); §14 tried codebook init | ignored | **no level embedding**; rows of different levels are unrelated | no branch embedding | no relation between `<s0>` and `<t*>` rows | **ignored** (embedding of an item ≠ sum of its code vectors; §14's codebook init survives training, correlation 0.83–0.97, but gives no gain) | n/a |
| decoder input, chain (`T5`, teacher forcing) | `[start, f, t1, t2, v0, v1]` | one item | position = level | fixed order: text before image | f first | ignored | **ignored**: image codes are conditioned on the text codes |
| decoder input, MFD (`mfd_model.MFDT5`) | two rows `[start, f, <T>, t1]`, `[start, f, <I>, i1]` | one item | position = level | **explicit** `<T>` / `<I>` rows | **explicit**: f is the shared root | ignored | **used**: rows cannot attend to each other |
| output head / softmax | full vocabulary (≈ 33.4k tokens) at every step, train and test | n/a | **not restricted** to the level's codes in training (eval restricts via trie / catalog columns but the log-softmax is full-vocabulary) | not restricted | not restricted | ignored | n/a |
| loss (`finetune_ext` / `MFDT5.forward`) | token cross-entropy, uniform weights: chain 6 targets (5 codes + `</s>`), MFD 5 (f once) | n/a | **f has 1/6 (chain) or 1/5 (MFD) of the loss although it decides every hit (Part 1)** | equal | equal | ignored | MFD factorised loss |
| inference, chain (`test_ddp_save`, trie) | beam 20 over full SIDs, score = sum of full-vocabulary log-probs | n/a | trie per level | text then image | f first; beam can drop f early (coverage 0.18–0.34 on missed users) | ignored | ignored |
| inference, MFD (`mfd_eval.score_batch`) | top-K f by log P(f); every catalog item under a kept f scored exactly: w_f s_f + w_t s_t + w_i s_i | n/a | catalog enumeration per level | **explicit** branch sums | **explicit** | ignored | **used** |
| C's merge (`ensemble.py`, C / plug-in only) | two lists; agreement bonus (+1 for items in both) | n/a | n/a | two views | plug-in: both views share f, so lists are correlated | ignored | n/a |

What the code never uses: **additivity** (anywhere after the tokenizer), **explicit item boundaries / item-level recency**
in the encoder, **level identity** beyond the token string, the **content coherence of the first decision** (the SID's
first code is fine-grained by construction), and in the chain the **conditional independence given f**.

---

## Part 3. Inference-only checks (mfd122_mv, VALIDATION)

One exact scoring pass with every catalog f kept (as `mfd_sweep`), so all variants rank the same candidates. Base at
K_f = 100 reproduces setting (ii) exactly (Instruments 0.1540, Arts 0.1253).

| variant (how f is kept / how items are scored) | Instruments K=20 | Instruments K=100 | Arts K=20 | Arts K=100 |
|---|---|---|---|---|
| base: keep by log P(f); score w·(s_f, s_t, s_i) | 0.1514 / 0.1151 (0.395) | **0.1540** / 0.1160 (0.793) | 0.1244 / 0.0854 (0.441) | **0.1253** / 0.0858 (0.802) |
| **sum**: keep by log P(f) + log Σ_items exp(s_t + s_i) | **0.1540** / 0.1160 (0.338) | 0.1540 / 0.1160 (0.744) | **0.1252** / 0.0858 (0.375) | 0.1253 / 0.0858 (0.768) |
| sum_w: same with 0.5 s_t + 0.5 s_i | 0.1525 / 0.1155 (0.393) | 0.1540 / 0.1160 (0.792) | 0.1245 / 0.0855 (0.439) | 0.1253 / 0.0858 (0.802) |
| max: keep by log P(f) + max_items (s_t + s_i) | 0.1537 / 0.1159 (0.309) | 0.1540 / 0.1160 (0.706) | 0.1251 / 0.0858 (0.338) | 0.1253 / 0.0858 (0.737) |
| renorm: branch scores renormalised over the catalog items under f | 0.1508 / 0.1149 (0.395) | 0.1538 / 0.1158 (0.793) | 0.1255 / 0.0861 (0.441) | 0.1264 / 0.0865 (0.802) |
| tta_f: log P(f) averaged over the full / text / image views of the history | 0.1516 / 0.1153 (0.395) | 0.1538 / 0.1160 (0.795) | 0.1244 / 0.0854 (0.442) | 0.1259 / 0.0860 (0.804) |
| tta_all: all three terms averaged over the three views | 0.1521 / 0.1154 (0.395) | 0.1553 / 0.1164 (0.795) | 0.1239 / 0.0845 (0.442) | 0.1254 / 0.0851 (0.804) |

Cells: validation HR@10 / NDCG@10 (coverage).

- **The requested "sum" rescoring does not raise HR@10** (0.1540 → 0.1540; Arts 0.1253 → 0.1253) and lowers coverage
  (0.793 → 0.744). But it reaches the K = 100 result **at K = 20** on both datasets (0.1540 vs base 0.1514 at K = 20).
  The top-10 lists at K = 100 already equal those over all f; f's whose items reach the top 10 are those with large item
  mass. Coverage beyond that is unused: it counts f's whose items never make the list. So the limit at K = 100 is the
  f term's ranking, not the K_f cut-off.
- Other checks: renorm +0.0011 on Arts / −0.0002 on Instruments; tta_all +0.0013 on Instruments / +0.0001 on Arts. Both
  below 0.002 and inconsistent across datasets.
- **f regions (`expl_group.py`).** Does the model's f mass at least land in the right region when its top-1 f is
  wrong? Regions = k-means of the f codebook vectors (16/32/64 groups) vs random partitions of the same sizes:

  | | Instruments valid, k-means / random | Arts valid, k-means / random |
  |---|---|---|
  | group top-1, 16 groups | 0.221 / 0.182 | 0.255 / 0.178 |
  | group top-1, 64 groups | 0.152 / 0.133 | 0.170 / 0.128 |
  | group top-1 when f top-1 is wrong, 64 groups | 0.052 / 0.027 | 0.074 / 0.024 |
  | coverage at K_f = 100, keep f by log P(f) + log P(group) vs base | 0.783 vs 0.793 | 0.793 vs 0.802 |

  The mass is slightly regional (+0.02–0.08 over random), far from C's coarse first level (C text L1 test top-1
  0.169 / 0.216 over 58 / 55 codes vs our 64 k-means groups 0.132 / 0.149), and keeping f by group mass lowers
  coverage. The codebook's own geometry is also a weak content partition: the share of items whose text NN falls in
  the same k-means group of f is 0.41 vs 0.33 for random groups (58 groups, Instruments; Arts 0.46 vs 0.39), against
  0.77 for C's text L1. This agrees with the §14 data check (codebook distance is a weak next-f signal) and the
  geometry-aware training (no gain). A coarse level, if any, must come from item content neighbourhoods, not from the
  f codebook vectors.

---

## Part 4. Proposals (ranked)

All are generic (any multi-code SID, no dataset-specific features), keep the v1 SID and use no CF features or reranking.

### 1. Coarse-to-fine first decision: factor f into a content-coherent (coarse, fine) pair

**What.** Re-encode each f code as two tokens (a, b) through a fixed bijection f ↔ (a, b) (e.g. 32 × 8), where the coarse
token a groups f codes whose items are content neighbours. The grouping maximises the share of item nearest-neighbour
pairs (text and image features, the tokenizer's own inputs) that fall in the same group. The SID's information is
unchanged (the map is invertible, the v1 files stay as they are); only the first decision becomes a 32-way content-coherent
choice followed by an 8-way one, like C's coarse first level.

**Evidence.** Part 1: wins and losses are decided at the first step (only-C users: C's first code top-1 0.46–0.63 vs our f
0.06–0.11). C's first level is content-coherent (NN share 0.77) and coarse (5.5 bits); f is fine (7.7 bits) and splits
near-duplicates (0.32–0.38), and C wins precisely on targets with close neighbours. Part 3: the codebook's geometry does
not give such a grouping (0.41 vs 0.33 random), so the grouping must be built from item neighbourhoods.

**Risk.** An f code holds items from several content regions (f is not a refinement of C's L1, purity 0.40), so no
grouping of whole f codes can reach C's 0.77; the attainable NN share bounds the benefit. A wrong coarse token now also
removes all f's in other groups (MFD can keep several coarse tokens, as it keeps several f).

**Smallest test.** (a) Offline, CPU: build the grouping (greedy / spectral clustering of the 256 × 256 f co-neighbour
count matrix into 32 groups); report the NN share vs random groups. Proceed only if it is clearly above the k-means
0.44 (G = 32) and approaches C's level. (b) One chain-decoder run, Instruments seed 42, SID read through the bijection
(7 tokens instead of 6 decoder steps), same recipe as the chain control; decide on validation HR@10 and first-step accuracy.

### 2. Give the first decision its share of the training signal

**What.** Weight the f token's cross-entropy (×2–3) or train f with the full catalog-restricted softmax (log-softmax
over the 256 f codes rather than the 33k vocabulary).

**Evidence.** Part 2: f is 1/6 (chain) or 1/5 (MFD) of the loss and is trained against the full vocabulary. Part 1: f
decides every hit. MFD Part B: up-weighting f at inference (w_t = w_i = 0.5) was the single largest gain (+0.003 HR@10),
which says the branches are over-trusted relative to f; training emphasis is the training-side analogue.

**Risk.** Shifting loss mass can cost branch accuracy; §13/§14 showed the f top-1 is hard to move (0.097–0.103 for every
recipe tried), so the ceiling may be the information in the history, not the loss weight.

**Smallest test.** mfd122_mv recipe with f-loss weight 2 and 3 (two runs, packed); validation f top-1/top-20 and HR@10
at setting (ii).

### 3. Item-level structure in the encoder

**What.** Add two small learned embeddings to every history token: its code level (5 rows) and its item's recency
(index from the end, 20 rows). Optional stronger variant: one input vector per item, the sum of its code embeddings
(uses additivity; 5× shorter input, exact item positions within T5's buckets).

**Evidence.** Part 2: nothing marks item boundaries or levels; T5's relative buckets are exact only within ±8 tokens
(< 2 items), so beyond the last item recency is coarse. Recency matters for the next f: in the §14 data check, the last
item alone predicts the next f as well as the whole history (Instruments test top-1 0.057 vs 0.050, using transition
counts only as a diagnostic, not as a feature).

**Risk.** The model may already infer boundaries from the `<s0_…>` tokens; the composed-item variant loses within-item
order. Expected gain is small.

**Smallest test.** Chain control + level/recency embeddings (one run); then MFD if it gains on validation.

### 4. Cheaper f selection by item mass (cost, not accuracy)

**What.** Keep f by log P(f) + log(catalog mass of its level-1 branch codes), computed in pass 2 for all f (2 × 256
sequences per user instead of 2 × K), then run the deeper passes only for the top 20.

**Evidence.** Part 3: keeping f by total item mass gives the K = 100 HR@10 at K = 20 on both datasets (0.1540 vs 0.1514;
0.1252 vs 0.1244). At K = 20 MFD decodes in 6.8 ms/user vs 29.4 at K = 100.

**Risk.** The exact "sum" needs every item's full branch score (K = all); a level-1-only approximation may lose the
effect. No accuracy gain by construction.

**Smallest test.** Inference only: implement the level-1 mass selection in `mfd_eval`, validation HR@10 and ms/user at
K = 20 vs base K = 100.

### 5. Small inference tweaks (keep as candidates, do not pursue alone)

Test-time view ensembling of all terms (+0.0013 Instruments, +0.0001 Arts) and catalog-renormalised branch scores
(+0.0011 Arts, −0.0002 Instruments). Both are free at training time and could be included in any future validation
sweep, but neither clears 0.002 on both datasets.

### Not proposed

- More codebook geometry was listed here on the Instruments evidence (§14 runs, data check, Part 3). **Correction
  (2026-11-03): the Arts geometry runs finished after this report.** On Arts, codebook init + soft f labels (AB2) gains
  significantly: chain +0.0046 test HR@10 [+0.0016, +0.0077], MFD at setting (ii) +0.0114 [+0.0084, +0.0143], with f top-1
  0.098 → 0.110. **Update (2026-11-05): `docs/MFD_REPORT.md` §15 attributes it to the true-codebook initialization** (codebook
  init alone equals AB2; shuffling loses 0.004–0.005; Games +0.0053; Instruments none). Proposal 3 was tested in §16: no HR@10
  gain on either dataset.
- Larger K_f or wider beams: coverage is not binding (Part 1 finding 2, Part 3 "sum").
- Changing the tokenizer (a coarser first shared level, i.e. a v2 SID) is the direct version of proposal 1 but is out of
  scope here (v1 locked); proposal 1 tests the same hypothesis on v1.
