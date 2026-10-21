# MFD: Modality-Factorized Decoding (old implementation)

What the GRID code does, cited to `legacy/grid/`. All paths are relative to it, and `mfd.py`
below means `src/models/modules/semantic_id/mfd_generation_model.py`. For the port, see
`PORTING.md`.

## 1. Plain-language summary

A normal (chain) generative recommender emits the next item's semantic ID one token at a time:
`f`, then `t` given `f`, then `i` given `f` and `t`. MFD uses the MD-RQ-VAE structure, where `t`
and `i` are siblings under `f` (see `MDRQVAE.md` §1), and **drops the `t → i` edge**:

```
chain   P(s|x) = P(f|x) · P(t|x,f) · P(i|x,f,t)
MFD     P(s|x) = P(f|x) · P(t|x,f) · P(i|x,f)          (mfd.py:5-8)
```

Once `f` is chosen, one decoder pass yields the full softmax over **all** text codes and over
**all** image codes. Every `(t, i)` combination under that `f` can then be scored with a lookup,
with no extra decoder work. Candidates are built compositionally:
1. keep the top-`K_f` shared codes;
2. for each, read both heads;
3. combine text × image codes;
4. keep only combinations that are real catalog items;
5. rank by the summed log-probabilities.

The reason this is cheap: in a chain beam, widening the text or image beam costs more decoder
states; in MFD, `K_t` and `K_i` are free.

The data license for dropping the edge is that, given `f`, the text and image codes are nearly
independent: corrected CMI(t; i | f) ≈ 1.7% of H(t | f) (`MDRQVAE.md` §6).

## 2. Model and decoder block layout

`MFDMultiBehaviorEncoderDecoder(SemanticIDMultiBehaviorEncoderDecoder)` (`mfd.py:66`) overrides
only the training step and the generators. Everything else is the chain model's:
- **Encoder.** Unchanged (`mfd.py:14-17`): a T5 encoder fed `inputs_embeds` (its own token tables
  are deleted, `AGENTS.md` §6). The input is the flat history `[b, f, t, i, b, f, t, i, …]`
  with stride 4 (`tiger_generation_model.py:1552-1622`), embedded with a (1+H)-periodic offset into
  one concatenated table `[L0 | L1 | L2 | behavior]` (`tiger_generation_model.py:1359-1362`,
  `1492-1497`). Backbone: d_model 128, 6 heads, d_ff 1024, 4 layers (`configs/experiment/tiger_mb_mfd_flat.yaml:5-6`).
- **Heads.** One `Linear(d, k_level, bias=False)` per SID level
  (`tiger_generation_model.py:1393-1396`), plus `behavior_mlp` (`:1399`).
- **Catalog.** The sorted unique packed keys `(f·k1 + t)·k2 + i` of every item's SID
  (`mfd.py:213-221`). Membership is tested with `searchsorted` (`mfd.py:300-308`).

### 2a. Plain MFD block (no query slots)
Decoder input `[BOS, b, f]`, standard causal mask (`mfd.py:19-28`, training at `mfd.py:733-756`):

| position | input | its hidden state predicts |
|---|---|---|
| p0 | BOS | behavior `b` (`behavior_mlp(h_bos)`) |
| p1 | behavior `b` | shared `f` (`decoder_mlp[0](h_b)`) |
| p2 | shared `f` | **both** `t` (`decoder_mlp[1](h_f)`) and `i` (`decoder_mlp[2](h_f)`) |

`t` and `i` are never decoder inputs, only targets. Both heads read the same `h_f`.

### 2b. PMQ / Query-f block (the shipped "Run-C")
Learnable query slots give each prediction its own hidden state (`mfd.py:225-262`):

```
use_pmq=false, use_query_f=false : [BOS, beh, f]
use_pmq=true                     : [BOS, beh, f, q_t | q_i]
use_pmq=true, use_query_f=true   : [BOS, beh, q_f, f, q_t | q_i]     <- Run-C
```

- `q_f`, `q_t`, `q_i` are free vectors initialized with `randn(d)·0.02` (`mfd.py:250-255`).
- The f-head reads `h(q_f)` (or `h(beh)` without q_f) (`mfd.py:245-248`). `q_f` sits **before**
  `f`, so the causal mask alone hides the teacher-forced `f*` from it.
- **`q_t` and `q_i` never share a pass.** Each runs as the last slot of its own block,
  `[…, f, q_t]` and `[…, f, q_i]` (`mfd.py:359-366`, `425-439`). They are therefore conditionally
  independent given `f` by construction, and every pass uses T5's standard causal mask
  (`attention_mask=None`, `mfd.py:400-416`). No custom 4D mask is needed.
- `decoder_slot_type_emb`: a zero-initialized `Embedding(n_slots, d)` added per slot, indexed by
  the slot's canonical position (`mfd.py:256-262`, `394-397`). T5 has no absolute position
  embedding, so this is the only slot-identity signal beyond relative bias. Zero init means a
  warm start begins exactly at the previous checkpoint.
- `mfd_warmstart_path` loads a prior checkpoint with `strict=False`, so the new slot parameters
  stay at their initialization (`mfd.py:288-295`).

**Training loss** (`mfd.py:666-708` slot mode; `mfd.py:746-779` plain):
`CE(b) + CE(f) + CE(t|f) + CE(i|f)`, with teacher forcing at the ground-truth `b, f`. The label
pipeline is unchanged (labels stay `[b, f, t, i]`; MFD slices them, `mfd.py:721-724`).

### 2c. Dropping the behavior token (needed for Amazon)
Amazon has a single behavior, so the `beh` slot and `behavior_mlp` disappear:
- **Block.** Becomes `[BOS, (q_f), f, q_t | q_i]`. The f-head reads `h(q_f)` when Query-f is
  on, otherwise `h(BOS)`. In the current code, without q_f the f-head reads `h(beh)`
  (`mfd.py:247`), so that position must be redirected.
- **Loss.** Drops `CE(b)` (`mfd.py:694-695`, `748-751`).
- **Stage A.** `_stageA_all_behaviors` (`mfd.py:441-466`) loops the f-head over all `nb` behaviors.
  It collapses to a single pass, and `score_root = logP(f)` with no `w_b·logP(b)` term
  (`mfd.py:796-822`).
- **Decoder-state count.** Drops from `nb·(2+K_f)` to `(1+K_f)`, plus one more pass per root
  under PMQ (§4).
- **Encoder.** The history becomes `[f, t, i, …]` (stride 3). In MACRec it is a text token
  stream instead (`PORTING.md`).
- **Generators.** `generate_conditioned` (the t2 track, behavior forced to ground truth,
  `mfd.py:840-865`) is the template for the only generator needed. "Free" (t3) and
  "conditioned" (t2) become the same thing.

## 3. Scoring

For root `f` with candidate pair `(t, i)` (`mfd.py:32-43`, `566-578`):

```
score(f, t, i) = logP(f | x) + (1/τ) · [ w_t·logP(t | x, f) + w_i·logP(i | x, f) ]
```

- **τ.** `mfd_tau`, default 1.0, exact product of experts. It is deliberately **not** learned, so
  the training path is unchanged; it is swept at inference only (`mfd.py:39-42`).
- **Single-branch ablation.** `w_t, w_i ∈ {0, 1}` via `mfd_score_text` / `mfd_score_image`
  (`mfd.py:572-573`). At least one must be on (`mfd.py:131-132`).
- **Tmall-only behavior term.** On Tmall the root score also carries `w_b·logP(b)`
  (`mfd.py:807`, `819`). Drop it on Amazon.
- **Learned fusion.** Re-weights `[logP(b), logP(f), logP(t|f), logP(i|f)]` on the *selected*
  pool only (`mfd.py:559-562`, `601-609`). Selection is unchanged; this is an ordering-only knob.
  OBSOLETE.

## 4. Candidate construction

### 4a. Dense compositional beam (the path the evaluator uses)
`generate_multibehavior` → `_modality_heads` → `_compose_and_pool` (`mfd.py:784-835`, `517-610`):

1. **Roots.** Top-`K_f` shared codes by `logP(f)` (`mfd.py:805-806`; per behavior on Tmall).
2. **Heads.** One decoder forward per root (two under PMQ) gives `t_logp (B,K_f,k1)` and
   `i_logp (B,K_f,k2)`, the **full** softmaxes (`mfd.py:425-439`, `517-544`).
3. **Grid.** `topk(K_t)` of text and `topk(K_i)` of image per root, giving a
   `K_f × K_t × K_i` additive score grid (`mfd.py:564-590`).
4. **Catalog constraint.** Combinations whose packed key is absent from the catalog get
   `-inf` (`mfd.py:592-595`). This replaces the chain decoder's validity-by-construction (trie).
5. **Pool.** Global `topk(K_pool)` over the grid (`mfd.py:597-598`). The evaluator then takes
   the top-10 by score.

### 4b. Wide-catalog pool (`mfd_wide_catalog_pool`, G5)
At `K_t = K_i = 1024` the dense grid is infeasible (`K_f·1024²` per behavior). Instead, every
**unique catalog tuple** is scored directly in one vectorized pass (`mfd.py:958-1017`):
- `code2root` maps each shared code to its root index, or −1 if `f` is not among the top-`K_f`
  (`mfd.py:964-967`);
- each catalog tuple gathers its root's `logP(t)` and `logP(i)` at its own `t` and `i`
  (`mfd.py:969-973`);
- tuples whose `f` was pruned get `-inf`;
- the **only** truncation is `topk(K_pool)` (`mfd.py:987`).

This is equivalent to `K_t = K_i = full head width` and costs about the same as the dense path.
**It exists only in the audit/dump path `_audit_forward`** (`mfd.py:159-163`), which fed the
offline reranker. It is not in `generate_multibehavior`. A port should put it into the main
generator: on Amazon the catalog is only 6k–14k tuples, so scoring all of them is trivial.

### 4c. K_pool truncation: the measured binding constraint
- With the wide pool, coverage at the **grid** (catalog tuples under the kept roots) rose to
  about 0.756.
- At `K_pool = 500`, coverage at the **pool** stayed flat at 0.2618 (`AGENTS.md:1038-1041`).
  The extra branch-covered ground truths have low generator scores and are cut by the
  hand-set-score `topk(K_pool)` before any reranker sees them.
- Raising `K_pool` to 2000–5000 recovers only a small fraction: about 2.3% of newly covered
  ground truths reach the top-10 (`rec-tmall/GOAL.md:55-56`).
- Lesson for the port: report `coverage@grid`, `coverage@pool` and `P(rank<10 | covered)`
  separately (`rec-tmall/analyze_mfd_audit.py`), so it is clear which cut binds.

### 4d. Decoder-state accounting
`fwd_states_per_item = nb · (2 + K_f)` (`mfd.py:1184-1188`): for each behavior, the behavior and
f slots, plus one modality forward per kept root. `K_t`, `K_i` and `K_pool` add **no** decoder
states, which is the whole point of factorization.
- Caveat 1: under PMQ each root actually runs **two** short passes (`[…, f, q_t]` and
  `[…, f, q_i]`, `mfd.py:431-433`), but the formula counts one.
- Caveat 2: the matched-compute comparison used **808 states**, i.e. `nb=4, K_f=200`
  (`sbatchfiles/matched_compute.sbatch`, `rec-tmall/analyze_matched_compute.py`).
- For a no-behavior port: `(1 + K_f)` states, or `(1 + 2·K_f)` short passes under PMQ.

## 5. Configurations that were run

| run | knobs | source |
|---|---|---|
| MFD (original) | `K_f=K_t=K_i=10`, `K_pool=50`, `τ=1` | `configs/experiment/tiger_mb_mfd_flat.yaml:156-165` |
| **Run-C** (shipped decoder) | PMQ + Query-f + slot-type emb, `K_f=50`, `K_t=K_i=10`, `K_pool=50`, warm start from the plain MFD | `sbatchfiles/train_mfd_pmq_qf.sbatch:13`, `:29`, `:53-58` |
| best final (Tmall-only extras) | Run-C ckpt, wide-catalog, `K_f=200`, `K_t=K_i=1024`, `K_pool=2000`, history forcing, plus reranker | `sbatchfiles/mfd_force_hist.sbatch` |

Run-C checkpoint:
`/scratch/yw8866/logs/train/mfd_pmq_qf/checkpoints/checkpoint_epoch=000_step=033750.ckpt`
(`AGENTS.md:974`).

## 6. Results on Tmall (read with the caveats)

| | t2@10 (behavior-conditioned SID retrieval) | source |
|---|---|---|
| chain baselines, **after** the t2 beam-collapse fix | **0.2032 – 0.2063** (four baselines) | `rec-tmall/results/reeval_baselines_t2fix_metrics.txt:7-10` |
| MFD (original) | 0.1996 | `AGENTS.md:1051` |
| MFD Run-C | **0.2094** | `AGENTS.md:1055` |

- **Do not use the "+27% t2" claim** (`AGENTS.md:834`). The chain t2 numbers it compares
  against (0.157) came from a bug: identical beams collapsed t2 to recall@1, which is why chain
  t2_recall@10 == t2_ndcg@10 in `AGENTS.md:1048-1050`. The fix is at
  `tiger_generation_model.py:1823-1830`. Against the fixed baselines, Run-C leads by only
  **+0.003 to +0.006**. The pools are also unmatched: chain beam 10 vs MFD `K_f=50`, pool 50.
- **End-to-end (t3@10, with the reranker, matched at 808 decoder states, 5 seeds):** MFD 0.2152
  vs chain 0.2135, lead +0.0017, 95% t-interval [+0.0011, +0.0023] (`rec-tmall/GOAL.md:30-35`).
  History forcing (Tmall-only) raised the lead to +0.0024–0.0025 (`rec-tmall/GOAL.md:150-160`).
  The decomposition: `recall@10 = coverage × P(rank<10 | cov)`. MFD has more coverage
  (0.297 vs 0.247) but worse ordering (0.725 vs 0.864) (`rec-tmall/GOAL.md:49-51`).
- So the transferable claim is not "MFD is much more accurate". It is **"MFD buys coverage
  cheaply (K_t/K_i are free), and the open problem is ordering inside the bigger pool."**

## 7. Tmall-only or obsolete: do not port

| feature | where | why obsolete |
|---|---|---|
| behavior marginalization (`mfd_behavior_marginalized`, `mfd_behavior_weight`) | `mfd.py:83-84`, `796-822` | single behavior on Amazon |
| t1/t2/t3 tracks, `generate_conditioned` as a separate track | `mfd.py:840-865`; `src/components/eval_metrics.py:392` | one track (next item) on Amazon |
| history forcing (`mfd_force_history`) | `mfd.py:95`, `165-171`, `939-947`, `975-985` | Tmall repeat-purchase exploit (≈92% of buys are repeats, `AGENTS.md` §12f) |
| reranker dump (`mfd_dump_memory`, `mfd_audit_*`) and the second-stage reranker | `mfd.py:146-158`, `1093-1124`; `rec-tmall/train_mfd_reranker.py` | separate project; revisit only after the base port works |
| learned fusion, branch contrastive, CMIR | `mfd.py:100-102`, `92-94`, `107-114`, `264-286` | side experiments. CMIR got t3@10 0.2024, slightly below Run-C (`AGENTS.md:1056`) |
| behavior-weighted SID loss, focal behavior loss | `mfd.py:696-705`; parent class | behavior-imbalance fixes |
