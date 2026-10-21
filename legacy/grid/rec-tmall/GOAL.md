# GOAL: MFD must beat chain on t3 recall@10 by >= +0.01 absolute

Living document. Updated after every job. Sections 1-4 are the standing frame; section 5 is the
running log (newest last); section 6 is the queue.

---

## 1. The bar

> MFD beats the chain baseline on t3 recall@10 by a margin that is (a) **>= +0.01 absolute**,
> (b) significant under the paired bootstrap, (c) stable across seeds, (d) explained by a mechanism
> chain cannot replicate.

Set 2026-09-24 (supersedes the earlier "1% relative" reading, which was +0.00214).
Chain sits at 0.21351 (mean of 5 seeds), so **MFD must reach >= 0.22351**, from 0.21524 today:
**+0.00827 to find, on top of a lead that is currently +0.00173.** The bar applies to the mean over
seeds, with the across-seed t-interval of the mean lead entirely above +0.01.

**This is 5.8x the current lead, and larger than any single result in the ledger** (the best feature
ever found, hard history matching, was worth +0.0060 and lifted both arms). It is not reachable by
another feature of that size. Section 3 identifies the only population large enough to supply it.

---

## 2. Where we stand (2026-09-21, 5 seeds, settled)

> **Superseded** — this is the 2026-09-21 baseline. For the current best (+0.0024–0.0025 with history forcing)
> see **CURRENT STATE** at the top of section 5.

| | coverage | P(rank<10 \| cov) | t3@10 mean | t3@10 std |
|---|---|---|---|---|
| MFD (P3 wide, K_pool=2000, R2_rw1, capped) | 0.2968 | 0.7252 | **0.21524** | 0.0005 |
| chain (808 states, K_pool=500, R2_rw1, capped) | 0.2471 | 0.8642 | **0.21351** | 0.0003 |

- **Lead: +0.00173, across-seed std 0.00047, 95% t-interval [+0.00114, +0.00231].**
- 5/5 seeds MFD wins; 5/5 single-seed paired CIs exclude 0.
- **Relative lead = 0.81%. Against the +0.01 absolute bar we are short by +0.00827 = 687 events.**
- Seed noise (0.00047) > within-seed event-bootstrap SE (0.0004): a single-seed CI understates the
  uncertainty of the lead. **Every future claim about the lead must be multi-seed.** The single-seed
  +0.0022 that motivated this goal was the luckiest of five draws.

Source: `seedvar/seed_variance_report.txt`, `seedvar/seed_variance.json`.

---

## 3. The identity, and the only population big enough for +0.01

```
recall@10 = coverage x P(rank<10 | covered)
MFD   0.2968 x 0.7252 = 0.2152      ceiling (= coverage) 0.2968, headroom 0.0816
chain 0.2471 x 0.8642 = 0.2135      ceiling 0.2471,        headroom 0.0336
```
+0.00827 of t3@10 = **687 more test events in the top-10**. Where can 687 events come from?

**Not coverage.** Newly covered GTs reach the top-10 at a measured 2.3% (K_pool 500->2000) and 1.0%
(->5000). K_pool 500->5000 netted +0.0005/+0.0007. Coverage 0.50 would need K_pool ~ 100,000.

**Not conversions.** fav+cart+buy are 5,172 test events (6.2%) at t3@10 0.94-0.96 with coverage ~0.99.
Driving all three to 1.00 is worth at most **+0.0031**, and chain is already level or ahead there
(chain cart 0.963 / buy 0.946 vs MFD 0.960 / 0.941).

**Not "MFD orders worse, fix the eviction".** The pcov drop from 0.821 (K_pool=500) to 0.725
(K_pool=2000) looks like eviction but is composition: of the 24,636 covered events at K=2000, 21,733
were already covered at K=500 and 2,903 are new; new ones convert at 2.3% (~67 hits), and total hits
moved 17,838 -> 17,863. **Eviction from widening costs ~42 events.** There is no eviction bug to fix.

**MEASURED (headroom audit, 5 seeds, `headroom_report.txt`).** The lead decomposes by pool membership:

| population | events | MFD recovery | chain recovery | contribution to the lead | MFD misses |
|---|---|---|---|---|---|
| BOTH | 19,280 | 0.9086 | **0.9137** | **-0.00119** | 1,762 |
| **MFD-ONLY** | **5,355** | **0.0649** | 0 by construction | **+0.00419** | **5,008** |
| CHAIN-ONLY | 1,227 | 0 (GT not in MFD's pool) | 0.0861 | -0.00127 | 1,227 |
| NEITHER | 57,143 | — | — | 0 | — |

Three things fall out, two of them new:
1. **MFD's entire lead is its extra coverage** (+0.00419), and it is being eaten by two deficits.
2. **Chain is AHEAD of MFD on the events both arms cover** (0.9137 vs 0.9086 = -0.00119). MFD does not
   merely "order worse overall because its pool is bigger" — it orders worse on identical GTs.
3. **MFD-ONLY converts terribly: 6.5%.** Its ceiling is +0.0603, so the bar needs **13.7% of its 5,008
   misses**. That is the target, and chain gains nothing there.

Where those misses sit after reranking is the crucial detail:
`10-19: 273 · 20-49: 380 · 50-99: 426 · 100-499: 1,601 · 500+: 2,329` — **73% are still at rank >= 100
after the reranker has run.** Recovering every near-miss (rank 10-49, 653 events) yields +0.00786,
which does not quite reach +0.00827 on its own. So the bar requires reaching into the 50-499 bands,
and that is a statement about the SCORE FORM, not about features: with
`score = base_scale * base + bounded delta`, a candidate the generator ranked near 1000 cannot
reach the top 10 no matter how good its features are. Hence the gate variant in 6.1.

This also reinterprets the CF-C result. CF-C was "generic" because it lifted the events **both** arms
already cover. A feature does not have to be exotic to be MFD-specific in effect — it has to move the
own-only population, where chain has nothing to gain by construction.

Supporting facts that stay true: MFD's **base score** is the weak part, not its reranker (base ->
reranked is 0.2041 -> 0.2152 for MFD but 0.2120 -> 0.2135 for chain), and on shared events MFD's
reranker is already clearly better (BOTH/10-49 0.6495 vs 0.4136).

---

## 4. Ledger (do not re-litigate)

**Works** — hard history matching (+0.0060, MFD gain 5.3x chain's); K_pool=2000; rw1 rank weighting
(BOTH/50+ 0.2236 -> 0.2656 at no aggregate cost); paired bootstrap protocol (unpaired inflates the CI 5x,
arms correlate at 0.96).

**Dead** — modality views as candidate representation (random vectors tie learned ones, twice);
chain conditional logP(i|f,t) as a rescoring term (weight 0.0764, +0.0001); branch score calibration
(z-standardisation collapsed pcov 0.78 -> 0.60; rank variant -> 0.42 — **absolute log-probs carry real
information**); reranker capacity (466k -> 3.46M moved 0.0002); time decay / popularity / session /
user stats / branch entropy; shared-code CF (CF-A, CF-B: -0.0013 and -0.0007, CI entirely below zero,
despite univariate AUC 0.92); further coverage expansion.

**Overturned** — "chain structurally cannot recover deep-ranked candidates" was a feature-set artefact:
item-item co-occurrence raised chain's own-only/50+ from 0.0034 to 0.0626, above MFD's 0.0392. The
claim now holds only for **BOTH/50+ (MFD 0.2133 vs chain 0.0894)** or in the history-features-only setting.

**Method rules** — univariate AUC does not predict incremental value; always run the chain arm
alongside and report generic gains as generic; assert dataset shape at generation time; report the NET
effect decomposed into gained and lost; never call a verdict from a handful of events; **and now: never
call a lead from one seed.**

---

## 5. Log

### >>> SUMMARY TABLE — every variant, MFD vs chain (added 2026-10-14) <<<
808 decoder states, R2_rw1 reranker, capped, TEST 83,005. lead = MFD - chain (t3@10, absolute); rel = lead / chain.

| # | variant | seeds | MFD | chain | lead | rel | note |
|---|---|---|---|---|---|---|---|
| 1 | baseline (P3 wide pool, K_pool=2000) | 5 | 0.2152 | 0.2135 | +0.00173 | +0.81% | t-int [+0.0011, +0.0023] |
| 2 | + BPF / gate / rw2 / hn64 (VAL-best) | 3 | 0.2155 | 0.2137 | +0.00180 | +0.84% | change -0.00016 |
| 3 | + reserved new-item slot P1 | 5 (offline) | 0.2139 | 0.2121 | +0.00186 | +0.88% | both arms drop; VAL rejects |
| 4 | + item dedup D | 3 | ~0.145 | ~0.14 | — | — | catastrophic |
| 5 | + new-item reweight w_new=3 | 3 | 0.2149 | 0.2134 | +0.0015 | +0.70% | null |
| 6 | history injection, both arms | 3 | 0.2148 | 0.2139 | +0.00094 | +0.44% | not significant |
| 7 | calibrated mixture + new-item specialist | 3 | 0.2153 | 0.2137 | +0.00153 | +0.72% | VAL picks P0 |
| 8 | + repeat-consumption features | 3 | 0.2154 | 0.2135 | +0.00194 | +0.91% | null |
| 9 | **history forcing** | 3 | 0.2162 | 0.2138 | **+0.00241** | +1.13% | 3/3 seeds significant |
| 10 | forcing + per-behavior bias | MFD 3 / chain 1 | 0.2170 | 0.2145 | ~+0.0025 | ~+1.17% | chain seed 0 only |
| 11 | forcing + both specialists (P0) | 3 | 0.2161 | 0.2136 | +0.00254 | +1.19% | t-int [+0.0012, +0.0038] |
| — | ceiling: forced pool, correct behavior | 1 | 0.2245 | 0.2203 | +0.0043 | +1.9% | not achievable |
| — | target | | | | +0.0100 | ~+4.7% | |

Only #9-#11 reach ~+0.0025 absolute, and all three are the same mechanism (history forcing): #11's VAL choice is
P0 (= plain forcing) and its gap to #9 is chain-retrain seed noise (across-seed std ~0.0005); #10's bias adds
~+0.0007 to MFD and ~+0.0004 to chain. Robust statement: history forcing gives a stable ~+0.0024-0.0025 lead.

### >>> CURRENT STATE (updated 2026-09-27) — read this first <<<
**Best configuration:** MFD with HISTORY FORCING (history shared codes kept among the K_f=200 roots, history
tuples kept in K_pool=2000 with true scores; 808 decoder states) + R2_rw1 reranker, capped. Chain unchanged.
**Lead +0.0024 to +0.0025 (3 seeds, every per-seed paired CI > 0). Bar +0.01 NOT met; short ~+0.0075.**

| step | lead (3 seeds) |
|---|---|
| baseline (5 seeds) | +0.00173 |
| + history forcing (job 18643702) | +0.00241 |
| + per-behavior score bias, VAL-tuned (job 18656572; MFD 0.2170 vs chain 0.2145) | ~+0.0025 |
| + new-item AND repeat specialists in the calibrated mixture (job 18665870) | +0.00238 (VAL keeps P0/M-1) |

**Where the remaining gap is (forced pool, decomposition):** MFD-ONLY repeats (1,020, in MFD's pool, not
chain's) convert at 0.384 -> +0.0047; BOTH new items -0.0011 (mostly SID-collision repeats in chain's space);
CHAIN-ONLY new -0.0004. **Ceiling with 10 distinct items and the correct behavior: MFD 0.2245 vs chain 0.2203,
lead +0.0043** (job 18671431). The binding constraint is ITEM-level ranking of hard repeats: the correct item
reaches the top-10 distinct items for only 0.587 of MFD-only repeats (0.983 for BOTH). With every covered
repeat ranked perfectly in both arms the forced-pool lead would be ~+0.0105 -- the bar is reachable only
through near-perfect "which past item comes back" prediction.

**Revisit heuristics (CPU):** every recency/frequency/behavior rule lands at 0.56-0.59 on the forced-only
repeats, the same as the reranker (0.587) -- the hard repeats sit at the ceiling of the available signal.
**Revisit model (job 18678090, DONE 2026-09-28):** a transformer over history slots trained on 4,000,003
repeat events from the full training split (100x the rerankers' data), structural features only. VAL
item-level top-10 hit 0.965 (vs most-recent-first 0.948). TEST: BOTH repeats 0.980 (reranker 0.983),
**forced-only repeats 0.580 (reranker 0.587)**, all pv repeats 0.952 (0.954), conversions 0.998.
**Confirmed: more data on the same position-based signal does not move the hard repeats. The ~0.59 wall is
a property of the input (item order without times), not of model capacity or training volume.**

**Every input in the data is now tested on the hard repeats (2026-09-28)** — item order/position (0.591),
a 4M-event revisit model (0.580), second-level timestamps (0.598 using request time, 0.580 leakage-free), and
full SigLIP text+image content (0.604 best, combined with position), against the reranker's 0.587. The returning
item among a hard-repeat user's ~29 past items is not identifiable from anything this dataset records beyond
~0.60. With that item-level ceiling the forced-pool lead ceiling (perfect behavior) is ~+0.0043; the measured
lead is +0.0024-0.0025. **+0.01 is not reachable under a fair comparison on this data.**

**Timestamps tested (2026-09-28): NO help.** 100% aligned; best time rule 0.598 on forced-only repeats (0.580
leakage-free) vs 0.591 position / 0.587 reranker. These are ~20 h-gap revisits (6.4% within 30 min). Neither
the order nor the timing of the history identifies the returning item; model capacity (4M-event revisit model:
0.580) and more signals (repeat features, repeat specialist) all land at ~0.59.

(Original plan, kept for the record:) **A NEW input: timestamps.** The TFRecords (and so the dumps) keep item ORDER but drop TIMES;
the raw logs have second-level timestamps spanning months (e.g. one item viewed at 16:22:32 and again at
16:25:47). Time since an item's last view / same-session views is the classic repeat-consumption signal and
list position is only a proxy for it. Job 18684595 (`ts_sidecar.py`, CPU) recovers each dump event's 40
history timestamps + target timestamp from the filtered logs (builder sorts by (user, timestamp); alignment
verified per event against the behavior sequence). Go/no-go (`time_heuristics.py`): item-level top-10 hit on
the 1,020 forced-only repeats with time-based ranking vs the ~0.59 wall -- in two versions, one using the
request (target) time and one using only the user's last history action (no target time, leakage-free).


### 2026-09-21 — seed variance (settles the gate)
5 seeds, both arms, identical protocol. Result in section 2. **The lead is +0.00173 +- 0.00047** —
real and significant, but far below the bar. The +0.0022 single-seed figure was the maximum of five
draws (range 0.00125 to 0.00225). No feature work should be judged against +0.0022.

### 2026-09-24 — bar raised to +0.01 absolute; plan re-derived
The gap to close went from +0.00041 to **+0.00827 (687 events)**. Section 3 was rewritten: coverage,
conversions and "fix the eviction" are all arithmetically too small, and the only population large
enough *and* structurally MFD's is **the events MFD covers and chain does not**. Added a STEP 0
headroom audit (`headroom.py`, no training, runs off the cached per-event results) that measures that
population exactly — size, recovery, miss-rank distribution, and the recovery rate it would need to
carry the whole gap — so the feasibility question is answered before GPU time is spent.

### 2026-09-24 — headroom audit RUN (5 seeds): the target is now measured, not assumed
`headroom.py` on the cached per-event results. Full decomposition in section 3. Headlines:
- **MFD-ONLY = 5,355 events at 6.5% recovery**, ceiling +0.0603; the bar needs **13.7% of its misses**.
  Feasible on paper, and chain cannot follow there.
- **Chain beats MFD on the 19,280 events both cover** (0.9137 vs 0.9086), costing -0.00119. MFD's
  deficit is not just "bigger pool, more competitors" — it is worse ordering on identical GTs.
- **CHAIN-ONLY costs another -0.00127** and is unreachable by reranking (the GT is not in MFD's pool).
- **73% of MFD-ONLY misses are still at rank >= 100 after reranking.** Recovering every near-miss
  (rank 10-49) gives +0.00786 — just short of the +0.00827 needed. The bar therefore cannot be met by
  near-miss recovery alone; it needs candidates promoted from the 50-499 bands, which the current
  score form (`base_scale * base + bounded delta`) structurally prevents.

This last point is why the job now screens a **learned per-candidate base anchor** (GateRanker:
`2*sigmoid(gate(h)) * base_scale * base + delta`, identity at init) alongside the features.

**Migration analysis (same audit) — what the reranker actually does to each population:**

| population | arm | base hits | reranked hits | promoted | demoted | median rank before -> after |
|---|---|---|---|---|---|---|
| BOTH | MFD | 16,694 | 17,518 | 974 | 150 | 1 -> 1 |
| BOTH | chain | 17,489 | 17,617 | 288 | 160 | 1 -> 1 |
| MFD-ONLY | MFD | 251 | 347 | 206 | 110 | **429 -> 378** |

Two conclusions that redirect the work:
- **The BOTH deficit is inherited from the generator and the reranker already repairs 87% of it.**
  MFD starts -795 events behind chain on shared events and ends -99. Closing it completely is worth
  only **+99 events (+0.0012)** — that avenue is nearly exhausted, and it was never chain-proof anyway.
- **On MFD-ONLY the reranker is nearly inert: the median GT rank moves 429 -> 378**, and it promotes
  206 while demoting 110. It clearly *can* promote (974 promotions on BOTH) — but only from near the
  top. **To clear the bar, MFD-ONLY promotions must rise from 206 to ~890 (4.3x), or ~790 (3.8x) if the
  BOTH deficit is also fully closed.** That is the number to beat, and it is a property of the score
  form, not of the feature set.

### 2026-09-24 — migration analysis: the reranker is inert on the population that matters
Added to `headroom.py`. Table in section 3. The BOTH deficit turns out to be generator-inherited and
already 87% repaired (only 99 events left there), and on MFD-ONLY the reranker moves the median GT
rank only 429 -> 378. **Quantified target: MFD-ONLY promotions 206 -> ~790-890, a 3.8-4.3x increase.**
This is the prediction the gate variant has to satisfy; if it moves the median rank of MFD-ONLY
substantially and still does not promote ~790, the score form is not the binding constraint and the
remaining explanation is that those GTs are genuinely indistinguishable at rank ~400.

### 2026-09-25/26 — NIP and INJECTION: every remaining reranker/candidate lever is closed
**NIP (job 18512956, 12 runs, both arms, 3 seeds):** VAL chose **P0 (today's list) in every run.**
Reserved new-item slots P1-P3 are worse everywhere (confirming the offline estimate); w_new=3 gives
nothing (MFD 0.2149 vs 0.2153). **Item dedup D is catastrophic (0.14 vs 0.21)** — and it exposes that
~4 of every 10 top-10 slots are the SAME item under a different behavior, in both arms (MFD 3.8-4.0,
chain 3.6-4.5). Those are not wasted slots: they are how the reranker hedges its uncertainty about WHICH
behavior comes next; dropping them drops the correct (behavior, item) pair. (The job's final report
crashed on a wrong assertion — the repeat flag legitimately differs between the two SID spaces; fixed.)

**Injection (job 18514124):** giving BOTH arms their users' history items as candidates (identical,
uninformative generator score) helped NEITHER: MFD 0.2155 -> 0.2148, chain 0.2136 -> 0.2139. Coverage
rose (chain +0.012, MFD +0.006) but the reranker cannot promote candidates with a bottom-of-pool base.

| comparison | lead | t-interval |
|---|---|---|
| neither injected (today) | +0.00196 | [+0.00082, +0.00311] |
| MFD vs chain + injection | +0.00162 | [-0.00230, +0.00554] |
| both injected | +0.00094 | [-0.00183, +0.00372] |

Chain did not *replicate* the lead, but **the lead does not survive a symmetric change to the candidate
set** — its significance across seeds is gone once both arms carry the repeats.

### 2026-09-26 — GOAL RE-AFFIRMED by the user; the new-item conversion gap is a RERANKER failure
User's correction, accepted: chain's reranker converts **5.9%** of its covered pv-new GTs (P(r<10|c)=0.0589,
MFD partition) while MFD's converts **0.1%**, although MFD's own generator had 2.8% before reranking. That
is not a signal ceiling. The previous "unreachable" conclusion assumed MFD's new-item conversion was fixed.

Breakdown (`predictable_breakdown.txt`, 5 seeds): PREDICTABLE events MFD cov 0.973 x P 0.950 vs chain
0.944 x 0.965 -> lead +0.0033; NEW events MFD cov 0.092 x P 0.001 vs chain 0.036 x 0.059 -> -0.0016.
Caveat: the NEW-row P depends on which SID space defines "repeat" (chain's own def: chain 0.017, MFD 0.012).

**Oracle-gate ceiling** (`oracle_gate.txt`): repeat events keep today's list, new events get a top-10 of
non-history candidates by BASE score. New-item P: MFD 0.001 -> 0.056, chain 0.017 -> 0.1025.
Symmetric lead +0.0033; MFD-only gate +0.0056. **MFD covers 3,643 more genuinely new items than chain**
(5,839 vs 2,196). With equal new-item P for both arms, that coverage is worth P x 0.0439 of t3@10, so the
bar needs a LEARNED new-item ranker around P ~ 0.15 inside MFD's pool. That IS a mechanism chain cannot
replicate: its beam does not reach those items.

**Build (job 18580730, `nip_mix.sbatch`):** calibrated per-event mixture, both arms identically, chosen on
VAL: P(c) = G' P_main(c|history) for history candidates, (1-G') P_new(c) otherwise; G = main reranker's
mass on history, logit(G') = logit(G) - beta; P_new = new-item specialist (non-repeat events, history
masked). Verified end-to-end on a synthetic pool; extreme-beta semantics checked.

Also: the contrastive fine-tune (18562804) never ran — Hydra rejected '=' in the warm-start path. Fixed
with a symlink; held until the mixture result says whether base new-item scores are the bottleneck.

### 2026-09-27 — mixture + specialist, and repeat-consumption features: both null (3 seeds, both arms)
| experiment | lead | MFD-only repeats | BOTH new items MFD / chain |
|---|---|---|---|
| baseline | +0.00196 | 0.51 | 0.001 / 0.084 |
| calibrated repeat/new mixture + new-item specialist (18580730) | +0.00153 | 0.516 | 0.004 / 0.079 |
| repeat-consumption features (18605626) | +0.00194 | 0.509 | 0.005 / 0.083 |
- The new-item specialist (5,149 non-repeat training events) reaches P ~0.054 on VAL = the base score.
- A history-only repeat/new gate reaches AUC 0.64 (`gate_audit.py`); VAL picks beta=-1/0 (= today's list).
- **Top-10 diagnostic** (`topk_diag.txt`): MFD's top-10 is ~10/10 history items. MFD-only repeat misses
  lose to OTHER history items (70% at rank 10-19; no behavior-hedge slot above them); covered new items
  never enter (9.95/10 history). Repeat features did not change this -> the reranker already has that info.
- **Correction to a premise**: chain's "5.9% on covered new items" is 1.7% when each arm's history is read in
  its own SID space (MFD 1.2%); the gap is mostly SID-collision repeats. The one clean per-type deficit is
  the 1,176 BOTH new-item events (MFD 0.005 vs chain 0.083 on identical GTs). Job 18641005 compares both
  arms' top-10 on those events to find why chain's list leaves room for them.

### 2026-09-27 — the lead is capped by POOL coverage of predictable events; history forcing lifts the cap
**Behavior-hedge cap / ceiling** (`cap_eval2.txt`, saved seed-0 models): the top-10 holds the same item under
~3 behaviors because the reranker's top behavior for the right item is wrong ~1/3 of the time (cap-1: BOTH
repeats 0.97 -> 0.66). Freeing hedge slots lifts MFD-ONLY repeats 0.545 -> 0.736, but VAL keeps no cap.
**Ceiling with 10 distinct items and the correct behavior: MFD 0.2231, chain 0.2203, lead +0.0028.** At
that ceiling predictable events convert at ~0.99 for both, so the lead = P x (MFD's extra predictable pool
coverage): 563 events -> <= +0.0067. **+0.01 needs ~840+ more predictable events in MFD's pool than chain's.**
Also: chain's "8.3% on shared new items" is 27/1,124 (2.4%) under its own SID space -- collision repeats.

**History forcing** (`+model.mfd_force_history`, job 18643702): keep the user's history shared codes among the
K_f=200 roots (1 root each; t/i heads score every tuple under them for free) and put history tuples in the
pool with their TRUE scores. Decoder states unchanged (808, validated). Test dump:

| | MFD before | MFD forced | chain |
|---|---|---|---|
| pv repeat coverage | 0.9662 | **1.0000** | 0.9279 |
| predictable coverage | 0.9733 | **0.9993** | 0.9441 |
| predictable in pool | 18,796 | **19,298** | 18,233 |

**Advantage 1,065 events (> the ~840 needed).** The coverage cap on the lead is lifted. Whether it converts
depends on P(rank<10|covered) on the forced repeats; the rerank (3 seeds) is running in the same job.

### 2026-09-28 — RAW CONTENT (SigLIP2 text + image, 1152-d) does not break the wall either
Job 18711859: sidecar extended to recover each history slot's ITEM ID (alignment 1.0000 on 83,005 TEST events;
text and image embeddings present for 100% of slots), then `content_heuristics.py`: rank each user's past items
by cosine similarity of their full SigLIP embedding to the mean of the user's last k items (k = 1, 3, 5), alone
and combined with the best position rule. Item-level top-10 hit:

| ranker | BOTH pv repeats | forced-only pv repeats | all pv repeats |
|---|---|---|---|
| best POSITION rule | 0.977 | 0.591 | 0.949 |
| CONTENT alone (text k=5, best single) | 0.878 | 0.478 | 0.849 |
| CONTENT alone (image k=3) | 0.938 | 0.448 | 0.903 |
| POSITION + 2 x image k=3 | 0.981 | **0.604** | 0.954 |
| POSITION + text k=3 + image k=3 | 0.983 | 0.601 | 0.956 |
| reranker (forced pool, seed 0) | 0.983 | 0.587 | 0.954 |

Best gain on the hard repeats +0.013 item-level (~13 of 1,020 events) = ~+0.0002 of t3@10. Content similarity to
recent views is a weaker predictor than position on its own, and adds almost nothing on top of it.

### 2026-09-28 — TIMESTAMPS (a genuinely new input) do not break the wall either
Job 18711015 (`ts_sidecar.py`, CPU, 10 min after resizing the request from 200 GB to 64 GB): recovered each
dump event's 40 history timestamps + target timestamp from the filtered raw logs. **Alignment 1.0000 on all
six dumps** (MFD forced test/val/train, chain test/val/train), verified per event against the behavior
sequence. Go/no-go (`time_heuristics.py`, `force_exp/time_heuristics.txt`), item-level top-10 hit:

| ranker | BOTH pv repeats | forced-only pv repeats | all pv repeats |
|---|---|---|---|
| best POSITION rule (freq + recency + conv) | 0.977 | 0.591 | 0.949 |
| TIME: most recent in seconds | 0.963 | 0.562 | 0.934 |
| TIME: views in session, then recency | 0.963 | 0.562 | 0.934 |
| TIME: -log gap + log count + conv (uses request time) | **0.978** | **0.598** | 0.950 |
| TIME+POS combined | 0.974 | 0.582 | 0.946 |
| NO-TARGET-TIME (leakage-free) variants | 0.974-0.975 | 0.578-0.580 | 0.945-0.946 |
| reranker (forced pool, seed 0) | 0.983 | 0.587 | 0.954 |

Why: for the forced-only repeats the median gap from the user's last action to the request is **72,339 s
(~20 h)**, and the returning item was viewed within 30 min of the request in only **6.4%** of cases. These are
long-gap revisits, not in-session repeats; neither order nor time identifies which of ~29 past items returns.

### 2026-09-28 — revisit model on 4M repeat events: 0.580 on the hard repeats (confirms the wall)
Job 18678090 (`revisit.py`): 2-layer transformer over the 40 history slots, structural slot features (recency,
behavior, occurrence count, first/last position, the item's behavior set, #distinct), softmax over
last-occurrence slots; trained on 4,000,003 repeat events from 400k training users (3 epochs).

| item-level top-10 hit | revisit model | reranker (forced pool, seed 0) |
|---|---|---|
| VAL (last-position repeat events) | 0.965 (most-recent-first 0.948) | — |
| TEST BOTH pv repeats (13,120) | 0.980 | 0.983 |
| TEST **forced-only pv repeats (1,020)** | **0.580** | **0.587** |
| TEST all pv repeats (14,140) | 0.952 | 0.954 |
| TEST conversions (repeat, 5,153) | 0.998 | 0.992 |

100x the training data on the same input lands on the same ~0.59. The limit is the input -- item ORDER without
TIMES -- which is why the next step is the timestamp sidecar (job 18684595).

### 2026-09-27 — revisit heuristics: the hard repeats sit at the signal ceiling (~0.59)
Item-level top-10 hit on TEST pv repeat events (forced pool; CPU, no training):

| ranker | BOTH repeats (13,120) | forced-only repeats (1,020) | all pv repeats |
|---|---|---|---|
| most recent first | 0.963 | 0.561 | 0.934 |
| most frequent, then recent | 0.962 | 0.568 | 0.933 |
| frequency x recency | 0.975 | 0.582 | 0.947 |
| recency + past fav/cart/buy bonus | 0.973 | 0.591 | 0.946 |
| frequency + recency + conversion | 0.977 | 0.591 | 0.949 |
| **reranker (forced pool, seed 0)** | **0.983** | **0.587** | **0.954** |

Forced-only repeat users have a median of 29 distinct past items; every structural rule and the trained
reranker (which also cross-attends over the full content history) land at 0.56-0.59. The +0.01 path on the
forced pool needs item-level ranking near 1.0 on these events; the available signal stops at ~0.59. The
revisit model (job 18678090, same structural signal, more data) is kept queued as confirmation.

### 2026-09-27 — history forcing: rerank result (job 18643702, 3 seeds)
MFD 0.2160 / 0.2163 / 0.2162 vs chain 0.2141 / 0.2134 / 0.2138 -> **lead +0.00241** (CIs [+0.0011,+0.0027],
[+0.0022,+0.0037], [+0.0017,+0.0032]). pv repeats: MFD cov 1.000, recall 0.9226 vs chain cov 0.928, recall
0.8971 (+0.0044); pv new -0.0016; conversions -0.0004. The 1,020 forced-only repeats convert at 0.386: coverage
is no longer the cap, conversion of hard repeats is. (The job's report step crashed on chain results cached
before the mixture policies existed -- KeyError 'M-1' -- fixed; numbers computed from the cached files.)

### 2026-09-27 — per-behavior score bias, tuned on VAL (job 18656572, saved models)
Grid over (pv, fav, cart, buy) biases, chosen per arm on VAL: MFD seeds 0/1/2 TEST 0.2160->0.2167,
0.2165->0.2174, 0.2163->0.2168 (best bias pv +0.5, conversions 0/-1); chain 0.2141->0.2145. **Lead ~+0.0025.**
A global bias cannot fix per-item behavior ordering.

### 2026-09-27 — both specialists with the feature bug FIXED (job 18665870, 3 seeds, both arms)
Bug found: each specialist has its own random code tables, but at evaluation it was fed the MAIN model's soft
history-match features. Fixed. With the fix: new-item specialist VAL hit@10 still ~0.054 (= base score);
repeat specialist (history-only, no base anchor) VAL repeat hit@10 0.93 (= main reranker). **Lead +0.00254
(P0) / +0.00238 (VAL-chosen).** The earlier null mixture result was not caused by the bug.

### 2026-09-27 — forced-pool ceiling (job 18671431)
10 distinct items, correct behavior: **MFD 0.2245, chain 0.2203, lead +0.0043.** Item-level top-10 hit:
BOTH repeats 0.983 / 0.988, forced-only repeats **0.587**, all pv repeats 0.954 / 0.917, conversions
0.992 / 0.985. Cap-2 behaviors raises the lead to +0.0081 but lowers BOTH arms' recall (MFD 0.2001), so VAL
correctly rejects it -- choosing a policy that shrinks both arms to widen the gap is not legitimate.

### CONCLUSION (2026-09-26) — SUPERSEDED on 2026-09-27: history forcing lifted the coverage cap it relied on
**The +0.01 absolute bar is not reachable under a fair comparison with these generators on this metric.**
The evidence chain:
1. t3@10 here is almost entirely **predictable events** — page-view repeats and conversions; new-item
   pv recall is ~0 for both systems (MFD 6 hits, chain 134, out of 63,693).
2. MFD already holds 97.3% of the predictable events in its pool (chain 94.4%). **Perfect reranking of
   every predictable event, applied to both arms, caps the lead near +0.0052.**
3. Every reranker lever was tested and closed: features (BPF), score form (gate), objective (hn64),
   deep weighting (rw2), pool selection (tau), new-item slots (P_k), item dedup (D), new-item reweighting.
4. MFD's current lead (+0.0017 to +0.0020, 5/5 seeds) is real but rests on repeat coverage, and loses
   significance under symmetric candidate injection — criterion (d) is not robustly met even for it.

What WOULD have to change for +0.01 to be reachable: MFD's generator would need to cover essentially
every predictable event AND the baseline be barred from repeat-candidate generation (a protocol choice a
reviewer would challenge), OR MFD would need to predict genuinely new items far better than any model
here does. The one untested generator-side mechanism (within-branch contrastive fine-tune) is re-queued
below with that expectation stated; it is the last lever, not a likely one.

### 2026-09-25 — THE FAIR CEILING: perfect reranking of the current pools reaches ~+0.005, half the bar
Almost every hit either system gets is a PREDICTABLE event — a page-view repeat or a conversion (19,312
of 83,005 events). MFD has 18,796 of them in its pool and hits 17,860 (936 in-pool misses); chain has
18,233 and hits 17,589 (644 misses).

| scenario (current generators and pools) | lead |
|---|---|
| today | +0.0017 |
| MFD recovers ALL its in-pool predictable misses, chain frozen | +0.0130 |
| **both arms recover all of theirs (any symmetric, fair method)** | **+0.0052** |
| only the MFD-ONLY repeats recovered (chain cannot follow those) | +0.0056 |

**The +0.01 bar is reachable on the current pools only by improving MFD's reranker and not chain's,
which is not a legitimate comparison.** Under a symmetric method even perfect reranking of every
predictable event tops out near +0.005. Clearing +0.01 fairly therefore needs MFD's GENERATOR to put
more predictable events in its pool than chain's does — pure pool composition, not ordering.

**And a validity risk for the lead we already have:** MFD's whole lead is repeat coverage, and adding
the user's own history items to the candidate set is standard repeat-candidate generation that any
system can do at zero decoder cost. Job 18514124 (`inject_rerank.sbatch`) gives BOTH arms exactly that
(identical construction, uninformative generator score) and asks whether MFD keeps any lead once both
arms have the repeats. If chain + injection matches MFD, the current lead fails criterion (d).

### 2026-09-25 — NIP offline estimate: reserving slots for new items is net-NEGATIVE for both arms
`nip_offline.py` (CPU, cached 5-seed results + dumps). Best k = 0 for BOTH arms:

| k reserved | MFD | chain | lead |
|---|---|---|---|
| 0 | 0.2152 | 0.2135 | +0.00173 |
| 1 | 0.2139 | 0.2121 | +0.00186 |
| 2 | 0.2117 | 0.2103 | +0.00140 |
| 3 | 0.2088 | 0.2075 | +0.00131 |

One reserved slot recovers ~58 MFD new-item hits (BOTH new 0.0009 -> 0.0223, MFD-ONLY new 0.0010 ->
0.0081) but costs ~139 repeat hits. **The reranker's "eviction" of new items was rational, not a bug**:
the marginal repeat slot is worth more than the generator's best new-item guess. A new-item specialist
would need ~2.4x the base score's top-1 new-item accuracy to break even (built: `--specialist`, not run).

Added to the pending GPU job instead: **policy D, one slot per ITEM.** t3 ranks the union of four
behavior pools, so (pv, X) and (cart, X) can both take top-10 slots; MFD's per-behavior pools are 4x
chain's, and its MFD-ONLY repeats sit at median rank ~9, right at the boundary a freed slot would move.
The job also measures how many of today's top-10 slots are duplicate items. Verified end-to-end on a
synthetic pool through the real Ranker: P0 reproduces metrics() exactly.

### 2026-09-25 — THE REFRAME: t3@10 on page-views is repeat retrieval, and MFD's reranker evicts new items
Two cheap analyses on the cached per-event results (5 seeds) changed the picture more than any run.

**1. The head-sharpness dose-response was a repeat artefact.** MFD-ONLY conversion climbs from 0.009
(diffuse heads) to 0.455 (t*=i*=0), but 82% of the sharp items are REPEATS (GT tuple already in the
user's history). Non-repeat MFD-ONLY items convert at ~0.000 even when both heads rank the GT first
(103 such items, 0 hits). Sharp heads were a proxy for "the user has seen this item".

**2. For pv, both systems succeed almost only on repeats:**

| pv (77,833 events) | hits from repeats (14,140 events) | hits from new items (63,693) |
|---|---|---|
| MFD | 12,951 | **6** |
| chain | 12,669 | **134** |

MFD's lead is repeat coverage (+0.0034) partly given back on new items (-0.0016).

**3. On new items MFD's reranker destroys what its generator found:**

| pv new items | n | MFD base -> rerank | chain base -> rerank |
|---|---|---|---|
| BOTH | 1,176 | **0.0570 -> 0.0009** | 0.0842 -> 0.0847 |
| MFD-ONLY | 4,663 | **0.0212 -> 0.0010** | (not in chain's pool) |

~160 generator hits evicted; the 94 on MFD-ONLY items are ones chain cannot follow. MFD's pool is 4x
chain's and carries far more history-matched candidates, and a listwise model trained mostly on
repeat-GT events fills the top 10 with them.

**4. Repeat headroom and a caution.** 1,189 pv repeats MFD still misses (+0.0143 if all recovered);
478 are not in MFD's pool, of which only 175 were even scored (root kept). But **admitting history
items to the pool is standard repeat-candidate generation — any system can do it**, and chain misses
MORE repeats than MFD (1,020 vs 478 not in pool). Applied fairly to both arms, history injection would
likely SHRINK the lead, because MFD's lead is precisely its better repeat coverage. Not pursued as an
MFD mechanism for that reason; recorded because it bounds how robust the current lead is.

**Consequences.** (a) The within-branch contrastive fine-tune (job 18512147) was cancelled before it
started: its target items convert at ~0 under the current reranker however sharp their heads are.
It is NOT refuted in combination with a reranker that stops evicting new items — then sharper heads
would convert — so it is queued behind NIP. (b) NIP (job 18512956) tests exactly the eviction.

**Honest budget, all optimistic:** current +0.0017 · stop new-item eviction <= +0.0019 · MFD-ONLY
repeats 0.54 -> 0.9 <= +0.0028 · total ~+0.0064. **Reaching +0.01 additionally requires MFD to
predict genuinely new items better than its generator does today** (MFD-ONLY new items at even 5%
would add ~+0.0016; chain's in-pool new-item rate of 8.5% would add ~+0.0048).

### 2026-09-25 — tau selection screen: closed (job 18511348, 4 min)
Re-selecting the stored top-5000 at nine tau values (exact at tau=1: reproduces base 0.2041, coverage
0.2968, MFD-ONLY 5,355). tau trades coverage for ordering; best pre-reranker gain +0.0018 at tau=0.7
with coverage falling 0.2968 -> 0.2892, and the reranker already repairs most of that ordering. With
round 1, **both re-ordering and re-selection of the frozen generator's pool are now closed.**

### 2026-09-25 — COMPONENT AUDIT: the failure is within-branch item discrimination, not the shared code
`component_audit.py` (mmap read of per-event teacher-forced GT factors, pv only):

| population | n | f* rank | t* rank | i* rank | logP(f*) | logP(t*\|f) | logP(i*\|f) |
|---|---|---|---|---|---|---|---|
| BOTH | 14,176 | 0 | 0 | 0 | -1.42 | -0.31 | -0.30 |
| MFD-ONLY | 5,325 | 1 | 6 | 3 | -2.57 | -3.87 | -3.19 |
| MFD-ONLY rescued | 362 | **22** | **0** | **0** | -5.45 | -0.52 | -0.50 |
| MFD-ONLY missed | 4,963 | **1** | 6 | 3 | -2.42 | -3.97 | -3.29 |

**This overturns the working hypothesis.** The gap MFD-ONLY vs BOTH is 84% in the modality heads (text
-3.56 nats, image -2.89) and only 16% in the shared code (-1.15). Inside MFD-ONLY, the reranker rescues
exactly the items whose t and i are rank 0 — even under a weak root (f* rank 22) — and misses the ones
with the right CATEGORY (f* rank 1!) but diffuse item heads. GT text code in the head's top-10: 61% of
MFD-ONLY vs 96% of BOTH. **The model knows the category and cannot pin the item within it.**

Why no reranker could fix this: every reranker feature is derived from these same heads, so the GT
genuinely looks worse than its in-branch competitors by every signal the reranker had.

What it points to, and why it is MFD-specific: a train/inference mismatch that only MFD has. Its t and i
heads are trained by independent cross-entropy over all 1,024 codes, but inference only ever ranks
CATALOG-VALID (t,i) pairs INSIDE the GT's branch. A within-branch contrastive loss over every catalog
tuple under f* trains the heads on exactly what inference asks. MFD gets this for free — one forward
pass yields both full heads, so all ~3,000 tuples in a branch score at no extra decoder cost — while
chain would need a sequential decode per negative. **If the missed items' t and i reached the rescued
profile, their summed score (~-3.4) would sit near BOTH's (-2.79): the ceiling is 4,963 events, and
converting 15% of them is +0.009.** -> section 6.6.

### 2026-09-25 — ROUND 1 RESULT: the reranking avenue is closed (job 18472175, 4h48m)
Four mechanisms screened at seed 0 on VAL (baseline 0.2138): **+BPF 0.2134, +BPF_gate 0.2139,
+BPF_gate_rw2 0.2133, +BPF_gate_hn64 0.2133.** The best is +0.0001 — inside seed noise (VAL std
~0.0005). The winner (+BPF_gate) confirmed on 3 seeds x 2 arms:

| | mean lead | t-interval | vs bar |
|---|---|---|---|
| baseline | +0.00196 | [+0.00082, +0.00311] | short by +0.00804 (667 events) |
| +BPF_gate | +0.00180 | [+0.00068, +0.00293] | short by +0.00820 (680 events) |

**Lead change -0.00016, CI [-0.00180, +0.00148] — GENERIC, no movement.** The deciding measurement:

| cell | MFD-ONLY promotions | median rank after | own recovery |
|---|---|---|---|
| baseline | 205 | 379 | 0.0648 |
| +BPF_gate | 202 | 350 | 0.0652 |
| +BPF_gate_hn64 | 235 | 458 | 0.0697 |

**Target was 790-890. None of the four moved it.** The four diagnostic cells got slightly *worse* with
BPF (own/10-49 0.1902 -> 0.1754, own/50+ 0.0243 -> 0.0205).

What this rules out, and it is a real finding rather than a failed tweak: the gate had complete freedom
to discount the generator's base score per candidate and produced nothing, so **the score form was not
the binding constraint**; hn64 made deep promotion the literal training objective and produced 235
instead of 205, so **the objective was not it either**; and the branch-posterior features carried no
usable signal, so **it is not an information gap**. Per the decision rule fixed in advance (6.3):
*the reranker cannot reorder this pool — the ranks themselves have to change, which means the
generator, not the reranker.*

**Round 2 (stacking +BPF/gate/hn) is CANCELLED, not run.** Stacking four independent nulls cannot
produce +0.008, and the per-mechanism numbers above say why. Spending 6 GPU-hours to confirm that
would be going through the motions.

### 2026-09-24 — BPF: branch-posterior position features (job 18472175, now complete — see above)
First attempt at the ordering lever, and the first experiment built on the factorisation itself
(queue item 2 of the old priority list). See section 6.1. Job: `sbatchfiles/bpf_rerank.sbatch`,
code `bpf_rerank.py` (AST-checked). NOTE: `bash -n` on the sbatch could not be run — the login
node's /tmp tmpfs filled up and killed shell output capture. The file is structurally identical to
`cf_rerank.sbatch` (which passed) and contains no escaped `\$` variables, but it is unverified.

---

## 6. Queue

### 6.0 Headroom audit — is +0.01 reachable by reranking at all? (STEP 0 of the BPF job)
`headroom.py`, no training, no dumps. Splits the 83,005 test events by pool membership into BOTH /
MFD-ONLY / CHAIN-ONLY / NEITHER and reports, per population: size, each arm's recovery, its
contribution to the current lead, the rank distribution of MFD's misses, and the recovery rate that
population would need to supply the whole +0.00827 alone. Asserts the split shape and the cross-arm
user_id join. Outputs `headroom_report.txt` / `headroom.json`.

Three possible readings, all actionable:
- MFD-ONLY misses >= +0.00827 at 100% recovery, needing a plausible fraction -> that is the target,
  and 6.1 is aimed at it.
- MFD-ONLY is too small even at 100% -> the bar cannot be met by reranking alone; it needs new MFD
  coverage that converts, or gains on BOTH events that chain can also chase. **Report that plainly
  rather than tuning toward the number.**
- Most misses sit at rank 500+ -> ordering features will not reach them; the pool composition is.

### 6.1 BPF — where does this candidate sit inside its own branch's posterior? (RUNNING)

**Mechanism, and why chain cannot replicate it.** MFD reads the complete 1024-way P(t|f,b) and
P(i|f,b) from one forward pass per (b,f) root, so *every* candidate's position inside its branch's
posterior is free at 808 decoder states. Chain would need to decode ~1024 continuations per root to
learn the same thing. The features are therefore structurally MFD-side even though both arms get them.

**Why it is not the calibration failure again.** Calibration *replaced* the absolute log-probs with
within-branch z-scores and collapsed pcov (0.78 -> 0.60) — the absolute level carries real information.
BPF *adds* relative position alongside the untouched absolute score, so the model can use both.

**14 features per candidate**, all from the existing dumps (no regeneration): margin of logP(t|f) to its
branch's best t (and the same for i); the candidate's share of its branch's pooled t-mass (and i);
the branch's log pooled mass, distinct-code count and top1-top2 gap for t and i; margin of logP(f) to
the best root of the same behavior; log branch size; the candidate's base-score margin inside its
branch; and inside the whole event. Z-scored on train candidates, clipped to +-10.

**Design.** Two phases so TEST never selects anything. PHASE 1 screens four variants at seed 0 on MFD,
chosen on VAL; PHASE 2 confirms the winner on 3 seeds x 2 arms against the reused seedvar baselines.
The four are each aimed at one candidate explanation for the measured blocker (206 deep promotions,
median rank 429 -> 378):

| variant | hypothesis it tests |
|---|---|
| `+BPF` | the reranker lacks the *information* to spot good deep candidates |
| `+BPF_gate` | it has the information but the *score form* forbids acting on it |
| `+BPF_gate_rw2` | ... and the deep rank bands are under-weighted in training |
| `+BPF_gate_hn64` | ... and the *objective* is wrong: an 8,000-way listwise softmax puts almost no gradient on a GT at rank ~400, so train GT-vs-top-64 instead — literally "beat the 10th-ranked item" |

Each variant isolates a different link in the chain, so a null result localises the cause rather than
just failing.

**Decision rule, fixed in advance.**
- ABSOLUTE: per-seed paired CI of (+BPF - baseline) within each arm, plus the across-seed mean gain.
- DIFFERENTIAL: mean lead with BPF vs mean lead without, with the across-seed t-interval; a feature
  that lifts both arms equally is reported as GENERIC.
- BAR: mean lead >= **+0.01 absolute** and the t-interval lower bound above it; the shortfall is
  reported in events, not hidden behind a percentage.
- SPLIT: the BPF effect is decomposed into its MFD-ONLY part (chain cannot follow) and its BOTH part
  (chain can chase it). A gain that lives on BOTH events does not advance this goal even if it is
  statistically real.
- Also required: coverage@pool unchanged, pv and conversions both reported, four diagnostic cells.

**Honest expectation.** BPF is aimed at the right population — MFD-only GTs are, by construction,
items chain's beam pruned at level 1, whose signature is a sharp t and i under a weak f-root, which is
exactly what t_margin / i_margin / t_share / i_share measure and what chain cannot compute at 808
states. But the largest feature effect ever measured here is +0.0060, and +0.00827 is needed. One
feature family is unlikely to be sufficient on its own; 6.0 will say how much of the gap it has to
find, and 6.2 is the follow-on if the direction is alive.

### 6.2 If the gate/hn variants move deep promotions but not far enough
Push the same lever harder: exact 1024-way branch statistics (true rank of t and i within the head, CDF
position, head entropy) instead of the pool-truncated approximations, which needs a re-dump; and a
two-stage reranker whose first stage prunes 8,000 -> ~200 so the listwise loss is no longer dominated
by the head of the list.

### 6.3 If deep promotion cannot be trained at all
Then the ranks themselves have to change, which means the generator, not the reranker. MFD's base
score is 0.2041 vs chain's 0.2120 — its pool has better coverage but materially worse ordering, and
the MFD-ONLY GTs sit at median rank 378 under the hand-set composition
`logP(b)+logP(f)+logP(t|f)+logP(i|f)`. Retraining MFD with a listwise/contrastive objective at the
composition level is the remaining structural move. It is expensive and outside everything the ledger
has tried, so it needs the gate result first: if the gate can already reorder the frozen pool, the
generator does not need retraining; if it cannot, no reranker will.

### 6.4 Fallback claim if the bar proves unreachable
Measure how many decoder states chain needs to reach MFD's coverage of 0.2968 (sweep chain beams up;
MFD stays at 808). If chain needs 10-100x, the defensible claim becomes efficiency at matched quality
rather than quality at matched compute. **This does not satisfy the +0.01 bar** — it is what to report
if the bar cannot be met, not a way of meeting it.

### 6.5 Not queued, and why
Pretrained ID embeddings (dense version of CF-C): expected generic, chain-favouring — run only to lift
absolute numbers, with the chain control. Deep-recovery beyond rw1: own-only/50+ has ~2,036 events, so
tripling recovery is worth ~+0.0025 — mechanism evidence, not aggregate gain.

---

## 7. Execution chain (autonomous)

Rounds are submitted back-to-back without waiting for a prompt. A background waiter watches the
running job and wakes this session when it leaves the queue; the next round is chosen from the result
and submitted immediately. `bpf_rerank.py --variants 'name:bpf,rw,hn,gate;...'` configures a round
without a code edit, and each round uses its own cache directory so nothing is silently reused.

| round | job | variants | status |
|---|---|---|---|
| 1 | 18472175 `bpf_rerank.sbatch` | +BPF / +BPF_gate / +BPF_gate_rw2 / +BPF_gate_hn64, 3 seeds | queued |
| 2 | `bpf_round2.sbatch` | +stack / +stack_rw2 / +gatehn_nobpf | **cancelled** — round 1 showed every component null |
| 3 | 18511348 `tau_select.sbatch` | 9 tau values, offline re-selection | done — closed (+0.0018 pre-reranker) |
| 4 | 18512147 `mfd_branch_ctr.sbatch` | within-branch contrastive fine-tune | cancelled before start; re-queued behind NIP |
| 5 | 18512956 `nip_rerank.sbatch` | P_k / **D (one slot per item)** x w_new in {1,3}, 3 seeds, both arms | queued (GPU quota) |
| 5b | 18513156 `nip_offline.sbatch` (CPU) | P_k from cached results | done — P_k net-negative for both arms |
| 6 | **18514124 `inject_rerank.sbatch`** | history-candidate injection, BOTH arms, 3 seeds | queued — validity check for (d) |

Round 2 stacks whatever round 1 separates and adds the ablation that matters: `+gatehn_nobpf` drops
the branch-posterior features while keeping the gate and the deep-promotion objective, so if it
matches `+stack` the features were never the point and the whole gain is the score form plus the loss.
It runs 5 seeds because the bar is on the across-seed t-interval, not the mean.

**Decision rule for what follows round 2**, fixed now so it is not chosen after seeing the numbers:
- MFD-ONLY promotions >= ~790 and the 5-seed t-interval clears +0.01 -> done; write it up.
- Promotions rise materially (say 206 -> 400+) but the interval falls short -> 6.2 (exact 1024-way
  branch statistics via a re-dump, plus a two-stage reranker so the listwise loss is not dominated by
  the head of the list).
- Promotions barely move -> 6.3: the reranker cannot reorder this pool, and the ranks themselves have
  to change, which means the generator. Report that the reranking avenue is closed before spending
  a training run on it.

---

## 8. Standing protocol

808 decoder states per item on both arms. Fit train / select VAL / report TEST, never select on TEST.
Capped setting (36,864 training events) is the fair comparison. Assert 83,005 test events with
pv 77,833 / fav 2,042 / cart 1,554 / buy 1,576 and halt on mismatch. Assert coverage@pool unchanged by
reranking. Read SID widths from the files (MFD 2048/1024/1024, chain 1024/1024/1024) — never hard-code.
Report per-behavior t3@10 and state whether pv and conversions both improve. Multi-seed for any claim
about the lead.
