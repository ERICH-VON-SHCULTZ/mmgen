"""Ceiling of a per-event REPEAT/NEW mixture, for both arms (CPU, cached results + dumps).

Identity: recall@10 = coverage x P(rank<10 | covered). On covered pv-NEW events chain's reranker converts
5.9%, MFD's 0.1% -- although MFD's own generator had 2.8% before reranking. MFD's reranker fills the top-10
with the user's history items (it carries more of them than chain: repeat coverage 0.966 vs 0.928), which
wins on repeat events and loses on new-item events. A fixed reserved-slot policy (P_k) paid the repeat cost
on EVERY event and was net-negative. The right object is a per-event mixture:
    P(GT = c) = P(repeat) P(c | repeat)   for history candidates
              + P(new)    P(c | new)      for non-history candidates
This script computes its ceiling with an ORACLE gate (knows whether the event is a repeat):
    repeat event -> today's reranked list (lower bound: a history-only ranker could only do better)
    new event    -> top-10 among NON-history candidates, ranked by the generator's base score
Both arms get the identical computation. If the ceiling clears the bar, a learned gate + a new-item ranker is
the build; if not, the new-item ranker must beat the base score, and this says by how much.
"""

import numpy as np
import torch

from nip_offline import per_event

C = "/scratch/yw8866/rec-tmall/seedvar/cache"
SEEDS = range(5)
out = []


def log(m):
    print(m, flush=True); out.append(m)


arms = {}
for arm, dump, kp in (("MFD", "/scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt", 2000),
                      ("CH", "/scratch/yw8866/rec-tmall/matched_rerank/chain808_test_dump.pt", None)):
    uid, nhr, gth, gb = per_event(dump, kp, 400, 1.0)
    o = np.argsort(uid)
    arms[arm] = {"uid": uid[o], "nhr": nhr[o], "gth": gth[o], "gb": gb[o]}
assert np.array_equal(arms["MFD"]["uid"], arms["CH"]["uid"])
N = len(arms["MFD"]["uid"]); gb = arms["MFD"]["gb"]; pv = gb == 0


def res(arm, sd):
    r = torch.load(f"{C}/{arm}__seed{sd}.pt", map_location="cpu", weights_only=False)
    o = np.argsort(r["uid"]); assert np.array_equal(r["uid"][o], arms[arm]["uid"])
    return r["has_gt"][o].astype(bool), r["hit_after"][o] > 0


log("=" * 100)
log("ORACLE-GATE CEILING: repeat events keep today's list; new events get a top-10 of NON-history candidates")
log("=" * 100)
hits = {}
for arm in ("MFD", "CH"):
    A = arms[arm]
    rep = A["gth"]                                   # GT is a history item, in this arm's own SID space
    for sd in SEEDS:
        cov, hit = res(arm, sd)
        newhit = cov & ~rep & (A["nhr"] >= 0) & (A["nhr"] < 10)
        hits[(arm, sd, "today")] = hit
        hits[(arm, sd, "oracle")] = np.where(rep, hit, newhit)
        hits[(arm, sd, "newonly")] = newhit
for arm in ("MFD", "CH"):
    A = arms[arm]; rep = A["gth"]
    cov = res(arm, 0)[0]
    for nm, m in (("pv new (own def)", pv & ~rep), ("all non-repeat", ~rep)):
        q = m & cov
        t = np.mean([hits[(arm, sd, "today")][q].mean() for sd in SEEDS])
        o = np.mean([hits[(arm, sd, "oracle")][q].mean() for sd in SEEDS])
        log(f"  {arm:<4}{nm:<18} covered {int(q.sum()):>6,}  P(r<10|c) today {t:.4f} -> non-history top-10 {o:.4f}"
            f"   (recall {t*q.sum()/m.sum():.4f} -> {o*q.sum()/m.sum():.4f})")
log("")
for kind in ("today", "oracle"):
    m = np.array([hits[("MFD", sd, kind)].mean() for sd in SEEDS]); c = np.array([hits[("CH", sd, kind)].mean() for sd in SEEDS])
    log(f"  {kind:<7} MFD {m.mean():.4f}  chain {c.mean():.4f}  lead {(m-c).mean():+.5f} (std {(m-c).std(ddof=1):.5f})")
mo = np.array([hits[("MFD", sd, "oracle")].mean() for sd in SEEDS]); ct = np.array([hits[("CH", sd, "today")].mean() for sd in SEEDS])
log(f"  oracle MFD vs TODAY's chain (upper bound if only MFD had the gate): {(mo-ct).mean():+.5f}")
log("\n  bar: lead >= +0.0100 with the SAME method applied to both arms (line 'oracle').")
open("/scratch/yw8866/rec-tmall/oracle_gate.txt", "w").write("\n".join(out) + "\n")
