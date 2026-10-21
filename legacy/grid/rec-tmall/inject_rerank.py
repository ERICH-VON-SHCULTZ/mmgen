"""Is MFD's lead robust to standard repeat-candidate injection? (validity check for criterion (d))

FINDINGS THAT MOTIVATE THIS (cached 5-seed results):
  * Almost every hit either system gets is a PREDICTABLE event: a page-view REPEAT (GT tuple already in
    the user's history) or a conversion. MFD: 12,951 pv-repeat hits vs 6 pv new-item hits.
  * MFD's lead is REPEAT COVERAGE: its pool contains 13,662 of the 14,140 pv repeats, chain's 13,120.
  * Adding the user's own history items to the candidate set is standard repeat-candidate generation --
    it costs no decoder states and any system can do it. If chain does it, it gains the repeats it misses
    (1,020 not in its pool vs MFD's 478), and MFD's lead may disappear. That would fail criterion (d),
    "explained by a mechanism chain cannot replicate", for the lead we already have.

THE TEST. Each arm's candidate set is augmented, per behavior branch, with the tuples of the user's own
history items (up to H per behavior), deduplicated against the existing pool. The injected candidates get
an UNINFORMATIVE generator score -- each component log-prob set to the event's minimum over its valid
pool -- identically for both arms, so neither arm gets a generator-side advantage from injection; the
reranker must rank them from its other signals (history match, cross-attention). Then the same reranker
(R2_rw1, capped, 3 seeds) is trained on each augmented arm and compared with the NON-injected baselines
(reused from seedvar/cache) under the fixed paired protocol.

Readings, fixed in advance:
  * chain+inject >= MFD (no injection): MFD's current lead is replicable by a standard heuristic.
  * MFD+inject - chain+inject vs today's +0.0017: does MFD keep a lead once both arms have the repeats?
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

from coverage_expansion import Pool, canon, log
from cf_rerank import paired_ci
import nip_rerank as NIP

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def inject(d, fh, name, wb=1.0):
    """Append each event's history tuples as candidates in EVERY behavior branch; dedupe; recompute GT."""
    N, nb, C0, H = d.N, d.nb, d.C, d.H
    hf, ht, hi = d.hist_f.long(), d.hist_t.long(), d.hist_i.long()
    hv = d.hist_b.long() >= 0                                           # (N,H)
    # new candidates: for behavior b, slot h -> (hf, ht, hi); layout [beh0 x H | beh1 x H | ...]
    nf = hf.repeat(1, nb); nt = ht.repeat(1, nb); ni = hi.repeat(1, nb)  # (N, nb*H)
    nbr = torch.arange(nb).repeat_interleave(H)                         # (nb*H,)
    nvalid = hv.repeat(1, nb).clone()
    # drop duplicates within the history (same tuple twice) -- keep the first occurrence per branch
    key = (nf * 4096 + nt) * 4096 + ni
    for b in range(nb):
        blk = slice(b * H, (b + 1) * H)
        k = key[:, blk]
        earlier = torch.tril(torch.ones(H, H, dtype=torch.bool), diagonal=-1)
        dup = ((k.unsqueeze(2) == k.unsqueeze(1)) & earlier.unsqueeze(0) & hv.unsqueeze(1)).any(-1)
        nvalid[:, blk] &= ~dup
    # drop history tuples already in that behavior's existing pool
    pf, pt, pi = d.f.long(), d.t.long(), d.i.long()
    already = torch.zeros_like(nvalid)
    for s0 in range(0, N, 2048):
        s1 = min(N, s0 + 2048)
        for b in range(nb):
            cols = (d.branch == b).nonzero().squeeze(1)
            pk = (pf[s0:s1, cols] * 4096 + pt[s0:s1, cols]) * 4096 + pi[s0:s1, cols]
            pv = d.valid[s0:s1, cols]
            blk = slice(b * H, (b + 1) * H)
            nk = key[s0:s1, blk]
            already[s0:s1, blk] = ((nk.unsqueeze(2) == pk.unsqueeze(1)) & pv.unsqueeze(1)).any(-1)
    nvalid &= ~already
    # uninformative generator score: each component = the event's minimum over its valid pool
    lp = d.lp.float()
    big = torch.finfo(torch.float32).max
    mins = torch.where(d.valid.unsqueeze(-1), lp, torch.full_like(lp, big)).min(1).values   # (N,4)
    mins = torch.where(mins >= big, torch.zeros_like(mins), mins)
    beh_lp = torch.stack([lp[:, (d.branch == b).nonzero()[0, 0], 0] for b in range(nb)], 1)  # logP(b) per event
    nlp = mins.unsqueeze(1).repeat(1, nb * H, 1)
    nlp[..., 0] = beh_lp[:, nbr]                                        # the behavior prior is known exactly
    nbase = wb * nlp[..., 0] + nlp[..., 1:].sum(-1)
    # append
    d.f = torch.cat([d.f, nf.to(d.f.dtype)], 1); d.t = torch.cat([d.t, nt.to(d.t.dtype)], 1)
    d.i = torch.cat([d.i, ni.to(d.i.dtype)], 1)
    d.lp = torch.cat([d.lp, nlp.to(d.lp.dtype)], 1); d.base = torch.cat([d.base, nbase.to(d.base.dtype)], 1)
    d.valid = torch.cat([d.valid, nvalid], 1); d.branch = torch.cat([d.branch, nbr])
    d.C = C0 + nb * H
    # recompute GT candidate / coverage over the augmented set
    br = d.branch.view(1, -1)
    sid = (d.f.long() == d.gt_f[:, None]) & (d.t.long() == d.gt_t[:, None]) & (d.i.long() == d.gt_i[:, None])
    gtc = sid & (br == d.gt_b[:, None]) & d.valid
    cov0 = float(d.has_gt.float().mean())
    d.has_gt = gtc.any(1)
    d.gt_idx = torch.where(d.has_gt, gtc.float().argmax(1), torch.full((N,), -1, dtype=torch.long))
    d.train_idx = torch.nonzero(d.has_gt, as_tuple=False).squeeze(1)
    log(f"  [{name}] injected {int(nvalid.sum()):,} history candidates ({float(nvalid.sum())/N:.1f}/event, "
        f"{int(already.sum()):,} already in pool); C {C0:,} -> {d.C:,}; coverage {cov0:.4f} -> "
        f"{float(d.has_gt.float().mean()):.4f}  (behavior-weight used for the injected base: {wb:.3f})", fh)


def ms(x):
    x = np.asarray(x, float); n = len(x); s = x.std(ddof=1) if n > 1 else float("nan")
    return x.mean(), s, (T975.get(n - 1, 1.96) * s / np.sqrt(n)) if n > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--base_cache", default="/scratch/yw8866/rec-tmall/seedvar/cache")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bar", type=float, default=0.01)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/inject_exp")
    args = ap.parse_args(); args.specialist = False
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "inject_report.txt"), "a")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(args.spec)); seeds = [int(s) for s in args.seeds.split(",")]
    log("=" * 100 + f"\n=== INJECTION: is MFD's lead robust to standard repeat-candidate injection?  ({time.strftime('%Y-%m-%d %H:%M')}) ===\n" + "=" * 100, fh)

    res = {}
    for arm in ("MFD", "CH"):
        sp = spec[arm]
        log(f"\n{'#' * 100}\n# ARM {arm}\n{'#' * 100}", fh)
        pool = Pool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"))
        assert pool.te.N == 83005
        for nm, d in (("tr", pool.tr), ("va", pool.va), ("te", pool.te)):
            inject(d, fh, f"{arm}/{nm}", wb=args.wb)
        for sd in seeds:
            cp = os.path.join(args.cache, f"{arm}__inject__seed{sd}.pt")
            if not os.path.exists(cp):
                a = copy.copy(args); a.seed = sd
                log(f"\n  >>> [{arm}] +inject seed {sd}", fh)
                torch.save(NIP.run_cell(pool, 1.0, args.cap, dev, a, fh), cp)
            res[(arm, "inj", sd)] = NIP.canon2(torch.load(cp, map_location="cpu", weights_only=False))
            r = res[(arm, "inj", sd)]
            log(f"  [{arm}] +inject seed {sd}: t3@10 P0 {r['test_k']['P0']:.4f}  cov {r['after']['cov']:.4f}", fh)
            b = torch.load(os.path.join(args.base_cache, f"{arm}__seed{sd}.pt"), map_location="cpu", weights_only=False)
            res[(arm, "base", sd)] = canon(b)
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()

    u = res[("MFD", "base", seeds[0])]["uid"]
    for k in res:
        assert np.array_equal(res[k]["uid"], u), f"join failed {k}"
    rng = np.random.default_rng(0)
    H = lambda arm, kind, sd: (res[(arm, kind, sd)]["hit_k"]["P0"] if kind == "inj" else res[(arm, kind, sd)]["hit_after"])
    log("\n" + "=" * 100 + "\n=== RESULTS (t3@10, unmodified list P0, mean over seeds) ===\n" + "=" * 100, fh)
    for arm in ("MFD", "CH"):
        for kind in ("base", "inj"):
            m, s, _ = ms([H(arm, kind, sd).mean() for sd in seeds])
            log(f"  {arm:<4}{'no injection' if kind == 'base' else '+ injection':<14} {m:.4f} (std {s:.4f})", fh)
    comps = [("MFD vs chain, neither injected (today)", ("MFD", "base"), ("CH", "base")),
             ("MFD (no injection) vs chain + injection", ("MFD", "base"), ("CH", "inj")),
             ("MFD + injection vs chain + injection", ("MFD", "inj"), ("CH", "inj"))]
    log("", fh)
    for lab, (a1, k1), (a2, k2) in comps:
        ls, cis = [], []
        for sd in seeds:
            hm, hc = H(a1, k1, sd), H(a2, k2, sd)
            lo, hi, p = paired_ci(hm, hc, rng, args.boot); ls.append(hm.mean() - hc.mean()); cis.append((lo, hi))
        m, s, h = ms(ls)
        log(f"  {lab:<44} lead {m:+.5f} (t-interval [{m-h:+.5f}, {m+h:+.5f}])  per-seed "
            + " ".join(f"[{a:+.4f},{b:+.4f}]" for a, b in cis), fh)
    log("\n  reading (fixed in advance): if chain + injection matches or beats MFD without it, MFD's current lead", fh)
    log("  is replicable by a standard heuristic and does not satisfy criterion (d). The third line is the lead", fh)
    log("  that survives once BOTH arms have the repeats.", fh)
    fh.close()


if __name__ == "__main__":
    main()
