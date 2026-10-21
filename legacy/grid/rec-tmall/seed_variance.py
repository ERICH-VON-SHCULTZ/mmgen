"""Seed variance of the current best setting: MFD (P3 wide pool, K_pool=2000) vs chain (808 states),
reranker R2_rw1, capped -- the SAME code path as the cf_rerank baseline cells (0.2161 vs 0.2139 at seed 0).

Everything the seed touches is re-drawn per seed: reranker init, the random code tables (seed+4242),
the capped training subsample (seed+99), the epoch shuffles. Pools, events, protocol are fixed.
Fit on TRAIN, select on VAL, report on TEST. Seed 0 is taken from the cf_rerank cache when present
(identical code path), so it is not paid for twice.

Report: per seed (both arms: t3@10 and the other metrics, VAL best, epochs; the lead with its paired
event-bootstrap CI); across seeds: mean, sample std, min/max of each arm and of the lead, the
t-interval of the mean lead, how many seeds have lead>0 / CI lower bound>0, and the seed std of the
lead compared with the within-seed bootstrap SE (which noise dominates).
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

from train_mfd_reranker import BEH
from coverage_expansion import Pool, canon, log
from cf_rerank import run_cell, paired_ci

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seed0_cache", default="", help="dir holding CF_{arm}__baseline__R2_rw1__capped.pt (seed 0, same code path)")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/seedvar")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "seed_variance_report.txt"), "w")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(args.spec)); seeds = [int(s) for s in args.seeds.split(",")]
    log("=" * 100 + f"\n=== SEED VARIANCE: MFD K_pool=2000 vs chain, R2_rw1 capped, seeds {seeds} ===\n" + "=" * 100, fh)

    res = {}
    for arm in ("MFD", "CH"):
        sp = spec[arm]
        log(f"\n{'#' * 100}\n# ARM {arm}: {os.path.basename(sp['te'])}  K_pool={sp.get('kpool')}\n{'#' * 100}", fh)
        pool = Pool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"))
        assert pool.te.N == 83005
        for sd in seeds:
            cp = os.path.join(args.cache, f"{arm}__seed{sd}.pt")
            if not os.path.exists(cp) and sd == 0 and args.seed0_cache:
                src = os.path.join(args.seed0_cache, f"CF_{arm}__baseline__R2_rw1__capped.pt")
                if os.path.exists(src):
                    torch.save(torch.load(src, map_location="cpu", weights_only=False), cp)
                    log(f"  [{arm}] seed 0 taken from the cf_rerank baseline cell ({src})", fh)
            if os.path.exists(cp):
                log(f"  [{arm}] seed {sd}: cached", fh)
            else:
                a = copy.copy(args); a.seed = sd
                log(f"\n  >>> [{arm}] seed {sd}", fh)
                torch.save(run_cell(pool, (), args.cap, dev, a, fh), cp)
            r = res[(arm, sd)] = canon(torch.load(cp, map_location="cpu", weights_only=False))
            log(f"  [{arm}] seed {sd}: t3@10 {r['before']['t3@10']:.4f} -> {r['after']['t3@10']:.4f}  VAL best {r['val']:.4f}  "
                f"{r['epochs_run']} ep  {r['train_secs']/60:.1f} min", fh)
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- report
    u0 = res[("MFD", seeds[0])]["uid"]
    for k in res:
        assert np.array_equal(res[k]["uid"], u0), f"join failed {k}"
    assert np.array_equal(res[("MFD", seeds[0])]["gt_b"], res[("CH", seeds[0])]["gt_b"])
    rng = np.random.default_rng(0)

    log("\n" + "=" * 100 + "\n=== PER SEED (TEST) ===\n" + "=" * 100, fh)
    log(f"  {'seed':>4} {'arm':<4}{'t3@10':>8}{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}{'P(r<10|c)':>10}{'t2@10':>8}{'VAL':>8}{'ep':>4}"
        + "".join(f"{BEH[b]:>9}" for b in range(4)), fh)
    for sd in seeds:
        for arm in ("MFD", "CH"):
            a = res[(arm, sd)]["after"]; r = res[(arm, sd)]
            log(f"  {sd:>4} {arm:<4}{a['t3@10']:>8.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}{a['t3n@5']:>8.4f}{a['t3n@10']:>8.4f}"
                f"{a['pcov']:>10.4f}{a['t2@10']:>8.4f}{r['val']:>8.4f}{r['epochs_run']:>4}"
                + "".join(f"{a['perbeh'][BEH[b]]:>9.4f}" for b in range(4)), fh)
    log(f"\n  lead MFD - chain per seed (paired event bootstrap, B={args.boot}):", fh)
    leads, se_within, los = [], [], []
    for sd in seeds:
        hm, hc = res[("MFD", sd)]["hit_after"], res[("CH", sd)]["hit_after"]
        lead = hm.mean() - hc.mean(); lo, hi, p = paired_ci(hm, hc, rng, args.boot)
        leads.append(lead); se_within.append((hi - lo) / 3.92); los.append(lo)
        log(f"    seed {sd}: {lead:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(MFD>chain)={p:.3f}  "
            f"{'MFD wins (CI>0)' if lo > 0 else 'chain wins (CI<0)' if hi < 0 else 'not distinguishable'}", fh)

    log("\n" + "=" * 100 + "\n=== ACROSS SEEDS ===\n" + "=" * 100, fh)
    n = len(seeds)
    def stats(x):
        x = np.asarray(x, dtype=np.float64)
        return x.mean(), (x.std(ddof=1) if n > 1 else float("nan")), x.min(), x.max()
    for arm in ("MFD", "CH"):
        for key in ("t3@10", "t3@5", "t3@20", "t3n@10", "pcov"):
            m, s, lo, hi = stats([res[(arm, sd)]["after"][key] for sd in seeds])
            log(f"  {arm:<4}{key:<8} mean {m:.4f}  std {s:.4f}  min {lo:.4f}  max {hi:.4f}", fh)
    m, s, lo, hi = stats(leads)
    tcrit = T975.get(n - 1, 1.96)
    half = tcrit * s / np.sqrt(n) if n > 1 else float("nan")
    log(f"\n  LEAD (MFD - chain, t3@10): mean {m:+.4f}  std across seeds {s:.4f}  min {lo:+.4f}  max {hi:+.4f}", fh)
    log(f"    95% t-interval of the mean lead over {n} seeds: [{m-half:+.4f}, {m+half:+.4f}]", fh)
    log(f"    seeds with lead > 0: {sum(l > 0 for l in leads)}/{n};  seeds with paired-CI lower bound > 0: {sum(l > 0 for l in los)}/{n}", fh)
    log(f"    noise budget: seed-to-seed std of the lead {s:.4f} vs within-seed event-bootstrap SE {np.mean(se_within):.4f} -> "
        + ("SEED noise dominates: a single-seed CI understates the uncertainty of the lead" if s > np.mean(se_within)
           else "event sampling dominates: single-seed CIs are a fair summary"), fh)
    x = np.array([res[("MFD", sd)]["after"]["t3@10"] for sd in seeds]); y = np.array([res[("CH", sd)]["after"]["t3@10"] for sd in seeds])
    if n > 2:
        log(f"    corr(MFD, chain) across seeds = {np.corrcoef(x, y)[0, 1]:+.3f} (arms use independent subsamples/inits; near 0 expected)", fh)
    json.dump({"seeds": seeds, "mfd": x.tolist(), "chain": y.tolist(), "lead": leads,
               "lead_mean": float(m), "lead_std": float(s), "t_interval": [float(m - half), float(m + half)]},
              open(os.path.join(args.out_dir, "seed_variance.json"), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
