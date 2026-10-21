"""K_pool sweep on the wide P3 pool: does enlarging K_pool convert MFD's extra coverage into recall?

One dump per split stores the top-5000 per behavior (C = 20,000 candidates/event). Each K_pool in the
sweep truncates the stored, score-sorted pool slots to K per behavior (so K=500 IS the old pool, and the
K=500 pool is a subset of every wider one), trains the fixed reranker R2_rw1 in the capped setting, and
evaluates on TEST. The chain arm is reused from the reranker cache (trained at 808 states, never retrained).

Deciding metric, relative to K_pool=500 (same events, same reranker, same protocol):
  NET t3@10 = newly-covered GTs that reach top-10  +  previously-covered GTs regained
              -  previously-covered GTs LOST from the top-10 (more candidates = more competitors),
  with a PAIRED bootstrap CI over events. Gross recovery is reported but the verdict is on the NET.
Verdict fixed in advance: NET clearly positive and significant -> K_pool is a live lever; NET ~0 or
negative -> the coverage axis closes. All K_pool values are reported, not the best one.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

from train_mfd_reranker import BEH
from coverage_expansion import Pool, canon, run_one, log


def paired_ci(a, b, rng, B):
    """CI of mean(a) - mean(b) resampling the SAME events for both (a, b aligned per event)."""
    N = a.shape[0]; d = np.empty(B)
    for k in range(B):
        ii = rng.integers(0, N, N); d[k] = a[ii].mean() - b[ii].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr", required=True); ap.add_argument("--va", required=True); ap.add_argument("--te", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--ks", default="500,2000,5000")
    ap.add_argument("--cache", required=True, help="reranker result cache; CH__R2_rw1__capped.pt must exist here")
    ap.add_argument("--chain_key", default="CH__R2_rw1__capped.pt")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", default="", help='in-model ladder for the report, "K:cov,..."')
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/coverage_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "kpool_sweep_report.txt"), "w")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(x) for x in args.ks.split(",")]
    assert ks[0] == 500, "the reference K_pool must be first (500)"
    ladder = {int(a): float(b) for a, b in (x.split(":") for x in args.ladder.split(",") if x)}

    log("=" * 100 + "\n=== K_pool SWEEP on the wide P3 pool (K_f=200, K_t=K_i=1024, 808 decoder states), R2_rw1, capped ===\n" + "=" * 100, fh)
    chain_p = os.path.join(args.cache, args.chain_key)
    if not os.path.exists(chain_p):
        raise SystemExit(f"** chain result {chain_p} missing -- it must be reused, not retrained **")
    ch = canon(torch.load(chain_p, map_location="cpu", weights_only=False))
    log(f"  chain (reused, 808 states): t3@10 {ch['before']['t3@10']:.4f} -> {ch['after']['t3@10']:.4f}  cov={ch['after']['cov']:.4f}", fh)

    # ---- train / load one reranker per K_pool ----
    res = {}
    for K in ks:
        cp = os.path.join(args.cache, f"P3K{K}__R2_rw1__capped.pt")
        if os.path.exists(cp):
            log(f"\n  [cached] K_pool={K}: {cp}", fh)
        else:
            log(f"\n  >>> K_pool={K}: loading the stored pool truncated to the top-{K} per behavior", fh)
            pool = Pool(f"P3K{K}", args.tr, args.va, args.te, args.sid, args.wb, dev, fh, kpool=K)
            assert pool.te.Kp == K and pool.te.C == pool.nb * K, "truncation did not take"
            cov = float(pool.te.has_gt.float().mean())
            if K in ladder:
                log(f"  TEST coverage@pool={cov:.4f} vs in-model ladder {ladder[K]:.4f} (diff {cov-ladder[K]:+.4f})", fh)
            log(f"  >>> RUN P3 K_pool={K} / R2_rw1 / capped  (C={pool.te.C:,} candidates/event, "
                f"covered train={int(pool.tr.has_gt.sum()):,}, cap={args.cap:,})", fh)
            r = run_one(pool, "rw1", args.cap, dev, args, fh)
            torch.save(r, cp)
            del pool
            if dev == "cuda":
                torch.cuda.empty_cache()
        res[K] = canon(torch.load(cp, map_location="cpu", weights_only=False))
        log(f"  K_pool={K}: t3@10 {res[K]['before']['t3@10']:.4f} -> {res[K]['after']['t3@10']:.4f}  cov={res[K]['after']['cov']:.4f}  "
            f"train {res[K].get('train_secs', float('nan'))/60:.1f} min / {res[K].get('epochs_run', '?')} ep / peak GPU {res[K].get('peak_gpu_gb', float('nan')):.1f} GB", fh)

    N = res[500]["uid"].shape[0]
    for K in ks:
        assert np.array_equal(res[K]["uid"], res[500]["uid"]), f"canonical join failed for K={K}"
    assert np.array_equal(ch["uid"], res[500]["uid"]), "chain join on user_id failed"
    assert np.array_equal(ch["gt_b"], res[500]["gt_b"]), "chain gt_b disagrees after join"
    rng = np.random.default_rng(args.seed)

    # ---- full metrics ----
    log("\n" + "=" * 100 + "\n=== FULL METRICS (TEST, capped, R2_rw1) ===\n" + "=" * 100, fh)
    log(f"  {'pool':<11}{'C/ev':>7}{'cov':>8}{'t3@10 bef':>11}{'aft':>8}{'gain':>9}{'P(r<10|c)':>10}"
        f"{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}{'t2@10':>8}{'t2n@10':>8}{'params':>9}{'n_tr':>8}{'min':>7}{'ep':>4}{'GPU GB':>8}", fh)
    rows = [(f"P3 K={K}", res[K]) for K in ks] + [("chain", ch)]
    for lab, r in rows:
        b, a = r["before"], r["after"]
        log(f"  {lab:<11}{r.get('C', 2000):>7,}{a['cov']:>8.4f}{b['t3@10']:>11.4f}{a['t3@10']:>8.4f}{a['t3@10']-b['t3@10']:>+9.4f}"
            f"{a['pcov']:>10.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}{a['t3n@5']:>8.4f}{a['t3n@10']:>8.4f}{a['t2@10']:>8.4f}{a['t2n@10']:>8.4f}"
            f"{r['params']:>9,}{r['n_train']:>8,}{r.get('train_secs', float('nan'))/60:>7.1f}{str(r.get('epochs_run', '?')):>4}"
            f"{r.get('peak_gpu_gb', float('nan')):>8.1f}", fh)
    log("\n  per-behavior t3@10 (before -> after):", fh)
    for lab, r in rows:
        bb, aa = r["before"]["perbeh"], r["after"]["perbeh"]
        log(f"    {lab:<11}" + "  ".join(f"{k}:{bb[k]:.4f}->{aa[k]:.4f}" for k in aa), fh)

    # ---- THE DECIDING METRIC: each K vs K=500 ----
    log("\n" + "=" * 100 + "\n=== DECIDING METRIC: each K_pool vs K_pool=500 (same events, same reranker protocol) ===\n" + "=" * 100, fh)
    ref = res[500]; c0 = ref["has_gt"].astype(bool); h0 = ref["hit_after"] > 0
    verdicts = []
    for K in ks[1:]:
        r = res[K]; cK = r["has_gt"].astype(bool); hK = r["hit_after"] > 0
        new = cK & ~c0; dropped = c0 & ~cK; n_new = int(new.sum())
        log(f"\n  [K_pool={K}] coverage {c0.mean():.4f} -> {cK.mean():.4f}   newly covered GTs = {n_new:,}   "
            f"(K=500-covered but absent at K={K}: {int(dropped.sum()):,} -- must be 0, the pools are nested)", fh)
        assert int(dropped.sum()) == 0, "nested-pool invariant violated"
        rk = r["r3_after"][new]
        if n_new:
            fr = {k: float((rk < k).mean()) for k in (10, 20, 50)}
            hit_new = (rk < 10).astype(np.float64)
            bs = np.array([hit_new[rng.integers(0, n_new, n_new)].mean() for _ in range(args.boot)])
            log(f"    newly-covered reaching top-10 = {fr[10]:.4f} ({int(hit_new.sum()):,}/{n_new:,})  "
                f"95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]   top-20 = {fr[20]:.4f}   top-50 = {fr[50]:.4f}", fh)
            q = np.percentile(rk, [10, 25, 50, 75, 90])
            log(f"    rank distribution of newly-covered GTs after reranking (10/25/50/75/90 pct): {[int(x) for x in q]}", fh)
            z = r["ent_z"][new]
            log(f"    by GT-branch entropy z (flat = high):", fh)
            for lo, hi, nm in ((-1e9, -1., "<-1"), (-1., 0., "-1..0"), (0., 1., "0..1"), (1., 1e9, ">1")):
                mm = (z >= lo) & (z < hi)
                if mm.sum():
                    log(f"      z {nm:<6} n={int(mm.sum()):>7,}  top10={float((rk[mm] < 10).mean()):.4f}", fh)
            gb = r["gt_b"][new]
            log(f"    per behavior top-10 among newly covered: " + "  ".join(
                f"{BEH[b]}={float((rk[gb==b] < 10).mean()):.4f}(n={int((gb==b).sum()):,})" for b in range(4) if (gb == b).sum()), fh)
        new_gain = int((new & hK).sum()); prev_lost = int((c0 & h0 & ~hK).sum()); prev_regained = int((c0 & ~h0 & hK).sum())
        net = r["after"]["t3@10"] - ref["after"]["t3@10"]
        log(f"    previously-covered GTs LOST from the top-10 = {prev_lost:,}   regained = {prev_regained:,}   "
            f"(previous widening: lost 203 / regained 109 / net -94)", fh)
        log(f"    NET t3@10 (K={K} - K=500) = {net:+.4f}  =  newly-covered gain {new_gain:,}/{N:,} ({new_gain/N:+.4f})"
            f"  +  regained {prev_regained:,} ({prev_regained/N:+.4f})  -  lost {prev_lost:,} ({-prev_lost/N:+.4f})", fh)
        chk = (new_gain + prev_regained - prev_lost) / N
        log(f"    decomposition check: {chk:+.4f} vs measured {net:+.4f} -> {'OK' if abs(chk-net) < 1e-6 else 'MISMATCH'}", fh)
        lo, hi, p = paired_ci(hK.astype(np.float64), h0.astype(np.float64), rng, args.boot)
        log(f"    NET 95% CI (paired over events, B={args.boot}) = [{lo:+.4f}, {hi:+.4f}]  P(net>0)={p:.4f}", fh)
        verdicts.append((K, net, lo, hi, n_new, new_gain, prev_lost))

    # ---- significance vs chain, fixed paired protocol ----
    log("\n" + "=" * 100 + "\n=== SIGNIFICANCE vs chain (R2_rw1 capped both arms, joined on user_id, PAIRED bootstrap, B=2000) ===\n" + "=" * 100, fh)
    log("  reference at K_pool=500: +0.0012, 95% CI [+0.0004, +0.0020]", fh)
    hc = ch["hit_after"]
    for K in ks:
        hm = res[K]["hit_after"]
        lo, hi, p = paired_ci(hm, hc, rng, args.boot)
        log(f"  P3 K_pool={K:<5} - chain: {hm.mean()-hc.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(MFD>chain)={p:.4f}  "
            f"{'DISTINGUISHABLE' if (lo > 0 or hi < 0) else 'not distinguishable'}", fh)

    # ---- verdict, fixed in advance ----
    log("\n" + "=" * 100 + "\n=== VERDICT (fixed in advance; judged on the NET, all K_pool values reported) ===\n" + "=" * 100, fh)
    for K, net, lo, hi, n_new, new_gain, prev_lost in verdicts:
        if lo > 0 and net >= 0.002:
            v = "LIVE LEVER -- coverage converts: net gain clearly positive and significant."
        elif hi < 0:
            v = "CLOSES -- net change significantly NEGATIVE: the extra candidates cost more top-10 hits than they add."
        else:
            v = "CLOSES -- net ~0 / not significant: the extra coverage does not convert."
        log(f"  K_pool={K}: newly covered {n_new:,}, of which top-10 {new_gain:,}; previously-covered lost {prev_lost:,}; "
            f"NET {net:+.4f} CI [{lo:+.4f}, {hi:+.4f}] -> {v}", fh)
    json.dump({str(K): {"net": float(n), "lo": float(lo), "hi": float(hi), "new": int(nn), "new_gain": int(ng), "lost": int(pl)}
               for K, n, lo, hi, nn, ng, pl in verdicts}, open(os.path.join(args.out_dir, "kpool_sweep_verdict.json"), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
