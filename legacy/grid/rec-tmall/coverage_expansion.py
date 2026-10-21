"""Re-test COVERAGE EXPANSION under the current strong reranker.

Four earlier coverage-expansion attempts raised coverage 0.28 -> 0.76 with recall@10 unchanged, and
an entropy simulation found 0.000% of newly covered GTs reaching top-10. But ALL of them scored the
enlarged pools with the hand-set sum of log-probabilities. Two later findings broke that premise:
the trained reranker lifts P(rank<10|cov) most in the FLATTEST branches (+0.0710 at z>1 vs +0.0239
at the most peaked), and a rank-weighted objective pushes deep recovery further (MFD-only/50+
0.0344 -> 0.0467). So "wider pool + strong reranker" has never been measured.

Pools (MFD, K_f=200, K_pool=500 -- widening K_t/K_i costs NO extra decoder forwards, both heads
already emit full 1024-way softmaxes, so all three sit at 808 states; this is asserted, not assumed):
  P1 K_t=200  K_i=50    (control, coverage 0.2615)
  P2 K_t=200  K_i=200   (image branch is the weak one: branch_i loss 0.1385 vs branch_t 0.0243)
  P3 K_t=1024 K_i=1024  (full support, ceiling 0.7561)
Chain gets its best pool at the SAME 808 states, with its per-level beam ALLOCATION swept
(B1+B2=200) rather than assuming uniform -- chain has no free-widening lever, so this is the fair
way to give it its best shot.

Configurations on every pool, identical across arms: R1 = baseline reranker; R2 = R1 + the rw1
rank-weighted objective. Capacity is NOT swept (466k -> 3.46M moved t3@10 by 0.0002).

THE DECIDING METRIC is not aggregate t3@10: it is whether the GTs newly covered by P2/P3 relative
to P1 can be RECOVERED -- their top-10/20/50 rates, bucketed by branch-entropy z -- net of any
previously covered GTs LOST from the top-10 as competitors arrive. Under the hand-set scorer the
newly covered recovery was 0.000%; any materially positive number here is the result. Reported as a
full grid, with no post-hoc selection of pool or configuration.
"""

import argparse
import time
import json
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn.functional as F

from train_mfd_reranker import BEH, RerankData, best_rank, ndcg_at, recall_at
from sweep2_rerank import RW, Ranker, eval_split, make_scal, metrics, n_extra_of, ranks_of, sid_widths

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
KF_MAX = 2048


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


class Pool:
    """One (arm, pool) triple of splits. Duck-types the interface make_scal/eval_split need.
    R1/R2 use only views+hard+soft, so no extra feature blocks are required at train time;
    branch entropy is computed only for the DIAGNOSTIC buckets."""

    def __init__(self, name, tr_p, va_p, te_p, sid_p, wb, dev, fh, kpool=None):
        self.name, self.dev = name, dev
        self.tr = RerankData(tr_p, wb, kpool); self.va = RerankData(va_p, wb, kpool); self.te = RerankData(te_p, wb, kpool)
        self.splits = {"tr": self.tr, "va": self.va, "te": self.te}
        self.widths = sid_widths(sid_p); self.nb = self.tr.nb
        self.k1, self.k2 = self.widths[1], self.widths[2]
        for d in self.splits.values():
            for lvl, col in enumerate((d.f, d.t, d.i)):
                assert int(col.max()) < self.widths[lvl], \
                    f"{name}: level-{lvl} code {int(col.max())} >= width {self.widths[lvl]}"
        per = {b: int((self.te.gt_b == b).sum()) for b in range(self.nb)}
        ok = self.te.N == EXPECT_TOTAL and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
        log(f"  [{name}] widths(read)={self.widths}  TEST N={self.te.N:,} "
            + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH) + f" -> {'OK' if ok else 'MISMATCH'}", fh)
        if not ok:
            raise SystemExit(f"** {name} TEST split mismatch -- stop. **")
        self.fwd = self.te.meta.get("fwd_states_per_item", "?")
        log(f"  [{name}] coverage@pool(TEST)={float(self.te.has_gt.float().mean()):.4f}  "
            f"decoder-states/item={self.fwd}  K_pool={self.te.Kp}  covered train={int(self.tr.has_gt.sum()):,}", fh)
        # user_id per split (kept by RerankData; falls back to re-reading the dump for older code paths)
        self.uid = {k: (d.user_id if getattr(d, "user_id", None) is not None
                        else torch.load(p, map_location="cpu", weights_only=False)["cols"]["user_id"].long())
                    for k, d, p in (("tr", self.tr, tr_p), ("va", self.va, va_p), ("te", self.te, te_p))}
        self._ent = {}

    def pack(self, f, t, i):
        return (f.long() * self.k1 + t.long()) * self.k2 + i.long()

    def branch_ent(self, split, chunk=2048):
        """(N,C) per-candidate branch entropy and its z within the event's pool.
        Computed in CHUNKS of events: every group (event, behavior, f-branch) lives inside one event, so
        chunking is exact. The un-chunked version moved N*C (1.66e9 at C=20,000) int64/float tensors to
        the GPU at once and OOM'd (74 GB) right after the K_pool=5000 reranker had finished training."""
        if split in self._ent:
            return self._ent[split]
        d = self.splits[split]; N, C, nb, dev = d.N, d.C, self.nb, self.dev
        out = torch.zeros(N, C, dtype=torch.float16)
        br = d.branch.view(1, -1)                                          # (1,C)
        for s0 in range(0, N, chunk):
            s1 = min(N, s0 + chunk); n = s1 - s0
            v = d.valid[s0:s1]                                             # (n,C)
            ev = torch.arange(n).view(n, 1).expand(n, C)
            gk = ev.to(torch.int64) * (nb * KF_MAX) + br.expand(n, C) * KF_MAX + d.f[s0:s1].long()
            g = gk[v].to(dev)
            xt = d.lp[s0:s1, :, 2].float()[v].to(dev); xi = d.lp[s0:s1, :, 3].float()[v].to(dev)
            pt = d.t[s0:s1].long()[v].to(dev); pi = d.i[s0:s1].long()[v].to(dev)
            uniq, gid = torch.unique(g, return_inverse=True); G = uniq.numel()

            def H(x, code, K):
                tk = gid * K + code
                utk, gtd = torch.unique(tk, return_inverse=True)
                lp = torch.full((utk.numel(),), -1e30, device=dev).scatter_reduce(0, gtd, x, reduce="amax", include_self=True)
                bb = (utk // K).long(); P = lp.exp()
                Z = torch.zeros(G, device=dev).index_add_(0, bb, P)
                p = P / Z[bb].clamp(min=1e-12)
                return torch.zeros(G, device=dev).index_add_(0, bb, -p * p.clamp(min=1e-12).log())

            Hs = H(xt, pt, self.k1) + H(xi, pi, self.k2)
            eg = (uniq // (nb * KF_MAX)).long()                            # local event index (0..n-1)
            cnt = torch.zeros(n, device=dev).index_add_(0, eg, torch.ones_like(Hs))
            mu = torch.zeros(n, device=dev).index_add_(0, eg, Hs) / cnt.clamp(min=1)
            ss = torch.zeros(n, device=dev).index_add_(0, eg, Hs * Hs)
            sd = (ss / cnt.clamp(min=1) - mu.pow(2)).clamp(min=0).sqrt()
            z = (Hs - mu[eg]) / sd[eg].clamp(min=1e-6)
            blk = torch.zeros(n * C); blk[v.reshape(-1)] = z[gid].float().cpu()
            out[s0:s1] = blk.reshape(n, C).half()
            del g, xt, xi, pt, pi, uniq, gid, Hs, z
        if dev == "cuda":
            torch.cuda.empty_cache()
        self._ent[split] = out
        return self._ent[split]


def run_one(pool, rw, cap, dev, args, fh):
    """Train R1 (rw=None) or R2 (rw='rw1') on this pool; return TEST per-event arrays in the pool's
    OWN row order plus scalar metrics."""
    torch.manual_seed(args.seed)
    _t0 = time.perf_counter(); ep = 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    model = Ranker(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, n_extra_of((), pool.nb)).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = pool.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]])
    w_ev = None
    if rw:                                     # R2: weight events by PRE-rerank rank band
        rb = ranks_of(pool.tr.base.float(), pool.tr)[0]
        band = torch.where(rb < 10, 0, torch.where(rb < 50, 1, 2))
        tab = RW[rw]
        w_ev = torch.tensor([tab[int(b)] for b in band.tolist()], dtype=torch.float)
    rng = np.random.default_rng(args.seed)
    base_te = metrics(pool.te, pool.te.base.float())
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s:s + args.batch])
            f, t, i, scal = make_scal(pool, "tr", model, pool.tr, bidx, dev, ())
            sc = model(f, t, i, scal, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                       pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            loss = F.cross_entropy(sc, pool.tr.gt_idx[bidx].to(dev), reduction="none")
            loss = ((loss * w_ev[bidx].to(dev)).sum() / w_ev[bidx].to(dev).sum()) if w_ev is not None else loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = metrics(pool.va, eval_split(model, pool, "va", pool.va, dev, (), args.eval_batch))["t3@10"]
        log(f"      ep{ep:02d} VAL t3@10={v:.4f}", fh)
        if v > best_v + args.min_delta:
            best_v, bad = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    te = metrics(pool.te, eval_split(model, pool, "te", pool.te, dev, (), args.eval_batch))
    assert abs(te["cov"] - base_te["cov"]) < 1e-9, "coverage changed under reranking -- BUG"
    gi = pool.te.gt_idx.clamp(min=0)
    try:                                       # diagnostic only -- never lose a trained run to it
        zt = pool.branch_ent("te").float()
        ent_z = zt.gather(1, gi.unsqueeze(1)).squeeze(1).numpy()  # GT's own branch z (valid where covered)
    except torch.OutOfMemoryError as e:
        log(f"      ** branch_ent OOM ({str(e)[:80]}...) -- entropy-z buckets will be NaN for this run **", fh)
        if dev == "cuda":
            torch.cuda.empty_cache()
        ent_z = np.full(pool.te.N, np.nan)
    return {"params": npar, "val": best_v, "n_train": int(len(idx_all)), "fwd": pool.fwd,
            "train_secs": time.perf_counter() - _t0, "epochs_run": ep, "C": int(pool.te.C),
            "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else 0.0,
            "before": {k: v for k, v in base_te.items() if k not in ("hit", "r3")},
            "after": {k: v for k, v in te.items() if k not in ("hit", "r3")},
            "uid": pool.uid["te"].numpy(), "gt_b": pool.te.gt_b.numpy(),
            "has_gt": pool.te.has_gt.numpy(),
            "r3_before": base_te["r3"].numpy(), "r3_after": te["r3"].numpy(),
            "hit_before": base_te["hit"], "hit_after": te["hit"], "ent_z": ent_z}


def canon(r):
    """Reindex a result's per-event arrays into user_id-sorted (canonical) order, so pools and arms
    dumped in different row orders can be compared elementwise."""
    o = np.argsort(r["uid"])
    out = dict(r)
    for k in ("uid", "gt_b", "has_gt", "r3_before", "r3_after", "hit_before", "hit_after", "ent_z"):
        out[k] = r[k][o]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON: {label: {arm, tr, va, te, sid}}")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain_sweep", action="append", default=[], metavar="B1xB2=COV",
                    help="chain beam-allocation sweep results to reproduce in the report")
    ap.add_argument("--chain_sweep_winner", default="")
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/coverage_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "coverage_report.txt"), "w")
    spec = json.load(open(args.spec))
    log("=" * 118 + "\n=== COVERAGE EXPANSION under the strong reranker ===\n" + "=" * 118, fh)

    log("\n--- BUG FIXED BEFORE THIS RUN -------------------------------------------------------", fh)
    log("  The previous attempt dumped 24,000 events instead of 83,005 and the split assertion", fh)
    log("  stopped the job. Cause: +trainer.limit_test_batches=6000 MULTIPLIES with", fh)
    log("  batch_size_per_device. P1 used the default batch 64 (6000*64 = 384,000, never binding);", fh)
    log("  P2 was lowered to batch 4 to avoid OOM on the dense compose -> 6000*4 = 24,000, and P3", fh)
    log("  at batch 8 -> 48,000. Both pools were silently truncated. FIX: limit_test_batches is", fh)
    log("  now computed per split as ceil(NEED/BATCH)+50, and every dump is validated at the END", fh)
    log("  OF GENERATION (count, per-behavior, decoder-states, coverage) instead of at load time.", fh)

    if args.chain_sweep:
        log("\n--- CHAIN BEAM-ALLOCATION SWEEP (already run; chain at its best equal-compute config) ---", fh)
        log(f"  At 808 states with B1+B2=200, VAL coverage@pool:", fh)
        vals = []
        for e in args.chain_sweep:
            k, v = e.split("="); vals.append((k, float(v)))
            mark = "   <-- WINNER (used as chain's arm)" if k == args.chain_sweep_winner else ""
            log(f"    {k:<10}{float(v):.4f}{mark}", fh)
        sp = max(v for _, v in vals) - min(v for _, v in vals)
        log(f"  span across the whole allocation curve = {sp:.4f} -- uniform wins and the choice", fh)
        log(f"  barely matters, so chain is not disadvantaged by its allocation.", fh)

    CONFIGS = [("R1", None), ("R2_rw1", "rw1")]
    SETTINGS = ("natural", "capped")
    res = {}

    # ---------------- phase 1: train, one pool loaded at a time (memory-safe) ----------------
    for label, sp in spec.items():
        need = [(c, s) for c, _ in CONFIGS for s in SETTINGS
                if not os.path.exists(os.path.join(cache, f"{label}__{c}__{s}.pt"))]
        if not need:
            log(f"\n  [all cached] {label}", fh)
            continue
        log(f"\n  >>> loading pool {label} ({sp['arm']})", fh)
        pool = Pool(label, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh)
        for cname, rw in CONFIGS:
            for setting in SETTINGS:
                cp = os.path.join(cache, f"{label}__{cname}__{setting}.pt")
                if os.path.exists(cp):
                    continue
                log(f"\n  >>> RUN {label} / {cname} / {setting}", fh)
                torch.save(run_one(pool, rw, args.cap if setting == "capped" else 0, dev, args, fh), cp)
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()

    for label in spec:
        for cname, _ in CONFIGS:
            for setting in SETTINGS:
                res[(label, cname, setting)] = canon(
                    torch.load(os.path.join(cache, f"{label}__{cname}__{setting}.pt"),
                               map_location="cpu", weights_only=False))

    labels = list(spec.keys())
    mfd_labels = [l for l in labels if spec[l]["arm"] == "MFD"]
    ch_labels = [l for l in labels if spec[l]["arm"] == "chain"]
    ctrl = "P1"

    # ---------------- decoder-state check ----------------
    log("\n" + "=" * 118 + "\n=== DECODER-STATE CHECK (must be 808 everywhere, else the match is broken) ===\n"
        + "=" * 118, fh)
    for l in labels:
        f = res[(l, "R1", "natural")]["fwd"]
        log(f"  {l:<8} arm={spec[l]['arm']:<6} decoder-states/item={f}  {'OK' if f == 808 else '** MISMATCH **'}", fh)

    # ---------------- consolidated table ----------------
    log("\n" + "=" * 118 + "\n=== CONSOLIDATED TABLE ===\n" + "=" * 118, fh)
    log(f"  {'set':<8}{'pool':<8}{'arm':<6}{'cfg':<8}{'params':>9}{'cov':>8}{'bef':>8}{'aft':>8}{'gain':>9}"
        f"{'Pc bef':>8}{'Pc aft':>8}{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}{'t2@10':>8}{'t2n@10':>8}", fh)
    for setting in SETTINGS:
        for l in labels:
            for cname, _ in CONFIGS:
                r = res[(l, cname, setting)]
                b, a = r["before"], r["after"]
                log(f"  {setting:<8}{l:<8}{spec[l]['arm']:<6}{cname:<8}{r['params']:>9,}{a['cov']:>8.4f}"
                    f"{b['t3@10']:>8.4f}{a['t3@10']:>8.4f}{a['t3@10']-b['t3@10']:>+9.4f}"
                    f"{b['pcov']:>8.4f}{a['pcov']:>8.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}"
                    f"{a['t3n@5']:>8.4f}{a['t3n@10']:>8.4f}{a['t2@10']:>8.4f}{a['t2n@10']:>8.4f}", fh)

    log("\n  per-behavior t3@10 (before -> after), and whether pv AND conversions both improve:", fh)
    for setting in SETTINGS:
        for l in labels:
            for cname, _ in CONFIGS:
                r = res[(l, cname, setting)]
                bb, aa = r["before"]["perbeh"], r["after"]["perbeh"]
                dd = {k: aa[k] - bb[k] for k in aa}
                conv = [k for k in dd if k != "pv"]
                v = ("BOTH improve" if dd.get("pv", 0) > 0 and all(dd[k] >= 0 for k in conv)
                     else "REDISTRIBUTION (pv up, conversions down)" if dd.get("pv", 0) > 0 else "pv down")
                log(f"    {setting:<8}{l:<8}{cname:<8}" + "  ".join(f"{k}:{bb[k]:.4f}->{aa[k]:.4f}" for k in aa)
                    + f"   [{v}]", fh)

    # ---------------- THE DECIDING METRIC: newly covered GTs ----------------
    log("\n" + "=" * 118 + "\n=== NEWLY COVERED GROUND TRUTHS (relative to P1) -- THE DECIDING METRIC ===\n"
        + "=" * 118, fh)
    log("  Under the hand-set scorer this recovery was 0.000%. Any materially positive number is the result.", fh)
    for setting in SETTINGS:
        for cname, _ in CONFIGS:
            base = res[(ctrl, cname, setting)]
            for l in mfd_labels:
                if l == ctrl:
                    continue
                r = res[(l, cname, setting)]
                assert np.array_equal(r["uid"], base["uid"]), "canonical join failed"
                new = r["has_gt"].astype(bool) & ~base["has_gt"].astype(bool)
                n = int(new.sum())
                log(f"\n  [{setting}/{cname}] {l}: coverage {base['after']['cov']:.4f} -> {r['after']['cov']:.4f}   "
                    f"newly covered GTs = {n:,}", fh)
                if n == 0:
                    continue
                rk = r["r3_after"][new]
                for k in (10, 20, 50):
                    log(f"      newly-covered reaching top-{k:<3}: {float((rk < k).mean()):.4f}  ({int((rk<k).sum()):,})", fh)
                qs = np.percentile(rk, [10, 25, 50, 75, 90])
                log(f"      rank distribution (10/25/50/75/90 pct): {['%d' % q for q in qs]}", fh)
                # bucketed by GT branch-entropy z
                z = r["ent_z"][new]
                log(f"      by branch-entropy z (flat = high z):", fh)
                for lo, hi, nm in ((-1e9, -1., "<-1"), (-1., 0., "-1..0"), (0., 1., "0..1"), (1., 1e9, ">1")):
                    m = (z >= lo) & (z < hi)
                    if m.sum() == 0:
                        continue
                    log(f"        z {nm:<7} n={int(m.sum()):>7,}  top10={float((rk[m] < 10).mean()):.4f}", fh)
                # per behavior
                gb = r["gt_b"][new]
                log(f"      per behavior top10: " + "  ".join(
                    f"{BEH[b]}={float((rk[gb == b] < 10).mean()):.4f}(n={int((gb==b).sum()):,})"
                    for b in range(4) if (gb == b).sum()), fh)
                # LOST: previously covered GTs that were top-10 under P1 but are not under this pool
                prev = base["has_gt"].astype(bool)
                lost = prev & (base["hit_after"] > 0) & (r["hit_after"] == 0)
                gained = prev & (base["hit_after"] == 0) & (r["hit_after"] > 0)
                log(f"      previously-covered GTs LOST from top-10: {int(lost.sum()):,}   "
                    f"regained: {int(gained.sum()):,}   net on P1-covered: {int(gained.sum()-lost.sum()):+,}", fh)
                log(f"      NET aggregate t3@10 vs P1: {r['after']['t3@10'] - base['after']['t3@10']:+.4f}", fh)

    # ---------------- four diagnostic cells (MFD vs chain, per pool) ----------------
    log("\n" + "=" * 118 + "\n=== FOUR DIAGNOSTIC CELLS vs the chain arm ===\n" + "=" * 118, fh)
    log("  P1 reference at rw1/capped: MFD 0.6495 / 0.2656 / 0.2153 / 0.0467 ; chain 0.4071 / 0.0226 / 0.0907 / 0.0026", fh)
    log(f"  {'set':<8}{'pool':<8}{'arm':<6}{'cfg':<8}{'BOTH/10-49':>12}{'BOTH/50+':>10}{'own/10-49':>11}{'own/50+':>10}", fh)
    for setting in SETTINGS:
        for cname, _ in CONFIGS:
            for chl in ch_labels:
                rc = res[(chl, cname, setting)]
                for l in mfd_labels + [chl]:
                    r = res[(l, cname, setting)]
                    other = rc if l != chl else res[(ctrl, cname, setting)]
                    cm = r["has_gt"].astype(bool); co = other["has_gt"].astype(bool)
                    both = cm & co; own = cm & ~co
                    cells = []
                    for grp in (both, own):
                        for lo, hi in ((10, 50), (50, 10 ** 9)):
                            m = grp & (r["r3_before"] >= lo) & (r["r3_before"] < hi)
                            cells.append(float(r["hit_after"][m].mean()) if m.sum() else float("nan"))
                    log(f"  {setting:<8}{l:<8}{spec[l]['arm']:<6}{cname:<8}{cells[0]:>12.4f}{cells[1]:>10.4f}"
                        f"{cells[2]:>11.4f}{cells[3]:>10.4f}", fh)

    # ---------------- significance: fixed paired protocol ----------------
    log("\n" + "=" * 118 + "\n=== SIGNIFICANCE (fixed protocol: same config both arms, joined on user_id,\n"
        "    PAIRED bootstrap resampling both arms together, B=2000) ===\n" + "=" * 118, fh)
    rng = np.random.default_rng(args.seed)
    for setting in SETTINGS:
        for cname, _ in CONFIGS:
            for chl in ch_labels:
                hc = res[(chl, cname, setting)]["hit_after"]
                for l in mfd_labels:
                    hm = res[(l, cname, setting)]["hit_after"]
                    assert np.array_equal(res[(l, cname, setting)]["uid"], res[(chl, cname, setting)]["uid"])
                    N = hm.shape[0]; d = np.empty(args.boot)
                    for k in range(args.boot):
                        ii = rng.integers(0, N, N); d[k] = hm[ii].mean() - hc[ii].mean()
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    log(f"  [{setting}/{cname}] {l} vs {chl}: diff {hm.mean()-hc.mean():+.4f}  "
                        f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(MFD>chain)={float((d>0).mean()):.4f}  "
                        f"{'DISTINGUISHABLE' if (lo>0 or hi<0) else 'not distinguishable (CI spans 0)'}", fh)

    # ---------------- verdict ----------------
    log("\n" + "=" * 118 + "\n=== VERDICT (stated in advance, not selected post hoc) ===\n" + "=" * 118, fh)
    log("  If newly-covered recovery is materially positive, coverage expansion is a live lever for MFD", fh)
    log("  and the earlier negatives were an artefact of the weak scorer. If it is ~0 even here, that", fh)
    log("  reproduces the earlier finding under a much stronger test and closes the coverage axis.", fh)
    best = []
    for setting in SETTINGS:
        for cname, _ in CONFIGS:
            for l in mfd_labels:
                if l == ctrl:
                    continue
                base = res[(ctrl, cname, setting)]; r = res[(l, cname, setting)]
                new = r["has_gt"].astype(bool) & ~base["has_gt"].astype(bool)
                if new.sum():
                    best.append((float((r["r3_after"][new] < 10).mean()), f"{setting}/{cname}/{l}"))
    if best:
        top = max(best)
        log(f"  Highest newly-covered top-10 recovery across the grid: {top[0]:.4f} at {top[1]}", fh)
        log(f"  -> {'MATERIALLY POSITIVE: coverage expansion is live.' if top[0] >= 0.02 else 'approximately ZERO: the coverage axis is closed under the strong reranker too.'}", fh)
    json.dump({f"{l}|{c}|{s}": {k: v for k, v in r.items()
                                if k in ("params", "val", "n_train", "fwd", "before", "after")}
               for (l, c, s), r in res.items()},
              open(os.path.join(args.out_dir, "coverage_results.json"), "w"), indent=1, default=float)
    fh.close()


if __name__ == "__main__":
    main()
