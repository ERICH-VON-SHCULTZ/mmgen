"""Repair the cross-arm event alignment in the matched head-to-head, then produce the grouped
breakdown that explains WHERE the reranker's gain comes from.

WHAT WENT WRONG: the previous compare tested `torch.equal(user_id_mfd, user_id_chain)` -- an
ORDER-SENSITIVE check. The two dumps hold the same 83,005 events (gt_b multisets are equal) but in
different ROW ORDER (gt_b agreement only ~0.893, first mismatch at row 82): the two arms were dumped
with different test batch sizes and tf.data interleaving makes row order batch-size dependent. So row
order is not a key. The repair is to JOIN on the stored per-event identifier instead of assuming
alignment, and to PROVE the join with an independent field (gt_behavior) rather than trusting it.

Join strategy, in order, with the choice reported:
  1. user_id  -- the natural key (leave-one-out: one test event per user). Requires non-degenerate
     and unique ids covering the same set on both sides.
  2. fallback fingerprint -- (gt_behavior, history-behavior sequence, history length). Behaviors are
     NOT tokenized, so this channel is identical across the two SID spaces. Used only if user_id is
     degenerate; the collision rate is reported and the run ABORTS if the key is not near-unique,
     rather than emitting a wrong partition.
Either way the join is validated by post-join gt_behavior agreement (must be ~1.0).

Then: partition covered GTs into BOTH / MFD-only / chain-only, and decompose each arm's t3@10 gain
into per-group contributions that sum to the whole; split by pre-rerank rank band (10-49 / 50+);
per behavior; MFD-only entropy-z buckets; and bootstrap CIs over EVENTS for each gain and for the
MFD-minus-chain difference.
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
KF_MAX = 2048


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


def sid_widths(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    s = o if torch.is_tensor(o) else next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2)
    if s.shape[0] > s.shape[1]:
        s = s.t().contiguous()
    return [int(s[l].max()) + 1 for l in range(s.shape[0])]


class Arm:
    def __init__(self, name, dump_path, ranks_path, sid_path, fh):
        d = torch.load(dump_path, map_location="cpu", weights_only=False)
        c, self.meta = d["cols"], d["meta"]
        self.name = name
        N, nb, Kp = c["pool_f"].shape
        self.N, self.nb, self.Kp, self.C = N, nb, Kp, nb * Kp
        f32 = lambda x: x.float() if x.dtype == torch.float16 else x
        flat = lambda x: x.reshape(N, nb * Kp)
        self.uid = c["user_id"].long()
        self.pf, self.pt, self.pi = flat(c["pool_f"]).long(), flat(c["pool_t"]).long(), flat(c["pool_i"]).long()
        self.valid = f32(flat(c["pool_sidscore"])) > -1e30
        self.lp_t, self.lp_i = f32(flat(c["pool_lpt"])), f32(flat(c["pool_lpi"]))
        self.branch = torch.arange(nb).repeat_interleave(Kp)
        self.gt_b = c["gt_b"].long(); self.gt_f = c["gt_f"].long()
        self.gt_t = c["gt_t"].long(); self.gt_i = c["gt_i"].long()
        self.hist = c["hist_ids"].long() if "hist_ids" in c else None
        r = torch.load(ranks_path, map_location="cpu", weights_only=False)
        self.br3, self.bh3 = r["base_r3"], r["base_h3"]
        self.rr3, self.rh3 = r["rr3"], r["rh3"]
        self.has_gt = r["has_gt"]
        self.n_train = r.get("n_train_events")
        # widths READ from the SID file, then asserted against every dumped code id
        self.widths = sid_widths(sid_path)
        for lvl, (nm, col) in enumerate((("l0", self.pf), ("l1", self.pt), ("l2", self.pi))):
            mx = int(col.max())
            assert mx < self.widths[lvl], f"{name} {nm}: code {mx} >= width {self.widths[lvl]}"
        log(f"  [{name}] N={N:,} Kp={Kp} widths(read)={self.widths} covered={int(self.has_gt.sum()):,} "
            f"train_events={self.n_train} fwd_states={self.meta.get('fwd_states_per_item','?')}", fh)
        per = {b: int((self.gt_b == b).sum()) for b in range(nb)}
        ok = N == EXPECT_TOTAL and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
        log(f"  [{name}] split " + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH)
            + f" -> {'OK' if ok else 'MISMATCH'}", fh)
        if not ok:
            raise SystemExit(f"** {name} split mismatch -- stop. **")

    def fingerprint(self):
        """Cross-arm-invariant event fingerprint: gt behavior + the history BEHAVIOR channel (not
        tokenized, so identical in both SID spaces) + history length."""
        if self.hist is None:
            return None
        H = self.hist.shape[1] // 4
        hb = self.hist[:, :H * 4].reshape(self.N, H, 4)[:, :, 0]          # (N,H) behaviors, -1 pad
        valid = (hb >= 0)
        ln = valid.sum(1)
        key = torch.zeros(self.N, dtype=torch.int64)
        for j in range(H):                                                # rolling hash over behaviors
            key = key * 5 + torch.where(valid[:, j], hb[:, j] + 1, torch.zeros_like(hb[:, j]))
            key = key % (2 ** 61 - 1)
        return key * 8 + self.gt_b * 2 + (ln % 2)

    def permute(self, perm):
        for a in ("uid", "pf", "pt", "pi", "valid", "lp_t", "lp_i", "gt_b", "gt_f", "gt_t", "gt_i",
                  "br3", "bh3", "rr3", "rh3", "has_gt"):
            v = getattr(self, a)
            if v is not None:
                setattr(self, a, v[perm])
        if self.hist is not None:
            self.hist = self.hist[perm]


def align(mfd, ch, fh):
    log("\n" + "=" * 92 + "\n=== ALIGNMENT DIAGNOSIS + REPAIR ===\n" + "=" * 92, fh)
    um, uc = mfd.uid, ch.uid
    log(f"  stored event identifier: 'user_id' (int32 in both dumps)", fh)
    for nm, u in (("MFD", um), ("chain", uc)):
        log(f"    [{nm}] min={int(u.min())} max={int(u.max())} unique={int(torch.unique(u).numel()):,}/{u.numel():,} "
            f"all_zero={bool((u == 0).all())}", fh)
    log(f"  row-order equality: user_id identical={bool(torch.equal(um, uc))}  "
        f"gt_b identical={bool(torch.equal(mfd.gt_b, ch.gt_b))}  "
        f"gt_b agreement in row order={float((mfd.gt_b == ch.gt_b).float().mean()):.4f}", fh)
    log(f"  multiset equality: user_id={bool(torch.equal(um.sort().values, uc.sort().values))}  "
        f"gt_b={bool(torch.equal(mfd.gt_b.sort().values, ch.gt_b.sort().values))}", fh)
    log("  -> same event SET, different ROW ORDER (tf.data interleaving is batch-size dependent;", fh)
    log("     the arms were dumped at different test batch sizes). Row order is NOT a key.", fh)

    usable = (not bool((um == 0).all()) and not bool((uc == 0).all())
              and int(torch.unique(um).numel()) == um.numel()
              and int(torch.unique(uc).numel()) == uc.numel()
              and bool(torch.equal(um.sort().values, uc.sort().values)))
    if usable:
        key_m, key_c, how = um, uc, "user_id"
    else:
        log("  user_id unusable (degenerate / non-unique / different sets) -> falling back to a", fh)
        log("  reconstructed fingerprint: gt_behavior + history-BEHAVIOR sequence + history length", fh)
        log("  (the behavior channel is not tokenized, so it is identical across the two SID spaces).", fh)
        key_m, key_c, how = mfd.fingerprint(), ch.fingerprint(), "history fingerprint"
        if key_m is None or key_c is None:
            raise SystemExit("** no usable join key (no user_id, no hist_ids) -- stop. **")
        for nm, k in (("MFD", key_m), ("chain", key_c)):
            u = int(torch.unique(k).numel())
            log(f"    [{nm}] fingerprint unique={u:,}/{k.numel():,} (collision rate "
                f"{1 - u / k.numel():.4f})", fh)
            if u / k.numel() < 0.999:
                raise SystemExit("** fingerprint not near-unique -- refusing to emit a wrong partition. **")

    # join: reorder CHAIN rows into MFD row order
    order_m = torch.argsort(key_m); order_c = torch.argsort(key_c)
    if not bool(torch.equal(key_m[order_m], key_c[order_c])):
        raise SystemExit("** join keys do not match as sets -- stop. **")
    perm = torch.empty(mfd.N, dtype=torch.long)
    perm[order_m] = order_c                       # mfd row i  <->  chain row perm[i]
    ch.permute(perm)
    agree = float((mfd.gt_b == ch.gt_b).float().mean())
    log(f"\n  JOIN: on {how};  joined events = {mfd.N:,}/{EXPECT_TOTAL:,}  missing from either side = 0", fh)
    log(f"  VALIDATION: post-join gt_behavior agreement = {agree:.6f}  "
        f"(independent of the key -- must be ~1.0)", fh)
    if agree < 0.999:
        raise SystemExit("** post-join gt_behavior disagreement -- join is wrong, refusing to continue. **")
    log("  -> alignment REPAIRED.", fh)
    return how


def hits(r, h, k=10):
    return (h & (r >= 0) & (r < k))


def entropy_z(arm, dev):
    N, C, nb = arm.N, arm.C, arm.nb
    ev = torch.arange(N).view(N, 1).expand(N, C)
    gk = (ev.to(torch.int64) * (nb * KF_MAX) + arm.branch.view(1, -1).expand(N, C) * KF_MAX + arm.pf)
    v = arm.valid
    g = gk[v].to(dev); xt = arm.lp_t[v].to(dev); xi = arm.lp_i[v].to(dev)
    pt = arm.pt[v].to(dev); pi = arm.pi[v].to(dev)
    uniq, gid = torch.unique(g, return_inverse=True); G = uniq.numel()

    def H(x, code, K):
        tk = gid * K + code
        utk, gtd = torch.unique(tk, return_inverse=True)
        lp = torch.full((utk.numel(),), -1e30, device=dev).scatter_reduce(0, gtd, x, reduce="amax", include_self=True)
        bb = (utk // K).long(); P = lp.exp()
        Z = torch.zeros(G, device=dev).index_add_(0, bb, P)
        p = P / Z[bb].clamp(min=1e-12)
        return torch.zeros(G, device=dev).index_add_(0, bb, -p * p.clamp(min=1e-12).log())

    Hs = H(xt, pt, 1024) + H(xi, pi, 1024)
    eg = (uniq // (nb * KF_MAX)).long()
    n = torch.zeros(N, device=dev).index_add_(0, eg, torch.ones_like(Hs))
    mu = torch.zeros(N, device=dev).index_add_(0, eg, Hs) / n.clamp(min=1)
    ss = torch.zeros(N, device=dev).index_add_(0, eg, Hs * Hs)
    sd = (ss / n.clamp(min=1) - mu.pow(2)).clamp(min=0).sqrt()
    gtk = (torch.arange(N) * (nb * KF_MAX) + arm.gt_b * KF_MAX + arm.gt_f).to(dev)
    pos = torch.searchsorted(uniq, gtk)
    avail = (pos < G) & (uniq[pos.clamp(max=G - 1)] == gtk)
    z = ((Hs[pos.clamp(max=G - 1)] - mu) / sd.clamp(min=1e-6)).cpu()
    return z, avail.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_dump", required=True); ap.add_argument("--chain_dump", required=True)
    ap.add_argument("--mfd_ranks", required=True); ap.add_argument("--chain_ranks", required=True)
    ap.add_argument("--mfd_ranks_sub", default="")
    ap.add_argument("--md_sid", required=True); ap.add_argument("--tg_sid", required=True)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/matched_rerank")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "aligned_breakdown.txt"), "w")
    log("=" * 92 + "\n=== MATCHED HEAD-TO-HEAD: aligned grouped breakdown ===\n" + "=" * 92, fh)

    mfd = Arm("MFD", args.mfd_dump, args.mfd_ranks, args.md_sid, fh)
    ch = Arm("chain", args.chain_dump, args.chain_ranks, args.tg_sid, fh)
    how = align(mfd, ch, fh)

    N = mfd.N
    hm_b, hm_a = hits(mfd.br3, mfd.bh3), hits(mfd.rr3, mfd.rh3)
    hc_b, hc_a = hits(ch.br3, ch.bh3), hits(ch.rr3, ch.rh3)
    cm, cc = mfd.has_gt, ch.has_gt
    log("\n" + "=" * 92 + "\n=== HEADLINE (aligned) ===\n" + "=" * 92, fh)
    for nm, hb, ha, cov in (("MFD", hm_b, hm_a, cm), ("chain", hc_b, hc_a, cc)):
        log(f"  {nm:<6} coverage={cov.float().mean():.4f}  t3@10 {hb.float().mean():.4f} -> {ha.float().mean():.4f} "
            f"({ha.float().mean()-hb.float().mean():+.4f})  P(r<10|cov) "
            f"{hb[cov].float().mean():.4f} -> {ha[cov].float().mean():.4f}", fh)

    # ---------------- coverage partition ----------------
    both, only_m, only_c = cm & cc, cm & ~cc, cc & ~cm
    log("\n" + "=" * 92 + f"\n=== COVERAGE PARTITION (exact; joined on {how} + GT SID tuple per arm) ===\n" + "=" * 92, fh)
    log(f"  BOTH={int(both.sum()):,}   MFD-only={int(only_m.sum()):,}   chain-only={int(only_c.sum()):,}   "
        f"(MFD covered {int(cm.sum()):,}, chain covered {int(cc.sum()):,})", fh)
    log(f"\n  {'arm / group':<22}{'n':>9}{'P(r<10) before':>16}{'after':>9}{'moved in':>10}{'moved out':>11}"
        f"{'contrib to gain':>17}", fh)
    contrib = {}
    for arm, hb, ha, groups in (("MFD", hm_b, hm_a, (("BOTH", both), ("MFD-only", only_m))),
                                ("chain", hc_b, hc_a, (("BOTH", both), ("chain-only", only_c)))):
        tot = 0.0
        for gname, g in groups:
            n = int(g.sum())
            if n == 0:
                continue
            mi = int((ha & ~hb & g).sum()); mo = int((hb & ~ha & g).sum())
            cgain = (mi - mo) / N
            tot += cgain
            log(f"  {arm+' / '+gname:<22}{n:>9,}{hb[g].float().mean():>16.4f}{ha[g].float().mean():>9.4f}"
                f"{mi:>10,}{mo:>11,}{cgain:>+17.4f}", fh)
        # uncovered events contribute nothing (GT not in pool -> never a hit)
        full = float(ha.float().mean() - hb.float().mean())
        log(f"  {arm+' / TOTAL':<22}{'':>9}{'':>16}{'':>9}{'':>10}{'':>11}{tot:>+17.4f}"
            f"   (arm total gain {full:+.4f}; parts sum check {'OK' if abs(tot-full)<1e-9 else 'MISMATCH'})", fh)
        contrib[arm] = tot
    log("\n  READ: if the gain sits in BOTH, MFD merely ranked shared GTs worse and the headroom is", fh)
    log("  bounded as P(r<10|cov) approaches chain's. If MFD-only carries it, the wider pool is being", fh)
    log("  converted and coverage expansion is worth revisiting.", fh)

    # ---------------- rank-band split ----------------
    log("\n" + "=" * 92 + "\n=== BY PRE-RERANK RANK BAND ===\n" + "=" * 92, fh)
    log(f"  {'arm / group / band':<34}{'n':>9}{'-> top10 after':>16}{'moved in':>10}", fh)
    for arm, hb, ha, r_b, groups in (("MFD", hm_b, hm_a, mfd.br3, (("BOTH", both), ("MFD-only", only_m))),
                                     ("chain", hc_b, hc_a, ch.br3, (("BOTH", both), ("chain-only", only_c)))):
        for gname, g in groups:
            for lo, hi, bn in ((10, 50, "10-49"), (50, 10 ** 9, "50+")):
                gg = g & (r_b >= lo) & (r_b < hi) & ~hb
                n = int(gg.sum())
                if n == 0:
                    continue
                log(f"  {arm+' / '+gname+' / '+bn:<34}{n:>9,}{ha[gg].float().mean():>16.4f}{int((ha&gg).sum()):>10,}", fh)

    # ---------------- per behavior ----------------
    log("\n" + "=" * 92 + "\n=== PER BEHAVIOR (t3@10 before -> after) ===\n" + "=" * 92, fh)
    for arm, hb, ha in (("MFD", hm_b, hm_a), ("chain", hc_b, hc_a)):
        log(f"  {arm:<6} " + "   ".join(
            f"{BEH[b]}: {hb[mfd.gt_b==b].float().mean():.4f}->{ha[mfd.gt_b==b].float().mean():.4f}"
            for b in range(mfd.nb) if (mfd.gt_b == b).any()), fh)

    # ---------------- MFD entropy-z buckets ----------------
    log("\n" + "=" * 92 + "\n=== MFD-ONLY: lift by GT-branch entropy z (flat = high z) ===\n" + "=" * 92, fh)
    z, avail = entropy_z(mfd, dev)
    covz = cm & avail
    log(f"  {'z bucket':<12}{'n':>9}{'P(r<10) before':>16}{'after':>9}{'lift':>9}", fh)
    for lo, hi, nm in ((-1e9, -1., "<-1"), (-1., -.5, "-1..-.5"), (-.5, 0., "-.5..0"),
                       (0., .5, "0..0.5"), (.5, 1., "0.5..1"), (1., 1e9, ">1")):
        g = covz & (z >= lo) & (z < hi)
        n = int(g.sum())
        if n == 0:
            continue
        pb, pa = float(hm_b[g].float().mean()), float(hm_a[g].float().mean())
        log(f"  {nm:<12}{n:>9,}{pb:>16.4f}{pa:>9.4f}{pa-pb:>+9.4f}", fh)
    log("  READ: lift concentrated in the FLAT (high-z) buckets => the history signal reaches the", fh)
    log("  low-confidence region where MFD's extra coverage lives (and entropy-driven budget", fh)
    log("  reallocation is worth re-testing under this ranker). Lift only in peaked buckets => not.", fh)

    # ---------------- bootstrap over EVENTS ----------------
    log("\n" + "=" * 92 + f"\n=== BOOTSTRAP over EVENTS (B={args.boot}, resampling events not candidates) ===\n" + "=" * 92, fh)
    rng = np.random.default_rng(args.seed)
    a_m_b, a_m_a = hm_b.numpy().astype(np.float64), hm_a.numpy().astype(np.float64)
    a_c_b, a_c_a = hc_b.numpy().astype(np.float64), hc_a.numpy().astype(np.float64)
    gm = np.empty(args.boot); gc = np.empty(args.boot); df = np.empty(args.boot)
    for k in range(args.boot):
        idx = rng.integers(0, N, N)
        mb, ma = a_m_b[idx].mean(), a_m_a[idx].mean()
        cb, ca = a_c_b[idx].mean(), a_c_a[idx].mean()
        gm[k], gc[k], df[k] = ma - mb, ca - cb, ma - ca
    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    lm, hm_ = ci(gm); lc, hc_ = ci(gc); ld, hd = ci(df)
    log(f"  MFD gain           = {a_m_a.mean()-a_m_b.mean():+.4f}  95% CI [{lm:+.4f}, {hm_:+.4f}]", fh)
    log(f"  chain gain         = {a_c_a.mean()-a_c_b.mean():+.4f}  95% CI [{lc:+.4f}, {hc_:+.4f}]", fh)
    log(f"  MFD - chain (after)= {a_m_a.mean()-a_c_a.mean():+.4f}  95% CI [{ld:+.4f}, {hd:+.4f}]  "
        f"P(MFD>chain)={float((df>0).mean()):.4f}", fh)
    sig = "DISTINGUISHABLE from noise" if (ld > 0 or hd < 0) else "NOT distinguishable from noise (CI spans 0)"
    log(f"  -> the post-rerank lead is {sig}.", fh)

    # ---------------- subsampled MFD ----------------
    if args.mfd_ranks_sub and os.path.exists(args.mfd_ranks_sub):
        rs = torch.load(args.mfd_ranks_sub, map_location="cpu", weights_only=False)
        # the sub run used the SAME MFD dump -> same row order as mfd BEFORE permutation (mfd was
        # never permuted; only chain was), so no reordering is needed here.
        sb, sa = hits(rs["base_r3"], rs["base_h3"]), hits(rs["rr3"], rs["rh3"])
        log(f"\n  MFD subsampled to chain's {rs.get('n_train_events')} train events: "
            f"t3@10 {sb.float().mean():.4f} -> {sa.float().mean():.4f} ({sa.float().mean()-sb.float().mean():+.4f})", fh)
    fh.close()


if __name__ == "__main__":
    main()
