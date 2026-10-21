"""Matched-COMPUTE comparison: chain sequential vs MFD factorized decoding at EQUAL decoder-state
budget, rather than at equal scored-pool size.

Decoder-state evaluations per item (the compute metric being matched):
  chain : 1 (behavior) + 3 (teacher-forced levels) + nb*(1 + 2*Bw)          -> grows with the beam Bw
  MFD   : nb*2 (Stage-A behavior+f slots) + nb*Kf (modality heads at kept roots)
          == nb*(2 + Kf).  K_t / K_i are FREE: text/image are marginal head-reads off the shared
          (b,f) root and add ZERO decoder states. That is the factorization.

So the matched points are:
  chain@808  : Bw=100   -> 4 + 4*(1+200)  = 808
  chain@4008 : Bw=500   -> 4 + 4*(1+1000) = 4008
  MFD@808    : Kf=200   -> 4*(2+200)      = 808     (the cited reference config; K_t=200,K_i=50)
  MFD@4008   : Kf=1000  -> 4*(2+1000)     = 4008    (K_t,K_i widened to codebook-full for FREE)

Because K_t/K_i are free, the honest allocation of an increased decoder-state budget in MFD is
degenerate: the entire budget goes to Kf (the only decoder-state-costing beam); K_t/K_i are then
widened to codebook-full at no cost. f_pruned is also the largest branch loss, so this matches the
"allocate by branch loss" rule as well. To separate the PAID lever (Kf) from the FREE lever
(K_t,K_i) we also read an auxiliary point MFD@808+free (Kf=200, K_t=K_i=full) -> the 808->4008 MFD
move then decomposes as [free-lever] + [paid Kf].

recall@10 is invariant to K_pool (K_pool=500 on every arm), so scoring is identical across points and
does not bias the compute comparison. Everything is per behavior (t3 is ~93% pv; conversion-only
effects are invisible in aggregate).

Reports, per point: exact fwd_states achieved (+ residual mismatch vs the 808/4008 target), coverage
@pool (ALL + per behavior), the coverage-loss decomposition in pruning order, recall@10,
P(rank<10|cov), pool diversity (unique l0 / 2-prefix / largest group), and wall-clock sec/1000 events.
Then one four-row headline table and the paid-vs-free decomposition.
"""

import argparse

import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}


def bar(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cols = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in d["cols"].items()}
    return cols, d["meta"]


def build_cand(c):
    N, nb, Kp = c["pool_f"].shape
    flat = lambda x: x.reshape(N, nb * Kp)
    return {
        "branch": torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1).contiguous(),
        "f": flat(c["pool_f"]).long(), "t": flat(c["pool_t"]).long(), "i": flat(c["pool_i"]).long(),
        "sid": flat(c["pool_sidscore"]), "valid": flat(c["pool_sidscore"]) > -1e30,
    }, N, nb, Kp


def union_score(cand, c, wb):
    s = wb * torch.gather(c["beh_logp"], 1, cand["branch"]) + cand["sid"]
    return s.masked_fill(~cand["valid"], float("-inf"))


def best_rank(score, match, eligible):
    score = score.masked_fill(~eligible, float("-inf"))
    m = torch.where(match & eligible, score, torch.full_like(score, float("-inf")))
    best = m.max(1).values
    has = best > -1e30
    rank = (score > best.unsqueeze(1)).sum(1)
    return torch.where(has, rank, torch.full_like(rank, -1)), has


def recall_at(rank, has, k, denom=None):
    hit = has & (rank >= 0) & (rank < k)
    return (hit[denom] if denom is not None else hit).float().mean().item()


def in_pool_mask(cand, c):
    gt_f, gt_t, gt_i, gt_b = c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long(), c["gt_b"].long()
    sid = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    beh = cand["branch"] == gt_b[:, None]
    return (sid & beh & cand["valid"]).any(1)


def fwd_states(meta, arm):
    if "fwd_states_per_item" in meta:
        return int(meta["fwd_states_per_item"])
    nb = meta["nb"]
    if arm == "MFD":
        return nb * (2 + int(meta["Kf"]))          # legacy dump (pre-meta-field)
    return 1 + 3 + nb * (1 + 2 * int(meta["beam"]))  # legacy chain dump


def sec_per_1k(meta):
    if meta.get("gen_events", 0):
        return 1000.0 * float(meta["gen_seconds"]) / float(meta["gen_events"])
    return None


def diversity(cand, gt_b, nb, cap=4000):
    """unique-l0 / unique-2-prefix / largest-l0-group over the per-event union pool. Returns dict
    metric -> [ALL, b0, b1, ...]. Loops are O(pool); subsample <=cap events per group."""
    f_all, t_all, i_all, val = cand["f"], cand["t"], cand["i"], cand["valid"]
    k1 = int(t_all.max().item()) + 1
    pre2 = f_all * k1 + t_all

    def over(mask, fn):
        idx = torch.nonzero(mask).squeeze(1)
        if idx.numel() > cap:
            idx = idx[torch.randperm(idx.numel())[:cap]]
        if idx.numel() == 0:
            return float("nan")
        return float(sum(fn(e) for e in idx.tolist()) / idx.numel())

    u_l0 = lambda e: torch.unique(f_all[e][val[e]]).numel()
    u_p2 = lambda e: torch.unique(pre2[e][val[e]]).numel()

    def lg(e):
        v = f_all[e][val[e]]
        return int(torch.bincount(v).max()) if v.numel() else 0

    groups = [torch.ones_like(gt_b, dtype=torch.bool)] + [gt_b == b for b in range(nb)]
    return {
        "uniq_l0": [over(g, u_l0) for g in groups],
        "uniq_pre2": [over(g, u_p2) for g in groups],
        "largest_grp": [over(g, lg) for g in groups],
    }


def compute(label, arm, path, wb):
    c, meta = load(path)
    nb = meta["nb"]
    gt_b = c["gt_b"].long()
    N = gt_b.shape[0]
    cand, _, _, Kp = build_cand(c)
    covered = in_pool_mask(cand, c)

    score = union_score(cand, c, wb)
    sid = (cand["f"] == c["gt_f"].long()[:, None]) & (cand["t"] == c["gt_t"].long()[:, None]) & \
          (cand["i"] == c["gt_i"].long()[:, None])
    beh = cand["branch"] == gt_b[:, None]
    rank, has = best_rank(score, sid & beh, cand["valid"])

    # coverage-loss decomposition (pruning order), mutually exclusive
    fr, tr, ir = c["f_star_rank"], c["tf_t_rank"], c["tf_i_rank"]
    gv = c["gt_in_catalog"].bool()
    if arm == "MFD":
        c0, c1, c2 = meta["Kf"], meta["Kt"], meta["Ki"]
        names = ("f_pruned", "branch_t", "branch_i")
    else:
        beam = meta["beam"]; c0 = c1 = c2 = beam
        names = ("level0", "level1", "level2")
    p0 = fr >= c0
    p1 = (~p0) & (tr >= c1)
    p2 = (~p0) & (tr < c1) & (ir >= c2)
    composed_ok = (~p0) & (tr < c1) & (ir < c2) & gv
    trunc = composed_ok & (~covered)
    n_inv = int((~gv).sum())
    assert n_inv == 0, f"CATALOG INVALID = {n_inv} (>0) for {label}: constraint set is WRONG"

    groups = [torch.ones_like(gt_b, dtype=torch.bool)] + [gt_b == b for b in range(nb)]
    per = lambda vec: [vec[g].float().mean().item() for g in groups]
    r10 = lambda m: recall_at(rank, has, 10, m)
    p_cov = lambda g: (recall_at(rank, has, 10, g & covered) if (g & covered).any() else 0.0)

    div = diversity(cand, gt_b, nb, cap=4000)
    fs = fwd_states(meta, arm)
    return {
        "label": label, "arm": arm, "path": path, "meta": meta, "nb": nb, "N": N, "Kp": Kp,
        "fwd_states": fs, "sec_per_1k": sec_per_1k(meta),
        "config": (f"K_f={meta['Kf']} K_t={meta['Kt']} K_i={meta['Ki']}" if arm == "MFD"
                   else f"beam={meta['beam']} (l0/l1/l2)"),
        "cov": per(covered), "recall10": [r10(g) for g in groups], "p_cov": [p_cov(g) for g in groups],
        "decomp_names": names,
        "decomp": {names[0]: per(p0), names[1]: per(p1), names[2]: per(p2), "trunc": per(trunc)},
        "div": div,
    }


def row(label, cells, w=10, lw=16):
    return f"  {label:<{lw}}" + "".join(f"{x:>{w}}" for x in cells)


def fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_point(r):
    nb = r["nb"]
    bar(f"{r['label']}   [{r['arm']}]   fwd_states/item = {r['fwd_states']}")
    tgt = 808 if r["fwd_states"] <= 2000 else 4008
    print(f"  dump: {r['path']}   events={r['N']:,}   {r['config']}   K_pool={r['Kp']}")
    print(f"  decoder-state target={tgt}  achieved={r['fwd_states']}  residual={r['fwd_states'] - tgt:+d}"
          f"  ({100.0*(r['fwd_states']-tgt)/tgt:+.1f}%)")
    sp = r["sec_per_1k"]
    print(f"  wall-clock generation: {'n/a (legacy dump, no timer)' if sp is None else f'{sp:.2f} sec / 1000 events'}")
    hdr = row("metric", ["ALL"] + [BEH[b] for b in range(nb)])
    print("\n" + hdr)
    print(row("coverage@pool", [fmt(x) for x in r["cov"]]))
    print(row("recall@10", [fmt(x) for x in r["recall10"]]))
    print(row("P(r<10|cov)", [fmt(x) for x in r["p_cov"]]))
    print("  coverage-loss decomposition (mutually exclusive, pruning order):")
    for nm in list(r["decomp_names"]) + ["trunc"]:
        print(row("  " + nm, [fmt(x) for x in r["decomp"][nm]]))
    print("  pool diversity (per-event union pool):")
    print(row("  uniq l0", [fmt(x, 1) for x in r["div"]["uniq_l0"]]))
    print(row("  uniq 2-prefix", [fmt(x, 1) for x in r["div"]["uniq_pre2"]]))
    print(row("  largest l0 grp", [fmt(x, 1) for x in r["div"]["largest_grp"]]))


def headline(pts):
    bar("HEADLINE — matched decoder compute (four rows)")
    cols = ["fwd_st", "cov@pool", "recall@10", "P(r<10|c)", "uniqL0", "uniqP2", "sec/1k"]
    print(row("point", cols, w=11, lw=14))
    for r in pts:
        cells = [str(r["fwd_states"]), fmt(r["cov"][0]), fmt(r["recall10"][0]), fmt(r["p_cov"][0]),
                 fmt(r["div"]["uniq_l0"][0], 1), fmt(r["div"]["uniq_pre2"][0], 1),
                 fmt(r["sec_per_1k"], 1) if r["sec_per_1k"] is not None else "n/a"]
        print(row(r["label"], cells, w=11, lw=14))

    nb = pts[0]["nb"]
    print("\n  coverage@pool per behavior:")
    print(row("point", ["ALL"] + [BEH[b] for b in range(nb)], lw=14))
    for r in pts:
        print(row(r["label"], [fmt(x) for x in r["cov"]], lw=14))
    print("\n  recall@10 per behavior:")
    print(row("point", ["ALL"] + [BEH[b] for b in range(nb)], lw=14))
    for r in pts:
        print(row(r["label"], [fmt(x) for x in r["recall10"]], lw=14))


def paid_vs_free(pmap):
    """MFD@808 -> MFD@808+free (free K_t/K_i) -> MFD@4008 (paid K_f). Needs the three MFD points."""
    need = ["MFD@808", "MFD@808+free", "MFD@4008"]
    if not all(k in pmap for k in need):
        print("\n  [paid-vs-free decomposition skipped: need MFD@808, MFD@808+free, MFD@4008]")
        return
    bar("MFD PAID (K_f) vs FREE (K_t,K_i) decomposition of the 808->4008 move")
    a, b, c = (pmap[k] for k in need)
    print("  All three MFD points, ALL-behavior coverage@pool / recall@10:")
    print(f"    MFD@808       (Kf=200,  Kt=200, Ki=50 )  fwd={a['fwd_states']:<5}  "
          f"cov={a['cov'][0]:.4f}  r@10={a['recall10'][0]:.4f}")
    print(f"    MFD@808+free  (Kf=200,  Kt/Ki=full   )  fwd={b['fwd_states']:<5}  "
          f"cov={b['cov'][0]:.4f}  r@10={b['recall10'][0]:.4f}   [FREE lever: +0 decoder states]")
    print(f"    MFD@4008      (Kf=1000, Kt/Ki=full   )  fwd={c['fwd_states']:<5}  "
          f"cov={c['cov'][0]:.4f}  r@10={c['recall10'][0]:.4f}   [PAID: Kf 200->1000]")
    print(f"\n  cov  delta:  free-lever (808->808+free) = {b['cov'][0]-a['cov'][0]:+.4f}"
          f"   |   paid-Kf (808+free->4008) = {c['cov'][0]-b['cov'][0]:+.4f}"
          f"   |   total (808->4008) = {c['cov'][0]-a['cov'][0]:+.4f}")
    print(f"  r@10 delta:  free-lever = {b['recall10'][0]-a['recall10'][0]:+.4f}"
          f"   |   paid-Kf = {c['recall10'][0]-b['recall10'][0]:+.4f}"
          f"   |   total = {c['recall10'][0]-a['recall10'][0]:+.4f}")


def interpret(pmap):
    bar("INTERPRETATION (reported, not tuned toward)")
    if "chain@808" in pmap and "MFD@808" in pmap:
        d = pmap["MFD@808"]["cov"][0] - pmap["chain@808"]["cov"][0]
        verdict = ("MFD reaches HIGHER coverage than chain at equal 808-state compute -> efficiency result: "
                   "factorized wins per unit compute." if d > 0.002 else
                   "chain matches or beats MFD coverage at equal 808-state compute -> the MFD edge is "
                   "NOT a per-unit-compute win at this budget." if d < -0.002 else
                   "MFD and chain are within noise at equal 808-state compute -> comparable coverage per unit compute.")
        print(f"  [equal budget 808]  cov MFD-chain = {d:+.4f}.  {verdict}")
    if "MFD@808" in pmap and "MFD@4008" in pmap:
        d = pmap["MFD@4008"]["cov"][0] - pmap["MFD@808"]["cov"][0]
        verdict = ("MFD keeps climbing with more compute -> more coverage per unit compute." if d > 0.005 else
                   "MFD@4008 barely improves on MFD@808 -> MFD SATURATES early; the honest framing is "
                   "'same coverage, far cheaper', not 'more coverage per unit compute'.")
        print(f"  [MFD scaling 808->4008]  cov delta = {d:+.4f}.  {verdict}")
    if "chain@808" in pmap and "chain@4008" in pmap:
        d = pmap["chain@4008"]["cov"][0] - pmap["chain@808"]["cov"][0]
        print(f"  [chain scaling 808->4008]  cov delta = {d:+.4f}  (5x compute).")
    print("  Read coverage@pool per behavior (esp. pv, where the loss concentrates) alongside the "
          "decomposition:\n  whichever branch dominates the loss at each budget is where added compute would pay off.")


def main():
    ap = argparse.ArgumentParser()
    # each --point is: LABEL ARM PATH   (ARM in {MFD, chain})
    ap.add_argument("--point", nargs=3, action="append", metavar=("LABEL", "ARM", "PATH"), required=True)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--order", default="chain@808,chain@4008,MFD@808,MFD@4008",
                    help="comma list of labels for the four-row headline table")
    args = ap.parse_args()

    pmap = {}
    for label, arm, path in args.point:
        assert arm in ("MFD", "chain"), f"arm must be MFD|chain, got {arm}"
        pmap[label] = compute(label, arm, path, args.wb)

    bar("MATCHED-COMPUTE POOL COMPARISON — chain sequential vs MFD factorized")
    print("  decoder-state model:  chain = 4 + nb*(1+2*Bw)   |   MFD = nb*(2+Kf)  (K_t,K_i are free)")
    for label in pmap:
        r = pmap[label]
        print(f"    {label:<14} {r['arm']:<6} events={r['N']:,}  {r['config']}  fwd={r['fwd_states']}")

    for r in pmap.values():
        print_point(r)

    order = [x for x in args.order.split(",") if x in pmap]
    headline([pmap[x] for x in order])
    paid_vs_free(pmap)
    interpret(pmap)


if __name__ == "__main__":
    main()