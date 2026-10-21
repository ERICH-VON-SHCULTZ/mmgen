"""Matched head-to-head: the SAME listwise reranker, trained the SAME way, on the MFD pool and on
the chain pool. Question: is the reranker's GAIN larger on MFD's wider pool?

recall@10 = coverage x P(rank<10|cov). MFD covers more but orders worse, so it holds a bigger stock
of recoverable GTs (covered-but-outside-top-10: MFD 0.0537 of events vs chain 0.0351, +53%). If the
history/engagement signal recovers a similar FRACTION on both sides, MFD gains more absolutely and
can overtake. If it mostly recovers GTs BOTH pools already cover (repeat items), the deltas match and
MFD's wider pool buys nothing. The grouped breakdown decides this -- not the headline delta.

All evaluation lives HERE so both arms use byte-identical metric code. Models are rebuilt from their
saved checkpoints; the generative models stay frozen throughout.
"""
import argparse, os, sys
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))
import numpy as np, torch
from train_mfd_reranker import (BEH, MFDReranker, RerankData, best_rank, hist_batch, load_geometry,
                                make_repr, ndcg_at, recall_at)

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def log(m, fh=None):
    print(m, flush=True)
    if fh: fh.write(m + "\n"); fh.flush()


def widths_from_sid(sid_path):
    o = torch.load(sid_path, map_location="cpu", weights_only=False)
    sid = o if torch.is_tensor(o) else next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2)
    if sid.shape[0] > sid.shape[1]: sid = sid.t().contiguous()
    return [int(sid[l].max()) + 1 for l in range(sid.shape[0])]


def rebuild(ckpt_path, data, device, fh):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]; d_model = int(data.meta.get("d_model", data.mem.shape[2]))
    if a.get("geom", "md") == "random":
        w = widths_from_sid(a["sid_path"]) if a.get("sid_path") else \
            [int(data.f.max()) + 1, int(data.t.max()) + 1, int(data.i.max()) + 1]
        g = torch.Generator().manual_seed(a["seed"] + 4242)
        U_f = (torch.randn(w[0], d_model, generator=g) * 0.02).to(device)
        U_t = (torch.randn(w[1], d_model, generator=g) * 0.02).to(device)
        U_i = (torch.randn(w[2], d_model, generator=g) * 0.02).to(device)
        log(f"    geometry=random  widths={w}", fh)
    else:
        U_f, U_t, U_i = load_geometry(a["mdrqvae_ckpt"], a["centroid_cache"], d_model, a["seed"], device)
        log(f"    geometry=md  widths={[U_f.shape[0], U_t.shape[0], U_i.shape[0]]}", fh)
    for nm, col, tab in (("l0", "f", U_f), ("l1", "t", U_t), ("l2", "i", U_i)):
        mx = int(getattr(data, col).max())
        assert mx < tab.shape[0], f"{nm} code {mx} >= width {tab.shape[0]}"
    m = MFDReranker(U_f, U_t, U_i, d_model, layers=a["layers"], heads=a["heads"], drop=a["dropout"],
                    num_behaviors=int(data.meta.get("nb", 4)), use_hard=a["use_hard"], use_soft=a["use_soft"],
                    cand_repr=a.get("cand_repr", "md3")).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


@torch.no_grad()
def score_all(model, data, device, bs=64):
    out = torch.empty(data.N, data.C)
    for s in range(0, data.N, bs):
        e = slice(s, min(s + bs, data.N)); idx = torch.arange(e.start, e.stop)
        f = data.f[idx].long().to(device); t = data.t[idx].long().to(device); i = data.i[idx].long().to(device)
        valid = data.valid[idx].to(device); base = data.base[idx].float().to(device)
        lp = data.lp[idx].float().to(device); mem = data.mem[idx].float().to(device)
        mp = data.mem_pad[idx].to(device)
        h = hist_batch(data, idx, device) if (model.use_hard or model.use_soft) else None
        out[e] = model(f, t, i, lp, base, valid, mem, mp, h).cpu()
    return out


def ranks_of(score, data):
    sid = (data.f == data.gt_f[:, None]) & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None])
    beh = data.branch.view(1, -1) == data.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, data.valid)
    r2, h2 = best_rank(score, sid, data.valid & beh)
    return r3, h3, r2, h2


def mech(score, data):
    cov = data.has_gt
    gt_is = ((data.branch.view(1, -1) == data.gt_b[:, None]) & (data.f == data.gt_f[:, None])
             & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]) & data.valid)
    gts = torch.where(gt_is, score, torch.full_like(score, float("-inf"))).max(1).values
    srt = score.sort(1, descending=True).values
    gap = (srt[:, 9] - srt[:, 10])[cov].mean().item()
    band = (((score >= gts.unsqueeze(1) - .5) & (score <= gts.unsqueeze(1) + .5) & data.valid).sum(1))[cov].float().mean().item()
    return gap, band


def full(tag, data, score, fh):
    r3, h3, r2, h2 = ranks_of(score, data); cov = data.has_gt; gt_b = data.gt_b
    gap, band = mech(score, data)
    log(f"\n  [{tag}]", fh)
    log(f"    t3 R@5={recall_at(r3,h3,5):.4f} R@10={recall_at(r3,h3,10):.4f} R@20={recall_at(r3,h3,20):.4f} "
        f"N@5={ndcg_at(r3,h3,5):.4f} N@10={ndcg_at(r3,h3,10):.4f}", fh)
    log(f"    t2 R@5={recall_at(r2,h2,5):.4f} R@10={recall_at(r2,h2,10):.4f} R@20={recall_at(r2,h2,20):.4f} "
        f"N@5={ndcg_at(r2,h2,5):.4f} N@10={ndcg_at(r2,h2,10):.4f}", fh)
    log(f"    P(rank<10|cov)={recall_at(r3,h3,10,cov):.4f}  coverage@pool={cov.float().mean().item():.4f}  "
        f"gap10-11={gap:.4f}  #within0.5={band:.2f}", fh)
    log(f"    t3@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r3,h3,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    log(f"    t2@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r2,h2,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    return {"r3": r3, "h3": h3, "t3@10": recall_at(r3, h3, 10), "pcov": recall_at(r3, h3, 10, cov),
            "gap": gap, "band": band, "cov": cov}


KF_MAX = 2048


def entropy_buckets(data, b, r, dev, fh):
    """MFD-only: bucket P(rank<10|cov) before/after reranking by the GT branch's entropy z within its
    event. If the reranker lifts the FLAT buckets, the extra signal reaches the low-confidence region
    where MFD's extra coverage lives; if only the peaked buckets lift, the wider pool is not converted."""
    N, C, nb = data.N, data.C, data.nb
    ev = torch.arange(N).view(N, 1).expand(N, C)
    gkey = ev.to(torch.int64) * (nb * KF_MAX) + data.branch.view(1, -1).expand(N, C) * KF_MAX + data.f.long()
    v = data.valid
    gk = gkey[v].to(dev); xt = data.lp[..., 2].float()[v].to(dev); xi = data.lp[..., 3].float()[v].to(dev)
    pt = data.t.long()[v].to(dev); pi = data.i.long()[v].to(dev)
    uniq, gid = torch.unique(gk, return_inverse=True); G = uniq.numel()
    def H(x, code, K):
        tk = gid * K + code
        utk, gtd = torch.unique(tk, return_inverse=True)
        lp = torch.full((utk.numel(),), -1e30, device=dev).scatter_reduce(0, gtd, x, reduce="amax", include_self=True)
        bb = (utk // K).long(); P = lp.exp()
        Z = torch.zeros(G, device=dev).index_add_(0, bb, P); p = P / Z[bb].clamp(min=1e-12)
        return torch.zeros(G, device=dev).index_add_(0, bb, -p * p.clamp(min=1e-12).log())
    Hs = H(xt, pt, 1024) + H(xi, pi, 1024)
    eg = (uniq // (nb * KF_MAX)).long()
    n = torch.zeros(N, device=dev).index_add_(0, eg, torch.ones_like(Hs))
    mu = torch.zeros(N, device=dev).index_add_(0, eg, Hs) / n.clamp(min=1)
    ss = torch.zeros(N, device=dev).index_add_(0, eg, Hs * Hs)
    sd = (ss / n.clamp(min=1) - mu.pow(2)).clamp(min=0).sqrt()
    gt_key = (torch.arange(N) * (nb * KF_MAX) + data.gt_b * KF_MAX + data.gt_f).to(dev)
    pos = torch.searchsorted(uniq, gt_key)
    avail = (pos < G) & (uniq[pos.clamp(max=G - 1)] == gt_key)
    z = ((Hs[pos.clamp(max=G - 1)] - mu) / sd.clamp(min=1e-6)).cpu()
    cov = data.has_gt & avail.cpu()
    log("\n  MFD branch-entropy buckets (GT branch entropy z within its event; flat = high z):", fh)
    log(f"  {'z bucket':<12}{'n':>9}{'top10 before':>14}{'top10 after':>13}{'gain':>9}", fh)
    edges = [(-1e9, -1.0, "<-1"), (-1.0, -0.5, "-1..-.5"), (-0.5, 0.0, "-.5..0"),
             (0.0, 0.5, "0..0.5"), (0.5, 1.0, "0.5..1"), (1.0, 1e9, ">1")]
    for lo, hi, nm in edges:
        m = cov & (z >= lo) & (z < hi)
        k = int(m.sum())
        if k == 0: continue
        pb = recall_at(b["r3"], b["h3"], 10, m); pa = recall_at(r["r3"], r["h3"], 10, m)
        log(f"  {nm:<12}{k:>9,}{pb:>14.4f}{pa:>13.4f}{pa-pb:>+9.4f}", fh)


def assert_split(data, tag, fh):
    per = {b: int((data.gt_b == b).sum()) for b in range(data.nb)}
    ok = data.N == EXPECT_TOTAL and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  [{tag}] split total={data.N:,} " + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH)
        + f" -> {'OK' if ok else 'MISMATCH'}", fh)
    if not ok: raise SystemExit(f"** {tag} split mismatch -- stop. **")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_test", required=True); ap.add_argument("--chain_test", required=True)
    ap.add_argument("--mfd_ckpt", required=True); ap.add_argument("--chain_ckpt", required=True)
    ap.add_argument("--mfd_ckpt_sub", default=""); ap.add_argument("--chain_ckpt_sub", default="")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/matched_rerank")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")

    log("=" * 92 + "\n=== MATCHED HEAD-TO-HEAD: same reranker on the MFD pool vs the chain pool ===\n" + "=" * 92, fh)
    mfd = RerankData(args.mfd_test, args.wb); ch = RerankData(args.chain_test, args.wb)
    assert_split(mfd, "MFD", fh); assert_split(ch, "chain", fh)
    log(f"  decoder-state evals/item: MFD={mfd.meta.get('fwd_states_per_item','?')}  "
        f"chain={ch.meta.get('fwd_states_per_item','?')}  (must both be 808)", fh)
    log(f"  K_pool: MFD={mfd.Kp} chain={ch.Kp}", fh)

    # exact cross-arm alignment: same events, same order
    um = torch.load(args.mfd_test, map_location="cpu", weights_only=False)["cols"]["user_id"]
    uc = torch.load(args.chain_test, map_location="cpu", weights_only=False)["cols"]["user_id"]
    same_u = bool(torch.equal(um, uc)); same_b = bool(torch.equal(mfd.gt_b, ch.gt_b))
    log(f"  cross-arm alignment: user_id identical={same_u}  gt_behavior identical={same_b}", fh)
    if not (same_u and same_b):
        raise SystemExit("** arms are not event-aligned -- the grouped breakdown would be wrong. **")

    settings = [("natural", args.mfd_ckpt, args.chain_ckpt)]
    if args.mfd_ckpt_sub and args.chain_ckpt_sub:
        settings.append(("subsampled(equal train data)", args.mfd_ckpt_sub, args.chain_ckpt_sub))

    for sname, mck, cck in settings:
        log("\n" + "#" * 92 + f"\n# SETTING: {sname}\n" + "#" * 92, fh)
        res = {}
        for arm, data, ckp in (("MFD", mfd, mck), ("chain", ch, cck)):
            log(f"\n  --- arm: {arm} ---", fh)
            model, ck = rebuild(ckp, data, dev, fh)
            log(f"    trained on {ck.get('args',{}).get('_n_train_events','?')} events; "
                f"best val t3@10={ck.get('val_t3_recall_10',float('nan')):.4f} @ep{ck.get('epoch','?')}", fh)
            b = full(f"{arm} BEFORE rerank (union score)", data, data.base.float(), fh)
            sc = score_all(model, data, dev)
            r = full(f"{arm} AFTER rerank", data, sc, fh)
            assert abs(b["cov"].float().mean() - r["cov"].float().mean()) < 1e-9
            log(f"    coverage unchanged by reranking: OK ({b['cov'].float().mean().item():.4f})", fh)
            log(f"    GAIN t3@10 = {r['t3@10']-b['t3@10']:+.4f}   P(rank<10|cov) {b['pcov']:.4f} -> {r['pcov']:.4f}", fh)
            res[arm] = (b, r)

        # ---------------- decisive breakdown ----------------
        log("\n  " + "=" * 88 + "\n  DECISIVE BREAKDOWN\n  " + "=" * 88, fh)
        cm, cc = res["MFD"][0]["cov"], res["chain"][0]["cov"]
        both, only_m, only_c = cm & cc, cm & ~cc, cc & ~cm
        log(f"  coverage groups: both={int(both.sum()):,}  only-MFD={int(only_m.sum()):,}  "
            f"only-chain={int(only_c.sum()):,}  (event-aligned, exact)", fh)
        log(f"\n  {'arm/group':<26}{'n':>9}{'top10 before':>14}{'top10 after':>13}{'gain':>9}", fh)
        for arm, grp, gname in (("MFD", both, "both-covered"), ("MFD", only_m, "ONLY-MFD"),
                                ("chain", both, "both-covered"), ("chain", only_c, "ONLY-chain")):
            b, r = res[arm]; n = int(grp.sum())
            if n == 0: continue
            pb = recall_at(b["r3"], b["h3"], 10, grp); pa = recall_at(r["r3"], r["h3"], 10, grp)
            log(f"  {arm+' / '+gname:<26}{n:>9,}{pb:>14.4f}{pa:>13.4f}{pa-pb:>+9.4f}", fh)
        log("  READ: if the gain concentrates on 'both-covered' (repeat items), MFD's wider pool buys", fh)
        log("  nothing. If ONLY-MFD converts at a similar rate, the extra coverage is being realised.", fh)

        log(f"\n  by BASELINE rank band (covered events):", fh)
        log(f"  {'arm/band':<26}{'n':>9}{'-> top10 after':>16}", fh)
        for arm in ("MFD", "chain"):
            b, r = res[arm]
            for lo, hi, nm in ((10, 50, "10-49"), (50, 10 ** 9, "50+")):
                grp = b["cov"] & (b["r3"] >= lo) & (b["r3"] < hi)
                n = int(grp.sum())
                if n == 0: continue
                log(f"  {arm+' / '+nm:<26}{n:>9,}{recall_at(r['r3'],r['h3'],10,grp):>16.4f}", fh)

        entropy_buckets(mfd, res["MFD"][0], res["MFD"][1], dev, fh)

        log(f"\n  per behavior, t3@10 gain:", fh)
        for arm in ("MFD", "chain"):
            b, r = res[arm]
            log(f"    {arm:<6} " + "  ".join(
                f"{BEH[x]}: {recall_at(b['r3'],b['h3'],10,mfd.gt_b==x):.4f}->{recall_at(r['r3'],r['h3'],10,mfd.gt_b==x):.4f}"
                for x in range(mfd.nb) if (mfd.gt_b == x).any()), fh)

        gm = res["MFD"][1]["t3@10"] - res["MFD"][0]["t3@10"]
        gc = res["chain"][1]["t3@10"] - res["chain"][0]["t3@10"]
        fm, fc = res["MFD"][1]["t3@10"], res["chain"][1]["t3@10"]
        log(f"\n  SUCCESS CRITERION (fixed in advance): MFD gain > chain gain AND MFD+rerank > chain+rerank.", fh)
        log(f"    gain: MFD {gm:+.4f} vs chain {gc:+.4f}   |   final t3@10: MFD {fm:.4f} vs chain {fc:.4f}", fh)
        verdict = ("SUPPORTED" if (gm > gc and fm > fc) else
                   "NOT SUPPORTED -- chain+reranker >= MFD+reranker; this closes the line of investigation"
                   if fc >= fm else "PARTIAL -- MFD wins on final but not on gain (or vice versa)")
        log(f"    VERDICT: {verdict}", fh)
    fh.close()


if __name__ == "__main__":
    main()
