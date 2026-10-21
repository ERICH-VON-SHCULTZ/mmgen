"""Simulate, offline, whether reallocating the per-branch candidate budget by branch entropy would
improve recall. No regeneration, no training -- a feasibility CEILING estimate.

MFD prunes each f-branch to top K_t=200 text / K_i=50 image codes uniformly. Flat (high-entropy)
branches are where the GT is likely pruned before scoring. Hypothesis: move budget from peaked
branches (top-1 already right, extra slots wasted) to flat branches, at fixed TOTAL budget.

What the dump gives (exact): tf_t_rank / tf_i_rank = the GT's t*/i* rank WITHIN its f-branch (full
head), f_star_rank, and the GT's logps. So GT survives reallocated branch pruning iff
tf_t_rank < K_t'(its branch) and tf_i_rank < K_i'(its branch) and f_star_rank < K_f -- exact.
Branch entropy is the TRUNCATED entropy of the branch's pooled candidates (full head not dumped);
available only for branches present in the pool (excluded fraction reported).

STAGE 1  coverage ceiling: survival rate per allocation rule (proportional to H, to exp(H), to
         entropy rank, and a two-tier flat/peaked multiplier sweep), budget preserved per event.
STAGE 2  the decisive question: rank each NEWLY-COVERED GT with the CURRENT score against the pool
         (OPTIMISTIC -- reallocation would add unseen flat-branch competitors). If < ~20% reach
         top-10, it repeats "coverage without recall" -> not worth implementing. Net recall@10 too.

Do NOT modify the score (z-standardization already failed). Do NOT tune toward the 20% criterion.
Asserts test split 83,005 (pv/fav/cart/buy 77,833/2,042/1,554/1,576). Refs: recall@10 0.2041 hand-set
/ 0.2065 fusion; coverage 0.2615.
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

from train_mfd_reranker import BEH, best_rank, recall_at

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
KF_MAX = 2048


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


def within_group_rank(keys, vals):
    n = keys.numel()
    o1 = torch.argsort(vals, stable=True)
    order = o1[torch.argsort(keys[o1], stable=True)]
    ks = keys[order]
    is_start = torch.ones(n, dtype=torch.bool); is_start[1:] = ks[1:] != ks[:-1]
    idxs = torch.arange(n)
    grp_start = torch.cummax(torch.where(is_start, idxs, torch.zeros(n, dtype=torch.long)), 0).values
    rank = torch.empty(n, dtype=torch.long); rank[order] = idxs - grp_start
    _, inv, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
    return rank.float() / (cnt[inv].float() - 1).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dump", required=True)
    ap.add_argument("--wb", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/branch_realloc"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")
    torch.manual_seed(args.seed)

    d = torch.load(args.test_dump, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    nb = meta["nb"]; N, _, Kp = c["pool_f"].shape
    Kf, Kt, Ki = meta.get("Kf", 200), meta.get("Kt", 200), meta.get("Ki", 50)
    f32 = lambda x: x.float() if x.dtype == torch.float16 else x
    flat = lambda x: x.reshape(N, nb * Kp)
    gt_b, gt_f = c["gt_b"].long(), c["gt_f"].long()
    per = {b: int((gt_b == b).sum()) for b in range(nb)}
    ok = (N == EXPECT_TOTAL) and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  split total={N:,} " + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH) + f" -> {'OK' if ok else 'MISMATCH'}", fh)
    if not ok:
        raise SystemExit("** split mismatch -- stop. **")
    log(f"  budget: K_f={Kf} K_t={Kt} K_i={Ki} K_pool={Kp}", fh)

    pf, pt, pi = flat(c["pool_f"]).long(), flat(c["pool_t"]).long(), flat(c["pool_i"]).long()
    valid = f32(flat(c["pool_sidscore"])) > -1e30
    branch = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1)
    sidscore = f32(flat(c["pool_sidscore"])); lp_t = f32(flat(c["pool_lpt"])); lp_i = f32(flat(c["pool_lpi"]))
    beh = f32(c["beh_logp"])
    fsr, ttr, tir = c["f_star_rank"].long(), c["tf_t_rank"].long(), c["tf_i_rank"].long()
    fsl, ttl, til = f32(c["f_star_logp"]), f32(c["tf_t_logp"]), f32(c["tf_i_logp"])
    gt_is = ((branch == gt_b[:, None]) & (pf == gt_f[:, None]) & (pt == c["gt_t"].long()[:, None])
             & (pi == c["gt_i"].long()[:, None]) & valid)
    has_gt = gt_is.any(1)

    # -------- per-(event,branch) truncated entropy from pooled candidates --------
    ev = torch.arange(N).view(N, 1).expand(N, nb * Kp)
    gkey = (ev.to(torch.int64) * (nb * KF_MAX) + branch * KF_MAX + pf)
    v = valid
    gk = gkey[v].to(dev); xt = lp_t[v].to(dev); xi = lp_i[v].to(dev); ptv = pt[v].to(dev); piv = pi[v].to(dev)
    uniq, gid = torch.unique(gk, return_inverse=True); G = uniq.numel()
    cnt = torch.zeros(G, device=dev).index_add_(0, gid, torch.ones_like(xt))
    def entropy(x, code, K):
        tk = gid * K + code
        utk, gtd = torch.unique(tk, return_inverse=True)
        lp = torch.full((utk.numel(),), -1e30, device=dev).scatter_reduce(0, gtd, x, reduce="amax", include_self=True)
        b_ = (utk // K).long(); P = lp.exp()
        Z = torch.zeros(G, device=dev).index_add_(0, b_, P); p = P / Z[b_].clamp(min=1e-12)
        H = torch.zeros(G, device=dev).index_add_(0, b_, -p * p.clamp(min=1e-12).log())
        return H, H.exp()
    H_t, eff_t = entropy(xt, ptv, 1024); H_i, eff_i = entropy(xi, piv, 1024)
    Hsum = H_t + H_i
    ev_of_g = (uniq // (nb * KF_MAX)).long()
    n_br = torch.zeros(N, device=dev).index_add_(0, ev_of_g, torch.ones_like(Hsum))
    ev_mean = torch.zeros(N, device=dev).index_add_(0, ev_of_g, Hsum) / n_br.clamp(min=1)
    ev_ss = torch.zeros(N, device=dev).index_add_(0, ev_of_g, Hsum * Hsum)
    ev_std = (ev_ss / n_br.clamp(min=1) - ev_mean.pow(2)).clamp(min=0).sqrt()
    erank = within_group_rank(ev_of_g.cpu(), Hsum.cpu()).to(dev) + 1e-3       # entropy rank within event

    # -------- locate the GT's branch group --------
    gt_gkey = (torch.arange(N) * (nb * KF_MAX) + gt_b * KF_MAX + gt_f).to(dev)
    pos = torch.searchsorted(uniq, gt_gkey)
    avail = (pos < G) & (uniq[pos.clamp(max=G - 1)] == gt_gkey)              # GT branch present in pool
    gg = pos.clamp(max=G - 1)
    log(f"\n  GT-branch entropy available for {int(avail.sum()):,}/{N:,} events ({avail.float().mean():.4f}); "
        f"rest (GT branch not in pool) excluded from the simulation.", fh)

    fsr_d, ttr_d, tir_d = fsr.to(dev), ttr.to(dev), tir.to(dev)
    surv_uni = (fsr_d < Kf) & (ttr_d < Kt) & (tir_d < Ki)
    ev_i = torch.arange(N, device=dev)

    def kprime(wt, wi):                                                      # budget-preserving per-event
        sw_t = torch.zeros(N, device=dev).index_add_(0, ev_of_g, wt); sw_i = torch.zeros(N, device=dev).index_add_(0, ev_of_g, wi)
        Ktg = (n_br * Kt) * wt[gg] / sw_t[ev_i].clamp(min=1e-9)
        Kig = (n_br * Ki) * wi[gg] / sw_i[ev_i].clamp(min=1e-9)
        return Ktg.round().clamp(1, 1024), Kig.round().clamp(1, 1024)

    def twotier(thr, m):
        z = (Hsum - ev_mean[ev_of_g]) / ev_std[ev_of_g].clamp(min=1e-6)
        flatg = (z > thr).float()
        n_flat = torch.zeros(N, device=dev).index_add_(0, ev_of_g, flatg)
        n_peak = (n_br - n_flat).clamp(min=1e-6)
        p_ev = ((n_br - n_flat * m) / n_peak).clamp(min=0.05)                # peaked multiplier (budget preserving)
        is_flat_gt = (z[gg] > thr)
        mult = torch.where(is_flat_gt, torch.full_like(p_ev, m)[ev_i], p_ev[ev_i])
        return (Kt * mult).round().clamp(1, 1024), (Ki * mult).round().clamp(1, 1024)

    rules = {"prop-H": kprime(H_t, H_i), "prop-expH": kprime(eff_t, eff_i), "prop-Hrank": kprime(erank, erank)}
    for thr in (0.0, 0.5, 1.0):
        for m in (1.5, 2.0, 3.0):
            rules[f"2tier(z>{thr},x{m})"] = twotier(thr, m)

    # -------- Stage 2 prep: GT score vs current pool --------
    gt_sid = (fsl + ttl + til).to(dev)
    gt_union = (args.wb * beh.gather(1, gt_b.unsqueeze(1)).squeeze(1)).to(dev) + gt_sid
    pool_union = (args.wb * torch.gather(beh, 1, branch) + sidscore)
    r_cur, h_cur = best_rank(pool_union, (pf == gt_f[:, None]) & (pt == c["gt_t"].long()[:, None])
                             & (pi == c["gt_i"].long()[:, None]) & (branch == gt_b[:, None]), valid)
    pu = pool_union.to(dev); vd = valid.to(dev)
    r_new = ((pu > gt_union.unsqueeze(1)) & vd).sum(1)                       # GT rank vs CURRENT pool (optimistic)
    cov = has_gt.to(dev); rcur = r_cur.to(dev); gtb = gt_b.to(dev)
    base_r10 = float((cov & (rcur < 10)).float().mean())

    log("\n" + "=" * 90 + "\n=== STAGE 1: coverage ceiling (branch-survival; K_pool truncation NOT applied) ===\n" + "=" * 90, fh)
    log(f"  uniform branch-survival rate = {float((surv_uni & avail).float().mean() + (surv_uni & ~avail).float().mean()):.4f}"
        f"  |  actual coverage@pool (has_gt) = {float(has_gt.float().mean()):.4f} (0.2615, K_pool-truncated)", fh)
    log(f"  {'rule':<20}{'surv(ceil)':>11}{'newlyRec':>10}{'newlyLost':>11}{'net':>8}", fh)
    stage2 = {}
    for name, (Ktg, Kig) in rules.items():
        surv = avail & (fsr_d < Kf) & (ttr_d < Ktg.long()) & (tir_d < Kig.long())
        newly_rec = surv & (~surv_uni)
        newly_lost = avail & surv_uni & (~surv)
        ceil = float((surv | (surv_uni & ~avail)).float().mean())           # excluded events keep uniform survival
        log(f"  {name:<20}{ceil:>11.4f}{int(newly_rec.sum()):>10,}{int(newly_lost.sum()):>11,}"
            f"{int(newly_rec.sum())-int(newly_lost.sum()):>8,}", fh)
        stage2[name] = (newly_rec, newly_lost)

    log("\n" + "=" * 90 + "\n=== STAGE 2: where do newly-covered GTs rank? (current score, OPTIMISTIC vs current pool) ===\n" + "=" * 90, fh)
    log("  NOTE: real reallocated pool would add UNSEEN flat-branch competitors -> true rank is DEEPER.", fh)
    log("  So these top-k fractions are an UPPER bound; if they fail the 20% bar here, they fail for real.", fh)
    for name, (newly_rec, newly_lost) in stage2.items():
        nr = int(newly_rec.sum())
        if nr == 0:
            log(f"\n  [{name}] no newly-covered GTs.", fh); continue
        rr = r_new[newly_rec]
        f10 = float((rr < 10).float().mean()); f50 = float((rr < 50).float().mean())
        # net recall@10: keep covered<10 unless newly-lost; add newly-rec<10
        lost10 = cov & (rcur < 10) & newly_lost
        keep = cov & (rcur < 10) & (~newly_lost)
        rule_r10 = float((keep.float().sum() + (newly_rec & (r_new < 10)).float().sum()) / N)
        log(f"\n  [{name}]  newlyRec={nr:,}  top10={f10:.3f} top50={f50:.3f} beyond50={1-f50:.3f}", fh)
        log(f"    net recall@10 = {rule_r10:.4f}  (vs current {base_r10:.4f}; refs 0.2041 hand / 0.2065 fusion)  "
            f"newly-lost that were top10 = {int(lost10.sum()):,}", fh)
        verdict = "PASS (>=20% reach top10)" if f10 >= 0.20 else "FAIL -- coverage without recall; not worth implementing"
        log(f"    DECISION: {f10*100:.1f}% of newly-covered reach top10 -> {verdict}", fh)
        # per behavior
        pb = "  ".join(f"{BEH[b]}: rec={int((newly_rec&(gtb==b)).sum())} top10={float((r_new[newly_rec&(gtb==b)]<10).float().mean()) if int((newly_rec&(gtb==b)).sum())>0 else float('nan'):.3f}" for b in range(nb))
        log(f"    per-beh: {pb}", fh)

    log("\n=== VERDICT ===", fh)
    log("  Stage 1 shows the coverage ceiling; Stage 2 is decisive. If the best rule's newly-covered", fh)
    log("  top-10 fraction is < ~20% (optimistically), budget reallocation repeats the coverage-without-", fh)
    log("  recall pattern and is not worth implementing. Reported as the ceiling; not tuned.", fh)
    fh.close()


if __name__ == "__main__":
    main()
