"""Test whether MFD's in-pool ordering deficit is a SCORE COMPARABILITY problem across f-branches,
and fix it by calibrating branch scores before merging. Pure offline rescoring on existing dumps.

MFD scores logP(f)+logP(t|f)+logP(i|f) where the two conditionals are INDEPENDENT per-f softmaxes.
A peaked branch (few items) and a flat branch (crowded) put their logP on different scales, so
summing raw log-probs across branches compares incommensurate numbers. Chain's logP(i|f,t) is one
joint factorization, natively commensurate.

STAGE 1 diagnostic (run first, decides go/no-go): per f-branch, entropy / top-1 / effective count /
size of P(t|f) and P(i|f); within-event spread of branch entropy; and the key test -- is the GT
pushed out of top-10 more often when ITS branch is FLAT (high entropy) relative to the other
branches in the same pool? P(rank<10|cov) bucketed by the GT branch's within-event entropy
percentile; a monotone decline confirms the hypothesis. (Chain dump lacks per-level conditional
logps, so its analog is a best-effort l0-group score-dispersion proxy -- stated as such.)

STAGE 2 three calibration variants (coverage invariant): (a) within-branch standardization (z-score
+ rank-based), (b) entropy-driven temperature tau_f, (c) learned branch MLP -> (a_f,b_f). Winner is
also combined with the adopted learned fusion weights (mfd_fusion_weights.pt).

Asserts: test split 83,005 (pv/fav/cart/buy 77,833/2,042/1,554/1,576); coverage@pool 0.2615 invariant.
Reference (do NOT tune toward): hand-set t3@10 0.2042, fusion 0.2065, MADR 0.2078, listwise 0.2150,
chain 0.2120. P(rank<10|cov) current 0.7945, chain 0.8581, break-even 0.81.
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mfd_reranker import BEH, best_rank, ndcg_at, recall_at

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
KF_MAX = 2048


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


class Data:
    """Flat per-candidate component tensors from an MFD (or chain) pool dump. No enc_mem needed."""

    def __init__(self, dump_path, wb):
        d = torch.load(dump_path, map_location="cpu", weights_only=False)
        c, meta = d["cols"], d["meta"]
        self.meta = meta; nb = meta["nb"]; N, _, Kp = c["pool_f"].shape
        self.N, self.nb, self.Kp, self.C, self.wb = N, nb, Kp, nb * Kp, wb
        f32 = lambda x: x.float() if x.dtype == torch.float16 else x
        flat = lambda x: x.reshape(N, nb * Kp)
        self.pf, self.pt, self.pi = flat(c["pool_f"]).long(), flat(c["pool_t"]).long(), flat(c["pool_i"]).long()
        self.valid = f32(flat(c["pool_sidscore"])) > -1e30
        self.branch = torch.arange(nb).repeat_interleave(Kp)                 # (C,) behavior of each cand
        self.gt_b, self.gt_f, self.gt_t, self.gt_i = (c["gt_b"].long(), c["gt_f"].long(),
                                                      c["gt_t"].long(), c["gt_i"].long())
        self.lp_beh = torch.gather(f32(c["beh_logp"]), 1, self.branch.view(1, -1).expand(N, -1))
        self.sidscore = f32(flat(c["pool_sidscore"]))
        self.has_comp = "pool_lpf" in c
        if self.has_comp:
            self.lp_f = f32(flat(c["pool_lpf"])); self.lp_t = f32(flat(c["pool_lpt"])); self.lp_i = f32(flat(c["pool_lpi"]))
        else:                                                               # chain: only the total sid score
            self.lp_f = self.sidscore; self.lp_t = torch.zeros_like(self.sidscore); self.lp_i = torch.zeros_like(self.sidscore)
        gt_is = ((self.branch.view(1, -1) == self.gt_b[:, None]) & (self.pf == self.gt_f[:, None])
                 & (self.pt == self.gt_t[:, None]) & (self.pi == self.gt_i[:, None]) & self.valid)
        self.gt_is = gt_is
        self.has_gt = gt_is.any(1)
        self.gt_idx = torch.where(self.has_gt, gt_is.float().argmax(1), torch.full((N,), -1, dtype=torch.long))
        self.base = (wb * self.lp_beh + self.lp_f + self.lp_t + self.lp_i)
        log(f"[data] {os.path.basename(dump_path)} N={N:,} C={self.C} covered={int(self.has_gt.sum()):,}"
            f" ({self.has_gt.float().mean():.4f}) comp={'yes' if self.has_comp else 'NO(chain)'}")


def assert_split(data, fh):
    per = {b: int((data.gt_b == b).sum()) for b in range(data.nb)}
    ok = (data.N == EXPECT_TOTAL) and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  TEST split total={data.N:,} " + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH)
        + f" -> {'OK' if ok else 'MISMATCH'}", fh)
    if not ok:
        raise SystemExit("** TEST split mismatch -- refusing to emit numbers. **")


# --------------------------------------------------------------------------- branch statistics
def within_group_rank(keys, vals):
    """Normalized within-group rank in [0,1] (fraction of same-key items with smaller val)."""
    n = keys.numel()
    o1 = torch.argsort(vals, stable=True)
    order = o1[torch.argsort(keys[o1], stable=True)]                        # sort by (key, val)
    ks = keys[order]
    start = torch.zeros(n, dtype=torch.long)
    is_start = torch.ones(n, dtype=torch.bool); is_start[1:] = ks[1:] != ks[:-1]
    idxs = torch.arange(n)
    grp_start = torch.cummax(torch.where(is_start, idxs, torch.zeros(n, dtype=torch.long)), 0).values
    pos = idxs - grp_start
    rank = torch.empty(n, dtype=torch.long); rank[order] = pos
    _, inv, cnt = torch.unique(keys, return_inverse=True, return_counts=True)
    size = cnt[inv]
    return rank.float() / (size.float() - 1).clamp(min=1), size


def branch_features(data, device):
    """Per-candidate branch statistics computed over the pooled candidates of each (behavior,f) branch
    within each event. Returns a dict of (N,C) tensors (valid candidates only meaningful)."""
    N, C = data.N, data.C
    ev = torch.arange(N).view(N, 1).expand(N, C)
    blocal = data.branch.view(1, -1).expand(N, C) * KF_MAX + data.pf       # (N,C) (beh,f) key in [0, nb*KF_MAX)
    gkey = ev.to(torch.int64) * (data.nb * KF_MAX) + blocal                # global (event,branch) key
    v = data.valid
    gk = gkey[v].to(device); xt = data.lp_t[v].to(device); xi = data.lp_i[v].to(device); xf = data.lp_f[v].to(device)
    pt = data.pt[v].to(device); pi = data.pi[v].to(device)
    uniq, gid = torch.unique(gk, return_inverse=True)
    G = uniq.numel()
    cnt = torch.zeros(G, device=device).index_add_(0, gid, torch.ones_like(xt))
    def mstd(x):
        s = torch.zeros(G, device=device).index_add_(0, gid, x)
        ss = torch.zeros(G, device=device).index_add_(0, gid, x * x)
        mean = s / cnt.clamp(min=1); var = (ss / cnt.clamp(min=1) - mean * mean).clamp(min=1e-10)
        return mean, var.sqrt()
    mean_t, std_t = mstd(xt); mean_i, std_i = mstd(xi)
    def shape(x, code, K):
        tk = gid * K + code
        utk, gtd = torch.unique(tk, return_inverse=True)
        lp_d = torch.full((utk.numel(),), -1e30, device=device).scatter_reduce(0, gtd, x, reduce="amax", include_self=True)
        bdt = (utk // K).long()
        Pd = lp_d.exp()
        Z = torch.zeros(G, device=device).index_add_(0, bdt, Pd)
        p = Pd / Z[bdt].clamp(min=1e-12)
        H = torch.zeros(G, device=device).index_add_(0, bdt, -p * p.clamp(min=1e-12).log())
        top1 = torch.zeros(G, device=device).scatter_reduce(0, bdt, Pd, reduce="amax", include_self=True)
        return H, top1, H.exp()
    H_t, top1_t, eff_t = shape(xt, pt, 1024); H_i, top1_i, eff_i = shape(xi, pi, 1024)
    lpf_g = torch.zeros(G, device=device).index_add_(0, gid, xf) / cnt.clamp(min=1)
    ev_of_g = (uniq // (data.nb * KF_MAX)).long()
    brank_f, _ = within_group_rank(ev_of_g.cpu(), lpf_g.cpu())             # branch rank by logP(f) within event
    brank_f = brank_f.to(device)
    # within-event entropy spread (std across branches / mean)
    Hsum = H_t + H_i
    ev_mean = torch.zeros(N, device=device).index_add_(0, ev_of_g, Hsum) / torch.zeros(N, device=device).index_add_(0, ev_of_g, torch.ones_like(Hsum)).clamp(min=1)
    ev_ss = torch.zeros(N, device=device).index_add_(0, ev_of_g, Hsum * Hsum)
    ev_cnt = torch.zeros(N, device=device).index_add_(0, ev_of_g, torch.ones_like(Hsum)).clamp(min=1)
    ev_std = (ev_ss / ev_cnt - ev_mean.pow(2)).clamp(min=0).sqrt()
    # scatter group stats back to candidates
    def back(g):
        full = torch.zeros(N * C, device=device)
        full[v.reshape(-1)] = g[gid]; return full.reshape(N, C).cpu()
    feats = {k: back(x) for k, x in dict(mean_t=mean_t, std_t=std_t, mean_i=mean_i, std_i=std_i,
             H_t=H_t, H_i=H_i, top1_t=top1_t, top1_i=top1_i, eff_t=eff_t, eff_i=eff_i,
             nbranch=cnt, brank_f=brank_f, Hsum=Hsum).items()}
    # within-branch normalized rank of each candidate's logP(t|f) and logP(i|f)
    rk_t, _ = within_group_rank(gid.cpu(), xt.cpu()); rk_i, _ = within_group_rank(gid.cpu(), xi.cpu())
    rt = torch.zeros(N * C); rt[v.reshape(-1)] = rk_t; feats["rank_t"] = rt.reshape(N, C)
    ri = torch.zeros(N * C); ri[v.reshape(-1)] = rk_i; feats["rank_i"] = ri.reshape(N, C)
    feats["ev_H_std"] = ev_std.cpu(); feats["ev_H_mean"] = ev_mean.cpu()
    return feats


# --------------------------------------------------------------------------- ranking / metrics
def ranks(score, data):
    sid = ((data.pf == data.gt_f[:, None]) & (data.pt == data.gt_t[:, None]) & (data.pi == data.gt_i[:, None]))
    beh = data.branch.view(1, -1) == data.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, data.valid)
    r2, h2 = best_rank(score, sid, beh & data.valid)
    return r3, h3, r2, h2


def mechanism(score, data):
    cov = data.has_gt
    gts = torch.where(data.gt_is, score, torch.full_like(score, float("-inf"))).max(1).values
    srt = score.sort(1, descending=True).values
    gap = (srt[:, 9] - srt[:, 10])[cov].mean().item()
    band = (((score >= gts.unsqueeze(1) - 0.5) & (score <= gts.unsqueeze(1) + 0.5) & data.valid).sum(1))[cov].float().mean().item()
    r3, h3, _, _ = ranks(score, data); rk = r3[cov]
    b = [(rk < 10).float().mean().item(), ((rk >= 10) & (rk < 50)).float().mean().item(), (rk >= 50).float().mean().item()]
    return gap, band, b


def report(tag, data, score, fh):
    r3, h3, r2, h2 = ranks(score, data); gt_b = data.gt_b; cov = data.has_gt
    gap, band, bnd = mechanism(score, data)
    log(f"\n  [{tag}]", fh)
    log(f"    t3 R@5={recall_at(r3,h3,5):.4f} R@10={recall_at(r3,h3,10):.4f} R@20={recall_at(r3,h3,20):.4f} "
        f"N@5={ndcg_at(r3,h3,5):.4f} N@10={ndcg_at(r3,h3,10):.4f}", fh)
    log(f"    t2 R@5={recall_at(r2,h2,5):.4f} R@10={recall_at(r2,h2,10):.4f} R@20={recall_at(r2,h2,20):.4f} "
        f"N@5={ndcg_at(r2,h2,5):.4f} N@10={ndcg_at(r2,h2,10):.4f}", fh)
    log(f"    P(rank<10|cov)={recall_at(r3,h3,10,cov):.4f} (cur 0.7945, chain 0.8581, break-even 0.81)  "
        f"coverage={cov.float().mean().item():.4f}", fh)
    log(f"    gap10-11={gap:.4f} (cur 0.2063, chain 0.3213)  #within0.5={band:.2f} (cur 27.50, chain 22.09)  "
        f"bands <10/10-49/50+ = {bnd[0]:.4f}/{bnd[1]:.4f}/{bnd[2]:.4f}", fh)
    log(f"    t3@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r3,h3,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    log(f"    t2@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r2,h2,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    return {"t3@10": recall_at(r3, h3, 10), "pcov": recall_at(r3, h3, 10, cov), "gap": gap, "band": band,
            "perbeh": {b: recall_at(r3, h3, 10, gt_b == b) for b in range(data.nb)}}


# --------------------------------------------------------------------------- STAGE 1
def stage1(data, feats, fh, chain=False):
    tag = "chain (l0-group PROXY: dump lacks per-level conditionals)" if chain else "MFD"
    log(f"\n=== STAGE 1 diagnostic -- {tag} ===", fh)
    cov = data.has_gt
    log(f"  within-event branch-entropy spread: std/mean quantiles (covered events):", fh)
    r = feats["ev_H_std"][cov] / feats["ev_H_mean"][cov].clamp(min=1e-6)
    log(f"    std/mean 10/25/50/75/90 = " + "/".join(f"{np.percentile(r.numpy(),p):.3f}" for p in (10,25,50,75,90)), fh)
    # GT branch entropy percentile within its event -> P(rank<10|cov)
    r3, h3, _, _ = ranks(data.base, data)
    gt_H = feats["Hsum"].gather(1, data.gt_idx.clamp(min=0).unsqueeze(1)).squeeze(1)   # GT branch entropy
    # percentile of GT branch entropy among the event's branches: use ev_H_mean/std z, or rank via Hsum
    # approximate percentile: (gt_H - ev_mean)/ev_std -> z; bucket
    z = ((gt_H - feats["ev_H_mean"]) / feats["ev_H_std"].clamp(min=1e-6))
    log("  P(rank<10|cov) bucketed by GT-branch entropy z within its event (flat branch = high z):", fh)
    edges = [-1e9, -1.0, -0.5, 0.0, 0.5, 1.0, 1e9]
    names = ["<-1", "-1..-.5", "-.5..0", "0..0.5", "0.5..1", ">1"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = cov & (z >= lo) & (z < hi)
        if m.sum() == 0:
            continue
        log(f"    z in [{nm:>8}]  n={int(m.sum()):>7,}  P(rank<10|cov)={recall_at(r3,h3,10,m):.4f}", fh)
    log("  A monotone DECLINE as z increases (GT in a flatter-than-peers branch) confirms the scale", fh)
    log("  hypothesis. Flat/no dependence => hypothesis wrong, do not proceed to calibration.", fh)


# --------------------------------------------------------------------------- STAGE 2 variants
def z_calib(data, feats):
    zt = (data.lp_t - feats["mean_t"]) / feats["std_t"].clamp(min=1e-6)
    zi = (data.lp_i - feats["mean_i"]) / feats["std_i"].clamp(min=1e-6)
    return data.wb * data.lp_beh + data.lp_f + zt + zi


def rank_calib(data, feats):
    # centered within-branch rank (robust to outliers)
    return data.wb * data.lp_beh + data.lp_f + (feats["rank_t"] - 0.5) * 2 + (feats["rank_i"] - 0.5) * 2


def fit_temperature(tr, ftr, device, steps=400, lr=0.05):
    """tau_f = exp(gamma*(H_event_mean - H_branch)); flat branch (high H) -> small tau. Fit (gamma,s)."""
    idx = torch.nonzero(tr.has_gt).squeeze(1)                              # covered events only (subset first)
    Hs = ftr["Hsum"][idx].to(device); Hm = ftr["ev_H_mean"][idx].unsqueeze(1).to(device)
    cond = (tr.lp_t + tr.lp_i)[idx].to(device); base_bf = (tr.wb * tr.lp_beh + tr.lp_f)[idx].to(device)
    valid = tr.valid[idx].to(device); gt = tr.gt_idx[idx].to(device)
    gamma = torch.zeros((), device=device, requires_grad=True); s = torch.ones((), device=device, requires_grad=True)
    opt = torch.optim.Adam([gamma, s], lr=lr)
    for _ in range(steps):
        tau = (gamma * (Hm - Hs)).exp()
        sc = (base_bf + s * tau * cond).masked_fill(~valid, float("-inf"))
        loss = F.cross_entropy(sc, gt)
        opt.zero_grad(); loss.backward(); opt.step()
    return gamma.detach(), s.detach()


def temp_calib(data, feats, gamma, s, device):
    Hs = feats["Hsum"]; Hm = feats["ev_H_mean"].unsqueeze(1).expand_as(Hs)
    tau = (gamma.cpu() * (Hm - Hs)).exp()
    return data.wb * data.lp_beh + data.lp_f + s.cpu() * tau * (data.lp_t + data.lp_i)


class BranchMLP(nn.Module):
    def __init__(self, n_in, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, x):
        o = self.net(x); return F.softplus(o[..., 0]), o[..., 1]           # a_f>=0, b_f


BSTAT = ["H_t", "H_i", "top1_t", "top1_i", "eff_t", "eff_i", "nbranch", "brank_f"]


def bstats_tensor(data, feats):
    x = torch.stack([feats[k] for k in BSTAT], dim=-1)                     # (N,C,8)
    x[..., BSTAT.index("nbranch")] = (x[..., BSTAT.index("nbranch")] + 1).log()   # log branch size
    x[..., BSTAT.index("eff_t")] = (x[..., BSTAT.index("eff_t")] + 1).log()
    x[..., BSTAT.index("eff_i")] = (x[..., BSTAT.index("eff_i")] + 1).log()
    return x


def fit_mlp(tr, ftr, device, epochs=200, lr=1e-2, wd=1e-3, seed=0, batch=2048):
    """Minibatch over covered EVENTS -- full-batch (98k events x 2000 cand x 32 hidden) OOMs."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.empty_cache()
    mlp = BranchMLP(len(BSTAT)).to(device)
    idx_all = torch.nonzero(tr.has_gt).squeeze(1).numpy()
    X = bstats_tensor(tr, ftr)                                             # (N,C,8) CPU
    cond = (tr.lp_t + tr.lp_i); base_bf = (tr.wb * tr.lp_beh + tr.lp_f)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, len(idx_all), batch):
            bidx = torch.as_tensor(idx_all[s:s + batch])
            a, b = mlp(X[bidx].to(device))
            sc = (base_bf[bidx].to(device) + a * cond[bidx].to(device) + b).masked_fill(~tr.valid[bidx].to(device), float("-inf"))
            loss = F.cross_entropy(sc, tr.gt_idx[bidx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return mlp


def mlp_ab(data, feats, mlp, device, batch=2048):
    """(a_f, b_f) per candidate as CPU (N,C) tensors, minibatched over events (avoids the OOM)."""
    X = bstats_tensor(data, feats)
    A = torch.empty(data.N, data.C); B = torch.empty(data.N, data.C)
    with torch.no_grad():
        for s in range(0, data.N, batch):
            e = slice(s, min(s + batch, data.N))
            a, b = mlp(X[e].to(device)); A[e] = a.cpu(); B[e] = b.cpu()
    return A, B


def mlp_calib(data, feats, mlp, device):
    a, b = mlp_ab(data, feats, mlp, device)
    return data.wb * data.lp_beh + data.lp_f + a * (data.lp_t + data.lp_i) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_pool", required=True)
    ap.add_argument("--train_dump", required=True); ap.add_argument("--val_dump", required=True); ap.add_argument("--test_dump", required=True)
    ap.add_argument("--fusion_weights", default="/scratch/yw8866/rec-tmall/mfd_fusion_weights.pt")
    ap.add_argument("--wb", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/branch_calib"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- STAGE 1 ----
    te = Data(args.test_dump, args.wb); assert_split(te, fh)
    fte = branch_features(te, device)
    stage1(te, fte, fh, chain=False)
    ch = Data(args.chain_pool, args.wb)
    if ch.has_comp:
        stage1(ch, branch_features(ch, device), fh, chain=True)
    else:
        log("\n=== STAGE 1 -- chain: dump has no per-level conditional logps; entropy analog not reproducible.", fh)
        log("  (The comparability hypothesis is native to MFD's two independent softmaxes; chain's single", fh)
        log("  joint factorization has no equivalent per-branch scale, which is the point.)", fh)

    log("\n>>> If Stage 1 shows the monotone decline, proceed. Calibration variants follow regardless,", fh)
    log(">>> but read Stage 1 as the go/no-go.", fh)

    # ---- STAGE 2 ----
    tr = Data(args.train_dump, args.wb); va = Data(args.val_dump, args.wb)
    ftr = branch_features(tr, device)
    if device == "cuda":
        torch.cuda.empty_cache()
    fva = branch_features(va, device)
    if device == "cuda":
        torch.cuda.empty_cache()
    log("\n" + "=" * 84 + "\n=== BASELINE + calibration variants on TEST ===\n" + "=" * 84, fh)
    base = report("BASELINE union (hand-set)", te, te.base, fh)

    results = {"baseline": base}
    results["(a) z-standard"] = report("(a) within-branch z-standardization", te, z_calib(te, fte), fh)
    results["(a) rank"] = report("(a') within-branch rank-based", te, rank_calib(te, fte), fh)
    gamma, s = fit_temperature(tr, ftr, device)
    log(f"\n  (b) fitted tau: gamma={gamma.item():+.4f} s={s.item():+.4f}", fh)
    results["(b) temperature"] = report("(b) entropy-driven temperature", te, temp_calib(te, fte, gamma, s, device), fh)
    mlp = fit_mlp(tr, ftr, device, seed=args.seed)
    results["(c) learned MLP"] = report("(c) learned branch calibration", te, mlp_calib(te, fte, mlp, device), fh)

    # winner (by VAL P(rank<10|cov)) combined with fusion weights
    def score_of(name, d, f):
        if name == "(a) z-standard": return z_calib(d, f)
        if name == "(a) rank": return rank_calib(d, f)
        if name == "(b) temperature": return temp_calib(d, f, gamma, s, device)
        if name == "(c) learned MLP": return mlp_calib(d, f, mlp, device)
        return d.base
    val_p = {k: recall_at(*ranks(score_of(k, va, fva), va)[:2], 10, va.has_gt) for k in results if k != "baseline"}
    winner = max(val_p, key=val_p.get)
    log(f"\n  VAL P(rank<10|cov) by variant: " + "  ".join(f"{k}={v:.4f}" for k, v in val_p.items()), fh)
    log(f"  WINNER on VAL = {winner}", fh)

    if os.path.exists(args.fusion_weights):
        w = torch.load(args.fusion_weights, map_location="cpu", weights_only=False)["weights"].float()
        # combine: fusion weights on [b,f] and on the calibrated conditionals (apply w_t,w_i as scale)
        def combo(d, f):
            if winner == "(a) z-standard":
                zt = (d.lp_t - f["mean_t"]) / f["std_t"].clamp(min=1e-6); zi = (d.lp_i - f["mean_i"]) / f["std_i"].clamp(min=1e-6)
                ct, ci = zt, zi
            elif winner == "(a) rank":
                ct, ci = (f["rank_t"] - 0.5) * 2, (f["rank_i"] - 0.5) * 2
            elif winner == "(b) temperature":
                Hs = f["Hsum"]; Hm = f["ev_H_mean"].unsqueeze(1).expand_as(Hs); tau = (gamma.cpu() * (Hm - Hs)).exp()
                ct, ci = s.cpu() * tau * d.lp_t, s.cpu() * tau * d.lp_i
            else:
                a, b = mlp_ab(d, f, mlp, device)
                ct, ci = a * d.lp_t + b / 2, a * d.lp_i + b / 2
            return w[0] * d.lp_beh + w[1] * d.lp_f + w[2] * ct + w[3] * ci
        log(f"\n  fusion weights w={w.tolist()}", fh)
        results[f"{winner}+fusion"] = report(f"WINNER {winner} + learned fusion weights", te, combo(te, fte), fh)

    # coverage invariance
    cb = ranks(te.base, te)[1].float().mean().item()
    cw = ranks(z_calib(te, fte), te)[1].float().mean().item()
    log(f"\n  coverage invariance: base={cb:.4f} calibrated={cw:.4f} {'OK' if abs(cb-cw)<1e-6 else '** DRIFT (bug) **'}", fh)
    assert abs(cb - cw) < 1e-6, "coverage drifted under rescoring -- BUG"

    # summary
    log("\n" + "=" * 84 + "\n=== SUMMARY ===\n" + "=" * 84, fh)
    log(f"  {'variant':<26}{'t3@10':>9}{'P(r<10|c)':>11}{'gap':>9}{'band':>8}", fh)
    for k, r in results.items():
        log(f"  {k:<26}{r['t3@10']:>9.4f}{r['pcov']:>11.4f}{r['gap']:>9.4f}{r['band']:>8.2f}", fh)
    log("\n  per-behavior t3@10 (baseline -> winner):", fh)
    win_r = results.get(f"{winner}+fusion", results[winner])
    for b in range(te.nb):
        log(f"    {BEH[b]:<5} {base['perbeh'][b]:.4f} -> {win_r['perbeh'][b]:.4f}", fh)
    log("  Honest read: does calibration raise P(rank<10|cov) toward 0.81 break-even / 0.8581 chain, widen", fh)
    log("  the rank10-11 gap, shrink the +/-0.5 band -- and improve BOTH pv and conversions or trade them?", fh)
    fh.close()


if __name__ == "__main__":
    main()
