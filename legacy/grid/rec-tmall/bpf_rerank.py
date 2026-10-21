"""BPF -- branch-posterior position features: attack MFD's ORDERING deficit with the factorisation itself.

The identity says ordering is the lever: recall@10 = coverage x P(rank<10|cov); MFD is 0.2968 x 0.7252,
chain 0.2471 x 0.8642, so +0.01 of MFD ordering is worth +0.0030 of t3@10 while +0.01 of coverage is
worth ~+0.0002 after the measured 2.3% conversion rate of newly covered GTs.

MECHANISM. MFD reads the COMPLETE 1024-way P(t|f,b) and P(i|f,b) from one forward pass per (b,f) root,
so where a candidate sits INSIDE its branch's posterior is free at 808 decoder states. Chain would need
~1024 continuations per root to know the same thing. Both arms get the features (the chain control is
mandatory); chain's are computed over its ~100 beams, MFD's over the branch's full pooled support.

NOT the calibration failure. Branch z-standardisation REPLACED the absolute log-probs and collapsed
P(rank<10|cov) from 0.78 to 0.60 -- absolute level carries real information. BPF ADDS relative position
next to the untouched base score, so the model can use both.

FEATURES (14/candidate, no behavior split; branch g = (behavior, f-root) within an event):
   1 t_margin      logP(t|f) - max over g          8 i_nuniq       log1p |distinct i in g|
   2 i_margin      logP(i|f) - max over g          9 t_gap12       top1 - top2 of logP(t|f) in g
   3 t_share       share of g's pooled t-mass     10 i_gap12
   4 i_share                                      11 f_margin      logP(f|b) - max root of that b
   5 t_mass        log pooled t-mass of g         12 branch_size   log1p |candidates in g|
   6 i_mass                                       13 in_branch     base - max base in g
   7 t_nuniq       log1p |distinct t in g|        14 in_event      base - max base in the event
All z-scored with TRAIN-candidate statistics and clipped to +-10; invalid candidates get 0.

PROTOCOL. R2_rw1, capped, fit train / select VAL / report TEST. 3 seeds x 2 arms x {baseline, +BPF};
baselines are reused from the seed-variance cache (identical code path and seeds). Verdicts: ABSOLUTE
(per-seed paired CI of the gain within an arm + across-seed mean), DIFFERENTIAL (does the LEAD move,
across seeds), and the 1% bar (mean lead >= 0.01 x mean chain t3@10, t-interval lower bound above it).
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
import torch.nn.functional as F
from torch import nn

from train_mfd_reranker import BEH
from sweep2_rerank import Ranker, make_scal, metrics, ranks_of, RW
from coverage_expansion import Pool, canon, log
from cf_rerank import paired_ci, four_cells

NEG = -1e30
FEATS = ["t_margin", "i_margin", "t_share", "i_share", "t_mass", "i_mass", "t_nuniq", "i_nuniq",
         "t_gap12", "i_gap12", "f_margin", "branch_size", "in_branch", "in_event"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}

# Screened variants, all aimed at the SAME measured target: the 5,008 MFD-ONLY misses, 73% of which
# still sit at rank >= 100 after reranking. (name, use_bpf, rank-weighting, hard-neg window, base gate)
#   +BPF        does knowing the branch posterior help at all?
#   +BPF_gate   ... and does the model also need permission to discount the generator's base score?
#   +BPF_gate_rw2  ... and does it need the deep rank bands weighted harder (9x at rank>=50)?
# Screening runs at ONE seed and is selected on VAL; the winner is then confirmed on every seed and
# both arms. TEST is never used to choose.
SCREEN = [("+BPF", True, "rw1", 0, False),          # do the branch-posterior features help at all?
          ("+BPF_gate", True, "rw1", 0, True),       # + permission to discount the generator's base score
          ("+BPF_gate_rw2", True, "rw2", 0, True),   # + deep rank bands weighted 9x
          ("+BPF_gate_hn64", True, "rw1", 64, True)] # + an objective that IS deep promotion:
# hn=64 replaces the 8,000-way listwise softmax with cross-entropy over {GT} u {current top-64}.
# Why: the measured blocker is that the reranker promotes 974 events on BOTH but only 206 on MFD-ONLY,
# whose median GT rank moves 429 -> 378. Under a full listwise softmax over 8,000 candidates the loss is
# dominated by the head of the list, so a GT at rank ~400 contributes almost no gradient. Restricting the
# loss to the GT against the candidates currently occupying the top makes "beat the 10th-ranked item" the
# literal training objective, independent of how deep the GT starts.


class GateRanker(Ranker):
    """Reranker whose BASE-SCORE anchor is per-candidate and learned.

    Measured motivation (headroom audit, 5 seeds): of the 5,008 MFD-ONLY misses -- the only population
    where a gain moves the LEAD one-for-one -- 73% still sit at rank >= 100 AFTER reranking. The score
    is `base_scale * base + delta` with a bounded delta, so a candidate whose generator score puts it
    near rank 1000 can essentially never reach the top 10 however good the features are. Here the
    anchor becomes `2*sigmoid(gate(h)) * base_scale * base + delta`: the model can DISCOUNT the
    generator score for a specific candidate (gate -> 0) when the branch-posterior features say it
    deserves promotion, or trust it (gate -> 2). Zero-init gate => 2*sigmoid(0) = 1.0, so together with
    the zero-init head the model still starts EXACTLY at the baseline score (identity start preserved).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = nn.Linear(self.head.in_features, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, f, t, i, scal, base, valid, mem, mem_pad):
        B, C = f.shape
        if self.mem_proj is not None:
            mem = self.mem_proj(mem)
        uf = self.U_f[f]
        x = self.in_drop(self.in_norm(
            self.view_proj(torch.cat([uf, uf + self.U_t[t], uf + self.U_i[i]], -1)) + self.scalar_proj(scal)))
        for blk in self.blocks:
            x = blk(x, mem, mem_pad)
        h = self.out_norm(x)
        delta = self.head(h).squeeze(-1)
        g = 2.0 * torch.sigmoid(self.gate(h).squeeze(-1))          # 1.0 at init
        return (self.base_scale * g * base + delta).masked_fill(~valid, float("-inf"))


class BPFPool(Pool):
    """Pool + within-branch position features computed from the stored pool (no regeneration)."""

    def __init__(self, name, tr, va, te, sid, wb, dev, fh, kpool=None, seed=0):
        super().__init__(name, tr, va, te, sid, wb, dev, fh, kpool=kpool)
        self.W = self.widths[0]; self.G = self.nb * self.W
        self.gen = torch.Generator().manual_seed(seed + 777)
        self._standardise(fh)

    @torch.no_grad()
    def bpf_raw(self, d, idx, dev):
        f = d.f[idx].long().to(dev); t = d.t[idx].long().to(dev); i = d.i[idx].long().to(dev)
        v = d.valid[idx].to(dev)                                   # (B,C)
        lp = d.lp[idx].float().to(dev)                             # (B,C,4) [beh, f, t|f, i|f]
        lpf, lpt, lpi = lp[..., 1], lp[..., 2], lp[..., 3]
        base = d.base[idx].float().to(dev)
        B, C = f.shape; G = self.G; nb = self.nb
        br = d.branch.view(1, -1).expand(B, C).to(dev)              # candidate's behavior
        gid = br * self.W + f                                       # (B,C) branch id inside the event
        rowg = torch.arange(B, device=dev).view(B, 1) * G + gid     # (B,C) flat (event, branch)

        def amax_flat(keys, vals, size):
            """max of vals (B,C) over groups given by flat keys (B*C,), invalid candidates excluded."""
            o = torch.full((size,), NEG, device=dev)
            return o.scatter_reduce_(0, keys, torch.where(v, vals, torch.full_like(vals, NEG)).reshape(-1),
                                     reduce="amax", include_self=True)

        kg = rowg.reshape(-1)
        b_maxg = amax_flat(kg, base, B * G)                         # best base score in the branch
        cnt_g = torch.zeros(B * G, device=dev).index_add_(0, kg, v.float().reshape(-1))

        def branch_stats(code, lpx, K):
            """Per branch, over DISTINCT codes: log pooled mass, count, top1, top1-top2."""
            key = (rowg.to(torch.int64) * K + code)[v]              # unique (event, branch, code)
            val = lpx[v]
            u, inv = torch.unique(key, return_inverse=True)
            ul = torch.full((u.numel(),), NEG, device=dev).scatter_reduce_(0, inv, val, reduce="amax", include_self=True)
            ug = (u // K)                                           # flat (event, branch)
            m1 = torch.full((B * G,), NEG, device=dev).scatter_reduce_(0, ug, ul, reduce="amax", include_self=True)
            ex = torch.exp((ul - m1[ug]).clamp(min=-60.0))
            sm = torch.zeros(B * G, device=dev).index_add_(0, ug, ex)
            mass = m1 + torch.log(sm.clamp(min=1e-30))
            nun = torch.zeros(B * G, device=dev).index_add_(0, ug, torch.ones_like(ex))
            u2 = torch.where(ul < m1[ug] - 1e-6, ul, torch.full_like(ul, NEG))
            m2 = torch.full((B * G,), NEG, device=dev).scatter_reduce_(0, ug, u2, reduce="amax", include_self=True)
            gap = torch.where(m2 < -1e29, torch.full_like(m1, 20.0), (m1 - m2).clamp(max=20.0))
            return m1, mass, nun, gap

        t1, tmass, tnun, tgap = branch_stats(t, lpt, self.k1)
        i1, imass, inun, igap = branch_stats(i, lpi, self.k2)

        rb = (torch.arange(B, device=dev).view(B, 1) * nb + br).reshape(-1)
        f_max = amax_flat(rb, lpf, B * nb)                          # best root log-prob of that behavior
        e_max = amax_flat(torch.arange(B, device=dev).view(B, 1).expand(B, C).reshape(-1), base, B)

        g = lambda x: x[rowg]                                       # (B,C) branch-level -> candidate-level
        out = torch.stack([
            lpt - g(t1), lpi - g(i1),
            torch.exp((lpt - g(tmass)).clamp(min=-60.0)), torch.exp((lpi - g(imass)).clamp(min=-60.0)),
            g(tmass), g(imass), torch.log1p(g(tnun)), torch.log1p(g(inun)), g(tgap), g(igap),
            lpf - f_max[rb.view(B, C)],
            torch.log1p(g(cnt_g)), base - g(b_maxg), base - e_max.view(B, 1),
        ], -1)
        return torch.where(v.unsqueeze(-1), out, torch.zeros_like(out))

    def _standardise(self, fh):
        d = self.tr
        s = torch.arange(0, d.N, max(1, d.N // 1500))
        raws = []
        for c in range(0, s.numel(), 16):
            idx = s[c:c + 16]
            raws.append(self.bpf_raw(d, idx, self.dev)[d.valid[idx].to(self.dev)].cpu())
        raw = torch.cat(raws)
        self.mu = raw.mean(0); self.sd = raw.std(0).clamp(min=1e-6)
        log(f"\n  [{self.name}] BPF statistics from {raw.shape[0]:,} TRAIN candidates "
            f"(branch = (behavior, f-root) inside one event; pool-truncated view of the branch posterior):", fh)
        for j, nm in enumerate(FEATS):
            log(f"    {nm:<12} mu={float(self.mu[j]):>9.4f}  sd={float(self.sd[j]):>8.4f}  "
                f"nonzero={float((raw[:, j] != 0).float().mean()):.4f}  "
                f"p10/50/90={np.percentile(raw[:, j].numpy(), [10, 50, 90]).round(3).tolist()}", fh)

    def bpf_feats(self, d, idx, dev):
        raw = self.bpf_raw(d, idx, dev)
        return ((raw - self.mu.to(dev)) / self.sd.to(dev)).clamp(-10.0, 10.0)


def precheck(pool, dev, fh, n_events=2000):
    """Univariate AUC (candidate is the GT) + coverage, on covered TRAIN events."""
    d = pool.tr
    ci = pool.tr.train_idx
    s = ci[torch.randperm(ci.numel(), generator=pool.gen)[:n_events]]
    vals, labs = [], []
    for c in range(0, s.numel(), 16):
        idx = s[c:c + 16]
        raw = pool.bpf_raw(d, idx, dev); v = d.valid[idx].to(dev)
        lab = torch.zeros(idx.numel(), d.C, dtype=torch.bool, device=dev)
        lab[torch.arange(idx.numel(), device=dev), d.gt_idx[idx].to(dev)] = True
        vals.append(raw[v].cpu()); labs.append(lab[v].cpu())
    X = torch.cat(vals); y = torch.cat(labs); n_neg = int((~y).sum())
    log(f"\n  [{pool.name}] PRE-CHECK (univariate; it does NOT predict incremental value -- recorded, not used to gate): "
        f"{X.shape[0]:,} candidates, {int(y.sum()):,} positives", fh)
    auc = {}
    for j, nm in enumerate(FEATS):
        x = X[:, j].double(); xp, xn = x[y], torch.sort(x[~y]).values
        lo = torch.searchsorted(xn, xp, right=False).double(); hi = torch.searchsorted(xn, xp, right=True).double()
        auc[nm] = float(((lo + 0.5 * (hi - lo)) / n_neg).mean())
        log(f"    {nm:<12} AUC={auc[nm]:.4f}", fh)
    return auc


def run_cell(pool, use_bpf, cap, dev, args, fh, rw="rw1", hn=0, gate=False):
    """rw = rank-band event weighting; hn = hard-negative window (0 = full listwise); gate = learned
    per-candidate base anchor (GateRanker) instead of the fixed scalar base_scale."""
    torch.manual_seed(args.seed); _t0 = time.perf_counter(); ep = 0
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    cls = GateRanker if gate else Ranker
    model = cls(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, len(FEATS) if use_bpf else 0).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = pool.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]])
    rb = ranks_of(pool.tr.base.float(), pool.tr)[0]
    band = torch.where(rb < 10, 0, torch.where(rb < 50, 1, 2)); tab = RW[rw]
    w_ev = torch.tensor([tab[int(b)] for b in band.tolist()], dtype=torch.float)
    rng = np.random.default_rng(args.seed)

    def scal(split, d, idx):
        f, t, i, s = make_scal(pool, split, model, d, idx, dev, ())
        return f, t, i, (torch.cat([s, pool.bpf_feats(d, idx, dev)], -1) if use_bpf else s)

    def ev(d, split):
        model.eval(); sc = torch.empty(d.N, d.C)
        with torch.no_grad():
            for s in range(0, d.N, args.eval_batch):
                idx = torch.arange(s, min(s + args.eval_batch, d.N))
                f, t, i, sl = scal(split, d, idx)
                sc[idx] = model(f, t, i, sl, d.base[idx].float().to(dev), d.valid[idx].to(dev),
                                d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev)).cpu()
        return sc

    base_te = metrics(pool.te, pool.te.base.float())
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s:s + args.batch])
            f, t, i, sl = scal("tr", pool.tr, bidx)
            sc = model(f, t, i, sl, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                       pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            gt = pool.tr.gt_idx[bidx].to(dev)
            if hn:                                    # loss over GT + the nearest-ranked window only
                k = min(hn, sc.size(1))
                sel = torch.cat([gt.unsqueeze(1), sc.detach().topk(k, dim=-1).indices], 1)
                sub = torch.gather(sc, 1, sel)        # GT placed first -> target index 0
                loss = F.cross_entropy(sub, torch.zeros(sub.size(0), dtype=torch.long, device=dev), reduction="none")
            else:
                loss = F.cross_entropy(sc, gt, reduction="none")
            w = w_ev[bidx].to(dev); loss = (loss * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = metrics(pool.va, ev(pool.va, "va"))["t3@10"]
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
    te = metrics(pool.te, ev(pool.te, "te"))
    assert abs(te["cov"] - base_te["cov"]) < 1e-9, "coverage changed under reranking -- BUG"
    return {"params": npar, "val": best_v, "n_train": int(len(idx_all)), "n_scalar": 28 + (len(FEATS) if use_bpf else 0),
            "cfg": {"bpf": bool(use_bpf), "rw": rw, "hn": int(hn), "gate": bool(gate)},
            "train_secs": time.perf_counter() - _t0, "epochs_run": ep, "C": int(pool.te.C),
            "before": {k: v for k, v in base_te.items() if k not in ("hit", "r3")},
            "after": {k: v for k, v in te.items() if k not in ("hit", "r3")},
            "uid": pool.uid["te"].numpy(), "gt_b": pool.te.gt_b.numpy(), "has_gt": pool.te.has_gt.numpy(),
            "r3_before": base_te["r3"].numpy(), "r3_after": te["r3"].numpy(),
            "hit_before": base_te["hit"], "hit_after": te["hit"], "ent_z": np.zeros(pool.te.N)}


def ms(x, n):
    x = np.asarray(x, float)
    s = x.std(ddof=1) if n > 1 else float("nan")
    half = (T975.get(n - 1, 1.96) * s / np.sqrt(n)) if n > 1 else float("nan")
    return x.mean(), s, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--base_cache", default="", help="seedvar cache with {arm}__seed{s}.pt baselines")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--stage", default="all", choices=["precheck", "all"])
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bar", type=float, default=0.01)
    ap.add_argument("--variants", default="",
                    help="override the screened variants without editing code: "
                         "'name:bpf,rw,hn,gate;name2:...' e.g. '+stack:1,rw1,64,1;+stack_rw2:1,rw2,64,1'")
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/bpf_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "bpf_report.txt"), "a")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(args.spec)); seeds = [int(s) for s in args.seeds.split(",")]
    log("=" * 100 + f"\n=== BPF: branch-posterior position features -- the ORDERING lever  ({time.strftime('%Y-%m-%d %H:%M')}) ===\n"
        + "=" * 100, fh)
    log(f"  seeds={seeds}   BAR: mean(MFD - chain) >= {args.bar:+.3f} ABSOLUTE, across-seed t-interval above it.", fh)
    log("  Measured target (headroom audit, 5 seeds): the lead +0.00173 decomposes into MFD-ONLY +0.00419,", fh)
    log("  BOTH -0.00119 (chain is AHEAD on shared events, 0.9137 vs 0.9086) and CHAIN-ONLY -0.00127.", fh)
    log("  MFD-ONLY = 5,355 events at 6.5% recovery, 5,008 misses, ceiling +0.0603; the bar needs 13.7% of", fh)
    log("  them. 73% of those misses are still at rank >= 100 AFTER reranking -- which is what the base-score", fh)
    log("  gate variant exists to test.", fh)

    if args.variants:                       # next-round configuration with no code edit
        screen = []
        for spec in args.variants.split(";"):
            nm, rest = spec.split(":")
            bp, rw, hn, gt_ = rest.split(",")
            screen.append((nm, bool(int(bp)), rw, int(hn), bool(int(gt_))))
    else:
        screen = list(SCREEN)
    log("  screened variants: " + ", ".join(f"{n}(bpf={b},rw={r},hn={h},gate={g})" for n, b, r, h, g in screen), fh)

    res = {}
    winner = None
    for arm in ("MFD", "CH"):
        sp = spec[arm]
        log(f"\n{'#' * 100}\n# ARM {arm}: {os.path.basename(sp['te'])}  K_pool={sp.get('kpool')}\n{'#' * 100}", fh)
        pool = BPFPool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"), seed=args.seed)
        assert pool.te.N == 83005, "test split"
        precheck(pool, dev, fh)
        if args.stage == "precheck":
            del pool
            if dev == "cuda":
                torch.cuda.empty_cache()
            continue

        def show(cell, sd, r):
            log(f"  [{arm}] {cell:<14} seed {sd}: t3@10 {r['after']['t3@10']:.4f}  "
                f"P(r<10|cov)={r['after']['pcov']:.4f}  cov={r['after']['cov']:.4f}  "
                f"VAL {r['val']:.4f}  {r['epochs_run']} ep  {r['train_secs']/60:.1f} min", fh)

        # ---- baselines: reuse the seed-variance runs (identical code path and seeds) ----
        for sd in seeds:
            cp = os.path.join(args.cache, f"{arm}__baseline__seed{sd}.pt")
            if not os.path.exists(cp) and args.base_cache:
                src = os.path.join(args.base_cache, f"{arm}__seed{sd}.pt")
                if os.path.exists(src):
                    torch.save(torch.load(src, map_location="cpu", weights_only=False), cp)
                    log(f"  [{arm}] baseline seed {sd} reused from {src}", fh)
            if not os.path.exists(cp):
                a_ = copy.copy(args); a_.seed = sd
                log(f"\n  >>> [{arm}] baseline seed {sd}", fh)
                torch.save(run_cell(pool, False, args.cap, dev, a_, fh), cp)
            res[(arm, "baseline", sd)] = canon(torch.load(cp, map_location="cpu", weights_only=False))
            show("baseline", sd, res[(arm, "baseline", sd)])

        # ---- PHASE 1: screen the variants on MFD at one seed, select on VAL ----
        if arm == "MFD" and winner is None:
            log(f"\n  --- PHASE 1: screening {len(screen)} variants at seed {seeds[0]}; selection is on VAL, never TEST ---", fh)
            scr = {}
            for nm, use, rw, hn, gt_ in screen:
                cp = os.path.join(args.cache, f"MFD__{nm}__seed{seeds[0]}.pt")
                if not os.path.exists(cp):
                    a_ = copy.copy(args); a_.seed = seeds[0]
                    log(f"\n  >>> [MFD] {nm} seed {seeds[0]}  (bpf={use}, rw={rw}, hn={hn}, gate={gt_}, "
                        f"n_scalar={28 + (len(FEATS) if use else 0)})", fh)
                    torch.save(run_cell(pool, use, args.cap, dev, a_, fh, rw=rw, hn=hn, gate=gt_), cp)
                scr[nm] = torch.load(cp, map_location="cpu", weights_only=False)
                show(nm, seeds[0], scr[nm])
            base_val = res[("MFD", "baseline", seeds[0])]["val"]
            for nm in scr:
                log(f"    {nm:<14} VAL {scr[nm]['val']:.4f} (baseline {base_val:.4f}, "
                    f"{scr[nm]['val']-base_val:+.4f})", fh)
            best = max(scr, key=lambda k: scr[k]["val"])
            winner = next(c for c in screen if c[0] == best)
            log(f"  PHASE 1 WINNER (highest VAL): {best}"
                + ("" if scr[best]["val"] > base_val else "  ** no variant beat the baseline on VAL **"), fh)
            wp = os.path.join(args.cache, f"MFD__+BPF__seed{seeds[0]}.pt")
            if not os.path.exists(wp):
                torch.save(scr[best], wp)

        # ---- PHASE 2: confirm the winner on every seed, both arms ----
        nm, use, rw, hn, gt_ = winner
        for sd in seeds:
            cp = os.path.join(args.cache, f"{arm}__+BPF__seed{sd}.pt")
            if not os.path.exists(cp):
                a_ = copy.copy(args); a_.seed = sd
                log(f"\n  >>> [{arm}] +BPF({nm}) seed {sd}", fh)
                torch.save(run_cell(pool, use, args.cap, dev, a_, fh, rw=rw, hn=hn, gate=gt_), cp)
            res[(arm, "+BPF", sd)] = canon(torch.load(cp, map_location="cpu", weights_only=False))
            show(f"+BPF({nm})", sd, res[(arm, "+BPF", sd)])
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()
    if args.stage == "precheck":
        fh.close(); return

    u0 = res[("MFD", "baseline", seeds[0])]["uid"]
    for k in res:
        assert np.array_equal(res[k]["uid"], u0), f"join failed for {k}"
    assert np.array_equal(res[("MFD", "baseline", seeds[0])]["gt_b"], res[("CH", "baseline", seeds[0])]["gt_b"])
    rng = np.random.default_rng(0); n = len(seeds); N = u0.shape[0]

    log("\n" + "=" * 100 + "\n=== PER SEED (TEST) ===\n" + "=" * 100, fh)
    log(f"  {'seed':>4} {'arm':<4}{'cell':<10}{'cov':>8}{'P(r<10|c)':>11}{'t3@10':>9}{'ident':>9}{'t3@5':>8}{'t3@20':>8}"
        f"{'n@10':>8}{'t2@10':>8}{'VAL':>8}{'ep':>4}" + "".join(f"{BEH[b]:>9}" for b in range(4)), fh)
    for sd in seeds:
        for arm in ("MFD", "CH"):
            for cell in ("baseline", "+BPF"):
                a = res[(arm, cell, sd)]["after"]; r = res[(arm, cell, sd)]
                log(f"  {sd:>4} {arm:<4}{cell:<10}{a['cov']:>8.4f}{a['pcov']:>11.4f}{a['t3@10']:>9.4f}"
                    f"{a['cov']*a['pcov']:>9.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}{a['t3n@10']:>8.4f}{a['t2@10']:>8.4f}"
                    f"{r['val']:>8.4f}{r['epochs_run']:>4}" + "".join(f"{a['perbeh'][BEH[b]]:>9.4f}" for b in range(4)), fh)
    log("  (ident = coverage x P(rank<10|cov); must equal t3@10)", fh)

    log("\n" + "=" * 100 + "\n=== VERDICT 1 -- ABSOLUTE: does BPF help each arm? ===\n" + "=" * 100, fh)
    gains = {}
    for arm in ("MFD", "CH"):
        gs = []
        for sd in seeds:
            hb, ha = res[(arm, "baseline", sd)]["hit_after"], res[(arm, "+BPF", sd)]["hit_after"]
            gain = ha.mean() - hb.mean(); lo, hi, p = paired_ci(ha, hb, rng, args.boot); gs.append(gain)
            dp = res[(arm, "+BPF", sd)]["after"]["pcov"] - res[(arm, "baseline", sd)]["after"]["pcov"]
            log(f"  {arm:<4} seed {sd}: gain {gain:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0)={p:.3f}   "
                f"d P(rank<10|cov) {dp:+.4f}  -> {'helps' if lo > 0 else 'hurts' if hi < 0 else 'flat'}", fh)
        m, s, half = ms(gs, n); gains[arm] = (m, s, half, gs)
        log(f"  {arm:<4} ACROSS SEEDS: mean gain {m:+.4f}  std {s:.4f}  95% t-interval [{m-half:+.4f}, {m+half:+.4f}]"
            f"  -> {'WORKS' if m - half > 0 else 'HURTS' if m + half < 0 else 'not distinguishable from 0'}", fh)

    log("\n" + "=" * 100 + "\n=== VERDICT 2 -- DIFFERENTIAL: does the LEAD move, and does it clear the bar? ===\n" + "=" * 100, fh)
    log("  a feature that lifts both arms equally is GENERIC: absolute numbers rise, the lead does not widen.", fh)
    leads = {}
    for cell in ("baseline", "+BPF"):
        ls = []
        for sd in seeds:
            hm, hc = res[("MFD", cell, sd)]["hit_after"], res[("CH", cell, sd)]["hit_after"]
            lo, hi, p = paired_ci(hm, hc, rng, args.boot); ls.append(hm.mean() - hc.mean())
            log(f"  {cell:<9} seed {sd}: lead {ls[-1]:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  P(MFD>chain)={p:.3f}", fh)
        leads[cell] = ls
        m, s, half = ms(ls, n)
        ch_mean = float(np.mean([res[("CH", cell, sd)]["after"]["t3@10"] for sd in seeds]))
        mf_mean = float(np.mean([res[("MFD", cell, sd)]["after"]["t3@10"] for sd in seeds]))
        need = args.bar                     # ABSOLUTE bar (set 2026-09-24): mean(MFD - chain) >= +0.01
        log(f"  {cell:<9} ACROSS SEEDS: MFD {mf_mean:.4f}  chain {ch_mean:.4f}  mean lead {m:+.5f} (std {s:.5f}, "
            f"t-interval [{m-half:+.5f}, {m+half:+.5f}])  = {100*m/ch_mean:+.2f}% relative", fh)
        log(f"  {cell:<9} BAR (absolute {need:+.3f}): mean lead {m:+.5f} -> {'MET' if m >= need else 'NOT met'}"
            f", short by {max(0.0, need-m):+.5f} = {max(0.0, need-m)*len(res[('MFD',cell,seeds[0])]['uid']):,.0f} events; "
            f"t-interval lower bound {m-half:+.5f} -> {'above the bar (stable)' if m - half >= need else 'below the bar'}", fh)
    dl = np.array(leads["+BPF"]) - np.array(leads["baseline"])
    m, s, half = ms(dl, n)
    log(f"\n  LEAD CHANGE (+BPF minus baseline), paired by seed: mean {m:+.5f}  std {s:.5f}  "
        f"95% t-interval [{m-half:+.5f}, {m+half:+.5f}]", fh)
    gm, gc = gains["MFD"][0], gains["CH"][0]
    ratio = gm / gc if abs(gc) > 1e-5 else float("inf") if gm > 0 else float("nan")
    log(f"  MFD gain {gm:+.4f} / chain gain {gc:+.4f} = ratio {ratio:.2f}:1 (hard-match reference 5.3:1)  -> "
        + ("MFD-SPECIFIC: the lead widens significantly" if m - half > 0 else
           "CHAIN-favouring: the lead narrows significantly" if m + half < 0 else
           "GENERIC: the lead does not move (its interval covers 0)"), fh)

    log("\n" + "=" * 100 + "\n=== SUPPORTING ===\n" + "=" * 100, fh)
    log("  per-behavior t3@10, +BPF vs baseline (mean over seeds):", fh)
    for arm in ("MFD", "CH"):
        b = {k: np.mean([res[(arm, "baseline", sd)]["after"]["perbeh"][k] for sd in seeds]) for k in (BEH[0], BEH[1], BEH[2], BEH[3])}
        a = {k: np.mean([res[(arm, "+BPF", sd)]["after"]["perbeh"][k] for sd in seeds]) for k in b}
        d = {k: a[k] - b[k] for k in b}
        note = ("BOTH improve" if d["pv"] > 0 and all(d[k] >= 0 for k in d if k != "pv")
                else "pv up, conversions traded down" if d["pv"] > 0 else "pv down")
        log(f"    {arm:<4}" + "  ".join(f"{k}:{b[k]:.4f}->{a[k]:.4f}({d[k]:+.4f})" for k in b) + f"  [{note}]", fh)
    log("\n  four diagnostic cells (seed 0; reference K_pool=500 rw1: MFD 0.6495/0.2656/0.2153/0.0467, "
        "chain 0.4136/0.0194/0.0992/0.0034):", fh)
    mc = res[("MFD", "baseline", seeds[0])]["has_gt"].astype(bool); cc = res[("CH", "baseline", seeds[0])]["has_gt"].astype(bool)
    for arm, other in (("MFD", cc), ("CH", mc)):
        for cell in ("baseline", "+BPF"):
            f4 = four_cells(res[(arm, cell, seeds[0])], other)
            log(f"    {arm:<4}{cell:<10} BOTH/10-49={f4[0]:.4f}  BOTH/50+={f4[1]:.4f}  own/10-49={f4[2]:.4f}  own/50+={f4[3]:.4f}", fh)
    log("\n  THE DECIDING POPULATION -- events MFD covers and chain does NOT (pool membership is", fh)
    log("  reranker-invariant, so these sets are fixed; every recovery here moves the LEAD one-for-one", fh)
    log("  because chain's pool does not contain the GT at all):", fh)
    own = mc & ~cc; both = mc & cc; n_own, n_both = int(own.sum()), int(both.sum())
    rec = {}
    for cell in ("baseline", "+BPF"):
        hs = [res[("MFD", cell, sd)]["hit_after"] > 0 for sd in seeds]
        rec[cell] = (float(np.mean([h[own].mean() for h in hs])), float(np.mean([h[both].mean() for h in hs])))
        log(f"    {cell:<10} MFD-ONLY n={n_own:,} recovery={rec[cell][0]:.4f} "
            f"(= {rec[cell][0]*n_own/N:+.5f} of t3@10)   BOTH n={n_both:,} recovery={rec[cell][1]:.4f}", fh)
    d_own = (rec["+BPF"][0] - rec["baseline"][0]) * n_own / N
    d_both = (rec["+BPF"][1] - rec["baseline"][1]) * n_both / N
    log(f"    BPF effect split: MFD-ONLY {d_own:+.5f} (chain cannot follow)  +  BOTH {d_both:+.5f} "
        f"(chain can chase this part)  -> {'mechanism is MFD-specific' if d_own > abs(d_both) else 'gain is mostly on shared events'}", fh)
    json.dump({"seeds": seeds, "own_only": {"n": n_own, "rec": rec}, "leads": {k: list(map(float, v)) for k, v in leads.items()},
               "gain_mfd": list(map(float, gains["MFD"][3])), "gain_chain": list(map(float, gains["CH"][3]))},
              open(os.path.join(args.out_dir, "bpf_results.json"), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
