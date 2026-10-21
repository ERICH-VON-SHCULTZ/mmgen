"""Collaborative-filtering features for the listwise reranker: generic gain or MFD-specific?

Both generative models frozen; offline only. Arms: MFD = P3 wide pool (K_f=200, full t/i support, 808
states) truncated to K_pool=2000 (current best, t3@10 0.2155); chain = 808-state pool, K_pool=500.
Reranker R2_rw1, capped (36,864 training events), identical architecture / optimiser / schedule /
early stopping in every cell of both arms. Fit on TRAIN, select on VAL, report on TEST.

Feature groups (all statistics from the TRAINING dump's HISTORY items only -- never val/test, never
the training labels; leave-one-out on the training split so a training event never sees its own
history pairs in its own features):
  CF-A  shared-code co-occurrence: W x W within-sequence pair counts over L0 codes (MFD: 2048-wide
        cross-modal commonality code; chain: its own 1024-wide L0), PPMI-normalised. Per candidate and
        history behavior: max / mean PPMI(code(cand), code(hist)).                          -> 2*nb
  CF-B  shared-code match: history items with the candidate's L0 code, per behavior:
        log1p(count), recency-weighted sum (positional decay), max recency.                 -> 3*nb
  CF-C  item-item co-occurrence: item = full SID tuple (the finest identity the reranker can see);
        within-sequence pair counts >= min_cnt, cosine-normalised, top-N neighbours per item kept.
        Per candidate and history behavior: max / mean / top-3 mean.                        -> 3*nb
Every CF feature is z-scored with TRAIN-candidate statistics before it enters the 28-wide scalar block.
Joins are on packed SID keys; a brute-force recount on random table entries verifies them.

Pre-check: univariate AUC (candidate is the GT) + coverage of every feature on TRAIN pools; a group with
no univariate signal is reported and skipped.  Grid: baseline, +A, +B, +A+B, +C, +all, both arms.
Verdicts (reported separately): ABSOLUTE -- gain over the arm's own baseline, paired bootstrap over
events; DIFFERENTIAL -- MFD gain / chain gain, plus the paired CI of (MFD gain - chain gain) on events
joined by user_id: if it covers 0 the feature is GENERIC (numbers rise, lead does not widen).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn.functional as F

from train_mfd_reranker import BEH
from sweep2_rerank import Ranker, make_scal, metrics, ranks_of, RW
from coverage_expansion import Pool, canon, log

CELLS = [("baseline", ()), ("+CF-A", ("A",)), ("+CF-B", ("B",)), ("+CF-A+B", ("A", "B")),
         ("+CF-C", ("C",)), ("+all", ("A", "B", "C"))]
REF = {"MFD": {"t3@10": 0.2155, "cov": 0.2968, "cells": (0.6495, 0.2656, 0.2153, 0.0467)},
       "CH": {"t3@10": 0.2140, "cov": 0.2471, "cells": (0.4136, 0.0194, 0.0992, 0.0034)}}


def feat_names(groups, nb):
    n = []
    if "A" in groups:
        n += [f"A.ppmi_{s}_{BEH[b]}" for b in range(nb) for s in ("max", "mean")]
    if "B" in groups:
        n += [f"B.code_{s}_{BEH[b]}" for b in range(nb) for s in ("log1pcnt", "recsum", "recmax")]
    if "C" in groups:
        n += [f"C.cooc_{s}_{BEH[b]}" for b in range(nb) for s in ("max", "mean", "top3")]
    return n


def n_cf(groups, nb):
    return len(feat_names(groups, nb))


# =========================================================================== CF statistics + features
class CFPool(Pool):
    """Pool + collaborative-filtering statistics built from the TRAIN split's history items."""

    def __init__(self, name, tr, va, te, sid, wb, dev, fh, kpool=None, min_cnt=2, topn=100, seed=0):
        super().__init__(name, tr, va, te, sid, wb, dev, fh, kpool=kpool)
        self.W = self.widths[0]; self.min_cnt = min_cnt; self.topn = topn
        self.gen = torch.Generator().manual_seed(seed + 777)
        self.build_cf(fh)

    # ---------------------------------------------------------------- statistics
    def _hist(self, d, idx=None):
        sl = slice(None) if idx is None else idx
        hb = d.hist_b[sl].long(); hv = hb >= 0
        hf = d.hist_f[sl].long(); ht = d.hist_t[sl].long(); hi = d.hist_i[sl].long()
        return hf, ht, hi, hb, hv, d.hist_recency[sl].float()

    def build_cf(self, fh):
        d, dev, W = self.tr, self.dev, self.W
        N, H = d.N, d.H
        hf, ht, hi, hb, hv, _ = self._hist(d)
        keys = self.pack(hf, ht, hi)                                   # (N,H) tuple keys (pad rows masked by hv)
        uid = d.user_id
        n_u = int(torch.unique(uid).numel())
        log(f"\n  [{self.name}] CF statistics: source = TRAIN dump HISTORY items only (N={N:,} events, H={H}, "
            f"{int(hv.sum()):,} history items); GT/next items are NOT used; val/test never touched.", fh)
        log(f"    train user_ids unique: {n_u:,}/{N:,} ({n_u/N:.4f}) -> "
            + ("one event per user: a user's future never enters another training event's statistics"
               if n_u == N else "REPEATED users: same-user later events could leak future items into earlier events' stats"), fh)

        # ---- CF-A: L0 pair counts (ordered pairs of distinct positions) + unigrams ----
        C = torch.zeros(W * W, dtype=torch.float64, device=dev)
        u = torch.zeros(W, dtype=torch.float64, device=dev)
        eye = torch.eye(H, dtype=torch.bool, device=dev)
        for s in range(0, N, 4096):
            f = hf[s:s + 4096].to(dev); v = hv[s:s + 4096].to(dev)
            pm = v[:, :, None] & v[:, None, :] & ~eye
            pk = (f[:, :, None] * W + f[:, None, :])[pm]
            C += torch.bincount(pk, minlength=W * W).double()
            u += torch.bincount(f[v], minlength=W).double()
        self.A_C = C.view(W, W); self.A_u = u
        T, U = float(C.sum()), float(u.sum())
        self.A_T, self.A_U = T, U
        pmi = torch.log((self.A_C * (U * U)).clamp(min=1e-300) / (T * (u[:, None] * u[None, :]).clamp(min=1.0)))
        self.A_ppmi = torch.where(self.A_C > 0, pmi.clamp(min=0), torch.zeros_like(pmi)).float()  # (W,W)
        nz = float((self.A_C > 0).float().mean())
        log(f"    CF-A: {W}x{W} L0 co-occurrence, {int(T):,} ordered pairs, {nz:.4f} of cells non-zero, "
            f"PPMI mean over non-zero cells={float(self.A_ppmi[self.A_C > 0].mean()):.3f}, "
            f"diagonal PPMI mean={float(self.A_ppmi.diag().mean()):.3f} (kept: same-code co-occurrence is real co-occurrence)", fh)

        # ---- CF-C: item (tuple) vocabulary + pair counts ----
        vk = keys[hv]
        self.item_keys, inv, cnt = torch.unique(vk, return_inverse=True, return_counts=True)   # CPU copy (verification)
        V = self.item_keys.numel(); self.V = V
        self.ik_dev = self.item_keys.to(dev)                                            # device copy (features)
        self.item_u = cnt.double().to(dev)
        idx_all = torch.full((N, H), -1, dtype=torch.long); idx_all[hv] = inv        # item index per history slot
        pairs = []
        tri = torch.triu(torch.ones(H, H, dtype=torch.bool, device=dev), diagonal=1)   # unordered p<q
        for s in range(0, N, 4096):
            ix = idx_all[s:s + 4096].to(dev); v = ix >= 0
            pm = v[:, :, None] & v[:, None, :] & tri
            a = ix[:, :, None].expand(-1, -1, H)[pm]; b = ix[:, None, :].expand(-1, H, -1)[pm]
            lo, hi_ = torch.minimum(a, b), torch.maximum(a, b)
            ne_ = lo != hi_                                                          # an item is not its own neighbour
            pairs.append((lo * V + hi_)[ne_])
        pairs = torch.cat(pairs)
        n_pairs_total = pairs.numel()
        pk, pc = torch.unique(pairs, return_counts=True); del pairs
        keep = pc >= self.min_cnt
        pk, pc = pk[keep], pc[keep]
        x, y = pk // V, pk % V
        sxy = pc.double() / torch.sqrt(self.item_u[x] * self.item_u[y])
        # both directions, then top-N per row by cosine
        rows = torch.cat([x, y]); cols = torch.cat([y, x]); s_all = torch.cat([sxy, sxy]); c_all = torch.cat([pc, pc])
        o = torch.argsort(s_all, descending=True)                                     # by strength desc ...
        rows, cols, s_all, c_all = rows[o], cols[o], s_all[o], c_all[o]
        o = torch.sort(rows, stable=True).indices                                    # ... then stable by row
        rows, cols, s_all, c_all = rows[o], cols[o], s_all[o], c_all[o]
        first = torch.searchsorted(rows, rows)                                        # first position of each row
        rank = torch.arange(rows.numel(), device=dev) - first
        keep = rank < self.topn
        rows, cols, s_all, c_all = rows[keep], cols[keep], s_all[keep], c_all[keep]
        self.C_keys = rows * V + cols                                                 # sorted (rows sorted, cols arbitrary)
        o = torch.argsort(self.C_keys); self.C_keys = self.C_keys[o]; self.C_s = s_all[o].float(); self.C_cnt = c_all[o].double()
        n_items_with = int(torch.unique(rows).numel())
        log(f"    CF-C: item = SID tuple; |vocab|={V:,} distinct history tuples; {n_pairs_total:,} unordered pairs -> "
            f"{int(keep.numel()):,} directed entries with count>={self.min_cnt} -> top-{self.topn}/item kept: "
            f"{self.C_keys.numel():,} entries over {n_items_with:,} items ({n_items_with/V:.4f} of vocab); "
            f"cosine strength mean={float(self.C_s.mean()):.4f}", fh)

        # ---- join verification by brute-force recount ----
        self._verify(fh, hf, hv, idx_all)
        # ---- standardisation stats from a TRAIN candidate sample (LOO applied) ----
        self._standardise(fh)

    def _verify(self, fh, hf, hv, idx_all):
        dev, W, N = self.dev, self.W, self.tr.N
        # CF-A: 64 random non-zero cells; recount = sum_e n_e(x) n_e(y) - [x==y] n_e(x)
        nzc = torch.nonzero(self.A_C > 0)
        pick = nzc[torch.randint(0, nzc.shape[0], (64,), generator=self.gen)]
        ok = 0
        for x, y in pick.tolist():
            nx = ((hf == x) & hv).sum(1).double(); ny = ((hf == y) & hv).sum(1).double()
            rec = float((nx * ny).sum() - (nx.sum() if x == y else 0.0))
            ok += int(abs(rec - float(self.A_C[x, y])) < 0.5)
        log(f"    JOIN VERIFICATION CF-A: brute-force recount of 64 random co-occurrence cells from the raw "
            f"training histories matches the matrix in {ok}/64 cases" + ("" if ok == 64 else "  ** MISMATCH **"), fh)
        # CF-C: 64 random table entries; recount over item indices
        pick = torch.randint(0, self.C_keys.numel(), (64,), generator=self.gen)
        ok2 = 0
        for p in pick.tolist():
            k = int(self.C_keys[p]); x, y = k // self.V, k % self.V
            nx = (idx_all == x).sum(1).double(); ny = (idx_all == y).sum(1).double()
            rec = float((nx * ny).sum())
            ok2 += int(abs(rec - float(self.C_cnt[p])) < 0.5)
        # candidate-side join: the packed key of a random candidate found in the vocabulary must unpack to the same tuple
        d = self.te; s = torch.randint(0, d.N, (2000,), generator=self.gen)
        ck = self.pack(d.f[s].long(), d.t[s].long(), d.i[s].long())
        pos = torch.searchsorted(self.item_keys, ck).clamp(max=self.V - 1); hit = self.item_keys[pos] == ck
        kk = self.item_keys[pos][hit]
        f_, t_, i_ = kk // (self.k1 * self.k2), (kk // self.k2) % self.k1, kk % self.k2
        same = bool(((f_ == d.f[s].long()[hit]) & (t_ == d.t[s].long()[hit]) & (i_ == d.i[s].long()[hit])).all())
        log(f"    JOIN VERIFICATION CF-C: brute-force recount of 64 random (item,item) table entries matches in "
            f"{ok2}/64 cases; TEST-candidate key round-trip (unpack == stored tuple) on {int(hit.sum()):,} matched "
            f"candidates: {'OK' if same else '** MISMATCH **'}; candidate vocab hit-rate (TEST, 2000 events) = {float(hit.float().mean()):.4f}"
            + ("" if ok2 == 64 else "  ** MISMATCH **"), fh)
        self.join_ok = (ok == 64) and (ok2 == 64) and same
        if not self.join_ok:
            raise SystemExit(f"** {self.name}: CF join verification FAILED -- a misaligned lookup would inject noise; stop. **")

    def _standardise(self, fh):
        d = self.tr; dev = self.dev
        s = torch.arange(0, d.N, max(1, d.N // 1500))
        raws = []
        for c in range(0, s.numel(), 16):
            idx = s[c:c + 16]
            raw = self.cf_raw("tr", d, idx, dev, ("A", "B", "C"))              # (B,C,n)
            raws.append(raw[d.valid[idx].to(dev)].cpu())
        raw = torch.cat(raws)                                                    # (n_cand, n)
        self.mu = raw.mean(0); self.sd = raw.std(0).clamp(min=1e-6)
        names = feat_names(("A", "B", "C"), self.nb)
        self.coverage = {nm: float((raw[:, j] != 0).float().mean()) for j, nm in enumerate(names)}
        log(f"    standardisation (TRAIN candidates, LOO applied, {raw.shape[0]:,} candidates): "
            + "  ".join(f"{nm}: mu={self.mu[j]:.3f} sd={self.sd[j]:.3f} cov={self.coverage[nm]:.3f}" for j, nm in enumerate(names)), fh)
        # self-inclusion check: how much did LOO remove on train for candidates with a non-zero feature?
        idx = s[:64]
        with_loo = self.cf_raw("tr", d, idx, dev, ("A", "C"))
        no_loo = self.cf_raw("te", d, idx, dev, ("A", "C"))                    # split!='tr' -> no LOO, same data
        v = d.valid[idx].to(dev)
        diff = (no_loo - with_loo)[v]
        log(f"    self-inclusion check (train, 64 events): mean |feature change from leave-one-out| = "
            f"{float(diff.abs().mean()):.4f} raw units, on {float((diff.abs() > 1e-6).float().mean()):.4f} of candidates "
            f"-> LOO applied on train (a training event never sees its own history pairs)", fh)

    # ---------------------------------------------------------------- features
    def _hist_dev(self, d, idx, dev):
        hf, ht, hi, hb, hv, rec = self._hist(d, idx)
        return (hf.to(dev), ht.to(dev), hi.to(dev), hb.to(dev), hv.to(dev), rec.to(dev))

    @torch.no_grad()
    def cf_raw(self, split, d, idx, dev, groups):
        f = d.f[idx].long().to(dev); t = d.t[idx].long().to(dev); i = d.i[idx].long().to(dev)
        hf, ht, hi, hb, hv, rec = self._hist_dev(d, idx, dev)
        B, Cn = f.shape; H = hf.shape[1]; nb = self.nb; loo = split == "tr"
        out = []
        bmask = [((hb == b) & hv) for b in range(nb)]                            # (B,H) each

        def masked_stats(val, m, top3=False):
            # val (B,C,H) >= 0; m (B,H) mask; -> max, mean (0 if none) [, top-3 mean]
            vm = val.masked_fill(~m[:, None, :], 0.0)
            mx = vm.amax(-1); mean = vm.sum(-1) / m.sum(1, keepdim=True).float().clamp(min=1)
            if not top3:
                return [mx, mean]
            k = min(3, H)
            t3 = vm.topk(k, dim=-1).values.sum(-1) / m.sum(1, keepdim=True).float().clamp(min=1, max=3)
            return [mx, mean, t3]

        if "A" in groups:
            if loo:
                ne = torch.zeros(B, self.W, device=dev).scatter_add_(1, hf.masked_fill(~hv, 0), hv.float())   # n_e(code)
                Cxy = self.A_C[f[:, :, None], hf[:, None, :]].float()            # (B,C,H)
                ne_c = ne.gather(1, f)                                           # n_e(code(cand))  (B,C)
                ne_h = ne.gather(1, hf.masked_fill(~hv, 0))                      # n_e(code(hist))  (B,H)
                own = ne_c[:, :, None] * ne_h[:, None, :] - (f[:, :, None] == hf[:, None, :]).float() * ne_c[:, :, None]
                Cl = (Cxy - own).clamp(min=0)
                ua = (self.A_u[f].float() - ne_c).clamp(min=1)[:, :, None]
                uh = (self.A_u[hf].float() - ne_h).clamp(min=1)[:, None, :]
                pmi = torch.log((Cl * (self.A_U ** 2)).clamp(min=1e-30) / (self.A_T * ua * uh))
                ppmi = torch.where(Cl > 0, pmi.clamp(min=0), torch.zeros_like(pmi))
            else:
                ppmi = self.A_ppmi[f[:, :, None], hf[:, None, :]]              # (B,C,H)
            for b in range(nb):
                out += masked_stats(ppmi, bmask[b])
        if "B" in groups:
            m = (f[:, :, None] == hf[:, None, :])                                # same L0 code
            for b in range(nb):
                mb = (m & bmask[b][:, None, :]).float()
                out += [torch.log1p(mb.sum(-1)), (mb * rec[:, None, :]).sum(-1), (mb * rec[:, None, :]).amax(-1)]
        if "C" in groups:
            ck = self.pack(f, t, i); hk = self.pack(hf, ht, hi)
            ik = self.ik_dev
            cpos = torch.searchsorted(ik, ck).clamp(max=self.V - 1); chit = ik[cpos] == ck        # (B,C)
            hpos = torch.searchsorted(ik, hk).clamp(max=self.V - 1); hhit = (ik[hpos] == hk) & hv  # (B,H)
            pkey = cpos[:, :, None] * self.V + hpos[:, None, :]                    # (B,C,H)
            ppos = torch.searchsorted(self.C_keys, pkey).clamp(max=self.C_keys.numel() - 1)
            phit = (self.C_keys[ppos] == pkey) & chit[:, :, None] & hhit[:, None, :]
            if loo:
                cnt = self.C_cnt[ppos].float()
                nc = (ck[:, :, None] == hk[:, None, :]).float().mul(hv[:, None, :]).sum(-1)       # n_e(cand tuple) (B,C)
                nh = (hk[:, :, None] == hk[:, None, :]).float().mul(hv[:, None, :]).sum(-1)       # n_e(hist tuple) (B,H)
                own = nc[:, :, None] * nh[:, None, :] - (ck[:, :, None] == hk[:, None, :]).float() * nc[:, :, None]
                cl = cnt - own
                uc = (self.item_u[cpos].float() - nc).clamp(min=1)[:, :, None]
                uh = (self.item_u[hpos].float() - nh).clamp(min=1)[:, None, :]
                s = torch.where(phit & (cl >= self.min_cnt), cl / torch.sqrt(uc * uh), torch.zeros_like(cl))
            else:
                s = torch.where(phit, self.C_s[ppos], torch.zeros_like(pkey, dtype=torch.float))
            for b in range(nb):
                out += masked_stats(s, bmask[b], top3=True)
        return torch.stack(out, -1)

    def cf_feats(self, split, d, idx, dev, groups):
        raw = self.cf_raw(split, d, idx, dev, groups)
        names_all = feat_names(("A", "B", "C"), self.nb); sel = [names_all.index(n) for n in feat_names(groups, self.nb)]
        mu = self.mu[sel].to(dev); sd = self.sd[sel].to(dev)
        return ((raw - mu) / sd).clamp(-10.0, 10.0)          # z-scored; clipped so a rare sparse feature cannot blow up


# =========================================================================== pre-check
def precheck(pool, dev, fh, n_events=2000):
    """Univariate AUC (candidate == GT) and coverage per CF feature on TRAIN pools (covered events)."""
    d = pool.tr
    cov_idx = pool.tr.train_idx
    s = cov_idx[torch.randperm(cov_idx.numel(), generator=pool.gen)[:n_events]]
    names = feat_names(("A", "B", "C"), pool.nb)
    vals, labs = [], []
    for c in range(0, s.numel(), 16):
        idx = s[c:c + 16]
        raw = pool.cf_raw("tr", d, idx, dev, ("A", "B", "C"))
        v = d.valid[idx].to(dev)
        lab = torch.zeros(idx.numel(), d.C, dtype=torch.bool, device=dev)
        lab[torch.arange(idx.numel(), device=dev), d.gt_idx[idx].to(dev)] = True
        vals.append(raw[v].cpu()); labs.append(lab[v].cpu())
    X = torch.cat(vals); y = torch.cat(labs)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    log(f"\n  [{pool.name}] PRE-CHECK on {s.numel():,} covered TRAIN events: {X.shape[0]:,} candidates, {n_pos:,} positives", fh)
    log(f"    {'feature':<22}{'AUC':>8}{'corr':>8}{'coverage':>10}{'cov|pos':>9}{'cov|neg':>9}", fh)
    auc = {}
    for j, nm in enumerate(names):
        x = X[:, j].double(); xp, xn = x[y], torch.sort(x[~y]).values
        less = torch.searchsorted(xn, xp, right=False).double(); leq = torch.searchsorted(xn, xp, right=True).double()
        a = float(((less + 0.5 * (leq - less)) / n_neg).mean())
        yf = y.double(); corr = float(((x - x.mean()) * (yf - yf.mean())).mean() / (x.std() * yf.std() + 1e-12))
        auc[nm] = a
        log(f"    {nm:<22}{a:>8.4f}{corr:>8.4f}{float((x != 0).float().mean()):>10.4f}"
            f"{float((xp != 0).float().mean()):>9.4f}{float((xn != 0).double().mean()):>9.4f}", fh)
    verdict = {}
    for g in ("A", "B", "C"):
        gn = [n for n in names if n.startswith(g + ".")]
        best = max(abs(auc[n] - 0.5) for n in gn)
        verdict[g] = best >= 0.01
        log(f"    group CF-{g}: max |AUC-0.5| = {best:.4f} -> {'has univariate signal, TRAIN' if verdict[g] else 'NO univariate signal (< 0.01) -> SKIP its training cells'}", fh)
    return verdict, auc


# =========================================================================== train / eval
def eval_cf(model, pool, split, d, dev, groups, bs):
    model.eval(); sc = torch.empty(d.N, d.C)
    with torch.no_grad():
        for s in range(0, d.N, bs):
            idx = torch.arange(s, min(s + bs, d.N))
            f, t, i, scal = make_scal(pool, split, model, d, idx, dev, ())
            if groups:
                scal = torch.cat([scal, pool.cf_feats(split, d, idx, dev, groups)], -1)
            sc[idx] = model(f, t, i, scal, d.base[idx].float().to(dev), d.valid[idx].to(dev),
                            d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev)).cpu()
    return sc


def run_cell(pool, groups, cap, dev, args, fh):
    torch.manual_seed(args.seed); _t0 = time.perf_counter(); ep = 0
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    model = Ranker(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, n_cf(groups, pool.nb)).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = pool.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]])
    rb = ranks_of(pool.tr.base.float(), pool.tr)[0]
    band = torch.where(rb < 10, 0, torch.where(rb < 50, 1, 2)); tab = RW["rw1"]
    w_ev = torch.tensor([tab[int(b)] for b in band.tolist()], dtype=torch.float)
    rng = np.random.default_rng(args.seed)
    base_te = metrics(pool.te, pool.te.base.float())
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s:s + args.batch])
            f, t, i, scal = make_scal(pool, "tr", model, pool.tr, bidx, dev, ())
            if groups:
                scal = torch.cat([scal, pool.cf_feats("tr", pool.tr, bidx, dev, groups)], -1)
            sc = model(f, t, i, scal, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                       pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            loss = F.cross_entropy(sc, pool.tr.gt_idx[bidx].to(dev), reduction="none")
            w = w_ev[bidx].to(dev); loss = (loss * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = metrics(pool.va, eval_cf(model, pool, "va", pool.va, dev, groups, args.eval_batch))["t3@10"]
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
    te = metrics(pool.te, eval_cf(model, pool, "te", pool.te, dev, groups, args.eval_batch))
    assert abs(te["cov"] - base_te["cov"]) < 1e-9, "coverage changed under reranking -- BUG"
    return {"params": npar, "val": best_v, "n_train": int(len(idx_all)), "n_scalar": 28 + n_cf(groups, pool.nb),
            "train_secs": time.perf_counter() - _t0, "epochs_run": ep, "C": int(pool.te.C),
            "before": {k: v for k, v in base_te.items() if k not in ("hit", "r3")},
            "after": {k: v for k, v in te.items() if k not in ("hit", "r3")},
            "uid": pool.uid["te"].numpy(), "gt_b": pool.te.gt_b.numpy(), "has_gt": pool.te.has_gt.numpy(),
            "r3_before": base_te["r3"].numpy(), "r3_after": te["r3"].numpy(),
            "hit_before": base_te["hit"], "hit_after": te["hit"], "ent_z": np.zeros(pool.te.N)}


def paired_ci(a, b, rng, B):
    N = a.shape[0]; d = np.empty(B)
    for k in range(B):
        ii = rng.integers(0, N, N); d[k] = a[ii].mean() - b[ii].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def four_cells(r, other_cov):
    cm = r["has_gt"].astype(bool); both, own = cm & other_cov, cm & ~other_cov
    cells = []
    for grp in (both, own):
        for lo, hi in ((10, 50), (50, 10 ** 9)):
            mm = grp & (r["r3_before"] >= lo) & (r["r3_before"] < hi)
            cells.append(float((r["hit_after"][mm] > 0).mean()) if mm.sum() else float("nan"))
    return cells


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help='JSON {"MFD": {tr,va,te,sid,kpool}, "CH": {...}}')
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stage", default="all", choices=["precheck", "all"])
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min_cnt", type=int, default=2)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/cf_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "cf_report.txt"), "a")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(args.spec))
    log("=" * 100 + f"\n=== CF features for the reranker: generic or MFD-specific?  ({time.strftime('%Y-%m-%d %H:%M')}) ===\n" + "=" * 100, fh)

    res, skip = {}, {}
    for arm in ("MFD", "CH"):
        sp = spec[arm]
        log(f"\n{'#' * 100}\n# ARM {arm}: {os.path.basename(sp['te'])}  K_pool={sp.get('kpool')}\n{'#' * 100}", fh)
        pool = CFPool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"),
                      min_cnt=args.min_cnt, topn=args.topn, seed=args.seed)
        assert pool.te.N == 83005, "test split"
        pj = os.path.join(args.out_dir, f"precheck_{arm}.json")
        if args.stage == "all" and os.path.exists(pj):
            verdict = json.load(open(pj))["signal"]
            log(f"  [{arm}] pre-check loaded from {pj}: signal={verdict}", fh)
        else:
            verdict, auc = precheck(pool, dev, fh)
            json.dump({"auc": auc, "coverage": pool.coverage, "signal": verdict}, open(pj, "w"), indent=1)
        skip[arm] = {g for g, ok in verdict.items() if not ok}
        if args.stage == "precheck":
            del pool
            if dev == "cuda":
                torch.cuda.empty_cache()
            continue
        for cname, groups in CELLS:
            if any(g in skip[arm] for g in groups):
                log(f"\n  [{arm}] {cname}: SKIPPED (a member group has no univariate signal)", fh); continue
            cp = os.path.join(args.cache, f"CF_{arm}__{cname}__R2_rw1__capped.pt")
            if os.path.exists(cp):
                log(f"\n  [{arm}] {cname}: cached", fh)
            else:
                log(f"\n  >>> [{arm}] {cname}: groups={groups}  n_scalar={28 + n_cf(groups, pool.nb)}  C={pool.te.C:,}", fh)
                r = run_cell(pool, groups, args.cap, dev, args, fh)
                torch.save(r, cp)
            res[(arm, cname)] = canon(torch.load(cp, map_location="cpu", weights_only=False))
            r = res[(arm, cname)]
            log(f"  [{arm}] {cname}: t3@10 {r['before']['t3@10']:.4f} -> {r['after']['t3@10']:.4f}  "
                f"({r['train_secs']/60:.1f} min, {r['epochs_run']} ep)", fh)
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()
    if args.stage == "precheck":
        fh.close(); return

    # ---------------------------------------------------------------- report
    cells_done = [c for c, _ in CELLS if ("MFD", c) in res and ("CH", c) in res]
    m0, c0 = res[("MFD", "baseline")], res[("CH", "baseline")]
    N = m0["uid"].shape[0]
    assert np.array_equal(m0["uid"], c0["uid"]) and np.array_equal(m0["gt_b"], c0["gt_b"]), "cross-arm join on user_id failed"
    for k in res:
        assert np.array_equal(res[k]["uid"], m0["uid"])
    rng = np.random.default_rng(args.seed)
    mcov, ccov = m0["has_gt"].astype(bool), c0["has_gt"].astype(bool)

    log("\n" + "=" * 100 + "\n=== FULL METRICS (TEST, capped, R2_rw1) ===\n" + "=" * 100, fh)
    log(f"  reference: MFD K_pool=2000 t3@10={REF['MFD']['t3@10']} cov={REF['MFD']['cov']}; chain t3@10={REF['CH']['t3@10']} cov={REF['CH']['cov']}", fh)
    log(f"  {'arm':<4}{'cell':<10}{'nscal':>6}{'cov':>8}{'t3@10 bef':>11}{'aft':>8}{'gain':>9}{'P(r<10|c)':>10}{'t3@5':>8}{'t3@20':>8}"
        f"{'n@5':>8}{'n@10':>8}{'t2@10':>8}{'t2n@10':>8}{'min':>6}{'ep':>4}", fh)
    for arm in ("MFD", "CH"):
        for c in cells_done:
            r = res[(arm, c)]; b, a = r["before"], r["after"]
            log(f"  {arm:<4}{c:<10}{r['n_scalar']:>6}{a['cov']:>8.4f}{b['t3@10']:>11.4f}{a['t3@10']:>8.4f}{a['t3@10']-b['t3@10']:>+9.4f}"
                f"{a['pcov']:>10.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}{a['t3n@5']:>8.4f}{a['t3n@10']:>8.4f}{a['t2@10']:>8.4f}{a['t2n@10']:>8.4f}"
                f"{r['train_secs']/60:>6.1f}{r['epochs_run']:>4}", fh)
    log("\n  per-behavior t3@10 (cell vs the arm's baseline cell):", fh)
    for arm in ("MFD", "CH"):
        base = res[(arm, "baseline")]["after"]["perbeh"]
        for c in cells_done:
            aa = res[(arm, c)]["after"]["perbeh"]; dd = {k: aa[k] - base[k] for k in aa}
            conv = [k for k in dd if k != "pv"]
            note = ("BOTH improve" if dd.get("pv", 0) > 0 and all(dd[k] >= 0 for k in conv)
                    else "pv up, conversions mixed/down" if dd.get("pv", 0) > 0 else "pv down")
            log(f"    {arm:<4}{c:<10}" + "  ".join(f"{k}:{aa[k]:.4f}({dd[k]:+.4f})" for k in aa) + f"  [{note if c != 'baseline' else 'baseline'}]", fh)

    log("\n" + "=" * 100 + "\n=== FOUR DIAGNOSTIC CELLS (BOTH/10-49, BOTH/50+, own/10-49, own/50+) ===\n" + "=" * 100, fh)
    log(f"  reference K_pool=500 rw1 capped: MFD {REF['MFD']['cells']}  chain {REF['CH']['cells']}", fh)
    for arm, other in (("MFD", ccov), ("CH", mcov)):
        for c in cells_done:
            fc = four_cells(res[(arm, c)], other)
            log(f"  {arm:<4}{c:<10} BOTH/10-49={fc[0]:.4f}  BOTH/50+={fc[1]:.4f}  own/10-49={fc[2]:.4f}  own/50+={fc[3]:.4f}", fh)

    log("\n" + "=" * 100 + "\n=== VERDICT 1 -- ABSOLUTE: gain over the arm's own baseline (paired bootstrap over events, B=2000) ===\n" + "=" * 100, fh)
    gains = {}
    for arm in ("MFD", "CH"):
        hb = res[(arm, "baseline")]["hit_after"]
        for c in cells_done:
            if c == "baseline":
                continue
            ha = res[(arm, c)]["hit_after"]; g = ha.mean() - hb.mean()
            lo, hi, p = paired_ci(ha, hb, rng, args.boot); gains[(arm, c)] = (g, lo, hi)
            vs_ref = res[(arm, c)]["after"]["t3@10"] - REF[arm]["t3@10"]
            log(f"  {arm:<4}{c:<10} gain {g:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0)={p:.3f}  vs fixed ref {REF[arm]['t3@10']}: {vs_ref:+.4f}"
                f"  -> {'WORKS (lower bound > 0)' if lo > 0 else 'HURTS (upper bound < 0)' if hi < 0 else 'no significant gain'}", fh)

    log("\n" + "=" * 100 + "\n=== VERDICT 2 -- DIFFERENTIAL: MFD gain / chain gain, and does MFD's lead widen? ===\n" + "=" * 100, fh)
    log("  hard match reference ratio 5.3:1. Paired CI of (MFD gain - chain gain) on events joined by user_id; if it covers 0", fh)
    log("  the feature is GENERIC: absolute numbers rise but MFD's lead does not widen.", fh)
    hmb, hcb = m0["hit_after"], c0["hit_after"]
    lead0, l0lo, l0hi, _ = (hmb.mean() - hcb.mean(),) + paired_ci(hmb, hcb, rng, args.boot)[:3]
    log(f"  baseline lead MFD - chain: {lead0:+.4f}  95% CI [{l0lo:+.4f}, {l0hi:+.4f}]   (current reference +0.0014 [+0.0007, +0.0022])", fh)
    for c in cells_done:
        if c == "baseline":
            continue
        gm, gc = gains[("MFD", c)][0], gains[("CH", c)][0]
        ratio = gm / gc if abs(gc) > 1e-5 else float("inf") if gm > 0 else float("nan")
        dm = res[("MFD", c)]["hit_after"] - hmb; dc = res[("CH", c)]["hit_after"] - hcb
        lo, hi, p = paired_ci(dm, dc, rng, args.boot)
        hm, hc = res[("MFD", c)]["hit_after"], res[("CH", c)]["hit_after"]
        llo, lhi, lp = paired_ci(hm, hc, rng, args.boot)
        kind = ("MFD-SPECIFIC (lead widens significantly)" if lo > 0 else
                "CHAIN-favouring (lead narrows significantly)" if hi < 0 else
                "GENERIC (lead does not widen: CI of the gain difference covers 0)")
        log(f"  {c:<10} MFD gain {gm:+.4f} / chain gain {gc:+.4f} = ratio {ratio:.2f}:1   gain difference {gm-gc:+.4f} CI [{lo:+.4f}, {hi:+.4f}]"
            f"   lead with feature {hm.mean()-hc.mean():+.4f} CI [{llo:+.4f}, {lhi:+.4f}] P(MFD>chain)={lp:.3f}   -> {kind}", fh)
    sc_cells = [c for c in ("+CF-A", "+CF-B", "+CF-A+B") if c in cells_done]
    if sc_cells:
        widen = [c for c in sc_cells if paired_ci(res[("MFD", c)]["hit_after"] - hmb, res[("CH", c)]["hit_after"] - hcb, rng, args.boot)[0] > 0]
        log(f"\n  EXPECTATION CHECK -- shared-code features (MFD L0 = cross-modal commonality code, entropy ratio 0.929; chain L0 = first residual):", fh)
        log(f"    expected to differentiate; cells that significantly widen MFD's lead: {widen if widen else 'NONE'} -> "
            f"{'expectation HOLDS' if widen else 'expectation does NOT hold: the shared-code CF signal is generic'}", fh)
    json.dump({f"{a}|{c}": {"t3@10": res[(a, c)]["after"]["t3@10"], "gain": gains.get((a, c), (0, 0, 0))}
               for a, c in res}, open(os.path.join(args.out_dir, "cf_results.json"), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
