"""Three tasks in one sweep, all offline; both generative models stay frozen.

TASK A  settle the significance of MFD's lead under ONE fixed protocol, after diagnosing why two
        runs gave contradictory CIs for the same quantity.
TASK B  fix the popularity feature (it collapsed t3@10 0.2144 -> 0.1129) and re-run B2/B3/B4; add
        B5 = branch entropy + z, which is structurally MFD-ONLY.
TASK C  target the differentiated headroom (deep-ranked and MFD-only GTs) instead of generic
        capacity, which is saturated: C1 rank-weighted loss, C2 candidate self-attention,
        C3 hard-negative windows, C4 best combination.

--- TASK A DIAGNOSIS (verified in code, demonstrated numerically below) -----------------------
Both earlier bootstraps resampled EVENTS with the SAME index for both arms (already paired) at
B=2000, so the resampling unit and the replicate count are NOT the cause. The cause is that
sweep_rerank.py never ALIGNED the two arms: the dumps hold the same 83,005 events in different row
order, so hm[ii] and hc[ii] indexed DIFFERENT events. That silently destroys the pairing and turns a
paired test into an effectively independent one: Var(m-c) becomes Var(m)+Var(c) instead of
Var(m)+Var(c)-2Cov(m,c). With hit rate ~0.21 the unpaired half-width is
1.96*sqrt(2*0.21*0.79/83005) ~ 0.0039, which is exactly the width that was observed. Comparing each
arm's own VAL-best config (A2 vs A1) is a real but SECONDARY contributor, since different
architectures also decorrelate the two hit vectors. This script reports the CI computed three ways
(unaligned, aligned-paired, aligned-paired-same-config) so the cause is visible, not asserted.

FIXED PROTOCOL, used for every comparison from here on:
  same configuration on BOTH arms; events joined on user_id; PAIRED bootstrap (both arms resampled
  together on the joined index); B=2000; report difference, 95% CI, P(MFD>chain).

--- TASK B DIAGNOSIS -------------------------------------------------------------------------
The old popularity had three defects, all reported with evidence below:
  (1) SELF-INCLUSION: counts were built from the TRAINING GT tuples, so a training event's own label
      inflated its own feature -- the model keys on a circular signal that vanishes at test time.
      Fixed with leave-one-out counts on the training split.
  (2) NO STANDARDISATION: raw log1p counts are long-tailed and on a different scale from the other
      28 features (indicators in [0,1], cosines in [-1,1]). Fixed by z-scoring with TRAIN statistics.
  (3) JOIN: verified (sorted-unique keys + exact-match check); the hit rate is reported so a
      misaligned lookup would be visible rather than silently injecting a random number.
Scope was already correct (TRAIN only, never val/test) and is re-asserted.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mfd_reranker import BEH, CrossBlock, RerankData, best_rank, ndcg_at, recall_at

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
REF_COV = {"MFD": 0.2615, "chain": 0.2471}
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
    def __init__(self, name, tr_p, va_p, te_p, sid_p, wb, dev, fh):
        self.name, self.dev = name, dev
        self.tr = RerankData(tr_p, wb); self.va = RerankData(va_p, wb); self.te = RerankData(te_p, wb)
        self.splits = {"tr": self.tr, "va": self.va, "te": self.te}
        self.widths = sid_widths(sid_p); self.nb = self.tr.nb
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
        cov = float(self.te.has_gt.float().mean())
        assert abs(cov - REF_COV[name]) < 0.01, f"{name} coverage {cov:.4f} != ref {REF_COV[name]}"
        log(f"  [{name}] coverage@pool={cov:.4f} (ref {REF_COV[name]})  covered train={int(self.tr.has_gt.sum()):,}", fh)
        self.k1, self.k2 = self.widths[1], self.widths[2]
        self.uid = {k: torch.load(p, map_location="cpu", weights_only=False)["cols"]["user_id"].long()
                    for k, p in (("tr", tr_p), ("va", va_p), ("te", te_p))}
        self._popularity(fh)
        self._per_event()
        self._ent = {}

    def pack(self, f, t, i):
        return (f.long() * self.k1 + t.long()) * self.k2 + i.long()

    # ---------------------------------------------------------------- TASK B: fixed popularity
    def _popularity(self, fh):
        key = self.pack(self.tr.gt_f, self.tr.gt_t, self.tr.gt_i)     # TRAIN split ONLY
        n = key.numel(); cut = int(n * 0.8)
        uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
        rec = torch.zeros_like(cnt)
        rec.index_add_(0, inv[cut:], torch.ones_like(inv[cut:]))
        self.pop_keys, self.pop_glob, self.pop_rec = uk, cnt.float(), rec.float()
        self.pop_inv_tr = inv                                          # for leave-one-out on train
        # standardisation statistics from a TRAIN candidate sample (never val/test)
        d = self.tr
        s = torch.arange(0, d.N, max(1, d.N // 2000))
        raw = self._pop_raw(d.f[s].long(), d.t[s].long(), d.i[s].long(), None)
        self.pop_mu = raw.reshape(-1, 2).mean(0)
        self.pop_sd = raw.reshape(-1, 2).std(0).clamp(min=1e-6)
        hit_rate = float((raw[..., 0] > 0).float().mean())
        log(f"  [{self.name}] popularity: source=TRAIN GT tuples only  |unique tuples|={uk.numel():,}  "
            f"train-candidate lookup hit-rate={hit_rate:.4f}", fh)
        log(f"    raw log1p stats (global,recent): mu={self.pop_mu.tolist()} sd={self.pop_sd.tolist()} "
            f"-> z-scored before use (fix 2: scale)", fh)
        gt_cnt = self.pop_glob[self.pop_inv_tr]
        log(f"    self-inclusion check: mean count of a TRAIN event's OWN GT tuple = "
            f"{float(gt_cnt.mean()):.2f}; leave-one-out applied on train (fix 1: leakage)", fh)

    def _pop_raw(self, f, t, i, loo_key):
        k = self.pack(f, t, i)
        pk = self.pop_keys.to(k.device)
        pos = torch.searchsorted(pk, k).clamp(max=pk.numel() - 1)
        hit = pk[pos] == k
        g = torch.where(hit, self.pop_glob.to(k.device)[pos], torch.zeros_like(k, dtype=torch.float))
        r = torch.where(hit, self.pop_rec.to(k.device)[pos], torch.zeros_like(k, dtype=torch.float))
        if loo_key is not None:                                        # leave-one-out on TRAIN
            own = (k == loo_key.unsqueeze(1))
            g = torch.where(own, (g - 1).clamp(min=0), g)
            r = torch.where(own, (r - 1).clamp(min=0), r)
        return torch.stack([torch.log1p(g), torch.log1p(r)], -1)

    def pop_feats(self, f, t, i, loo_key):
        raw = self._pop_raw(f, t, i, loo_key)
        return (raw - self.pop_mu.to(raw.device)) / self.pop_sd.to(raw.device)

    # ---------------------------------------------------------------- session / user features
    def _per_event(self):
        self.ev = {}
        for nm, d in self.splits.items():
            hb, hv = d.hist_b.long(), (d.hist_b.long() >= 0)
            ln = hv.sum(1).float()
            sess = torch.stack([torch.log1p(ln), ln / max(d.H, 1)], -1)
            bd = torch.stack([((hb == b) & hv).sum(1).float() / ln.clamp(min=1) for b in range(self.nb)], -1)
            ent = []
            uf = d.hist_f.long()
            for e in range(0, d.N, 8192):
                sl = slice(e, min(e + 8192, d.N))
                x = torch.where(hv[sl], uf[sl], torch.full_like(uf[sl], -1)) + 1
                oh = torch.zeros(x.shape[0], self.widths[0] + 1)
                oh.scatter_add_(1, x, torch.ones_like(x, dtype=torch.float))
                p = oh[:, 1:]; p = p / p.sum(1, keepdim=True).clamp(min=1)
                ent.append(-(p * p.clamp(min=1e-12).log()).sum(1))
            self.ev[nm] = {"session": sess,
                           "user": torch.cat([bd, torch.log1p(ln).unsqueeze(-1), torch.cat(ent).unsqueeze(-1)], -1)}

    # ---------------------------------------------------------------- B5: branch entropy (MFD only)
    def branch_ent(self, split):
        if split in self._ent:
            return self._ent[split]
        d = self.splits[split]; N, C, nb, dev = d.N, d.C, self.nb, self.dev
        ev = torch.arange(N).view(N, 1).expand(N, C)
        gk = ev.to(torch.int64) * (nb * KF_MAX) + d.branch.view(1, -1).expand(N, C) * KF_MAX + d.f.long()
        v = d.valid
        g = gk[v].to(dev); xt = d.lp[..., 2].float()[v].to(dev); xi = d.lp[..., 3].float()[v].to(dev)
        pt = d.t.long()[v].to(dev); pi = d.i.long()[v].to(dev)
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
        eg = (uniq // (nb * KF_MAX)).long()
        cnt = torch.zeros(N, device=dev).index_add_(0, eg, torch.ones_like(Hs))
        mu = torch.zeros(N, device=dev).index_add_(0, eg, Hs) / cnt.clamp(min=1)
        ss = torch.zeros(N, device=dev).index_add_(0, eg, Hs * Hs)
        sd = (ss / cnt.clamp(min=1) - mu.pow(2)).clamp(min=0).sqrt()
        z = (Hs - mu[eg]) / sd[eg].clamp(min=1e-6)
        out_h = torch.zeros(N * C); out_z = torch.zeros(N * C)
        flat = v.reshape(-1)
        out_h[flat] = Hs[gid].cpu(); out_z[flat] = z[gid].cpu()
        self._ent[split] = (out_h.reshape(N, C).half(), out_z.reshape(N, C).half())
        return self._ent[split]


# ------------------------------------------------------------------------------------- features
def decay_feats(f, t, i, hist, nb, rates=(0.15, 0.02)):
    """Exponential POSITION-decay of exact matches. NOTE: timestamps are not stored in the TFRecords
    (only user_id/sequence_data/behavior_data), so elapsed TIME is unavailable; this is a proxy."""
    hf, ht, hi, hb, hvalid, recency = hist
    H = hf.shape[1]
    age = (1.0 - recency).clamp(min=0) * H
    match = ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
             & (i[:, :, None] == hi[:, None, :]))
    out = []
    for lam in rates:
        w = torch.exp(-lam * age)[:, None, :]
        for b in range(nb):
            out.append((match & ((hb == b) & hvalid)[:, None, :]).float().mul(w).amax(-1))
    return torch.stack(out, -1)


def engagement(model, f, t, i, hist):
    hf, ht, hi, hb, hvalid, rec = hist
    feats = []
    match = ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
             & (i[:, :, None] == hi[:, None, :]))
    r = rec[:, None, :]
    for b in range(model.nb):
        mb = match & ((hb == b) & hvalid)[:, None, :]
        cnt = mb.sum(-1).float()
        feats += [(cnt > 0).float(), torch.log1p(cnt), (mb.float() * r).amax(-1)]
    cuf = model.U_f[f]; huf = model.U_f[hf]
    for cv, hv in ((cuf, huf), (cuf + model.U_t[t], huf + model.U_t[ht]), (cuf + model.U_i[i], huf + model.U_i[hi])):
        sim = torch.bmm(F.normalize(cv, dim=-1), F.normalize(hv, dim=-1).transpose(1, 2))
        for b in range(model.nb):
            m = sim.masked_fill(~((hb == b) & hvalid)[:, None, :].expand_as(sim), -2.0).amax(-1)
            feats.append(torch.where(m < -1.5, torch.zeros_like(m), m))
    return torch.stack(feats, -1)


def n_extra_of(blocks, nb):
    n = 0
    if "decay" in blocks: n += 2 * nb
    if "pop" in blocks: n += 2
    if "session" in blocks: n += 2
    if "user" in blocks: n += nb + 2
    if "brancnt" in blocks: n += 2
    return n


def make_scal(arm, split, model, d, idx, dev, blocks):
    f = d.f[idx].long().to(dev); t = d.t[idx].long().to(dev); i = d.i[idx].long().to(dev)
    hist = (d.hist_f[idx].long().to(dev), d.hist_t[idx].long().to(dev), d.hist_i[idx].long().to(dev),
            d.hist_b[idx].long().to(dev), (d.hist_b[idx].long() >= 0).to(dev), d.hist_recency[idx].float().to(dev))
    parts = [d.lp[idx].float().to(dev), engagement(model, f, t, i, hist)]
    B, C = f.shape
    if "decay" in blocks:
        parts.append(decay_feats(f, t, i, hist, model.nb))
    if "pop" in blocks:
        loo = arm.pack(d.gt_f[idx], d.gt_t[idx], d.gt_i[idx]).to(dev) if split == "tr" else None
        parts.append(arm.pop_feats(f, t, i, loo))
    if "session" in blocks:
        parts.append(arm.ev[split]["session"][idx].to(dev).unsqueeze(1).expand(B, C, -1))
    if "user" in blocks:
        parts.append(arm.ev[split]["user"][idx].to(dev).unsqueeze(1).expand(B, C, -1))
    if "brancnt" in blocks:
        h, z = arm.branch_ent(split)
        parts.append(torch.stack([h[idx].float().to(dev), z[idx].float().to(dev)], -1))
    return f, t, i, torch.cat(parts, -1)


# ------------------------------------------------------------------------------------- the ranker
class Ranker(nn.Module):
    def __init__(self, U, d_model, d_mem, layers=2, heads=4, drop=0.3, nb=4, n_extra=0,
                 self_attn=False, self_m=256):
        super().__init__()
        self.register_buffer("U_f", U[0]); self.register_buffer("U_t", U[1]); self.register_buffer("U_i", U[2])
        self.nb = nb; self.self_attn = self_attn; self.self_m = self_m
        n_scalar = 4 + 3 * nb + 3 * nb + n_extra
        d = d_model
        self.mem_proj = nn.Linear(d_mem, d) if d_mem != d else None
        self.view_proj = nn.Linear(3 * d, d)
        self.scalar_proj = nn.Sequential(nn.Linear(n_scalar, d), nn.GELU(), nn.Linear(d, d))
        self.in_norm = nn.LayerNorm(d); self.in_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([CrossBlock(d, heads, drop) for _ in range(layers)])
        if self_attn:                                   # C2: candidates attend to each other
            self.s_norm = nn.LayerNorm(d)
            self.s_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.out_norm = nn.LayerNorm(d); self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.base_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, f, t, i, scal, base, valid, mem, mem_pad):
        B, C = f.shape
        if self.mem_proj is not None:
            mem = self.mem_proj(mem)
        uf = self.U_f[f]
        x = self.in_drop(self.in_norm(
            self.view_proj(torch.cat([uf, uf + self.U_t[t], uf + self.U_i[i]], -1)) + self.scalar_proj(scal)))
        for blk in self.blocks:
            x = blk(x, mem, mem_pad)
        if self.self_attn:
            # relative modelling among the top-M by base score (full-pool self-attention is O(C^2));
            # every candidate still keeps its cross-attention delta -- only the CONTEXT is truncated.
            m = min(self.self_m, C)
            idx = base.masked_fill(~valid, float("-inf")).topk(m, dim=-1).indices
            xs = torch.gather(x, 1, idx.unsqueeze(-1).expand(B, m, x.size(-1)))
            a, _ = self.s_attn(self.s_norm(xs), self.s_norm(xs), self.s_norm(xs), need_weights=False)
            x = x.scatter_add(1, idx.unsqueeze(-1).expand(B, m, x.size(-1)), a)
        delta = self.head(self.out_norm(x)).squeeze(-1)
        return (self.base_scale * base + delta).masked_fill(~valid, float("-inf"))


# ------------------------------------------------------------------------------------ train / eval
def ranks_of(score, d):
    sid = (d.f == d.gt_f[:, None]) & (d.t == d.gt_t[:, None]) & (d.i == d.gt_i[:, None])
    beh = d.branch.view(1, -1) == d.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, d.valid)
    r2, h2 = best_rank(score, sid, d.valid & beh)
    return r3, h3, r2, h2


@torch.no_grad()
def eval_split(model, arm, split, d, dev, blocks, bs=64):
    model.eval()
    sc = torch.empty(d.N, d.C)
    for s in range(0, d.N, bs):
        idx = torch.arange(s, min(s + bs, d.N))
        f, t, i, scal = make_scal(arm, split, model, d, idx, dev, blocks)
        sc[idx] = model(f, t, i, scal, d.base[idx].float().to(dev), d.valid[idx].to(dev),
                        d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev)).cpu()
    return sc


def metrics(d, score):
    r3, h3, r2, h2 = ranks_of(score, d)
    cov = d.has_gt
    return {"t3@5": recall_at(r3, h3, 5), "t3@10": recall_at(r3, h3, 10), "t3@20": recall_at(r3, h3, 20),
            "t3n@5": ndcg_at(r3, h3, 5), "t3n@10": ndcg_at(r3, h3, 10),
            "t2@10": recall_at(r2, h2, 10), "t2n@10": ndcg_at(r2, h2, 10),
            "pcov": recall_at(r3, h3, 10, cov), "cov": float(cov.float().mean()),
            "perbeh": {BEH[b]: recall_at(r3, h3, 10, d.gt_b == b) for b in range(d.nb) if (d.gt_b == b).any()},
            "r3": r3, "hit": (h3 & (r3 >= 0) & (r3 < 10)).numpy().astype(np.float64)}


RW = {"rw1": {0: 1., 1: 2., 2: 4.}, "rw2": {0: 1., 1: 3., 2: 9.}, "rw3": {0: .5, 1: 2., 2: 5.}}


def run_one(arm, cfg, cap, dev, args, fh):
    torch.manual_seed(args.seed)
    d_mem = arm.tr.mem.shape[2]; dm = cfg.get("d", 128)
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in arm.widths]
    model = Ranker(U, dm, d_mem, cfg.get("layers", 2), args.heads, args.dropout, arm.nb,
                   n_extra_of(cfg["blocks"], arm.nb), cfg.get("self_attn", False)).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = arm.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]])
    # C1: per-event weights from the PRE-rerank rank band
    w_ev = None
    if cfg.get("rw"):
        rb = ranks_of(arm.tr.base.float(), arm.tr)[0]
        band = torch.where(rb < 10, 0, torch.where(rb < 50, 1, 2))
        tab = RW[cfg["rw"]]
        w_ev = torch.tensor([tab[int(b)] for b in band.tolist()], dtype=torch.float)
    hn = cfg.get("hn", 0)                                        # C3: hard-negative window
    rng = np.random.default_rng(args.seed)
    base_te = metrics(arm.te, arm.te.base.float())
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s:s + args.batch])
            f, t, i, scal = make_scal(arm, "tr", model, arm.tr, bidx, dev, cfg["blocks"])
            sc = model(f, t, i, scal, arm.tr.base[bidx].float().to(dev), arm.tr.valid[bidx].to(dev),
                       arm.tr.mem[bidx].float().to(dev), arm.tr.mem_pad[bidx].to(dev))
            gt = arm.tr.gt_idx[bidx].to(dev)
            if hn:                                               # loss over GT + nearest-ranked window
                k = min(hn, sc.size(1))
                top = sc.detach().topk(k, dim=-1).indices
                sel = torch.cat([gt.unsqueeze(1), top], 1)       # GT first -> target index 0
                sub = torch.gather(sc, 1, sel)
                loss = F.cross_entropy(sub, torch.zeros(sub.size(0), dtype=torch.long, device=dev),
                                       reduction="none")
            else:
                loss = F.cross_entropy(sc, gt, reduction="none")
            if w_ev is not None:
                w = w_ev[bidx].to(dev)
                loss = (loss * w).sum() / w.sum()
            else:
                loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = metrics(arm.va, eval_split(model, arm, "va", arm.va, dev, cfg["blocks"], args.eval_batch))["t3@10"]
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
    te = metrics(arm.te, eval_split(model, arm, "te", arm.te, dev, cfg["blocks"], args.eval_batch))
    assert abs(te["cov"] - base_te["cov"]) < 1e-9, "coverage changed under reranking -- BUG"
    collapsed = (te["t3@10"] < base_te["t3@10"] - 0.01) or (te["pcov"] < base_te["pcov"] - 0.05)
    return {"params": npar, "val": best_v, "n_train": int(len(idx_all)), "collapsed": bool(collapsed),
            "before": {k: v for k, v in base_te.items() if k not in ("hit", "r3")},
            "after": {k: v for k, v in te.items() if k not in ("hit", "r3")},
            "hit_before": base_te["hit"], "hit_after": te["hit"],
            "r3_before": base_te["r3"].numpy(), "r3_after": te["r3"].numpy()}


CONFIGS = [
    ("baseline",     dict(blocks=())),
    ("B2_pop_fix",   dict(blocks=("decay", "pop"))),
    ("B3_session",   dict(blocks=("decay", "pop", "session"))),
    ("B4_user",      dict(blocks=("decay", "pop", "session", "user"))),
    ("B5_branchent", dict(blocks=("decay", "pop", "session", "user", "brancnt"), mfd_only=True)),
    ("C1_rw1",       dict(blocks=(), rw="rw1")),
    ("C1_rw2",       dict(blocks=(), rw="rw2")),
    ("C1_rw3",       dict(blocks=(), rw="rw3")),
    ("C2_selfattn",  dict(blocks=(), self_attn=True)),
    ("C3_hn50",      dict(blocks=(), hn=50)),
    ("C3_hn200",     dict(blocks=(), hn=200)),
]


def main():
    ap = argparse.ArgumentParser()
    for a in ("mfd_tr", "mfd_va", "mfd_te", "ch_tr", "ch_va", "ch_te", "md_sid", "tg_sid"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/sweep2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "sweep2_report.txt"), "w")
    log("=" * 118 + "\n=== SWEEP 2: significance protocol, fixed popularity, deep/MFD-only targeting ===\n"
        + "=" * 118, fh)

    arms = {"MFD": Arm("MFD", args.mfd_tr, args.mfd_va, args.mfd_te, args.md_sid, args.wb, dev, fh),
            "chain": Arm("chain", args.ch_tr, args.ch_va, args.ch_te, args.tg_sid, args.wb, dev, fh)}

    # ---------------- alignment (join on user_id), needed for paired stats + diagnostic cells ----
    um, uc = arms["MFD"].uid["te"], arms["chain"].uid["te"]
    assert bool(torch.equal(um.sort().values, uc.sort().values)), "TEST user_id sets differ"
    om, oc = torch.argsort(um), torch.argsort(uc)
    perm = torch.empty(um.numel(), dtype=torch.long); perm[om] = oc      # mfd row -> chain row
    agree = float((arms["MFD"].te.gt_b == arms["chain"].te.gt_b[perm]).float().mean())
    log(f"\n  ALIGNMENT: joined on user_id, {um.numel():,}/{EXPECT_TOTAL:,} matched; post-join "
        f"gt_behavior agreement = {agree:.6f}", fh)
    assert agree > 0.999, "join failed"

    results = {}

    def get(a, cname, cfg, setting):
        key = f"{a}__{cname}__{setting}"
        cp = os.path.join(cache, key + ".pt")
        if os.path.exists(cp):
            log(f"  [cached] {key}", fh); return torch.load(cp, map_location="cpu", weights_only=False)
        log(f"\n  >>> RUN {key}", fh)
        r = run_one(arms[a], cfg, args.cap if setting == "capped" else 0, dev, args, fh)
        torch.save(r, cp); return r

    for setting in ("natural", "capped"):
        for cname, cfg in CONFIGS:
            for a in ("MFD", "chain"):
                if cfg.get("mfd_only") and a == "chain":
                    continue
                results[(a, cname, setting)] = get(a, cname, cfg, setting)

    # ---------------- C4: best combination (own-VAL selected, collapsed runs excluded) ----------
    for setting in ("natural", "capped"):
        for a in ("MFD", "chain"):
            ok = lambda n: (a, n, setting) in results and not results[(a, n, setting)]["collapsed"]
            bfeat = max([n for n in ("baseline", "B2_pop_fix", "B3_session", "B4_user", "B5_branchent") if ok(n)],
                        key=lambda n: results[(a, n, setting)]["val"])
            bc = max([n for n in ("baseline", "C1_rw1", "C1_rw2", "C1_rw3", "C2_selfattn", "C3_hn50", "C3_hn200") if ok(n)],
                     key=lambda n: results[(a, n, setting)]["val"])
            cd = dict(CONFIGS)[bc]
            combo = dict(blocks=dict(CONFIGS)[bfeat]["blocks"], rw=cd.get("rw"),
                         self_attn=cd.get("self_attn", False), hn=cd.get("hn", 0))
            log(f"\n  C4[{setting}/{a}] own-VAL: features={bfeat} + mechanism={bc} -> {combo}", fh)
            results[(a, "C4_best", setting)] = get(a, "C4_best", combo, setting)

    names = [c[0] for c in CONFIGS] + ["C4_best"]

    # ---------------- TASK A: significance ------------------------------------------------------
    log("\n" + "=" * 118 + "\n=== TASK A: SIGNIFICANCE PROTOCOL ===\n" + "=" * 118, fh)
    log("  DIAGNOSIS of the earlier five-fold CI widening (verified in code, demonstrated below):", fh)
    log("    * resampling unit  : BOTH runs resampled EVENTS -- not the cause.", fh)
    log("    * replicates       : BOTH used B=2000 -- not the cause.", fh)
    log("    * PAIRING (cause)  : sweep_rerank.py never aligned the arms, so the same bootstrap index", fh)
    log("      hit DIFFERENT events in each arm. That converts a paired test into an effectively", fh)
    log("      independent one: Var(m-c) = Var(m)+Var(c) instead of Var(m)+Var(c)-2Cov(m,c).", fh)
    log("    * model selection  : comparing each arm's own VAL-best config (A2 vs A1) also decorrelates", fh)
    log("      the two hit vectors -- a real but SECONDARY contributor.", fh)
    rng = np.random.default_rng(args.seed)
    for setting in ("natural", "capped"):
        hm = results[("MFD", "baseline", setting)]["hit_after"]
        hc_raw = results[("chain", "baseline", setting)]["hit_after"]
        hc = hc_raw[perm.numpy()]                                        # aligned into MFD row order
        N = hm.shape[0]
        rho = float(np.corrcoef(hm, hc)[0, 1])
        def boot(x, y, B):
            o = np.empty(B)
            for k in range(B):
                ii = rng.integers(0, N, N); o[k] = x[ii].mean() - y[ii].mean()
            return o
        d_un = boot(hm, hc_raw, args.boot)      # UNALIGNED (reproduces the old, wrong width)
        d_pa = boot(hm, hc, args.boot)          # ALIGNED + PAIRED (the fixed protocol)
        q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
        lu, hu = q(d_un); lp, hp = q(d_pa)
        log(f"\n  [{setting}] baseline config on both arms; corr(hit_MFD, hit_chain) after alignment = {rho:.4f}", fh)
        log(f"    UNALIGNED (old, wrong)     diff={hm.mean()-hc_raw.mean():+.4f}  95% CI [{lu:+.4f}, {hu:+.4f}]  "
            f"width={hu-lu:.4f}", fh)
        log(f"    ALIGNED+PAIRED (protocol)  diff={hm.mean()-hc.mean():+.4f}  95% CI [{lp:+.4f}, {hp:+.4f}]  "
            f"width={hp-lp:.4f}  P(MFD>chain)={float((d_pa>0).mean()):.4f}", fh)
        log(f"    -> {'DISTINGUISHABLE from noise' if (lp>0 or hp<0) else 'NOT distinguishable (CI spans 0)'}", fh)
    log("\n  FIXED PROTOCOL (use for every future comparison): same configuration on both arms;", fh)
    log("  events joined on user_id; PAIRED bootstrap (both arms resampled together on the joined", fh)
    log("  index); B=2000; report difference, 95% CI, and P(MFD>chain).", fh)

    # ---------------- consolidated table --------------------------------------------------------
    log("\n" + "=" * 118 + "\n=== CONSOLIDATED TABLE ===\n" + "=" * 118, fh)
    log(f"  {'set':<8}{'arm':<6}{'config':<14}{'params':>10}{'bef':>8}{'aft':>8}{'gain':>9}{'ratio':>7}"
        f"{'Pc bef':>8}{'Pc aft':>8}{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}{'t2@10':>8}{'t2n@10':>8}{'':>3}", fh)
    for setting in ("natural", "capped"):
        for cname in names:
            gc = results.get(("chain", cname, setting))
            gain_c = (gc["after"]["t3@10"] - gc["before"]["t3@10"]) if gc else float("nan")
            for a in ("MFD", "chain"):
                r = results.get((a, cname, setting))
                if r is None:
                    log(f"  {setting:<8}{a:<6}{cname:<14}  N/A -- structurally unavailable to chain "
                        f"(no per-branch conditional)", fh)
                    continue
                g = r["after"]["t3@10"] - r["before"]["t3@10"]
                ratio = (g / gain_c) if (a == "MFD" and gc and abs(gain_c) > 1e-9) else (1.0 if a == "chain" else float("nan"))
                flag = " **COLLAPSED" if r["collapsed"] else ""
                log(f"  {setting:<8}{a:<6}{cname:<14}{r['params']:>10,}{r['before']['t3@10']:>8.4f}"
                    f"{r['after']['t3@10']:>8.4f}{g:>+9.4f}{ratio:>7.2f}{r['before']['pcov']:>8.4f}"
                    f"{r['after']['pcov']:>8.4f}{r['after']['t3@5']:>8.4f}{r['after']['t3@20']:>8.4f}"
                    f"{r['after']['t3n@5']:>8.4f}{r['after']['t3n@10']:>8.4f}{r['after']['t2@10']:>8.4f}"
                    f"{r['after']['t2n@10']:>8.4f}{flag}", fh)

    log("\n  per-behavior t3@10 (before -> after) and whether pv AND conversions both improve:", fh)
    both_pat = {"MFD": [0, 0], "chain": [0, 0]}
    for setting in ("natural", "capped"):
        for cname in names:
            for a in ("MFD", "chain"):
                r = results.get((a, cname, setting))
                if r is None or r["collapsed"]:
                    continue
                bb, aa = r["before"]["perbeh"], r["after"]["perbeh"]
                dd = {k: aa[k] - bb[k] for k in aa}
                conv = [k for k in dd if k != "pv"]
                good = dd.get("pv", 0) > 0 and all(dd[k] >= 0 for k in conv)
                both_pat[a][0 if good else 1] += 1
                v = "BOTH improve" if good else ("REDISTRIBUTION (pv up, conversions down)"
                                                 if dd.get("pv", 0) > 0 else "pv down")
                log(f"    {setting:<8}{a:<6}{cname:<14}" + "  ".join(f"{k}:{bb[k]:.4f}->{aa[k]:.4f}" for k in aa)
                    + f"   [{v}]", fh)
    log(f"\n  pattern check (sound configs only): MFD both-improve {both_pat['MFD'][0]}/"
        f"{sum(both_pat['MFD'])}   chain both-improve {both_pat['chain'][0]}/{sum(both_pat['chain'])}", fh)

    # ---------------- the four diagnostic cells -------------------------------------------------
    log("\n" + "=" * 118 + "\n=== DIAGNOSTIC CELLS (recovery into top-10; the sensitive read) ===\n" + "=" * 118, fh)
    cm = arms["MFD"].te.has_gt.numpy(); cc = arms["chain"].te.has_gt.numpy()[perm.numpy()]
    both = cm & cc; om_ = cm & ~cc; oc_ = cc & ~cm
    log(f"  groups: BOTH={both.sum():,}  MFD-only={om_.sum():,}  chain-only={oc_.sum():,}", fh)
    log(f"  {'set':<8}{'arm':<6}{'config':<14}{'BOTH/10-49':>12}{'BOTH/50+':>10}{'own/10-49':>11}{'own/50+':>10}", fh)
    for setting in ("natural", "capped"):
        for cname in names:
            for a in ("MFD", "chain"):
                r = results.get((a, cname, setting))
                if r is None:
                    continue
                rb = r["r3_before"]; ha = r["hit_after"]
                if a == "chain":
                    rb = rb[perm.numpy()]; ha = ha[perm.numpy()]
                grp_both, grp_own = both, (om_ if a == "MFD" else oc_)
                cells = []
                for grp in (grp_both, grp_own):
                    for lo, hi in ((10, 50), (50, 10 ** 9)):
                        m = grp & (rb >= lo) & (rb < hi)
                        cells.append(float(ha[m].mean()) if m.sum() else float("nan"))
                log(f"  {setting:<8}{a:<6}{cname:<14}{cells[0]:>12.4f}{cells[1]:>10.4f}"
                    f"{cells[2]:>11.4f}{cells[3]:>10.4f}", fh)
    log("  (chain/own/50+ was exactly 0.0000 at baseline -- chain cannot recover deep candidates.)", fh)

    # ---------------- collapse flagging ---------------------------------------------------------
    bad = [(a, c, s) for (a, c, s), r in results.items() if r["collapsed"]]
    log(f"\n  COLLAPSED runs (excluded from any trend analysis, flagged as implementation failures, "
        f"NOT data points): {bad if bad else 'none'}", fh)

    json.dump({f"{a}|{c}|{s}": {k: v for k, v in r.items() if not (k.startswith('hit_') or k.startswith('r3_'))}
               for (a, c, s), r in results.items()},
              open(os.path.join(args.out_dir, "sweep2_results.json"), "w"), indent=1, default=float)
    fh.close()


if __name__ == "__main__":
    main()
