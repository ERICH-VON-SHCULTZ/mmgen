"""Scale the reranker along capacity / features / two-stage, on BOTH arms, and test whether MFD's
advantage over chain grows. The headline is the RATIO of gains, not either arm alone.

Baseline (466,690 params, 2 layers, d=128, views+hard+soft) at matched 808-decoder-state compute:
  MFD   t3@10 0.2042 -> 0.2151 (+0.0110)   chain 0.2120 -> 0.2140 (+0.0020)   ratio 5.3x
MFD holds more recoverable GTs (covered-but-outside-top-10 5.37% of events vs chain 3.51%), so a
stronger ranker should keep favouring it. If the ratio collapses toward 1:1 as signal is added, the
signal is saturating and the advantage is bounded -- report whichever the data shows.

AXES (every config runs on BOTH arms, in BOTH the natural and the capped setting):
  A capacity : layers {2,4} x d_model {128,256}   (d_model != d_mem -> the encoder memory is projected)
  B features : cumulative -- decay, +popularity, +session, +user
  C two-stage: a cheap scalar-only stage-1 shortlists top-50, the cross-attn stage-2 refines it
  best-of    : winning capacity + winning feature set (+ two-stage if it helped), resolved on VAL

!! B1 CAVEAT (stated, not silently substituted): the TFRecords store only user_id / sequence_data /
   behavior_data. Sequences are SORTED by timestamp but timestamps are NOT stored anywhere, so the
   elapsed TIME since a matching interaction is not recoverable from any artefact. B1 is therefore
   implemented as an exponential POSITION-decay (two rates), which is a different signal from the
   existing linear recency but a proxy for the requested one. Reported as such.

Protocol: identical optimizer/schedule/early-stopping everywhere; fit on train, select on VAL,
report on TEST. Splits and coverage are asserted per arm; SID widths are read from the files.
Every (arm, config, setting) result is cached, so the sweep is resumable.
"""

import argparse
import itertools
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


# ------------------------------------------------------------------ per-arm feature precomputation
class Arm:
    """Holds the three splits of one arm plus everything the feature blocks need."""

    def __init__(self, name, tr_p, va_p, te_p, sid_p, wb, fh):
        self.name = name
        self.tr = RerankData(tr_p, wb); self.va = RerankData(va_p, wb); self.te = RerankData(te_p, wb)
        self.widths = sid_widths(sid_p)
        self.nb = self.tr.nb
        for d in (self.tr, self.va, self.te):
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
        log(f"  [{name}] coverage@pool={cov:.4f} (ref {REF_COV[name]:.4f})  "
            f"fwd_states={self.te.meta.get('fwd_states_per_item','?')}  Kp={self.te.Kp}  "
            f"covered train={int(self.tr.has_gt.sum()):,}", fh)
        assert abs(cov - REF_COV[name]) < 0.01, f"{name} coverage {cov:.4f} != ref"
        self.k1, self.k2 = self.widths[1], self.widths[2]
        self._popularity()
        self._per_event()

    def _pack(self, f, t, i):
        return (f.long() * self.k1 + t.long()) * self.k2 + i.long()

    def _popularity(self):
        """Tuple frequency from the TRAINING split ONLY (never val/test). 'recent window' = the last
        20% of training rows (a row-order proxy: timestamps are not stored)."""
        key = self._pack(self.tr.gt_f, self.tr.gt_t, self.tr.gt_i)
        n = key.numel(); cut = int(n * 0.8)
        uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
        rec = torch.zeros_like(cnt); rec.index_add_(0, inv[cut:], torch.ones_like(inv[cut:]))
        self.pop_keys, self.pop_glob, self.pop_rec = uk, cnt.float(), rec.float()

    def pop_feats(self, f, t, i):
        k = self._pack(f, t, i)
        pos = torch.searchsorted(self.pop_keys.to(k.device), k).clamp(max=self.pop_keys.numel() - 1)
        hit = self.pop_keys.to(k.device)[pos] == k
        g = torch.where(hit, self.pop_glob.to(k.device)[pos], torch.zeros_like(k, dtype=torch.float))
        r = torch.where(hit, self.pop_rec.to(k.device)[pos], torch.zeros_like(k, dtype=torch.float))
        return torch.stack([torch.log1p(g), torch.log1p(r)], -1)          # (B,C,2)

    def _per_event(self):
        """Per-event (broadcast over candidates) session + user features, per split."""
        self.ev = {}
        for nm, d in (("tr", self.tr), ("va", self.va), ("te", self.te)):
            if not d.has_hist:
                self.ev[nm] = None; continue
            hb, hv = d.hist_b.long(), (d.hist_b.long() >= 0)
            ln = hv.sum(1).float()
            sess = torch.stack([torch.log1p(ln), ln / max(d.H, 1)], -1)                  # (N,2)
            bd = torch.stack([((hb == b) & hv).sum(1).float() / ln.clamp(min=1) for b in range(self.nb)], -1)
            uf = d.hist_f.long()
            ent = []
            for e in range(0, d.N, 8192):                                                 # shared-code entropy
                sl = slice(e, min(e + 8192, d.N))
                x = uf[sl].clone(); m = hv[sl]
                x = torch.where(m, x, torch.full_like(x, -1))
                oh = torch.zeros(x.shape[0], self.widths[0] + 1, device=x.device)
                oh.scatter_add_(1, (x + 1), torch.ones_like(x, dtype=torch.float))
                oh = oh[:, 1:]
                p = oh / oh.sum(1, keepdim=True).clamp(min=1)
                ent.append(-(p * p.clamp(min=1e-12).log()).sum(1))
            ent = torch.cat(ent).unsqueeze(-1)
            self.ev[nm] = {"session": sess, "user": torch.cat([bd, torch.log1p(ln).unsqueeze(-1), ent], -1)}


def decay_feats(f, t, i, hist, nb, rates=(0.15, 0.02)):
    """Exponential POSITION-decay of exact history matches, per behavior, at two rates.
    PROXY for elapsed time: timestamps are not stored anywhere (see module docstring)."""
    hf, ht, hi, hb, hvalid, recency = hist
    H = hf.shape[1]
    age = (1.0 - recency).clamp(min=0) * H                                   # 0 = newest
    match = ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
             & (i[:, :, None] == hi[:, None, :]))
    out = []
    for lam in rates:
        w = torch.exp(-lam * age)[:, None, :]
        for b in range(nb):
            bm = ((hb == b) & hvalid)[:, None, :]
            out.append((match & bm).float().mul(w).amax(-1))
    return torch.stack(out, -1)                                              # (B,C,2*nb)


# ------------------------------------------------------------------------------------ the ranker
class SweepRanker(nn.Module):
    """Identical to MFDReranker at (layers=2, d=128, no extra blocks, no two-stage); adds a memory
    projection when d_model != d_mem, extra feature blocks, and an optional cheap stage-1."""

    def __init__(self, U_f, U_t, U_i, d_model, d_mem, layers=2, heads=4, drop=0.3, nb=4,
                 n_extra=0, two_stage=False, k1_short=50):
        super().__init__()
        self.register_buffer("U_f", U_f); self.register_buffer("U_t", U_t); self.register_buffer("U_i", U_i)
        self.nb = nb; self.two_stage = two_stage; self.k1_short = k1_short
        n_scalar = 4 + 3 * nb + 3 * nb + n_extra          # logps + hard + soft + extra
        d = d_model
        self.mem_proj = nn.Linear(d_mem, d) if d_mem != d else None
        self.view_proj = nn.Linear(3 * d, d)
        self.scalar_proj = nn.Sequential(nn.Linear(n_scalar, d), nn.GELU(), nn.Linear(d, d))
        self.in_norm = nn.LayerNorm(d); self.in_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([CrossBlock(d, heads, drop) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d); self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        if two_stage:                                     # cheap scalar-only shortlister
            self.stage1 = nn.Sequential(nn.Linear(n_scalar, 4 * d), nn.GELU(), nn.Linear(4 * d, 1))
            nn.init.zeros_(self.stage1[-1].weight); nn.init.zeros_(self.stage1[-1].bias)

    def tokens(self, f, t, i, scal):
        uf = self.U_f[f]
        rep = torch.cat([uf, uf + self.U_t[t], uf + self.U_i[i]], -1)
        return self.in_drop(self.in_norm(self.view_proj(rep) + self.scalar_proj(scal)))

    def forward(self, f, t, i, scal, base, valid, mem, mem_pad):
        B, C = f.shape
        if self.mem_proj is not None:
            mem = self.mem_proj(mem)
        x = self.tokens(f, t, i, scal)
        if not self.two_stage:
            for blk in self.blocks:
                x = blk(x, mem, mem_pad)
            delta = self.head(self.out_norm(x)).squeeze(-1)
            return (self.base_scale * base + delta).masked_fill(~valid, float("-inf"))
        s1 = self.stage1(scal).squeeze(-1).masked_fill(~valid, float("-inf"))     # cheap full-pool pass
        k = min(self.k1_short, C)
        idx = s1.detach().topk(k, dim=-1).indices                                 # (B,k) shortlist
        xs = torch.gather(x, 1, idx.unsqueeze(-1).expand(B, k, x.size(-1)))
        for blk in self.blocks:
            xs = blk(xs, mem, mem_pad)                                            # cross-attn on 50, not 2000
        ds = self.head(self.out_norm(xs)).squeeze(-1)                             # (B,k)
        delta = torch.zeros_like(s1).scatter(1, idx, ds)
        return (self.base_scale * base + s1 + delta).masked_fill(~valid, float("-inf"))


# --------------------------------------------------------------------------------- feature assembly
def hist_of(d, idx, dev):
    return (d.hist_f[idx].long().to(dev), d.hist_t[idx].long().to(dev), d.hist_i[idx].long().to(dev),
            d.hist_b[idx].long().to(dev), (d.hist_b[idx].long() >= 0).to(dev),
            d.hist_recency[idx].float().to(dev))


def engagement(model, f, t, i, hist):
    hf, ht, hi, hb, hvalid, rec = hist
    feats = []
    match = ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
             & (i[:, :, None] == hi[:, None, :]))
    r = rec[:, None, :]
    for b in range(model.nb):
        bm = ((hb == b) & hvalid)[:, None, :]
        mb = match & bm
        cnt = mb.sum(-1).float()
        feats += [(cnt > 0).float(), torch.log1p(cnt), (mb.float() * r).amax(-1)]
    cuf = model.U_f[f]; cvt = cuf + model.U_t[t]; cvi = cuf + model.U_i[i]
    huf = model.U_f[hf]; hvt = huf + model.U_t[ht]; hvi = huf + model.U_i[hi]
    for cv, hv in ((cuf, huf), (cvt, hvt), (cvi, hvi)):
        sim = torch.bmm(F.normalize(cv, dim=-1), F.normalize(hv, dim=-1).transpose(1, 2))
        for b in range(model.nb):
            bm = ((hb == b) & hvalid)[:, None, :]
            m = sim.masked_fill(~bm.expand_as(sim), -2.0).amax(-1)
            feats.append(torch.where(m < -1.5, torch.zeros_like(m), m))
    return torch.stack(feats, -1)


def make_scal(arm, split, model, d, idx, dev, blocks):
    f = d.f[idx].long().to(dev); t = d.t[idx].long().to(dev); i = d.i[idx].long().to(dev)
    lp = d.lp[idx].float().to(dev)
    hist = hist_of(d, idx, dev)
    parts = [lp, engagement(model, f, t, i, hist)]
    B, C = f.shape
    if "decay" in blocks:
        parts.append(decay_feats(f, t, i, hist, model.nb))
    if "pop" in blocks:
        parts.append(arm.pop_feats(f, t, i))
    ev = arm.ev[split]
    if "session" in blocks:
        parts.append(ev["session"][idx].to(dev).unsqueeze(1).expand(B, C, -1))
    if "user" in blocks:
        parts.append(ev["user"][idx].to(dev).unsqueeze(1).expand(B, C, -1))
    return f, t, i, torch.cat(parts, -1)


def n_extra_of(blocks, nb):
    n = 0
    if "decay" in blocks: n += 2 * nb
    if "pop" in blocks: n += 2
    if "session" in blocks: n += 2
    if "user" in blocks: n += nb + 2
    return n


# ------------------------------------------------------------------------------------- train / eval
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
            "hit": (h3 & (r3 >= 0) & (r3 < 10)).numpy().astype(np.float64)}


def run_one(arm, cfg, cap, dev, args, fh):
    torch.manual_seed(args.seed)
    d_mem = arm.tr.mem.shape[2]; dm = cfg["d"]
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in arm.widths]
    model = SweepRanker(U[0], U[1], U[2], dm, d_mem, cfg["layers"], args.heads, args.dropout,
                        arm.nb, n_extra_of(cfg["blocks"], arm.nb), cfg.get("two_stage", False)).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = arm.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]]
        idx_all.sort()
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
            loss = F.cross_entropy(sc, arm.tr.gt_idx[bidx].to(dev))
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
    return {"params": npar, "val_t3@10": best_v, "n_train": int(len(idx_all)),
            "before": {k: v for k, v in base_te.items() if k != "hit"},
            "after": {k: v for k, v in te.items() if k != "hit"},
            "hit_before": base_te["hit"], "hit_after": te["hit"]}


CONFIGS = [
    ("baseline", dict(layers=2, d=128, blocks=())),
    ("A1_4L_128", dict(layers=4, d=128, blocks=())),
    ("A2_2L_256", dict(layers=2, d=256, blocks=())),
    ("A3_4L_256", dict(layers=4, d=256, blocks=())),
    ("B1_decay", dict(layers=2, d=128, blocks=("decay",))),
    ("B2_pop", dict(layers=2, d=128, blocks=("decay", "pop"))),
    ("B3_session", dict(layers=2, d=128, blocks=("decay", "pop", "session"))),
    ("B4_user", dict(layers=2, d=128, blocks=("decay", "pop", "session", "user"))),
    ("C1_twostage", dict(layers=2, d=128, blocks=(), two_stage=True)),
]
SIGNAL = {"baseline": 0, "B1_decay": 1, "B2_pop": 2, "B3_session": 3, "B4_user": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_tr", required=True); ap.add_argument("--mfd_va", required=True); ap.add_argument("--mfd_te", required=True)
    ap.add_argument("--ch_tr", required=True); ap.add_argument("--ch_va", required=True); ap.add_argument("--ch_te", required=True)
    ap.add_argument("--md_sid", required=True); ap.add_argument("--tg_sid", required=True)
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/sweep")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache"); os.makedirs(cache_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "sweep_report.txt"), "w")
    log("=" * 110 + "\n=== RERANKER SWEEP: does MFD's advantage over chain grow with a stronger ranker? ===\n"
        + "=" * 110, fh)
    log("  B1 CAVEAT: timestamps are not stored in the TFRecords (only user_id/sequence_data/"
        "behavior_data),\n  so 'elapsed time' is NOT recoverable; B1 is an exponential POSITION-decay proxy.", fh)

    arms = {"MFD": Arm("MFD", args.mfd_tr, args.mfd_va, args.mfd_te, args.md_sid, args.wb, fh),
            "chain": Arm("chain", args.ch_tr, args.ch_va, args.ch_te, args.tg_sid, args.wb, fh)}

    results = {}

    def get(arm_name, cname, cfg, setting):
        cap = args.cap if setting == "capped" else 0
        key = f"{arm_name}__{cname}__{setting}"
        cp = os.path.join(cache_dir, key + ".pt")
        if os.path.exists(cp):
            log(f"  [cached] {key}", fh)
            return torch.load(cp, map_location="cpu", weights_only=False)
        log(f"\n  >>> RUN {key}", fh)
        r = run_one(arms[arm_name], cfg, cap, dev, args, fh)
        torch.save(r, cp)
        return r

    for setting in ("natural", "capped"):
        for cname, cfg in CONFIGS:
            for a in ("MFD", "chain"):
                results[(a, cname, setting)] = get(a, cname, cfg, setting)

    # ---- best-of, resolved on VAL (never TEST) ----
    # best-of is resolved on EACH ARM'S OWN VAL split (never TEST), so neither arm is tuned on the
    # other's selection -- the search itself is identical on both.
    for setting in ("natural", "capped"):
        for a in ("MFD", "chain"):
            bv = lambda ns: max(ns, key=lambda n: results[(a, n, setting)]["val_t3@10"])
            cap_win = bv(["baseline", "A1_4L_128", "A2_2L_256", "A3_4L_256"])
            feat_win = bv(["baseline", "B1_decay", "B2_pop", "B3_session", "B4_user"])
            ts_help = results[(a, "C1_twostage", setting)]["val_t3@10"] > results[(a, "baseline", setting)]["val_t3@10"]
            cap_d = dict(CONFIGS)[cap_win]; feat_d = dict(CONFIGS)[feat_win]
            combo = dict(layers=cap_d["layers"], d=cap_d["d"], blocks=feat_d["blocks"], two_stage=ts_help)
            log(f"\n  best-of[{setting}/{a}] (own-VAL): capacity={cap_win} features={feat_win} "
                f"two_stage={ts_help} -> {combo}", fh)
            results[(a, "BESTOF", setting)] = get(a, "BESTOF", combo, setting)

    # ------------------------------------------------------------------ consolidated table
    names = [c[0] for c in CONFIGS] + ["BESTOF"]
    log("\n" + "=" * 110 + "\n=== CONSOLIDATED TABLE ===\n" + "=" * 110, fh)
    hdr = (f"  {'setting':<9}{'arm':<6}{'config':<13}{'params':>10}{'t3@10 bef':>11}{'aft':>8}{'gain':>9}"
           f"{'ratio':>7}{'P(r<10)b':>10}{'aft':>8}{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}"
           f"{'t2@10':>8}{'t2n@10':>8}")
    log(hdr, fh)
    for setting in ("natural", "capped"):
        for cname in names:
            gc = results[("chain", cname, setting)]
            gain_c = gc["after"]["t3@10"] - gc["before"]["t3@10"]
            for a in ("MFD", "chain"):
                r = results[(a, cname, setting)]
                g = r["after"]["t3@10"] - r["before"]["t3@10"]
                ratio = (g / gain_c) if (a == "MFD" and abs(gain_c) > 1e-9) else (1.0 if a == "chain" else float("nan"))
                log(f"  {setting:<9}{a:<6}{cname:<13}{r['params']:>10,}{r['before']['t3@10']:>11.4f}"
                    f"{r['after']['t3@10']:>8.4f}{g:>+9.4f}{ratio:>7.2f}{r['before']['pcov']:>10.4f}"
                    f"{r['after']['pcov']:>8.4f}{r['after']['t3@5']:>8.4f}{r['after']['t3@20']:>8.4f}"
                    f"{r['after']['t3n@5']:>8.4f}{r['after']['t3n@10']:>8.4f}{r['after']['t2@10']:>8.4f}"
                    f"{r['after']['t2n@10']:>8.4f}", fh)

    log("\n  per-behavior t3@10 (before -> after), and whether pv and conversions BOTH improve:", fh)
    for setting in ("natural", "capped"):
        for cname in names:
            for a in ("MFD", "chain"):
                r = results[(a, cname, setting)]
                bb, aa = r["before"]["perbeh"], r["after"]["perbeh"]
                dd = {k: aa[k] - bb[k] for k in aa}
                conv = [k for k in dd if k != "pv"]
                verdict = ("BOTH improve" if dd.get("pv", 0) > 0 and all(dd[k] >= 0 for k in conv)
                           else "REDISTRIBUTION (pv up, conversions down)" if dd.get("pv", 0) > 0
                           else "pv down")
                log(f"    {setting:<9}{a:<6}{cname:<13}" + "  ".join(f"{k}:{bb[k]:.4f}->{aa[k]:.4f}" for k in aa)
                    + f"   [{verdict}]", fh)

    # ------------------------------------------------------------------ gain vs signal strength
    log("\n" + "=" * 110 + "\n=== GAIN vs SIGNAL STRENGTH (feature axis; the two slopes are the headline) ===\n"
        + "=" * 110, fh)
    for setting in ("natural", "capped"):
        log(f"  [{setting}]  {'signal':<8}{'config':<13}{'MFD gain':>10}{'chain gain':>12}{'ratio':>8}", fh)
        xs, ym, yc = [], [], []
        for cname, s in sorted(SIGNAL.items(), key=lambda kv: kv[1]):
            rm, rc = results[("MFD", cname, setting)], results[("chain", cname, setting)]
            gm = rm["after"]["t3@10"] - rm["before"]["t3@10"]; gcv = rc["after"]["t3@10"] - rc["before"]["t3@10"]
            xs.append(s); ym.append(gm); yc.append(gcv)
            log(f"           {s:<8}{cname:<13}{gm:>+10.4f}{gcv:>+12.4f}"
                f"{(gm/gcv if abs(gcv)>1e-9 else float('nan')):>8.2f}", fh)
        sm = float(np.polyfit(xs, ym, 1)[0]); sc_ = float(np.polyfit(xs, yc, 1)[0])
        log(f"    slope(gain vs signal): MFD {sm:+.5f}/step   chain {sc_:+.5f}/step   "
            f"-> {'MFD slope steeper (advantage GROWS)' if sm > sc_ else 'chain slope >= MFD (advantage does NOT grow)'}", fh)
        r0 = (results[('MFD','baseline',setting)]['after']['t3@10'] - results[('MFD','baseline',setting)]['before']['t3@10'])
        r4 = (results[('MFD','B4_user',setting)]['after']['t3@10'] - results[('MFD','B4_user',setting)]['before']['t3@10'])
        c0 = (results[('chain','baseline',setting)]['after']['t3@10'] - results[('chain','baseline',setting)]['before']['t3@10'])
        c4 = (results[('chain','B4_user',setting)]['after']['t3@10'] - results[('chain','B4_user',setting)]['before']['t3@10'])
        log(f"    ratio at signal 0 = {(r0/c0 if abs(c0)>1e-9 else float('nan')):.2f}   "
            f"at signal 4 = {(r4/c4 if abs(c4)>1e-9 else float('nan')):.2f}   "
            f"-> {'ratio HOLDS/GROWS' if abs(c4)>1e-9 and abs(c0)>1e-9 and r4/c4 >= r0/c0 else 'ratio COLLAPSES toward 1:1 (signal saturating, advantage bounded)'}", fh)

    # marginal contribution of popularity (B2 - B1)
    log("\n  marginal contribution of POPULARITY (B2 minus B1):", fh)
    for setting in ("natural", "capped"):
        for a in ("MFD", "chain"):
            d1 = results[(a, "B1_decay", setting)]["after"]["t3@10"]
            d2 = results[(a, "B2_pop", setting)]["after"]["t3@10"]
            log(f"    {setting:<9}{a:<6} t3@10 {d1:.4f} -> {d2:.4f}  ({d2-d1:+.4f})", fh)

    # ------------------------------------------------------------------ bootstrap on the best config
    log("\n" + "=" * 110 + "\n=== BOOTSTRAP over EVENTS: MFD - chain at the best configuration ===\n" + "=" * 110, fh)
    rng = np.random.default_rng(args.seed)
    for setting in ("natural", "capped"):
        cand = max(names, key=lambda n: results[("MFD", n, setting)]["val_t3@10"])
        hm = results[("MFD", cand, setting)]["hit_after"]; hc = results[("chain", cand, setting)]["hit_after"]
        N = hm.shape[0]
        df = np.empty(args.boot)
        for k in range(args.boot):
            ii = rng.integers(0, N, N)
            df[k] = hm[ii].mean() - hc[ii].mean()
        lo, hi = float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))
        log(f"  [{setting}] best-on-VAL config = {cand}   MFD {hm.mean():.4f} vs chain {hc.mean():.4f}   "
            f"diff {hm.mean()-hc.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(MFD>chain)={float((df>0).mean()):.4f}", fh)
        log(f"    -> {'DISTINGUISHABLE from noise' if (lo>0 or hi<0) else 'NOT distinguishable from noise (CI spans 0)'}", fh)

    json.dump({f"{a}|{c}|{s}": {k: v for k, v in r.items() if not k.startswith("hit_")}
               for (a, c, s), r in results.items()},
              open(os.path.join(args.out_dir, "sweep_results.json"), "w"), indent=1, default=float)
    fh.close()


if __name__ == "__main__":
    main()
