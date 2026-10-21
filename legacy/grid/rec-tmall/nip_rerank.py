"""NIP -- New-Item Protection: stop MFD's reranker from evicting the new-item hits its generator found.

MEASURED (5 seeds, cached per-event results). For page-views, both systems succeed almost only on
REPEATS (items already in the user's history): MFD 12,951 repeat hits vs 6 new-item hits, chain
12,669 vs 134. On the new items, MFD's RERANKER destroys what its generator got right:

    pv new items          n       MFD base -> rerank      chain base -> rerank
    BOTH              1,176        0.0570 -> 0.0009        0.0842 -> 0.0847
    MFD-ONLY          4,663        0.0212 -> 0.0010        (not in chain's pool)

MFD's pool is 4x larger than chain's and carries far more history-matched candidates, and a listwise
model trained mostly on repeat-GT events fills the top 10 with them. ~160 generator new-item hits are
evicted; the 94 on MFD-ONLY items are hits chain cannot follow.

Two fixes, both chosen on VAL, never TEST:
  1. FINAL-LIST POLICY (no retraining of the generator; a property of how the list is assembled):
     P_k = top-(10-k) by the reranker  u  top-k by the GENERATOR's base score among candidates that are
     NOT in the user's history and not already chosen. k in {0,1,2,3}; k = 0 is today's list.
     Reserving k slots for the generator's own best new-item guesses protects them from eviction.
  2. TRAINING REWEIGHT: multiply the rw1 event weight by w_new for training events whose GT is NOT a
     repeat, so the listwise loss stops being dominated by repeat-GT events.
Both arms get exactly the same treatment (same policy family, same k grid, same reweight, k chosen on
each arm's own VAL), so any change in the LEAD is attributable to MFD's pool, not to unequal handling.

Report: per seed and across seeds, the lead at each arm's VAL-chosen policy with a paired event
bootstrap, and the lead decomposed by pool membership x {repeat, new item} -- the deciding cell is
MFD-ONLY x new item, which chain cannot reach.
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

from train_mfd_reranker import BEH
from sweep2_rerank import Ranker, make_scal, metrics, ranks_of, RW
from coverage_expansion import Pool, canon, log
from cf_rerank import paired_ci

KS = (0, 1, 2, 3)
BETAS = (-1, 0, 1, 2, 3, 4, 5, 6)
POLICIES = ("P0", "P1", "P2", "P3", "D") + tuple(f"M{b}" for b in BETAS)
# P_k: k reserved new-item slots; D: one slot per ITEM; M{beta}: calibrated per-event REPEAT/NEW mixture
#   P(c) = G' * P_main(c | history)  for history candidates,  (1-G') * P_new(c)  for the rest,
#   G = main reranker's probability mass on history candidates, logit(G') = logit(G) - beta (beta chosen on VAL),
#   P_new = softmax over NON-history candidates of the new-item SPECIALIST (or the base score without one).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def hist_match(d, idx, f, t, i, dev):
    """(B,C) bool: candidate tuple equals some valid history item's tuple (any behavior)."""
    hf, ht, hi = d.hist_f[idx].long().to(dev), d.hist_t[idx].long().to(dev), d.hist_i[idx].long().to(dev)
    hv = (d.hist_b[idx].long() >= 0).to(dev)
    return ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
            & (i[:, :, None] == hi[:, None, :]) & hv[:, None, :]).any(-1)


def dedup_top10(S, key, M=64):
    """Top-10 keeping only the highest-scoring behavior per ITEM (f,t,i). t3 ranks the UNION of four
    behavior pools, so (pv, X) and (cart, X) can both occupy top-10 slots; for a pv ground truth the
    second is wasted. Walks the top-M by score, drops items already seen, keeps the first 10."""
    B, C = S.shape
    top = S.topk(min(M, C), dim=-1)
    k = torch.gather(key, 1, top.indices)
    Mm = k.shape[1]
    earlier = torch.tril(torch.ones(Mm, Mm, dtype=torch.bool, device=S.device), diagonal=-1)
    dup = ((k.unsqueeze(2) == k.unsqueeze(1)) & earlier.unsqueeze(0)).any(-1)
    keep = ~dup & (top.values > float("-inf"))
    sel = keep & (keep.long().cumsum(1) <= 10)
    chosen = torch.zeros(B, C, dtype=torch.bool, device=S.device)
    chosen.scatter_(1, top.indices, sel)
    # duplicate slots inside today's top-10 (diagnostic): positions 0..9 whose item appeared earlier
    dup10 = (dup[:, :10] & (top.values[:, :10] > float("-inf"))).sum(1)
    return chosen, dup10


REPFEAT = False          # set from --repfeat: add repeat-consumption features for history candidates
N_REP = 11


def rep_feats(d, idx, f, t, i, dev):
    """Per-candidate repeat-consumption features (zeros for non-history candidates, except the event-level one).
    The top-10 is almost entirely history items and MFD-only repeats lose to the user's OTHER history items
    (70% sit at rank 10-19), so the missing signal is WHICH past item comes back next.
      log1p(count) | last position | first position | 1/(1+recency rank among distinct items) |
      log1p(recency rank) | behavior of the most recent interaction (4 one-hot) | mean gap between
      occurrences | distinct history items / H"""
    hf, ht, hi = d.hist_f[idx].long().to(dev), d.hist_t[idx].long().to(dev), d.hist_i[idx].long().to(dev)
    hb = d.hist_b[idx].long().to(dev); hv = hb >= 0
    rec = d.hist_recency[idx].float().to(dev)                               # (B,H) in (0,1], 1 = most recent
    B, H = hf.shape
    m = ((f[:, :, None] == hf[:, None, :]) & (t[:, :, None] == ht[:, None, :])
         & (i[:, :, None] == hi[:, None, :]) & hv[:, None, :])               # (B,C,H)
    cnt = m.sum(-1).float()
    lastp = torch.where(m, rec[:, None, :], torch.zeros_like(rec[:, None, :])).amax(-1)
    firstp = torch.where(m, rec[:, None, :], torch.full_like(rec[:, None, :], 2.0)).amin(-1)
    firstp = torch.where(cnt > 0, firstp, torch.zeros_like(firstp))
    # distinct history items: slot j is the LAST occurrence of its item
    hk = (hf * 4096 + ht) * 4096 + hi
    same_h = (hk[:, :, None] == hk[:, None, :]) & hv[:, :, None] & hv[:, None, :]
    last_rec_h = torch.where(same_h, rec[:, None, :], torch.zeros_like(rec[:, None, :])).amax(-1)   # (B,H)
    is_last = hv & (rec >= last_rec_h - 1e-6)
    rank = ((last_rec_h[:, None, :] > lastp[:, :, None] + 1e-6) & is_last[:, None, :]).sum(-1).float()
    jlast = torch.where(m, rec[:, None, :], torch.full_like(rec[:, None, :], -1.0)).argmax(-1)       # (B,C)
    lb = hb.gather(1, jlast)
    onehot = torch.stack([(lb == b) for b in range(4)], -1).float() * (cnt > 0).float().unsqueeze(-1)
    gap = (lastp - firstp) / (cnt - 1).clamp(min=1)
    nd = (is_last.sum(1).float() / H).unsqueeze(1).expand(B, f.shape[1])
    has = (cnt > 0).float()
    out = torch.stack([torch.log1p(cnt), lastp, firstp, has / (1 + rank), has * torch.log1p(rank), gap * has, nd], -1)
    return torch.cat([out[..., :5], onehot, out[..., 5:]], -1)             # (B,C,11)


def scal_of(pool, split, model, d, idx, dev):
    f, t, i, scal = make_scal(pool, split, model, d, idx, dev, ())
    if REPFEAT:
        scal = torch.cat([scal, rep_feats(d, idx, f, t, i, dev)], -1)
    return f, t, i, scal


def gt_is_repeat(d):
    """(N,) bool: the ground-truth tuple appears in the event's history."""
    hv = d.hist_b.long() >= 0
    return ((d.hist_f.long() == d.gt_f[:, None]) & (d.hist_t.long() == d.gt_t[:, None])
            & (d.hist_i.long() == d.gt_i[:, None]) & hv).any(1).numpy()


@torch.no_grad()
def eval_policies(model, pool, split, d, dev, bs, spec=None, rspec=None):
    """Returns per-event hit arrays for every policy P_k, plus the raw reranker score matrix.
    spec = optional new-item SPECIALIST reranker: when given, the k reserved slots are filled by the
    specialist's score among non-history candidates instead of by the generator's raw base score."""
    model.eval()
    if spec is not None:
        spec.eval()
    if rspec is not None:
        rspec.eval()
    N = d.N
    hits = {p: np.zeros(N, dtype=bool) for p in POLICIES}
    dups = np.zeros(N, dtype=np.int64)
    sc_all = torch.empty(N, d.C)
    for s in range(0, N, bs):
        idx = torch.arange(s, min(s + bs, N))
        f, t, i, scal = scal_of(pool, split, model, d, idx, dev)
        valid = d.valid[idx].to(dev)
        S = model(f, t, i, scal, d.base[idx].float().to(dev), valid,
                  d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev))
        sc_all[idx] = S.cpu()
        base = d.base[idx].float().to(dev).masked_fill(~valid, float("-inf"))
        if spec is not None:                       # specialist score replaces the base for the reserved slots
            _, _, _, scal_s = scal_of(pool, split, spec, d, idx, dev)      # its OWN feature tables
            base = spec(f, t, i, scal_s, d.base[idx].float().to(dev), valid,
                        d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev))
        hm = hist_match(d, idx, f, t, i, dev)
        gt = d.gt_idx[idx].to(dev); has = gt >= 0
        B = f.shape[0]
        for k in KS:
            chosen = torch.zeros_like(valid)
            chosen.scatter_(1, S.topk(10 - k, dim=-1).indices, True)
            if k > 0:
                b2 = base.masked_fill(hm | chosen, float("-inf"))      # generator's best NON-history picks
                tb = b2.topk(k, dim=-1)
                add = torch.zeros_like(chosen)
                add.scatter_(1, tb.indices, tb.values > float("-inf"))   # never admit an invalid slot
                chosen |= add
            chosen &= valid
            hit = has & chosen.gather(1, gt.clamp(min=0).unsqueeze(1)).squeeze(1)
            hits[f"P{k}"][idx.numpy()] = hit.cpu().numpy()
        key = (f * pool.k1 + t) * pool.k2 + i
        chosen, dup10 = dedup_top10(S, key)
        chosen &= valid
        hits["D"][idx.numpy()] = (has & chosen.gather(1, gt.clamp(min=0).unsqueeze(1)).squeeze(1)).cpu().numpy()
        dups[idx.numpy()] = dup10.cpu().numpy()
        # ---- calibrated repeat/new mixture ----
        NEG = float("-inf")
        lpm = torch.log_softmax(S, dim=-1)                                 # main model over all valid
        nonh = valid & ~hm
        logG = torch.logsumexp(lpm.masked_fill(~(hm & valid), NEG), dim=-1)
        log1mG = torch.logsumexp(lpm.masked_fill(~nonh, NEG), dim=-1)
        sn = base.masked_fill(~nonh, NEG)                                  # base, or the specialist's score
        lpn = torch.log_softmax(sn, dim=-1)
        logitG = logG - log1mG
        if rspec is not None:                     # within-history distribution from the repeat specialist
            _, _, _, scal_r = scal_of(pool, split, rspec, d, idx, dev)     # its OWN feature tables
            R = rspec(f, t, i, scal_r, d.base[idx].float().to(dev), valid, d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev))
            lph = torch.log_softmax(R.masked_fill(~(hm & valid), NEG), dim=-1)
        else:
            lph = lpm - logG.unsqueeze(1)          # main model's P(c | history)
        for bta in BETAS:
            lg = torch.nn.functional.logsigmoid(logitG - bta)
            l1 = torch.nn.functional.logsigmoid(-(logitG - bta))
            fin = torch.where(hm, lg.unsqueeze(1) + lph, l1.unsqueeze(1) + lpn)
            fin = torch.nan_to_num(fin, nan=NEG).masked_fill(~valid, NEG)
            chosen = torch.zeros_like(valid)
            tk = fin.topk(10, dim=-1)
            chosen.scatter_(1, tk.indices, tk.values > NEG)
            hits[f"M{bta}"][idx.numpy()] = (has & chosen.gather(1, gt.clamp(min=0).unsqueeze(1)).squeeze(1)).cpu().numpy()
    hits["_dup10"] = dups
    return hits, sc_all


def train_specialist(pool, dev, args, fh):
    """New-item specialist: trained ONLY on non-repeat covered training events, with every history-matched
    candidate masked out of the listwise softmax -- the task the main reranker neglects. Early-stopped on
    VAL hit@10 over non-repeat events with history candidates masked (its own task, never TEST)."""
    torch.manual_seed(args.seed + 7)
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4243)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    m = Ranker(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, N_REP if REPFEAT else 0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    rep_tr = gt_is_repeat(pool.tr)
    idx_all = pool.tr.train_idx.numpy()
    idx_all = idx_all[~rep_tr[idx_all]].copy()               # non-repeat covered events only
    if args.cap and args.cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 98).permutation(len(idx_all))[:args.cap]])
    log(f"      specialist: {len(idx_all):,} non-repeat covered training events", fh)
    va_idx = pool.va.train_idx.numpy(); va_idx = va_idx[~gt_is_repeat(pool.va)[va_idx]]
    rng = np.random.default_rng(args.seed + 7)

    def masked(sc, d, bidx, f, t, i):
        return sc.masked_fill(hist_match(d, bidx, f, t, i, dev), float("-inf"))

    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        m.train(); rng.shuffle(idx_all)
        for s0 in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s0:s0 + args.batch])
            f, t, i, scal = scal_of(pool, "tr", m, pool.tr, bidx, dev)
            sc = m(f, t, i, scal, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                   pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            loss = F.cross_entropy(masked(sc, pool.tr, bidx, f, t, i), pool.tr.gt_idx[bidx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hits = []
        with torch.no_grad():
            for s0 in range(0, len(va_idx), args.eval_batch):
                bidx = torch.as_tensor(va_idx[s0:s0 + args.eval_batch])
                f, t, i, scal = scal_of(pool, "va", m, pool.va, bidx, dev)
                sc = m(f, t, i, scal, pool.va.base[bidx].float().to(dev), pool.va.valid[bidx].to(dev),
                       pool.va.mem[bidx].float().to(dev), pool.va.mem_pad[bidx].to(dev))
                sc = masked(sc, pool.va, bidx, f, t, i)
                gt = pool.va.gt_idx[bidx].to(dev)
                r = (sc > sc.gather(1, gt.unsqueeze(1))).sum(1)
                hits.append((r < 10).cpu())
        v = float(torch.cat(hits).float().mean()) if hits else 0.0
        log(f"      specialist ep{ep:02d} VAL new-item hit@10 (history masked) = {v:.4f}", fh)
        if v > best_v + args.min_delta:
            best_v, bad = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, best_v


def train_repeat_specialist(pool, dev, args, fh):
    """REPEAT specialist: trained only on covered REPEAT events, softmax over HISTORY candidates only, with the
    base-score anchor switched off (base_scale starts at 0; the generator's log-probs remain input features).
    Motivation: the 1,020 repeats that only MFD's forced pool covers were outside its pool precisely because the
    generator scores them low, so a base-anchored score keeps them down however strong their history signal."""
    torch.manual_seed(args.seed + 11)
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4244)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    m = Ranker(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, N_REP if REPFEAT else 0).to(dev)
    m.base_scale.data.fill_(0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    rep_tr = gt_is_repeat(pool.tr)
    idx_all = pool.tr.train_idx.numpy(); idx_all = idx_all[rep_tr[idx_all]].copy()
    if args.cap and args.cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 97).permutation(len(idx_all))[:args.cap]])
    log(f"      repeat specialist: {len(idx_all):,} repeat covered training events", fh)
    va_idx = pool.va.train_idx.numpy(); va_idx = va_idx[gt_is_repeat(pool.va)[va_idx]]
    rng = np.random.default_rng(args.seed + 11)
    msk = lambda sc, d, b_, f, t, i: sc.masked_fill(~hist_match(d, b_, f, t, i, dev), float("-inf"))
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        m.train(); rng.shuffle(idx_all)
        for s0 in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s0:s0 + args.batch])
            f, t, i, scal = scal_of(pool, "tr", m, pool.tr, bidx, dev)
            sc = m(f, t, i, scal, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                   pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            loss = F.cross_entropy(msk(sc, pool.tr, bidx, f, t, i), pool.tr.gt_idx[bidx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hits = []
        with torch.no_grad():
            for s0 in range(0, len(va_idx), args.eval_batch):
                bidx = torch.as_tensor(va_idx[s0:s0 + args.eval_batch])
                f, t, i, scal = scal_of(pool, "va", m, pool.va, bidx, dev)
                sc = msk(m(f, t, i, scal, pool.va.base[bidx].float().to(dev), pool.va.valid[bidx].to(dev),
                           pool.va.mem[bidx].float().to(dev), pool.va.mem_pad[bidx].to(dev)), pool.va, bidx, f, t, i)
                gt = pool.va.gt_idx[bidx].to(dev); hits.append(((sc > sc.gather(1, gt[:, None])).sum(1) < 10).cpu())
        v = float(torch.cat(hits).float().mean()) if hits else 0.0
        log(f"      repeat specialist ep{ep:02d} VAL repeat hit@10 (history only) = {v:.4f}", fh)
        if v > best_v + args.min_delta:
            best_v, bad = v, 0; best_state = {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, best_v


def run_cell(pool, w_new, cap, dev, args, fh):
    torch.manual_seed(args.seed); t0 = time.perf_counter(); ep = 0
    d_mem = pool.tr.mem.shape[2]; dm = 128
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, dm, generator=g) * 0.02).to(dev) for w in pool.widths]
    model = Ranker(U, dm, d_mem, 2, args.heads, args.dropout, pool.nb, N_REP if REPFEAT else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = pool.tr.train_idx.numpy().copy()
    if cap and cap < len(idx_all):
        idx_all = np.sort(idx_all[np.random.default_rng(args.seed + 99).permutation(len(idx_all))[:cap]])
    rb = ranks_of(pool.tr.base.float(), pool.tr)[0]
    band = torch.where(rb < 10, 0, torch.where(rb < 50, 1, 2)); tab = RW["rw1"]
    w_ev = torch.tensor([tab[int(b)] for b in band.tolist()], dtype=torch.float)
    rep_tr = torch.as_tensor(gt_is_repeat(pool.tr))
    if w_new != 1.0:
        w_ev = torch.where(rep_tr, w_ev, w_ev * w_new)
    sub = torch.as_tensor(idx_all)
    log(f"      training events {len(idx_all):,}: GT is a repeat in {float(rep_tr[sub].float().mean()):.3f}"
        f" (w_new={w_new}: new-item events weighted x{w_new})", fh)
    rng = np.random.default_rng(args.seed)
    best_v, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(idx_all)
        for s in range(0, len(idx_all), args.batch):
            bidx = torch.as_tensor(idx_all[s:s + args.batch])
            f, t, i, scal = scal_of(pool, "tr", model, pool.tr, bidx, dev)
            sc = model(f, t, i, scal, pool.tr.base[bidx].float().to(dev), pool.tr.valid[bidx].to(dev),
                       pool.tr.mem[bidx].float().to(dev), pool.tr.mem_pad[bidx].to(dev))
            loss = F.cross_entropy(sc, pool.tr.gt_idx[bidx].to(dev), reduction="none")
            w = w_ev[bidx].to(dev); loss = (loss * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        vh, _ = eval_policies(model, pool, "va", pool.va, dev, args.eval_batch)
        v = float(vh["P0"].mean())                                # early stopping on the unmodified list
        log(f"      ep{ep:02d} VAL t3@10={v:.4f}  (P1 {vh['P1'].mean():.4f}  P2 {vh['P2'].mean():.4f}  "
            f"P3 {vh['P3'].mean():.4f}  D {vh['D'].mean():.4f}; dup slots/top-10 {vh['_dup10'].mean():.3f})", fh)
        if v > best_v + args.min_delta:
            best_v, bad = v, 0
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    spec, spec_val = None, None
    if args.specialist:
        spec, spec_val = train_specialist(pool, dev, args, fh)
    rspec, rspec_val = None, None
    if getattr(args, "repeat_spec", False):
        rspec, rspec_val = train_repeat_specialist(pool, dev, args, fh)
    vh, _ = eval_policies(model, pool, "va", pool.va, dev, args.eval_batch, spec=spec, rspec=rspec)
    val_k = {p: float(vh[p].mean()) for p in POLICIES}
    k_star = max(POLICIES, key=lambda p: val_k[p])                # chosen on VAL
    th, sc = eval_policies(model, pool, "te", pool.te, dev, args.eval_batch, spec=spec, rspec=rspec)
    if getattr(args, "_model_path", None):                       # keep weights for top-k diagnostics
        torch.save({"main": model.state_dict(), "spec": spec.state_dict() if spec is not None else None,
                    "rspec": rspec.state_dict() if rspec is not None else None},
                   args._model_path)
    base_te = metrics(pool.te, pool.te.base.float())
    te = metrics(pool.te, sc)
    agree = float(((te["hit"] > 0) == th["P0"]).mean())
    assert agree > 0.999, f"policy P0 disagrees with metrics() on {1-agree:.4f} of events"
    assert abs(te["cov"] - base_te["cov"]) < 1e-9, "coverage changed under reranking -- BUG"
    return {"w_new": w_new, "specialist": bool(args.specialist), "spec_val": spec_val, "rspec_val": rspec_val,
            "val_k": val_k, "k_star": k_star, "val": best_v, "epochs_run": ep,
            "dup10_test": float(th["_dup10"].mean()), "dup10_val": float(vh["_dup10"].mean()),
            "train_secs": time.perf_counter() - t0, "C": int(pool.te.C),
            "test_k": {p: float(th[p].mean()) for p in POLICIES},
            "hit_k": {p: th[p].astype(np.float64) for p in POLICIES},
            "hit_after": th[k_star].astype(np.float64), "hit_before": base_te["hit"],
            "before": {k: v for k, v in base_te.items() if k not in ("hit", "r3")},
            "after": {k: v for k, v in te.items() if k not in ("hit", "r3")},
            "uid": pool.uid["te"].numpy(), "gt_b": pool.te.gt_b.numpy(), "has_gt": pool.te.has_gt.numpy(),
            "rep": gt_is_repeat(pool.te), "r3_before": base_te["r3"].numpy(), "r3_after": te["r3"].numpy(),
            "ent_z": np.zeros(pool.te.N)}


def canon2(r):
    o = np.argsort(r["uid"]); out = dict(r)
    for k in ("uid", "gt_b", "has_gt", "rep", "hit_after", "hit_before", "r3_before", "r3_after", "ent_z"):
        out[k] = r[k][o]
    out["hit_k"] = {k: v[o] for k, v in r["hit_k"].items()}
    return out


def ms(x):
    x = np.asarray(x, float); n = len(x)
    s = x.std(ddof=1) if n > 1 else float("nan")
    return x.mean(), s, (T975.get(n - 1, 1.96) * s / np.sqrt(n)) if n > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--w_new", default="1,3")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bar", type=float, default=0.01)
    ap.add_argument("--repfeat", action="store_true", help="add repeat-consumption features (see rep_feats)")
    ap.add_argument("--repeat_spec", action="store_true", help="repeat specialist replaces the within-history distribution")
    ap.add_argument("--specialist", action="store_true",
                    help="fill the reserved slots with a new-item specialist reranker instead of the base score")
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/nip_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "nip_report.txt"), "a")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(args.spec)); seeds = [int(s) for s in args.seeds.split(",")]
    wns = [float(x) for x in args.w_new.split(",")]
    global REPFEAT
    REPFEAT = bool(args.repfeat)
    log("=" * 104 + f"\n=== NIP: new-item protection  ({time.strftime('%Y-%m-%d %H:%M')})  seeds={seeds} w_new={wns} ===\n" + "=" * 104, fh)

    res = {}
    for arm in ("MFD", "CH"):
        sp = spec[arm]
        log(f"\n{'#' * 104}\n# ARM {arm}: {os.path.basename(sp['te'])}  K_pool={sp.get('kpool')}\n{'#' * 104}", fh)
        pool = Pool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"))
        assert pool.te.N == 83005
        for wn in wns:
            for sd in seeds:
                cp = os.path.join(args.cache, f"{arm}__wnew{wn:g}{'__spec' if args.specialist else ''}{'__rep' if args.repfeat else ''}{'__rspec' if args.repeat_spec else ''}__seed{sd}.pt")
                if not os.path.exists(cp):
                    a = copy.copy(args); a.seed = sd; a._model_path = cp[:-3] + ".model.pt"
                    log(f"\n  >>> [{arm}] w_new={wn:g} seed {sd}", fh)
                    torch.save(run_cell(pool, wn, args.cap, dev, a, fh), cp)
                r = res[(arm, wn, sd)] = canon2(torch.load(cp, map_location="cpu", weights_only=False))
                log(f"  [{arm}] w_new={wn:g} seed {sd}: VAL " + " ".join(f"{p}={r['val_k'][p]:.4f}" for p in POLICIES if p in r['val_k'])
                    + f" -> chosen {r['k_star']}   TEST " + " ".join(f"{p}={r['test_k'][p]:.4f}" for p in POLICIES if p in r['test_k'])
                    + f"   duplicate-item slots in today's top-10: {r['dup10_test']:.3f}", fh)
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()

    k0 = res[("MFD", wns[0], seeds[0])]
    for key in res:
        assert np.array_equal(res[key]["uid"], k0["uid"]), f"user_id join failed for {key}"
    N = len(k0["uid"]); rng = np.random.default_rng(0)
    mc = k0["has_gt"].astype(bool); cc = res[("CH", wns[0], seeds[0])]["has_gt"].astype(bool)
    rep = k0["rep"].astype(bool); pv = k0["gt_b"] == 0
    # Each arm's history is tokenized in ITS OWN SID space (MD-RQ-VAE vs tiger_1024) with different
    # collisions, so "GT tuple appears in history" can legitimately differ between arms. The split
    # below uses MFD's definition throughout; the disagreement rate is reported, not asserted away.
    rep_ch = res[("CH", wns[0], seeds[0])]["rep"].astype(bool)
    log(f"  repeat-flag agreement between the two SID spaces: {float((rep == rep_ch).mean()):.4f} "
        f"(MFD {rep.mean():.4f} vs chain {rep_ch.mean():.4f}); decomposition uses MFD's", fh)

    log("\n" + "=" * 104 + "\n=== RESULTS: lead at each arm's VAL-chosen policy (and at the unmodified list P0) ===\n" + "=" * 104, fh)
    for wn in wns:
        for pol in ("P0", "k*"):
            ls, lo_hi, mf, chs = [], [], [], []
            for sd in seeds:
                m, c = res[("MFD", wn, sd)], res[("CH", wn, sd)]
                hm = m["hit_k"]["P0"] if pol == "P0" else m["hit_after"]
                hc = c["hit_k"]["P0"] if pol == "P0" else c["hit_after"]
                lo, hi, p = paired_ci(hm, hc, rng, args.boot)
                ls.append(hm.mean() - hc.mean()); mf.append(hm.mean()); chs.append(hc.mean())
                lo_hi.append((lo, hi))
            m_, s_, h_ = ms(ls)
            ks = "" if pol == "P0" else "  k*: MFD " + ",".join(str(res[("MFD", wn, sd)]["k_star"]) for sd in seeds) \
                + " / chain " + ",".join(str(res[("CH", wn, sd)]["k_star"]) for sd in seeds)
            log(f"  w_new={wn:g} {pol:<3} MFD {np.mean(mf):.4f}  chain {np.mean(chs):.4f}  lead {m_:+.5f} "
                f"(std {s_:.5f}, t-interval [{m_-h_:+.5f}, {m_+h_:+.5f}])  per-seed CIs "
                + " ".join(f"[{a:+.4f},{b:+.4f}]" for a, b in lo_hi) + ks
                + f"  -> BAR {'MET' if m_ - h_ >= args.bar else 'not met'} (short {max(0, args.bar - m_):+.5f})", fh)

    log("\n" + "=" * 104 + "\n=== DECOMPOSITION of the lead: pool membership x {repeat, new item}, pv only, mean over seeds ===\n" + "=" * 104, fh)
    pops = {"BOTH": mc & cc, "MFD-ONLY": mc & ~cc, "CHAIN-ONLY": cc & ~mc}
    for wn in wns:
        log(f"  w_new={wn:g} (k* policy; P0 in brackets):", fh)
        for nm, p in pops.items():
            for rnm, rr in (("repeat", rep), ("new item", ~rep)):
                q = p & rr & pv
                if not q.sum():
                    continue
                mh = np.mean([res[("MFD", wn, sd)]["hit_after"][q].mean() for sd in seeds])
                ch = np.mean([res[("CH", wn, sd)]["hit_after"][q].mean() for sd in seeds])
                mh0 = np.mean([res[("MFD", wn, sd)]["hit_k"]["P0"][q].mean() for sd in seeds])
                ch0 = np.mean([res[("CH", wn, sd)]["hit_k"]["P0"][q].mean() for sd in seeds])
                log(f"    {nm:<11}{rnm:<9} n={int(q.sum()):>6,}  MFD {mh:.4f} [{mh0:.4f}]  chain {ch:.4f} [{ch0:.4f}]  "
                    f"lead contrib {(mh-ch)*q.sum()/N:+.5f} [{(mh0-ch0)*q.sum()/N:+.5f}]", fh)
    log("\n  reading: the deciding cell is MFD-ONLY x new item -- ground truths chain's pool does not contain.", fh)
    log("  Any gain there moves the lead one-for-one and is attributable to MFD's pool, not to the policy.", fh)
    fh.close()


if __name__ == "__main__":
    main()
