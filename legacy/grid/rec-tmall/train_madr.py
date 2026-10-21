"""MADR: Modality-Aware Dual-channel Ranker. Rescores the frozen MFD candidate pool with TWO
structurally separated scoring channels (text, image) instead of one network over concatenated
views. Trained offline on the existing dumps; the main model is frozen.

Per candidate c=(f,t,i), views lifted through the same orthogonal W_up: v_t=W_up(e_f+e_t)=U_f[f]+U_t[t],
v_i=U_f[f]+U_i[i]. Each channel has its OWN cross-attention over the FULL encoder memory and its own
MLP (no shared params):
  q_x = CrossAttn_x(Linear_x(v_x), enc_mem)          # attended text/image evidence
  s_x = MLP_x([ q_x * v_x , q_x , v_x ])             # scalar channel score
Fusion (able to express cross-modal AGREEMENT, not a weighted sum):
  score = MLP_g([ s_t, s_i, s_t - s_i, s_t * s_i, logP(b), logP(f), logP(t|f), logP(i|f) ])

Listwise CE over the real pool, GT positive, ONLY on covered events; negatives are the model's own
generated candidates. Select on VAL, report on TEST. Channels small (2 layers, d=128, dropout) --
~98k covered training events.

Diagnostics (primary output): corr(s_t,s_i) [>0.9 => channels collapsed]; q_t vs q_i attention
divergence over the memory; per-channel ablation (text/image/both); view substitution (random views).
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mfd_reranker import (BEH, RerankData, best_rank, load_geometry, ndcg_at, orth_lift,
                                 recall_at)

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


def assert_split(data, fh):
    per = {b: int((data.gt_b == b).sum()) for b in range(data.nb)}
    ok = (data.N == EXPECT_TOTAL) and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  TEST split: total={data.N:,}  " + "  ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH)
        + f"  -> {'OK' if ok else 'MISMATCH'}", fh)
    if not ok:
        raise SystemExit("** TEST split mismatch -- refusing to emit numbers. **")


# =============================================================================== model
class Channel(nn.Module):
    """One modality channel: query = Linear(view) cross-attends the memory; MLP over [q*v, q, v]."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.qproj = nn.Linear(d, d)
        self.norm_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, 2 * d), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(2 * d, 1))

    def forward(self, v, mem, mem_pad, need_attn=False):
        q = self.norm_q(self.qproj(v))                              # (B,C,d) query per candidate
        ctx, attw = self.attn(q, mem, mem, key_padding_mask=mem_pad,
                              need_weights=need_attn, average_attn_weights=True)
        s = self.mlp(torch.cat([ctx * v, ctx, v], dim=-1)).squeeze(-1)  # (B,C)
        return s, attw                                             # attw (B,C,S) or None


class MADR(nn.Module):
    def __init__(self, U_f, U_t, U_i, d, heads=4, drop=0.3, channels="both", n_logp=4):
        super().__init__()
        self.register_buffer("U_f", U_f); self.register_buffer("U_t", U_t); self.register_buffer("U_i", U_i)
        self.channels = channels
        self.ch_t = Channel(d, heads, drop)
        self.ch_i = Channel(d, heads, drop)
        self.fuse = nn.Sequential(nn.Linear(4 + n_logp, 4 * d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(4 * d, 1))

    def views(self, f, t, i):
        vf = self.U_f[f]
        return vf + self.U_t[t], vf + self.U_i[i]                  # v_t, v_i  (B,C,d)

    def forward(self, f, t, i, lp, mem, mem_pad, need_attn=False):
        v_t, v_i = self.views(f, t, i)
        s_t, at = self.ch_t(v_t, mem, mem_pad, need_attn)
        s_i, ai = self.ch_i(v_i, mem, mem_pad, need_attn)
        if self.channels == "text":
            s_i = torch.zeros_like(s_i)
        elif self.channels == "image":
            s_t = torch.zeros_like(s_t)
        g = torch.stack([s_t, s_i, s_t - s_i, s_t * s_i], dim=-1)  # (B,C,4)
        g = torch.cat([g, lp], dim=-1)                            # + [logP(b),logP(f),logP(t|f),logP(i|f)]
        score = self.fuse(g).squeeze(-1)                          # (B,C)
        return score, s_t, s_i, at, ai


# =============================================================================== batching
def batch_events(data, idx, device):
    f = data.f[idx].long().to(device); t = data.t[idx].long().to(device); i = data.i[idx].long().to(device)
    lp = data.lp[idx].float().to(device)
    mem = data.mem[idx].float().to(device); mp = data.mem_pad[idx].to(device)
    valid = data.valid[idx].to(device)
    return f, t, i, lp, mem, mp, valid


@torch.no_grad()
def all_scores(model, data, device, eb=16):
    model.eval()
    out = torch.empty(data.N, data.C)
    for s in range(0, data.N, eb):
        e = slice(s, min(s + eb, data.N))
        f, t, i, lp, mem, mp, valid = batch_events(data, e, device)
        sc, *_ = model(f, t, i, lp, mem, mp)
        out[e] = sc.masked_fill(~valid, float("-inf")).cpu()
    return out


# =============================================================================== eval / metrics
def ranks(score, data):
    sid = ((data.f == data.gt_f[:, None]) & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]))
    beh = data.branch.view(1, -1) == data.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, data.valid)
    r2, h2 = best_rank(score, sid, beh & data.valid)
    return r3, h3, r2, h2


def mechanism(score, data):
    cov = data.has_gt
    gt_is = ((data.branch.view(1, -1) == data.gt_b[:, None]) & (data.f == data.gt_f[:, None])
             & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]) & data.valid)
    gts = torch.where(gt_is, score, torch.full_like(score, float("-inf"))).max(1).values
    srt = score.sort(1, descending=True).values
    gap = (srt[:, 9] - srt[:, 10])[cov]
    band = ((score >= (gts.unsqueeze(1) - 0.5)) & (score <= (gts.unsqueeze(1) + 0.5)) & data.valid).sum(1)[cov]
    return gap.mean().item(), band.float().mean().item()


def full_report(tag, data, score, fh):
    r3, h3, r2, h2 = ranks(score, data); gt_b = data.gt_b; cov = data.has_gt
    gap, band = mechanism(score, data)
    log(f"\n  [{tag}]", fh)
    log(f"    t3 R@5={recall_at(r3,h3,5):.4f} R@10={recall_at(r3,h3,10):.4f} R@20={recall_at(r3,h3,20):.4f} "
        f"N@5={ndcg_at(r3,h3,5):.4f} N@10={ndcg_at(r3,h3,10):.4f}", fh)
    log(f"    t2 R@5={recall_at(r2,h2,5):.4f} R@10={recall_at(r2,h2,10):.4f} R@20={recall_at(r2,h2,20):.4f} "
        f"N@5={ndcg_at(r2,h2,5):.4f} N@10={ndcg_at(r2,h2,10):.4f}", fh)
    log(f"    P(rank<10|cov)={recall_at(r3,h3,10,cov):.4f} (base 0.7897, chain 0.8581)  coverage={cov.float().mean().item():.4f}", fh)
    log(f"    rank10->11 gap={gap:.4f} (base 0.1485, chain 0.3213)   #within +/-0.5 of GT={band:.2f} (base 32.59, chain 22.09)", fh)
    log(f"    t3@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r3,h3,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    log(f"    t2@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r2,h2,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    return {"t3@10": recall_at(r3, h3, 10), "pcov": recall_at(r3, h3, 10, cov), "gap": gap, "band": band,
            "perbeh3": {b: recall_at(r3, h3, 10, gt_b == b) for b in range(data.nb)}}


# =============================================================================== training
def train_one(tr, va, te, U_f, U_t, U_i, d, device, args, channels, tag, fh, epochs):
    torch.manual_seed(args.seed)
    model = MADR(U_f, U_t, U_i, d, args.heads, args.dropout, channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx = tr.train_idx.numpy(); rng = np.random.default_rng(args.seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n=== training MADR [{tag}] channels={channels}  params={n_params/1e3:.1f}k  epochs<= {epochs} ===", fh)

    def val_t3():
        r3, h3, _, _ = ranks(all_scores(model, va, device, args.eval_batch), va)
        return recall_at(r3, h3, 10)

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); rng.shuffle(idx); tl = 0.0
        for s in range(0, len(idx), args.batch):
            bidx = torch.as_tensor(idx[s:s + args.batch])
            f, t, i, lp, mem, mp, valid = batch_events(tr, bidx, device)
            gt = tr.gt_idx[bidx].to(device)
            sc, *_ = model(f, t, i, lp, mem, mp)
            sc = sc.masked_fill(~valid, float("-inf"))
            loss = F.cross_entropy(sc, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(bidx)
        vt3 = val_t3()
        log(f"  [{tag}] ep{ep:02d} train_ce={tl/len(idx):.4f}  VAL t3@10={vt3:.4f}", fh)
        if vt3 > best + 1e-5:
            best, best_state, bad = vt3, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  [{tag}] early stop at ep{ep} (best VAL t3@10={best:.4f})", fh); break
    if best_state:
        model.load_state_dict(best_state)
    return model


# =============================================================================== diagnostics
@torch.no_grad()
def channel_diagnostics(model, data, device, fh, sample=3000, eb=8):
    """corr(s_t,s_i) across candidates, and mean JS divergence between q_t and q_i attention maps."""
    model.eval()
    cov = torch.nonzero(data.has_gt).squeeze(1)
    if len(cov) > sample:
        cov = cov[torch.randperm(len(cov))[:sample]]
    st_all, si_all, js_all = [], [], []
    for s in range(0, len(cov), eb):
        idx = cov[s:s + eb]
        f, t, i, lp, mem, mp, valid = batch_events(data, idx, device)
        _, s_t, s_i, at, ai = model(f, t, i, lp, mem, mp, need_attn=True)
        m = valid
        st_all.append(s_t[m].cpu()); si_all.append(s_i[m].cpu())
        # JS divergence between the two attention distributions over memory positions
        p = at.clamp(min=1e-9); qd = ai.clamp(min=1e-9)
        mid = 0.5 * (p + qd)
        js = 0.5 * (p * (p / mid).log()).sum(-1) + 0.5 * (qd * (qd / mid).log()).sum(-1)  # (B,C)
        js_all.append(js[m].cpu())
    st = torch.cat(st_all); si = torch.cat(si_all); js = torch.cat(js_all)
    corr = torch.corrcoef(torch.stack([st, si]))[0, 1].item()
    log("\n=== CHANNEL DIAGNOSTICS (primary 'both' model) ===", fh)
    log(f"  corr(s_t, s_i) across candidates = {corr:.4f}  "
        f"{'** COLLAPSED (>0.9): separation too weak for two channels **' if abs(corr)>0.9 else '(distinct channels)'}", fh)
    log(f"  q_t vs q_i attention JS divergence over memory: mean={js.mean().item():.4f}  "
        f"(0 = identical attention; larger = the two queries read different history positions)", fh)
    log(f"  channel score stats: s_t mean={st.mean():.3f} std={st.std():.3f}  s_i mean={si.mean():.3f} std={si.std():.3f}", fh)


def siglip_displacers(score, data, sid_path, tf_mm, id2row_t, if_mm, id2row_i, fh, sample=4000, seed=0):
    from attribute_ordering_gap import build_tuple_to_item
    keys, items, w1, w2 = build_tuple_to_item(sid_path)

    def lookup(f, t, i):
        k = int((int(f) * w1 + int(t)) * w2 + int(i)); p = np.searchsorted(keys, k)
        return int(items[p]) if p < len(keys) and keys[p] == k else -1

    def feat(iid):
        rt = id2row_t.get(int(iid), -1); ri = id2row_i.get(int(iid), -1)
        if rt < 0 or ri < 0:
            return None
        tt = tf_mm[rt]; ii = if_mm[ri]
        tt = tt / (np.linalg.norm(tt) + 1e-9); ii = ii / (np.linalg.norm(ii) + 1e-9)
        return np.concatenate([tt, ii])

    r3, h3, _, _ = ranks(score, data)
    rng = np.random.default_rng(seed)
    cov = torch.nonzero(data.has_gt & (r3 >= 1)).squeeze(1).numpy()
    if len(cov) > sample:
        cov = rng.choice(cov, sample, replace=False)
    sims = []
    gt_is = ((data.branch.view(1, -1) == data.gt_b[:, None]) & (data.f == data.gt_f[:, None])
             & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]) & data.valid)
    for e in cov.tolist():
        gtf = feat(lookup(data.gt_f[e], data.gt_t[e], data.gt_i[e]))
        if gtf is None:
            continue
        sc = score[e]; gt_sc = torch.where(gt_is[e], sc, torch.full_like(sc, float("-inf"))).max()
        outr = torch.nonzero((sc > gt_sc) & data.valid[e]).squeeze(1).tolist()[:20]
        for j in outr:
            df = feat(lookup(data.f[e][j], data.t[e][j], data.i[e][j]))
            if df is not None:
                sims.append(float(df @ gtf))
    log(f"  SigLIP2 GT-vs-displacer similarity (MADR ranking): mean={np.mean(sims):.4f} over {len(sims):,} pairs "
        f"(base 1.5049)" if sims else "  SigLIP2 displacer check: no mappable pairs", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dump", required=True); ap.add_argument("--val_dump", required=True)
    ap.add_argument("--test_dump", required=True); ap.add_argument("--mdrqvae_ckpt", required=True)
    ap.add_argument("--centroid_cache", default="/scratch/yw8866/rec-tmall/mdrqvae_centroids.pt")
    ap.add_argument("--fusion_weights", default="/scratch/yw8866/rec-tmall/mfd_fusion_weights.pt",
                    help="learned 4-term fusion weights (from promote_fusion) -> the baseline to beat")
    ap.add_argument("--md_sid", default=""); ap.add_argument("--text_npy", default=""); ap.add_argument("--text_ids", default="")
    ap.add_argument("--image_npy", default=""); ap.add_argument("--image_ids", default="")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--abl_epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/madr")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr = RerankData(args.train_dump, args.wb); va = RerankData(args.val_dump, args.wb); te = RerankData(args.test_dump, args.wb)
    assert_split(te, fh)
    d = tr.mem.shape[2]
    U_f, U_t, U_i = load_geometry(args.mdrqvae_ckpt, args.centroid_cache, d, args.seed, device)
    log(f"  geometry d={d}  K_f={U_f.shape[0]} K_t={U_t.shape[0]} K_i={U_i.shape[0]}", fh)
    cov = te.has_gt.float().mean().item()

    # baseline to beat = the learned 4-term fusion (0.7897); fall back to hand-set union if absent.
    if os.path.exists(args.fusion_weights):
        w = torch.load(args.fusion_weights, map_location="cpu", weights_only=False)["weights"].float()
        base_score = (te.lp.float() * w).sum(-1)
        base_tag = f"BASELINE learned fusion w={w.tolist()}"
    else:
        base_score = te.base.float()
        base_tag = "BASELINE hand-set union (fusion weights not found)"
    log("\n" + "=" * 84 + f"\n=== {base_tag} ===\n" + "=" * 84, fh)
    base_te = full_report(base_tag, te, base_score, fh)
    full_report("(reference) hand-set union", te, te.base.float(), fh)

    # ---- PRIMARY: both channels ----
    m_both = train_one(tr, va, te, U_f, U_t, U_i, d, device, args, "both", "BOTH", fh, args.epochs)
    sc_both = all_scores(m_both, te, device, args.eval_batch)
    log("\n" + "=" * 84 + "\n=== TEST -- MADR (both channels, PRIMARY) ===\n" + "=" * 84, fh)
    r_both = full_report("MADR both", te, sc_both, fh)
    channel_diagnostics(m_both, te, device, fh)
    if args.md_sid and args.text_npy:
        from attribute_ordering_gap import load_id_map, open_embeddings
        id2t = load_id_map(args.text_ids); id2i = load_id_map(args.image_ids)
        tfm = open_embeddings(args.text_npy, len(id2t)); ifm = open_embeddings(args.image_npy, len(id2i))
        siglip_displacers(sc_both, te, args.md_sid, tfm, id2t, ifm, id2i, fh, seed=args.seed)
    torch.save({"state": m_both.state_dict(), "channels": "both"}, os.path.join(args.out_dir, "madr_both.pt"))

    # coverage invariance
    cb = ranks(te.base.float(), te)[1].float().mean().item(); cm = ranks(sc_both, te)[1].float().mean().item()
    log(f"\n  coverage invariance: base={cb:.4f} MADR={cm:.4f} {'OK' if abs(cb-cm)<1e-6 else '** DRIFT (bug) **'}", fh)
    assert abs(cb - cm) < 1e-6, "coverage drifted under rescoring -- BUG"

    # ---- ABLATIONS: text-only, image-only, random-views ----
    log("\n" + "=" * 84 + "\n=== ABLATIONS ===\n" + "=" * 84, fh)
    m_t = train_one(tr, va, te, U_f, U_t, U_i, d, device, args, "text", "TEXT-ONLY", fh, args.abl_epochs)
    r_t = full_report("MADR text-only", te, all_scores(m_t, te, device, args.eval_batch), fh)
    m_i = train_one(tr, va, te, U_f, U_t, U_i, d, device, args, "image", "IMAGE-ONLY", fh, args.abl_epochs)
    r_i = full_report("MADR image-only", te, all_scores(m_i, te, device, args.eval_batch), fh)
    # view substitution: fixed random views, same architecture
    Rf = orth_lift(torch.randn_like(U_f.cpu()), d, args.seed + 7).to(device)
    Rt = orth_lift(torch.randn_like(U_t.cpu()), d, args.seed + 8).to(device)
    Ri = orth_lift(torch.randn_like(U_i.cpu()), d, args.seed + 9).to(device)
    m_r = train_one(tr, va, te, Rf, Rt, Ri, d, device, args, "both", "RANDOM-VIEWS", fh, args.abl_epochs)
    r_r = full_report("MADR random-views", te, all_scores(m_r, te, device, args.eval_batch), fh)

    # ---- summary ----
    log("\n" + "=" * 84 + "\n=== SUMMARY ===\n" + "=" * 84, fh)
    log(f"  {'variant':<20}{'t3@10':>9}{'P(r<10|c)':>11}{'gap10-11':>10}{'#band':>8}", fh)
    for nm, r in [("baseline", base_te), ("MADR both", r_both), ("text-only", r_t),
                  ("image-only", r_i), ("random-views", r_r)]:
        log(f"  {nm:<20}{r['t3@10']:>9.4f}{r['pcov']:>11.4f}{r['gap']:>10.4f}{r['band']:>8.2f}", fh)
    log("\n  per-behavior t3@10 (baseline -> MADR both):", fh)
    for b in range(te.nb):
        log(f"    {BEH[b]:<5} {base_te['perbeh3'][b]:.4f} -> {r_both['perbeh3'][b]:.4f}", fh)
    log("\n  Reads: (a) does MADR both beat the 0.7897 P(rank<10|cov) baseline and widen the rank10-11", fh)
    log("  gap / shrink the crowding band? (b) do BOTH channels beat either alone (complementary)?", fh)
    log("  (c) does random-views DEGRADE vs learned views (evidence the views carry usable info,", fh)
    log("  unlike the concat ranker)? (d) does MADR fix the conversion regression or repeat it?", fh)
    log("  If corr(s_t,s_i)>0.9 the channels collapsed -> separation too weak; report that plainly.", fh)
    fh.close()


if __name__ == "__main__":
    main()
