"""Chain-conditional RESCORING on top of the frozen MFD + PMQ + Q_f pool.

MFD drops the chain edge t->i to decode in parallel. That edge is unnecessary for GENERATION but
not for SCORING: once the pool is fixed, logP(i | f, t) is one batched forward pass, no beam search.
This script keeps MFD's factorized pool untouched (coverage is invariant) and only RE-RANKS it.

STAGE A  feasibility (on the existing MFD pool dump, no model): does the chain conditional carry
         signal on FREELY GENERATED candidates? Fraction of correct-t | correct-f (per behavior),
         GT-vs-rest separation of the parallel score among correct-f candidates, and the image-order
         headroom among candidates that share the GT (f,t). Prints a GO / NO-GO.
STAGE B  chain-conditional head P(i | f, t): a small cross-attention module reads the encoder memory
         (the same context the frozen decoder's f-slot hidden attends to) conditioned on the lifted
         (f,t) code embeddings, and outputs a distribution over the image codebook. Trained TEACHER-
         FORCED on GT (f,t)->i, everything else frozen. Train/inference mismatch is explicit (train
         sees GT t, inference sees candidate t, often wrong); optional scheduled sampling (default off).
STAGE C  learned fusion over the five terms logP(b), logP(f), logP(t|f), logP(i|f), logP(i|f,t),
         fit on the eval split; reports the weights and the weight the chain term earns, vs the
         hand-set baseline (tau=1, w_b=1) and vs a chain-term-ablated fusion.

Coverage cannot change (generation untouched): asserted score-invariant. Test split counts asserted.
Metrics use the SAME best_rank / recall_at / ndcg_at primitives as the reranker.
Reference (do NOT tune toward): MFD alone t3@10 0.1904 t2@10 0.2016; chain t3@10 0.1942 t2@10 0.2032;
MFD+listwise reranker t3@10 0.2150 t2@10 0.2225. P(rank<10|cov) baseline 0.7805, chain ref 0.8581.
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mfd_reranker import (BEH, CrossBlock, RerankData, best_rank, load_geometry,
                                 ndcg_at, recall_at)

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n"); fh.flush()


def assert_split(data, fh):
    n = data.N
    per = {b: int((data.gt_b == b).sum()) for b in range(data.nb)}
    ok = (n == EXPECT_TOTAL) and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  TEST split check: total={n:,} (expect {EXPECT_TOTAL:,})  "
        + "  ".join(f"{BEH[b]}={per.get(b,0):,}(exp {EXPECT_PERBEH[b]:,})" for b in EXPECT_PERBEH), fh)
    if not ok:
        raise SystemExit("** TEST split mismatch -- refusing to emit numbers (the 45k-pv-subsample bug). **")


# ============================================================================ STAGE A
def stage_a(dump_path, wb, fh):
    log("\n" + "=" * 88 + "\n=== STAGE A: feasibility of the chain conditional on FREELY GENERATED candidates ===\n"
        + "=" * 88, fh)
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    nb = meta["nb"]; N, _, Kp = c["pool_f"].shape
    f32 = lambda x: x.float() if x.dtype == torch.float16 else x
    flat = lambda x: x.reshape(N, nb * Kp)
    pf, pt, pi = flat(c["pool_f"]).long(), flat(c["pool_t"]).long(), flat(c["pool_i"]).long()
    valid = f32(flat(c["pool_sidscore"])) > -1e30
    branch = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1)
    gt_b, gt_f, gt_t, gt_i = c["gt_b"].long(), c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    lp_f, lp_t, lp_i = f32(flat(c["pool_lpf"])), f32(flat(c["pool_lpt"])), f32(flat(c["pool_lpi"]))
    lp_b = torch.gather(f32(c["beh_logp"]), 1, branch)
    par = wb * lp_b + lp_f + lp_t + lp_i                      # parallel union score
    log(f"  pool dump: N={N:,}  nb={nb}  Kp={Kp}  (from {os.path.basename(dump_path)})", fh)

    correct_f = (branch == gt_b[:, None]) & (pf == gt_f[:, None]) & valid
    correct_ft = correct_f & (pt == gt_t[:, None])
    # (1) fraction of correct-t among correct-f candidates, overall + per behavior
    log("\n  (1) among candidates with the correct f (and behavior), fraction that ALSO have correct t:", fh)
    log(f"      {'group':<8}{'#events w/correct-f':>22}{'mean #correct-f':>18}{'frac t correct':>16}", fh)
    for grp, name in [(torch.ones(N, dtype=torch.bool), "ALL")] + [(gt_b == b, BEH[b]) for b in range(nb)]:
        cf = correct_f[grp]; cft = correct_ft[grp]
        has_cf = cf.any(1)
        if has_cf.sum() == 0:
            log(f"      {name:<8}{'0':>22}", fh); continue
        n_cf = cf.sum(1).float()
        frac_t = (cft.sum(1).float()[has_cf] / n_cf[has_cf].clamp(min=1)).mean().item()
        log(f"      {name:<8}{int(has_cf.sum()):>22,}{n_cf[has_cf].mean().item():>18.2f}{frac_t:>16.4f}", fh)

    # (2) GT vs rest separation among correct-f candidates, under the parallel score
    gt_is = (branch == gt_b[:, None]) & (pf == gt_f[:, None]) & (pt == gt_t[:, None]) & (pi == gt_i[:, None]) & valid
    has_gt = gt_is.any(1)
    par_gt = torch.where(gt_is, par, torch.full_like(par, float("-inf"))).max(1).values
    rest = correct_f & (~gt_is)
    par_rest_mean = (torch.where(rest, par, torch.zeros_like(par)).sum(1) / rest.sum(1).clamp(min=1))
    sep = (par_gt - par_rest_mean)[has_gt & rest.any(1)]
    # AUC-like: P(par(GT) > max par over correct-f non-GT)
    par_rest_max = torch.where(rest, par, torch.full_like(par, float("-inf"))).max(1).values
    auc_like = ((par_gt > par_rest_max)[has_gt & rest.any(1)]).float().mean().item()
    log("\n  (2) parallel-score separation of GT vs other correct-f candidates:", fh)
    log(f"      mean(par_GT - mean par_rest) = {sep.mean().item():+.4f}   "
        f"P(par_GT > max par_rest) = {auc_like:.4f}   (over {int((has_gt&rest.any(1)).sum()):,} events)", fh)

    # (3) image-order headroom: among candidates sharing GT (f,t), rank of GT's image code by the
    #     PARALLEL image term logP(i|f). If GT i is often NOT top among (f,t)-siblings, a chain
    #     conditional logP(i|f,t) has room to reorder.
    sib = correct_ft & valid                                 # candidates with GT f AND GT t
    has_sib = sib.any(1) & has_gt
    lp_i_masked = torch.where(sib, lp_i, torch.full_like(lp_i, float("-inf")))
    gt_lp_i = torch.where(gt_is, lp_i, torch.full_like(lp_i, float("-inf"))).max(1).values
    rank_i = (lp_i_masked > gt_lp_i.unsqueeze(1)).sum(1)      # #siblings with higher logP(i|f) than GT
    n_sib = sib.sum(1)
    hs = has_sib
    log("\n  (3) image-order headroom among GT-(f,t) siblings (candidates sharing GT f AND GT t):", fh)
    log(f"      events with >=1 sibling & GT covered = {int(hs.sum()):,}   mean #siblings = "
        f"{n_sib[hs].float().mean().item():.2f}", fh)
    log(f"      GT's image rank by parallel logP(i|f): mean={rank_i[hs].float().mean().item():.3f}  "
        f"frac GT already top-1 = {(rank_i[hs]==0).float().mean().item():.4f}", fh)

    # GO / NO-GO
    go = (auc_like < 0.95) and (n_sib[hs].float().mean().item() > 1.05) and ((rank_i[hs] > 0).float().mean().item() > 0.02)
    log("\n  --- STAGE A VERDICT ---", fh)
    if go:
        log("  GO: the parallel score does NOT perfectly separate GT among correct-f candidates, and GT-(f,t)", fh)
        log("  siblings exist whose image order the parallel logP(i|f) gets wrong -- logP(i|f,t) has room to", fh)
        log("  help. Proceed to Stages B/C (which give the direct empirical answer via P(rank<10|covered)).", fh)
    else:
        log("  NO-GO signal: the parallel score already orders correct-f candidates near-perfectly and/or GT", fh)
        log("  rarely has competing (f,t)-siblings, so a chain conditional has little to reorder. Stages B/C", fh)
        log("  still run below, but expect little gain -- report the negative rather than tuning toward it.", fh)
    return {"auc_like": auc_like, "go": go}


# ============================================================================ STAGE B
class ChainHead(nn.Module):
    """logP(i | f, t): lifted (f,t) code embeddings -> a query token that cross-attends into the
    frozen encoder memory (the context the decoder's f-slot hidden reads), -> image-codebook logits."""

    def __init__(self, U_f, U_t, d_model, k_i, layers=1, heads=4, drop=0.1):
        super().__init__()
        self.register_buffer("U_f", U_f)          # frozen lifted centroids (K_f, d)
        self.register_buffer("U_t", U_t)          # (K_t, d)
        self.q = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.blocks = nn.ModuleList([CrossBlock(d_model, heads, drop) for _ in range(layers)])
        self.out = nn.Linear(d_model, k_i)

    def forward(self, f, t, mem, mem_pad):
        # f,t (B,Q) long; mem (B,S,d); mem_pad (B,S) True=pad -> log P(i|f,t) (B,Q,k_i)
        x = self.q(torch.cat([self.U_f[f], self.U_t[t]], dim=-1))
        for blk in self.blocks:
            x = blk(x, mem, mem_pad)
        return torch.log_softmax(self.out(x), dim=-1)


def train_head(head, tr, va, device, args, fh):
    log("\n" + "=" * 88 + "\n=== STAGE B: training the chain-conditional head P(i|f,t) (teacher-forced) ===\n"
        + "=" * 88, fh)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.wd)
    idx_tr = tr.train_idx.numpy()                 # GT-in-pool events (GT (f,t,i) available in pool)
    rng = np.random.default_rng(args.seed)

    def run_eval(data):
        head.eval(); tot, corr, ce = 0, 0, 0.0
        with torch.no_grad():
            for s in range(0, data.N, args.eval_batch):
                e = slice(s, min(s + args.eval_batch, data.N))
                B = e.stop - e.start
                mem = data.mem[e].float().to(device); mp = data.mem_pad[e].to(device)
                f = data.gt_f[e].view(B, 1).to(device); t = data.gt_t[e].view(B, 1).to(device)
                gi = data.gt_i[e].to(device)
                lg = head(f, t, mem, mp)[:, 0, :]        # (B,k_i)
                ce += F.nll_loss(lg, gi, reduction="sum").item()
                corr += (lg.argmax(1) == gi).sum().item(); tot += B
        return ce / tot, corr / tot

    best_ce, best_state, bad = 1e9, None, 0
    for ep in range(1, args.head_epochs + 1):
        head.train(); rng.shuffle(idx_tr); tl = 0.0
        for s in range(0, len(idx_tr), args.batch):
            bidx = torch.as_tensor(idx_tr[s:s + args.batch])
            B = bidx.numel()
            mem = tr.mem[bidx].float().to(device); mp = tr.mem_pad[bidx].to(device)
            f = tr.gt_f[bidx].view(B, 1).to(device)
            t = tr.gt_t[bidx].view(B, 1).clone()
            if args.ss_prob > 0:                    # scheduled sampling: mix in a wrong (pool) t
                mask = torch.rand(B) < args.ss_prob
                if mask.any():
                    rndc = torch.randint(0, tr.C, (int(mask.sum()),))
                    t[mask] = tr.t[bidx[mask], rndc].long().view(-1, 1)
            t = t.to(device); gi = tr.gt_i[bidx].to(device)
            lg = head(f, t, mem, mp)[:, 0, :]
            loss = F.nll_loss(lg, gi)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * B
        vce, vacc = run_eval(va)
        log(f"  ep{ep:02d}  train_nll={tl/len(idx_tr):.4f}   VAL nll={vce:.4f}  top1_acc(i|f,t)={vacc:.4f}", fh)
        if vce < best_ce - 1e-4:
            best_ce, best_state, bad = vce, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.head_patience:
                log(f"  early stop (patience {args.head_patience}) at ep{ep}", fh); break
    if best_state:
        head.load_state_dict(best_state)
    log(f"  best VAL nll={best_ce:.4f}", fh)


@torch.no_grad()
def head_scores(head, data, device, eval_batch):
    """Per-candidate logP(i|f,t) over the whole pool -> (N, C)."""
    head.eval()
    out = torch.empty(data.N, data.C)
    for s in range(0, data.N, eval_batch):
        e = slice(s, min(s + eval_batch, data.N))
        mem = data.mem[e].float().to(device); mp = data.mem_pad[e].to(device)
        f = data.f[e].long().to(device); t = data.t[e].long().to(device)
        i = data.i[e].long()
        lg = head(f, t, mem, mp)                                  # (B,C,k_i)
        gathered = lg.gather(2, i.to(device).unsqueeze(-1)).squeeze(-1)  # (B,C) = logP(cand_i | f,t)
        out[e] = gathered.cpu()
    return out


# ============================================================================ STAGE C
def five_terms(data, chain, wb):
    """Stack the 5 fusion terms (N,C,5): logP(b), logP(f), logP(t|f), logP(i|f), logP(i|f,t)."""
    lp = data.lp.float()                                          # (N,C,4)=[b,f,t|f,i|f]
    return torch.stack([lp[..., 0], lp[..., 1], lp[..., 2], lp[..., 3], chain], dim=-1)


def fit_fusion(terms, data, device, use_terms, fh, steps=400, lr=0.05, seed=0):
    """Fit linear weights over the selected term indices by listwise CE on GT-in-pool events."""
    torch.manual_seed(seed)
    idx = data.train_idx.to(device)                              # GT covered on THIS (val) split
    T = terms[..., use_terms].to(device)                         # (N,C,k)
    valid = data.valid.to(device); gt_idx = data.gt_idx.to(device)
    w = torch.zeros(len(use_terms), device=device, requires_grad=True)
    with torch.no_grad():                                        # init log-prob terms at 1 (= hand-set sum)
        for j, u in enumerate(use_terms):
            if u < 4:
                w[j] = 1.0
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        sc = (T[idx] * w).sum(-1).masked_fill(~valid[idx], float("-inf"))
        loss = F.cross_entropy(sc, gt_idx[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach()


# ============================================================================ EVAL
def score_ranks(score, data):
    valid = data.valid
    sid = ((data.f == data.gt_f[:, None]) & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]))
    beh = data.branch.view(1, -1) == data.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, valid)                  # union / behavior-marginalized (t3)
    r2, h2 = best_rank(score, sid, beh & valid)                 # routed to GT behavior (t2)
    return r3, h3, r2, h2


def report(tag, data, score, fh, coverage=None):
    r3, h3, r2, h2 = score_ranks(score, data)
    gt_b = data.gt_b
    covered = data.has_gt
    p_cov = recall_at(r3, h3, 10, covered)
    log(f"\n  [{tag}]  t3: R@5={recall_at(r3,h3,5):.4f} R@10={recall_at(r3,h3,10):.4f} "
        f"R@20={recall_at(r3,h3,20):.4f} N@5={ndcg_at(r3,h3,5):.4f} N@10={ndcg_at(r3,h3,10):.4f}", fh)
    log(f"        t2: R@5={recall_at(r2,h2,5):.4f} R@10={recall_at(r2,h2,10):.4f} "
        f"R@20={recall_at(r2,h2,20):.4f} N@5={ndcg_at(r2,h2,5):.4f} N@10={ndcg_at(r2,h2,10):.4f}", fh)
    log(f"        P(rank<10 | covered) = {p_cov:.4f}   (baseline 0.7805, chain ref 0.8581)", fh)
    if coverage is not None:
        log(f"        coverage@pool = {coverage:.4f} (score-invariant; ref 0.2615)", fh)
    p3 = "  ".join(f"{BEH[b]}={recall_at(r3,h3,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any())
    p2 = "  ".join(f"{BEH[b]}={recall_at(r2,h2,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any())
    log(f"        t3@10 per-beh: {p3}", fh)
    log(f"        t2@10 per-beh: {p2}", fh)
    return recall_at(r3, h3, 10), p_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_cmp_dump", required=True, help="existing MFD pool dump for Stage A")
    ap.add_argument("--train_dump", required=True)
    ap.add_argument("--val_dump", required=True)
    ap.add_argument("--test_dump", required=True)
    ap.add_argument("--mdrqvae_ckpt", required=True)
    ap.add_argument("--centroid_cache", default="/scratch/yw8866/rec-tmall/mdrqvae_centroids.pt")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--head_layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--head_epochs", type=int, default=30)
    ap.add_argument("--head_patience", type=int, default=4)
    ap.add_argument("--ss_prob", type=float, default=0.0, help="scheduled sampling prob (default off)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/chain_rescore")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- STAGE A (pure analysis on the existing pool dump) ----
    stage_a(args.pool_cmp_dump, args.wb, fh)

    # ---- load split dumps + geometry ----
    log("\nloading split dumps (enc_mem required) ...", fh)
    tr = RerankData(args.train_dump, args.wb)
    va = RerankData(args.val_dump, args.wb)
    te = RerankData(args.test_dump, args.wb)
    assert_split(te, fh)
    d_model = tr.mem.shape[2]
    U_f, U_t, U_i = load_geometry(args.mdrqvae_ckpt, args.centroid_cache, d_model, args.seed, device)
    k_i = U_i.shape[0]
    log(f"  geometry: d_model={d_model}  K_f={U_f.shape[0]} K_t={U_t.shape[0]} K_i={k_i}", fh)

    # ---- STAGE B: train the chain head ----
    head = ChainHead(U_f, U_t, d_model, k_i, args.head_layers, args.heads, args.dropout).to(device)
    train_head(head, tr, va, device, args, fh)

    # ---- per-candidate logP(i|f,t) on val + test ----
    chain_va = head_scores(head, va, device, args.eval_batch)
    chain_te = head_scores(head, te, device, args.eval_batch)

    # ---- STAGE C: fit fusion on VAL (val is the eval split for weight selection) ----
    log("\n" + "=" * 88 + "\n=== STAGE C: learned score fusion (fit on the eval/VAL split) ===\n" + "=" * 88, fh)
    terms_va = five_terms(va, chain_va, args.wb)
    terms_te = five_terms(te, chain_te, args.wb)
    names = ["logP(b)", "logP(f)", "logP(t|f)", "logP(i|f)", "logP(i|f,t)"]
    w5 = fit_fusion(terms_va, va, device, [0, 1, 2, 3, 4], fh, seed=args.seed)
    w4 = fit_fusion(terms_va, va, device, [0, 1, 2, 3], fh, seed=args.seed)   # chain-ablated
    log("  fused weights (fit on VAL):", fh)
    for nm, wv in zip(names, w5.tolist()):
        log(f"      {nm:<12} {wv:+.4f}", fh)
    log(f"  chain-term weight = {w5[4].item():+.4f}  (|w| share = "
        f"{abs(w5[4].item())/(w5.abs().sum().item()+1e-9):.3f} of total)", fh)

    cov = te.has_gt.float().mean().item()

    def fused(terms, w, use):
        return (terms[..., use].to(device) * w).sum(-1).cpu()

    # ---- EVAL on TEST ----
    log("\n" + "=" * 88 + "\n=== EVALUATION on TEST split ===\n" + "=" * 88, fh)
    base_r, base_p = report("BASELINE hand-set (w_b=1, tau=1)  [4 terms]", te, te.base.float(), fh, cov)
    abl_r, abl_p = report("FUSION chain-ABLATED (learned, 4 terms)", te, fused(terms_te, w4, [0, 1, 2, 3]), fh, cov)
    chn_r, chn_p = report("FUSION + chain conditional (learned, 5 terms)", te, fused(terms_te, w5, [0, 1, 2, 3, 4]), fh, cov)

    # coverage invariance (rescoring cannot change which GTs are in the pool)
    cov_base = score_ranks(te.base.float(), te)[1].float().mean().item()
    cov_chn = score_ranks(fused(terms_te, w5, [0, 1, 2, 3, 4]), te)[1].float().mean().item()
    log(f"\n  coverage invariance: base={cov_base:.4f}  chain-fused={cov_chn:.4f}  "
        f"{'OK' if abs(cov_base-cov_chn)<1e-6 else '** DRIFT (bug) **'}  (has_gt={cov:.4f})", fh)
    if abs(cov - 0.2615) > 0.02:
        log(f"  [note] test coverage {cov:.4f} differs from the 0.2615 reference sample -- flagged, not tuned.", fh)

    log("\n=== SUMMARY ===", fh)
    log(f"  P(rank<10|cov):  baseline {base_p:.4f} -> chain-ablated {abl_p:.4f} -> +chain {chn_p:.4f}"
        f"   (target beat 0.7805, chain ref 0.8581)", fh)
    log(f"  t3@10:           baseline {base_r:.4f} -> chain-ablated {abl_r:.4f} -> +chain {chn_r:.4f}", fh)
    log(f"  chain-term learned weight = {w5[4].item():+.4f}. Marginal gain of the chain conditional "
        f"= {chn_p - abl_p:+.4f} on P(rank<10|cov), {chn_r - abl_r:+.4f} on t3@10.", fh)
    log("  If the chain-term weight is ~0 and +chain does not beat chain-ablated, the conditional carries", fh)
    log("  no signal on freely generated candidates -- report the negative (Stage A predicted it).", fh)
    fh.close()


if __name__ == "__main__":
    main()
