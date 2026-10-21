"""TASK 1: promote the learned score fusion to the default scoring path.

The hand-set score is  w_b*logP(b) + logP(f) + (1/tau)*[logP(t|f)+logP(i|f)]  (w_b=1, tau=1).
Fitting the FOUR weights on the eval (VAL) split by listwise CE gives a better TEST ordering. This
script fits on VAL only (never test), PERSISTS the weights beside the checkpoint, and re-reports the
FULL metric set for hand-set vs learned, PAIRED, with honest per-behavior breakdowns and the
behavior-head precision/recall/F1. The learned fusion re-ranks the FIXED pool (selection unchanged),
so coverage@pool is invariant -- asserted.

The chain-conditional 5th term is dropped (its learned weight was +0.0764 / 2.1%, marginal ~0).

Secondary variant: a behavior-WEIGHTED fit (inverse-frequency event weights) to see whether the pv
gain can be kept without the conversion regression -- reported but NOT the primary number.
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import torch
import torch.nn.functional as F

from train_mfd_reranker import BEH, RerankData, best_rank, ndcg_at, recall_at

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}
TERM_NAMES = ["logP(b)", "logP(f)", "logP(t|f)", "logP(i|f)"]


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


def assert_split(data, fh):
    per = {b: int((data.gt_b == b).sum()) for b in range(data.nb)}
    ok = (data.N == EXPECT_TOTAL) and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    log(f"  TEST split: total={data.N:,} " + "  ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH)
        + f"  -> {'OK' if ok else 'MISMATCH'}", fh)
    if not ok:
        raise SystemExit("** TEST split mismatch -- refusing to emit numbers (45k-pv-subsample guard). **")


def terms4(data):
    """(N,C,4) = [logP(b), logP(f), logP(t|f), logP(i|f)] (RerankData.lp)."""
    return data.lp.float()


def fit_weights(data, device, fh, event_w=None, steps=500, lr=0.05, seed=0):
    torch.manual_seed(seed)
    idx = data.train_idx.to(device)
    T = terms4(data).to(device)
    valid = data.valid.to(device); gt_idx = data.gt_idx.to(device)
    w = torch.ones(4, device=device, requires_grad=True)          # init at the hand-set all-ones
    ew = None if event_w is None else event_w.to(device)[idx]
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        sc = (T[idx] * w).sum(-1).masked_fill(~valid[idx], float("-inf"))
        loss = F.cross_entropy(sc, gt_idx[idx], reduction="none")
        loss = (loss * ew).sum() / ew.sum() if ew is not None else loss.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach()


def scored(data, w, device):
    return (terms4(data).to(device) * w.to(device)).sum(-1).cpu()


def ranks(score, data):
    sid = ((data.f == data.gt_f[:, None]) & (data.t == data.gt_t[:, None]) & (data.i == data.gt_i[:, None]))
    beh = data.branch.view(1, -1) == data.gt_b[:, None]
    r3, h3 = best_rank(score, sid & beh, data.valid)             # t3 union
    r2, h2 = best_rank(score, sid, beh & data.valid)            # t2 routed
    return r3, h3, r2, h2


def behavior_head(beh_logp, gt_b, nb):
    pred = beh_logp.argmax(1)
    acc = (pred == gt_b).float().mean().item()
    rows = {}
    for b in range(nb):
        tp = int(((pred == b) & (gt_b == b)).sum()); fp = int(((pred == b) & (gt_b != b)).sum())
        fn = int(((pred != b) & (gt_b == b)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows[b] = (p, r, f1)
    return acc, rows


def full_report(tag, data, score, fh, beh_logp=None):
    r3, h3, r2, h2 = ranks(score, data)
    gt_b = data.gt_b; cov = data.has_gt
    log(f"\n  [{tag}]", fh)
    log(f"    t3: R@5={recall_at(r3,h3,5):.4f} R@10={recall_at(r3,h3,10):.4f} R@20={recall_at(r3,h3,20):.4f} "
        f"N@5={ndcg_at(r3,h3,5):.4f} N@10={ndcg_at(r3,h3,10):.4f}", fh)
    log(f"    t2: R@5={recall_at(r2,h2,5):.4f} R@10={recall_at(r2,h2,10):.4f} R@20={recall_at(r2,h2,20):.4f} "
        f"N@5={ndcg_at(r2,h2,5):.4f} N@10={ndcg_at(r2,h2,10):.4f}", fh)
    log(f"    P(rank<10|cov)={recall_at(r3,h3,10,cov):.4f}  coverage@pool={cov.float().mean().item():.4f}", fh)
    log(f"    t3@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r3,h3,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    log(f"    t2@10 per-beh: " + "  ".join(f"{BEH[b]}={recall_at(r2,h2,10,gt_b==b):.4f}" for b in range(data.nb) if (gt_b==b).any()), fh)
    if beh_logp is not None:
        acc, rows = behavior_head(beh_logp, gt_b, data.nb)
        log(f"    behavior head: accuracy={acc:.4f}   " + "  ".join(
            f"{BEH[b]}(P/R/F1={rows[b][0]:.3f}/{rows[b][1]:.3f}/{rows[b][2]:.3f})" for b in range(data.nb)), fh)
    return {"t3@10": recall_at(r3, h3, 10), "pcov": recall_at(r3, h3, 10, cov),
            "perbeh": {b: recall_at(r3, h3, 10, gt_b == b) for b in range(data.nb)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dump", required=True)
    ap.add_argument("--test_dump", required=True)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--weights_out", default="/scratch/yw8866/rec-tmall/mfd_fusion_weights.pt")
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/promote_fusion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")

    va = RerankData(args.val_dump, args.wb)
    te = RerankData(args.test_dump, args.wb)
    assert_split(te, fh)
    beh_logp_te = torch.load(args.test_dump, map_location="cpu", weights_only=False)["cols"]["beh_logp"].float()

    # ---- fit on VAL (plain, primary) ----
    w_plain = fit_weights(va, device, fh, event_w=None, seed=args.seed)
    log("\n=== FITTED WEIGHTS (VAL, plain listwise CE -- PRIMARY) ===", fh)
    for nm, wv in zip(TERM_NAMES, w_plain.tolist()):
        log(f"    {nm:<12} {wv:+.4f}", fh)

    # ---- secondary: behavior-weighted fit (inverse-frequency event weights) ----
    freq = torch.tensor([(va.gt_b == b).float().mean().item() for b in range(va.nb)]).clamp(min=1e-6)
    ev_w = (1.0 / freq)[va.gt_b]
    w_bw = fit_weights(va, device, fh, event_w=ev_w, seed=args.seed)
    log("=== FITTED WEIGHTS (VAL, behavior-weighted -- SECONDARY, not the headline) ===", fh)
    for nm, wv in zip(TERM_NAMES, w_bw.tolist()):
        log(f"    {nm:<12} {wv:+.4f}", fh)

    # ---- persist the PRIMARY weights beside the checkpoint (downstream runs load these) ----
    torch.save({"weights": w_plain.cpu(), "term_names": TERM_NAMES, "wb": args.wb,
                "fit": "val_plain_listwise_ce", "behavior_weighted": w_bw.cpu()}, args.weights_out)
    log(f"\n  persisted fusion weights -> {args.weights_out}", fh)

    # ---- TEST re-report, paired ----
    log("\n" + "=" * 80 + "\n=== TEST re-report (hand-set vs learned) ===\n" + "=" * 80, fh)
    r_hand = full_report("HAND-SET  w_b=1, tau=1 (baseline)", te, te.base.float(), fh, beh_logp_te)
    r_learn = full_report("LEARNED FUSION (plain fit, PRIMARY)", te, scored(te, w_plain, device), fh, beh_logp_te)
    r_bw = full_report("LEARNED FUSION (behavior-weighted, secondary)", te, scored(te, w_bw, device), fh, beh_logp_te)

    # coverage invariance
    cov_hand = ranks(te.base.float(), te)[1].float().mean().item()
    cov_learn = ranks(scored(te, w_plain, device), te)[1].float().mean().item()
    log(f"\n  coverage invariance: hand={cov_hand:.4f} learned={cov_learn:.4f} "
        f"{'OK' if abs(cov_hand-cov_learn)<1e-6 else '** DRIFT (bug) **'}", fh)
    assert abs(cov_hand - cov_learn) < 1e-6, "coverage drifted under rescoring -- BUG"

    log("\n=== VERDICT (Task 1) ===", fh)
    log(f"  t3@10 {r_hand['t3@10']:.4f} -> {r_learn['t3@10']:.4f}   P(rank<10|cov) {r_hand['pcov']:.4f} -> {r_learn['pcov']:.4f}", fh)
    log("  per-behavior t3@10 (hand -> learned -> beh-weighted):", fh)
    for b in range(te.nb):
        log(f"    {BEH[b]:<5} {r_hand['perbeh'][b]:.4f} -> {r_learn['perbeh'][b]:.4f} -> {r_bw['perbeh'][b]:.4f}", fh)
    log("  Honest read: the plain fit is a REDISTRIBUTION (pv up, conversions down); it wins in", fh)
    log("  aggregate only because pv is ~93% of events. Whether the behavior-weighted variant removes", fh)
    log("  the conversion regression without losing the pv gain is shown above -- the PRIMARY number is", fh)
    log("  the plain fit; the weighted one is reported, not selected as the headline.", fh)
    fh.close()


if __name__ == "__main__":
    main()
