"""TASK 2: attribute the in-pool ORDERING gap between MFD and chain.

MFD orders its pool worse than chain: P(rank<10|covered) 0.7806 vs 0.8581. The chain conditional
logP(i|f,t) has been ruled out (no signal on freely generated candidates). Leading hypothesis: the
gap is the PRICE OF DIVERSITY -- MFD's pool has ~327 unique 2-prefixes vs chain's ~143 at matched
compute, so more plausible competitors crowd the GT. Competing hypothesis: it's a pool-SIZE artefact.

On covered events (GT in pool), per arm and per behavior, using each arm's own union score
(w_b*logP(b) + pool_sidscore):
  - score margin GT vs top non-GT, and GT vs the rank-10 candidate (full quantile distribution)
  - crowding: #candidates within a +/- band around the GT score, band swept
  - within-pool score spread (std) and consecutive-rank gaps near the cutoff at rank 10
  - rank-band fractions: covered GT at 0-9 / 10-49 / 50+  (recoverable band made explicit)
  - SigLIP2 similarity between the GT and the candidates that OUTRANK it (shared item space), so
    both arms are comparable -- near-misses vs unrelated displacers
  - DIVERSITY-cap vs SIZE-cap subsampling sweeps -> P(rank<10|cov): if ordering recovers toward
    chain when DIVERSITY is matched (not merely when size shrinks), the gap is the cost of diversity.

Sweeps are reported in full; nothing is tuned to favour either hypothesis.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}


# ------------------------------------------------------------------- feature IO (raw float32 dumps)
def load_id_map(pkl):
    with open(pkl, "rb") as f:
        m = pickle.load(f)
    if isinstance(m, dict):
        return {int(k): int(v) for k, v in m.items()}
    return {int(iid): r for r, iid in enumerate(m)}


def open_embeddings(path, n_rows):
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    total = os.path.getsize(path) // 4
    dim = total // n_rows
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n_rows, dim))


# ------------------------------------------------------------------------------- per-arm loading
def load_arm(label, dump_path, wb, fh):
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    nb = meta["nb"]; N, _, Kp = c["pool_f"].shape
    f32 = lambda x: x.float() if x.dtype == torch.float16 else x
    flat = lambda x: x.reshape(N, nb * Kp)
    pf, pt, pi = flat(c["pool_f"]).long(), flat(c["pool_t"]).long(), flat(c["pool_i"]).long()
    ssc = f32(flat(c["pool_sidscore"]))
    valid = ssc > -1e30
    branch = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1)
    gt_b, gt_f, gt_t, gt_i = c["gt_b"].long(), c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    lp_b = torch.gather(f32(c["beh_logp"]), 1, branch)
    score = (wb * lp_b + ssc).masked_fill(~valid, float("-inf"))          # arm's own union score
    gt_is = ((branch == gt_b[:, None]) & (pf == gt_f[:, None]) & (pt == gt_t[:, None])
             & (pi == gt_i[:, None]) & valid)
    covered = gt_is.any(1)
    gt_score = torch.where(gt_is, score, torch.full_like(score, float("-inf"))).max(1).values
    rank = (score > gt_score.unsqueeze(1)).sum(1)                         # 0-based rank of GT
    fw = int(pt.max()) + 1
    pre2 = (pf * fw + pt)                                                 # 2-prefix key (f,t)
    log(f"  [{label}] N={N:,} nb={nb} Kp={Kp} covered={int(covered.sum()):,} "
        f"({covered.float().mean():.4f})  meta:{ {k:meta[k] for k in ('beam','Kf','Kt','Ki','fwd_states_per_item') if k in meta} }", fh)
    return {"label": label, "N": N, "nb": nb, "Kp": Kp, "score": score, "valid": valid,
            "branch": branch, "gt_b": gt_b, "pf": pf, "pt": pt, "pi": pi, "gt_is": gt_is,
            "covered": covered, "gt_score": gt_score, "rank": rank, "pre2": pre2, "meta": meta,
            "gt_f": gt_f, "gt_t": gt_t, "gt_i": gt_i}


def q(x, qs=(0, 10, 25, 50, 75, 90, 100)):
    x = np.asarray(x, float)
    return {p: float(np.percentile(x, p)) for p in qs} if len(x) else {p: float("nan") for p in qs}


# --------------------------------------------------------------------------- margins / crowding
def margins(arm, fh, bands=(0.5, 1.0, 2.0, 4.0)):
    cov = arm["covered"]
    score = arm["score"]; gt_is = arm["gt_is"]; gts = arm["gt_score"]
    non_gt = torch.where(gt_is, torch.full_like(score, float("-inf")), score)
    top_non_gt = non_gt.max(1).values
    m_top = (gts - top_non_gt)[cov].numpy()                              # GT - best non-GT
    srt = score.sort(1, descending=True).values
    r10 = srt[:, 9]                                                       # 10th-ranked score
    m_r10 = (gts - r10)[cov].numpy()
    log(f"\n  [{arm['label']}] score margins on covered events (quantiles 0/10/25/50/75/90/100):", fh)
    log(f"    GT - best non-GT : {q(m_top)}", fh)
    log(f"    GT - rank10 score: {q(m_r10)}", fh)
    # within-pool score std over VALID candidates (score has -inf at invalid slots -> mask first).
    v = arm["valid"]
    cnt = v.sum(1).float()
    sc0 = torch.where(v, score, torch.zeros_like(score))
    mean = sc0.sum(1) / cnt.clamp(min=1)
    var = (((sc0 - mean.unsqueeze(1)) ** 2) * v).sum(1) / (cnt - 1).clamp(min=1)
    std = var.sqrt()[cov].numpy()
    gap10 = (srt[:, 9] - srt[:, 10])[cov].numpy()                        # gap between rank10 and 11
    log(f"    within-pool score std: mean={np.nanmean(std):.3f}   rank10->11 gap: mean={np.mean(gap10):.4f}  {q(gap10)}", fh)
    log("  crowding -- mean #candidates within +/- band of the GT score (band sweep):", fh)
    for band in bands:
        cnt = ((score >= (gts.unsqueeze(1) - band)) & (score <= (gts.unsqueeze(1) + band)) & arm["valid"]).sum(1)
        log(f"    band=+/-{band:<4}: mean #in-band = {cnt[cov].float().mean().item():.2f}", fh)


def rank_bands(arm, fh):
    cov = arm["covered"]; rk = arm["rank"][cov]
    n = rk.numel()
    b09 = int((rk < 10).sum()); b1049 = int(((rk >= 10) & (rk < 50)).sum()); b50 = int((rk >= 50).sum())
    log(f"\n  [{arm['label']}] covered-GT rank bands: <10={b09/n:.4f}  10-49={b1049/n:.4f}  50+={b50/n:.4f}  "
        f"(recoverable 10-49 band = {b1049:,} events)", fh)


def per_behavior(arm, fh):
    log(f"\n  [{arm['label']}] per-behavior P(rank<10|cov) and mean unique-2prefix:", fh)
    for b in range(arm["nb"]):
        m = arm["covered"] & (arm["gt_b"] == b)
        if m.sum() == 0:
            continue
        pcov = (arm["rank"][m] < 10).float().mean().item()
        ups = [torch.unique(arm["pre2"][e][arm["valid"][e]]).numel() for e in torch.nonzero(m).squeeze(1)[:4000].tolist()]
        up = float(np.mean(ups)) if ups else float("nan")
        log(f"    {BEH[b]:<5} P(rank<10|cov)={pcov:.4f}  mean uniq-2prefix={up:.1f}  (n={int(m.sum()):,})", fh)


# ------------------------------------------------------------------- diversity / size sweeps
def sweeps(arm, fh, size_caps=(100000, 1000, 500, 300, 200, 143, 100, 50),
           div_caps=(100000, 400, 327, 250, 200, 143, 100, 50)):
    """Recompute P(rank<10|cov), always keeping the GT. SIZE cap = top-N by score. DIVERSITY cap =
    keep candidates only from the first D distinct 2-prefixes seen in score order. Both are computed
    from two per-event vectors precomputed ONCE: the GT's position in the score order, and each
    candidate's prefix-first-rank -- so all caps are O(1) thereafter (no per-candidate Python loop)."""
    cov_idx = torch.nonzero(arm["covered"]).squeeze(1).tolist()
    score, valid, pre2, gt_is = arm["score"], arm["valid"], arm["pre2"], arm["gt_is"]
    n = len(cov_idx)
    # accumulators: hit counts + (size,uniq) sums per cap
    sz_hit = {N: 0 for N in size_caps}; sz_sz = {N: 0.0 for N in size_caps}; sz_up = {N: 0.0 for N in size_caps}
    dv_hit = {D: 0 for D in div_caps}; dv_sz = {D: 0.0 for D in div_caps}; dv_up = {D: 0.0 for D in div_caps}
    for e in cov_idx:
        sc = score[e]; v = valid[e]
        order = torch.argsort(sc, descending=True)
        order = order[v[order]]                                          # valid candidate ids, score desc
        po = pre2[e][order].numpy()
        gt_pos_row = torch.nonzero(gt_is[e]).squeeze(1)[0].item()
        gt_pos = int((order == gt_pos_row).nonzero()[0, 0])              # GT's index within the score order
        # prefix-first-rank per candidate (vectorized): group id -> first-occurrence pos -> rank
        _, inv = np.unique(po, return_inverse=True)
        first_pos = np.full(inv.max() + 1, len(po), dtype=np.int64)
        np.minimum.at(first_pos, inv, np.arange(len(po)))
        grank = first_pos.argsort().argsort() + 1                       # 1-based rank of each group by first occ
        po_rank = grank[inv]                                            # (len order,) prefix-first-rank per cand
        ngroups = len(first_pos)
        for N in size_caps:                                            # top-N: GT kept -> rank = #kept above GT
            sz_hit[N] += (min(gt_pos, N) < 10)
            sz_sz[N] += min(len(po), N)
            sz_up[N] += int((first_pos < N).sum())
        for D in div_caps:                                            # keep prefixes ranked <= D (+GT)
            kept = po_rank <= D
            above = int(kept[:gt_pos].sum())                          # kept candidates scoring above GT
            dv_hit[D] += (above < 10)
            dv_sz[D] += int(kept.sum()) + (0 if po_rank[gt_pos] <= D else 1)
            dv_up[D] += min(D, ngroups)
    log(f"\n  [{arm['label']}] SIZE-cap sweep (keep top-N by score; GT always kept):", fh)
    log(f"    {'N':>6}{'P(rank<10|cov)':>17}{'mean size':>11}{'mean uniq-2pfx':>16}", fh)
    for N in size_caps:
        log(f"    {N:>6}{sz_hit[N]/n:>17.4f}{sz_sz[N]/n:>11.1f}{sz_up[N]/n:>16.1f}", fh)
    log(f"  [{arm['label']}] DIVERSITY-cap sweep (keep candidates from the first D distinct 2-prefixes, "
        f"score order; GT always kept):", fh)
    log(f"    {'D':>6}{'P(rank<10|cov)':>17}{'mean size':>11}{'mean uniq-2pfx':>16}", fh)
    for D in div_caps:
        log(f"    {D:>6}{dv_hit[D]/n:>17.4f}{dv_sz[D]/n:>11.1f}{dv_up[D]/n:>16.1f}", fh)


# ----------------------------------------------------------------- SigLIP near-miss similarity
def build_tuple_to_item(sid_path):
    o = torch.load(sid_path, map_location="cpu", weights_only=False)
    t = o if torch.is_tensor(o) else next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2)
    if t.shape[0] > t.shape[1]:
        t = t.t().contiguous()
    l0, l1, l2 = t[0].long(), t[1].long(), t[2].long()
    w1 = int(l1.max()) + 1; w2 = int(l2.max()) + 1
    key = (l0 * w1 + l1) * w2 + l2                                        # packed tuple key -> item_id (column)
    keyn = key.numpy().astype(np.int64)
    order = np.argsort(keyn, kind="stable")
    return keyn[order], order.astype(np.int64), w1, w2                   # sorted keys, item_ids, widths


def siglip_similarity(arm, sid_path, tf_mm, id2row_t, if_mm, id2row_i, fh, sample=4000, seed=0):
    keys, items, w1, w2 = build_tuple_to_item(sid_path)
    def feat(item_ids):
        rt = np.array([id2row_t.get(int(i), -1) for i in item_ids])
        ri = np.array([id2row_i.get(int(i), -1) for i in item_ids])
        ok = (rt >= 0) & (ri >= 0)
        F = np.full((len(item_ids), tf_mm.shape[1] + if_mm.shape[1]), np.nan, np.float32)
        if ok.any():
            tt = tf_mm[rt[ok]]; ii = if_mm[ri[ok]]
            tt = tt / (np.linalg.norm(tt, axis=1, keepdims=True) + 1e-9)
            ii = ii / (np.linalg.norm(ii, axis=1, keepdims=True) + 1e-9)
            F[ok] = np.concatenate([tt, ii], axis=1)
        return F
    def lookup(f, t, i):
        k = int((int(f) * w1 + int(t)) * w2 + int(i))
        p = np.searchsorted(keys, k)
        return int(items[p]) if p < len(keys) and keys[p] == k else -1

    rng = np.random.default_rng(seed)
    cov_idx = torch.nonzero(arm["covered"]).squeeze(1).numpy()
    if len(cov_idx) > sample:
        cov_idx = rng.choice(cov_idx, sample, replace=False)
    sims, n_disp, n_mapped = [], 0, 0
    for e in cov_idx.tolist():
        rk = int(arm["rank"][e])
        if rk == 0:
            continue
        sc = arm["score"][e]; v = arm["valid"][e]
        gtp = torch.nonzero(arm["gt_is"][e]).squeeze(1)
        gt_item = lookup(arm["gt_f"][e], arm["gt_t"][e], arm["gt_i"][e])
        if gt_item < 0:
            continue
        gtf = feat([gt_item])[0]
        if np.isnan(gtf).any():
            continue
        # candidates that outrank the GT
        outr = torch.nonzero((sc > sc[gtp].max()) & v).squeeze(1).tolist()[:20]
        its = [lookup(arm["pf"][e][j], arm["pt"][e][j], arm["pi"][e][j]) for j in outr]
        its = [x for x in its if x >= 0]
        if not its:
            continue
        FF = feat(its); good = ~np.isnan(FF).any(1)
        if not good.any():
            continue
        s = FF[good] @ gtf
        sims.extend(s.tolist()); n_disp += len(s); n_mapped += 1
    log(f"\n  [{arm['label']}] SigLIP2 similarity of GT vs its displacers (candidates outranking GT):", fh)
    if sims:
        log(f"    events used={n_mapped:,}  displacer pairs={n_disp:,}  cosine mean={np.mean(sims):.4f}  {q(sims)}", fh)
        log(f"    (higher = displacers are semantic NEAR-MISSES; low = unrelated items crowding the GT)", fh)
    else:
        log("    (no mappable displacer pairs -- tuple->item lookup missed; SigLIP check skipped)", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_dump", required=True); ap.add_argument("--chain_dump", required=True)
    ap.add_argument("--mfd_sid", required=True); ap.add_argument("--chain_sid", required=True)
    ap.add_argument("--text_npy", required=True); ap.add_argument("--text_ids", required=True)
    ap.add_argument("--image_npy", required=True); ap.add_argument("--image_ids", required=True)
    ap.add_argument("--extra_arm", nargs=3, action="append", default=[], metavar=("LABEL", "DUMP", "SID"),
                    help="additional matched-compute arms: LABEL DUMP SID")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--siglip_sample", type=int, default=4000)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/ordering_gap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")

    log("=" * 88 + "\n=== TASK 2: attribute the in-pool ordering gap (MFD vs chain) ===\n" + "=" * 88, fh)
    arms = [("MFD", args.mfd_dump, args.mfd_sid), ("chain", args.chain_dump, args.chain_sid)] + \
           [(l, d, s) for (l, d, s) in args.extra_arm]

    id2row_t = load_id_map(args.text_ids); id2row_i = load_id_map(args.image_ids)
    tf_mm = open_embeddings(args.text_npy, len(id2row_t)); if_mm = open_embeddings(args.image_npy, len(id2row_i))

    loaded = []
    for label, dump, sid in arms:
        arm = load_arm(label, dump, args.wb, fh)
        margins(arm, fh); rank_bands(arm, fh); per_behavior(arm, fh)
        siglip_similarity(arm, sid, tf_mm, id2row_t, if_mm, id2row_i, fh, args.siglip_sample, args.seed)
        sweeps(arm, fh)
        loaded.append(arm)

    log("\n" + "=" * 88 + "\n=== VERDICT ===\n" + "=" * 88, fh)
    log("  Read the DIVERSITY-cap sweep against the SIZE-cap sweep: if MFD's P(rank<10|cov) climbs", fh)
    log("  toward chain's 0.8581 as the 2-prefix diversity D is reduced to chain's ~143 -- while the", fh)
    log("  size-cap sweep at the SAME resulting pool size does NOT recover as much -- the gap is the", fh)
    log("  cost of diversity, not a defect. If neither recovers, the scoring is genuinely weaker on", fh)
    log("  MFD's candidates and the cause is still open. Also weigh the SigLIP similarity of displacers", fh)
    log("  (near-misses vs unrelated) and the crowding sweep. Reported as swept; not tuned.", fh)
    fh.close()


if __name__ == "__main__":
    main()
