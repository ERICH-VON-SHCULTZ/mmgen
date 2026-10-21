"""Pool comparison: chain sequential decoding vs MFD factorized decoding, at a matched budget.

Reads two audit dumps with a shared schema (per-behavior pools pool_f/t/i + pool_sidscore + beh_logp
+ per-level teacher-forced ranks f_star_rank/tf_t_rank/tf_i_rank + gt_* + gt_in_catalog):
  - MFD  arm: mfd_audit_dump (K_f,K_t,K_i,K_pool)   [levels = shared f / text t / image i]
  - chain arm: chain_pool_audit (beam, K_pool)       [levels = l0 / l1 / l2, sequential]

Per arm it prints: budget accounting (matched at scored=K_pool), coverage@pool overall + per
behavior, the coverage-loss decomposition in pruning order, pool diversity, P(rank<10|cov), and
the oracle-headroom. Then the coverage-lever sweep (coverage@grid from the per-level ranks) for
both, so the two coverage surfaces are directly comparable.

HONESTY: the per-level ranks are TEACHER-FORCED (later levels condition on TRUE earlier codes),
which flatters chain relative to free generation. The ACTUAL free-gen number is coverage@pool
(the real beam pool); the rank-based decomposition/sweep is the generous ceiling. Both are shown.
Everything is per behavior, since t3 is ~93% pv and a conversion-only effect is invisible in aggregate.
"""

import argparse
import os

import torch


BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
MD_CENTROIDS = "/scratch/yw8866/rec-tmall/mdrqvae_centroids.pt"
REPR_CACHE = "/scratch/yw8866/rec-tmall/cand_repr_tables.pt"   # holds tiger_cb for chain reconstructions


def bar(t):
    print("\n" + "=" * 82 + f"\n{t}\n" + "=" * 82)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cols = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in d["cols"].items()}
    return cols, d["meta"]


def build_cand(c, meta):
    N, nb, Kp = c["pool_f"].shape
    flat = lambda x: x.reshape(N, nb * Kp)
    return {
        "branch": torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1).contiguous(),
        "f": flat(c["pool_f"]).long(), "t": flat(c["pool_t"]).long(), "i": flat(c["pool_i"]).long(),
        "sid": flat(c["pool_sidscore"]), "valid": flat(c["pool_sidscore"]) > -1e30,
    }, N, nb, Kp


def union_score(cand, c, wb):
    s = wb * torch.gather(c["beh_logp"], 1, cand["branch"]) + cand["sid"]
    return s.masked_fill(~cand["valid"], float("-inf"))


def best_rank(score, match, eligible):
    score = score.masked_fill(~eligible, float("-inf"))
    m = torch.where(match & eligible, score, torch.full_like(score, float("-inf")))
    best = m.max(1).values
    has = best > -1e30
    rank = (score > best.unsqueeze(1)).sum(1)
    return torch.where(has, rank, torch.full_like(rank, -1)), has


def recall_at(rank, has, k, denom=None):
    hit = has & (rank >= 0) & (rank < k)
    return (hit[denom] if denom is not None else hit).float().mean().item()


# ---------------------------------------------------------------------------
def per_behavior_rows(name, values_fn, gt_b, nb):
    """values_fn(mask)->float; returns aligned per-behavior + ALL row string."""
    cells = [f"{values_fn(torch.ones_like(gt_b, dtype=torch.bool)):>9.4f}"]
    for b in range(nb):
        cells.append(f"{values_fn(gt_b == b):>9.4f}")
    return f"  {name:<26}" + "".join(cells)


def in_pool_mask(cand, c):
    gt_f, gt_t, gt_i, gt_b = c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long(), c["gt_b"].long()
    sid = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    beh = cand["branch"] == gt_b[:, None]
    return (sid & beh & cand["valid"]).any(1)   # GT present in its behavior's pool (union sense)


def analyze_arm(tag, c, meta, wb, arm, centroids=None):
    bar(f"ARM: {tag}   [{arm}]")
    nb = meta["nb"]; Kp = meta["Kpool"]
    gt_b = c["gt_b"].long()
    N = gt_b.shape[0]
    cand, _, _, _ = build_cand(c, meta)
    hdr = "  {:<26}".format("metric") + "".join(f"{n:>9}" for n in ["ALL"] + [BEH[b] for b in range(nb)])

    # ---- budget accounting ----
    if arm == "MFD":
        Kf, Kt, Ki = meta["Kf"], meta["Kt"], meta["Ki"]
        composed = Kf * Kt * Ki
        fwd_states = "stageA nb*(1+f) + heads nb*Kf  (=%d)" % (nb * 2 + nb * Kf)
        beams = f"K_f={Kf} K_t={Kt} K_i={Ki}"
    else:
        beam = meta["beam"]; composed = meta["composed_l2"]
        fwd_states = str(meta["fwd_states_per_item"])
        beams = f"beam={beam} (l0/l1/l2)"
    surv = c["v_cnt"].mean().item() if "v_cnt" in c else float("nan")   # MFD valid comps/behavior
    surv2 = c["v2_cnt"].mean().item() if "v2_cnt" in c else surv        # chain valid l2-comps/behavior
    print(f"  codebook widths (k0,k1,k2) = ({meta['k0']},{meta['k1']},{meta['k2']})")
    print(f"  BUDGET: beams[{beams}]  composed(before catalog)={composed:,}  "
          f"survived-catalog/behavior≈{surv2:,.0f}  scored=K_pool={Kp}  "
          f"decoder-state-evals/item={fwd_states}")

    # ---- coverage@pool (actual scored pool), per behavior ----
    covered = in_pool_mask(cand, c)
    print("\n  " + hdr[2:])
    print(per_behavior_rows("coverage@pool", lambda m: covered[m].float().mean().item(), gt_b, nb))

    # ---- recall@10 + P(rank<10|cov) + headroom, per behavior ----
    score = union_score(cand, c, wb)
    sid = (cand["f"] == c["gt_f"].long()[:, None]) & (cand["t"] == c["gt_t"].long()[:, None]) & \
          (cand["i"] == c["gt_i"].long()[:, None])
    beh = cand["branch"] == gt_b[:, None]
    rank, has = best_rank(score, sid & beh, cand["valid"])
    r10 = lambda m: recall_at(rank, has, 10, m)
    print(per_behavior_rows("recall@10 (union,w_b=1)", r10, gt_b, nb))
    def p_r10_cov(m):
        mm = m & covered
        return recall_at(rank, has, 10, mm) if mm.any() else 0.0
    print(per_behavior_rows("P(rank<10 | covered)", p_r10_cov, gt_b, nb))
    print(per_behavior_rows("oracle headroom (cov-r@10)",
                            lambda m: covered[m].float().mean().item() - r10(m), gt_b, nb))

    # ---- coverage-loss decomposition (pruning order), per behavior ----
    fr, tr, ir = c["f_star_rank"], c["tf_t_rank"], c["tf_i_rank"]
    gt_valid = c["gt_in_catalog"].bool()
    if arm == "MFD":
        c0, c1, c2 = meta["Kf"], meta["Kt"], meta["Ki"]; names = ("f_pruned", "branch_t", "branch_i")
    else:
        beam = meta["beam"]; c0 = c1 = c2 = beam; names = ("level0_pruned", "level1_pruned", "level2_pruned")
    p0 = fr >= c0
    p1 = (~p0) & (tr >= c1)
    p2 = (~p0) & (tr < c1) & (ir >= c2)
    composed_ok = (~p0) & (tr < c1) & (ir < c2) & gt_valid
    trunc = composed_ok & (~covered)
    cat_invalid = ~gt_valid
    n_inv = int(cat_invalid.sum())
    print("\n  coverage-loss decomposition (mutually exclusive, pruning order):")
    print("  " + hdr[2:])
    for nm, mask in [(names[0], p0), (names[1], p1), (names[2], p2), ("trunc", trunc),
                     ("cat_invalid", cat_invalid)]:
        print(per_behavior_rows(nm, lambda m: mask[m].float().mean().item(), gt_b, nb))
    assert n_inv == 0, f"CATALOG INVALID = {n_inv} (>0): the constraint set is WRONG for {tag}"
    print(f"  [catalog unit test OK: GT-in-catalog for all {N:,} events]")

    # ---- pool diversity (union pool per event), per behavior ----
    print("\n  pool diversity (per-event union pool; mean over events):")
    print("  " + hdr[2:])
    f_all, t_all, i_all, val = cand["f"], cand["t"], cand["i"], cand["valid"]
    pre2 = f_all * meta["k1"] + t_all
    def uniq_l0(m):
        idx = torch.nonzero(m).squeeze(1)
        return float(sum(torch.unique(f_all[e][val[e]]).numel() for e in idx.tolist()) / max(idx.numel(), 1))
    def uniq_pre2(m):
        idx = torch.nonzero(m).squeeze(1)
        return float(sum(torch.unique(pre2[e][val[e]]).numel() for e in idx.tolist()) / max(idx.numel(), 1))
    def largest_l0grp(m):
        idx = torch.nonzero(m).squeeze(1)
        tot = 0
        for e in idx.tolist():
            v = f_all[e][val[e]]
            tot += int(torch.bincount(v).max()) if v.numel() else 0
        return tot / max(idx.numel(), 1)
    # subsample events for the O(K) loops (diversity is stable); cap 4000/behavior
    def sub(m):
        idx = torch.nonzero(m).squeeze(1)
        if idx.numel() > 4000:
            idx = idx[torch.randperm(idx.numel())[:4000]]
        mm = torch.zeros_like(m); mm[idx] = True
        return mm
    print(per_behavior_rows("unique l0 codes / pool", lambda m: uniq_l0(sub(m)), gt_b, nb))
    print(per_behavior_rows("unique 2-prefixes / pool", lambda m: uniq_pre2(sub(m)), gt_b, nb))
    print(per_behavior_rows("largest l0-prefix group", lambda m: largest_l0grp(sub(m)), gt_b, nb))
    if centroids is not None:
        cf, ct, ci = centroids
        def mean_pairdist(m):
            idx = torch.nonzero(m).squeeze(1)
            if idx.numel() > 800:
                idx = idx[torch.randperm(idx.numel())[:800]]
            acc = 0.0; n = 0
            for e in idx.tolist():
                vv = val[e]
                if vv.sum() < 2:
                    continue
                rep = cf[f_all[e][vv]] + ct[t_all[e][vv]] + ci[i_all[e][vv]]   # additive reconstruction
                rep = torch.nn.functional.normalize(rep, dim=-1)
                sim = rep @ rep.t()
                k = rep.shape[0]
                acc += (1 - sim).sum().item() / (k * (k - 1)); n += 1
            return acc / max(n, 1)
        print(per_behavior_rows("mean pairwise cos-dist", lambda m: mean_pairdist(m), gt_b, nb))
    else:
        print("  (mean pairwise cos-dist: SKIPPED — centroids unavailable)")
    return {"covered": covered, "rank": rank, "has": has, "gt_b": gt_b,
            "fr": fr, "tr": tr, "ir": ir, "gt_valid": gt_valid, "meta": meta}


def coverage_grid(tag, r, arm):
    bar(f"COVERAGE LEVER (grid coverage from teacher-forced per-level ranks) — {tag} [{arm}]")
    fr, tr, ir, gv, gt_b = r["fr"], r["tr"], r["ir"], r["gt_valid"], r["gt_b"]
    meta = r["meta"]; pv, buy = meta["pv_id"], meta["buy_id"]
    if arm == "MFD":
        L0 = [10, 20, 50, 100, 200, 10 ** 9]; L1 = [10, 50, 200, 1024]; L2 = [10, 50, 1024]
        n0, n1, n2 = "K_f", "K_t", "K_i"
    else:
        L0 = [10, 50, 200, 500, 10 ** 9]; L1 = [10, 50, 200, 500]; L2 = [10, 50, 200, 500]
        n0, n1, n2 = "B0", "B1", "B2"
    print("  NOTE: teacher-forced (later levels see TRUE earlier codes) -> GENEROUS upper bound vs")
    print("        free generation. The honest free-gen number is coverage@pool above.\n")
    print(f"  Holding {n0}=max, widen later levels ({n1} x {n2}); cov_ALL / cov_pv / cov_buy:")
    print(f"  {n1:>6}{n2:>6}{'cov_ALL':>10}{'cov_pv':>9}{'cov_buy':>9}")
    for b1 in L1:
        for b2 in L2:
            comp = (fr < L0[-1]) & (tr < b1) & (ir < b2) & gv
            ca = comp.float().mean().item()
            cp = comp[gt_b == pv].float().mean().item()
            cb = comp[gt_b == buy].float().mean().item()
            print(f"  {b1:>6}{b2:>6}{ca:>10.4f}{cp:>9.4f}{cb:>9.4f}")
    print(f"\n  Full {n0} x {n1} x {n2} surface (cov_ALL):")
    print(f"  {n0:>7}" + "".join(f"{n1}={b1}".rjust(11) for b1 in L1))
    for b0 in L0:
        row = f"  {('inf' if b0 > 1e8 else b0):>7}"
        for b1 in L1:
            comp = (fr < b0) & (tr < b1) & (ir < L2[-1]) & gv
            row += f"{comp.float().mean().item():>11.4f}"
        print(row)
    # saturation: where does cov stop climbing as later levels widen (l0 fixed at max)?
    base = ((fr < L0[-1]) & (tr < L1[0]) & (ir < L2[0]) & gv).float().mean().item()
    full = ((fr < L0[-1]) & (tr < L1[-1]) & (ir < L2[-1]) & gv).float().mean().item()
    print(f"\n  widen-later-levels lift (l0 unbounded): {base:.4f} -> {full:.4f}  (+{full - base:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_dump", required=True)
    ap.add_argument("--chain_dump", required=True)
    ap.add_argument("--chain_tag", default="chain RQ-VAE (tiger_1024)")
    ap.add_argument("--wb", type=float, default=1.0)
    args = ap.parse_args()

    # centroids for pairwise cosine (optional): MD for MFD arm, TIGER for chain arm
    md_cent = tg_cent = None
    if os.path.exists(MD_CENTROIDS):
        g = torch.load(MD_CENTROIDS, map_location="cpu", weights_only=False)
        md_cent = (g["c_f"].float(), g["c_t"].float(), g["c_i"].float())
    if os.path.exists(REPR_CACHE):
        rc = torch.load(REPR_CACHE, map_location="cpu", weights_only=False)
        if "tiger_cb" in rc:
            tg_cent = (rc["tiger_cb"][0].float(), rc["tiger_cb"][1].float(), rc["tiger_cb"][2].float())

    cm, mm = load(args.mfd_dump)
    cc, mc = load(args.chain_dump)

    # guard: only use centroids whose per-level widths cover that arm's codes (else skip pairwise-cos)
    def fits(cent, meta):
        return cent is not None and all(cent[l].shape[0] >= meta[f"k{l}"] for l in range(3))
    if not fits(md_cent, mm):
        md_cent = None
    if not fits(tg_cent, mc):
        if tg_cent is not None:
            print(f"  [note] TIGER centroid widths {[t.shape[0] for t in tg_cent]} do not cover chain "
                  f"k=({mc['k0']},{mc['k1']},{mc['k2']}); chain pairwise-cos skipped.", flush=True)
        tg_cent = None
    bar("POOL COMPARISON — chain sequential vs MFD factorized  (matched: scored = 500 per arm)")
    print(f"  MFD  dump: {args.mfd_dump}  events={cm['gt_b'].shape[0]:,}")
    print(f"  chain dump: {args.chain_dump}  events={cc['gt_b'].shape[0]:,}")
    for b in range(mm["nb"]):
        print(f"    {BEH.get(b, b):<5} MFD={int((cm['gt_b'].long()==b).sum()):>8,}  "
              f"chain={int((cc['gt_b'].long()==b).sum()):>8,}")

    r_mfd = analyze_arm("MFD + PMQ + Q_f  (MD-RQ-VAE)", cm, mm, args.wb, "MFD", md_cent)
    r_ch = analyze_arm(args.chain_tag, cc, mc, args.wb, "chain", tg_cent)
    coverage_grid("MFD + PMQ + Q_f  (MD-RQ-VAE)", r_mfd, "MFD")
    coverage_grid(args.chain_tag, r_ch, "chain")

    bar("READ-OUT")
    print("  Compare coverage@pool per behavior (esp. pv, where the loss concentrates) and the")
    print("  coverage-lever surfaces: does widening later levels convert branch loss to coverage")
    print("  under factorization, and does the chain surface SATURATE earlier? If chain matches on")
    print("  coverage/diversity/headroom, narrow the claim to whichever dimensions the data supports.")


if __name__ == "__main__":
    main()