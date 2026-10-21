"""Three offline audits of an MFD free-generation pass (reads mfd_audit_dump.pt).

The dump is produced by one instrumented test pass (mfd_audit_dump=true) over a frozen MFD
checkpoint: per (stratified) event it stores the behavior head log-probs, the per-behavior
candidate pools (with the SID_score EXCLUDING the behavior term, so w_b can be swept here),
the teacher-forced GT-f branch reads, and the full ranks of f*/t*/i*. Nothing is recomputed
on the GPU — these three analyses are pure CPU tensor ops.

  0a  Evaluator pool routing : is the t1 collapse a scoring artifact (union pool) or real
                               cross-branch crowding? Re-rank union vs routed vs committed.
  0b  Contention test        : does sharing one h_f across the text+image heads cost per-level
                               accuracy beyond §12g's ~1.7% bound? (needs --chain_dump for the gap)
  0c  Coverage vs ordering   : is GT missing from the pool (widen K) or present-but-outranked
                               (rank/PMQ)? Coverage decomposition + in-pool rank + oracle headroom.

Usage:
  python analyze_mfd_audit.py --dump mfd_audit_dump.pt                 # 0a (run w_b), 0b(MFD), 0c
  python analyze_mfd_audit.py --dump ... --wb 1 2 4                    # 0a over several w_b
  python analyze_mfd_audit.py --dump ... --chain_dump chain_tf_audit.pt  # completes 0b's gap
"""

import argparse

import torch


BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}


def bar(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cols = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in d["cols"].items()}
    return cols, d["meta"]


# ---------------------------------------------------------------------------
# candidate table (flatten per-behavior pools) + ranking primitives
# ---------------------------------------------------------------------------
def build_candidates(c, meta):
    N, nb, Kp = c["pool_f"].shape
    flat = lambda x: x.reshape(N, nb * Kp)
    cand = {
        "branch": torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1).contiguous(),
        "f": flat(c["pool_f"]).long(),
        "t": flat(c["pool_t"]).long(),
        "i": flat(c["pool_i"]).long(),
        "sid": flat(c["pool_sidscore"]),               # SID_score sans behavior term
        "valid": flat(c["pool_sidscore"]) > -1e30,     # real (non-padding, non-filtered) slot
    }
    return cand, N, nb, Kp


def union_score(cand, c, wb):
    # w_b*logP(b_branch) + SID_score(.|branch)
    beh = c["beh_logp"]                                 # (N, nb)
    beh_of_cand = torch.gather(beh, 1, cand["branch"])  # (N, C)
    s = wb * beh_of_cand + cand["sid"]
    return s.masked_fill(~cand["valid"], float("-inf"))


def best_rank(score, match, eligible):
    """Per event: 0-based rank of the best matching+eligible candidate among eligible ones,
    or -1 if none. rank = #eligible candidates scoring strictly higher than that match."""
    score = score.masked_fill(~eligible, float("-inf"))
    m_score = torch.where(match & eligible, score, torch.full_like(score, float("-inf")))
    best = m_score.max(dim=1).values                    # (N,)
    has = best > -1e30
    rank = (score > best.unsqueeze(1)).sum(1)           # (N,) count strictly greater
    rank = torch.where(has, rank, torch.full_like(rank, -1))
    return rank, has


def recall_at(rank, has, k, denom_mask=None):
    hit = has & (rank >= 0) & (rank < k)
    if denom_mask is not None:
        return hit[denom_mask].float().mean().item()
    return hit.float().mean().item()


# ---------------------------------------------------------------------------
# 0a — evaluator pool routing
# ---------------------------------------------------------------------------
def audit_0a(c, meta, cand, N, wb, committed):
    bar(f"0a  POOL ROUTING  (w_b={wb})  — union vs routed vs committed")
    buy_id = meta["buy_id"]
    gt_b, gt_f, gt_t, gt_i = c["gt_b"].long(), c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    score = union_score(cand, c, wb)
    sid_match = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    beh_match = cand["branch"] == gt_b[:, None]
    all_elig = cand["valid"]
    branch_elig = cand["valid"] & beh_match
    buy_mask = gt_b == buy_id

    def line(name, rank, has, denom=None):
        r5, r10 = recall_at(rank, has, 5, denom), recall_at(rank, has, 10, denom)
        print(f"  {name:<34}{r5:>10.4f}{r10:>10.4f}")

    print(f"  {'scoring / track':<34}{'recall@5':>10}{'recall@10':>10}")
    print("  -- t1 (buy events, SID-only) --")
    r, h = best_rank(score, sid_match, all_elig);    line("t1 union", r, h, buy_mask)
    r, h = best_rank(score, sid_match, branch_elig); line("t1 routed (buy branch)", r, h, buy_mask)
    print(f"  {'t1 committed (cited)':<34}{committed['t1_5']:>10.4f}{committed['t1_10']:>10.4f}")
    print("  -- t2 (all events, SID-only, beh-conditioned) --")
    r, h = best_rank(score, sid_match, all_elig);    line("t2 union", r, h)
    r, h = best_rank(score, sid_match, branch_elig); line("t2 routed (GT-beh branch)", r, h)
    print(f"  {'t2 committed (cited)':<34}{'-':>10}{committed['t2_10']:>10.4f}")
    print("  -- t3 (free-gen joint = behavior+SID) --")
    r3, h3 = best_rank(score, sid_match & beh_match, all_elig); line("t3 union (marginalized)", r3, h3)
    print(f"  {'t3 committed (cited)':<34}{'-':>10}{committed['t3_10']:>10.4f}")

    # decision
    t1u10 = recall_at(*best_rank(score, sid_match, all_elig), 10, buy_mask)
    t1r10 = recall_at(*best_rank(score, sid_match, branch_elig), 10, buy_mask)
    t1r5 = recall_at(*best_rank(score, sid_match, branch_elig), 5, buy_mask)
    print("\n  DECISION:")
    if t1r10 >= committed["t1_10"] - 0.01 and t1r5 >= committed["t1_5"] - 0.02:
        print(f"    ROUTED RECOVERS COMMITTED (t1 routed @10={t1r10:.3f}≈{committed['t1_10']:.3f}, "
              f"@5={t1r5:.3f}≈{committed['t1_5']:.3f}) -> the t1 'collapse' is a scoring ARTIFACT of")
        print("    ranking behavior-conditioned tracks against the 4-way union pool. Standing readout:")
        print("    t3 from union/marginalized, t1/t2 from ROUTED. Update the eval path + §11.")
    else:
        print(f"    ROUTED STAYS DEPRESSED (t1 routed @10={t1r10:.3f} < committed {committed['t1_10']:.3f})")
        print("    -> REAL cross-branch crowding: per-branch scores aren't comparable across behaviors.")
        print("    Escalate to score normalization (per-branch standardization); use 0c ranks to localize.")
    return {"t3_union_10": recall_at(r3, h3, 10)}


# ---------------------------------------------------------------------------
# 0b — contention test
# ---------------------------------------------------------------------------
def _acc(rank, tops=(1, 5)):
    return {k: (rank == 0).float().mean().item() if k == 1 else (rank < k).float().mean().item() for k in tops}


def audit_0b(c, meta, chain):
    bar("0b  CONTENTION  — teacher-forced per-level accuracy at GT f (shared h_f)")
    mt, mi = c["tf_t_rank"], c["tf_i_rank"]
    at, ai = _acc(mt), _acc(mi)
    print(f"  MFD (text & image heads share h_f):")
    print(f"    acc(t | GT f)         top1={at[1]:.4f}  top5={at[5]:.4f}")
    print(f"    acc(i | GT f)         top1={ai[1]:.4f}  top5={ai[5]:.4f}")
    if chain is None:
        print("\n  (no --chain_dump given: MFD absolutes only. Run the chain config with")
        print("   +model.audit_tf_dump=true +model.audit_tf_path=chain_tf_audit.pt to get the gap.)")
        print("  Reminder for the decision: text has NO dropped edge (chain also predicts t|f),")
        print("  so any MFD text deficit vs chain is PURE contention; image may legitimately trail")
        print("  chain's i|f,t by up to §12g's ~1.7% of H(i|f).")
        return
    cc = chain[0]
    ct, ci = _acc(cc["tf_t_rank"]), _acc(cc["tf_i_rank"])
    print(f"  Chain (text head owns h_f; image head owns h_t):")
    print(f"    acc(t | GT f)         top1={ct[1]:.4f}  top5={ct[5]:.4f}")
    print(f"    acc(i | GT f, GT t)   top1={ci[1]:.4f}  top5={ci[5]:.4f}")
    tgap1, tgap5 = ct[1] - at[1], ct[5] - at[5]
    igap1, igap5 = ci[1] - ai[1], ci[5] - ai[5]
    print(f"\n  gaps (chain - MFD):  text top1={tgap1:+.4f} top5={tgap5:+.4f}   "
          f"image top1={igap1:+.4f} top5={igap5:+.4f}")
    print("\n  DECISION:")
    if tgap1 > 0.005 or tgap5 > 0.005:
        print(f"    TEXT deficit {max(tgap1, tgap5):+.4f} > 0.5pt with NO information excuse")
        print("    -> CONTENTION CONFIRMED, PMQ (per-modality query / relieve shared h_f) is licensed.")
        print(f"    Image gap {max(igap1, igap5):+.4f}: subtract the text (contention) gap to isolate the")
        print("    legitimate §12g ~1.7% component.")
    else:
        print(f"    TEXT accuracy matches chain within noise ({tgap1:+.4f}/{tgap5:+.4f})")
        print("    -> NO contention; the t1 drop is elsewhere (0a artifact / scoring). PMQ is NOT")
        print("    the lever — redirect Step-1 budget to the ranker.")


# ---------------------------------------------------------------------------
# 0c — coverage vs ordering
# ---------------------------------------------------------------------------
def audit_0c(c, meta, cand, N, wb, t3_union_10):
    bar(f"0c  COVERAGE vs ORDERING  (w_b={wb})")
    Kf, Kt, Ki, Kp = meta["Kf"], meta["Kt"], meta["Ki"], meta["Kpool"]
    nb = meta["nb"]
    gt_b, gt_f, gt_t, gt_i = c["gt_b"].long(), c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    fr, tr, ir = c["f_star_rank"], c["tf_t_rank"], c["tf_i_rank"]

    f_pruned = fr >= Kf
    bp_t = (~f_pruned) & (tr >= Kt)
    bp_i = (~f_pruned) & (tr < Kt) & (ir >= Ki)
    composed = (~f_pruned) & (tr < Kt) & (ir < Ki)

    # is GT actually present in its own behavior's pool?
    ar = torch.arange(N)
    pf = c["pool_f"][ar, gt_b].long(); pt = c["pool_t"][ar, gt_b].long(); pi = c["pool_i"][ar, gt_b].long()
    in_gt_pool = ((pf == gt_f[:, None]) & (pt == gt_t[:, None]) & (pi == gt_i[:, None])).any(1)
    # Split "composed but not in pool" into the REAL catalog bug vs mere Kpool truncation, using
    # the direct GT-in-catalog flag (backward-compat: assume valid if the flag is absent).
    gt_valid = c["gt_in_catalog"].bool() if "gt_in_catalog" in c else torch.ones(N, dtype=torch.bool)
    catalog_invalid = composed & (~gt_valid)                       # GT truly not in catalog -> BUG
    pool_truncated = composed & gt_valid & (~in_gt_pool)           # GT valid+composed but rank>Kpool

    # union-scored rank of GT joint candidate (same as 0a t3)
    score = union_score(cand, c, wb)
    beh_match = cand["branch"] == gt_b[:, None]
    sid_match = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    rank, has = best_rank(score, sid_match & beh_match, cand["valid"])

    def block(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<8} n=0"); return
        cov = in_gt_pool[mask].float().mean().item()
        print(f"  {name:<8} n={n:>7}   coverage@pool={cov:6.4f}   "
              f"f_pruned={f_pruned[mask].float().mean():.3f} "
              f"branch_t={bp_t[mask].float().mean():.3f} branch_i={bp_i[mask].float().mean():.3f} "
              f"trunc={pool_truncated[mask].float().mean():.4f} "
              f"cat_invalid={catalog_invalid[mask].float().mean():.4f}")
        hm = has & mask
        if hm.any():
            rr = rank[hm].float()
            print(f"           in-pool rank: P(<5)={ (rr < 5).float().mean():.3f} "
                  f"P(<10)={(rr < 10).float().mean():.3f} "
                  f"P(10-{Kp})={((rr >= 10) & (rr < Kp)).float().mean():.3f} median={int(rr.median())}")

    print("  loss decomposition + in-pool rank (by behavior; oracle = coverage@pool):")
    block("ALL", torch.ones(N, dtype=torch.bool))
    for b in range(nb):
        block(BEH.get(b, str(b)), gt_b == b)

    oracle = in_gt_pool.float().mean().item()
    n_inv = int(catalog_invalid.sum()); n_trunc = int(pool_truncated.sum())
    print(f"\n  ORACLE recall (coverage@pool, any reranker's ceiling) = {oracle:.4f}")
    print(f"  current t3@10 (union) = {t3_union_10:.4f}   -> ORDERING HEADROOM = {oracle - t3_union_10:+.4f}")
    print(f"  composed-but-not-pooled split: cat_invalid={n_inv} (real catalog bug)  "
          f"trunc={n_trunc} (GT valid, rank>Kpool={Kp})")
    print("\n  DECISION:")
    if n_inv > 0:
        print(f"    ** cat_invalid = {n_inv} events: GT tuples genuinely ABSENT from the catalog ->")
        print("       the constraint hash is WRONG. FIX the constraint before trusting anything.")
    else:
        print(f"    catalog OK (0 GT invalid). The {n_trunc} composed-but-not-pooled are POOL")
        print(f"    TRUNCATION (GT valid but ranked >Kpool={Kp}); they do NOT affect recall@10 (top-10")
        print("    only) and are fixed by a larger pool / reranker, not a catalog rebuild.")
    fp = f_pruned.float().mean().item(); bpt = bp_t.float().mean().item(); bpi = bp_i.float().mean().item()
    if fp > 0.05 or bpt > 0.05 or bpi > 0.05:
        worst = max(("f/K_f", fp), ("t/K_t", bpt), ("i/K_i", bpi), key=lambda x: x[1])
        print(f"    COVERAGE is the binding constraint, concentrated in {worst[0]} (={worst[1]:.3f}):")
        print(f"    cheap inference-only widening of that K is the first lever (see the K_f sweep).")
    elif oracle > 0.95 and (oracle - t3_union_10) > 0.02:
        print(f"    HIGH coverage ({oracle:.3f}) + ordering headroom ({oracle - t3_union_10:+.3f})")
        print("    -> ORDERING problem; the ranker (and PMQ) is the lever, K-widening is spent.")
    else:
        print(f"    coverage {oracle:.3f}, headroom {oracle - t3_union_10:+.3f}: modest on both.")


def audit_kf_sweep(c, meta):
    """Coverage@pool vs K_f (oracle ceiling), computed purely from the dumped FULL ranks -- no
    re-dump needed. Shows the knee: how much of f_pruned converts to coverage as K_f grows."""
    bar("K_f COVERAGE SWEEP  (oracle ceiling from full ranks; K_t=K_i fixed at the dump values)")
    nb, Kt, Ki = meta["nb"], meta["Kt"], meta["Ki"]
    gt_b = c["gt_b"].long()
    fr, tr, ir = c["f_star_rank"], c["tf_t_rank"], c["tf_i_rank"]
    gt_valid = c["gt_in_catalog"].bool() if "gt_in_catalog" in c else torch.ones_like(fr, dtype=torch.bool)
    branch_ok = (tr < Kt) & (ir < Ki) & gt_valid          # everything except the f-beam width
    print(f"  (branch/catalog ceiling with K_t={Kt},K_i={Ki}: the max coverage any K_f can reach)")
    header = "  {:<6}".format("K_f") + "".join(f"{('cov_' + BEH.get(b, str(b))):>11}" for b in range(nb)) + f"{'cov_ALL':>11}"
    print(header)
    for Kf in (10, 20, 50, 100, 200):
        row = f"  {Kf:<6}"
        for b in range(nb):
            m = gt_b == b
            cov = (branch_ok & m & (fr < Kf)).float().sum() / m.float().sum().clamp(min=1)
            row += f"{cov.item():>11.4f}"
        cov_all = (branch_ok & (fr < Kf)).float().mean().item()
        row += f"{cov_all:>11.4f}"
        print(row)
    ceil_all = branch_ok.float().mean().item()
    print(f"  K_f=inf ceiling (branch+catalog only) cov_ALL = {ceil_all:.4f}")
    print("  -> the knee is where cov_ALL stops climbing; that K_f is the cheap t3 win. Confirm the")
    print("     actual t3@10 at the chosen K_f with a real eval run (mfd_kf_sweep.sbatch).")


# G1..G6 grid: (name, K_f, K_t, K_i, K_pool). All computed from ONE dump's full-vocab ranks.
CONFIG_GRID = [
    ("G1", 200, 10, 10, 50),
    ("G2", 200, 10, 10, 200),
    ("G3", 200, 50, 20, 200),
    ("G4", 200, 200, 50, 200),
    ("G5", 200, 1024, 1024, 200),
    ("G6", 200, 1024, 1024, 500),
]


def audit_config_grid(c, meta, grid=CONFIG_GRID):
    """coverage@(K_f,K_t,K_i) for each requested config, from the stored FULL-vocab ranks. This is
    the ORACLE coverage (GT composed = f*/t*/i* all inside their K + catalog-valid); it does NOT
    include K_pool truncation (that needs the scored pool and is invisible to recall@10 anyway)."""
    bar("CONFIG GRID  (G1..G6)  — oracle coverage@(K_f,K_t,K_i) from full ranks")
    nb = meta["nb"]; pv = meta["pv_id"]
    gt_b = c["gt_b"].long()
    fr, tr, ir = c["f_star_rank"], c["tf_t_rank"], c["tf_i_rank"]
    gt_valid = c["gt_in_catalog"].bool() if "gt_in_catalog" in c else torch.ones_like(fr, dtype=torch.bool)
    print("  NOTE: K_pool is NOT an axis here — recall@{5,10} is K_pool-invariant (top-10 of a")
    print("        top-K_pool pool == top-10 overall), so G1==G2 and G5==G6 on both recall AND this")
    print("        oracle coverage. K_pool only shifts coverage@pool (truncation), which needs the")
    print("        scored pool; see 0c for that at the dump's K_pool.\n")
    print(f"  {'cfg':<4}{'K_f':>5}{'K_t':>6}{'K_i':>6}{'K_pool':>7}"
          f"{'cov_ALL':>10}{'cov_pv':>9}{'cov_buy':>9}{'f_pruned':>10}{'branch_t':>10}{'branch_i':>10}")
    for name, Kf, Kt, Ki, Kp in grid:
        composed = (fr < Kf) & (tr < Kt) & (ir < Ki) & gt_valid
        f_pr = (fr >= Kf).float().mean().item()
        b_t = ((fr < Kf) & (tr >= Kt)).float().mean().item()
        b_i = ((fr < Kf) & (tr < Kt) & (ir >= Ki)).float().mean().item()
        cov_all = composed.float().mean().item()
        cov_pv = composed[gt_b == pv].float().mean().item()
        cov_buy = composed[gt_b == meta["buy_id"]].float().mean().item()
        print(f"  {name:<4}{Kf:>5}{Kt:>6}{Ki:>6}{Kp:>7}"
              f"{cov_all:>10.4f}{cov_pv:>9.4f}{cov_buy:>9.4f}{f_pr:>10.4f}{b_t:>10.4f}{b_i:>10.4f}")
    print("\n  Reading it: G1->G3->G4->G5 shows branch widening converting branch_t/branch_i into")
    print("  coverage (text is the weak link, so K_t moves it most). G4 (K_t=200) vs G5 (K_t=1024)")
    print("  isolates how much text mass lives beyond the effective support. G2-G1 and G6-G5 are 0")
    print("  here (K_pool only) — the truncation they target is a coverage@pool / pool-rank effect,")
    print("  not a coverage@(K_f,K_t,K_i) or recall effect.")


def audit_pool_sweep(c, meta, cand, wb, kpools=(500, 1000, 2000)):
    """coverage@pool + P(rank<10) as K_pool grows, from GT's union score-rank (dump built at the
    largest K_pool). Isolates truncation: coverage rises with K_pool, but recall@10 = P(rank<10
    over ALL events) is K_pool-INVARIANT, so the extra coverage sits at rank>=10 (no recall gain)."""
    bar(f"K_pool SWEEP  (G4 branch width; coverage@pool + P(rank<10) from GT union rank, w_b={wb})")
    nb, pv, buy = meta["nb"], meta["pv_id"], meta["buy_id"]
    gt_b = c["gt_b"].long(); gt_f = c["gt_f"].long(); gt_t = c["gt_t"].long(); gt_i = c["gt_i"].long()
    score = union_score(cand, c, wb)
    beh_match = cand["branch"] == gt_b[:, None]
    sid_match = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    rank, has = best_rank(score, sid_match & beh_match, cand["valid"])   # GT union rank (t3 sense)
    stored = cand["branch"].shape[1]
    print(f"  (dump stored {stored} union candidates; coverage@K valid for K<=that. recall@10 is the")
    print(f"   K_pool-invariant P(rank<10) over ALL events.)")
    print(f"  {'K_pool':>7}{'cov_ALL':>10}{'cov_pv':>9}{'cov_buy':>9}"
          f"{'P(r<10|cov)':>13}{'recall@10':>11}{'median_r|cov':>13}")
    recall10 = (has & (rank >= 0) & (rank < 10)).float().mean().item()
    for K in kpools:
        cov = has & (rank >= 0) & (rank < K)
        cov_all = cov.float().mean().item()
        cov_pv = cov[gt_b == pv].float().mean().item()
        cov_buy = cov[gt_b == buy].float().mean().item()
        if cov.any():
            rr = rank[cov].float()
            p10 = (rr < 10).float().mean().item()
            med = int(rr.median().item())
        else:
            p10, med = 0.0, -1
        print(f"  {K:>7}{cov_all:>10.4f}{cov_pv:>9.4f}{cov_buy:>9.4f}{p10:>13.4f}{recall10:>11.4f}{med:>13}")
    print("\n  Reading it: cov_ALL climbs with K_pool (less truncation), but recall@10 is flat and")
    print("  P(r<10|cov) FALLS -- the newly-covered GTs land at high ranks, so pool widening buys")
    print("  coverage, not recall. Only a better ranker (or sharper branch posteriors, e.g. CMIR)")
    print("  converts that covered-but-outranked mass into top-10.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/scratch/yw8866/rec-tmall/mfd_audit_dump.pt")
    ap.add_argument("--chain_dump", default=None)
    ap.add_argument("--wb", type=float, nargs="+", default=None,
                    help="w_b values for 0a/0c (default: the run's w_b from meta)")
    # committed-beam reference (cited from the committed MFD run)
    ap.add_argument("--committed_t1_10", type=float, default=0.9403)
    ap.add_argument("--committed_t1_5", type=float, default=0.8932)
    ap.add_argument("--committed_t2_10", type=float, default=0.1996)
    ap.add_argument("--committed_t3_10", type=float, default=0.1899)
    args = ap.parse_args()

    c, meta = load(args.dump)
    N = c["gt_b"].shape[0]
    bar("MFD AUDIT")
    print(f"  dump: {args.dump}")
    print(f"  events={N:,}  nb={meta['nb']}  Kf={meta['Kf']} Kt={meta['Kt']} Ki={meta['Ki']} "
          f"Kpool={meta['Kpool']}  tau={meta['tau']}  w_b(run)={meta['w_b_run']}")
    for b in range(meta["nb"]):
        print(f"    {BEH.get(b, b):<5}: {int((c['gt_b'].long() == b).sum()):>8,} events")
    committed = {"t1_10": args.committed_t1_10, "t1_5": args.committed_t1_5,
                 "t2_10": args.committed_t2_10, "t3_10": args.committed_t3_10}

    chain = load(args.chain_dump) if args.chain_dump else None
    cand, N, nb, Kp = build_candidates(c, meta)
    wbs = args.wb if args.wb is not None else [meta["w_b_run"]]

    for wb in wbs:
        res = audit_0a(c, meta, cand, N, wb, committed)
        audit_0c(c, meta, cand, N, wb, res["t3_union_10"])
    audit_kf_sweep(c, meta)
    audit_config_grid(c, meta)
    audit_pool_sweep(c, meta, cand, wbs[0])
    audit_0b(c, meta, chain)


if __name__ == "__main__":
    main()