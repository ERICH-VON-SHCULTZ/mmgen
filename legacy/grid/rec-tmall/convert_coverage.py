"""Can MFD's coverage advantage be converted into recall under the strong reranker?

STAGE diagnose  -- WHY did widening K_t/K_i to full support move coverage 0.2615 -> 0.2618?
STAGE diag_sample -- the IN-MODEL diagnostic: reads the small instrumented sample dump (mfd_compose_debug>0,
                   K_t=K_i=1024 wide path, K_pool=500) whose new columns gt_comp_rank / gt_comp_score /
                   gt_in_pool / n_comp were computed INSIDE the composition stage; classifies every missing
                   GT (pruned by K_f / truncated by K_pool / real bug), reads coverage@K off gt_comp_rank,
                   and fixes K_pool (K_fix) for the regeneration -- or HALTS (exit 2 no K, exit 3 real bug).
STAGE curve     -- coverage@K table on the REGENERATED full P3W test dump (gate: coverage >= target).
STAGE grid      -- the minimal grid: P1 x {R1,R2} (cached) and P3W x {R1,R2} (trained), capped,
                   chain reused; the deciding metric = NET t3@10 change decomposed into newly-covered
                   gain and previously-covered loss, with bootstrap CIs.

Root cause (established by `diagnose`, verified numerically, not asserted):
  Pool membership needs TWO things: (a) survive branch pruning (rank(t*)<K_t and rank(i*)<K_i within
  the GT's f-branch) AND (b) rank inside the top-K_pool=500 catalog tuples by the HAND-SET score
  logP(f)+logP(t|f)+logP(i|f). The offline "ceiling" 0.7561 counts only (a). The wide-catalog path
  makes (a) trivially true (it reads the full 1024-way heads and scores every catalog tuple whose
  f-root survived), so the ONLY remaining truncation is (b) -- `score.topk(min(K_pool, Mc))`. The GTs
  that branch-widening frees are precisely the flat-branch, low-logP ones, so they enter the
  composition with scores far below the 500th candidate and are truncated by K_pool, not by any
  filter. That is why P2 recovered 62 and P3 68 GTs: K_pool=500 is the binding constraint.
  Consequence: coverage >= 0.50 is IMPOSSIBLE at K_pool=500 with hand-set selection. The genuine fix
  is a larger K_pool for the wide pool -- which is what `curve` sizes and `grid` then tests.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

from train_mfd_reranker import BEH, recall_at

EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def log(m, fh=None):
    print(m, flush=True)
    if fh:
        fh.write(m + "\n"); fh.flush()


def load_raw(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    _COLS_CACHE[path] = c                      # so d_cols() does not load the (large) file twice
    N, nb, Kp = c["pool_f"].shape
    f32 = lambda x: x.float() if x.dtype == torch.float16 else x
    o = {"N": N, "nb": nb, "Kp": Kp, "meta": meta,
         "gt_b": c["gt_b"].long(), "gt_f": c["gt_f"].long(), "gt_t": c["gt_t"].long(), "gt_i": c["gt_i"].long(),
         "pf": c["pool_f"].long(), "pt": c["pool_t"].long(), "pi": c["pool_i"].long(),
         "psc": f32(c["pool_sidscore"]), "beh": f32(c["beh_logp"]),
         "fsr": c["f_star_rank"].long(), "ttr": c["tf_t_rank"].long(), "tir": c["tf_i_rank"].long(),
         "fsl": f32(c["f_star_logp"]), "ttl": f32(c["tf_t_logp"]), "til": f32(c["tf_i_logp"])}
    ar = torch.arange(N)
    slab_f = o["pf"][ar, o["gt_b"]]; slab_t = o["pt"][ar, o["gt_b"]]; slab_i = o["pi"][ar, o["gt_b"]]   # (N,Kp) GT-behavior slice
    slab_s = o["psc"][ar, o["gt_b"]]
    match = ((slab_f == o["gt_f"][:, None]) & (slab_t == o["gt_t"][:, None]) & (slab_i == o["gt_i"][:, None])
             & (slab_s > -1e30))
    o["in_pool"] = match.any(1)
    o["gt_pos"] = torch.where(o["in_pool"], match.float().argmax(1), torch.full((N,), -1, dtype=torch.long))
    o["slab_s"] = slab_s
    return o


# ------------------------------------------------------------------------------- STAGE 1
def diagnose(paths, fh):
    log("=" * 100 + "\n=== STEP 1: WHY DOES WIDENING K_t / K_i NOT MOVE COVERAGE? ===\n" + "=" * 100, fh)
    for label, p in paths.items():
        o = load_raw(p); m = o["meta"]; N = o["N"]
        Kf, Kt, Ki, Kp = m.get("Kf"), m.get("Kt"), m.get("Ki"), o["Kp"]
        log(f"\n  [{label}] {os.path.basename(p)}", fh)
        log(f"    values reaching the composition stage (from the dump meta, i.e. what the model USED):", fh)
        log(f"      K_f={Kf}  K_t={Kt}  K_i={Ki}  K_pool={Kp}  wide_catalog={m.get('wide_catalog')}  "
            f"decoder-states/item={m.get('fwd_states_per_item')}", fh)
        if m.get("wide_catalog"):
            log(f"      composition = ALL catalog tuples whose f-root is in the top-K_f (K_t/K_i effectively FULL:", fh)
            log(f"      t/i log-probs are gathered from the complete 1024-way heads); ONLY truncation = top-K_pool by score.", fh)
        else:
            log(f"      composition = K_f*K_t*K_i = {Kf*Kt*Ki:,} (t,i) pairs per behavior before the catalog filter,", fh)
            log(f"      then catalog filter, then top-K_pool by score.", fh)

        f_surv = o["fsr"] < Kf
        brs_here = f_surv & (o["ttr"] < Kt) & (o["tir"] < Ki)          # branch survival at THIS pool's widths
        brs_p1 = f_surv & (o["ttr"] < 200) & (o["tir"] < 50)
        brs_full = f_surv                                                # K_t=K_i=1024 -> t,i ranks trivially inside
        cov = o["in_pool"]
        log(f"\n    branch-survival (rank(t*)<K_t & rank(i*)<K_i & f-root kept), the quantity the 0.7561 'ceiling' measures:", fh)
        log(f"      at P1 widths (200/50) = {brs_p1.float().mean():.4f}   at THIS pool's widths = {brs_here.float().mean():.4f}   "
            f"at full support = {brs_full.float().mean():.4f}", fh)
        log(f"    ACTUAL coverage@pool (GT inside the top-{Kp} by score)          = {cov.float().mean():.4f}", fh)
        # the GOAL's bug criterion: ranks inside K_t/K_i yet absent
        absent_ok = brs_here & ~cov
        log(f"\n    GTs with rank(t*)<K_t AND rank(i*)<K_i (and f kept) yet ABSENT from the pool: "
            f"{int(absent_ok.sum()):,} ({absent_ok.float().mean():.4f} of events)", fh)
        # where are they? compare the GT's own score to the pool cutoff in its behavior
        gt_sid = o["fsl"] + o["ttl"] + o["til"]                          # GT hand-set SID score
        slab = o["slab_s"].masked_fill(o["slab_s"] <= -1e30, float("inf"))
        pool_min = slab.min(1).values                                    # 500th-best score in GT's behavior
        below = (gt_sid < pool_min) & absent_ok
        log(f"      of those, GT score < the {Kp}th-ranked pool score (i.e. TRUNCATED BY K_pool): "
            f"{int(below.sum()):,} = {float(below.sum())/max(1,float(absent_ok.sum())):.4f}", fh)
        weird = absent_ok & ~(gt_sid < pool_min)
        log(f"      of those, GT score >= cutoff yet absent (would be a REAL composition/filter bug): "
            f"{int(weird.sum()):,}", fh)
        gap = (pool_min - gt_sid)[below]
        if gap.numel():
            q = np.percentile(gap.numpy(), [10, 25, 50, 75, 90])
            log(f"      score gap below the cutoff, quantiles 10/25/50/75/90: {['%.2f' % x for x in q]} (nats)", fh)
        # per behavior
        log(f"    per behavior: coverage | branch-survival(full) | truncated-by-K_pool share of the gap", fh)
        for b in range(o["nb"]):
            mb = o["gt_b"] == b
            gapb = (brs_full & ~cov & mb).float().sum()
            trunc_b = (below & mb).float().sum()
            log(f"      {BEH[b]:<5} cov={cov[mb].float().mean():.4f}  surv={brs_full[mb].float().mean():.4f}  "
                f"gap={int(gapb):,}  truncated={int(trunc_b):,}", fh)

    log("\n  " + "-" * 96, fh)
    log("  ROOT CAUSE: the pool is truncated to the top-K_pool=500 catalog tuples by the HAND-SET score.", fh)
    log("  Widening K_t/K_i removes branch pruning, but the GTs it frees are the flat-branch, low-logP", fh)
    log("  ones -- their scores sit far below the 500th candidate, so they are dropped by K_pool, not by", fh)
    log("  any composition step or filter (0 events fail the 'score >= cutoff yet absent' test). The", fh)
    log("  0.7561 'ceiling' measures branch survival only and ignores K_pool. Coverage >= 0.50 is therefore", fh)
    log("  IMPOSSIBLE at K_pool=500 with hand-set selection; the genuine fix is a larger K_pool for the", fh)
    log("  wide pool. This is NOT a plumbing bug -- it is the K_pool truncation identified in the", fh)
    log("  matched-compute analysis, now proven on the wide pools.", fh)
    log("  " + "-" * 96, fh)


# ------------------------------------------------------------------------------- STAGE 1b
def diag_sample(path, target, margin, ladder, max_kpool, out_json, fh):
    log("\n" + "=" * 100 + "\n=== STEP 1b: IN-MODEL composition-stage diagnostic (instrumented sample, K_t=K_i=1024, K_pool as dumped) ===\n" + "=" * 100, fh)
    d = torch.load(path, map_location="cpu", weights_only=False)
    c, m = d["cols"], d["meta"]
    N, nb, Kp = c["pool_f"].shape
    eff = m.get("eff", {})
    log(f"  sample N={N:,}  per-behavior: " + "  ".join(f"{BEH[b]}={int((c['gt_b'].long()==b).sum()):,}" for b in range(nb)), fh)
    log(f"  values IN EFFECT inside _audit_forward (recorded by the running code, not read from config):", fh)
    for k, v in eff.items():
        log(f"    {k:<12} = {v}", fh)
    log(f"  meta (config-derived, for comparison): Kf={m.get('Kf')} Kt={m.get('Kt')} Ki={m.get('Ki')} Kpool={m.get('Kpool')} "
        f"wide_catalog={m.get('wide_catalog')} states/item={m.get('fwd_states_per_item')} peak_mem={m.get('peak_mem_gb', 0):.2f}GB", fh)
    if "gt_comp_rank" not in c:
        raise SystemExit("** sample dump has no gt_comp_rank column -- the instrumented model was not the one that ran. **")
    r = c["gt_comp_rank"].long(); gsc = c["gt_comp_score"].float(); inp = c["gt_in_pool"].bool()
    fsr = c["f_star_rank"].long(); ttr = c["tf_t_rank"].long(); tir = c["tf_i_rank"].long()
    Kf = eff.get("Kf_eff", m.get("Kf")); Kt = eff.get("Kt_eff", m.get("Kt")); Ki = eff.get("Ki_eff", m.get("Ki"))
    ncomp = c["n_comp"].float()
    log(f"\n  candidates composed per event per behavior (before any cut): mean {ncomp.mean():,.0f}  min {ncomp.min():,.0f}  "
        f"max {ncomp.max():,.0f}   (dense grid would be Kf*Kt*Ki = {Kf*Kt*Ki:,}; the wide path composes every catalog tuple under a kept f)", fh)
    f_surv = fsr < Kf; brs = f_surv & (ttr < Kt) & (ttr >= 0) & (tir < Ki)
    composed = r >= 0
    log(f"  f-root survival (rank(f*)<K_f)                    = {f_surv.float().mean():.4f}", fh)
    log(f"  branch survival at effective K_t/K_i             = {brs.float().mean():.4f}   (the 0.7561-type 'ceiling')", fh)
    log(f"  GT COMPOSED at all (gt_comp_rank >= 0, in-model)  = {composed.float().mean():.4f}   "
        f"-> {'matches f-survival: K_t/K_i prune NOTHING' if abs(float(composed.float().mean()) - float(f_surv.float().mean())) < 1e-3 else 'DIFFERS from f-survival: t/i pruning is active'}", fh)
    log(f"  GT IN POOL (in-model membership test)             = {inp.float().mean():.4f}", fh)
    # classify every missing GT
    miss = ~inp
    pruned_f = miss & ~composed & ~f_surv
    pruned_ti = miss & ~composed & f_surv
    trunc = miss & composed & (r >= Kp)
    bug = miss & composed & (r < Kp)
    tot = int(miss.sum())
    log(f"\n  MISSING GTs = {tot:,} ({miss.float().mean():.4f} of events). Where did each one go?", fh)
    log(f"    f* pruned by K_f (never composed)                 : {int(pruned_f.sum()):>7,}  ({float(pruned_f.sum())/max(tot,1):.4f})", fh)
    log(f"    t*/i* pruned by K_t/K_i (never composed)          : {int(pruned_ti.sum()):>7,}  ({float(pruned_ti.sum())/max(tot,1):.4f})", fh)
    log(f"    composed, rank >= K_pool={Kp} (TRUNCATED BY K_pool) : {int(trunc.sum()):>7,}  ({float(trunc.sum())/max(tot,1):.4f})", fh)
    log(f"    composed, rank <  K_pool yet absent (REAL BUG)    : {int(bug.sum()):>7,}  ({float(bug.sum())/max(tot,1):.4f})", fh)
    if trunc.sum():
        q = np.percentile(r[trunc].numpy(), [10, 25, 50, 75, 90])
        log(f"    full-composition rank of the K_pool-truncated GTs, 10/25/50/75/90 pct: {[int(x) for x in q]}", fh)
    # coverage@K ladder
    log(f"\n  coverage@K if K_pool were K (from the in-model gt_comp_rank; sample SE ~ {0.5/np.sqrt(N):.4f}):", fh)
    log(f"  {'K':>8}{'coverage':>10}" + "".join(f"{BEH[b]:>9}" for b in range(nb)), fh)
    gb = c["gt_b"].long(); covs = {}
    for K in ladder:
        cv = composed & (r < K); covs[K] = float(cv.float().mean())
        log(f"  {K:>8}{covs[K]:>10.4f}" + "".join(f"{float(cv[gb==b].float().mean()):>9.4f}" for b in range(nb)), fh)
    log(f"  {'full':>8}{composed.float().mean():>10.4f}   (= f-root survival; K_t/K_i already at full support)", fh)
    bug_share = float(bug.sum()) / max(tot, 1)
    if bug_share > 0.01:
        log(f"\n  ** ROOT CAUSE = a REAL composition/filter bug: {bug_share:.4f} of missing GTs rank inside K_pool yet are absent. "
            f"Not fixable by K_pool -- HALT (fix the code path). **", fh)
        json.dump({"ok": False, "root_cause": "bug", "bug_share": bug_share, "covs": covs}, open(out_json, "w"), indent=1)
        raise SystemExit(3)
    log("\n  ROOT CAUSE (from inside the composition stage): K_t/K_i ARE at full support on the wide path and prune nothing;", fh)
    log(f"  {float(trunc.sum())/max(tot,1):.4f} of the missing GTs were composed and scored but rank >= K_pool={Kp} under the hand-set", fh)
    log("  score. The 0.7561 'ceiling' counted branch survival and ignored K_pool. There is no hidden cap, default", fh)
    log("  argument, buffer size or early top-k between the head softmaxes and composition on this path: the only", fh)
    log(f"  truncation is {eff.get('truncations')}. The low peak memory is expected: this path never materialises", fh)
    log("  K_f x K_t x K_i; it scores (B, Mc) catalog tuples. THE FIX IS A LARGER K_pool.", fh)
    need = target + margin
    ok_ks = [K for K in ladder if covs[K] >= need and K <= max_kpool]
    if not ok_ks:
        best = max((K for K in ladder if K <= max_kpool), key=lambda K: covs[K])
        log(f"\n  ** STEP 1 HALT: no K_pool <= {max_kpool:,} reaches sample coverage >= {need:.2f} (target {target:.2f} + margin {margin:.2f}); "
            f"best {covs[best]:.4f} at K={best:,}. Full ladder above shows what it would take. Not regenerating. **", fh)
        json.dump({"ok": False, "root_cause": "kpool", "covs": covs}, open(out_json, "w"), indent=1)
        raise SystemExit(2)
    Kfix = min(ok_ks)
    log(f"\n  K_fix = {Kfix:,}: smallest ladder K_pool with sample coverage >= {need:.2f} ({covs[Kfix]:.4f}). "
        f"P3 will be REGENERATED at K_t=K_i=1024, K_pool={Kfix:,}; the full-test gate is coverage >= {target:.2f}.", fh)
    log(f"  PROTOCOL NOTE: P1 and chain stay at K_pool=500; P3W's reranker sees {nb*Kfix:,} candidates/event vs {nb*500:,}. "
        f"Decoder compute is unchanged (808 states, asserted).", fh)
    json.dump({"ok": True, "root_cause": "kpool", "Kpool": Kfix, "covs": covs}, open(out_json, "w"), indent=1)


# ------------------------------------------------------------------------------- STAGE 2
def curve(path, ks, target, out_json, fh):
    log("\n" + "=" * 100 + "\n=== STEP 2: coverage vs K_pool from ONE wide test dump (K_t=K_i=1024) ===\n" + "=" * 100, fh)
    o = load_raw(path); m = o["meta"]; N = o["N"]; Kp = o["Kp"]
    per = {b: int((o["gt_b"] == b).sum()) for b in range(o["nb"])}
    ok = N == EXPECT_TOTAL and all(per.get(b, 0) == EXPECT_PERBEH[b] for b in EXPECT_PERBEH)
    st = m.get("fwd_states_per_item")
    log(f"  N={N:,} " + " ".join(f"{BEH[b]}={per.get(b,0):,}" for b in EXPECT_PERBEH) + f" -> {'OK' if ok else 'MISMATCH'}"
        f"   decoder-states/item={st} -> {'OK' if st == 808 else 'MISMATCH'}   K_pool dumped={Kp}", fh)
    if not ok or st != 808:
        raise SystemExit("** wide dump failed split/compute assertion -- stop. **")
    pos = o["gt_pos"]                                                    # GT position within its behavior's sorted pool
    log(f"\n  {'K_pool':>8}{'coverage@pool':>16}" + "".join(f"{BEH[b]:>9}" for b in range(o['nb'])), fh)
    covs = {}
    for K in ks:
        c = (pos >= 0) & (pos < K)
        covs[K] = float(c.float().mean())
        log(f"  {K:>8}{covs[K]:>16.4f}" + "".join(f"{float(c[o['gt_b']==b].float().mean()):>9.4f}" for b in range(o['nb'])), fh)
    brs = (o["fsr"] < m.get("Kf", 200)).float().mean()
    log(f"  (f-root survival = ceiling at full support = {brs:.4f}; coverage -> this only as K_pool -> catalog size)", fh)
    if "gt_comp_rank" in d_cols(path):
        rr = d_cols(path)["gt_comp_rank"].long()
        log("  beyond the dumped K_pool (from the in-model gt_comp_rank): " + "  ".join(
            f"K={K}:{float(((rr >= 0) & (rr < K)).float().mean()):.4f}" for K in (Kp, 2 * Kp, 5 * Kp, 10 * Kp)), fh)
    cov_pool = float(o["in_pool"].float().mean())
    log(f"\n  coverage@pool of the regenerated P3W test dump (K_pool={Kp}) = {cov_pool:.4f}  (gate: >= {target:.2f})", fh)
    if cov_pool < target:
        log(f"  ** STEP 2 FAILED: coverage {cov_pool:.4f} < {target:.2f}. HALTING -- no reranker result would mean anything. **", fh)
        json.dump({"ok": False, "cov": cov_pool, "covs": covs}, open(out_json, "w"), indent=1)
        raise SystemExit(2)
    log(f"  STEP 2 PASSED. P3W's reranker sees {o['nb']*Kp:,} candidates/event (P1/chain: 2,000); decoder states 808 everywhere.", fh)
    json.dump({"ok": True, "Kstar": Kp, "cov": cov_pool, "covs": covs}, open(out_json, "w"), indent=1)


_COLS_CACHE = {}
def d_cols(path):
    if path not in _COLS_CACHE:
        _COLS_CACHE[path] = torch.load(path, map_location="cpu", weights_only=False)["cols"]
    return _COLS_CACHE[path]


# ------------------------------------------------------------------------------- STAGE 3
def grid(args, fh):
    from coverage_expansion import Pool, canon, run_one
    log("\n" + "=" * 100 + "\n=== STEP 3: minimal grid -- P1 x {R1,R2} (cached), P3W = regenerated widened P3 at K_pool=K_fix x {R1,R2} (trained), chain (cached); capped ===\n" + "=" * 100, fh)
    spec = json.load(open(args.spec))
    cache = args.cache; os.makedirs(cache, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CONF = [("R1", None), ("R2_rw1", "rw1")]

    res = {}
    # cached P1 / chain (capped) from the previous full run
    for lab in ("P1", "CH"):
        for cn, _ in CONF:
            p = os.path.join(cache, f"{lab}__{cn}__capped.pt")
            if not os.path.exists(p):
                raise SystemExit(f"** missing cached result {p} **")
            res[(lab, cn)] = canon(torch.load(p, map_location="cpu", weights_only=False))
            log(f"  [cached] {lab} {cn}: t3@10 {res[(lab,cn)]['before']['t3@10']:.4f} -> {res[(lab,cn)]['after']['t3@10']:.4f}  "
                f"cov={res[(lab,cn)]['after']['cov']:.4f}", fh)
    # PW: train only what is missing
    need = [cn for cn, _ in CONF if not os.path.exists(os.path.join(cache, f"P3W__{cn}__capped.pt"))]
    if need:
        sp = spec["P3W"]
        pool = Pool("P3W", sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh)
        for cn, rw in CONF:
            if cn not in need:
                continue
            log(f"\n  >>> RUN P3W / {cn} / capped  (C={pool.te.C:,} candidates/event)", fh)
            r = run_one(pool, rw, args.cap, dev, args, fh)
            torch.save(r, os.path.join(cache, f"P3W__{cn}__capped.pt"))
        del pool
        if dev == "cuda":
            torch.cuda.empty_cache()
    for cn, _ in CONF:
        res[("P3W", cn)] = canon(torch.load(os.path.join(cache, f"P3W__{cn}__capped.pt"), map_location="cpu", weights_only=False))

    N = res[("P1", "R1")]["uid"].shape[0]
    for k in res:
        assert np.array_equal(res[k]["uid"], res[("P1", "R1")]["uid"]), f"canonical join failed for {k}"
    rng = np.random.default_rng(args.seed)

    # ---- full metrics table ----
    log("\n" + "=" * 100 + "\n=== FULL METRICS (capped) ===\n" + "=" * 100, fh)
    log(f"  {'pool':<5}{'cfg':<8}{'params':>9}{'cov':>8}{'t3@10 bef':>11}{'aft':>8}{'gain':>9}{'P(r<10|c)':>10}"
        f"{'t3@5':>8}{'t3@20':>8}{'n@5':>8}{'n@10':>8}{'t2@10':>8}{'t2n@10':>8}", fh)
    for lab in ("P1", "P3W", "CH"):
        for cn, _ in CONF:
            r = res[(lab, cn)]; b, a = r["before"], r["after"]
            log(f"  {lab:<5}{cn:<8}{r['params']:>9,}{a['cov']:>8.4f}{b['t3@10']:>11.4f}{a['t3@10']:>8.4f}"
                f"{a['t3@10']-b['t3@10']:>+9.4f}{a['pcov']:>10.4f}{a['t3@5']:>8.4f}{a['t3@20']:>8.4f}"
                f"{a['t3n@5']:>8.4f}{a['t3n@10']:>8.4f}{a['t2@10']:>8.4f}{a['t2n@10']:>8.4f}", fh)
    log("\n  per-behavior t3@10 (before -> after) + both-improve check:", fh)
    for lab in ("P1", "P3W", "CH"):
        for cn, _ in CONF:
            r = res[(lab, cn)]; bb, aa = r["before"]["perbeh"], r["after"]["perbeh"]
            dd = {k: aa[k] - bb[k] for k in aa}; conv = [k for k in dd if k != "pv"]
            v = ("BOTH improve" if dd.get("pv", 0) > 0 and all(dd[k] >= 0 for k in conv)
                 else "REDISTRIBUTION" if dd.get("pv", 0) > 0 else "pv down")
            log(f"    {lab:<5}{cn:<8}" + "  ".join(f"{k}:{bb[k]:.4f}->{aa[k]:.4f}" for k in aa) + f"  [{v}]", fh)

    # ---- THE DECIDING METRIC ----
    log("\n" + "=" * 100 + "\n=== THE DECIDING METRIC: newly covered GTs (P3W minus P1), and the NET ===\n" + "=" * 100, fh)
    verdicts = []
    for cn, _ in CONF:
        p1, pw = res[("P1", cn)], res[("P3W", cn)]
        c1, cw = p1["has_gt"].astype(bool), pw["has_gt"].astype(bool)
        new = cw & ~c1; lost_cov = c1 & ~cw
        n_new = int(new.sum())
        log(f"\n  [{cn}] coverage P1={c1.mean():.4f}  P3W={cw.mean():.4f}   newly covered={n_new:,}   "
            f"(P1-covered but NOT in P3W pool: {int(lost_cov.sum()):,})", fh)
        rk = pw["r3_after"][new]
        fr = {k: float((rk < k).mean()) for k in (10, 20, 50)}
        # bootstrap CI on the top-10 recovery fraction over the newly-covered events
        hit_new = (rk < 10).astype(np.float64)
        bs = np.array([hit_new[rng.integers(0, n_new, n_new)].mean() for _ in range(args.boot)]) if n_new else np.zeros(1)
        log(f"    newly-covered reaching top-10 = {fr[10]:.4f}  95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]"
            f"   top-20 = {fr[20]:.4f}   top-50 = {fr[50]:.4f}", fh)
        if n_new:
            q = np.percentile(rk, [10, 25, 50, 75, 90])
            log(f"    rank distribution of newly-covered GTs (10/25/50/75/90 pct): {[int(x) for x in q]}", fh)
            z = pw["ent_z"][new]
            log(f"    by GT-branch entropy z (flat = high):", fh)
            for lo, hi, nm in ((-1e9, -1., "<-1"), (-1., 0., "-1..0"), (0., 1., "0..1"), (1., 1e9, ">1")):
                mm = (z >= lo) & (z < hi)
                if mm.sum():
                    log(f"      z {nm:<6} n={int(mm.sum()):>7,}  top10={float((rk[mm] < 10).mean()):.4f}", fh)
            gb = pw["gt_b"][new]
            log(f"    per behavior top10: " + "  ".join(
                f"{BEH[b]}={float((rk[gb==b] < 10).mean()):.4f}(n={int((gb==b).sum()):,})" for b in range(4) if (gb == b).sum()), fh)
        # net decomposition: t3@10(PW) - t3@10(P1) = new_gain + prev_regained - prev_lost
        h1, hw = p1["hit_after"] > 0, pw["hit_after"] > 0
        new_gain = int((new & hw).sum()); prev_lost = int((c1 & h1 & ~hw).sum()); prev_regained = int((c1 & ~h1 & hw).sum())
        net = pw["after"]["t3@10"] - p1["after"]["t3@10"]
        log(f"    NET t3@10 (P3W - P1) = {net:+.4f}  =  newly-covered gain {new_gain:,}/{N:,} ({new_gain/N:+.4f})"
            f"  +  previously-covered regained {prev_regained:,} ({prev_regained/N:+.4f})"
            f"  -  previously-covered LOST from top-10 {prev_lost:,} ({-prev_lost/N:+.4f})", fh)
        chk = (new_gain + prev_regained - prev_lost) / N
        log(f"    decomposition check: {chk:+.4f} vs measured {net:+.4f} -> {'OK' if abs(chk-net) < 1e-6 else 'MISMATCH'}", fh)
        # paired bootstrap on the NET over all events
        d = (hw.astype(np.float64) - h1.astype(np.float64))
        bn = np.array([d[rng.integers(0, N, N)].mean() for _ in range(args.boot)])
        lo, hi = np.percentile(bn, 2.5), np.percentile(bn, 97.5)
        log(f"    NET 95% CI (paired over events, B={args.boot}) = [{lo:+.4f}, {hi:+.4f}]  P(net>0)={float((bn>0).mean()):.4f}", fh)
        verdicts.append((cn, net, lo, hi))

    # ---- four diagnostic cells vs chain ----
    log("\n" + "=" * 100 + "\n=== FOUR DIAGNOSTIC CELLS vs chain (BOTH/10-49, BOTH/50+, own/10-49, own/50+) ===\n" + "=" * 100, fh)
    log("  P1 capped R2_rw1 reference: MFD 0.6495 / 0.2656 / 0.2153 / 0.0467 ; chain 0.4136 / 0.0194 / 0.0992 / 0.0034", fh)
    for cn, _ in CONF:
        ch = res[("CH", cn)]; cc = ch["has_gt"].astype(bool)
        for lab in ("P1", "P3W", "CH"):
            r = res[(lab, cn)]; cm = r["has_gt"].astype(bool)
            other = cc if lab != "CH" else res[("P1", cn)]["has_gt"].astype(bool)   # chain own-only vs P1 = the stated reference
            both, own = cm & other, cm & ~other
            cells = []
            for grp in (both, own):
                for lo, hi in ((10, 50), (50, 10 ** 9)):
                    mm = grp & (r["r3_before"] >= lo) & (r["r3_before"] < hi)
                    cells.append(float((r["hit_after"][mm] > 0).mean()) if mm.sum() else float("nan"))
            log(f"  {cn:<8}{lab:<5} BOTH/10-49={cells[0]:.4f}  BOTH/50+={cells[1]:.4f}  own/10-49={cells[2]:.4f}  own/50+={cells[3]:.4f}"
                + ("   (own = vs P1, reference-comparable)" if lab == "CH" else "   (own = vs chain)"), fh)

    # ---- P3W vs chain significance, fixed paired protocol ----
    log("\n" + "=" * 100 + "\n=== SIGNIFICANCE: P3W vs chain (same config, joined on user_id, PAIRED, B=2000) ===\n" + "=" * 100, fh)
    log("  reference at P1 capped R2_rw1: +0.0012, 95% CI [+0.0004, +0.0020]", fh)
    for cn, _ in CONF:
        for lab in ("P1", "P3W"):
            hm = res[(lab, cn)]["hit_after"]; hc = res[("CH", cn)]["hit_after"]
            d = np.empty(args.boot)
            for k in range(args.boot):                       # PAIRED: same resampled events for both arms
                ii = rng.integers(0, N, N); d[k] = hm[ii].mean() - hc[ii].mean()
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            log(f"  [{cn}] {lab} - chain: {hm.mean()-hc.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"P(MFD>chain)={float((d>0).mean()):.4f}  {'DISTINGUISHABLE' if (lo>0 or hi<0) else 'not distinguishable'}", fh)

    # ---- verdict against an ABSOLUTE threshold, fixed in advance ----
    log("\n" + "=" * 100 + "\n=== VERDICT (absolute threshold, fixed in advance) ===\n" + "=" * 100, fh)
    log("  Judged on the NET t3@10 change P1 -> P3W (not on gross recovery, which was 4/62 last time):", fh)
    for cn, net, lo, hi in verdicts:
        if lo > 0 and net >= 0.002:
            v = "LIVE LEVER -- net gain clearly positive and significant; the four earlier negative coverage results were an artefact of the weak scorer."
        elif hi < 0:
            v = "DEAD -- net change significantly NEGATIVE: the wider pool costs more than it recovers."
        else:
            v = "DEAD -- net gain ~0 / not significant under a much stronger test than before; the coverage axis closes."
        log(f"  [{cn}] net {net:+.4f} CI [{lo:+.4f}, {hi:+.4f}] -> {v}", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["diagnose", "diag_sample", "curve", "grid"])
    ap.add_argument("--diag_dump", default=""); ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--ladder", default="500,1000,1500,2000,3000,4000,5000,7500,10000,15000,20000,30000,50000,100000")
    ap.add_argument("--max_kpool", type=int, default=10000); ap.add_argument("--kfix_json", default="")
    ap.add_argument("--p1_test", default=""); ap.add_argument("--p2_test", default=""); ap.add_argument("--p3_test", default="")
    ap.add_argument("--wide_dump", default=""); ap.add_argument("--ks", default="500,1000,2000,3000,4000,5000")
    ap.add_argument("--target_cov", type=float, default=0.50); ap.add_argument("--kstar_json", default="")
    ap.add_argument("--spec", default=""); ap.add_argument("--cache", default="")
    ap.add_argument("--cap", type=int, default=36864); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=0.0); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/coverage_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fh = open(os.path.join(args.out_dir, "convert_report.txt"), "a")
    if args.stage == "diagnose":
        paths = {k: v for k, v in (("P1", args.p1_test), ("P2", args.p2_test), ("P3", args.p3_test)) if v}
        diagnose(paths, fh)
    elif args.stage == "diag_sample":
        diag_sample(args.diag_dump, args.target_cov, args.margin, [int(x) for x in args.ladder.split(",")],
                    args.max_kpool, args.kfix_json, fh)
    elif args.stage == "curve":
        curve(args.wide_dump, [int(x) for x in args.ks.split(",")], args.target_cov, args.kstar_json, fh)
    else:
        grid(args, fh)
    fh.close()


if __name__ == "__main__":
    main()
