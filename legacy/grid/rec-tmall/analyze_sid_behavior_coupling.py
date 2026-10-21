"""Probe SID<->behavior coupling and per-behavior modality asymmetry.

Two diagnostics, each making a go/no-go call for a downstream research step. Fits on the
TRAINING split, evaluates on the EVAL split (honest under the natural ~93.6% pv prior).

  Part A (--part mi):   Is behavior predictable from the SID, beyond what previous behavior
                        already gives?  -> gates Step 2 (SID-conditioned behavior head).
  Part B (--part asym): Do different behaviors rank their next item better under the text
                        view vs the image view?  -> gates Step 3 (behavior-gated BM-TV).
  Part G (--part factor): From semantic_ids.pt ALONE (no behavior, no model), is the text
                        code L1 and image code L2 near-independent given the shared code L0?
                        Measures CMI(L1;L2|L0) with a within-group shuffle bias floor. If it
                        is small vs H(L1|L0), H(L2|L0) the factorization is near-lossless ->
                        licenses MFD by the data before training (§12g companion to A.4).

Follows the analyze_tfrecords_mb.py pattern: bounded Python, py310 + tensorflow, prints a
report to stdout (redirect to expoutfile/), and also writes a coupling_artifacts.pt so Step 3's
gate-supervision targets can be loaded without re-running.

Inputs:
  --data_dir     tfrecords_mb_userdrop (verified; rebalanced train, natural eval -> prior
                 correction applied). Swap to tfrecords_mb (natural) to make correction a no-op.
  --sid_path     semantic_ids.pt  [3, N]  widths [2048,1024,1024], column == item_id.
  --ckpt_path    MD-RQ-VAE 2048_1024 checkpoint (Part B only; loaded mmap to skip the 50GB).

Usage:
  python analyze_sid_behavior_coupling.py --part both --max_users 2000     # smoke
  python analyze_sid_behavior_coupling.py --part mi                        # full Part A
  python analyze_sid_behavior_coupling.py --part asym                      # full Part B
  python analyze_sid_behavior_coupling.py --part factor                    # §12g CMI test
"""

import argparse
import collections
import os
import sys
import time

import numpy as np

BEHAVIOR_NAMES = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
BUY = 3
LEVEL_NAMES = ("shared", "text", "img")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bar(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------------------
# TFRecord reading (matches analyze_tfrecords_mb.py)
# ---------------------------------------------------------------------------
def iterate_users(data_dir, split, max_users=None):
    """Yield (items int64[L], behaviors int64[L]) per user in a split."""
    import tensorflow as tf

    d = os.path.join(data_dir, split)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tfrecord.gz"))
    spec = {
        "user_id": tf.io.VarLenFeature(tf.int64),
        "sequence_data": tf.io.VarLenFeature(tf.int64),
        "behavior_data": tf.io.VarLenFeature(tf.int64),
    }
    ds = tf.data.TFRecordDataset(paths, compression_type="GZIP", num_parallel_reads=8)
    ds = ds.map(lambda r: tf.io.parse_single_example(r, spec), num_parallel_calls=tf.data.AUTOTUNE)
    n = 0
    for ex in ds:
        items = tf.sparse.to_dense(ex["sequence_data"]).numpy().astype(np.int64)
        behs = tf.sparse.to_dense(ex["behavior_data"]).numpy().astype(np.int64)
        yield items, behs
        n += 1
        if max_users is not None and n >= max_users:
            return


# ---------------------------------------------------------------------------
# Shared: load SIDs + coverage guard
# ---------------------------------------------------------------------------
def load_sids(sid_path):
    import torch

    sids = torch.load(sid_path, map_location="cpu", weights_only=False)
    sids = sids.numpy().astype(np.int64)  # [3, N]
    N = sids.shape[1]
    allzero = (sids == 0).all(axis=0)  # [N] placeholder columns
    log(f"SIDs {sids.shape}  N={N:,}  placeholder(all-zero) columns={int(allzero.sum()):,}")
    return sids, N, allzero


def clean_sequence(items, behs, N, allzero):
    """Drop events whose item_id is out of range or maps to a placeholder SID.

    Returns (items_valid, behs_valid, n_skipped_here). Dropping keeps the sequence
    contiguous so transition (b_prev) adjacency is over VALID events only.
    """
    valid = (items >= 0) & (items < N)
    valid[valid] &= ~allzero[items[valid]]
    return items[valid], behs[valid], int((~valid).sum())


def _prior_behavior_masks(items, behs):
    """4-bit prior-behavior mask per position (for Part A repeat-evidence).

    out[t] has bit b set iff item[t] appeared EARLIER in this sequence (index < t) with
    behavior b. So it summarises "how has the user interacted with this exact item before".
    Sequential single pass, O(L).
    """
    seen = {}
    out = np.zeros(len(items), np.int64)
    for t in range(len(items)):
        it = int(items[t])
        out[t] = seen.get(it, 0)
        seen[it] = seen.get(it, 0) | (1 << int(behs[t]))
    return out


# ---------------------------------------------------------------------------
# Part G (§12g): is the SID's text/image factorization near-lossless?
#   Count-level, from semantic_ids.pt ALONE (no behavior, no model). Over all 5.4M
#   items, measures how much the text code L1 and image code L2 still share once the
#   fusion/shared code L0 is known:  CMI(L1;L2 | L0).  If CMI is small relative to
#   H(L1|L0) and H(L2|L0), then given L0 the two modality codes are nearly independent
#   -> the factorization P(L1,L2|L0)=P(L1|L0)P(L2|L0) is near-lossless, and modality-
#   factorized decoding (MFD) is licensed by the DATA before any model is trained.
#   This is the structural companion to A.4 (which showed the residuals carry behavior
#   info); §12g shows they are near-orthogonal given the shared code.
# ---------------------------------------------------------------------------
def _cmi_from_triples(l0, l1, l2, W1, W2, K0, N):
    """Plug-in H(L1|L0), H(L2|L0), H(L1,L2|L0), CMI(L1;L2|L0) in bits from index arrays.

    All counts come from bincount / np.unique over integer keys -- no dense L0xL1xL2
    array (that would be 2.1e9 cells). Nv = len(l0) is the total event count used to
    normalise. Returns a dict of the four entropy/MI quantities plus the joint-support size.
    """
    Nv = l0.shape[0]
    C0 = np.bincount(l0, minlength=K0).astype(np.float64)              # [K0]
    C01 = np.bincount(l0 * W1 + l1, minlength=K0 * W1).astype(np.float64)  # [K0*W1]
    C02 = np.bincount(l0 * W2 + l2, minlength=K0 * W2).astype(np.float64)  # [K0*W2]

    def cond_H(Cxy, W):  # H(Lx|L0) = -(1/Nv) sum Cxy*(log2 Cxy - log2 C0[l0])
        nz = Cxy > 0
        idx0 = np.nonzero(nz)[0] // W
        c = Cxy[nz]
        return -float(np.sum(c * (np.log2(c) - np.log2(C0[idx0]))) / Nv)

    H1g = cond_H(C01, W1)
    H2g = cond_H(C02, W2)

    # joint (l0,l1,l2) is sparse: at most Nv distinct triples -> np.unique on int64 key.
    key = (l0.astype(np.int64) * W1 + l1) * W2 + l2
    uk, cnt = np.unique(key, return_counts=True)
    n_triples = uk.shape[0]
    t_l0 = (uk // (W1 * W2)).astype(np.int64)
    t_l1 = ((uk // W2) % W1).astype(np.int64)
    t_l2 = (uk % W2).astype(np.int64)
    cnt = cnt.astype(np.float64)
    # H(L1,L2|L0) = -(1/Nv) sum n*(log2 n - log2 C0[l0])
    H12g = -float(np.sum(cnt * (np.log2(cnt) - np.log2(C0[t_l0]))) / Nv)
    # CMI = (1/Nv) sum n * log2( n * C0[l0] / (C01[l0,l1] * C02[l0,l2]) )
    num = np.log2(cnt) + np.log2(C0[t_l0])
    den = np.log2(C01[t_l0 * W1 + t_l1]) + np.log2(C02[t_l0 * W2 + t_l2])
    cmi = float(np.sum(cnt * (num - den)) / Nv)
    return {"H1g": H1g, "H2g": H2g, "H12g": H12g, "cmi": cmi, "n_triples": n_triples}


def part_g(args, sids, N, allzero):
    bar("PART G (§12g) - is the L1/L2 (text/image) factorization near-lossless?")
    keep = ~allzero
    l0 = sids[0, keep].astype(np.int64)
    l1 = sids[1, keep].astype(np.int64)
    l2 = sids[2, keep].astype(np.int64)
    Nv = l0.shape[0]
    K0 = int(l0.max()) + 1
    W1 = int(l1.max()) + 1
    W2 = int(l2.max()) + 1
    # widen the packing radix to the codebook width so shuffles that move codes around
    # never collide keys; use the declared widths when larger than the observed max.
    W1 = max(W1, 1024)
    W2 = max(W2, 1024)
    log(f"items used={Nv:,}  (dropped {int(allzero.sum()):,} placeholders)  "
        f"L0∈[0,{K0}) L1 radix={W1} L2 radix={W2}")

    # marginal entropies (descriptive)
    def H(x, W):
        c = np.bincount(x, minlength=W).astype(np.float64)
        p = c[c > 0] / c.sum()
        return -float(np.sum(p * np.log2(p)))
    H0, H1, H2 = H(l0, K0), H(l1, W1), H(l2, W2)

    # unconditional MI(L1;L2) for contrast (how coupled BEFORE conditioning on the shared code)
    C12 = np.bincount(l1 * W2 + l2, minlength=W1 * W2).astype(np.float64)
    p12 = C12.reshape(W1, W2) / Nv
    p1 = p12.sum(1, keepdims=True); p2 = p12.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p12 * (np.log2(p12) - np.log2(p1) - np.log2(p2))
    mi_uncond = float(np.nansum(term))

    R = _cmi_from_triples(l0, l1, l2, W1, W2, K0, Nv)
    H1g, H2g, cmi_raw = R["H1g"], R["H2g"], R["cmi"]

    # ---- plug-in bias floor via within-L0 shuffles of L2 ----
    # Under the null (L1 ⟂ L2 | L0) the true CMI is 0, but the plug-in estimator is
    # positively biased on sparse per-group joints. Permuting L2 *within each L0 group*
    # preserves C0, C01, C02 exactly while destroying any real L1<->L2 coupling, so the
    # shuffled CMI estimates the bias. Corrected CMI = raw - mean(shuffle).
    rng = np.random.default_rng(args.seed)
    order_stable = np.argsort(l0, kind="stable")
    n_shuf = 3
    shuf_cmis = []
    for s in range(n_shuf):
        order_rand = np.lexsort((rng.random(Nv), l0))  # per-group random order
        l2_shuf = np.empty_like(l2)
        l2_shuf[order_stable] = l2[order_rand]         # permute L2 within each L0 group
        shuf_cmis.append(_cmi_from_triples(l0, l1, l2_shuf, W1, W2, K0, Nv)["cmi"])
        log(f"  shuffle {s+1}/{n_shuf}: CMI_null={shuf_cmis[-1]:.4f} bits")
    cmi_null = float(np.mean(shuf_cmis))
    cmi_corr = cmi_raw - cmi_null

    print("\n  marginal entropies (bits):")
    print(f"    H(L0 shared) = {H0:.3f}   H(L1 text) = {H1:.3f}   H(L2 img) = {H2:.3f}")
    print(f"    unconditional MI(L1;L2)        = {mi_uncond:.4f} bits   "
          f"({mi_uncond/min(H1,H2)*100:.1f}% of min(H1,H2))")
    print("\n  conditional on the shared code L0 (bits):")
    print(f"    H(L1|L0) = {H1g:.4f}")
    print(f"    H(L2|L0) = {H2g:.4f}")
    print(f"    H(L1,L2|L0) = {R['H12g']:.4f}   (identity check: H1g+H2g-CMI = "
          f"{H1g + H2g - cmi_raw:.4f})")
    print("\n  CMI(L1;L2 | L0) — residual text/image coupling AFTER the shared code:")
    print(f"    plug-in (raw)       = {cmi_raw:.4f} bits")
    print(f"    null floor (shuffle)= {cmi_null:.4f} bits   <- estimator bias")
    print(f"    bias-corrected      = {cmi_corr:.4f} bits")
    frac1 = cmi_corr / H1g if H1g > 0 else 0.0
    frac2 = cmi_corr / H2g if H2g > 0 else 0.0
    fracj = cmi_corr / R["H12g"] if R["H12g"] > 0 else 0.0
    print("\n  normalised (corrected CMI as a fraction of the code entropy it could couple):")
    print(f"    CMI / H(L1|L0)   = {frac1*100:6.2f}%   <- of the text code is coupled to image | L0")
    print(f"    CMI / H(L2|L0)   = {frac2*100:6.2f}%   <- of the image code is coupled to text | L0")
    print(f"    CMI / H(L1,L2|L0)= {fracj*100:6.2f}%   <- of the joint conditional entropy")

    bar("G  DECISION GATE  (is MFD licensed by the data?)")
    lossy = max(frac1, frac2)
    print(f"  residual coupling = {lossy*100:.2f}% of the smaller conditional code entropy")
    print(f"  (unconditional coupling was {mi_uncond/min(H1,H2)*100:.1f}% -> the shared code L0 "
          f"absorbs {(1 - cmi_corr/max(mi_uncond,1e-9))*100:.1f}% of it)")
    if lossy < 0.05:
        verdict = ("GREEN  -> given L0 the text/image codes are near-independent; the "
                   "factorization P(L1,L2|L0)=P(L1|L0)P(L2|L0) is near-lossless. MFD is "
                   "licensed by the data before any training.")
    elif lossy < 0.10:
        verdict = ("AMBER  -> mostly factorized but a residual L1<->L2 dependence remains; "
                   "MFD is defensible but a small joint-rerank head may recover the tail.")
    else:
        verdict = ("RED    -> the modality codes stay coupled given L0; a fully factorized "
                   "decoder would drop real joint structure -- keep chained/joint decoding.")
    print(f"  VERDICT: {verdict}")

    return {"H0": H0, "H1": H1, "H2": H2, "mi_uncond": mi_uncond,
            "H1g": H1g, "H2g": H2g, "H12g": R["H12g"], "n_triples": R["n_triples"],
            "cmi_raw": cmi_raw, "cmi_null": cmi_null, "cmi_corr": cmi_corr,
            "frac_H1g": frac1, "frac_H2g": frac2, "frac_joint": fracj, "n_items": Nv}


# ---------------------------------------------------------------------------
# Part A: is behavior predictable from the SID?
# ---------------------------------------------------------------------------
def part_a(args, sids, N, allzero):
    bar("PART A - behavior predictability from SID")
    K = [2048, 1024, 1024]
    lvl = [sids[0], sids[1], sids[2]]  # per-level code lookup by item_id

    # ---- training counting pass ----
    K2 = 1024 * 1024  # triple key = shared*K2 + text*1024 + img
    C = [np.zeros((K[l], 4), np.float64) for l in range(3)]      # C[l][code, b]
    C_bprev = np.zeros((4, 4), np.float64)                        # [b_prev, b]
    C_shared_prev = np.zeros((4, K[0], 4), np.float64)           # [b_prev, shared, b]
    train_bcount = np.zeros(4, np.float64)
    # (a) modality additivity: dense pairwise joints + sparse triple (built via unique)
    T_stext = np.zeros((K[0], K[1], 4), np.float64)             # [shared, text, b]
    T_simg = np.zeros((K[0], K[2], 4), np.float64)              # [shared, img,  b]
    tri_keys, tri_b = [], []
    # (b) repeat evidence: prior-behavior mask (4-bit) x b_prev x shared -> b
    B3 = np.zeros((16, 4, K[0], 4), np.float64)
    B2 = np.zeros((16, 4, 4), np.float64)                       # mask x b_prev -> b (backoff)
    full_tuple = collections.defaultdict(lambda: np.zeros(4, np.float64)) if args.full_tuple else None
    skip = tot = users = 0
    t0 = time.time()

    for items, behs in iterate_users(args.data_dir, "training", args.max_users):
        items, behs, sk = clean_sequence(items, behs, N, allzero)
        skip += sk
        tot += len(items) + sk
        users += 1
        if len(items) == 0:
            continue
        s0, s1, s2 = lvl[0][items], lvl[1][items], lvl[2][items]
        np.add.at(C[0], (s0, behs), 1.0)
        np.add.at(C[1], (s1, behs), 1.0)
        np.add.at(C[2], (s2, behs), 1.0)
        np.add.at(T_stext, (s0, s1, behs), 1.0)
        np.add.at(T_simg, (s0, s2, behs), 1.0)
        tri_keys.append(s0.astype(np.int64) * K2 + s1.astype(np.int64) * 1024 + s2.astype(np.int64))
        tri_b.append(behs.astype(np.int8))
        train_bcount += np.bincount(behs, minlength=4)
        masks = _prior_behavior_masks(items, behs)
        if len(items) > 1:
            bp, bc, s0c = behs[:-1], behs[1:], s0[1:]
            m = masks[1:]
            np.add.at(C_bprev, (bp, bc), 1.0)
            np.add.at(C_shared_prev, (bp, s0c, bc), 1.0)
            np.add.at(B3, (m, bp, s0c, bc), 1.0)
            np.add.at(B2, (m, bp, bc), 1.0)
        if full_tuple is not None:
            for a, b_, c, bb in zip(s0, s1, s2, behs):
                full_tuple[(int(a), int(b_), int(c))][bb] += 1.0
        if users % 200000 == 0:
            log(f"  train {users:,} users ({(time.time()-t0)/60:.1f} min)")

    cov = skip / max(tot, 1)
    log(f"training: {users:,} users, {int(tot):,} events, coverage-skip {cov*100:.3f}%")
    if cov > 0.02:
        log("!! skip rate > 2% -- item sets may be mismatched; investigate before trusting.")

    # sparse triple table for (a): per-key [4] counts, keep only >=50 train support.
    tri_keys = np.concatenate(tri_keys)
    tri_b = np.concatenate(tri_b).astype(np.int64)
    uniq_key, inv = np.unique(tri_keys, return_inverse=True)
    tri_tab = np.zeros((len(uniq_key), 4), np.float64)
    np.add.at(tri_tab, (inv, tri_b), 1.0)
    keep = tri_tab.sum(1) >= 50
    keep_keys = uniq_key[keep]                                  # sorted (from np.unique)
    keep_tab = tri_tab[keep]
    log(f"triple table: {len(uniq_key):,} distinct, {int(keep.sum()):,} with >=50 support")
    del tri_keys, tri_b, inv, uniq_key, tri_tab

    # ---- eval counting pass (MI on natural distribution) + scoring collection ----
    EV = [np.zeros((K[l], 4), np.float64) for l in range(3)]
    EV_bprev = np.zeros((4, 4), np.float64)
    eval_bcount = np.zeros(4, np.float64)
    # per-eval-transition scoring inputs (t>=1 events only, which always have b_prev)
    sc_bp, sc_s0, sc_s1, sc_s2, sc_b, sc_mask = [], [], [], [], [], []
    # ALL eval events (for (a) held-out predictive-MI cross-entropy)
    ev_s0, ev_s1, ev_s2, ev_b = [], [], [], []
    eskip = etot = eusers = 0

    for items, behs in iterate_users(args.data_dir, "evaluation", args.max_users):
        items, behs, sk = clean_sequence(items, behs, N, allzero)
        eskip += sk
        etot += len(items) + sk
        eusers += 1
        if len(items) == 0:
            continue
        s0, s1, s2 = lvl[0][items], lvl[1][items], lvl[2][items]
        np.add.at(EV[0], (s0, behs), 1.0)
        np.add.at(EV[1], (s1, behs), 1.0)
        np.add.at(EV[2], (s2, behs), 1.0)
        eval_bcount += np.bincount(behs, minlength=4)
        ev_s0.append(s0); ev_s1.append(s1); ev_s2.append(s2); ev_b.append(behs)
        if len(items) > 1:
            EV_bprev_bp, EV_bprev_bc = behs[:-1], behs[1:]
            np.add.at(EV_bprev, (EV_bprev_bp, EV_bprev_bc), 1.0)
            sc_bp.append(behs[:-1]); sc_b.append(behs[1:])
            sc_s0.append(s0[1:]); sc_s1.append(s1[1:]); sc_s2.append(s2[1:])
            sc_mask.append(_prior_behavior_masks(items, behs)[1:])

    ecov = eskip / max(etot, 1)
    log(f"eval: {eusers:,} users, {int(etot):,} events, coverage-skip {ecov*100:.3f}%")
    sc_bp = np.concatenate(sc_bp); sc_b = np.concatenate(sc_b)
    sc_s0 = np.concatenate(sc_s0); sc_s1 = np.concatenate(sc_s1); sc_s2 = np.concatenate(sc_s2)
    sc_mask = np.concatenate(sc_mask)
    ev_s0 = np.concatenate(ev_s0); ev_s1 = np.concatenate(ev_s1)
    ev_s2 = np.concatenate(ev_s2); ev_b = np.concatenate(ev_b)
    log(f"eval scoring transitions: {len(sc_b):,}  (all events {len(ev_b):,})")

    train_prior = train_bcount / train_bcount.sum()
    eval_prior = eval_bcount / eval_bcount.sum()
    print("\n  behavior priors:")
    for b in range(4):
        print(f"    {BEHAVIOR_NAMES[b]:<5}: train={train_prior[b]*100:6.2f}%  eval={eval_prior[b]*100:6.2f}%")

    # ---- Mutual information (descriptive, on eval joint) ----
    def mutual_info(J):
        p = J / J.sum()
        pk = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = p * (np.log2(p) - np.log2(pk) - np.log2(pb))
        return float(np.nansum(term))

    HB = -float(np.nansum(eval_prior * np.log2(eval_prior + 1e-12)))
    mi_lvl = [mutual_info(EV[l]) for l in range(3)]
    mi_bprev = mutual_info(EV_bprev)
    bar("A.1  Mutual information (eval split)")
    print(f"  H(B) = {HB:.4f} bits")
    print(f"  {'signal':<18}{'MI(bits)':>10}{'MI/H(B)':>10}")
    for l in range(3):
        print(f"  MI(B; ID_{LEVEL_NAMES[l]:<7}){mi_lvl[l]:>10.4f}{mi_lvl[l]/HB:>10.3f}")
    print(f"  {'MI(B; b_prev)':<18}{mi_bprev:>10.4f}{mi_bprev/HB:>10.3f}   <- reference bar")
    if mi_lvl[0] < mi_bprev:
        print("  note: MI(B;ID_shared) < MI(B;b_prev) -> SID signal weak in absolute terms;")
        print("        only the CONDITIONAL Bayes test (A.3) can rescue it.")

    # ---- Buy lift table (train buy-rate per shared code, eval traffic) ----
    bar("A.2  Buy lift by shared code (train rate >=200 occ, eval traffic)")
    code_tot = C[0].sum(1)
    global_buy = train_bcount[BUY] / train_bcount.sum()
    mask = code_tot >= 200
    code_buy_rate = np.divide(C[0][:, BUY], code_tot, out=np.zeros(2048), where=code_tot > 0)
    lift = np.divide(code_buy_rate, global_buy, out=np.zeros(2048), where=np.ones(2048, bool))
    ev_traffic = EV[0].sum(1)
    ev_total = ev_traffic.sum()
    print(f"  global buy base rate (train): {global_buy*100:.3f}%   qualifying codes: {int(mask.sum())}/2048")
    print(f"  {'lift':>6}{'#codes':>9}{'eval traffic %':>16}{'mean buy rate':>15}")
    for thr in (2, 5, 10):
        sel = mask & (lift > thr)
        traf = ev_traffic[sel].sum() / ev_total * 100
        mbr = code_buy_rate[sel].mean() * 100 if sel.any() else 0.0
        print(f"  >{thr:>4}x{int(sel.sum()):>9}{traf:>16.2f}{mbr:>15.3f}")

    # ---- Held-out Bayes classifier (fit train, prior-correct, score eval) ----
    def dirichlet(counts):  # counts[...,4] -> P(b|.) with alpha=1
        return (counts + 1.0) / (counts.sum(-1, keepdims=True) + 4.0)

    correction = eval_prior / train_prior  # prior-shift recalibration

    # model (i): P(b | b_prev)
    P_bprev = dirichlet(C_bprev)  # [4,4]
    # model (iii): P(b | ID_shared)
    P_shared = dirichlet(C[0])    # [2048,4]
    # model (ii): P(b | ID_shared, b_prev) with backoff under min-count 50
    cnt_sp = C_shared_prev                       # [4,2048,4]
    tot_sp = cnt_sp.sum(-1)                       # [4,2048]
    P_sp = dirichlet(cnt_sp)                      # [4,2048,4]

    def score(model, correct):
        """Return per-eval-event P(buy|.) for the requested model."""
        if model == "i":
            P = P_bprev[sc_bp]                                  # [M,4]
        elif model == "iii":
            P = P_shared[sc_s0]
        elif model == "ii":
            P = P_sp[sc_bp, sc_s0]                              # [M,4]
            weak = tot_sp[sc_bp, sc_s0] < 50                    # backoff mask
            P[weak] = P_bprev[sc_bp[weak]]
        if correct:
            P = P * correction[None, :]
            P = P / P.sum(1, keepdims=True)
        return P[:, BUY]

    is_buy = (sc_b == BUY).astype(np.int64)

    def max_f1(buy_prob):
        # sweep threshold over quantiles of the positive+negative score range
        thrs = np.unique(np.quantile(buy_prob, np.linspace(0.5, 0.9999, 400)))
        best = (0.0, 0.0, 0.0, 0.0)  # f1, prec, rec, thr
        P_pos = is_buy.sum()
        for thr in thrs:
            pred = buy_prob >= thr
            tp = np.logical_and(pred, is_buy).sum()
            if tp == 0:
                continue
            prec = tp / pred.sum()
            rec = tp / P_pos
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best[0]:
                best = (f1, prec, rec, float(thr))
        return best

    bar("A.3  Held-out buy detection (max-F1 over threshold)")
    print(f"  {'model':<34}{'buy_F1':>8}{'prec':>7}{'rec':>7}{'thr':>9}")
    results = {}
    for name, model in [
        ("(i)   P(b|b_prev)  history-only", "i"),
        ("(ii)  P(b|shared,b_prev)", "ii"),
        ("(iii) P(b|ID_shared)  attribution", "iii"),
    ]:
        for tag, corr in [("uncorrected", False), ("prior-corrected", True)]:
            f1, prec, rec, thr = max_f1(score(model, corr))
            results[(model, corr)] = (f1, prec, rec, thr)
            print(f"  {name+' ['+tag+']':<34}{f1:>8.3f}{prec:>7.3f}{rec:>7.3f}{thr:>9.4f}")

    f1_i = results[("i", True)][0]
    f1_ii = results[("ii", True)][0]
    delta = f1_ii - f1_i
    bar("A  DECISION GATE")
    print(f"  buy_F1: (i) history-only = {f1_i:.3f}   (ii) +SID = {f1_ii:.3f}   Delta = {delta:+.3f}")
    print(f"  reference: current model's focal buy_F1 ceiling ~ 0.34")
    if delta >= 0.05:
        verdict = "GREEN  -> Step 2 (SID-conditioned behavior head) has real signal."
    elif delta >= 0.02:
        verdict = "AMBER  -> run only cheap post-hoc 2b on frozen ckpts before committing to 2a."
    else:
        verdict = "RED    -> SID-conditioned behavior is dead; go to Step 4 / metric reframe (publishable negative)."
    print(f"  VERDICT: {verdict}")
    if f1_ii >= 0.32:
        print(f"  ** note: lookup-table buy_F1 ({f1_ii:.3f}) approaches/beats the learned head (0.34)")
        print("     -> a count table rivals the model; the architecture argument writes itself.")

    # ==================================================================
    # A.4  Modality-additivity of behavior MI (held-out predictive MI)
    #   Plug-in MI on the 2048x1024x1024 joint would overfit (~every triple unique),
    #   so we fit P(B|code) on TRAIN (Dirichlet a=1, >=50 support, hierarchical backoff to
    #   the shared model, prior-corrected) and score cross-entropy on EVAL. Predictive MI =
    #   H(B) - CE. If shared+text+img > shared by a real margin, modality residuals carry
    #   behavior information -- the count-level proof, before any model is trained.
    # ==================================================================
    bar("A.4  Modality additivity - held-out predictive MI (train-fit, eval-scored)")

    def cn(P):  # prior-shift correct (last axis = behavior) + renormalize
        Pc = P * correction
        return Pc / Pc.sum(-1, keepdims=True)

    P_sh = cn(dirichlet(C[0]))  # [2048,4]

    def backoff_pair(T):  # [2048,Kx,4] -> prior-corrected, backoff to shared where <50
        P = cn(dirichlet(T))
        weak = T.sum(-1) < 50  # [2048,Kx]
        P[weak] = P_sh[np.nonzero(weak)[0]]
        return P

    P_st = backoff_pair(T_stext)
    P_si = backoff_pair(T_simg)
    keep_P = cn(dirichlet(keep_tab)) if len(keep_keys) else np.zeros((0, 4))

    def pred_mi(logp):
        return HB - (-logp.mean())  # H(B) - cross-entropy (bits)

    lp_sh = np.log2(P_sh[ev_s0, ev_b] + 1e-12)
    lp_st = np.log2(P_st[ev_s0, ev_s1, ev_b] + 1e-12)
    lp_si = np.log2(P_si[ev_s0, ev_s2, ev_b] + 1e-12)
    # triple: >=50-support triples use their own P, else back off to shared
    ek = ev_s0.astype(np.int64) * K2 + ev_s1.astype(np.int64) * 1024 + ev_s2.astype(np.int64)
    P_tri = P_sh[ev_s0].copy()  # [M_all,4] default = shared backoff
    if len(keep_keys):
        idx = np.clip(np.searchsorted(keep_keys, ek), 0, len(keep_keys) - 1)
        hit = keep_keys[idx] == ek
        P_tri[hit] = keep_P[idx[hit]]
    lp_tri = np.log2(P_tri[np.arange(len(ev_b)), ev_b] + 1e-12)

    mi = {"shared": pred_mi(lp_sh), "shared+text": pred_mi(lp_st),
          "shared+img": pred_mi(lp_si), "shared+text+img": pred_mi(lp_tri)}
    print(f"  H(B)={HB:.4f} bits   (prior-corrected, Dirichlet a=1, backoff->shared @ min50)")
    print(f"  {'code set':<22}{'pred-MI':>9}{'Δ vs shared':>13}")
    print(f"  {'shared':<22}{mi['shared']:>9.4f}{'-':>13}")
    for key in ("shared+text", "shared+img", "shared+text+img"):
        print(f"  {key:<22}{mi[key]:>9.4f}{mi[key]-mi['shared']:>+13.4f}")
    add = mi["shared+text+img"] - mi["shared"]
    if add > 0.0005:
        print(f"  -> modality residuals ADD behavior info: +{add:.4f} bits "
              f"({add/max(mi['shared'],1e-9)*100:.0f}% over shared) — count-level additivity proof.")
    else:
        print("  -> no additive behavior info from the modality residuals.")

    # ==================================================================
    # A.5  Repeat-evidence value (how much of the behavior win is the repeat feature)
    # ==================================================================
    bar("A.5  Repeat-evidence value (count-based)")
    has = lambda bit: ((sc_mask >> bit) & 1).astype(bool)
    base = eval_prior[BUY]
    cats = [
        ("not in history", sc_mask == 0),
        ("pv only", sc_mask == 1),
        ("has fav (no cart)", has(1) & ~has(2)),
        ("has cart", has(2)),
        ("has buy", has(3)),
    ]
    print(f"  next-item prior interaction -> P(buy)  (eval)")
    print(f"  {'category':<20}{'n(eval)':>13}{'P(buy)':>9}{'lift':>7}")
    for name, mk in cats:
        n = int(mk.sum())
        pr = float((sc_b[mk] == BUY).mean()) if n else 0.0
        print(f"  {name:<20}{n:>13,}{pr*100:>8.2f}%{pr/base:>7.1f}")

    # Bayes attribution: buy_F1 with progressively richer features (all prior-corrected).
    Pc_bprev = cn(dirichlet(C_bprev))         # [4,4]
    Pc_B2 = cn(dirichlet(B2)); tot_B2 = B2.sum(-1)         # [16,4,4], [16,4]
    Pc_B3 = cn(dirichlet(B3)); tot_B3 = B3.sum(-1)         # [16,4,2048,4], [16,4,2048]
    # (b1) P(buy | repeat-set, b_prev), backoff -> P(buy | b_prev)
    Pb1 = Pc_B2[sc_mask, sc_bp].copy()
    wk = tot_B2[sc_mask, sc_bp] < 50
    Pb1[wk] = Pc_bprev[sc_bp[wk]]
    # (b2) P(buy | repeat-set, b_prev, ID_shared), backoff -> b1
    Pb2 = Pc_B3[sc_mask, sc_bp, sc_s0].copy()
    wk3 = tot_B3[sc_mask, sc_bp, sc_s0] < 50
    Pb2[wk3] = Pb1[wk3]
    f1_b1 = max_f1(Pb1[:, BUY])[0]
    f1_b2 = max_f1(Pb2[:, BUY])[0]
    print(f"\n  buy detection max-F1 (prior-corrected):")
    print(f"    (i)   b_prev only               {f1_i:.3f}")
    print(f"    (ii)  + ID_shared               {f1_ii:.3f}   (Δ vs i {f1_ii-f1_i:+.3f})")
    print(f"    (b1)  + repeat-set (no SID)      {f1_b1:.3f}   (Δ vs i {f1_b1-f1_i:+.3f})")
    print(f"    (b2)  + repeat-set + ID_shared   {f1_b2:.3f}   (Δ vs b1 {f1_b2-f1_b1:+.3f}  = SID's add on top of repeat)")
    print(f"  attribution: repeat feature contributes {f1_b1-f1_i:+.3f}, SID adds {f1_b2-f1_b1:+.3f} on top;")
    print(f"               SID alone over history was {delta:+.3f} (A.3).")

    return {
        "C_shared": C[0], "C_text": C[1], "C_img": C[2],
        "C_bprev": C_bprev, "C_shared_prev": C_shared_prev,
        "train_prior": train_prior, "eval_prior": eval_prior,
        "mi_levels": mi_lvl, "mi_bprev": mi_bprev, "H_B": HB,
        "bayes_results": results, "delta_buy_f1": delta,
        "pred_mi_additivity": mi,
        "repeat_bayes_f1": {"i": f1_i, "ii": f1_ii, "b1_repeat": f1_b1, "b2_repeat_sid": f1_b2},
    }


# ---------------------------------------------------------------------------
# Part B: per-behavior modality asymmetry
# ---------------------------------------------------------------------------
def _install_src_stub():
    """Allow unpickling the MD-RQ-VAE Lightning ckpt WITHOUT importing GRID's `src` package.

    The ckpt's peripheral objects (config/loss/strategy refs, partials) are pickled under
    `src.*`; we only read the codebook tensors from state_dict. Importing the real `src`
    would pull hydra/rootutils and their import-time side effects (wrong CWD). Instead we
    fabricate a harmless dummy class for any `src.*` symbol the unpickler asks for.
    """
    import importlib.abc
    import importlib.machinery
    import types

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            cls = type(name, (_Dummy,), {})
            setattr(self, name, cls)
            return cls

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "src" or fullname.startswith("src."):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            return _Mod(spec.name)

        def exec_module(self, module):
            pass

    if not any(getattr(f, "_src_stub", False) for f in sys.meta_path):
        f = _Finder()
        f._src_stub = True
        sys.meta_path.insert(0, f)


def load_centroids(ckpt_path):
    import torch

    log(f"loading centroids from {ckpt_path} (mmap) ...")
    _install_src_stub()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = obj["state_dict"] if "state_dict" in obj else obj
    tables = {}
    for k, v in sd.items():
        if k.endswith(".centroids") and hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] == 512:
            tables[k] = v
            log(f"  centroid table: {k}  {tuple(v.shape)}")
    E_f = E_t = E_i = None
    for k, v in tables.items():
        vf = v.float().clone()
        if v.shape[0] == 2048:
            E_f = vf
        elif "text" in k:
            E_t = vf
        elif "image" in k or "img" in k:
            E_i = vf
    assert E_f is not None and E_t is not None and E_i is not None, \
        f"could not identify all 3 codebooks from {list(tables)}"
    return E_f, E_t, E_i


def part_b(args, sids, N, allzero):
    import torch

    bar("PART B - per-behavior modality asymmetry")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {dev}")
    E_f, E_t, E_i = load_centroids(args.ckpt_path)

    s = torch.from_numpy(sids).long().clamp(min=0)  # [3,N]
    def build(view_res):
        v = (E_f[s[0]] + view_res).half()
        v = torch.nn.functional.normalize(v.float(), dim=-1).half()
        return v
    log("building normalized views V_t, V_i, V_shared ...")
    V_t = build(E_t[s[1]])       # [N,512] fp16
    V_i = build(E_i[s[2]])
    V_s = torch.nn.functional.normalize(E_f[s[0]].float(), dim=-1).half()
    valid = torch.from_numpy(~allzero)
    valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(1).numpy()
    log(f"views built; valid items {len(valid_ids):,}")

    # fixed negative pool
    rng = np.random.default_rng(args.seed)
    neg = torch.from_numpy(rng.choice(valid_ids, args.neg_pool, replace=False))
    Neg_t = V_t[neg].to(dev); Neg_i = V_i[neg].to(dev); Neg_s = V_s[neg].to(dev)

    # ---- sample transitions from eval ----
    gt, beh, rep, last, histk = [], [], [], [], []
    k = args.hist_k
    pv_kept = 0
    n_users = 0
    for items, behs in iterate_users(args.data_dir, "evaluation", args.max_users):
        items, behs, _ = clean_sequence(items, behs, N, allzero)
        n_users += 1
        seen = set()
        for t in range(1, len(items)):
            b = int(behs[t])
            if b == 0:  # pv downsample
                if pv_kept >= args.pv_cap:
                    seen.add(int(items[t - 1])); continue
                pv_kept += 1
            gt.append(int(items[t])); beh.append(b)
            rep.append(1 if int(items[t]) in seen else 0)
            last.append(int(items[t - 1]))
            h = items[max(0, t - k):t]
            pad = np.full(k, -1, np.int64); pad[-len(h):] = h
            histk.append(pad)
            seen.add(int(items[t - 1]))
        seen.add(int(items[-1])) if len(items) else None
    gt = np.array(gt); beh = np.array(beh); rep = np.array(rep)
    last = np.array(last); histk = np.stack(histk) if histk else np.zeros((0, k), np.int64)
    log(f"transitions: {len(gt):,}  (pv kept {pv_kept:,})  from {n_users:,} users")
    for b in range(4):
        m = beh == b
        log(f"  {BEHAVIOR_NAMES[b]:<5}: {int(m.sum()):>9,}  repeat {int(rep[m].sum()):>8,}")

    gt_t = torch.from_numpy(gt); last_t = torch.from_numpy(last); histk_t = torch.from_numpy(histk)

    def hist_vecs(V, rep_mode):
        if rep_mode == "last":
            return V[last_t]
        h = histk_t.clone(); valid_h = h >= 0; h[~valid_h] = 0
        g = V[h.reshape(-1)].float().reshape(len(gt), k, -1)
        m = valid_h.unsqueeze(-1).float()
        mean = (g * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(mean, dim=-1).half()

    @torch.no_grad()
    def rank_all(V, Neg, rep_mode):
        H = hist_vecs(V, rep_mode); G = V[gt_t]
        r10 = np.zeros(len(gt), bool); r100 = np.zeros(len(gt), bool); mrr = np.zeros(len(gt))
        bs = 4096
        for i in range(0, len(gt), bs):
            hb = H[i:i+bs].to(dev).float(); gb = G[i:i+bs].to(dev).float()
            gs = (hb * gb).sum(1, keepdim=True)                 # [b,1]
            ns = hb @ Neg.float().t()                           # [b,neg]
            rank = 1 + (ns > gs).sum(1)                         # 1..neg+1
            rank = rank.cpu().numpy()
            sl = slice(i, i+len(rank))
            r10[sl] = rank <= 10; r100[sl] = rank <= 100; mrr[sl] = 1.0 / rank
        return r10, r100, mrr

    # ---- compute rankings for both views + shared, both history modes ----
    metrics = {}
    for rep_mode in ("last", "mean"):
        metrics[rep_mode] = {
            "text": rank_all(V_t, Neg_t, rep_mode),
            "img": rank_all(V_i, Neg_i, rep_mode),
            "shared": rank_all(V_s, Neg_s, rep_mode),
        }
        log(f"ranked view scores ({rep_mode} history)")

    # ---- readout tables + paired bootstrap ----
    def report(rep_mode):
        """Print R@10 by behavior x view for {non-repeat, repeat, all}; return the
        non-repeat per-behavior paired-bootstrap draws of Δ(text-img)."""
        bar(f"B.1  R@10 by behavior x view  ({rep_mode} history)")
        m = metrics[rep_mode]
        rt_all, ri_all, rs_all = m["text"][0], m["img"][0], m["shared"][0]
        nonrep_draws = {}
        for split_name, split_mask in [("non-repeat", rep == 0), ("repeat", rep == 1), ("all", np.ones(len(gt), bool))]:
            print(f"\n  [{split_name}]")
            print(f"  {'behavior':<8}{'R@10 text':>11}{'R@10 img':>11}{'Δ(t-i)':>9}{'95% CI':>18}{'shared':>9}")
            for b in range(4):
                idx = np.nonzero(split_mask & (beh == b))[0]
                nb = len(idx)
                if nb == 0:
                    continue
                rt, ri, rs = rt_all[idx].mean(), ri_all[idx].mean(), rs_all[idx].mean()
                # paired bootstrap: one resample per draw applied to BOTH views.
                draws = np.array([
                    rt_all[idx[s]].mean() - ri_all[idx[s]].mean()
                    for s in (np.random.default_rng(args.seed + r).integers(0, nb, nb)
                              for r in range(args.boot))
                ])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                if split_name == "non-repeat":
                    nonrep_draws[b] = draws
                print(f"  {BEHAVIOR_NAMES[b]:<8}{rt:>11.3f}{ri:>11.3f}{rt-ri:>9.3f}"
                      f"{f'[{lo:+.3f},{hi:+.3f}]':>18}{rs:>9.3f}")
        return nonrep_draws

    draws_by_mode = {rm: report(rm) for rm in ("last", "mean")}

    # ---- interaction Δ_buy - Δ_pv (last-item history, non-repeat) ----
    m = metrics["last"]
    nr = rep == 0
    nd = draws_by_mode["last"]
    inter_mean = inter_lo = inter_hi = None
    if 0 in nd and BUY in nd:
        inter = nd[BUY] - nd[0]  # difference of two independent group bootstraps
        inter_mean = float(inter.mean())
        inter_lo, inter_hi = (float(x) for x in np.percentile(inter, [2.5, 97.5]))

    bar("B  DECISION GATE (last-item history, non-repeat)")
    if inter_mean is not None:
        print(f"  Δ_buy - Δ_pv (R@10 text-advantage) = {inter_mean*100:+.2f} pts   "
              f"95% CI [{inter_lo*100:+.2f}, {inter_hi*100:+.2f}]")
        licensed = (inter_lo > 0 or inter_hi < 0) and abs(inter_mean) * 100 >= 2.0
        if licensed:
            print("  VERDICT: GREEN -> Step 3 (behavior-gated BM-TV) licensed; supervise gate toward measured Δ.")
        else:
            print("  VERDICT: RED  -> gating hypothesis dead (CI includes 0 or |gap| < 2 pts);")
            print("           BM-TV stays shelved, and this retroactively explains the flat ungated BM-TV.")
    else:
        print("  (insufficient buy/pv non-repeat transitions to compute the interaction)")

    # ---- bonus: does shared-only ranking ~ the full views? (e_f dominance test) ----
    for b in (0, BUY):
        idx = np.nonzero(nr & (beh == b))[0]
        if len(idx):
            full = 0.5 * (m["text"][0][idx].mean() + m["img"][0][idx].mean())
            sh = m["shared"][0][idx].mean()
            print(f"  bonus [{BEHAVIOR_NAMES[b]}]: shared-only R@10={sh:.3f} vs full-avg {full:.3f}"
                  f"  ({'shared dominates geometry (explains rerank failure)' if sh >= full - 0.01 else 'residuals add signal'})")

    return {
        "beh": beh, "rep": rep,
        "metrics_last": metrics["last"], "metrics_mean": metrics["mean"],
        "nonrepeat_draws_last": draws_by_mode["last"],
        "interaction_last_nonrepeat": (inter_mean, inter_lo, inter_hi),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["mi", "asym", "factor", "both"], default="both")
    ap.add_argument("--max_users", type=int, default=None)
    ap.add_argument("--data_dir", default="/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop")
    ap.add_argument("--sid_path",
                    default="/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt")
    ap.add_argument("--ckpt_path",
                    default="/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt")
    ap.add_argument("--out_artifacts", default="/scratch/yw8866/rec-tmall/coupling_artifacts.pt")
    ap.add_argument("--pv_cap", type=int, default=500000)
    ap.add_argument("--neg_pool", type=int, default=8192)
    ap.add_argument("--hist_k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full_tuple", action="store_true")
    args = ap.parse_args()

    bar("SID <-> behavior coupling probe")
    print(f"  data_dir : {args.data_dir}")
    print(f"  sid_path : {args.sid_path}")
    print(f"  part={args.part}  max_users={args.max_users}")
    is_natural = args.data_dir.rstrip("/").endswith("tfrecords_mb")
    print(f"  prior correction: {'~identity (natural build)' if is_natural else 'ACTIVE (userdrop train prior != eval)'}")

    sids, N, allzero = load_sids(args.sid_path)
    artifacts = {}
    if args.part == "factor":
        artifacts["part_g"] = part_g(args, sids, N, allzero)
    if args.part in ("mi", "both"):
        artifacts["part_a"] = part_a(args, sids, N, allzero)
    if args.part in ("asym", "both"):
        artifacts["part_b"] = part_b(args, sids, N, allzero)

    import torch
    torch.save(artifacts, args.out_artifacts)
    log(f"artifacts saved: {args.out_artifacts}")


if __name__ == "__main__":
    sys.exit(main())