"""Extended-relevance re-evaluation of the chain vs MFD candidate pools (offline; no regeneration).

The test set is last-item leave-one-out, so NO interactions exist AFTER the target. The extended
relevance set is therefore the user's LAST N interactions (target + N-1 preceding), deduplicated on
the (behavior, l0, l1, l2) 4-tuple -- a "recent-interest cluster" probe, NOT literal future (the
N-1 extra items are recent past, some inside the model's history window; this rewards re-surfacing
recent interests, which BOTH arms can do equally). N=1 == the single next item -> reproduces the
existing single-GT numbers exactly (asserted as a correctness check).

Pipeline: read the test tfrecords -> per user_id the full time-ordered item sequence + behaviors;
map each item id -> that arm's SID (MD-RQ-VAE for MFD, tiger_1024 for chain) via semantic_ids.pt;
take the last N items -> dedup 4-tuples; join to the dump events by user_id. Metrics per arm, per
N in {1,3,5,10,20}, per behavior (event's target behavior).
"""

import argparse
import glob
import os

import numpy as np
import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
NS = [1, 3, 5, 10, 20]


# ---------------------------------------------------------------------------
def read_test_sequences(tfdir):
    """{user_id -> (item_ids int64[L], behaviors int64[L])} from the testing tfrecords."""
    import tensorflow as tf
    files = sorted(glob.glob(os.path.join(tfdir, "*.tfrecord.gz")))
    if not files:
        files = [f for f in sorted(glob.glob(os.path.join(tfdir, "*"))) if os.path.isfile(f)]
    feat = {"user_id": tf.io.FixedLenFeature([1], tf.int64),
            "sequence_data": tf.io.VarLenFeature(tf.int64),
            "behavior_data": tf.io.VarLenFeature(tf.int64)}
    uid2seq = {}
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")   # builder writes GZIP .tfrecord.gz
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feat)
        uid = int(ex["user_id"].numpy()[0])
        seq = tf.sparse.to_dense(ex["sequence_data"]).numpy().astype(np.int64)
        beh = tf.sparse.to_dense(ex["behavior_data"]).numpy().astype(np.int64)
        uid2seq[uid] = (seq, beh)
    print(f"[tfrecords] {tfdir}: {len(uid2seq):,} users, {len(files)} files", flush=True)
    return uid2seq


def load_sid(path):
    t = torch.load(path, map_location="cpu", weights_only=False).long()   # (levels, N_items)
    if t.shape[0] > t.shape[1]:
        raise ValueError(f"{path}: shape {tuple(t.shape)} looks transposed")
    return t


def build_relevance(uids, uid2seq, sid, nmax):
    """For each event (aligned to `uids`), the last-`nmax` items' (behavior, l0,l1,l2) with a valid
    mask. Returns dict N -> (b,l0,l1,l2,mask) each (Nev, R_N) int64, R_N<=N (after dedup)."""
    Nev = len(uids)
    Ncat = sid.shape[1]
    # gather last-nmax raw items + behaviors per event (pad short seqs)
    lastb = np.full((Nev, nmax), -1, np.int64)
    lasti = np.full((Nev, nmax), -1, np.int64)
    miss = 0
    for e, u in enumerate(uids):
        rec = uid2seq.get(int(u))
        if rec is None:
            miss += 1
            continue
        seq, beh = rec
        L = seq.shape[0]
        k = min(nmax, L)
        lasti[e, nmax - k:] = seq[L - k:]              # oldest..newest within the window (newest last)
        lastb[e, nmax - k:] = beh[L - k:]
    if miss:
        print(f"[relevance] WARNING: {miss:,}/{Nev:,} events had no matching user_id in tfrecords", flush=True)
    # map item id -> SID (invalid item -> -1)
    lasti_t = torch.from_numpy(lasti)
    valid_item = (lasti_t >= 0) & (lasti_t < Ncat)
    idx = lasti_t.clamp(min=0)
    l0 = sid[0][idx]; l1 = sid[1][idx]; l2 = sid[2][idx]
    b = torch.from_numpy(lastb)
    l0 = torch.where(valid_item, l0, torch.full_like(l0, -1))
    l1 = torch.where(valid_item, l1, torch.full_like(l1, -1))
    l2 = torch.where(valid_item, l2, torch.full_like(l2, -1))
    out = {}
    for N in NS:
        # take the last N columns (most recent N), dedup per event on the 4-tuple
        bb = b[:, nmax - N:]; a0 = l0[:, nmax - N:]; a1 = l1[:, nmax - N:]; a2 = l2[:, nmax - N:]
        m = (a0 >= 0) & (bb >= 0)
        out[N] = _dedup(bb, a0, a1, a2, m, N)
    return out


def _dedup(bb, a0, a1, a2, m, N):
    """Per-row dedup of (b,l0,l1,l2) among valid entries; returns padded (Nev,N) + mask."""
    Nev = bb.shape[0]
    ob = torch.full((Nev, N), -1, dtype=torch.long)
    o0 = ob.clone(); o1 = ob.clone(); o2 = ob.clone(); om = torch.zeros(Nev, N, dtype=torch.bool)
    for e in range(Nev):
        seen = set(); w = 0
        for j in range(bb.shape[1]):
            if not bool(m[e, j]):
                continue
            key = (int(bb[e, j]), int(a0[e, j]), int(a1[e, j]), int(a2[e, j]))
            if key in seen:
                continue
            seen.add(key)
            ob[e, w], o0[e, w], o1[e, w], o2[e, w], om[e, w] = key[0], key[1], key[2], key[3], True
            w += 1
    return ob, o0, o1, o2, om


# ---------------------------------------------------------------------------
def build_pool(c):
    N, nb, Kp = c["pool_f"].shape
    flat = lambda x: x.reshape(N, nb * Kp)
    branch = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1).contiguous()
    f = flat(c["pool_f"]).long(); t = flat(c["pool_t"]).long(); i = flat(c["pool_i"]).long()
    sid = flat(c["pool_sidscore"]); valid = sid > -1e30
    beh_of = torch.gather(c["beh_logp"], 1, branch)
    score = (beh_of + sid).masked_fill(~valid, float("-inf"))     # union score, w_b=1
    return branch, f, t, i, valid, score, nb, Kp


@torch.no_grad()
def rel_ranks_for_N(pool, rel, device, chunk=4000):
    """(Nev, R) 0-based rank of each relevance 4-tuple in the union pool (-1 if not present)."""
    branch, f, t, i, valid, score, nb, Kp = pool
    rb, r0, r1, r2, rm = rel
    Nev, R = rb.shape
    out = torch.full((Nev, R), -1, dtype=torch.long)
    for s in range(0, Nev, chunk):
        e = min(Nev, s + chunk)
        br = branch[s:e].to(device); ff = f[s:e].to(device); tt = t[s:e].to(device); ii = i[s:e].to(device)
        va = valid[s:e].to(device); sc = score[s:e].to(device)
        sc_v = sc.masked_fill(~va, float("-inf"))
        for r in range(R):
            mk = rm[s:e, r].to(device)
            if not bool(mk.any()):
                continue
            match = (br == rb[s:e, r, None].to(device)) & (ff == r0[s:e, r, None].to(device)) & \
                    (tt == r1[s:e, r, None].to(device)) & (ii == r2[s:e, r, None].to(device)) & va
            has = match.any(1) & mk
            best = torch.where(match, sc, torch.full_like(sc, float("-inf"))).max(1).values
            rk = (sc_v > best.unsqueeze(1)).sum(1)
            rk = torch.where(has, rk, torch.full_like(rk, -1))
            out[s:e, r] = rk.cpu()
    return out


def report_arm(tag, c, uid2seq, sid, device, ref_r10):
    print("\n" + "#" * 84 + f"\n# ARM: {tag}\n" + "#" * 84)
    uids = c["user_id"].long().numpy()
    gt_b = c["gt_b"].long()
    Nev = gt_b.shape[0]
    pool = build_pool(c)
    rels = build_relevance(uids, uid2seq, sid, max(NS))

    # ---- N=1 correctness: relevance == single GT, and recall@10 reproduces reference ----
    rb, r0, r1, r2, rm = rels[1]
    ok = (rm[:, -1] & (rb[:, -1] == gt_b) & (r0[:, -1] == c["gt_f"].long()) &
          (r1[:, -1] == c["gt_t"].long()) & (r2[:, -1] == c["gt_i"].long()))
    frac_ok = ok.float().mean().item()
    print(f"  [N=1 join check] last item's (beh,SID) == dump GT for {frac_ok:.4f} of events "
          f"(expect ~1.0; low => user_id join / SID map mismatch)")
    assert frac_ok > 0.97, f"N=1 join check FAILED ({frac_ok:.3f}): tfrecord<->dump alignment is wrong for {tag}"

    hdr = "  {:<22}".format("") + "".join(f"{n:>9}" for n in ["ALL"] + [BEH[b] for b in range(pool[6])])
    for N in NS:
        rel = rels[N]
        rr = rel_ranks_for_N(pool, rel, device)             # (Nev, R)
        rm_N = rel[4]
        setsize = rm_N.sum(1)                                # unique relevant per event
        covered = ((rr >= 0) & rm_N).any(1)
        hit10 = ((rr >= 0) & (rr < 10) & rm_N).any(1)
        hit20 = ((rr >= 0) & (rr < 20) & rm_N).any(1)
        rec10_num = ((rr >= 0) & (rr < 10) & rm_N).sum(1).float()
        recall10 = rec10_num / setsize.clamp(min=1).float()  # fraction of the set retrieved @10
        print("\n  " + "=" * 78)
        print(f"  N={N}   (relevance = last {N} interactions, dedup 4-tuples)")
        print(hdr)
        def row(name, fn):
            cells = [f"{fn(torch.ones(Nev, dtype=torch.bool)):>9.4f}"]
            for b in range(pool[6]):
                cells.append(f"{fn(gt_b == b):>9.4f}")
            print(f"  {name:<22}" + "".join(cells))
        row("mean set size", lambda m: setsize[m].float().mean().item())
        row("coverage@pool", lambda m: covered[m].float().mean().item())
        row("hit@10", lambda m: hit10[m].float().mean().item())
        row("hit@20", lambda m: hit20[m].float().mean().item())
        row("recall@10 (set denom)", lambda m: recall10[m].mean().item())
        def p_hit_cov(m):
            mm = m & covered
            return hit10[mm].float().mean().item() if mm.any() else 0.0
        row("P(hit@10 | covered)", p_hit_cov)
        if N == 1:
            r10_all = hit10.float().mean().item()
            print(f"  [N=1 check] recall@10={r10_all:.4f}  vs reference {ref_r10:.4f}  "
                  f"(diff {r10_all - ref_r10:+.4f})")
            assert abs(r10_all - ref_r10) < 0.02, \
                f"N=1 recall@10 {r10_all:.4f} does NOT reproduce single-GT reference {ref_r10:.4f}"
        # out-of-top10 rank distribution for in-pool-but-not-top10 relevant items
        inpool_not10 = (rr >= 10) & (rr < pool[7] * pool[6]) & rm_N
        rr_deep = rr[inpool_not10].float()
        if rr_deep.numel():
            qs = torch.tensor([0.25, 0.5, 0.75, 0.9])
            qv = torch.quantile(rr_deep, qs)
            print(f"  in-pool-not-top10 relevant items: n={int(inpool_not10.sum()):,}  "
                  f"rank p25/p50/p75/p90 = {qv[0]:.0f}/{qv[1]:.0f}/{qv[2]:.0f}/{qv[3]:.0f}  "
                  f"(mean {rr_deep.mean():.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_dump", default="/scratch/yw8866/rec-tmall/mfd_pool_cmp_dump.pt")
    ap.add_argument("--chain_dump", default="/scratch/yw8866/rec-tmall/chain_pool_cmp_dump.pt")
    ap.add_argument("--test_dir", default="/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop/testing")
    ap.add_argument("--md_sid", default="/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt")
    ap.add_argument("--tiger_sid", default="/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt")
    ap.add_argument("--ref_mfd_r10", type=float, default=0.2041)
    ap.add_argument("--ref_chain_r10", type=float, default=0.2127)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 84)
    print("EXTENDED RELEVANCE (last-N interactions) -- chain vs MFD pools, offline")
    print("  NOTE: last-item test => no post-target interactions. Relevance = user's last N items")
    print("        (target + recent past), dedup 4-tuples. N=1 reproduces single-GT (asserted).")
    print("=" * 84)

    uid2seq = read_test_sequences(args.test_dir)
    md_sid = load_sid(args.md_sid)
    tg_sid = load_sid(args.tiger_sid)

    cm = torch.load(args.mfd_dump, map_location="cpu", weights_only=False)["cols"]
    cm = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in cm.items()}
    cc = torch.load(args.chain_dump, map_location="cpu", weights_only=False)["cols"]
    cc = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in cc.items()}

    report_arm("MFD + PMQ + Q_f  (MD-RQ-VAE)", cm, uid2seq, md_sid, device, args.ref_mfd_r10)
    report_arm("chain RQ-VAE (tiger_1024)", cc, uid2seq, tg_sid, device, args.ref_chain_r10)

    print("\n" + "=" * 84)
    print("READ-OUT: compare coverage@pool / hit@10 / hit@20 as N grows, per behavior. If MFD")
    print("  overtakes chain as N grows, factorized decoding covers a broader recent-interest slice")
    print("  and single-GT understates it. If chain stays level/ahead at every N, the diversity")
    print("  claim has no downstream payoff here -- rest the paper on coverage + decoding cost.")
    print("  (Full sweep reported; no N selected post hoc.)")


if __name__ == "__main__":
    main()