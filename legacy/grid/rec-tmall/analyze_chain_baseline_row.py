"""Chain RQ-VAE (multi-behavior TIGER) baseline row — computed OFFLINE from chain pool-audit dumps.

FAST PATH for the results-table row. Reads one or more chain pool dumps (schema: per-behavior pools
pool_f/t/i + pool_sidscore, beh_logp, gt_*, gt_in_catalog, meta.fwd_states_per_item) and computes,
per dump, the SAME conventions as the project evaluator (MultiBehaviorSIDRetrievalEvaluator):

  t3 (free / union): joint behavior+SID match, scored on the behavior-marginalized union pool at
                     w_b=1 (score = w_b*logP(b) + SID_score).  recall/ndcg + behavior accuracy.
  t2 (conditioned):  routed to the GROUND-TRUTH behavior branch (NOT the union), scored by the
                     SID score alone (the pool_sidscore excludes the behavior term).  recall/ndcg.

recall@K = 1[GT in top-K]  (single relevant -> total_relevant.min(K)=1, matches Recall metric).
ndcg@K   = 1/log2(rank+2) if GT in top-K else 0  (single relevant -> ideal_dcg=1, matches NDCG).
behavior P/R/F1 = over argmax_b logP(b) vs GT behavior (matches BehaviorClassMetric).

CALIBER WARNING (printed): these dump numbers are CATALOG-CONSTRAINED sequential-beam pools, a
DIFFERENT decoder family and budget from the free-beam eval that produced the other table rows.
They are the point comparable to the MFD pool (matched_compute), NOT a drop-in for the free-beam
baseline row. The authoritative caliber-aligned row is the separate free-beam eval; the two are
reported side by side and any gap is a real catalog-constraint effect, to be surfaced not averaged.

SPLIT UNIT TEST (hard): asserts total==83,005 and per-behavior pv/fav/cart/buy ==
77,833/2,042/1,554/1,576. A mismatch means the dump silently subsampled (the 45k-pv bug) -> the
dump's numbers are SKIPPED rather than emitted.
"""

import argparse
import math

import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
EXPECT_TOTAL = 83005
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def bar(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cols = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in d["cols"].items()}
    return cols, d["meta"]


def build_cand(c):
    N, nb, Kp = c["pool_f"].shape
    flat = lambda x: x.reshape(N, nb * Kp)
    return {
        "branch": torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1).contiguous(),
        "f": flat(c["pool_f"]).long(), "t": flat(c["pool_t"]).long(), "i": flat(c["pool_i"]).long(),
        "sid": flat(c["pool_sidscore"]), "valid": flat(c["pool_sidscore"]) > -1e30,
    }, N, nb, Kp


def best_rank(score, match, eligible):
    """rank of the best matching, eligible candidate by descending score; -1 if none."""
    score = score.masked_fill(~eligible, float("-inf"))
    m = torch.where(match & eligible, score, torch.full_like(score, float("-inf")))
    best = m.max(1).values
    has = best > -1e30
    rank = (score > best.unsqueeze(1)).sum(1)
    return torch.where(has, rank, torch.full_like(rank, -1)), has


def recall_ndcg(rank, has, k, denom):
    hit = has & (rank >= 0) & (rank < k)
    r = hit[denom].float().mean().item()
    inv = 1.0 / torch.log2((rank.clamp(min=0) + 2).float())
    ndcg = torch.where(hit, inv, torch.zeros_like(inv))[denom].mean().item()
    return r, ndcg


def check_split(c):
    gt_b = c["gt_b"].long()
    N = gt_b.shape[0]
    per = {b: int((gt_b == b).sum()) for b in range(4)}
    ok = (N == EXPECT_TOTAL) and all(per[b] == EXPECT_PERBEH[b] for b in range(4))
    print(f"  SPLIT UNIT TEST: total={N:,} (expect {EXPECT_TOTAL:,})  "
          + "  ".join(f"{BEH[b]}={per[b]:,}(exp {EXPECT_PERBEH[b]:,})" for b in range(4)))
    if not ok:
        print("  ** SPLIT MISMATCH — dump silently subsampled; numbers SKIPPED for this dump. **")
    return ok


def behavior_head(c):
    beh_logp, gt_b = c["beh_logp"], c["gt_b"].long()
    pred = beh_logp.argmax(1)
    acc = (pred == gt_b).float().mean().item()
    rows = {}
    for b in range(4):
        tp = int(((pred == b) & (gt_b == b)).sum())
        fp = int(((pred == b) & (gt_b != b)).sum())
        fn = int(((pred != b) & (gt_b == b)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows[b] = (p, r, f1)
    return acc, rows


def analyze(label, path, wb):
    c, meta = load(path)
    nb = meta["nb"]
    fs = meta.get("fwd_states_per_item")
    if fs is None and "beam" in meta:
        fs = 1 + 3 + nb * (1 + 2 * int(meta["beam"]))
    bar(f"{label}   dump={path}   decoder-states/item={fs}   beam={meta.get('beam')}  K_pool={meta.get('Kpool')}")
    if not check_split(c):
        return None
    gt_b = c["gt_b"].long()
    gt_f, gt_t, gt_i = c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    cand, N, _, Kp = build_cand(c)
    sid_match = (cand["f"] == gt_f[:, None]) & (cand["t"] == gt_t[:, None]) & (cand["i"] == gt_i[:, None])
    branch_gt = cand["branch"] == gt_b[:, None]

    # --- t3: behavior-marginalized union, score = wb*logP(b) + SID_score ---
    s3 = (wb * torch.gather(c["beh_logp"], 1, cand["branch"]) + cand["sid"]).masked_fill(~cand["valid"], float("-inf"))
    r3, h3 = best_rank(s3, sid_match & branch_gt, cand["valid"])

    # --- t2: routed to GT behavior branch, score = SID_score only ---
    elig2 = branch_gt & cand["valid"]
    r2, h2 = best_rank(cand["sid"], sid_match, elig2)

    allm = torch.ones(N, dtype=torch.bool)
    groups = [("ALL", allm)] + [(BEH[b], gt_b == b) for b in range(nb)]

    print(f"\n  {'metric':<20}" + "".join(f"{g[0]:>10}" for g in groups))
    def line(name, fn):
        print(f"  {name:<20}" + "".join(f"{fn(g[1]):>10.4f}" for g in groups))

    for k in (5, 10, 20):
        line(f"t3_recall@{k}", lambda m, k=k: recall_ndcg(r3, h3, k, m)[0])
    for k in (5, 10):
        line(f"t3_ndcg@{k}", lambda m, k=k: recall_ndcg(r3, h3, k, m)[1])
    for k in (5, 10, 20):
        line(f"t2_recall@{k}", lambda m, k=k: recall_ndcg(r2, h2, k, m)[0])
    for k in (5, 10):
        line(f"t2_ndcg@{k}", lambda m, k=k: recall_ndcg(r2, h2, k, m)[1])

    acc, rows = behavior_head(c)
    print(f"\n  t3_behavior_accuracy = {acc:.4f}")
    print(f"  {'behavior head':<20}{'precision':>12}{'recall':>10}{'f1':>10}")
    for b in range(nb):
        p, r, f1 = rows[b]
        print(f"  {BEH[b]:<20}{p:>12.4f}{r:>10.4f}{f1:>10.4f}")

    res = {"label": label, "fwd": fs, "beam": meta.get("beam"),
           "t3r10": recall_ndcg(r3, h3, 10, allm)[0], "t2r10": recall_ndcg(r2, h2, 10, allm)[0],
           "t3n10": recall_ndcg(r3, h3, 10, allm)[1], "t2n10": recall_ndcg(r2, h2, 10, allm)[1],
           "beh_acc": acc}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--ref_t3r10", type=float, default=0.2127,
                    help="pool-comparison chain t3@10 (beam 500) — cross-check")
    args = ap.parse_args()

    bar("CHAIN RQ-VAE (multi-behavior TIGER) — dump-based row  [CATALOG-CONSTRAINED pool]")
    print("  conventions: t3 = union / behavior-marginalized (w_b=1); t2 = routed to GT behavior;")
    print("  recall@K = 1[GT in top-K]; ndcg@K = 1/log2(rank+2). SAME as the project evaluator.")
    print("  NOTE: these are catalog-constrained sequential-beam pools -> comparable to the MFD POOL")
    print("        (matched_compute), NOT a drop-in for the free-beam baseline row. See the separate")
    print("        free-beam eval in the same job; any t3@10 gap vs the dump is a catalog-constraint effect.")

    results = []
    for label, path in args.point:
        r = analyze(label, path, args.wb)
        if r is not None:
            results.append(r)

    if results:
        bar("SUMMARY (dump-based, catalog-constrained)")
        print(f"  {'point':<16}{'states':>8}{'t2@10':>9}{'t2ndcg':>9}{'t3@10':>9}{'t3ndcg':>9}{'beh_acc':>9}")
        for r in results:
            print(f"  {r['label']:<16}{str(r['fwd']):>8}{r['t2r10']:>9.4f}{r['t2n10']:>9.4f}"
                  f"{r['t3r10']:>9.4f}{r['t3n10']:>9.4f}{r['beh_acc']:>9.4f}")
        # cross-check any beam-500 / 4008-state dump against the pool-comparison reference
        for r in results:
            if r["beam"] == 500 or r["fwd"] == 4008:
                d = r["t3r10"] - args.ref_t3r10
                verdict = "AGREES" if abs(d) < 0.003 else "** DISCREPANCY — surface, do not average **"
                print(f"\n  cross-check {r['label']}: t3@10={r['t3r10']:.4f} vs pool-comparison ref "
                      f"{args.ref_t3r10:.4f}  (diff {d:+.4f})  {verdict}")


if __name__ == "__main__":
    main()