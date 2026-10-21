"""Why do MFD-ONLY ground truths sit at rank ~430? Which factor of the score is weak for them?

Every lever on the reranker and on pool selection is now closed (round 1: four mechanisms null; tau
screen: best +0.0018 pre-reranker). What remains is changing the generator, and a generator-side build
must target the right factor. The score is

    logP(b) + logP(f|b) + logP(t|f,b) + logP(i|f,b)

and each dump already carries, per test event, the TEACHER-FORCED rank and log-prob of the GT's own
factors inside the GT behavior: f_star_rank/f_star_logp (shared code among 2048), tf_t_rank/tf_t_logp
(text among 1024, given the GT f), tf_i_rank/tf_i_logp (image among 1024, given the GT f).

This audit splits events by pool membership (BOTH / MFD-ONLY / CHAIN-ONLY / NEITHER) and, inside
MFD-ONLY, by whether the reranker got them into the top 10, and compares the three factors. The
factor that separates MFD-ONLY from BOTH is the one a generator change has to fix.

Reads only the small per-event columns via torch.load(mmap=True), so the 27 GB dump is never paged in.
"""

import argparse
import os

import numpy as np
import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}


def sorted_by_uid(r):
    o = np.argsort(r["uid"])
    return {k: (v[o] if isinstance(v, np.ndarray) and v.shape[:1] == r["uid"].shape else v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt")
    ap.add_argument("--mfd", default="/scratch/yw8866/rec-tmall/seedvar/cache/MFD__seed0.pt")
    ap.add_argument("--ch", default="/scratch/yw8866/rec-tmall/seedvar/cache/CH__seed0.pt")
    ap.add_argument("--out", default="/scratch/yw8866/rec-tmall/component_audit.txt")
    args = ap.parse_args()
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    d = torch.load(args.dump, map_location="cpu", weights_only=False, mmap=True)
    c = d["cols"]
    uid = c["user_id"].long().numpy()
    o = np.argsort(uid)
    cols = {k: c[k].float().numpy()[o] for k in
            ("f_star_rank", "tf_t_rank", "tf_i_rank", "f_star_logp", "tf_t_logp", "tf_i_logp")}
    gtb = c["gt_b"].long().numpy()[o]
    beh_gt = torch.gather(c["beh_logp"].float(), 1, c["gt_b"].long().view(-1, 1)).squeeze(1).numpy()[o]
    m = sorted_by_uid(torch.load(args.mfd, map_location="cpu", weights_only=False))
    ch = sorted_by_uid(torch.load(args.ch, map_location="cpu", weights_only=False))
    assert np.array_equal(np.sort(uid), m["uid"]) and np.array_equal(m["uid"], ch["uid"]), "user_id join failed"
    assert np.array_equal(gtb, m["gt_b"]), "gt behavior disagrees between dump and reranker result"
    N = len(uid)
    assert N == 83005

    mc, cc = m["has_gt"].astype(bool), ch["has_gt"].astype(bool)
    hit = m["hit_after"] > 0
    pops = {"BOTH": mc & cc, "MFD-ONLY": mc & ~cc, "  MFD-ONLY hit": mc & ~cc & hit,
            "  MFD-ONLY miss": mc & ~cc & ~hit, "CHAIN-ONLY": cc & ~mc, "NEITHER": ~mc & ~cc}

    log("=" * 108)
    log("=== COMPONENT AUDIT: which factor of the MFD score is weak for the ground truths MFD alone covers? ===")
    log("=" * 108)
    log("  teacher-forced GT factors inside the GT behavior; ranks are 0-based (f among 2048, t and i among 1024)")
    log("  medians over the population; pv only unless stated (MFD-ONLY is 99.4% pv)\n")
    hdr = (f"  {'population':<17}{'n':>8}{'f* rank':>9}{'t* rank':>9}{'i* rank':>9}"
           f"{'logP(f*)':>10}{'logP(t*|f)':>12}{'logP(i*|f)':>12}{'logP(b*)':>10}{'sum':>9}")
    log(hdr)
    pv = gtb == 0
    for nm, p in pops.items():
        q = p & pv
        if q.sum() == 0:
            continue
        med = lambda k: float(np.median(cols[k][q]))
        s = cols["f_star_logp"][q] + cols["tf_t_logp"][q] + cols["tf_i_logp"][q] + beh_gt[q]
        log(f"  {nm:<17}{int(q.sum()):>8,}{med('f_star_rank'):>9,.0f}{med('tf_t_rank'):>9,.0f}{med('tf_i_rank'):>9,.0f}"
            f"{med('f_star_logp'):>10.2f}{med('tf_t_logp'):>12.2f}{med('tf_i_logp'):>12.2f}"
            f"{float(np.median(beh_gt[q])):>10.2f}{float(np.median(s)):>9.2f}")

    b, own = pops["BOTH"] & pv, pops["MFD-ONLY"] & pv
    log("\n  --- the gap: MFD-ONLY minus BOTH, pv, in nats (negative = MFD-ONLY is weaker on that factor) ---")
    gaps = {}
    for k, lab in (("f_star_logp", "shared code  logP(f*)"), ("tf_t_logp", "text         logP(t*|f)"),
                   ("tf_i_logp", "image        logP(i*|f)")):
        g = float(np.median(cols[k][own]) - np.median(cols[k][b])); gaps[lab] = g
        log(f"    {lab}: {g:+.2f}")
    worst = min(gaps, key=gaps.get)
    tot = sum(gaps.values())
    log(f"    total {tot:+.2f} nats; largest share: {worst.split()[0]} ({gaps[worst]/tot*100 if tot else 0:.0f}% of it)")

    log("\n  --- inside MFD-ONLY: what separates the 6.5% the reranker rescues from the rest? ---")
    h, x = pops["  MFD-ONLY hit"] & pv, pops["  MFD-ONLY miss"] & pv
    for k, lab in (("f_star_logp", "logP(f*)"), ("tf_t_logp", "logP(t*|f)"), ("tf_i_logp", "logP(i*|f)"),
                   ("f_star_rank", "f* rank"), ("tf_t_rank", "t* rank"), ("tf_i_rank", "i* rank")):
        log(f"    {lab:<12} rescued {np.median(cols[k][h]):>9.2f}   missed {np.median(cols[k][x]):>9.2f}")

    log("\n  --- share of MFD-ONLY pv GTs whose factor is individually TOP-ranked (rank < 10) ---")
    for k, lab in (("f_star_rank", "f*"), ("tf_t_rank", "t*|f"), ("tf_i_rank", "i*|f")):
        log(f"    {lab:<5} top-10 in {float((cols[k][own] < 10).mean()):.3f} of MFD-ONLY   vs "
            f"{float((cols[k][b] < 10).mean()):.3f} of BOTH")

    log("\n  --- reading ---")
    log("  If the shared-code factor carries most of the gap, a generator change must improve P(f|history, b)")
    log("  for pv; the modality heads are already doing their job on these items. If a modality factor")
    log("  carries it, the fix is in that head. If all three are uniformly weak, these items are simply")
    log("  hard pv next-items and no factor-specific change will move them.")
    open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
