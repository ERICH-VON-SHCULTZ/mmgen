"""Does changing the pool SELECTION score help, where changing the ORDERING did not?

Round 1 closed the reranking avenue: four mechanisms (features / score form / weighting / objective)
all failed to promote MFD-ONLY ground truths, whose median rank is ~378 and stays there. But every
experiment so far -- learned fusion, calibration, MADR, BPF -- rescored a FIXED pool; coverage@pool was
an asserted invariant. The selection score itself has never been varied.

    score(f,t,i | b) = w_b*logP(b) + logP(f|b) + (1/tau)*[ logP(t|f,b) + logP(i|f,b) ]

tau is frozen at 1.0 (exact product-of-experts) and only ever swept at inference for ordering. Lowering
tau amplifies the two modality terms relative to the shared-code term, which is exactly the MFD-ONLY
signature: a ground truth whose f-root is weak but whose t and i are sharp. Crucially this changes WHICH
candidates are selected, so the ground truths it admits arrive at GOOD ranks -- unlike K_pool widening,
which appended low-scoring candidates deep in the pool and converted at only 2.3%.

This screen is offline and exact on the stored pool: the p3k dumps keep the top-5,000 per behavior with
their component log-probs, so any tau can be re-scored and re-selected within them.

CAVEAT, stated up front: the stored 5,000 were themselves chosen by the tau=1 score, so re-selecting
inside them is biased toward tau=1 and UNDERSTATES what full-catalog selection at tau!=1 would find.
Every number here is therefore a lower bound on the tau effect. A tau that wins offline justifies a
re-dump (full catalog selection); a tau that loses offline cannot be rescued by one.

Reports per tau: coverage@2000 (the operating K_pool), base t3@10 (rank<10 over the union, no reranker),
the GT rank distribution, and -- the deciding column -- the MFD-ONLY population under that tau, joined
against chain's fixed pool membership by user_id.
"""

import argparse
import json
import os

import numpy as np
import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
EXPECT_PERBEH = {0: 77833, 1: 2042, 2: 1554, 3: 1576}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="p3k_test.pt (top-5000 stored)")
    ap.add_argument("--chain_result", required=True, help="cached CH per-event result, for MFD-ONLY")
    ap.add_argument("--taus", default="0.3,0.5,0.7,0.85,1.0,1.25,1.5,2.0")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--kpool", type=int, default=2000, help="operating K_pool to re-select to")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/tau_exp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    N, nb, Kp = c["pool_f"].shape
    log("=" * 100)
    log("=== TAU: does changing pool SELECTION help where changing ORDERING did not? ===")
    log("=" * 100)
    log(f"  dump {os.path.basename(args.dump)}  N={N:,}  nb={nb}  stored K_pool={Kp}  "
        f"re-selecting to K_pool={args.kpool}  wide_catalog={meta.get('wide_catalog')}  states={meta.get('fwd_states_per_item')}")
    per = {b: int((c["gt_b"].long() == b).sum()) for b in range(nb)}
    assert N == 83005 and per == EXPECT_PERBEH, f"split shape wrong: N={N} per={per}"
    assert meta.get("fwd_states_per_item") == 808, "decoder compute is not 808 states"
    log(f"  split asserted: 83,005 events, " + " ".join(f"{BEH[b]}={per[b]:,}" for b in range(nb)))
    log("  CAVEAT: the stored 5,000/behavior were selected by the tau=1 score, so these numbers are a")
    log("  LOWER BOUND on what full-catalog selection at tau!=1 would give.")

    gt_b = c["gt_b"].long(); gt_f = c["gt_f"].long(); gt_t = c["gt_t"].long(); gt_i = c["gt_i"].long()
    beh = c["beh_logp"].float()
    ch = torch.load(args.chain_result, map_location="cpu", weights_only=False)
    o = np.argsort(ch["uid"]); ch_cov_sorted = ch["has_gt"].astype(bool)[o]
    uid = c["user_id"].long().numpy()
    # map chain coverage onto this dump's row order via user_id
    order = np.argsort(uid)
    ch_cov = np.empty(N, dtype=bool)
    ch_cov[order] = ch_cov_sorted
    assert np.array_equal(np.sort(uid), ch["uid"][o]), "user_id sets differ between dump and chain result"
    log(f"  chain coverage joined on user_id: {ch_cov.mean():.4f} (reference 0.2471)")

    taus = [float(x) for x in args.taus.split(",")]
    log(f"\n  {'tau':>6}{'cov@2000':>11}{'base t3@10':>12}{'GT med rank':>13}{'r<10':>8}{'r<50':>8}"
        f"{'MFD-ONLY n':>12}{'own r<10':>10}{'own med':>9}")
    res = {}
    for tau in taus:
        inv = 1.0 / tau
        hit10 = np.zeros(N, dtype=bool); cov = np.zeros(N, dtype=bool)
        rank = np.full(N, -1, dtype=np.int64)
        for s0 in range(0, N, args.chunk):
            s1 = min(N, s0 + args.chunk); n = s1 - s0
            lpf = c["pool_lpf"][s0:s1].float()
            lpt = c["pool_lpt"][s0:s1].float()
            lpi = c["pool_lpi"][s0:s1].float()
            valid = c["pool_sidscore"][s0:s1].float() > -1e30
            sc = args.wb * beh[s0:s1].unsqueeze(-1) + lpf + inv * (lpt + lpi)     # (n,nb,Kp)
            sc = sc.masked_fill(~valid, float("-inf"))
            # GT candidate mask
            m = ((c["pool_f"][s0:s1].long() == gt_f[s0:s1].view(n, 1, 1))
                 & (c["pool_t"][s0:s1].long() == gt_t[s0:s1].view(n, 1, 1))
                 & (c["pool_i"][s0:s1].long() == gt_i[s0:s1].view(n, 1, 1))
                 & valid)
            bm = torch.arange(nb).view(1, nb, 1) == gt_b[s0:s1].view(n, 1, 1)
            gtm = m & bm                                                          # (n,nb,Kp), <=1 True
            has = gtm.any(-1).any(-1)
            gsc = torch.where(gtm, sc, torch.full_like(sc, float("-inf"))).reshape(n, -1).max(-1).values
            # union rank of the GT by the new score
            r = (sc.reshape(n, -1) > gsc.view(n, 1)).sum(-1)
            rank[s0:s1] = torch.where(has, r, torch.full_like(r, -1)).numpy()
            hit10[s0:s1] = (has & (r < 10)).numpy()
            # coverage after re-selecting the top-kpool WITHIN the GT's behavior
            slab = sc[torch.arange(n), gt_b[s0:s1]]                                # (n,Kp)
            gslab = torch.where(gtm[torch.arange(n), gt_b[s0:s1]], slab, torch.full_like(slab, float("-inf"))).max(-1).values
            rb = (slab > gslab.view(n, 1)).sum(-1)
            cov[s0:s1] = (has & (rb < args.kpool)).numpy()
            del sc, m, bm, gtm, lpf, lpt, lpi, valid
        own = cov & ~ch_cov
        rk = rank[rank >= 0]
        res[tau] = {"cov": float(cov.mean()), "t3@10": float(hit10.mean()), "own_n": int(own.sum()),
                    "own_hit10": float(hit10[own].mean()) if own.sum() else 0.0}
        log(f"  {tau:>6.2f}{cov.mean():>11.4f}{hit10.mean():>12.4f}{np.median(rk):>13,.0f}"
            f"{hit10.mean():>8.4f}"
            f"{float(((rank >= 0) & (rank < 50)).mean()):>8.4f}"
            f"{int(own.sum()):>12,}{(float(hit10[own].mean()) if own.sum() else 0):>10.4f}"
            f"{np.median(rank[own][rank[own] >= 0]) if own.sum() else -1:>9,.0f}")

    base = res[1.0] if 1.0 in res else None
    log("\n  --- reading ---")
    log("  base t3@10 at tau=1 is the pre-reranker reference (0.2041 in the live pipeline); the reranker")
    log("  adds ~+0.011 on top of it. A tau that raises BOTH coverage and base t3@10 is worth a re-dump.")
    if base:
        best = max(res, key=lambda t: res[t]["t3@10"])
        log(f"  best base t3@10: tau={best} ({res[best]['t3@10']:.4f} vs {base['t3@10']:.4f} at tau=1, "
            f"{res[best]['t3@10']-base['t3@10']:+.4f}); coverage {res[best]['cov']:.4f} vs {base['cov']:.4f}")
        bo = max(res, key=lambda t: res[t]["own_n"] * res[t]["own_hit10"])
        log(f"  best MFD-ONLY yield: tau={bo} -> {res[bo]['own_n']:,} events at {res[bo]['own_hit10']:.4f} "
            f"= {res[bo]['own_n']*res[bo]['own_hit10']/N:+.5f} of t3@10 (tau=1: "
            f"{base['own_n']*base['own_hit10']/N:+.5f})")
        gain = max(res[t]["t3@10"] for t in res) - base["t3@10"]
        log(f"  VERDICT: best offline base-score gain {gain:+.4f}. Remember this is a LOWER BOUND, but the")
        log("  reranker's +0.011 is not additive with it -- it partly repairs the same ordering. Re-dump only")
        log(f"  if the gain is large relative to the {0.01 - 0.00196:+.5f} still needed for the bar.")
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(args.out_dir, "tau_screen.json"), "w"), indent=1)
    open(os.path.join(args.out_dir, "tau_screen.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
