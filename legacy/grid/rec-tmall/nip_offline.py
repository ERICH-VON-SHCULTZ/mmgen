"""Offline estimate of NIP's reserved-slot policy, from cached results + dumps -- no GPU.

Policy P_k: top-(10-k) by the reranker  u  top-k by the GENERATOR's base score among candidates that are
not in the user's history (and not already chosen). Per event, from what is already on disk:
    hit_Pk ~= [ reranker rank of GT < 10-k ]  OR  [ GT is not a history item AND fewer than k non-history
                                                     candidates have a higher base score than the GT ]
The reranker rank comes from the cached per-event results (5 seeds); the base score and the history mask
come from the dump. The second term ignores that some higher-base non-history candidates may already sit
in the reranker's top-(10-k) (which would free a reserved slot for the GT), so it slightly UNDER-counts
what the real policy recovers. Both arms get the identical computation.

This answers, before the GPU job runs, whether reserving slots for the generator's new-item picks can
recover the new-item hits MFD's reranker evicts, and what it costs in displaced reranker hits.
"""

import argparse
import os

import numpy as np
import torch

KS = (0, 1, 2, 3)


def per_event(dump, kpool, chunk, wb):
    d = torch.load(dump, map_location="cpu", weights_only=False, mmap=True)
    c = d["cols"]
    N, nb, Kst = c["pool_f"].shape
    Kp = min(kpool or Kst, Kst)
    uid = c["user_id"].long().numpy()
    gb = c["gt_b"].long()
    gf, gt, gi = c["gt_f"].long(), c["gt_t"].long(), c["gt_i"].long()
    nh_rank = np.full(N, -1, dtype=np.int64)          # GT's base rank among NON-history candidates (-1: GT absent)
    gt_hist = np.zeros(N, dtype=bool)                 # GT tuple is itself in the history (a repeat)
    for s0 in range(0, N, chunk):
        s1 = min(N, s0 + chunk); n = s1 - s0
        pf = c["pool_f"][s0:s1, :, :Kp].long(); pt = c["pool_t"][s0:s1, :, :Kp].long(); pi = c["pool_i"][s0:s1, :, :Kp].long()
        valid = c["pool_sidscore"][s0:s1, :, :Kp].float() > -1e30
        base = (wb * c["beh_logp"][s0:s1].float().unsqueeze(-1) + c["pool_lpf"][s0:s1, :, :Kp].float()
                + c["pool_lpt"][s0:s1, :, :Kp].float() + c["pool_lpi"][s0:s1, :, :Kp].float())
        base = base.masked_fill(~valid, float("-inf")).reshape(n, -1)
        f = pf.reshape(n, -1); t = pt.reshape(n, -1); i = pi.reshape(n, -1)
        h = c["hist_ids"][s0:s1].long(); H = h.shape[1] // 4; h = h[:, :H * 4].reshape(n, H, 4)
        hv = h[:, :, 0] >= 0
        hm = ((f[:, :, None] == h[:, None, :, 1]) & (t[:, :, None] == h[:, None, :, 2])
              & (i[:, :, None] == h[:, None, :, 3]) & hv[:, None, :]).any(-1)          # (n, C)
        br = torch.arange(nb).repeat_interleave(Kp).view(1, -1)
        gtm = ((f == gf[s0:s1, None]) & (t == gt[s0:s1, None]) & (i == gi[s0:s1, None])
               & (br == gb[s0:s1, None]) & valid.reshape(n, -1))
        has = gtm.any(1)
        gsc = torch.where(gtm, base, torch.full_like(base, float("-inf"))).max(1).values
        nh = base.masked_fill(hm, float("-inf"))
        r = (nh > gsc.unsqueeze(1)).sum(1)
        gth = (hm & gtm).any(1)
        nh_rank[s0:s1] = torch.where(has, r, torch.full_like(r, -1)).numpy()
        gt_hist[s0:s1] = gth.numpy()
    return uid, nh_rank, gt_hist, gb.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfd_dump", required=True); ap.add_argument("--ch_dump", required=True)
    ap.add_argument("--cache", default="/scratch/yw8866/rec-tmall/seedvar/cache")
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--mfd_kpool", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=400); ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--out", default="/scratch/yw8866/rec-tmall/nip_exp/nip_offline.txt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    seeds = [int(s) for s in a.seeds.split(",")]
    arms = {}
    for arm, dump, kp in (("MFD", a.mfd_dump, a.mfd_kpool), ("CH", a.ch_dump, None)):
        uid, nhr, gth, gb = per_event(dump, kp, a.chunk, a.wb)
        o = np.argsort(uid)
        arms[arm] = {"uid": uid[o], "nhr": nhr[o], "gth": gth[o], "gb": gb[o]}
        log(f"  [{arm}] per-event non-history base ranks computed ({len(uid):,} events)")
    assert np.array_equal(arms["MFD"]["uid"], arms["CH"]["uid"]), "user_id join failed"
    N = len(arms["MFD"]["uid"])

    res = {}
    for arm in ("MFD", "CH"):
        for sd in seeds:
            r = torch.load(os.path.join(a.cache, f"{arm}__seed{sd}.pt"), map_location="cpu", weights_only=False)
            o = np.argsort(r["uid"]); assert np.array_equal(r["uid"][o], arms[arm]["uid"])
            ra = r["r3_after"][o]; has = r["has_gt"][o].astype(bool)
            hit0 = has & (ra >= 0) & (ra < 10)
            assert np.array_equal(hit0, r["hit_after"][o] > 0), "reranker rank does not reproduce the cached hits"
            for k in KS:
                rer = has & (ra >= 0) & (ra < 10 - k)
                res_k = rer | (has & ~arms[arm]["gth"] & (arms[arm]["nhr"] >= 0) & (arms[arm]["nhr"] < k))
                res[(arm, sd, k)] = res_k

    rep = arms["MFD"]["gth"]; pv = arms["MFD"]["gb"] == 0
    mc = torch.load(os.path.join(a.cache, "MFD__seed0.pt"), map_location="cpu", weights_only=False)
    cc = torch.load(os.path.join(a.cache, "CH__seed0.pt"), map_location="cpu", weights_only=False)
    mcov = mc["has_gt"][np.argsort(mc["uid"])].astype(bool); ccov = cc["has_gt"][np.argsort(cc["uid"])].astype(bool)

    log("\n" + "=" * 100 + "\n=== NIP OFFLINE ESTIMATE: recall@10 under P_k (reserve k slots for the generator's non-history picks) ===\n" + "=" * 100)
    log(f"  {'k':>3}{'MFD':>10}{'chain':>10}{'lead':>11}{'lead std':>10}   MFD per-seed / chain per-seed")
    best = {}
    for k in KS:
        m = [res[("MFD", sd, k)].mean() for sd in seeds]; c = [res[("CH", sd, k)].mean() for sd in seeds]
        lead = np.array(m) - np.array(c)
        best[k] = (np.mean(m), np.mean(c))
        log(f"  {k:>3}{np.mean(m):>10.4f}{np.mean(c):>10.4f}{lead.mean():>+11.5f}{lead.std(ddof=1):>10.5f}   "
            + " ".join(f"{x:.4f}" for x in m) + " / " + " ".join(f"{x:.4f}" for x in c))
    km = max(KS, key=lambda k: best[k][0]); kc = max(KS, key=lambda k: best[k][1])
    log(f"\n  each arm's best k on these TEST estimates (indicative only; the GPU job chooses k on VAL): MFD k={km}, chain k={kc}")
    log(f"  lead at (MFD k={km}, chain k={kc}): {best[km][0]-best[kc][1]:+.5f}   vs P0 lead {best[0][0]-best[0][1]:+.5f}")

    log("\n  where the P_k change comes from (MFD, mean over seeds; pv only):")
    for nm, p in (("BOTH", mcov & ccov), ("MFD-ONLY", mcov & ~ccov)):
        for rn, rr in (("repeat", rep), ("new item", ~rep)):
            q = p & rr & pv
            vals = [np.mean([res[("MFD", sd, k)][q].mean() for sd in seeds]) for k in KS]
            log(f"    {nm:<9}{rn:<9} n={int(q.sum()):>6,}  " + "  ".join(f"P{k}={v:.4f}" for k, v in zip(KS, vals))
                + f"   (P{km}-P0 = {(vals[km]-vals[0])*q.sum():+.0f} events)")
    open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
