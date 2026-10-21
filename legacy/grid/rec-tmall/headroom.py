"""Where can +0.01 absolute come from? A no-training, no-dump audit of the recoverable populations.

Runs off the cached per-event reranker results (seedvar/cache/{MFD,CH}__seed{s}.pt), which already
carry, per test event: user_id, gt behavior, whether the GT is in that arm's pool (has_gt), and the
GT's rank before/after reranking. Seconds of CPU, no GPU, no pool dumps.

The bar: MFD - chain >= +0.01 on t3@10 = +830 test events out of 83,005 relative to chain's score,
i.e. +687 more MFD top-10 hits than MFD has today (the current lead is already +0.00173 = 143 events).

The question this answers: which population of events could supply them, and at what recovery rate?
Populations are defined by POOL MEMBERSHIP, not by score:
  BOTH        GT in both arms' pools   -> a gain here is one chain can also chase
  MFD-ONLY    GT in MFD's pool only    -> chain cannot follow: every recovery moves the LEAD 1:1
  CHAIN-ONLY  GT in chain's pool only  -> MFD cannot reach these at all without a new pool
  NEITHER     unreachable by reranking either arm
For each: size, current recovery in each arm, the rank distribution of the misses, and the recovery
rate that population would need to carry the whole +0.00827. Averaged over seeds, with the spread.
"""

import argparse
import json
import os

import numpy as np
import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
BANDS = [(0, 10, "<10 (hit)"), (10, 20, "10-19"), (20, 50, "20-49"), (50, 100, "50-99"),
         (100, 500, "100-499"), (500, 10 ** 9, "500+")]


def load(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    o = np.argsort(r["uid"])
    out = {k: (r[k][o] if isinstance(r.get(k), np.ndarray) and r[k].shape[:1] == r["uid"].shape else r[k])
           for k in r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/scratch/yw8866/rec-tmall/seedvar/cache")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--target", type=float, default=0.01, help="required absolute lead")
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    log("=" * 100)
    log("=== HEADROOM AUDIT: which population can supply a +%.3f absolute lead? ===" % args.target)
    log("=" * 100)

    acc = {}
    N = None
    for sd in seeds:
        m = load(os.path.join(args.cache, f"MFD__seed{sd}.pt"))
        c = load(os.path.join(args.cache, f"CH__seed{sd}.pt"))
        assert np.array_equal(m["uid"], c["uid"]), f"seed {sd}: user_id join failed"
        assert np.array_equal(m["gt_b"], c["gt_b"]), f"seed {sd}: gt behavior disagrees after join"
        N = m["uid"].shape[0]
        assert N == 83005, f"expected 83,005 test events, got {N:,}"
        per = {b: int((m["gt_b"] == b).sum()) for b in range(4)}
        assert per == {0: 77833, 1: 2042, 2: 1554, 3: 1576}, f"per-behavior counts wrong: {per}"
        mc, cc = m["has_gt"].astype(bool), c["has_gt"].astype(bool)
        hm, hc = m["hit_after"] > 0, c["hit_after"] > 0
        pops = {"BOTH": mc & cc, "MFD-ONLY": mc & ~cc, "CHAIN-ONLY": cc & ~mc, "NEITHER": ~mc & ~cc}
        hmb, hcb = m["hit_before"] > 0, c["hit_before"] > 0
        for nm, p in pops.items():
            d = acc.setdefault(nm, {"n": [], "mh": [], "ch": [], "bands": [], "beh": [],
                                    "m_hb": [], "m_ha": [], "m_pro": [], "m_dem": [], "m_mrb": [], "m_mra": [],
                                    "c_hb": [], "c_ha": [], "c_pro": [], "c_dem": [], "c_mrb": [], "c_mra": []})
            for key, hbx, hax, rbx, rax in (("m", hmb, hm, m["r3_before"], m["r3_after"]),
                                            ("c", hcb, hc, c["r3_before"], c["r3_after"])):
                d[f"{key}_hb"].append(int((p & hbx).sum())); d[f"{key}_ha"].append(int((p & hax).sum()))
                d[f"{key}_pro"].append(int((p & ~hbx & hax).sum())); d[f"{key}_dem"].append(int((p & hbx & ~hax).sum()))
                rr = rbx[p]; ra = rax[p]
                d[f"{key}_mrb"].append(float(np.median(rr[rr >= 0])) if (rr >= 0).any() else -1.0)
                d[f"{key}_mra"].append(float(np.median(ra[ra >= 0])) if (ra >= 0).any() else -1.0)
            d["n"].append(int(p.sum())); d["mh"].append(int((p & hm).sum())); d["ch"].append(int((p & hc).sum()))
            r = m["r3_after"][p]
            d["bands"].append([int(((r >= lo) & (r < hi)).sum()) for lo, hi, _ in BANDS] + [int((r < 0).sum())])
            d["beh"].append([[int((p & (m["gt_b"] == b)).sum()), int((p & hm & (m["gt_b"] == b)).sum())] for b in range(4)])
        acc.setdefault("_lead", []).append(float(hm.mean() - hc.mean()))
        acc.setdefault("_mfd", []).append(float(hm.mean())); acc.setdefault("_ch", []).append(float(hc.mean()))

    lead = np.mean(acc["_lead"]); mfd = np.mean(acc["_mfd"]); ch = np.mean(acc["_ch"])
    need_ev = (args.target - lead) * N
    log(f"\n  seeds {seeds}   MFD {mfd:.5f}   chain {ch:.5f}   lead {lead:+.5f} "
        f"(std {np.std(acc['_lead'], ddof=1):.5f})")
    log(f"  BAR {args.target:+.3f} -> MFD must reach {ch + args.target:.5f}: "
        f"**{need_ev:,.0f} more MFD top-10 hits** (current lead = {lead*N:,.0f} events)")
    log("  (chain held fixed; if chain also improves, the requirement grows one-for-one)")

    log(f"\n  {'population':<11}{'events':>9}{'share':>8}{'MFD hits':>10}{'MFD rec':>9}{'chain hits':>12}"
        f"{'chain rec':>11}{'lead contrib':>14}{'misses':>9}")
    for nm in ("BOTH", "MFD-ONLY", "CHAIN-ONLY", "NEITHER"):
        d = acc[nm]; n = np.mean(d["n"]); mh = np.mean(d["mh"]); chh = np.mean(d["ch"])
        log(f"  {nm:<11}{n:>9,.0f}{n/N:>8.4f}{mh:>10,.0f}{(mh/n if n else 0):>9.4f}{chh:>12,.0f}"
            f"{(chh/n if n else 0):>11.4f}{(mh-chh)/N:>+14.5f}{n-mh:>9,.0f}")
    log("  lead contrib = (MFD hits - chain hits)/N within that population; the four sum to the lead.")

    log("\n  --- what each population would have to do to carry the whole gap, alone ---")
    for nm in ("BOTH", "MFD-ONLY", "CHAIN-ONLY", "NEITHER"):
        d = acc[nm]; n = np.mean(d["n"]); mh = np.mean(d["mh"])
        cur = mh / n if n else 0.0
        need = (mh + need_ev) / n if n else float("inf")
        if nm in ("CHAIN-ONLY", "NEITHER"):
            feas = "STRUCTURALLY IMPOSSIBLE by reranking (the GT is not in MFD's pool)"
            extra = "  [needs a different MFD pool, not a better reranker]"
        else:
            feas = ("IMPOSSIBLE (needs recovery > 1)" if need > 1 else
                    "implausible (> 2x current)" if need > 2 * cur + 1e-9 else "plausible")
            extra = ("  [chain cannot follow here: its pool lacks the GT]" if nm == "MFD-ONLY"
                     else "  [chain can chase every gain here -- lifts both arms, lead may not move]")
        log(f"  {nm:<11} recovery {cur:.4f} -> {min(need,9.9999):.4f} required   {feas}{extra}")

    log("\n  --- MFD-ONLY: where the misses sit (rank of the GT after reranking) ---")
    for nm in ("MFD-ONLY", "BOTH"):
        d = acc[nm]; b = np.mean(np.array(d["bands"], float), axis=0); n = np.mean(d["n"])
        log(f"  {nm:<11}" + "  ".join(f"{lab}={b[j]:,.0f}({b[j]/n:.3f})" for j, (_, _, lab) in enumerate(BANDS))
            + f"  not-in-pool={b[-1]:,.0f}")
    d = acc["MFD-ONLY"]; b = np.mean(np.array(d["bands"], float), axis=0)
    near = b[1] + b[2]
    log(f"  MFD-ONLY near misses (rank 10-49) = {near:,.0f} events = {near/N:+.5f} of t3@10 if ALL were recovered"
        f" -> {'enough alone' if near/N >= (args.target-lead) else 'NOT enough alone'} for the gap ({need_ev:,.0f} needed)")
    log(f"  MFD-ONLY total misses = {np.mean(d['n'])-np.mean(d['mh']):,.0f} = {(np.mean(d['n'])-np.mean(d['mh']))/N:+.5f} "
        f"if ALL were recovered (the hard ceiling of this population)")

    log("\n  --- MFD-ONLY by behavior (n, MFD hits, recovery) ---")
    beh = np.mean(np.array(acc["MFD-ONLY"]["beh"], float), axis=0)
    for b in range(4):
        n_b, h_b = beh[b]
        log(f"    {BEH[b]:<5} n={n_b:>8,.0f}  hits={h_b:>7,.0f}  recovery={(h_b/n_b if n_b else 0):.4f}  "
            f"all-recovered would add {(n_b-h_b)/N:+.5f}")

    log("\n  --- verdict ---")
    d = acc["MFD-ONLY"]; n = np.mean(d["n"]); mh = np.mean(d["mh"])
    ceil_own = (n - mh) / N
    ceil_both = (np.mean(acc["BOTH"]["n"]) - np.mean(acc["BOTH"]["mh"])) / N
    log(f"  reachable-by-reranking ceilings: MFD-ONLY {ceil_own:+.5f}, BOTH {ceil_both:+.5f}, "
        f"together {ceil_own+ceil_both:+.5f}; gap to close {args.target-lead:+.5f}")
    if ceil_own >= (args.target - lead):
        log(f"  -> the MFD-ONLY population ALONE can supply the bar: it needs "
            f"{100*need_ev/(n-mh):.1f}% of its misses recovered. This is the target; chain gains nothing here.")
    else:
        log(f"  -> MFD-ONLY alone is NOT sufficient even at 100% recovery ({ceil_own:+.5f} < {args.target-lead:+.5f}). "
            f"The bar additionally requires BOTH-population gains that chain can also chase, or a wider MFD pool "
            f"(new coverage), or both. Say so plainly rather than tuning toward it.")
    # ---------------------------------------------------------------- reranker migration
    log("\n  --- does the reranker MOVE these events, or leave them where the generator put them? ---")
    log(f"  {'population':<11}{'arm':<6}{'base hits':>11}{'rerank hits':>12}{'promoted':>10}{'demoted':>9}"
        f"{'net':>8}{'med rank bef':>14}{'med rank aft':>14}")
    mig = {}
    for nm in ("BOTH", "MFD-ONLY"):
        for arm, key in (("MFD", "m"), ("chain", "c")):
            hb = np.mean(acc[nm][f"{key}_hb"]); ha = np.mean(acc[nm][f"{key}_ha"])
            pr = np.mean(acc[nm][f"{key}_pro"]); de = np.mean(acc[nm][f"{key}_dem"])
            rb = np.mean(acc[nm][f"{key}_mrb"]); ra = np.mean(acc[nm][f"{key}_mra"])
            mig[(nm, arm)] = (hb, ha, pr, de)
            log(f"  {nm:<11}{arm:<6}{hb:>11,.0f}{ha:>12,.0f}{pr:>10,.0f}{de:>9,.0f}{ha-hb:>+8,.0f}"
                f"{rb:>14,.0f}{ra:>14,.0f}")
    log("  promoted = GT moved from rank >=10 into the top 10; demoted = the reverse; median ranks over the")
    log("  whole population (chain's ranks are -1 on MFD-ONLY by construction and are not shown as medians).")
    hb_m, ha_m, pro_m, dem_m = mig[("MFD-ONLY", "MFD")]
    log(f"\n  MFD-ONLY: the reranker promotes {pro_m:,.0f} and demotes {dem_m:,.0f} of 5,355 -> net {ha_m-hb_m:+,.0f}.")
    log(f"    It must promote {need_ev:,.0f} MORE to clear the bar, i.e. {need_ev/max(pro_m,1):.1f}x its current "
        f"promotion count on this population.")
    b_bef = mig[("BOTH", "MFD")][0] - mig[("BOTH", "chain")][0]
    b_aft = mig[("BOTH", "MFD")][1] - mig[("BOTH", "chain")][1]
    log(f"  BOTH: MFD - chain is {b_bef:+,.0f} events BEFORE reranking and {b_aft:+,.0f} AFTER -> "
        + ("MFD's reranker closes part of a base-score deficit" if b_aft > b_bef else
           "MFD's reranker LOSES ground on shared events")
        + f" (the -0.00119 deficit is {'inherited from the generator' if b_bef < 0 else 'created by the reranker'}).")

    json.dump({"lead": lead, "mfd": mfd, "chain": ch, "need_events": need_ev,
               "pops": {k: {"n": float(np.mean(v["n"])), "mfd_hits": float(np.mean(v["mh"])),
                            "chain_hits": float(np.mean(v["ch"]))} for k, v in acc.items() if not k.startswith("_")}},
              open(os.path.join(args.out_dir, "headroom.json"), "w"), indent=1)
    open(os.path.join(args.out_dir, "headroom_report.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
