"""§21.6 gain diagnostic report (CPU, inside the env): every mmgen_runs/geo/r21_gain/<D>/<run>_s<seed>.pt of seed 42 against
P1a (mfd122_mv_geoP1a_s42). Per run: mean log P(true f), mean / median margin log P(true f) - log P(top f), f top-1, HR@10.
Rescued = P1a misses (rank >= 10), run hits; lost = the reverse. Each is split by WHERE the target's rank changed:
  f newly kept / f dropped   the target's f crosses the K_f = 50 cut
  across      most of the rank change comes from candidates of OTHER f buckets (|delta n_other| > |delta n_within|)
  within      most of it comes from the target's OWN bucket (|delta n_within| > |delta n_other|)
  tie         equal
plus the mean change of the target's margin on rescued / lost users.  Usage: python r21_gain_report.py <D> -> geo/r21_gain_<D>.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

R = "/scratch/yw8866/mmgen_runs"
BIG = 10 ** 6


def load(p):
    X = torch.load(p, weights_only=False)
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in X.items()}


def split(ref, new, sel, direction):
    out = {"users": int(sel.sum())}
    if direction == "rescued":
        fk = sel & (ref["rank"] >= BIG)
    else:
        fk = sel & (new["rank"] >= BIG)
    both = sel & ~fk & (ref["rank"] < BIG) & (new["rank"] < BIG)
    do = (new["n_other"] - ref["n_other"])[both]
    dw = (new["n_within"] - ref["n_within"])[both]
    across, within = np.abs(do) > np.abs(dw), np.abs(dw) > np.abs(do)
    out.update({"f_crosses_Kf_cut": int(fk.sum()), "across_buckets": int(across.sum()), "within_bucket": int(within.sum()),
                "tie": int((~across & ~within).sum()), "median_delta_n_other": float(np.median(do)) if len(do) else None,
                "median_delta_n_within": float(np.median(dw)) if len(dw) else None,
                "mean_delta_margin": float(((new["lp_true"] - new["lp_top"]) - (ref["lp_true"] - ref["lp_top"]))[sel].mean()) if sel.any() else None,
                "share_true_f_rank_improved": float((new["f_rank"] < ref["f_rank"])[sel].mean()) if sel.any() else None})
    return out


def main():
    D = sys.argv[1]
    G = f"{R}/geo/r21_gain/{D}"
    ref = load(f"{G}/mfd122_mv_geoP1a_s42.pt")
    res = {}
    for p in sorted(glob.glob(f"{G}/*_s42.pt")):
        name = os.path.basename(p)[:-7]
        x = load(p)
        assert np.array_equal(x["target"], ref["target"])
        m = x["lp_true"] - x["lp_top"]
        row = {"hr10": float((x["rank"] < 10).mean()), "mean_log_p_true_f": float(x["lp_true"].mean()),
               "mean_margin": float(m.mean()), "median_margin": float(np.median(m)), "f_top1": float((x["f_rank"] == 0).mean()),
               "sanity": x["sanity_share_same_rank_as_r21_eval"]}
        if name != "mfd122_mv_geoP1a":
            row["rescued"] = split(ref, x, (ref["rank"] >= 10) & (x["rank"] < 10), "rescued")
            row["lost"] = split(ref, x, (ref["rank"] < 10) & (x["rank"] >= 10), "lost")
        res[name] = row
        print(name, json.dumps(row))
    json.dump(res, open(f"{R}/geo/r21_gain_{D}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
