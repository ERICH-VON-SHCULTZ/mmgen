"""mmgen addition (docs/MFD_REPORT.md §22.5, E5): region-mass score report. CPU; inputs <A01 run>/seed<S>/r22_e5/{e5.json,
e5_ranks.pt} (macrec/r22_e5_eval.py) for every A01 checkpoint that has them.
Per partition x checkpoint: chosen (beta, K_f), valid / test HR@10, NDCG@10, paired bootstrap CI vs plain A01 (K_f = 50, beta = 0;
2,000 resamples, seed 0, users aligned by dataset index), ms/user. Summary per partition: mean valid / test gain over checkpoints,
NMI vs C-text / C-image level 1. Best oracle and best own partition (max mean valid gain): split on C-hit / A01-miss users (C =
merged list of the same seed): rescued / lost (and over all users), and how the items above the target move (same f / other f in
the target's region / other region), users whose target is kept.
Out: geo/r22_e5.json + printed tables.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin_paired import paired  # noqa: E402
from round20_collect import c_lists_seed, rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
ORACLE = ("oracle_ctext", "oracle_cimage")


def main():
    out, per = {}, {}
    for f in sorted(glob.glob(f"{R}/mfd/*/r21_A01/seed*/r22_e5/e5.json")):
        D, S = f.split("/")[-5], int(f.split("/")[-3][4:])
        e = json.load(open(f))
        X = torch.load(f.replace("e5.json", "e5_ranks.pt"), weights_only=False)
        inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
        tg = np.array([inter[u][-1] for u in inter])
        boot = np.random.default_rng(0).integers(0, len(tg), size=(2000, len(tg)))
        pl = X["plain"].numpy()
        vpl = e["valid"]["oracle_ctext"]["K50_b0.0"]
        key = f"{D} seed {S}"
        rows = {}
        for k, t in e["test"].items():
            ch = e["chosen"][k]
            v = e["valid"][k][f"K{ch['K']}_b{ch['beta']}"]
            rk = X["partitions"][k]["rank"].numpy()
            rows[k] = {"K": ch["K"], "beta": ch["beta"], "valid_hr10": v["hr10"], "valid_ndcg10": v["ndcg10"],
                       "valid_gain": v["hr10"] - vpl["hr10"], "test_hr10": t["hr10"], "test_ndcg10": t["ndcg10"],
                       "ms_per_user": t["ms_per_user"], "ci_vs_plain": paired(rk, pl, boot), **e["partitions"][k]}
        per[key] = {"D": D, "seed": S, "valid_plain": vpl, "test_plain": e["test_plain"], "rows": rows,
                    "sanity": e["sanity_plain_valid_same_rank_as_r21_eval"], "gpu": e["gpu"], "X": X, "tg": tg}
    if not per:
        print("no e5 results"); return
    # summary per partition
    names = list(next(iter(per.values()))["rows"])
    summ = {}
    for k in names:
        vg = [p["rows"][k]["valid_gain"] for p in per.values()]
        tgain = [p["rows"][k]["ci_vs_plain"]["hr10"]["diff"] for p in per.values()]
        summ[k] = {"mean_valid_gain": float(np.mean(vg)), "mean_test_gain": float(np.mean(tgain)),
                   "test_ci_above_0": int(sum(p["rows"][k]["ci_vs_plain"]["hr10"]["ci95"][0] > 0 for p in per.values())),
                   "test_ci_below_0": int(sum(p["rows"][k]["ci_vs_plain"]["hr10"]["ci95"][1] < 0 for p in per.values())),
                   "nmi_ctext": float(np.mean([p["rows"][k]["nmi_ctext"] for p in per.values()])),
                   "nmi_cimage": float(np.mean([p["rows"][k]["nmi_cimage"] for p in per.values()])),
                   "G": next(iter(per.values()))["rows"][k]["G"], "checkpoints": len(per)}
    best_or = max(ORACLE, key=lambda k: summ[k]["mean_valid_gain"])
    best_own = max([k for k in names if k not in ORACLE], key=lambda k: summ[k]["mean_valid_gain"])
    # split on C-hit / A01-miss users
    split = {}
    for key, p in per.items():
        X, tg, pl = p["X"], p["tg"], p["X"]["plain"].numpy()
        cm = rank_of(c_lists_seed(p["D"], p["seed"]), tg)            # C merged top-10 of the same seed
        sel = (cm < 10) & (pl >= 10)
        split[key] = {"c_hit_a01_miss": int(sel.sum())}
        for k in (best_or, best_own):
            P = {n: v.numpy() for n, v in X["partitions"][k].items()}
            new = P["rank"]
            kept = sel & (new < 10 ** 6) & (pl < 10 ** 6)              # target among the candidates (BIG = 1e6)
            mv = lambda n: float(np.mean(P[n][kept] - P["plain_" + n][kept]))
            split[key][k] = {"rescued_in_set": int((sel & (new < 10)).sum()),
                             "all_users_rescued": int(((pl >= 10) & (new < 10)).sum()), "all_users_lost": int(((pl < 10) & (new >= 10)).sum()),
                             "set_users_target_kept": int(kept.sum()),
                             "mean_change_items_above_same_f": mv("same_f"),
                             "mean_change_items_above_other_f_same_region": mv("other_f_same_region"),
                             "mean_change_items_above_other_region": mv("other_region"),
                             "share_rank_improved": float((new[kept] < pl[kept]).mean()),
                             "median_rank_plain_new": [float(np.median(pl[kept])), float(np.median(new[kept]))]}
    for p in per.values():
        del p["X"], p["tg"]
    out = {"per_checkpoint": per, "summary": summ, "best_oracle": best_or, "best_own": best_own, "split": split}
    json.dump(out, open(f"{R}/geo/r22_e5.json", "w"), indent=1)
    for key, p in per.items():
        print(f"\n=== {key} ({p['gpu']}; sanity {p['sanity']:.4f}) plain valid {p['valid_plain']['hr10']:.4f} test "
              f"{p['test_plain']['hr10']:.4f} / {p['test_plain']['ndcg10']:.4f} ms {p['test_plain']['ms_per_user']:.1f}")
        for k, r in p["rows"].items():
            c = r["ci_vs_plain"]["hr10"]
            print(f"{k:14s} G{r['G']:4d} nmi C-t {r['nmi_ctext']:.2f} C-i {r['nmi_cimage']:.2f} | K{r['K']:3d} b{r['beta']:<4} "
                  f"valid {r['valid_hr10']:.4f} ({r['valid_gain']:+.4f}) test {r['test_hr10']:.4f} / {r['test_ndcg10']:.4f} "
                  f"diff {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] ms {r['ms_per_user']:.1f}")
    print("\nsummary (mean over checkpoints)")
    for k, s in sorted(summ.items(), key=lambda x: -x[1]["mean_valid_gain"]):
        print(f"{k:14s}", json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in s.items()}))
    print("\nbest oracle", best_or, "best own", best_own)
    for key, s in split.items():
        print(key, json.dumps(s))


if __name__ == "__main__":
    main()
