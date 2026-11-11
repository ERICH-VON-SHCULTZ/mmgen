"""§21.5 three-seed table (CPU, inside the env, next to round20_collect.py / plugin_paired.py).
Seeds 42/43/44 of P1a (mfd122_mv_geoP1a), A01 (r21_A01, grouped, one micro-batch) and C033 (r21_C033), all evaluated by
macrec/r21_eval.py (plain K_f = 50, +1c chosen on each seed's own validation). Mean / std (ddof 1) over seeds of test HR@10 /
NDCG@10 for P1a, P1a+1c, A01, A01+1c, C033, C033+1c, C merged and D (MACRec paper setting, merged); per seed paired CIs
(2,000 resamples, seed 0, 95% percentile, users aligned by dataset index) vs C merged of the SAME seed and vs P1a (plain and
+1c) of the same seed. Usage: python r21_seeds.py <D> -> mmgen_runs/geo/r21_seeds_<D>.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin_paired import ndcg, paired  # noqa: E402
from round20_collect import c_lists_seed, rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
SEEDS = (42, 43, 44)
RUNS = {"P1a": "mfd122_mv_geoP1a", "A01": "r21_A01", "C033": "r21_C033"}


def met(r):
    return {"hr10": float((r < 10).mean()), "ndcg10": float(ndcg(r).mean())}


def ms(x):
    a = np.array(x)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None, "per_seed": [float(v) for v in a]}


def main():
    D = sys.argv[1]
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    tg = np.array([inter[u][-1] for u in inter])
    boot = np.random.default_rng(0).integers(0, len(tg), size=(2000, len(tg)))
    per, rows = {}, {}
    for s in SEEDS:
        C = rank_of(c_lists_seed(D, s), tg)
        cr = json.load(open(f"{R}/baselines/{D}/mql/seed{s}/eval/test/results_ensemble_20.json"))
        assert abs(cr["hit@10"] - (C < 10).mean()) < 1e-6
        dr = json.load(open(f"{R}/baselines/{D}/macrecpaper/seed{s}/eval/test/results_ensemble_20.json"))
        rk = {}
        for name, run in RUNS.items():
            b = f"{R}/mfd/{D}/{run}/seed{s}/r21_eval"
            E = json.load(open(f"{b}/eval.json"))
            X = torch.load(f"{b}/ranks.pt", weights_only=False)
            for k in ("plain", "1c"):
                r = X["test"][k].numpy()
                assert abs(E["test"][k]["hr10"] - (r < 10).mean()) < 1e-6
                rk[f"{name}{'' if k == 'plain' else '+1c'}"] = r
            rows.setdefault(name, {})[s] = {"1c_config": E["test"]["1c"]["name"], "valid_plain_hr10": E["valid"]["plain"]["hr10"],
                                            "valid_1c_hr10": E["valid"]["1c"]["hr10"], "f_top1_test": E["test"]["f"]["f_top1"]}
        per[s] = {k: met(v) for k, v in rk.items()} | {"C": met(C), "D": {"hr10": dr["hit@10"], "ndcg10": dr["ndcg@10"]}}
        per[s]["ci"] = {}
        for k in ("A01", "A01+1c", "C033", "C033+1c", "P1a", "P1a+1c"):
            per[s]["ci"][f"{k} - C"] = paired(rk[k], C, boot)
            if not k.startswith("P1a"):
                per[s]["ci"][f"{k} - P1a"] = paired(rk[k], rk["P1a"], boot)
                per[s]["ci"][f"{k} - P1a+1c"] = paired(rk[k], rk["P1a+1c"], boot)
    keys = ("P1a", "P1a+1c", "A01", "A01+1c", "C033", "C033+1c", "C", "D")
    agg = {k: {m: ms([per[s][k][m] for s in SEEDS]) for m in ("hr10", "ndcg10")} for k in keys}
    json.dump({"dataset": D, "seeds": SEEDS, "mean_std": agg, "per_seed": {str(s): per[s] for s in SEEDS}, "runs": rows},
              open(f"{R}/geo/r21_seeds_{D}.json", "w"), indent=1)
    for k in keys:
        print(f"{k:8s} HR@10 {agg[k]['hr10']['mean']:.4f} ± {agg[k]['hr10']['std']:.4f}  NDCG@10 {agg[k]['ndcg10']['mean']:.4f} ± {agg[k]['ndcg10']['std']:.4f}  per seed {agg[k]['hr10']['per_seed']}")
    for s in SEEDS:
        for k, v in per[s]["ci"].items():
            p = v["hr10"]
            print(f"seed {s} {k}: HR@10 {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  NDCG@10 {v['ndcg10']['diff']:+.4f} [{v['ndcg10']['ci95'][0]:+.4f}, {v['ndcg10']['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
