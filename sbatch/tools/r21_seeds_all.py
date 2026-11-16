"""§21.9 A01 across datasets and seeds (CPU, inside the env, next to round20_collect.py / plugin_paired.py).
Datasets Instruments / Arts / Games x seeds 42 / 43 / 44. P1a (mfd122_mv_geoP1a) and A01 (r21_A01), both evaluated by
macrec/r21_eval.py (plain K_f = 50; +1c chosen on each seed's own validation). C (MQL4GRec) and D (MACRec paper setting)
merged lists. Per dataset: mean / std (ddof 1) over seeds of valid and test HR@10 / NDCG@10 for P1a, P1a+1c, A01, A01+1c,
C, D (D from its results files). Per dataset and seed: paired CIs (2,000 resamples, seed 0, 95% percentile) of A01 and A01+1c
vs P1a and vs C of the same seed, on VALID and TEST (C's valid lists rebuilt like the test ones; users aligned by index).
Pooled paired bootstrap (information only): per-user differences of all users of all three datasets, resampled within each
dataset (stratified), per seed and pooled over seeds (per-user difference averaged over the 3 seeds, same users).
Usage: python r21_seeds_all.py -> mmgen_runs/geo/r21_seeds_all.json (missing runs are listed and skipped)
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
DS, SEEDS = ("Instruments", "Arts", "Games"), (42, 43, 44)
RUNS = {"P1a": "mfd122_mv_geoP1a", "A01": "r21_A01"}
hr = lambda r: (r < 10).astype(float)


def ms(x):
    a = np.array(x, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None, "per_seed": [float(v) for v in a], "n": len(a)}


def main():
    out, missing, pooled = {"datasets": {}}, [], {"hr10": {}, "ndcg10": {}}
    diffs = {}                                                      # (contrast, metric) -> {seed: {D: per-user diff}}
    for D in DS:
        inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
        tgt = {"test": np.array([inter[u][-1] for u in inter]), "valid": np.array([inter[u][-2] for u in inter])}
        boot = np.random.default_rng(0).integers(0, len(tgt["test"]), size=(2000, len(tgt["test"])))
        per, vals = {}, {}
        for s in SEEDS:
            rk = {}
            try:
                for name, run in RUNS.items():
                    X = torch.load(f"{R}/mfd/{D}/{run}/seed{s}/r21_eval/ranks.pt", weights_only=False)
                    for sp in ("valid", "test"):
                        rk[(name, sp)] = X[sp]["plain"].numpy(); rk[(name + "+1c", sp)] = X[sp]["1c"].numpy()
            except FileNotFoundError as e:
                missing.append(f"{D} seed {s}: {e.filename}")
                continue
            for sp in ("valid", "test"):
                C = rank_of(c_lists_seed(D, s, sp), tgt[sp])
                cr = json.load(open(f"{R}/baselines/{D}/mql/seed{s}/eval/{sp}/results_ensemble_20.json"))
                assert abs(cr["hit@10"] - (C < 10).mean()) < 1e-6, (D, s, sp)
                rk[("C", sp)] = C
            dres = {sp: json.load(open(f"{R}/baselines/{D}/macrecpaper/seed{s}/eval/{sp}/results_ensemble_20.json")) for sp in ("valid", "test")}
            row = {}
            for sp in ("valid", "test"):
                for k in ("P1a", "P1a+1c", "A01", "A01+1c", "C"):
                    row[f"{k} {sp}"] = {"hr10": float(hr(rk[(k, sp)]).mean()), "ndcg10": float(ndcg(rk[(k, sp)]).mean())}
                row[f"D {sp}"] = {"hr10": dres[sp]["hit@10"], "ndcg10": dres[sp]["ndcg@10"]}
                for a in ("A01", "A01+1c"):
                    for b in ("P1a", "P1a+1c", "C"):
                        row[f"{a} - {b} {sp}"] = paired(rk[(a, sp)], rk[(b, sp)], boot)
                row[f"P1a - C {sp}"] = paired(rk[("P1a", sp)], rk[("C", sp)], boot)
            for a, b in (("A01", "P1a"), ("A01", "C")):
                for mname, f in (("hr10", hr), ("ndcg10", ndcg)):
                    diffs.setdefault((a, b, mname), {}).setdefault(s, {})[D] = f(rk[(a, "test")]) - f(rk[(b, "test")])
            per[str(s)] = row
        keys = [k for k in next(iter(per.values()), {}) if " - " not in k]
        agg = {k: {m: ms([per[s][k][m] for s in per]) for m in ("hr10", "ndcg10")} for k in keys}
        out["datasets"][D] = {"seeds_done": list(per), "mean_std": agg, "per_seed": per}
    rng = np.random.default_rng(0)
    for (a, b, mname), bys in diffs.items():
        res = {}
        full = {s: v for s, v in bys.items() if len(v) == len(DS)}
        for s, byd in list(full.items()) + ([("pooled_over_seeds", {D: np.mean([full[s][D] for s in full], 0) for D in DS})] if len(full) == len(SEEDS) else []):
            n = {D: len(v) for D, v in byd.items()}
            tot = sum(n.values())
            est = sum(v.sum() for v in byd.values()) / tot
            bs = np.zeros(2000)
            for D, v in byd.items():
                idx = rng.integers(0, len(v), size=(2000, len(v)))
                bs += v[idx].sum(1)
            bs /= tot
            res[str(s)] = {"diff": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "users": tot}
        pooled[mname][f"{a} - {b}"] = res
    out["pooled_test_bootstrap"] = pooled
    out["missing"] = missing
    json.dump(out, open(f"{R}/geo/r21_seeds_all.json", "w"), indent=1)
    for D, v in out["datasets"].items():
        print(f"== {D} seeds {v['seeds_done']}")
        for k, x in v["mean_std"].items():
            print(f"  {k:14s} HR@10 {x['hr10']['mean']:.4f} ± {x['hr10']['std'] or 0:.4f}  NDCG@10 {x['ndcg10']['mean']:.4f} ± {x['ndcg10']['std'] or 0:.4f}  {['%.4f' % y for y in x['hr10']['per_seed']]}")
        for s, row in v["per_seed"].items():
            for k, p in row.items():
                if " - " in k:
                    print(f"  seed {s} {k}: HR@10 {p['hr10']['diff']:+.4f} [{p['hr10']['ci95'][0]:+.4f}, {p['hr10']['ci95'][1]:+.4f}]  NDCG@10 {p['ndcg10']['diff']:+.4f} [{p['ndcg10']['ci95'][0]:+.4f}, {p['ndcg10']['ci95'][1]:+.4f}]")
    print("pooled:", json.dumps(pooled))
    print("missing:", missing)


if __name__ == "__main__":
    main()
