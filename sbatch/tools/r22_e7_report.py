"""mmgen addition (docs/MFD_REPORT.md §22.7, E7): structured item vector vs A01, CPU. For every E7 run with r21_eval output
(mfd/<D>/r22_E7{a,b}/seed<S>): plain K_f = 50 valid / test HR@1/5/10, NDCG@5/10, ms/user (H200, batch 64), paired bootstrap CIs vs
A01 of the same seed on valid and test (2,000 resamples, seed 0, users aligned by dataset index), f diagnostics (r21_eval: top-1,
median rank) and the E4 information gain of the f distribution (fdump: IG = CE_m - NLL, q = empirical target-f distribution over
the test users, renormalised log P(f)). Better run (higher seed-42 valid HR@10): E4 within-region split on C-hit / A01-miss users
(C merged, same seed): rescued / lost (all users), and the target f's rank against same-region competitors (f codes ranked above
the target f that contain an item of the target's C-text level-1 bucket) in A01 vs E7.
--stage: print the E7 variants whose Arts seed-42 valid HR@10 beats A01's (one per line) to geo/r22_e7_stage.txt.
Out: geo/r22_e7.json + printed tables.
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
MD = ".index_md122_a05_ep500_reassign.json"
KEYS = ("hr1", "hr5", "hr10", "ndcg5", "ndcg10")


def ev(run_dir):
    e = json.load(open(f"{run_dir}/r21_eval/eval.json"))
    r = torch.load(f"{run_dir}/r21_eval/ranks.pt", weights_only=False)
    return e, {s: r[s]["plain"].numpy() for s in ("valid", "test")}


def info_gain(path, D):
    if not os.path.exists(path):
        return None, None
    E = torch.load(path, weights_only=False)["f"]
    lp = E["lp"].float().numpy()
    col = {t: j for j, t in enumerate(E["tokens"])}
    idx = json.load(open(f"{R}/data/{D}/{D}{MD}"))
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    y = np.array([col[idx[str(s[-1])][0]] for s in inter.values()])
    q = np.bincount(y, minlength=lp.shape[1]) / len(y)
    ce = float(-np.log(q[y]).mean())
    nll = float(-lp[np.arange(len(y)), y].mean())
    return {"nll": nll, "ce_marginal": ce, "ig": ce - nll, "ig_norm": (ce - nll) / ce}, lp


def main():
    stage = "--stage" in sys.argv
    out = {}
    runs = sorted(glob.glob(f"{R}/mfd/*/r22_E7*/seed*/r21_eval/eval.json"))
    for p in runs:
        rd = p.rsplit("/r21_eval/", 1)[0]
        D, V, S = rd.split("/")[-3], rd.split("/")[-2][4:], int(rd.split("/")[-1][4:])
        inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
        n = len(inter)
        boot = np.random.default_rng(0).integers(0, n, size=(2000, n))
        e, rk = ev(rd)
        ea, ra = ev(f"{R}/mfd/{D}/r21_A01/seed{S}")
        ig, _ = info_gain(f"{R}/geo/r22/{D}/fdump_r22_{V}_s{S}.pt", D)
        ig_a, _ = info_gain(f"{R}/geo/r22/{D}/fdump_r21_A01_s{S}.pt", D)
        out[f"{D} {V} seed {S}"] = {
            "D": D, "var": V, "seed": S, "gpu": e.get("gpu"),
            "valid": {k: e["valid"]["plain"][k] for k in KEYS}, "test": {k: e["test"]["plain"][k] for k in KEYS},
            "A01_valid": {k: ea["valid"]["plain"][k] for k in KEYS}, "A01_test": {k: ea["test"]["plain"][k] for k in KEYS},
            "ms_per_user": e["test"]["plain"].get("ms_per_user"), "A01_ms_per_user": ea["test"]["plain"].get("ms_per_user"),
            "ci_valid_vs_A01": paired(rk["valid"], ra["valid"], boot), "ci_test_vs_A01": paired(rk["test"], ra["test"], boot),
            "f_test": e["test"].get("f"), "A01_f_test": ea["test"].get("f"), "f_info_gain": ig, "A01_f_info_gain": ig_a}
    if stage:
        win = [r["var"] for r in out.values() if r["D"] == "Arts" and r["seed"] == 42 and r["valid"]["hr10"] > r["A01_valid"]["hr10"]]
        open(f"{R}/geo/r22_e7_stage.txt", "w").write("\n".join(win) + ("\n" if win else ""))
        print("E7 variants beating A01 on Arts seed-42 valid HR@10:", win or "none")
    # within-region split for the better seed-42 Arts run
    s42 = {r["var"]: r for r in out.values() if r["D"] == "Arts" and r["seed"] == 42}
    split = {}
    if s42:
        V = max(s42, key=lambda v: (s42[v]["valid"]["hr10"], s42[v]["valid"]["ndcg10"]))
        D, S = "Arts", 42
        _, lp7 = info_gain(f"{R}/geo/r22/{D}/fdump_r22_{V}_s{S}.pt", D)
        _, lpa = info_gain(f"{R}/geo/r22/{D}/fdump_r21_A01_s{S}.pt", D)
        if lp7 is not None and lpa is not None:
            E = torch.load(f"{R}/geo/r22/{D}/fdump_r21_A01_s{S}.pt", weights_only=False)["f"]
            col = {t: j for j, t in enumerate(E["tokens"])}
            idx = json.load(open(f"{R}/data/{D}/{D}{MD}"))
            fo = {int(k): col[v[0]] for k, v in idx.items()}
            ct = {int(k): v[0] for k, v in json.load(open(f"{R}/data/{D}/{D}.index_lemb_mql.json")).items()}
            inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
            tg = np.array([s[-1] for s in inter.values()])
            nF = lpa.shape[1]
            fwc = {}
            for i, c in ct.items():
                fwc.setdefault(c, np.zeros(nF, bool))[fo[i]] = True
            A = ev(f"{R}/mfd/{D}/r21_A01/seed{S}")[1]["test"]
            N = ev(f"{R}/mfd/{D}/r22_{V}/seed{S}")[1]["test"]
            C = rank_of(c_lists_seed(D, S), tg)
            sel = (C < 10) & (A >= 10)
            reg_a, reg_7, fr_a, fr_7 = [], [], [], []
            for u in np.where(sel)[0]:
                t = fo[int(tg[u])]
                reg = fwc[ct[int(tg[u])]]
                reg_a.append(int(((lpa[u] > lpa[u, t]) & reg).sum()))
                reg_7.append(int(((lp7[u] > lp7[u, t]) & reg).sum()))
                fr_a.append(int((lpa[u] > lpa[u, t]).sum()))
                fr_7.append(int((lp7[u] > lp7[u, t]).sum()))
            reg_a, reg_7, fr_a, fr_7 = map(np.array, (reg_a, reg_7, fr_a, fr_7))
            split = {"run": V, "c_hit_a01_miss": int(sel.sum()), "rescued_in_set": int((N[sel] < 10).sum()),
                     "all_users_rescued": int(((A >= 10) & (N < 10)).sum()), "all_users_lost": int(((A < 10) & (N >= 10)).sum()),
                     "same_region_competitors_mean_A01_E7": [float(reg_a.mean()), float(reg_7.mean())],
                     "same_region_competitors_median_A01_E7": [float(np.median(reg_a)), float(np.median(reg_7))],
                     "share_users_fewer_same_region_competitors": float((reg_7 < reg_a).mean()),
                     "share_users_more_same_region_competitors": float((reg_7 > reg_a).mean()),
                     "target_f_rank_median_A01_E7": [float(np.median(fr_a)), float(np.median(fr_7))]}
    res = {"runs": out, "split": split}
    json.dump(res, open(f"{R}/geo/r22_e7.json", "w"), indent=1)
    for k, r in out.items():
        cv, ct_ = r["ci_valid_vs_A01"]["hr10"], r["ci_test_vs_A01"]["hr10"]
        print(f"{k}: valid {r['valid']['hr10']:.4f} (A01 {r['A01_valid']['hr10']:.4f}; {cv['diff']:+.4f} [{cv['ci95'][0]:+.4f}, {cv['ci95'][1]:+.4f}]) "
              f"| test {r['test']['hr10']:.4f} / {r['test']['ndcg10']:.4f} (A01 {r['A01_test']['hr10']:.4f} / {r['A01_test']['ndcg10']:.4f}; "
              f"{ct_['diff']:+.4f} [{ct_['ci95'][0]:+.4f}, {ct_['ci95'][1]:+.4f}]) | ms {r['ms_per_user']} | f {json.dumps(r['f_test'])} "
              f"| IG {json.dumps(r['f_info_gain'])} vs A01 {json.dumps(r['A01_f_info_gain'])}")
    print("split", json.dumps(split))


if __name__ == "__main__":
    main()
