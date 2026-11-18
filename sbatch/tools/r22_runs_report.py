"""mmgen addition (docs/MFD_REPORT.md §22.3 / 22.7 / 22.8 / 22.9): common report of round-22 training runs vs A01, CPU.
For every run mfd/<D>/r22_<V>/seed<S> with r21_eval output, V matching the given prefixes (default E1 E7 E8 E9):
 - valid / test HR@1/5/10, NDCG@5/10 (plain, K_f = 50), ms/user (H200, batch 64); paired bootstrap CIs vs A01 of the same seed
   (2,000 resamples, seed 0, users aligned by dataset index) on valid and test
 - f diagnostics: top-1, median rank of the true f (r21_eval, test), E4 information gain of the f distribution (fdump: IG =
   CE_marginal - NLL, normalised by CE_marginal)
 - E6 split (test; history = items[:-1], last 20): repeat-f (target f in history) vs new-f users: share, model / C HR@10, C - model;
   median rank of the true f on new-f users; share of competitors (f codes ranked above the true f) that are history f codes
   (all users and C-hit / model-miss users) -- also for A01 of the same seed (reference)
 - E4 within-region split: among C-hit / A01-miss users, rescued (model hit); all users rescued / lost vs A01; same-region
   competitors (f codes above the true f that contain an item of the target's C-text level-1 bucket) median, A01 vs model
The f space is the run's own SID set (E1: the 1+3+3 index; f codes are not comparable to A01's, competitor counts are).
--stage NAME PREFIX [PREFIX ..]: write the variants (matching PREFIX) whose Arts seed-42 valid HR@10 beats A01's to
geo/r22_<NAME>_stage.txt. Out: geo/r22_runs.json + printed tables.  Usage: python r22_runs_report.py [--stage e89 E8 E9] [PREFIX ..]
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
IDX = {"1+2+2": ".index_md122_a05_ep500_reassign.json", "1+3+3": ".index_md133_a06_ep500_reassign.json"}
KEYS = ("hr1", "hr5", "hr10", "ndcg5", "ndcg10")
_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = json.load(open(path))
    return _cache[path]


def ev(run_dir):
    e = json.load(open(f"{run_dir}/r21_eval/eval.json"))
    r = torch.load(f"{run_dir}/r21_eval/ranks.pt", weights_only=False)
    return e, {s: r[s]["plain"].numpy() for s in ("valid", "test")}


def f_analysis(D, fdump, sid, ranks, C):
    """IG, E6 split and per-user competitor info from a run's fdump (catalog f, renormalised log P)."""
    if not os.path.exists(fdump):
        return None
    E = torch.load(fdump, weights_only=False)["f"]
    lp = E["lp"].float().numpy()
    col = {t: j for j, t in enumerate(E["tokens"])}
    idx = load(f"{R}/data/{D}/{D}{IDX[sid]}")
    fo = {int(k): col[v[0]] for k, v in idx.items()}
    ct = {int(k): v[0] for k, v in load(f"{R}/data/{D}/{D}.index_lemb_mql.json").items()}
    seqs = list(load(f"{R}/data/{D}/{D}.inter.json").values())
    U, nF = lp.shape
    assert U == len(seqs) == len(ranks) == len(C)
    tg = np.array([s[-1] for s in seqs])
    tf = np.array([fo[t] for t in tg])
    q = np.bincount(tf, minlength=nF) / U
    ce = float(-np.log(q[tf]).mean())
    nll = float(-lp[np.arange(U), tf].mean())
    fwc = {}
    for i, c in ct.items():
        fwc.setdefault(c, np.zeros(nF, bool))[fo[i]] = True
    frank = np.empty(U, int)
    reg = np.empty(U, int)
    hist_share = np.full(U, np.nan)
    in_hist = np.zeros(U, bool)
    for u, s in enumerate(seqs):
        comp = lp[u] > lp[u, tf[u]]
        frank[u] = comp.sum()
        reg[u] = (comp & fwc[ct[int(tg[u])]]).sum()
        H = {fo[x] for x in s[:-1][-20:]}
        in_hist[u] = tf[u] in H
        if frank[u]:
            hm = np.zeros(nF, bool)
            hm[list(H - {tf[u]})] = True
            hist_share[u] = (comp & hm).sum() / frank[u]
    hit, c_hit = ranks < 10, C < 10
    grp = {}
    for g, m in (("repeat", in_hist), ("new_f", ~in_hist)):
        grp[g] = {"share": float(m.mean()), "hr10": float(hit[m].mean()), "C_hr10": float(c_hit[m].mean()),
                  "C_minus_model": float(c_hit[m].mean() - hit[m].mean()),
                  "true_f_rank_median": float(np.median(frank[m]))}
    cm = c_hit & ~hit
    out = {"info_gain": {"nll": nll, "ce_marginal": ce, "ig": ce - nll, "ig_norm": (ce - nll) / ce},
           "e6": dict(grp, competitors_from_history_all=float(np.nanmean(hist_share)),
                      competitors_from_history_C_hit_model_miss=float(np.nanmean(hist_share[cm])) if cm.any() else None)}
    return out, frank, reg


def main():
    argv = sys.argv[1:]
    stage = None
    if argv and argv[0] == "--stage":
        stage = (argv[1], argv[2:])
        argv = []
    prefixes = tuple(stage[1] if stage else (argv or ["E1", "E7", "E8", "E9"]))
    out, ref = {}, {}
    for p in sorted(glob.glob(f"{R}/mfd/*/r22_*/seed*/r21_eval/eval.json")):
        rd = p.rsplit("/r21_eval/", 1)[0]
        D, V, S = rd.split("/")[-3], rd.split("/")[-2][4:], int(rd.split("/")[-1][4:])
        if not V.startswith(prefixes) or not os.path.exists(f"{R}/mfd/{D}/r21_A01/seed{S}/r21_eval/ranks.pt"):
            continue
        sid = "1+3+3" if V == "E1" else "1+2+2"
        seqs = load(f"{R}/data/{D}/{D}.inter.json")
        n = len(seqs)
        boot = np.random.default_rng(0).integers(0, n, size=(2000, n))
        e, rk = ev(rd)
        ea, ra = ev(f"{R}/mfd/{D}/r21_A01/seed{S}")
        r = {"D": D, "var": V, "seed": S, "sid": sid, "gpu": e.get("gpu"),
             "valid": {k: e["valid"]["plain"][k] for k in KEYS}, "test": {k: e["test"]["plain"][k] for k in KEYS},
             "A01_valid": {k: ea["valid"]["plain"][k] for k in KEYS}, "A01_test": {k: ea["test"]["plain"][k] for k in KEYS},
             "ms_per_user": e["test"]["plain"].get("ms_per_user"), "A01_ms_per_user": ea["test"]["plain"].get("ms_per_user"),
             "ci_valid_vs_A01": paired(rk["valid"], ra["valid"], boot), "ci_test_vs_A01": paired(rk["test"], ra["test"], boot),
             "f_test": e["test"].get("f"), "A01_f_test": ea["test"].get("f")}
        if not stage:
            tg = np.array([s[-1] for s in seqs.values()])
            C = rank_of(c_lists_seed(D, S), tg)
            fa = f_analysis(D, f"{R}/geo/r22/{D}/fdump_r22_{V}_s{S}.pt", sid, rk["test"], C)
            key_a = (D, S)
            if key_a not in ref:
                ref[key_a] = f_analysis(D, f"{R}/geo/r22/{D}/fdump_r21_A01_s{S}.pt", "1+2+2", ra["test"], C)
            fr = ref[key_a]
            r["A01_f"] = fr[0] if fr else None
            if fa:
                r["f"] = fa[0]
                A, N = ra["test"], rk["test"]
                sel = (C < 10) & (A >= 10)
                r["e4_split"] = {"c_hit_a01_miss": int(sel.sum()), "rescued_in_set": int((N[sel] < 10).sum()),
                                 "all_users_rescued": int(((A >= 10) & (N < 10)).sum()), "all_users_lost": int(((A < 10) & (N >= 10)).sum())}
                if fr:
                    r["e4_split"]["same_region_competitors_median_A01_model"] = [float(np.median(fr[2][sel])), float(np.median(fa[2][sel]))]
                    r["e4_split"]["true_f_rank_median_A01_model"] = [float(np.median(fr[1][sel])), float(np.median(fa[1][sel]))]
        out[f"{D} {V} seed {S}"] = r
    if stage:
        name = stage[0]
        win = sorted(r["var"] for r in out.values() if r["D"] == "Arts" and r["seed"] == 42 and r["valid"]["hr10"] > r["A01_valid"]["hr10"])
        open(f"{R}/geo/r22_{name}_stage.txt", "w").write("\n".join(win) + ("\n" if win else ""))
        print(f"{name}: variants beating A01 on Arts seed-42 valid HR@10:", win or "none")
        for k, r in out.items():
            print(f"  {k}: valid {r['valid']['hr10']:.4f} vs A01 {r['A01_valid']['hr10']:.4f}")
        return
    json.dump(out, open(f"{R}/geo/r22_runs.json", "w"), indent=1)
    for k, r in out.items():
        cv, ct_ = r["ci_valid_vs_A01"], r["ci_test_vs_A01"]
        print(f"\n=== {k} ({r['sid']})")
        print(f"valid HR@10 {r['valid']['hr10']:.4f} (A01 {r['A01_valid']['hr10']:.4f}; {cv['hr10']['diff']:+.4f} "
              f"[{cv['hr10']['ci95'][0]:+.4f}, {cv['hr10']['ci95'][1]:+.4f}])")
        print("test  " + " ".join(f"{m} {r['test'][m]:.4f}" for m in KEYS) + " | A01 " + " ".join(f"{r['A01_test'][m]:.4f}" for m in KEYS))
        for m in ("hr10", "ndcg10"):
            print(f"  test {m} vs A01 {ct_[m]['diff']:+.4f} [{ct_[m]['ci95'][0]:+.4f}, {ct_[m]['ci95'][1]:+.4f}]")
        print(f"ms/user {r['ms_per_user']} (A01 {r['A01_ms_per_user']}) | f {json.dumps(r['f_test'])}")
        for name, f in (("model", r.get("f")), ("A01", r.get("A01_f"))):
            if f:
                g = f["e6"]
                print(f"  {name:5s} IG {f['info_gain']['ig_norm']:.3f} | repeat {g['repeat']['share']:.3f}: HR@10 {g['repeat']['hr10']:.4f} "
                      f"C-m {g['repeat']['C_minus_model']:+.4f} | new-f {g['new_f']['share']:.3f}: HR@10 {g['new_f']['hr10']:.4f} "
                      f"C-m {g['new_f']['C_minus_model']:+.4f} f-rank med {g['new_f']['true_f_rank_median']:.0f} | comp from hist "
                      f"{g['competitors_from_history_all']:.3f} / C-hit-miss {g['competitors_from_history_C_hit_model_miss']}")
        print("  E4 split", json.dumps(r.get("e4_split")))


if __name__ == "__main__":
    main()
