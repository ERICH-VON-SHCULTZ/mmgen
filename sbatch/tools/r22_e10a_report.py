"""mmgen addition (docs/MFD_REPORT.md §22.10, E10a): why does A01 help only on Arts? CPU, no model, existing checkpoints only.
 1 view disagreement at the f step (geo/r22/<D>/views_<RUN>_s<S>.pt from macrec/r22_e10_views.py, TEST users, renormalised
   over the catalog f codes): mean generalised JS of the 3 views = H(pbar) - mean_v H(p_v) (nats), pbar = mean of the 3 view
   probabilities (= A01's teacher); mean KL(pbar || p_full); share of users whose argmax pbar != argmax p_full. Per model
   (P1a, G0, G0acc, A01), dataset, seed; also on repeat-f / new-f users. Cross-check: A01 full view = the E6 fdump.
 2 E6 split of A01 - P1a (TEST; repeat-f = target f among the f codes of history[-20:]): share and HR@10 difference with a
   paired bootstrap CI inside each group; also split by P1a's view disagreement (ensemble top-f != full top-f).
 3 validation vs test: A01 - P1a HR@10 / NDCG@10 with paired CIs on both splits (and the E6 split on validation).
 4 G0 - P1a and A01 - G0 (and G0acc) where G0 exists, valid and test.
Bootstrap: 2,000 resamples of users, seed 0, aligned by dataset index (plain K_f = 50 ranks from r21_eval/ranks.pt).
Out: geo/r22_e10a.json + printed tables.  Usage: python r22_e10a_report.py
"""
import json
import os

import numpy as np
import torch

R = "/scratch/yw8866/mmgen_runs"
IDX = ".index_md122_a05_ep500_reassign.json"
DS = ("Arts", "Instruments", "Games")
SEEDS = (42, 43, 44)
MODELS = {"P1a": "mfd122_mv_geoP1a", "G0": "r21_G0", "G0acc": "r21_G0acc", "A01": "r21_A01"}
NB = 2000


def ranks(D, run, S):
    f = f"{R}/mfd/{D}/{run}/seed{S}/r21_eval/ranks.pt"
    if not os.path.exists(f):
        return None
    r = torch.load(f, weights_only=False)
    return {s: r[s]["plain"].numpy() for s in ("valid", "test")}


def hr(r):
    return (r < 10).astype(float)


def ndcg(r):
    return np.where(r < 10, 1.0 / np.log2(np.minimum(r, 10) + 2.0), 0.0)


def ci(d, rng):
    """mean and 95% percentile CI of a per-user difference vector (paired bootstrap)."""
    n = len(d)
    if n == 0:
        return {"diff": None, "ci95": None, "n": 0}
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NB)])
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(n)}


def pair(a, b, mask=None):
    m = np.ones(len(a), bool) if mask is None else mask
    out = {}
    for name, f in (("hr10", hr), ("ndcg10", ndcg)):
        out[name] = ci((f(a) - f(b))[m], np.random.default_rng(0))
    out["won"], out["lost"] = int(((a < 10) & (b >= 10) & m).sum()), int(((a >= 10) & (b < 10) & m).sum())
    return out


def repeat_flags(D):
    idx = json.load(open(f"{R}/data/{D}/{D}{IDX}"))
    fo = {int(k): v[0] for k, v in idx.items()}
    seqs = list(json.load(open(f"{R}/data/{D}/{D}.inter.json")).values())
    test = np.array([fo[s[-1]] in {fo[x] for x in s[:-1][-20:]} for s in seqs])
    valid = np.array([fo[s[-2]] in {fo[x] for x in s[:-2][-20:]} for s in seqs])
    return {"test": test, "valid": valid}, seqs, fo


def entropy(lp):
    return -(np.exp(lp) * lp).sum(-1)


def views(D, run, S, rep):
    f = f"{R}/geo/r22/{D}/views_{run}_s{S}.pt"
    if not os.path.exists(f):
        return None, None
    V = torch.load(f, weights_only=False)
    lf, lt, li = (V[v].double().numpy() for v in ("full", "text", "image"))
    pbar = (np.exp(lf) + np.exp(lt) + np.exp(li)) / 3
    lbar = np.log(np.clip(pbar, 1e-300, None))
    js = entropy(lbar) - (entropy(lf) + entropy(lt) + entropy(li)) / 3
    klbf = (pbar * (lbar - lf)).sum(-1)
    diff_top = lbar.argmax(-1) != lf.argmax(-1)
    pair_js = {}
    for (a, la), (b, lb) in ((("full", lf), ("text", lt)), (("full", lf), ("image", li)), (("text", lt), ("image", li))):
        m = np.log(np.clip((np.exp(la) + np.exp(lb)) / 2, 1e-300, None))
        pair_js[f"{a}-{b}"] = float((entropy(m) - (entropy(la) + entropy(lb)) / 2).mean())
    out = {}
    for g, m in (("all", np.ones(len(js), bool)), ("repeat", rep), ("new_f", ~rep)):
        out[g] = {"js3": float(js[m].mean()), "kl_teacher_full": float(klbf[m].mean()), "top_f_differs": float(diff_top[m].mean()),
                  "H_full": float(entropy(lf)[m].mean()), "n": int(m.sum())}
    out["pairwise_js"] = pair_js
    out["view_top1_agree"] = {"full=text": float((lf.argmax(-1) == lt.argmax(-1)).mean()),
                              "full=image": float((lf.argmax(-1) == li.argmax(-1)).mean())}
    fd = f"{R}/geo/r22/{D}/fdump_{run}_s{S}.pt"
    if os.path.exists(fd):
        E = torch.load(fd, weights_only=False)["f"]
        assert E["tokens"] == V["tokens"]
        out["fdump_full_max_abs_diff"] = float((E["lp"].double().numpy() - lf).__abs__().max())
    return out, diff_top


def fmt(c):
    return "   n/a" if c["diff"] is None else f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


def main():
    res = {"views": {}, "e6_split": {}, "valid_vs_test": {}, "g0": {}}
    for D in DS:
        rep, seqs, fo = repeat_flags(D)
        for S in SEEDS:
            rk = {k: ranks(D, v, S) for k, v in MODELS.items()}
            for k, v in MODELS.items():
                vs, dtop = views(D, v, S, rep["test"])
                if vs:
                    res["views"][f"{D} {k} s{S}"] = vs
                if k == "P1a":
                    dtop_p1a = dtop
            a, p = rk["A01"], rk["P1a"]
            if a and p:
                assert len(a["test"]) == len(p["test"]) == len(seqs)
                e6 = {}
                for split in ("valid", "test"):
                    r_ = rep[split]
                    e6[split] = {"repeat_share": float(r_.mean()),
                                 "repeat": pair(a[split], p[split], r_), "new_f": pair(a[split], p[split], ~r_),
                                 "P1a_hr10_repeat": float(hr(p[split])[r_].mean()), "P1a_hr10_new_f": float(hr(p[split])[~r_].mean())}
                if dtop_p1a is not None:
                    e6["test"]["P1a_views_disagree"] = pair(a["test"], p["test"], dtop_p1a)
                    e6["test"]["P1a_views_agree"] = pair(a["test"], p["test"], ~dtop_p1a)
                    e6["test"]["P1a_disagree_share"] = float(dtop_p1a.mean())
                res["e6_split"][f"{D} s{S}"] = e6
                res["valid_vs_test"][f"{D} s{S}"] = {sp: pair(a[sp], p[sp]) for sp in ("valid", "test")}
            for x, y in (("G0", "P1a"), ("A01", "G0"), ("G0acc", "P1a"), ("A01", "G0acc")):
                if rk[x] and rk[y]:
                    res["g0"][f"{D} s{S} {x}-{y}"] = {sp: pair(rk[x][sp], rk[y][sp]) for sp in ("valid", "test")}
    json.dump(res, open(f"{R}/geo/r22_e10a.json", "w"), indent=1)

    print("\n== 1 view disagreement at the f step (TEST; nats) ==")
    print(f"{'':26s} {'JS3':>7s} {'KL(t||full)':>11s} {'top-f diff':>10s} {'H_full':>7s} | JS3 rep/new | KL rep/new | fdump chk")
    for k, v in res["views"].items():
        a_, r_, n_ = v["all"], v["repeat"], v["new_f"]
        print(f"{k:26s} {a_['js3']:7.4f} {a_['kl_teacher_full']:11.4f} {a_['top_f_differs']:10.3f} {a_['H_full']:7.3f} | "
              f"{r_['js3']:.4f}/{n_['js3']:.4f} | {r_['kl_teacher_full']:.4f}/{n_['kl_teacher_full']:.4f} | {v.get('fdump_full_max_abs_diff', '-')}")
    print("seed means:")
    for D in DS:
        for k in MODELS:
            vs = [res["views"][f"{D} {k} s{S}"]["all"] for S in SEEDS if f"{D} {k} s{S}" in res["views"]]
            if vs:
                print(f"  {D:11s} {k:5s} JS3 {np.mean([x['js3'] for x in vs]):.4f}  KL {np.mean([x['kl_teacher_full'] for x in vs]):.4f}  "
                      f"top-f diff {np.mean([x['top_f_differs'] for x in vs]):.3f}  (n seeds {len(vs)})")
    print("\n== 2 E6 split of A01 - P1a, HR@10 (TEST) ==")
    for k, e in res["e6_split"].items():
        t = e["test"]
        line = (f"{k:16s} repeat {t['repeat_share']:.3f}: {fmt(t['repeat']['hr10'])} | new-f: {fmt(t['new_f']['hr10'])}")
        if "P1a_views_disagree" in t:
            line += f" | P1a views disagree ({t['P1a_disagree_share']:.3f}): {fmt(t['P1a_views_disagree']['hr10'])} agree: {fmt(t['P1a_views_agree']['hr10'])}"
        print(line)
    print("\n== 3 A01 - P1a, valid vs test ==")
    for k, e in res["valid_vs_test"].items():
        print(f"{k:16s} valid HR@10 {fmt(e['valid']['hr10'])} NDCG@10 {fmt(e['valid']['ndcg10'])} | "
              f"test HR@10 {fmt(e['test']['hr10'])} NDCG@10 {fmt(e['test']['ndcg10'])}")
    for k, e in res["e6_split"].items():
        v = e["valid"]
        print(f"  {k:16s} VALID E6: repeat {v['repeat_share']:.3f}: {fmt(v['repeat']['hr10'])} | new-f: {fmt(v['new_f']['hr10'])}")
    print("\n== 4 grouping vs KL ==")
    for k, e in res["g0"].items():
        print(f"{k:26s} valid HR@10 {fmt(e['valid']['hr10'])} | test HR@10 {fmt(e['test']['hr10'])} NDCG@10 {fmt(e['test']['ndcg10'])}")


if __name__ == "__main__":
    main()
