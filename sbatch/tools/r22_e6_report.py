"""mmgen addition (docs/MFD_REPORT.md §22.6, E6): are A01's within-region competitors the user's own history f codes? CPU.
Inputs: geo/r22/<D>/fdump_r21_A01_s<S>.pt (A01's full-view log P(f) per TEST user, macrec/r22_e4_first.py), A01's plain K_f = 50
test ranks (r21_eval), C's merged test list of the same seed, the locked v1 1+2+2 index (item -> f) and C's text index (level 1).
Users in dataset order; test history = items[:-1], last 20 (MACRec's max_his_len), recency 1 = the most recent item.
Competitors = the f codes A01 ranks ABOVE the target's f (log P(f) strictly greater).
 1 share of competitors that are history f codes (any / last 1 / 3 / 5 items); chance = the same share for random f codes drawn
   from q (the f marginal of training interactions, items[:-2] of every user) excluding the target f: sum_{f in H, f != f*} q(f) /
   (1 - q(f*)); users with >= 1 competitor. Also: share of the user's distinct non-target history f codes that are competitors.
 2 target f in history: rate, most recent position (1, 2-3, 4-5, 6-10, 11-20), A01's rank of the target f (median, top-1, top-10)
   when in history vs not
 3 repeat (target f in history) vs new f: share of users, A01 HR@10, C HR@10, gap, contribution of each group to the overall gap
 4 region link (E4 definition: a competitor is in the target's C-text region when it contains >= 1 catalog item of the target's
   C-text level-1 bucket): share of in-region and out-of-region competitors that are history f codes
User sets: all, C-hit / A01-miss, A01-hit / C-miss (hit = rank < 10). Out: geo/r22_e6.json + printed tables.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round20_collect import c_lists_seed, rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
MD = ".index_md122_a05_ep500_reassign.json"
BINS = (("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-10", 6, 10), ("11-20", 11, 20))


def mean(x):
    return float(np.mean(x)) if len(x) else None


def run(D, S, path):
    E = torch.load(path, weights_only=False)["f"]
    lp = E["lp"].float().numpy()
    col = {t: j for j, t in enumerate(E["tokens"])}
    nF = lp.shape[1]
    idx = json.load(open(f"{R}/data/{D}/{D}{MD}"))
    fo = {int(k): col[v[0]] for k, v in idx.items()}
    ct = {int(k): v[0] for k, v in json.load(open(f"{R}/data/{D}/{D}.index_lemb_mql.json")).items()}
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    seqs = list(inter.values())
    assert len(seqs) == lp.shape[0]
    tg = np.array([s[-1] for s in seqs])
    q = np.zeros(nF)
    for s in seqs:
        for x in s[:-2]:
            q[fo[x]] += 1
    q /= q.sum()
    f_with_ct = {}                                                   # C-text code -> bool mask over f codes containing it
    for i, c in ct.items():
        f_with_ct.setdefault(c, np.zeros(nF, bool))[fo[i]] = True
    A = torch.load(f"{R}/mfd/{D}/r21_A01/seed{S}/r21_eval/ranks.pt", weights_only=False)["test"]["plain"].numpy()
    C = rank_of(c_lists_seed(D, S), tg)
    a_hit, c_hit = A < 10, C < 10
    sets = {"all": np.ones(len(tg), bool), "C_hit_A01_miss": c_hit & ~a_hit, "A01_hit_C_miss": a_hit & ~c_hit}
    U = len(tg)
    tf = np.array([fo[t] for t in tg])
    frank = np.array([(lp[u] > lp[u, tf[u]]).sum() for u in range(U)])
    per = {k: np.full(U, np.nan) for k in ("hist", "last1", "last3", "last5", "chance", "hist_f_above", "inreg_hist", "outreg_hist")}
    in_hist = np.zeros(U, bool)
    recent = np.zeros(U, int)
    for u, s in enumerate(seqs):
        H = [fo[x] for x in s[:-1][-20:]][::-1]                      # newest first
        comp = lp[u] > lp[u, tf[u]]
        if tf[u] in H:
            in_hist[u] = True
            recent[u] = H.index(tf[u]) + 1
        nc = comp.sum()
        hs = set(H) - {tf[u]}
        if hs:
            per["hist_f_above"][u] = np.mean([comp[f] for f in hs])
        if nc == 0:
            continue
        for k, n in (("hist", 20), ("last1", 1), ("last3", 3), ("last5", 5)):
            m = np.zeros(nF, bool); m[list(set(H[:n]) - {tf[u]})] = True
            per[k][u] = (comp & m).sum() / nc
        per["chance"][u] = q[list(hs)].sum() / (1 - q[tf[u]]) if hs else 0.0
        hm = np.zeros(nF, bool); hm[list(hs)] = True
        reg = f_with_ct[ct[int(tg[u])]] & comp
        if reg.any():
            per["inreg_hist"][u] = (reg & hm).sum() / reg.sum()
        if (comp & ~reg).any():
            per["outreg_hist"][u] = (comp & ~reg & hm).sum() / (comp & ~reg).sum()
    out = {}
    gap_all = c_hit.mean() - a_hit.mean()
    for sname, sel in sets.items():
        r = {"users": int(sel.sum())}
        r["1_competitors_from_history"] = {k: mean(per[k][sel & ~np.isnan(per[k])]) for k in ("hist", "last1", "last3", "last5", "chance")}
        r["1_competitors_from_history"]["users_with_competitors"] = int((sel & ~np.isnan(per["hist"])).sum())
        r["1_share_of_history_f_ranked_above_target"] = mean(per["hist_f_above"][sel & ~np.isnan(per["hist_f_above"])])
        ih = sel & in_hist
        r["2_target_f_in_history"] = {"rate": float(ih.sum() / max(sel.sum(), 1)),
                                      "most_recent_position": {b: float(((recent >= lo) & (recent <= hi) & ih).sum() / max(ih.sum(), 1))
                                                               for b, lo, hi in BINS},
                                      "f_rank_in_history": {"median": float(np.median(frank[ih])) if ih.any() else None,
                                                            "top1": mean(frank[ih] == 0), "top10": mean(frank[ih] < 10)},
                                      "f_rank_not_in_history": {"median": float(np.median(frank[sel & ~in_hist])) if (sel & ~in_hist).any() else None,
                                                                "top1": mean(frank[sel & ~in_hist] == 0), "top10": mean(frank[sel & ~in_hist] < 10)}}
        r["4_region_link"] = {"in_region_competitors_from_history": mean(per["inreg_hist"][sel & ~np.isnan(per["inreg_hist"])]),
                              "out_of_region_competitors_from_history": mean(per["outreg_hist"][sel & ~np.isnan(per["outreg_hist"])])}
        out[sname] = r
    grp = {}
    for g, m in (("repeat", in_hist), ("new_f", ~in_hist)):
        grp[g] = {"share": float(m.mean()), "A01_hr10": float(a_hit[m].mean()), "C_hr10": float(c_hit[m].mean()),
                  "gap_C_minus_A01": float(c_hit[m].mean() - a_hit[m].mean()),
                  "contribution_to_overall_gap": float(m.mean() * (c_hit[m].mean() - a_hit[m].mean()))}
    out["3_repeat_vs_new"] = dict(grp, overall_gap_C_minus_A01=float(gap_all))
    return out


def show(key, r):
    print(f"\n=== {key}")
    for s in ("all", "C_hit_A01_miss", "A01_hit_C_miss"):
        x = r[s]
        c, t = x["1_competitors_from_history"], x["2_target_f_in_history"]
        print(f"{s:15s} n={x['users']:6d} | competitors from history: any {c['hist']:.3f} last1 {c['last1']:.3f} last3 {c['last3']:.3f} "
              f"last5 {c['last5']:.3f} chance {c['chance']:.3f} | history f above target {x['1_share_of_history_f_ranked_above_target']:.3f}")
        print(f"{'':15s} target f in history {t['rate']:.3f} recency {json.dumps({k: round(v, 3) for k, v in t['most_recent_position'].items()})} "
              f"| f rank in hist {t['f_rank_in_history']} | not {t['f_rank_not_in_history']}")
        print(f"{'':15s} region link: in-region competitors from history {x['4_region_link']['in_region_competitors_from_history']}, "
              f"out-of-region {x['4_region_link']['out_of_region_competitors_from_history']}")
    print("repeat vs new:", json.dumps(r["3_repeat_vs_new"]))


def main():
    out = {}
    for p in sorted(glob.glob(f"{R}/geo/r22/*/fdump_r21_A01_s*.pt")):
        D, S = p.split("/")[-2], int(p.rsplit("_s", 1)[1][:-3])
        key = f"{D} seed {S}"
        out[key] = run(D, S, p)
        show(key, out[key])
    json.dump(out, open(f"{R}/geo/r22_e6.json", "w"), indent=1)


if __name__ == "__main__":
    main()
