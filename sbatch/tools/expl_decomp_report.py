"""§20 Task 0 (docs/MFD_REPORT.md): error decomposition of P1a seed 42 vs C seed 42 on the test set. CPU, inside the env
(MACREC=<macrec dir>), next to expl_top10.py.

Inputs: mmgen_runs/expl/<D>/decomp_<tag>.pt (macrec/expl_decomp.py: every catalog item scored once), topk_<list tag>.pt
(P1a's top list, for top-10 diversity), C's saved beams (rebuilt by expl_top10.c_lists, asserted there against C's results).
Per test user: P1a / C-merged / C-text / C-image hit@10, target f rank among all f, kept or not (K_f), rank, rank within its f
bucket, bucket size, target popularity (train interactions = occurrences in items[:-2] over all users) and history length
(min(len - 1, 20)). Users aligned by dataset index; targets checked against <D>.inter.json.
Reports: (1) P1a misses = (a) f not kept + (b) f kept, rank >= 10; (2) the same on C-hit/P1a-miss and P1a-hit/C-miss;
(3) oracles: true f always kept / true f with the top f's log P(f); (4) C-hit/P1a-miss by popularity quintile, history-length
bucket, bucket-size quartile (each with the P1a and C HR@10 of all users in that bucket); (5) distinct f in the top 10.
Usage: python expl_decomp_report.py <D> <decomp tag> <topk tag>  -> mmgen_runs/expl/<D>/decomp.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expl_top10 import R, c_lists, rank_of  # noqa: E402

BIG = 10 ** 6


def pct(x):
    return float(np.mean(x)) if len(x) else float("nan")


def split_misses(sel, kept, r):
    n = int(sel.sum())
    a = int((sel & ~kept).sum())
    b = int((sel & kept & (r >= 10)).sum())
    return {"users": n, "f_not_kept": a, "f_kept_rank_ge10": b, "f_not_kept_share": a / max(n, 1), "f_kept_rank_ge10_share": b / max(n, 1)}


def buckets(name, key, edges, labels, cm, P, C):
    out = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (key >= lo) & (key < hi)
        out[lab] = {"users": int(m.sum()), "c_hit_p1a_miss": int((m & cm).sum()), "share_of_set": float((m & cm).sum() / max(cm.sum(), 1)),
                    "p1a_hr10": pct(P[m]), "c_hr10": pct(C[m])}
    return out


def main():
    D, tag, ltag = sys.argv[1], sys.argv[2], sys.argv[3]
    X = torch.load(f"{R}/expl/{D}/decomp_{tag}.pt", weights_only=False)
    t = {k: v.numpy() for k, v in X["test"].items()}
    K = X["setting"]["k_f"]
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    users = list(inter)
    assert np.array_equal(t["targets"], np.array([inter[u][-1] for u in users])), "user order"
    pop = Counter(i for u in users for i in inter[u][:-2])
    tpop = np.array([pop.get(i, 0) for i in t["targets"]])
    hlen = np.array([min(len(inter[u]) - 1, 20) for u in users])
    C = c_lists(D, "test")
    tg = t["targets"]
    rc = {n: rank_of(C[n], tg) for n in C}
    r = t["rank"]
    kept = t["f_rank"] < K
    P, Cm = r < 10, rc["merged"] < 10
    # consistency with the reported ranks of the run at this setting: (ii) = eval_ii, else the (iii) sweep's chosen setting
    b = f"{R}/mfd/{D}/mfd122_mv_geoP1a/seed42"
    st = X["setting"]
    if (st["k_f"], st["w_f"], st["w_t"], st["w_i"]) == (100, 1.0, 0.5, 0.5):
        E = torch.load(f"{b}/eval_ii/ranked_test.pt", weights_only=False)
        m = E["joint"][0][:, :10] == E["targets"][:, None]
        rep = torch.where(m.any(1), m.int().argmax(1), torch.full_like(E["targets"], 10)).numpy()
    else:
        ch = json.load(open(f"{b}/sweep/chosen.json"))
        rep = torch.load(f"{b}/sweep/apply_test_ranks.pt", weights_only=False)[ch["name"]].clamp(max=10).numpy()
    res_consistency = float((np.minimum(r, 10) == rep).mean())
    assert res_consistency > 0.999, f"ranks differ from the reported run ({res_consistency})"
    res = {"dataset": D, "setting": X["setting"], "users": int(len(r)), "share_users_rank_as_reported": res_consistency,
           "hr10": {"P1a": pct(P), "C_merged": pct(Cm), "C_text": pct(rc["text"] < 10), "C_image": pct(rc["image"] < 10)},
           "f_recall_at_K": pct(kept), "f_top1": pct(t["f_rank"] == 0), "f_top20": pct(t["f_rank"] < 20)}
    res["1_p1a_misses"] = split_misses(~P, kept, r)
    cm, pm = Cm & ~P, P & ~Cm
    res["2_c_hit_p1a_miss"] = split_misses(cm, kept, r)
    res["2_p1a_hit_c_miss_users"] = int(pm.sum())
    res["2_both_hit"], res["2_both_miss"] = int((P & Cm).sum()), int((~P & ~Cm).sum())
    for name, sel in (("c_hit_p1a_miss", cm), ("all_misses", ~P)):
        rr, fr = r[sel & kept], t["f_rank"][sel & ~kept]
        res[f"2_{name}_detail"] = {
            "kept_rank_11_20": pct((rr >= 10) & (rr < 20)), "kept_rank_21_50": pct((rr >= 20) & (rr < 50)),
            "kept_rank_51_100": pct((rr >= 50) & (rr < 100)), "kept_rank_gt100": pct(rr >= 100),
            "kept_within_bucket_rank_lt10": pct(t["rank_in_bucket"][sel & kept] < 10),
            "notkept_f_rank_median": float(np.median(fr)) if len(fr) else None,
            "notkept_f_rank_100_150": pct((fr >= 100) & (fr < 150)), "notkept_f_rank_ge150": pct(fr >= 150)}
    res["3_oracle"] = {"P1a": pct(P), "true_f_always_kept": pct(t["rank_oracle_keep"] < 10),
                       "true_f_ranked_first": pct(t["rank_oracle_top"] < 10),
                       "true_f_ranked_first_hr1": pct(t["rank_oracle_top"] < 1), "P1a_hr1": pct(r < 1),
                       "within_bucket_top10_given_any": pct(t["rank_in_bucket"] < 10)}
    q = np.quantile(tpop, [0, .2, .4, .6, .8, 1.0]); q[-1] += 1
    res["4_by_popularity_quintile"] = buckets("pop", tpop, q, [f"Q{j+1} [{int(q[j])}, {int(q[j+1])})" for j in range(5)], cm, P, Cm)
    he = [1, 4, 6, 11, 20, 21]
    res["4_by_history_length"] = buckets("hlen", hlen, he, ["1-3", "4-5", "6-10", "11-19", "20 (capped)"], cm, P, Cm)
    bq = np.quantile(t["bucket"], [0, .25, .5, .75, 1.0]); bq[-1] += 1
    res["4_by_bucket_size_quartile"] = buckets("bucket", t["bucket"], bq, [f"Q{j+1} [{int(bq[j])}, {int(bq[j+1])})" for j in range(4)], cm, P, Cm)
    L = torch.load(f"{R}/expl/{D}/topk_{ltag}.pt", weights_only=False)
    top = L["test"]["ids"].numpy()[:, :10]
    assert np.array_equal(L["test"]["targets"].numpy(), tg)
    fcode = L["item_f"].numpy()
    def dist(lists):
        c = np.where(lists >= 0, fcode[np.maximum(lists, 0)], -1)
        d = np.array([len(set(x[x >= 0])) for x in c])
        return {"mean": float(d.mean()), "median": float(np.median(d)), "hist": {str(k): float((d == k).mean()) for k in range(1, 11)}}
    res["5_distinct_f_top10"] = {"P1a": dist(top), "C_merged": dist(C["merged"])}
    json.dump(res, open(f"{R}/expl/{D}/decomp.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
