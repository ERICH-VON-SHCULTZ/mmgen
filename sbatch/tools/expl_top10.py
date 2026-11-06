"""Top-10 diversity (docs/MFD_REPORT.md §18). CPU, inside the env (MACREC=<macrec dir>), next to plugin_paired.py.

Our lists: mmgen_runs/expl/<D>/topk_<tag>.pt (macrec/expl_topk.py). C seed 42: text, image and merged lists rebuilt from
save_seq{rec,image}_20.json with ensemble.py's rule (single lists by beam score; merged: an item in both lists scores the
mean of its two scores + 1, then sorted by score); each list's HR@10 / NDCG@10 is asserted against C's results files.
1a (test): HR@k (k = 1, 2, 3, 5, 10); distinct first codes in the top 10 and the share under the top-ranked one; item overlap
with C merged; for users C merged hits in its top 10 and we miss: target f among the f codes of our top 10, target's rank in
our full scored list.
1b (--fix): per-f cap m in {1, 2, 3, 5} and greedy penalty lambda (score - lambda * items already chosen under the same f),
one setting per model chosen on VALIDATION HR@10 (tie NDCG@10), applied once to test; paired CIs vs the same model
without the fix and vs C merged. Writes mmgen_runs/expl/<D>/top10.json.
Usage: python expl_top10.py <D> tag [tag ...] [--fix]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plugin_paired  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
KS = (1, 2, 3, 5, 10)
CAPS = (1, 2, 3, 5)
LAMBDAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BIG = 10 ** 6


def hr(r, k):
    return float((r < k).mean())


def ndcg10(r):
    return float(np.where(r < 10, 1 / np.log2(r + 2.0), 0).mean())


def c_lists(D, split):
    """{list: (U, 10) item ids, -1 padded}, for text / image / merged, users in dataset order."""
    ev = f"{R}/baselines/{D}/mql/seed42/eval/{split}"
    infos, U = {}, None
    for task, fn in (("seqrec", ".index_lemb_mql.json"), ("seqimage", ".index_vitemb_mql.json")):
        sv = json.load(open(f"{ev}/save_{task}_20.json"))
        idx = json.load(open(f"{R}/data/{D}/{D}{fn}"))
        s2i = {"".join(v): int(k) for k, v in idx.items()}
        U = len(sv["all_users"])
        B = len(sv["all_outputs"]) // U
        rows = {}
        for k, u in enumerate(sv["all_users"]):
            o = [x.strip().replace(" ", "") for x in sv["all_outputs"][k * B:(k + 1) * B]]
            s = sv["all_scores"][k * B:(k + 1) * B]
            rows[u] = [(s2i[a], b) for a, b in zip(o, s) if a in s2i]
        infos[task] = rows
    out = {n: np.full((U, 10), -1, dtype=np.int64) for n in ("text", "image", "merged")}
    for u in range(U):
        t, i = infos["seqrec"][u], infos["seqimage"][u]
        for name, x in (("text", t), ("image", i)):
            top = [a for a, _ in sorted(x, key=lambda p: p[1], reverse=True)][:10]
            out[name][u, :len(top)] = top
        sc = {}
        for j in range(20):                                       # ensemble.get_topk_results_ensemble's order of updates
            for lst in (t, i):
                if j < len(lst):
                    a, b = lst[j]
                    sc[a] = (b + sc[a]) / 2 + 1 if a in sc else b
        top = [a for a, _ in sorted(sc.items(), key=lambda p: p[1], reverse=True)][:10]
        out["merged"][u, :len(top)] = top
    return out


def rank_of(lists, tgt):
    m = lists == tgt[:, None]
    return np.where(m.any(1), m.argmax(1), BIG)


def conc(lists, code):
    """Mean distinct first codes in the top 10 and share of the listed items under the top-ranked item's code."""
    valid = lists >= 0
    c = np.where(valid, code[np.maximum(lists, 0)], -1)
    distinct = np.array([len(set(x[x >= 0])) for x in c])
    share = (c == c[:, :1]).sum(1) / np.maximum(valid.sum(1), 1)
    return {"distinct_mean": float(distinct.mean()), "distinct_median": float(np.median(distinct)),
            "top_code_share": float(share.mean()), "share_users_all_ten_same_code": float((distinct == 1).mean())}


def paired(a, b):
    boot = np.random.default_rng(0).integers(0, len(a), size=(2000, len(a)))
    out = {}
    for n, f in (("hr10", lambda r: (r < 10).astype(float)), ("ndcg10", lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.))):
        d = f(a) - f(b)
        bs = d[boot].mean(1)
        out[n] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def fm(x):
    return f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def cap_list(ids, fc, m):
    """Next-best items with at most m per f; if the list runs out, positions are filled by the skipped items in order."""
    ids_t, fc_t = torch.from_numpy(ids), torch.from_numpy(fc)
    U, N = ids.shape
    occ = torch.zeros((U, N), dtype=torch.long)
    for j in range(N):                                           # occurrence number of each item's f within its user list
        same = (fc_t[:, :j] == fc_t[:, j:j + 1]) if j else torch.zeros((U, 0), dtype=torch.bool)
        occ[:, j] = same.sum(1)
    key = (occ >= m).long() * (N + 1) + torch.arange(N)[None]     # allowed items first, in rank order
    pick = key.argsort(1)[:, :10]
    return ids_t.gather(1, pick).numpy()


def penalty_list(ids, sc, fc, lam):
    S = torch.from_numpy(sc).double().clone()
    F = torch.from_numpy(fc)
    U, N = S.shape
    a = torch.arange(U)
    chosen = torch.zeros(U, N, dtype=torch.bool)
    picks = []
    for _ in range(10):
        j = S.masked_fill(chosen, float("-inf")).argmax(1)
        picks.append(j)
        chosen[a, j] = True
        S = S - lam * (F == F[a, j][:, None]).double()
    p = torch.stack(picks, 1)
    return torch.from_numpy(ids).gather(1, p).numpy()


def main():
    D = sys.argv[1]
    fix = "--fix" in sys.argv
    tags = [t for t in sys.argv[2:] if t != "--fix"]
    idx = json.load(open(f"{R}/data/{D}/{D}.index_md122_a05_ep500_reassign.json"))
    code = lambda fn: (lambda d: np.array([int(d[str(i)][0].split("_")[1][:-1]) for i in range(len(d))]))(json.load(open(f"{R}/data/{D}/{D}{fn}")))
    fcode, ctext, cimage = code(".index_md122_a05_ep500_reassign.json"), code(".index_lemb_mql.json"), code(".index_vitemb_mql.json")
    C = {sp: c_lists(D, sp) for sp in ("valid", "test")}
    res = {"dataset": D, "models": {}, "catalog_items": len(idx)}
    ev = f"{R}/baselines/{D}/mql/seed42/eval"
    T = {t: torch.load(f"{R}/expl/{D}/topk_{t}.pt", weights_only=False) for t in tags}
    tgt = {sp: T[tags[0]][sp]["targets"].numpy() for sp in ("valid", "test")}
    for sp in ("valid", "test"):                                  # our rebuild of C's lists equals its reported results
        for name, f in (("text", "results_seqrec_20.json"), ("image", "results_seqimage_20.json"), ("merged", "results_ensemble_20.json")):
            m = json.load(open(f"{ev}/{sp}/{f}"))
            m = m.get("mean_results", m)
            r = rank_of(C[sp][name], tgt[sp])
            assert abs(hr(r, 10) - m["hit@10"]) < 1e-9 and abs(ndcg10(r) - m["ndcg@10"]) < 1e-9, (sp, name, hr(r, 10), m["hit@10"])
    tt = tgt["test"]
    rC = {n: rank_of(C["test"][n], tt) for n in C["test"]}
    res["C"] = {n: {"hr": {str(k): hr(rC[n], k) for k in KS}, "ndcg10": ndcg10(rC[n])} for n in rC}
    res["C"]["text"]["conc"] = conc(C["test"]["text"], ctext)
    res["C"]["image"]["conc"] = conc(C["test"]["image"], cimage)
    res["C"]["merged"]["conc_text_code"] = conc(C["test"]["merged"], ctext)
    res["C"]["merged"]["conc_image_code"] = conc(C["test"]["merged"], cimage)
    res["C"]["merged"]["conc_f"] = conc(C["test"]["merged"], fcode)
    print(f"== {D}")
    for n in ("text", "image", "merged"):
        print(f"C {n:7s}", {k: round(v, 4) for k, v in res["C"][n]["hr"].items()}, "ndcg", round(res["C"][n]["ndcg10"], 4))
    print("C conc text", res["C"]["text"]["conc"]); print("C conc image", res["C"]["image"]["conc"])
    print("C merged by text code", res["C"]["merged"]["conc_text_code"]); print("C merged by image code", res["C"]["merged"]["conc_image_code"])
    print("C merged by our f", res["C"]["merged"]["conc_f"])
    for t in tags:
        X = T[t]
        ids = X["test"]["ids"].numpy()
        assert np.array_equal(X["test"]["targets"].numpy(), tt)
        top = ids[:, :10]
        r = X["test"]["rank"].numpy()
        assert (np.minimum(r, 10) == np.minimum(rank_of(top, tt), 10)).all()
        e = {"setting": X["setting"], "hr": {str(k): hr(r, k) for k in KS}, "ndcg10": ndcg10(r), "conc_f": conc(top, fcode),
             "conc_C_text_code": conc(top, ctext), "conc_C_image_code": conc(top, cimage)}
        mg = C["test"]["merged"]
        e["overlap_with_C_merged"] = float(np.mean([len(set(a) & set(b[b >= 0])) for a, b in zip(top, mg)]))
        miss = (rC["merged"] < 10) & (r >= 10)
        tf = fcode[tt]
        in_top = np.array([tf[u] in set(fcode[top[u]]) for u in range(len(tt))])
        def bucket(a, b):
            return float(((r[miss] >= a) & (r[miss] < b)).mean())
        e["miss"] = {"users": int(miss.sum()), "target_f_in_our_top10_f": float(in_top[miss].mean()), "rank_11_20": bucket(10, 20),
                     "rank_21_50": bucket(20, 50), "rank_51_100": bucket(50, 100), "rank_beyond_100_scored": bucket(100, BIG),
                     "target_f_not_kept": float((r[miss] >= BIG).mean())}
        e["paired_vs_C_merged"] = paired(np.minimum(r, 10), np.minimum(rC["merged"], 10))
        res["models"][t] = e
        print(f"-- {t} {X['setting']}", {k: round(v, 4) for k, v in e["hr"].items()}, "ndcg", round(e["ndcg10"], 4))
        print("   conc f", e["conc_f"]); print("   by C text code", e["conc_C_text_code"]); print("   by C image code", e["conc_C_image_code"])
        print("   overlap", round(e["overlap_with_C_merged"], 3), "miss", e["miss"], "vs C", fm(e["paired_vs_C_merged"]["hr10"]))
        if not fix:
            continue
        lists = {sp: (X[sp]["ids"].numpy(), X[sp]["scores"].numpy()) for sp in ("valid", "test")}
        depth = lists["valid"][0].shape[1]
        grid = [("cap", m) for m in CAPS] + [("penalty", x) for x in LAMBDAS]

        def apply(sp, g):
            ids_, sc_ = lists[sp]
            pad = ids_ >= len(idx)                                # rank_lists pads short users with N
            f_ = fcode[np.minimum(ids_, len(idx) - 1)].copy()
            f_[pad] = -1
            if g[0] == "cap":
                return cap_list(ids_, f_ + 1, g[1])
            sc2 = sc_.copy()
            sc2[pad] = -1e30
            return penalty_list(ids_, sc2, f_, g[1])
        table = []
        for g in grid:
            rv = rank_of(apply("valid", g), tgt["valid"])
            table.append({"fix": g[0], "value": g[1], "valid_hr10": hr(rv, 10), "valid_ndcg10": ndcg10(rv)})
        best = max(table, key=lambda x: (round(x["valid_hr10"], 10), round(x["valid_ndcg10"], 10)))
        g = (best["fix"], best["value"])
        rt = rank_of(apply("test", g), tt)
        base = np.minimum(r, 10)
        rv0 = np.minimum(X["valid"]["rank"].numpy(), 10)
        e["fix"] = {"depth": depth, "grid": table, "chosen": best, "valid_base": {"hr10": hr(rv0, 10), "ndcg10": ndcg10(rv0)},
                    "test": {"hr": {str(k): hr(rt, k) for k in KS}, "ndcg10": ndcg10(rt)}, "conc_f": conc(apply("test", g), fcode),
                    "paired_vs_same_model": paired(np.minimum(rt, 10), base),
                    "paired_vs_C_merged": paired(np.minimum(rt, 10), np.minimum(rC["merged"], 10))}
        print("   FIX grid", [(x["fix"], x["value"], round(x["valid_hr10"], 4)) for x in table])
        print(f"   FIX chosen {g}: valid {best['valid_hr10']:.4f}/{best['valid_ndcg10']:.4f} (base {hr(rv0, 10):.4f}/{ndcg10(rv0):.4f}) test", e["fix"]["test"]["hr"]["10"], e["fix"]["test"]["ndcg10"],
              "vs same", fm(e["fix"]["paired_vs_same_model"]["hr10"]), "NDCG", fm(e["fix"]["paired_vs_same_model"]["ndcg10"]), "vs C", fm(e["fix"]["paired_vs_C_merged"]["hr10"]))
    json.dump(res, open(f"{R}/expl/{D}/top10.json", "w"), indent=1)


if __name__ == "__main__":
    main()
