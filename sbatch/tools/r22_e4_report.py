"""mmgen addition (docs/MFD_REPORT.md §22.4, E4): is the shared code f harder to predict than a single-modality level-1 code?
CPU; inputs from macrec/r22_e4_first.py (geo/r22/<D>/e4_{A01,C,A,B}.pt, seed 42) + saved test lists / ranks.

Partitions (level 1 of each model's own SID): A01 f (MD 1+2+2 v1, first token), C text / C image level 1 (MQL4GRec),
TIGER A text level 1, TIGER B image level 1.
1 predictability, over all test users and the C-hit / A01-miss, A01-hit / C-miss sets (C = merged list, A01 = plain K_f = 50
  ranks of r21_eval, hit = rank < 10): top-1 / top-10 of the true level-1 code, MRR, median rank (0 = top), mean NLL (nats,
  distribution renormalised over the model's level-1 codes), marginal cross-entropy CE_m = mean -log q(code) with q = the
  empirical distribution of the target's level-1 code over ALL test users (= H(marginal) on all users), information gain
  IG = CE_m - NLL and IG / CE_m.
2 HR@10 of each model's own final list by the true level-1 code's rank bucket (1, 2-5, 6-20, >20): share of users, HR@10 in
  the bucket, contribution = share x HR@10. "C merged" uses min(text, image level-1 rank).
3 partition relations, over catalog items and over test targets (one row per user): NMI (arithmetic), ARI between f and each
  other partition; bucket spread both ways (for each item, the number of distinct other-partition codes among the items of its
  bucket: median over items = bucket-size-weighted median). On C-hit / A01-miss users: the target's C-text / C-image bucket
  split over f codes (number of f codes it spans; share of its items in the target's f), and whether the f codes A01 ranked
  above the target f contain items of that bucket (share of such codes vs the share of ALL f codes that do = chance).
Out: geo/r22_e4.json + printed tables.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expl_top10 import c_lists  # noqa: E402
from round20_collect import rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
BIG = 10 ** 9
MD = ".index_md122_a05_ep500_reassign.json"
PARTS = {  # name: (e4 file, key, index file, baseline run for the final list or None)
    "A01 f": ("A01", "f", MD, None),
    "C text": ("C", "seqrec", ".index_lemb_mql.json", None),
    "C image": ("C", "seqimage", ".index_vitemb_mql.json", None),
    "A text": ("A", "seqrec", ".index_lemb_tiger.json", "tiger_text"),
    "B image": ("B", "seqrec", ".index_vitemb_tiger.json", "tiger_image"),
}
BUCKETS = (("1", 0, 0), ("2-5", 1, 4), ("6-20", 5, 19), (">20", 20, BIG))


def single_list(D, run, fn):
    """top-10 item ids of a one-list baseline (test), users in dataset order (expl_top10.c_lists' parsing)."""
    sv = json.load(open(f"{R}/baselines/{D}/{run}/seed42/eval/test/save_seqrec_20.json"))
    idx = json.load(open(f"{R}/data/{D}/{D}{fn}"))
    s2i = {"".join(v): int(k) for k, v in idx.items()}
    U = len(sv["all_users"])
    B = len(sv["all_outputs"]) // U
    out = np.full((U, 10), -1, dtype=np.int64)
    for k, u in enumerate(sv["all_users"]):
        o = [x.strip().replace(" ", "") for x in sv["all_outputs"][k * B:(k + 1) * B]]
        s = sv["all_scores"][k * B:(k + 1) * B]
        top = [a for a, _ in sorted([(s2i[a], b) for a, b in zip(o, s) if a in s2i], key=lambda p: p[1], reverse=True)][:10]
        out[u, :len(top)] = top
    return out


def level1(D, fn):
    return {int(k): v[0] for k, v in json.load(open(f"{R}/data/{D}/{D}{fn}")).items()}


def predict_stats(lp, y, q, sel):
    lp, y = lp[sel], y[sel]
    ty = lp[np.arange(len(y)), y]
    rank = (lp > ty[:, None]).sum(1)
    nll = float(-ty.mean())
    ce = float(-np.log(q[y]).mean())
    return {"users": int(sel.sum()), "top1": float((rank == 0).mean()), "top10": float((rank < 10).mean()),
            "mrr": float((1 / (rank + 1)).mean()), "median_rank": float(np.median(rank)), "nll": nll,
            "ce_marginal": ce, "ig": ce - nll, "ig_norm": (ce - nll) / ce}


def spread(a, b):
    """per item: number of distinct b-codes in its a-bucket -> median over items (bucket-size weighted)."""
    n = {}
    for x, z in zip(a, b):
        n.setdefault(x, set()).add(z)
    return float(np.median([len(n[x]) for x in a]))


def codes(arr):
    _, inv = np.unique(np.asarray(arr), return_inverse=True)
    return inv


def run(D):
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    tg = np.array([inter[u][-1] for u in inter])
    U = len(tg)
    A01 = torch.load(f"{R}/mfd/{D}/r21_A01/seed42/r21_eval/ranks.pt", weights_only=False)["test"]["plain"].numpy()
    Cl = c_lists(D, "test")
    rc = {n: rank_of(Cl[n], tg) for n in Cl}
    final = {"A01 f": A01, "C text": rc["text"], "C image": rc["image"]}
    for name, (_, _, fn, run_) in PARTS.items():
        if run_:
            final[name] = rank_of(single_list(D, run_, fn), tg)
    sets = {"all": np.ones(U, bool), "C_hit_A01_miss": (rc["merged"] < 10) & (A01 >= 10),
            "A01_hit_C_miss": (A01 < 10) & (rc["merged"] >= 10)}
    E = {m: torch.load(f"{R}/geo/r22/{D}/e4_{m}.pt", weights_only=False) for m in ("A01", "C", "A", "B")}
    L1, Y, LP, RK, info = {}, {}, {}, {}, {}
    for name, (m, key, fn, _) in PARTS.items():
        e = E[m][key]
        lp = e["lp"].float().numpy()
        assert lp.shape[0] == U, (name, lp.shape, U)
        col = {t: j for j, t in enumerate(e["tokens"])}
        L1[name] = level1(D, fn)
        y = np.array([col[L1[name][int(t)]] for t in tg])
        q = np.bincount(y, minlength=lp.shape[1]) / U
        Y[name], LP[name] = y, lp
        RK[name] = (lp > lp[np.arange(U), y][:, None]).sum(1)
        pz = q[q > 0]
        info[name] = {"codebook_size": lp.shape[1], "codes_used_by_test_targets": int((q > 0).sum()),
                      "H_target_level1_nats": float(-(pz * np.log(pz)).sum()),
                      "predict": {s: predict_stats(lp, y, np.where(q > 0, q, 1.0), sel) for s, sel in sets.items()}}
    # 2. HR@10 by level-1 rank bucket
    hrb = {}
    for name in list(PARTS) + ["C merged"]:
        rk = np.minimum(RK["C text"], RK["C image"]) if name == "C merged" else RK[name]
        hit = (rc["merged"] if name == "C merged" else final[name]) < 10
        row = {"hr10": float(hit.mean())}
        for b, lo, hi in BUCKETS:
            s = (rk >= lo) & (rk <= hi)
            row[b] = {"share": float(s.mean()), "hr10": float(hit[s].mean()) if s.any() else None,
                      "contribution": float((hit & s).mean())}
        hrb[name] = row
    # 3. partition relations
    items = sorted(set.intersection(*[set(L1[n]) for n in PARTS]))
    rel = {}
    fi = codes([L1["A01 f"][i] for i in items])
    ft = codes([L1["A01 f"][int(t)] for t in tg])
    for name in PARTS:
        if name == "A01 f":
            continue
        oi = codes([L1[name][i] for i in items])
        ot = codes([L1[name][int(t)] for t in tg])
        rel[name] = {"items": {"nmi": float(normalized_mutual_info_score(fi, oi)), "ari": float(adjusted_rand_score(fi, oi)),
                               "other_codes_per_f_bucket_median": spread(fi, oi), "f_codes_per_other_bucket_median": spread(oi, fi),
                               "n_f": int(fi.max() + 1), "n_other": int(oi.max() + 1)},
                     "test_targets": {"nmi": float(normalized_mutual_info_score(ft, ot)), "ari": float(adjusted_rand_score(ft, ot))}}
    # C-hit / A01-miss: is the target's C bucket split over f, and do the f codes ranked above the target overlap it?
    sel = np.where(sets["C_hit_A01_miss"])[0]
    fcol = {t: j for j, t in enumerate(E["A01"]["f"]["tokens"])}
    item_f = {i: fcol[L1["A01 f"][i]] for i in items}
    nF = len(fcol)
    lpf = LP["A01 f"]
    for name in ("C text", "C image", "A text", "B image"):
        bucket = {}
        for i in items:
            bucket.setdefault(L1[name][i], []).append(i)
        span, own, ov, ch, top1 = [], [], [], [], []
        for u in sel:
            t = int(tg[u])
            its = bucket[L1[name][t]]
            fs = np.zeros(nF, bool)
            fs[[item_f[i] for i in its]] = True
            span.append(int(fs.sum()))
            own.append(float(np.mean([item_f[i] == item_f[t] for i in its])))
            above = np.where(lpf[u] > lpf[u, item_f[t]])[0]
            ch.append(float((fs.sum() - 1) / (nF - 1)))                   # chance: a random other f code contains bucket items
            if len(above):
                ov.append(float(fs[above].mean()))
                top1.append(bool(fs[int(lpf[u].argmax())]))
        rel[name]["C_hit_A01_miss"] = {"users": int(len(sel)), "f_codes_spanned_by_target_bucket_median": float(np.median(span)),
                                       "share_of_bucket_items_in_target_f_median": float(np.median(own)),
                                       "users_with_f_above_target": len(ov),
                                       "share_above_f_containing_bucket_items": float(np.mean(ov)),
                                       "top1_f_contains_bucket_items": float(np.mean(top1)),
                                       "chance_share_f_containing_bucket_items": float(np.mean(ch))}
    return {"users": U, "partitions": info, "hr10_by_level1_rank": hrb, "relations": rel}


def show(D, r):
    print(f"\n=== {D} ({r['users']} test users)")
    print("1. predictability          size used  H     | set              top1  top10  MRR   med  NLL   CE_m  IG    IG/CE")
    for n, x in r["partitions"].items():
        for s, p in x["predict"].items():
            print(f"{n:9s} {x['codebook_size']:4d} {x['codes_used_by_test_targets']:4d} {x['H_target_level1_nats']:.2f} | "
                  f"{s:16s} {p['top1']:.3f} {p['top10']:.3f} {p['mrr']:.3f} {p['median_rank']:4.0f} {p['nll']:.2f} "
                  f"{p['ce_marginal']:.2f} {p['ig']:.2f} {p['ig_norm']:.3f}  (n={p['users']})")
    print("2. HR@10 by level-1 rank: share / HR@10 / contribution")
    for n, x in r["hr10_by_level1_rank"].items():
        print(f"{n:9s} HR@10 {x['hr10']:.4f} | " + " | ".join(
            f"{b}: {x[b]['share']:.3f} / {x[b]['hr10'] if x[b]['hr10'] is None else round(x[b]['hr10'], 3)} / {x[b]['contribution']:.4f}"
            for b, _, _ in BUCKETS))
    print("3. relations to f")
    for n, x in r["relations"].items():
        print(f"{n:9s}", json.dumps(x))


def main():
    out = {}
    for D in ("Arts", "Instruments"):
        if os.path.exists(f"{R}/geo/r22/{D}/e4_B.pt"):
            out[D] = run(D)
            show(D, out[D])
    json.dump(out, open(f"{R}/geo/r22_e4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
