"""§22.2 E2 report (CPU, inside the env), Arts seed 42.
Step 1 (geo/r22/Arts/e2diag_r21_A01_s42.pt, macrec/r22_e2_diag.py): on C-hit / A01-miss TEST users (C merged seed 42 vs A01
plain K_f = 50), the share hit by C's text list only / image list only / both (top 10 of each single list); the target's t1 / i1
rank given its f under A01 (among all level codes and among codes occurring under f), for those users and for all users;
and how first-level text / image codes are spread over f in the catalog (one codebook per level, shared by all parents).
Step 2 (mfd/Arts/r22_E2g*/seed42/r22_e2eval): chosen w, valid / test, CIs vs A01 (test, chosen and w = 0), and the
Task-0 split vs A01 for the chosen config: rescued / lost users, rank change mostly across f buckets vs within.
Usage: python r22_e2_report.py -> mmgen_runs/geo/r22_e2.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expl_top10 import c_lists  # noqa: E402
from plugin_paired import paired  # noqa: E402
from round20_collect import rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
BIG = 10 ** 6


def stats(x):
    x = np.asarray(x)
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None, "mean": float(x.mean()) if len(x) else None,
            "top1": float((x == 0).mean()) if len(x) else None, "top5": float((x < 5).mean()) if len(x) else None,
            "top10": float((x < 10).mean()) if len(x) else None}


def split(ref, new, sel, rescued):
    fk = sel & ((ref["rank"] >= BIG) if rescued else (new["rank"] >= BIG))
    both = sel & ~fk & (ref["rank"] < BIG) & (new["rank"] < BIG)
    do = (new["n_other"] - ref["n_other"])[both]; dw = (new["n_within"] - ref["n_within"])[both]
    a, w = np.abs(do) > np.abs(dw), np.abs(dw) > np.abs(do)
    return {"users": int(sel.sum()), "f_crosses_Kf_cut": int(fk.sum()), "across_buckets": int(a.sum()), "within_bucket": int(w.sum()),
            "tie": int((~a & ~w).sum())}


def main():
    D = "Arts"
    out = {}
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    tg = np.array([inter[u][-1] for u in inter])
    boot = np.random.default_rng(0).integers(0, len(tg), size=(2000, len(tg)))
    A = torch.load(f"{R}/mfd/{D}/r21_A01/seed42/r21_eval/ranks.pt", weights_only=False)["test"]["plain"].numpy()
    p = f"{R}/geo/r22/{D}/e2diag_r21_A01_s42.pt"
    if os.path.exists(p):
        X = {k: v.numpy() for k, v in torch.load(p, weights_only=False).items()}
        Cl = c_lists("Arts", "test")
        rc = {n: rank_of(Cl[n], tg) for n in Cl}
        sel = (rc["merged"] < 10) & (A >= 10)
        t, i = rc["text"] < 10, rc["image"] < 10
        out["step1"] = {"c_hit_a01_miss": int(sel.sum()),
                        "share_text_only": float((sel & t & ~i).sum() / sel.sum()), "share_image_only": float((sel & i & ~t).sum() / sel.sum()),
                        "share_both": float((sel & t & i).sum() / sel.sum()), "share_neither_single_list": float((sel & ~t & ~i).sum() / sel.sum()),
                        "set": {k: stats(X[k][sel]) for k in ("rank_t1_all", "rank_t1_inf", "rank_i1_all", "rank_i1_inf", "n_t1_under_f", "n_i1_under_f")},
                        "all_users": {k: stats(X[k]) for k in ("rank_t1_all", "rank_t1_inf", "rank_i1_all", "rank_i1_inf", "n_t1_under_f", "n_i1_under_f")},
                        "codebooks": {"organisation": "one codebook per level (text_rq / image_rq vq_layers.l), shared by every parent",
                                      "t1_codes_used": int((X["t1_codes_n_f"] > 0).sum()), "i1_codes_used": int((X["i1_codes_n_f"] > 0).sum()),
                                      "t1_distinct_f_per_code": stats(X["t1_codes_n_f"][X["t1_codes_n_f"] > 0]),
                                      "i1_distinct_f_per_code": stats(X["i1_codes_n_f"][X["i1_codes_n_f"] > 0]),
                                      "t1_codes_under_more_than_one_f": float((X["t1_codes_n_f"] > 1).mean()),
                                      "i1_codes_under_more_than_one_f": float((X["i1_codes_n_f"] > 1).mean())}}
        print("step1", json.dumps(out["step1"])[:1500])
    G = torch.load(f"{R}/geo/r21_gain/{D}/r21_A01_s42.pt", weights_only=False)
    G = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in G.items()}
    out["step2"] = {}
    for d in sorted(glob.glob(f"{R}/mfd/{D}/r22_E2g*/seed42/r22_e2eval/e2_eval.json")):
        run = d.split("/")[-4]
        E = json.load(open(d))
        X = torch.load(d.replace("e2_eval.json", "e2_ranks.pt"), weights_only=False)
        w = E["chosen_w"]
        ch = {k: v.numpy() for k, v in X[f"w{w}"].items()}
        w0 = X["w0.0"]["rank"].numpy()
        row = {"chosen_w": w, "valid": E["valid"], "test": E["test"], "test_f": E["test_f"],
               "ci_chosen_vs_A01": paired(ch["rank"], A, boot), "ci_w0_vs_A01": paired(w0, A, boot), "ci_chosen_vs_w0": paired(ch["rank"], w0, boot),
               "split_vs_A01": {"rescued": split(G, ch, (G["rank"] >= 10) & (ch["rank"] < 10), True),
                                "lost": split(G, ch, (G["rank"] < 10) & (ch["rank"] >= 10), False)},
               "gpu": E["gpu"]}
        out["step2"][run] = row
        print(run, "w", w, "| test", json.dumps(E["test"]), "| vs A01", round(row["ci_chosen_vs_A01"]["hr10"]["diff"], 4),
              [round(x, 4) for x in row["ci_chosen_vs_A01"]["hr10"]["ci95"]], "| w0 vs A01", round(row["ci_w0_vs_A01"]["hr10"]["diff"], 4),
              [round(x, 4) for x in row["ci_w0_vs_A01"]["hr10"]["ci95"]], "| split", json.dumps(row["split_vs_A01"]))
    json.dump(out, open(f"{R}/geo/r22_e2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
