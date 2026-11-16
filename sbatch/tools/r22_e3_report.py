"""§22.1 E3 report (CPU, inside the env): every mfd/<D>/r21_A01/seed<S>/r22_e3 (macrec/r22_e3_smooth.py).
Per run: chosen (tau multiple, lambda) on valid, valid / test HR@10 / NDCG@10 of chosen and plain, paired test CI chosen -
plain (2,000 resamples, seed 0), ms/user. Diagnostic on C-hit / A01-miss TEST users (C merged of the same seed): the codes
ranked above the target's f by A01's P(f) (the first f_rank of the top-64 list, capped at 64): share of them that are among
the target f's 5 nearest codebook neighbours, vs the chance share 5 / (n_f - 1); and the share of users whose top-ranked code
(when it is not the target f) is one of those 5 neighbours, vs the same chance rate.
Usage: python r22_e3_report.py -> mmgen_runs/geo/r22_e3.json
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


def main():
    out = {}
    for d in sorted(glob.glob(f"{R}/mfd/*/r21_A01/seed*/r22_e3/e3.json")):
        D, s = d.split("/")[-5], int(d.split("/")[-3][4:])
        E = json.load(open(d))
        X = torch.load(d.replace("e3.json", "e3_ranks.pt"), weights_only=False)
        inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
        tg = np.array([inter[u][-1] for u in inter])
        boot = np.random.default_rng(0).integers(0, len(tg), size=(2000, len(tg)))
        rc, rp = X["test_chosen"].numpy(), X["test_plain"].numpy()
        ref = torch.load(f"{R}/mfd/{D}/r21_A01/seed{s}/r21_eval/ranks.pt", weights_only=False)["test"]["plain"].numpy()
        C = rank_of(c_lists_seed(D, s), tg)
        sel = (C < 10) & (rp >= 10)
        nn5 = X["nn5"].numpy(); tf = X["tf"].numpy(); fr = X["f_rank"].numpy(); top = X["top64"].numpy()
        nF = nn5.shape[0]
        share, top1 = [], []
        for u in np.where(sel)[0]:
            above = top[u, :min(fr[u], 64)]
            if len(above):
                share.append(np.isin(above, nn5[tf[u]]).mean())
                top1.append(above[0] in nn5[tf[u]])
        ch = E["chosen"]
        out[f"{D} seed {s}"] = {"chosen": ch, "chosen_cfg": E["valid"][ch], "valid_plain": E["valid"][[k for k in E["valid"] if k.endswith("lam0.0")][0]],
                                "test": E["test"], "ci_chosen_minus_plain": paired(rc, rp, boot),
                                "plain_equals_r21_eval_test": float((np.minimum(rp, 10) == np.minimum(ref, 10)).mean()),
                                "diag_c_hit_a01_miss": {"users": int(sel.sum()), "users_with_codes_above": len(share),
                                                        "share_above_codes_in_nn5": float(np.mean(share)) if share else None,
                                                        "share_top_code_in_nn5": float(np.mean(top1)) if top1 else None,
                                                        "chance": 5 / (nF - 1)},
                                "gpu": E["gpu"]}
        o = out[f"{D} seed {s}"]
        print(D, s, "chosen", ch, "valid", round(o["chosen_cfg"]["hr10"], 4), "vs plain", round(o["valid_plain"]["hr10"], 4),
              "| test", round(E["test"]["chosen"]["hr10"], 4), "vs", round(E["test"]["plain"]["hr10"], 4),
              "CI", round(o["ci_chosen_minus_plain"]["hr10"]["diff"], 4), [round(x, 4) for x in o["ci_chosen_minus_plain"]["hr10"]["ci95"]],
              "| ms", round(E["test"]["chosen"]["ms_per_user"], 1), round(E["test"]["plain"]["ms_per_user"], 1), "| diag", json.dumps(o["diag_c_hit_a01_miss"]))
    json.dump(out, open(f"{R}/geo/r22_e3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
