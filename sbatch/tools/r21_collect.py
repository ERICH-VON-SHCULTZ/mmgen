"""§21 collection (docs/MFD_REPORT.md §21). CPU, inside the env, next to round20_collect.py / expl_top10.py / plugin_paired.py.
Per run with <run>/r21_eval/eval.json (macrec/r21_eval.py): valid / test metrics at plain K_f = 50 and with 1c (chosen on
valid), f diagnostics, paired CIs on test vs P1a seed 42 plain, P1a seed 42 + 1c (both from P1a's own r21_eval, the
reference row) and C merged seed 42 (users aligned by dataset index, 2,000 resamples, seed 0, 95% percentile);
training: best epoch, epochs and steps run, steps per epoch, wall-clock, peak memory, GPU.
Usage: python r21_collect.py <D> [run ...] (default: every mfd/<D>/r21_*/seed42 with an eval) -> mmgen_runs/geo/r21_<D>.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin_paired import ndcg, paired  # noqa: E402
from round20_collect import c_lists_seed, rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
REF = "mfd122_mv_geoP1a"


def met(r):
    return {"hr10": float((r < 10).mean()), "ndcg10": float(ndcg(r).mean())}


def main():
    D = sys.argv[1]
    runs = sys.argv[2:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{R}/mfd/{D}/r21_*/seed42") )
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    tg = np.array([inter[u][-1] for u in inter])
    boot = np.random.default_rng(0).integers(0, len(tg), size=(2000, len(tg)))
    C = rank_of(c_lists_seed(D, 42), tg)
    cr = json.load(open(f"{R}/baselines/{D}/mql/seed42/eval/test/results_ensemble_20.json"))
    assert abs(cr["hit@10"] - (C < 10).mean()) < 1e-6
    def load(run):
        b = f"{R}/mfd/{D}/{run}/seed42"
        E = json.load(open(f"{b}/r21_eval/eval.json"))
        X = torch.load(f"{b}/r21_eval/ranks.pt", weights_only=False)
        rk = {s: {k: v.numpy() for k, v in X[s].items()} for s in X}
        for s in ("valid", "test"):
            for k in ("plain", "1c"):
                assert abs(E[s][k]["hr10"] - (rk[s][k] < 10).mean()) < 1e-6, (run, s, k)
        return b, E, rk
    _, Eref, Rref = load(REF)
    out = {"dataset": D, "C_seed42_test": met(C), "runs": {}}
    for run in [REF] + [r for r in runs if r != REF]:
        if not os.path.exists(f"{R}/mfd/{D}/{run}/seed42/r21_eval/eval.json"):
            out["runs"][run] = {"missing": True}
            continue
        b, E, rk = load(run)
        st = json.load(open(f"{b}/train_stats.json"))
        clip = (st.get("r21") or {}).get("clip")
        if clip is None:                                        # P1a: only every logging_steps-th norm is logged
            ts = json.load(open(f"{b}/trainer_state.json"))
            gn = [h["grad_norm"] for h in ts["log_history"] if "grad_norm" in h]
            clip = {"steps": len(gn), "logged_every": ts.get("logging_steps"), "max_grad_norm": 1.0,
                    "clipped_share": float(np.mean(np.array(gn) > 1.0)), "norm_min": min(gn), "norm_median": float(np.median(gn))}
        row = {"clip": clip, "valid": E["valid"], "test": E["test"],
               "train": {k: st.get(k) for k in ("best_epoch", "epochs_run", "global_steps", "train_wall_seconds", "peak_gpu_mem_gb", "gpu",
                                                 "train_examples_per_epoch")} | {"steps_per_epoch": st["global_steps"] / st["epochs_run"],
                                                                                  "r21": st.get("r21")},
               "ci_test": {}}
        for k in ("plain", "1c"):
            r = rk["test"][k]
            row["ci_test"][k] = {"vs_P1a_plain": paired(r, Rref["test"]["plain"], boot), "vs_P1a_1c": paired(r, Rref["test"]["1c"], boot),
                                 "vs_C": paired(r, C, boot)}
        out["runs"][run] = row
    json.dump(out, open(f"{R}/geo/r21_{D}.json", "w"), indent=1, default=str)
    f4 = lambda x: f"{x:.4f}"
    ci = lambda p: f"{p['hr10']['diff']:+.4f} [{p['hr10']['ci95'][0]:+.4f}, {p['hr10']['ci95'][1]:+.4f}]"
    for run, row in out["runs"].items():
        if row.get("missing"):
            print(run, "missing"); continue
        v, t = row["valid"], row["test"]
        print(run, "valid plain", f4(v["plain"]["hr10"]), "1c", f4(v["1c"]["hr10"]), "| test plain", f4(t["plain"]["hr10"]), f4(t["plain"]["ndcg10"]),
              "1c", f4(t["1c"]["hr10"]), f4(t["1c"]["ndcg10"]), "| plain vs P1a", ci(row["ci_test"]["plain"]["vs_P1a_plain"]),
              "vs P1a+1c", ci(row["ci_test"]["plain"]["vs_P1a_1c"]), "vs C", ci(row["ci_test"]["plain"]["vs_C"]),
              "| f top1 v/t", f4(v["f"]["f_top1"]), f4(t["f"]["f_top1"]), "median", v["f"]["f_rank_median"], t["f"]["f_rank_median"],
              "| clipped", round(row["clip"]["clipped_share"], 3), "| best ep", row["train"]["best_epoch"], "steps/ep", round(row["train"]["steps_per_epoch"], 1))


if __name__ == "__main__":
    main()
