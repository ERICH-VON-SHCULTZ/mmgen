"""mmgen addition (docs/MFD_REPORT.md §24.6): is the learning-rate gain specific to full P1a (item-level input + codebook init)? (CPU)
Four models, each at its default lr and at lr 2e-3, every other setting the model's own, Arts + Instruments seed 42:
  P1a          mfd/<D>/mfd122_mv_geoP1a              vs mfd/<D>/r24_lr2e3
  P1a-noinit   mfd/<D>/r24_noinit                    vs mfd/<D>/r24_noinit_lr2e3     (--geo none --item_enc)
  V2-MFD       mfd/<D>/mfd122_mv_geoV2               vs mfd/<D>/r24_v2_lr2e3         (codebook init, token-level input, 200 epochs)
  MD chain     baselines/<D>/md122_a05_ep500_reassign vs baselines_tuned/<D>/md122_a05_ep500_reassign_lr2e3 (token-level, no init)
MFD models: plain K_f = 50 ranks (r21_eval); the chain: its own beam-search list (ranks rebuilt from the saved outputs and checked against
the results file, tools/r24_baselines.ranks). Per model: pooled + per-dataset valid / test HR@10 and NDCG@10 at both lrs; the gain
lr 2e-3 - default (paired, per dataset and pooled stratified bootstrap); the difference in gains vs P1a, per user
[(P1a_2e3 - P1a_def) - (X_2e3 - X_def)] (same users, pooled stratified bootstrap); training-curve stability of both runs.
A missing run is listed and its model skipped. -> geo/r24_lrspec.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r22_e10_report as E  # noqa: E402
import r24_baselines as RB  # noqa: E402
from r24_report import stability  # noqa: E402

R, DS, SPS = E.R, ("Arts", "Instruments"), ("valid", "test")
CH = "md122_a05_ep500_reassign"
MODELS = {"P1a": ("mfd", "mfd122_mv_geoP1a", "r24_lr2e3"), "P1a-noinit": ("mfd", "r24_noinit", "r24_noinit_lr2e3"),
          "V2-MFD": ("mfd", "mfd122_mv_geoV2", "r24_v2_lr2e3"), "MD chain": ("chain", None, None)}
MET = (("hr10", E.hr), ("ndcg10", E.ndcg))


def rdir(kind, name, D, lr):
    if kind == "chain":
        return f"{R}/baselines_tuned/{D}/{CH}_lr2e3/seed42" if lr else f"{R}/baselines/{D}/{CH}/seed42"
    return f"{R}/mfd/{D}/{name}/seed42"


def load(model, lr):
    kind, a, b = MODELS[model]
    out = {}
    for D in DS:
        rd = rdir(kind, b if lr else a, D, lr)
        if kind == "chain":
            if not os.path.exists(f"{rd}/eval/test/results_seqrec_20.json"):
                return None, rd
            out[D] = {"rd": rd, **{sp: RB.ranks(D, CH, rd, sp) for sp in SPS}}
        else:
            if not os.path.exists(f"{rd}/r21_eval/ranks.pt"):
                return None, rd
            import torch
            r = torch.load(f"{rd}/r21_eval/ranks.pt", weights_only=False)
            out[D] = {"rd": rd, **{sp: r[sp]["plain"].numpy() for sp in SPS}}
    return out, None


def pooled(d, sp, f):
    return float(sum(f(d[D][sp]).sum() for D in DS) / sum(len(d[D][sp]) for D in DS))


def main():
    res, L, missing = {"models": {}, "missing": []}, {}, []
    for m in MODELS:
        x = {}
        for lr in (False, True):
            x[lr], miss = load(m, lr)
            if miss:
                missing.append(f"{m} {'lr 2e-3' if lr else 'default lr'}: {miss}")
        if x[False] is None or x[True] is None:
            continue
        L[m] = x
        rng = np.random.default_rng(0)
        r = {"runs": {("lr2e3" if lr else "default"): {D: x[lr][D]["rd"] for D in DS} for lr in (False, True)}}
        for sp in SPS:
            r[sp] = {}
            for lr in (False, True):
                k = "lr2e3" if lr else "default"
                r[sp][k] = {"pooled": {mm: pooled(x[lr], sp, f) for mm, f in MET},
                            **{D: {mm: float(f(x[lr][D][sp]).mean()) for mm, f in MET} for D in DS}}
            r[sp]["gain"] = {mm: {"pooled": E.strat_boot({D: f(x[True][D][sp]) - f(x[False][D][sp]) for D in DS}, rng),
                                  **{D: E.strat_boot({D: f(x[True][D][sp]) - f(x[False][D][sp])}, rng) for D in DS}} for mm, f in MET}
        st = {}
        for lr in (False, True):
            for D in DS:
                rd = x[lr][D]["rd"]
                s = json.load(open(f"{rd}/train_stats.json"))
                st[f"{'lr2e3' if lr else 'default'} {D}"] = stability(rd, s) | {"lr": (s.get("r24") or {}).get("learning_rate")}
        r["stability"] = st
        res["models"][m] = r
    if "P1a" in L:                                           # difference in gains vs P1a, per user (same test / valid users)
        P, rng = L["P1a"], np.random.default_rng(1)
        for m in L:
            if m == "P1a":
                continue
            X = L[m]
            res["models"][m]["P1a_gain_minus_this_gain"] = {
                sp: {mm: E.strat_boot({D: (f(P[True][D][sp]) - f(P[False][D][sp])) - (f(X[True][D][sp]) - f(X[False][D][sp])) for D in DS}, rng)
                     for mm, f in MET} for sp in SPS}
    res["missing"] = missing
    json.dump(res, open(f"{R}/geo/r24_lrspec.json", "w"), indent=1)
    fm = lambda c: f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
    print("missing (model skipped):", missing or "none")
    print("== pooled Arts + Instruments seed 42, HR@10 / NDCG@10: default lr -> lr 2e-3, gain (95% CI) ==")
    for m, r in res["models"].items():
        for sp in SPS:
            a, b, g = r[sp]["default"]["pooled"], r[sp]["lr2e3"]["pooled"], r[sp]["gain"]
            print(f"{m:11s} {sp:5s} {a['hr10']:.4f} / {a['ndcg10']:.4f} -> {b['hr10']:.4f} / {b['ndcg10']:.4f} | gain HR {fm(g['hr10']['pooled'])} "
                  f"NDCG {fm(g['ndcg10']['pooled'])} | " + " ".join(f"{D} HR {fm(g['hr10'][D])}" for D in DS))
        if "P1a_gain_minus_this_gain" in r:
            d = r["P1a_gain_minus_this_gain"]
            print(f"{'':11s} P1a's gain - this gain: " + " | ".join(f"{sp} HR {fm(d[sp]['hr10'])} NDCG {fm(d[sp]['ndcg10'])}" for sp in SPS))
        for k, s in r["stability"].items():
            print(f"{'':11s} stability {k}: " + json.dumps({a: (round(v, 4) if isinstance(v, float) else v) for a, v in s.items()}))


if __name__ == "__main__":
    main()
