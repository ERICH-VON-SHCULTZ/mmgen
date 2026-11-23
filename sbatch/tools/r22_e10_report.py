"""mmgen addition (docs/MFD_REPORT.md §22.10, E10b / E10c): report vs P1a and A01, pooled-validation selection, chaining, 3x3 table. CPU.
Runs: mfd/<D>/r22_E10c/seed<S> (A01's KL without grouping, P1a's path), mfd/<D>/r22_E10b{005,02,05}/seed<S> (A01, beta 0.05 / 0.2 / 0.5).
Per run: valid / test HR@1/5/10, NDCG@5/10 (plain K_f = 50, r21_eval); paired bootstrap CIs (2,000 resamples, seed 0, users aligned
by index) vs P1a and vs A01 of the same seed on valid and test; ms/user; f diagnostics (top-1, median true-f rank from r21_eval; IG from
the fdump); E6 split (repeat-f / new-f HR@10, C - model, new-f true-f median rank; tools/r22_runs_report.f_analysis) for the run, P1a
and A01; training: clipping (share clipped, pre-clip norm quantiles), wall seconds per epoch (pack-concurrent: indicative only), E10c
KL stats. P1a's fdump is written from the E10a full-view dump (same f_logprobs path; E10a cross-checks it against A01's fdump).
E10b: beta per dataset chosen on seed-42 VALID HR@10 (tie NDCG@10) among 0.05 / 0.1 (= A01) / 0.2 / 0.5; Arts keeps 0.1 (A01).
Selection (from round 22 on): POOLED validation HR@10 over Arts + Instruments + Games (all users of the three datasets; stratified
paired bootstrap). A candidate is chained when its pooled valid HR@10 beats BOTH P1a and A01 (point estimate).
Modes:  python r22_e10_report.py                       report of every E10b / E10c run -> geo/r22_e10.json
        python r22_e10_report.py --stage e10c|e10b     report + chaining decision -> geo/r22_e10_<name>_stage.txt ("D:VAR" per line,
                                                       the runs to train on seeds 43 / 44; empty = nothing chained)
        python r22_e10_report.py --final e10c|e10b     3 datasets x 3 seeds table vs P1a, A01, C with the pooled paired bootstrap
                                                       -> geo/r22_e10_<name>_final.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin_paired import ndcg, paired  # noqa: E402
from r22_runs_report import f_analysis  # noqa: E402
from round20_collect import c_lists_seed, rank_of  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"
DS, SEEDS = ("Arts", "Instruments", "Games"), (42, 43, 44)
KEYS = ("hr1", "hr5", "hr10", "ndcg5", "ndcg10")
BETAS = {"E10b005": 0.05, "A01": 0.1, "E10b02": 0.2, "E10b05": 0.5}
hr = lambda r: (r < 10).astype(float)


def run_dir(D, name, S):
    d = {"P1a": "mfd122_mv_geoP1a", "A01": "r21_A01"}.get(name, name if name.startswith("r24_") else f"r22_{name}")
    return f"{R}/mfd/{D}/{d}/seed{S}"


def ev(D, name, S):
    rd = run_dir(D, name, S)
    if not os.path.exists(f"{rd}/r21_eval/ranks.pt"):
        return None
    e = json.load(open(f"{rd}/r21_eval/eval.json"))
    r = torch.load(f"{rd}/r21_eval/ranks.pt", weights_only=False)
    return e, {s: r[s]["plain"].numpy() for s in ("valid", "test")}


def fdump_path(D, name, S):
    d = os.path.basename(os.path.dirname(run_dir(D, name, S)))
    p = f"{R}/geo/r22/{D}/fdump_{d}_s{S}.pt"
    v = f"{R}/geo/r22/{D}/views_{d}_s{S}.pt"
    if not os.path.exists(p) and os.path.exists(v):                       # P1a: from the E10a full-view dump
        V = torch.load(v, weights_only=False)
        torch.save({"f": {"lp": V["full"], "tokens": V["tokens"]}}, p)
    return p


def train_info(D, name, S):
    f = f"{run_dir(D, name, S)}/train_stats.json"
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    r21 = s.get("r21", {})
    return {"epochs_run": s.get("epochs_run"), "best_epoch": s.get("best_epoch"), "gpu": s.get("gpu"),
            "wall_s_per_epoch": s["train_wall_seconds"] / s["epochs_run"] if s.get("epochs_run") else None,
            "clip": r21.get("clip"), "e10c": r21.get("e10c"), "steps_per_epoch": r21.get("steps_per_epoch")}


def strat_boot(byd, rng):
    """pooled mean and 95% CI of per-user differences {D: vector}, resampled within each dataset."""
    tot = sum(len(v) for v in byd.values())
    est = sum(v.sum() for v in byd.values()) / tot
    bs = np.zeros(2000)
    for v in byd.values():
        bs += v[rng.integers(0, len(v), size=(2000, len(v)))].sum(1)
    bs /= tot
    return {"diff": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "users": int(tot)}


def fmt(c):
    return f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"


_C = {}


def C_ranks(D, S, sp):
    if (D, S, sp) not in _C:
        inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
        tg = np.array([s[-1] if sp == "test" else s[-2] for s in inter.values()])
        _C[(D, S, sp)] = rank_of(c_lists_seed(D, S, sp), tg)
    return _C[(D, S, sp)]


def report_runs():
    out = {}
    for p in sorted(glob.glob(f"{R}/mfd/*/r22_E10*/seed*/r21_eval/eval.json")):
        rd = p.rsplit("/r21_eval/", 1)[0]
        D, V, S = rd.split("/")[-3], rd.split("/")[-2][4:], int(rd.split("/")[-1][4:])
        x, ref = ev(D, V, S), {k: ev(D, k, S) for k in ("P1a", "A01")}
        if not all(ref.values()):
            continue
        e, rk = x
        n = len(rk["test"])
        boot = np.random.default_rng(0).integers(0, n, size=(2000, n))
        r = {"D": D, "var": V, "seed": S, "gpu": e.get("gpu"),
             "valid": {k: e["valid"]["plain"][k] for k in KEYS}, "test": {k: e["test"]["plain"][k] for k in KEYS},
             "ms_per_user": e["test"]["plain"].get("ms_per_user"), "f_test": e["test"].get("f"), "train": train_info(D, V, S)}
        for b in ("P1a", "A01"):
            eb, rb = ref[b]
            r[f"{b}_valid"] = {k: eb["valid"]["plain"][k] for k in KEYS}
            r[f"{b}_test"] = {k: eb["test"]["plain"][k] for k in KEYS}
            r[f"{b}_ms_per_user"] = eb["test"]["plain"].get("ms_per_user")
            r[f"{b}_f_test"] = eb["test"].get("f")
            r[f"{b}_train"] = train_info(D, b, S)
            for sp in ("valid", "test"):
                r[f"ci_{sp}_vs_{b}"] = paired(rk[sp], rb[sp], boot)
        C = C_ranks(D, S, "test")
        for name, ranks in ((V, rk["test"]), ("P1a", ref["P1a"][1]["test"]), ("A01", ref["A01"][1]["test"])):
            fa = f_analysis(D, fdump_path(D, name, S), "1+2+2", ranks, C)
            r[f"f_{'model' if name == V else name}"] = fa[0] if fa else None
        out[f"{D} {V} seed {S}"] = r
    return out


def choose_beta(D):
    """E10b: beta of D on seed-42 valid HR@10 (tie NDCG@10) among the existing runs."""
    cand = {}
    for name, b in BETAS.items():
        x = ev(D, name, 42)
        if x:
            cand[name] = (x[0]["valid"]["plain"]["hr10"], x[0]["valid"]["plain"]["ndcg10"], b)
    if D == "Arts" or len(cand) < len(BETAS):
        return "A01", cand
    return max(cand, key=lambda k: cand[k][:2]), cand


def pooled_valid(model_by_D, S=42, sp="valid"):
    """pooled HR@10 / NDCG@10 of a model (name per dataset) vs P1a and A01 on one split, stratified paired bootstrap."""
    res, rng = {}, np.random.default_rng(0)
    rk = {D: ev(D, model_by_D[D], S)[1][sp] for D in DS}
    tot = sum(len(v) for v in rk.values())
    res["model"] = {"hr10": float(sum(hr(v).sum() for v in rk.values()) / tot), "ndcg10": float(sum(ndcg(v).sum() for v in rk.values()) / tot)}
    for b in ("P1a", "A01"):
        rb = {D: ev(D, b, S)[1][sp] for D in DS}
        res[b] = {"hr10": float(sum(hr(v).sum() for v in rb.values()) / tot), "ndcg10": float(sum(ndcg(v).sum() for v in rb.values()) / tot)}
        res[f"vs_{b}"] = {m: strat_boot({D: f(rk[D]) - f(rb[D]) for D in DS}, rng) for m, f in (("hr10", hr), ("ndcg10", ndcg))}
    res["per_dataset_model"] = model_by_D
    return res


def final(name):
    """3 x 3 table (seeds 42-44) of candidate `name` vs P1a, A01, C; per-seed CIs and the pooled stratified bootstrap."""
    if name == "e10c":
        sel = {D: "E10c" for D in DS}
    elif name == "e10b":
        sel = {D: choose_beta(D)[0] for D in DS}
    else:                                                    # any fixed run name on every dataset (e.g. E9k4, §22.9)
        sel = {D: name for D in DS}
    out = {"model_by_dataset": sel, "datasets": {}, "pooled": {}}
    diffs, rng = {}, np.random.default_rng(0)
    for D in DS:
        per = {}
        for S in SEEDS:
            x = {k: ev(D, k, S) for k in ("P1a", "A01")} | {"M": ev(D, sel[D], S)}
            if not all(x.values()):
                continue
            n = len(x["M"][1]["test"])
            boot = np.random.default_rng(0).integers(0, n, size=(2000, n))
            row = {}
            for sp in ("valid", "test"):
                rk = {k: v[1][sp] for k, v in x.items()} | {"C": C_ranks(D, S, sp)}
                for k, v in rk.items():
                    row[f"{k} {sp}"] = {"hr10": float(hr(v).mean()), "ndcg10": float(ndcg(v).mean())}
                for b in ("P1a", "A01", "C"):
                    row[f"M - {b} {sp}"] = paired(rk["M"], rk[b], boot)
                    for m, f in (("hr10", hr), ("ndcg10", ndcg)):
                        diffs.setdefault((b, sp, m), {}).setdefault(S, {})[D] = f(rk["M"]) - f(rk[b])
            per[S] = row
        mean = {k: {m: float(np.mean([per[S][k][m] for S in per])) for m in ("hr10", "ndcg10")} for k in next(iter(per.values()), {}) if " - " not in k}
        out["datasets"][D] = {"seeds": list(per), "mean": mean, "per_seed": per}
    for (b, sp, m), bys in diffs.items():
        full = {S: v for S, v in bys.items() if len(v) == len(DS)}
        res = {str(S): strat_boot(v, rng) for S, v in full.items()}
        if len(full) == len(SEEDS):
            res["pooled_over_seeds"] = strat_boot({D: np.mean([full[S][D] for S in full], 0) for D in DS}, rng)
        out["pooled"][f"M - {b} {sp} {m}"] = res
    json.dump(out, open(f"{R}/geo/r22_e10_{name}_final.json", "w"), indent=1)
    print(f"== {name} final: model per dataset {sel}")
    for D, v in out["datasets"].items():
        print(f"{D} seeds {v['seeds']}: " + " | ".join(f"{k} {x['hr10']:.4f}/{x['ndcg10']:.4f}" for k, x in v["mean"].items() if k.endswith("test")))
        for S, row in v["per_seed"].items():
            print(f"  seed {S}: " + "  ".join(f"{k}: HR {fmt(p['hr10'])}" for k, p in row.items() if " - " in k and k.endswith("test")))
    for k, v in out["pooled"].items():
        print(f"pooled {k}: " + "  ".join(f"{s} {fmt(c)}" for s, c in v.items()))


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--final":
        return final(argv[1])
    stage = argv[1] if argv and argv[0] == "--stage" else None
    out = {"runs": report_runs()}
    beta = {D: choose_beta(D) for D in ("Instruments", "Games")}
    out["e10b_beta_choice"] = {D: {"chosen": c, "valid_hr10_ndcg10_beta": cand} for D, (c, cand) in beta.items()}
    have = lambda name_by_D: all(ev(D, n, 42) for D, n in name_by_D.items())
    cands = {"E10c": {D: "E10c" for D in DS}, "E10b": {"Arts": "A01", **{D: beta[D][0] for D in beta}}}
    out["pooled"] = {}
    for c, m in cands.items():
        if have(m):
            out["pooled"][c] = {sp: pooled_valid(m, 42, sp) for sp in ("valid", "test")}
    json.dump(out, open(f"{R}/geo/r22_e10.json", "w"), indent=1)

    for k, r in out["runs"].items():
        print(f"\n=== {k}")
        print("valid " + " ".join(f"{m} {r['valid'][m]:.4f}" for m in KEYS) + " | P1a " + " ".join(f"{r['P1a_valid'][m]:.4f}" for m in KEYS)
              + " | A01 " + " ".join(f"{r['A01_valid'][m]:.4f}" for m in KEYS))
        print("test  " + " ".join(f"{m} {r['test'][m]:.4f}" for m in KEYS) + " | P1a " + " ".join(f"{r['P1a_test'][m]:.4f}" for m in KEYS)
              + " | A01 " + " ".join(f"{r['A01_test'][m]:.4f}" for m in KEYS))
        for b in ("P1a", "A01"):
            print(f"  vs {b}: valid HR@10 {fmt(r[f'ci_valid_vs_{b}']['hr10'])} | test HR@10 {fmt(r[f'ci_test_vs_{b}']['hr10'])} "
                  f"NDCG@10 {fmt(r[f'ci_test_vs_{b}']['ndcg10'])}")
        print(f"  ms/user {r['ms_per_user']} (P1a {r['P1a_ms_per_user']}, A01 {r['A01_ms_per_user']}) | f {json.dumps(r['f_test'])} "
              f"| P1a f {json.dumps(r['P1a_f_test'])} | A01 f {json.dumps(r['A01_f_test'])}")
        for name in ("model", "P1a", "A01"):
            f = r.get(f"f_{name}")
            if f:
                g = f["e6"]
                print(f"  {name:5s} IG {f['info_gain']['ig_norm']:.3f} | repeat {g['repeat']['share']:.3f}: HR@10 {g['repeat']['hr10']:.4f} "
                      f"C-m {g['repeat']['C_minus_model']:+.4f} | new-f HR@10 {g['new_f']['hr10']:.4f} C-m {g['new_f']['C_minus_model']:+.4f} "
                      f"f-rank med {g['new_f']['true_f_rank_median']:.0f}")
        for name in ("train", "P1a_train", "A01_train"):
            t = r.get(name)
            if t:
                c = t.get("clip") or {}
                print(f"  {name:9s} epochs {t['epochs_run']} best {t['best_epoch']} s/epoch {t['wall_s_per_epoch']:.1f} ({t['gpu']}) "
                      f"clipped {c.get('clipped_share')} norm med {c.get('norm_median')} | e10c {json.dumps(t.get('e10c'))}")
    print("\nE10b beta choice (seed-42 valid):", json.dumps(out["e10b_beta_choice"]))
    for c, v in out["pooled"].items():
        for sp in ("valid", "test"):
            p = v[sp]
            print(f"POOLED {sp} {c} ({p['per_dataset_model']}): HR@10 {p['model']['hr10']:.4f} vs P1a {p['P1a']['hr10']:.4f} "
                  f"{fmt(p['vs_P1a']['hr10'])}, vs A01 {p['A01']['hr10']:.4f} {fmt(p['vs_A01']['hr10'])}")
    if stage:
        key = {"e10c": "E10c", "e10b": "E10b"}[stage]
        lines = []
        p = out["pooled"].get(key)
        if p and p["valid"]["vs_P1a"]["hr10"]["diff"] > 0 and p["valid"]["vs_A01"]["hr10"]["diff"] > 0:
            m = cands[key]
            lines = [f"{D}:{m[D]}" for D in DS if m[D] != "A01"]         # A01 (beta 0.1) seeds 43 / 44 exist
        open(f"{R}/geo/r22_e10_{stage}_stage.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
        print(f"{stage} stage: chain on seeds 43/44:", lines or "nothing")


if __name__ == "__main__":
    main()
