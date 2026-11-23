"""mmgen addition (docs/MFD_REPORT.md §24): round-24 report and selection (CPU, from saved ranks / fdumps / stats).
Every run mfd/<D>/r24_<VAR>/seed<S> with r21_eval (plain K_f = 50): valid / test HR@1/5/10, NDCG@5/10; paired CIs vs P1a and vs C
of the same seed; E6 split (repeat-f / new-f HR@10, C - model, true-f median rank on new-f users) from its fdump; f diagnostics;
best epoch, optimizer steps, wall time, clipping, examples per epoch; ms/user. POOLED seed-42 Arts + Instruments (the screening
set) per VAR: HR@10 / NDCG@10 vs P1a and C, stratified paired bootstrap (r22_e10_report.strat_boot), valid and test.
Selection (validation only): a VAR "qualifies" when its pooled Arts + Instruments seed-42 valid HR@10 > P1a's.
  --stage a1     A1 (lr3e4 lr1e3 do02 do03 wd01 gn5): best qualifying lr (lr3e4 / lr1e3) and best qualifying regularisation
                 (do02 do03 wd01 gn5) by pooled valid HR@10 (NDCG@10 tie-break); both carried into A2 / the combination (together if
                 both exist: a combination A1 never tested) -> geo/r24_stage_a1.json (carry, reasons)
  --stage ab     A-best = best qualifying Track A VAR (A1 + A2), B-best = best qualifying of B1 / B2; combo = A-best tokens
                 (+ the carried lr if A-best has none) + B token (if both exist) -> geo/r24_stage_ab.json
  --stage best   best of {A-best, B-best, combo} by pooled valid HR@10 (qualifying only) -> geo/r24_stage_best.json
Out: geo/r24_report.json + printed tables.  Usage: python r24_report.py [--stage a1|ab|best]"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r22_e10_report as E  # noqa: E402
from plugin_paired import paired  # noqa: E402
from r22_runs_report import ev, f_analysis  # noqa: E402

R = E.R
KEYS = ("hr1", "hr5", "hr10", "ndcg5", "ndcg10")
SCREEN = ("Arts", "Instruments")
A1 = ("lr3e4", "lr1e3", "do02", "do03", "wd01", "gn5")
REG = ("do02", "do03", "wd01", "gn5")


def runs():
    out = {}
    for p in sorted(glob.glob(f"{R}/mfd/*/r24_*/seed*/r21_eval/ranks.pt")):
        rd = p.rsplit("/r21_eval/", 1)[0]
        D, V, S = rd.split("/")[-3], rd.split("/")[-2][4:], int(rd.split("/")[-1][4:])
        out[(D, V, S)] = rd
    return out


def one(D, V, S, rd):
    e, rk = ev(rd)
    eb, rb = E.ev(D, "P1a", S)
    n = len(rk["test"])
    boot = np.random.default_rng(0).integers(0, n, size=(2000, n))
    r = {"D": D, "var": V, "seed": S}
    for sp in ("valid", "test"):
        rc = E.C_ranks(D, S, sp)
        r[sp] = {k: e[sp]["plain"][k] for k in KEYS}
        r[f"P1a_{sp}"] = {k: eb[sp]["plain"][k] for k in KEYS}
        r[f"C_{sp}"] = {"hr10": float(E.hr(rc).mean()), "ndcg10": float(E.ndcg(rc).mean())}
        r[f"vs_P1a_{sp}"] = paired(rk[sp], rb[sp], boot)
        r[f"vs_C_{sp}"] = paired(rk[sp], rc, boot)
    r["f_test"] = e["test"].get("f")
    r["ms_per_user"] = e["test"]["plain"].get("ms_per_user")
    Ct = E.C_ranks(D, S, "test")
    fa = f_analysis(D, f"{R}/geo/r22/{D}/fdump_r24_{V}_s{S}.pt", "1+2+2", rk["test"], Ct)
    fb = f_analysis(D, E.fdump_path(D, "P1a", S), "1+2+2", rb["test"], Ct)
    r["e6"] = fa[0]["e6"] if fa else None
    r["P1a_e6"] = fb[0]["e6"] if fb else None
    r["ig_norm"] = fa[0]["info_gain"]["ig_norm"] if fa else None
    st = json.load(open(f"{rd}/train_stats.json"))
    r["train"] = {k: st.get(k) for k in ("epochs_run", "best_epoch", "global_steps", "train_wall_seconds", "gpu",
                                         "train_examples_per_epoch", "train_examples_per_task", "peak_gpu_mem_gb")}
    r["train"]["r24"] = st.get("r24")
    return r


def pooled(V, S=42):
    """pooled Arts + Instruments seed S: model vs P1a and vs C, valid and test."""
    res, rng = {}, np.random.default_rng(0)
    for sp in ("valid", "test"):
        rk, rb, rc = {}, {}, {}
        for D in SCREEN:
            rd = f"{R}/mfd/{D}/r24_{V}/seed{S}"
            if not os.path.exists(f"{rd}/r21_eval/ranks.pt"):
                return None
            rk[D] = ev(rd)[1][sp]
            rb[D] = E.ev(D, "P1a", S)[1][sp]
            rc[D] = E.C_ranks(D, S, sp)
        tot = sum(len(v) for v in rk.values())
        m = lambda d, f: float(sum(f(v).sum() for v in d.values()) / tot)
        res[sp] = {"model": {"hr10": m(rk, E.hr), "ndcg10": m(rk, E.ndcg)}, "P1a": {"hr10": m(rb, E.hr), "ndcg10": m(rb, E.ndcg)},
                   "C": {"hr10": m(rc, E.hr), "ndcg10": m(rc, E.ndcg)}}
        for b, rr in (("P1a", rb), ("C", rc)):
            res[sp][f"vs_{b}"] = {mm: E.strat_boot({D: f(rk[D]) - f(rr[D]) for D in SCREEN}, rng)
                                  for mm, f in (("hr10", E.hr), ("ndcg10", E.ndcg))}
    res["qualifies"] = res["valid"]["model"]["hr10"] > res["valid"]["P1a"]["hr10"]
    return res


def key(p):
    return (p["valid"]["model"]["hr10"], p["valid"]["model"]["ndcg10"])


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else None
    rs = runs()
    out = {"runs": {f"{D} {V} s{S}": one(D, V, S, rd) for (D, V, S), rd in rs.items()}}
    vars42 = sorted({V for (D, V, S) in rs if S == 42})
    out["pooled_screen"] = {V: p for V in vars42 if (p := pooled(V))}
    json.dump(out, open(f"{R}/geo/r24_report.json", "w"), indent=1)
    P = out["pooled_screen"]
    fmt = lambda c: f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
    print("== pooled Arts + Instruments, seed 42 (valid | test) ==")
    for V, p in P.items():
        v, t = p["valid"], p["test"]
        print(f"{V:14s} valid HR@10 {v['model']['hr10']:.4f} (P1a {v['P1a']['hr10']:.4f}, C {v['C']['hr10']:.4f}) vs P1a {fmt(v['vs_P1a']['hr10'])}"
              f" | test HR@10 {t['model']['hr10']:.4f} vs P1a {fmt(t['vs_P1a']['hr10'])} vs C {fmt(t['vs_C']['hr10'])}"
              f" NDCG vs C {fmt(t['vs_C']['ndcg10'])} | qualifies {p['qualifies']}")
    for k, r in out["runs"].items():
        print(f"\n=== {k}")
        for sp in ("valid", "test"):
            print(f"{sp:5s} " + " ".join(f"{m} {r[sp][m]:.4f}" for m in KEYS) + f" | vs P1a HR {fmt(r[f'vs_P1a_{sp}']['hr10'])} NDCG "
                  f"{fmt(r[f'vs_P1a_{sp}']['ndcg10'])} | vs C HR {fmt(r[f'vs_C_{sp}']['hr10'])} NDCG {fmt(r[f'vs_C_{sp}']['ndcg10'])}")
        for name, g in (("model", r["e6"]), ("P1a", r["P1a_e6"])):
            if g:
                print(f"  E6 {name:5s} repeat {g['repeat']['share']:.3f}: HR@10 {g['repeat']['hr10']:.4f} C-m {g['repeat']['C_minus_model']:+.4f}"
                      f" | new-f HR@10 {g['new_f']['hr10']:.4f} C-m {g['new_f']['C_minus_model']:+.4f} f-rank med {g['new_f']['true_f_rank_median']:.0f}")
        t = r["train"]
        print(f"  f {json.dumps(r['f_test'])} | IG {r['ig_norm']} | ms/user {r['ms_per_user']}")
        print(f"  train epochs {t['epochs_run']} best {t['best_epoch']} steps {t['global_steps']} wall {t['train_wall_seconds']} s "
              f"{t['gpu']} examples/epoch {t['train_examples_per_task']} | {json.dumps((t['r24'] or {}).get('clip'))}")
    if stage == "a1":
        # lr is part of A-best (user, 2026-11-23): the best qualifying lr and the best qualifying regularisation are carried
        # into A2 and into the A+B combination, together if both exist (each A1 run changes one factor, so lr + reg together
        # is a combination A1 never tested).
        p1a = next(iter(P.values()))["valid"]["P1a"]["hr10"] if P else None
        q = lambda Vs: [V for V in Vs if V in P and P[V]["qualifies"]]
        lr, reg = q(("lr3e4", "lr1e3")), q(REG)
        best_lr = max(lr, key=lambda V: key(P[V])) if lr else ""
        best_reg = max(reg, key=lambda V: key(P[V])) if reg else ""
        carry = "_".join(t for t in (best_lr, best_reg) if t)
        why = []
        for V in A1:
            if V not in P:
                why.append(f"{V}: no pooled result (missing eval)")
                continue
            v = P[V]["valid"]["model"]
            tag = "carried (best lr)" if V == best_lr else "carried (best regularisation)" if V == best_reg else \
                ("qualifies but not the best of its kind" if P[V]["qualifies"] else "does not beat P1a")
            why.append(f"{V}: pooled valid HR@10 {v['hr10']:.4f} / NDCG@10 {v['ndcg10']:.4f} vs P1a {p1a:.4f} -> {tag}")
        res = {"qualifying_A1": q(A1), "best_lr": best_lr, "best_regularisation": best_reg, "carry": carry,
               "lr_and_reg_untested_together": bool(best_lr and best_reg), "reasons": why,
               "pooled_valid_hr10": {V: P[V]["valid"]["model"]["hr10"] for V in A1 if V in P}, "P1a_pooled_valid_hr10": p1a}
        json.dump(res, open(f"{R}/geo/r24_stage_a1.json", "w"), indent=1)
        print("\nSTAGE a1:", json.dumps(res, indent=1))
    elif stage == "ab":
        A = [V for V in P if P[V]["qualifies"] and not any(t in V.split("_") for t in ("B1", "B2"))]
        B = [V for V in ("B1", "B2") if V in P and P[V]["qualifies"]]
        a = max(A, key=lambda V: key(P[V])) if A else None
        b = max(B, key=lambda V: key(P[V])) if B else None
        a1 = json.load(open(f"{R}/geo/r24_stage_a1.json"))
        lr = a1.get("best_lr") or ""
        ca = a                                                   # the carried lr goes into the combination (user, 2026-11-23)
        if a and lr and not any(t.startswith("lr") for t in a.split("_")):
            ca = f"{a}_{lr}"
        res = {"A_best": a, "B_best": b, "combo": f"{ca}_{b}" if a and b else None, "A_qualifying": A, "B_qualifying": B,
               "carried_lr": lr or None, "combo_adds_carried_lr": ca != a,
               "note": None if ca == a else f"A-best {a} has no lr token; the carried lr {lr} is added to the combination (untested with {a})"}
        json.dump(res, open(f"{R}/geo/r24_stage_ab.json", "w"), indent=1)
        print("\nSTAGE ab:", json.dumps(res))
    elif stage == "best":
        ab = json.load(open(f"{R}/geo/r24_stage_ab.json"))
        cand = [V for V in (ab["A_best"], ab["B_best"], ab["combo"]) if V and V in P and P[V]["qualifies"]]
        res = {"candidates": cand, "best": max(cand, key=lambda V: key(P[V])) if cand else None,
               "pooled_valid_hr10": {V: P[V]["valid"]["model"]["hr10"] for V in cand}}
        json.dump(res, open(f"{R}/geo/r24_stage_best.json", "w"), indent=1)
        print("\nSTAGE best:", json.dumps(res))


if __name__ == "__main__":
    main()
