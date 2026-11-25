"""mmgen addition (docs/MFD_REPORT.md §24.4): baseline tuning fairness (CPU). Baselines C (mql), D (macrecpaper), TIGER-text at their own
settings (frozen runs, baselines/) and at the tuned learning rates (baselines_tuned/), Arts + Instruments seed 42, vs P1a.
Ranks come from the saved beam outputs (C / D merged with the MQL4GRec rule, TIGER single list) and every HR@10 is checked against the
evaluation's own results file. Per baseline: the configuration with the best POOLED Arts + Instruments validation HR@10 (NDCG@10
tie-break) is selected (validation only); per-dataset bests are listed for information. Test: paired comparison (per dataset and
pooled stratified bootstrap) of P1a lr 2e-3, and of P1a's validation-selected pure-lr run, vs each baseline's own-settings run and its
selected run. A missing configuration is listed and not used. -> geo/r24_baselines.json"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r22_e10_report as E  # noqa: E402
from plugin_paired import paired  # noqa: E402
from round20_collect import rank_of  # noqa: E402

R, DS = E.R, ("Arts", "Instruments")
TASKS = {"mql": (("seqrec", ".index_lemb_mql.json"), ("seqimage", ".index_vitemb_mql.json")),
         "macrecpaper": (("seqrec", ".index_lemb_macrecpaper.json"), ("seqimage", ".index_vitemb_macrecpaper.json")),
         "tiger_text": (("seqrec", ".index_lemb_tiger.json"),)}
NAME = {"mql": "C (MQL4GRec)", "macrecpaper": "D (MACRec)", "tiger_text": "TIGER-text"}
OWN = {"mql": "own: lr 5e-4, 256x4", "macrecpaper": "own: lr 1e-3, 1024x4", "tiger_text": "own: lr 5e-4, 256x4"}
PURE = ("lr1e3", "lr2e3", "lr3e3", "lr4e3")


def ranks(D, B, rd, sp):
    ev = f"{rd}/eval/{sp}"
    rows, U = [], None
    for task, fn in TASKS[B]:
        sv = json.load(open(f"{ev}/save_{task}_20.json"))
        s2i = {"".join(v): int(k) for k, v in json.load(open(f"{R}/data/{D}/{D}{fn}")).items()}
        U = len(sv["all_users"])
        nb = len(sv["all_outputs"]) // U
        d = {}
        for k, u in enumerate(sv["all_users"]):
            o = [x.strip().replace(" ", "") for x in sv["all_outputs"][k * nb:(k + 1) * nb]]
            d[u] = [(s2i[a], b) for a, b in zip(o, sv["all_scores"][k * nb:(k + 1) * nb]) if a in s2i]
        rows.append(d)
    top = np.full((U, 10), -1, dtype=np.int64)
    for u in range(U):
        sc = {}
        for j in range(20):
            for d in rows:
                if j < len(d[u]):
                    a, b = d[u][j]
                    sc[a] = ((b + sc[a]) / 2 + 1 if a in sc else b) if len(rows) == 2 else sc.get(a, b)
        t = [a for a, _ in sorted(sc.items(), key=lambda p: p[1], reverse=True)][:10]
        top[u, :len(t)] = t
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    rk = rank_of(top, np.array([s[-1] if sp == "test" else s[-2] for s in inter.values()]))
    f = f"{ev}/results_ensemble_20.json" if len(rows) == 2 else f"{ev}/results_seqrec_20.json"
    d = json.load(open(f))
    rep = d.get("mean_results", d)["hit@10"]
    assert abs(E.hr(rk).mean() - rep) < 1.5 / U, (D, B, rd, sp, float(E.hr(rk).mean()), rep)
    return rk


def curve(rd):
    s = json.load(open(f"{rd}/train_stats.json"))
    ls = [l for _, l in s.get("eval_history", [])]
    ups = [b - a for a, b in zip(ls, ls[1:])]
    return {"epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"], "steps": s["global_steps"], "best_eval_loss": s["best_eval_loss"],
            "max_epoch_rise": max(ups) if ups else None, "nonfinite": int(sum(not np.isfinite(x) for x in ls)), "wall_h": s["train_wall_seconds"] / 3600, "gpu": s["gpu"]}


def main():
    out, RK = {"configs": {}, "missing": []}, {}
    for B in TASKS:
        tags = {None} | {os.path.basename(p)[len(B) + 1:] for p in glob.glob(f"{R}/baselines_tuned/*/{B}_*")}
        for t in sorted(tags, key=lambda x: x or ""):
            rd = {D: f"{R}/baselines/{D}/{B}/seed42" if t is None else f"{R}/baselines_tuned/{D}/{B}_{t}/seed42" for D in DS}
            last = "ensemble" if len(TASKS[B]) == 2 else "seqrec"
            if not all(os.path.exists(f"{rd[D]}/eval/{sp}/results_{last}_20.json") for D in DS for sp in ("valid", "test")):
                out["missing"].append(f"{B} {t}")
                continue
            RK[(B, t)] = {sp: {D: ranks(D, B, rd[D], sp) for D in DS} for sp in ("valid", "test")}
            c = {"curve": {D: curve(rd[D]) for D in DS}}
            for sp in ("valid", "test"):
                rk = RK[(B, t)][sp]
                tot = sum(map(len, rk.values()))
                c[sp] = {D: {"hr10": float(E.hr(rk[D]).mean()), "ndcg10": float(E.ndcg(rk[D]).mean())} for D in DS}
                c[sp]["pooled"] = {"hr10": float(sum(E.hr(v).sum() for v in rk.values()) / tot), "ndcg10": float(sum(E.ndcg(v).sum() for v in rk.values()) / tot)}
            out["configs"][f"{B} {t or 'own'}"] = c
    M = {}
    for n in ("P1a",) + tuple(f"r24_{v}" for v in PURE + ("lr1e3_do02", "lr2e3_do02")):
        ok = all(os.path.exists(f"{E.run_dir(D, n, 42)}/r21_eval/ranks.pt") for D in DS)
        if ok:
            M[n] = {sp: {D: E.ev(D, n, 42)[1][sp] for D in DS} for sp in ("valid", "test")}
    pv = lambda rk: (sum(E.hr(v).sum() for v in rk.values()) / sum(map(len, rk.values())), sum(E.ndcg(v).sum() for v in rk.values()) / sum(map(len, rk.values())))
    pure = [n for n in M if n[4:] in PURE]
    psel = max(pure, key=lambda n: pv(M[n]["valid"])) if pure else None
    out["P1a_selected_pure_lr"] = psel
    out["P1a_configs"] = {n: {sp: dict(zip(("hr10", "ndcg10"), map(float, pv(M[n][sp])))) for sp in ("valid", "test")} for n in M}
    out["selected"], out["comparisons"] = {}, {}
    for B in TASKS:
        cand = [k for k in RK if k[0] == B]
        sel = max(cand, key=lambda k: pv(RK[k]["valid"]))
        out["selected"][B] = {"pooled_valid": sel[1] or "own", "n_configs": len(cand),
                              "per_dataset": {D: (max(cand, key=lambda k: E.hr(RK[k]["valid"][D]).mean())[1] or "own") for D in DS}}
        for n in dict.fromkeys(("r24_lr2e3", psel)):
            if n not in M:
                continue
            for lab, k in (("own", (B, None)), ("selected", sel)):
                r, rng, diffs = {}, np.random.default_rng(0), {}
                for m, f in (("hr10", E.hr), ("ndcg10", E.ndcg)):
                    diffs[m] = {D: f(M[n]["test"][D]) - f(RK[k]["test"][D]) for D in DS}
                    r[f"pooled {m}"] = E.strat_boot(diffs[m], rng)
                for D in DS:
                    nU = len(M[n]["test"][D])
                    r[D] = paired(M[n]["test"][D], RK[k]["test"][D], np.random.default_rng(0).integers(0, nU, size=(2000, nU)))
                out["comparisons"][f"{n} - {B} {lab} ({k[1] or 'own'})"] = r
    json.dump(out, open(f"{R}/geo/r24_baselines.json", "w"), indent=1)
    fm = lambda c: f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
    print("missing (not used):", out["missing"] or "none")
    for k, c in out["configs"].items():
        B = k.split()[0]
        cu = " ".join(f"{D}: ep {x['best_epoch']:.0f}/{x['epochs_run']:.0f} loss {x['best_eval_loss']:.4f} max rise {x['max_epoch_rise']:.3f} nonfinite {x['nonfinite']}" for D, x in c["curve"].items())
        print(f"{NAME[B]:14s} {k.split()[1]:12s} valid pooled {c['valid']['pooled']['hr10']:.4f}/{c['valid']['pooled']['ndcg10']:.4f} (Arts {c['valid']['Arts']['hr10']:.4f} "
              f"Instr {c['valid']['Instruments']['hr10']:.4f}) | test pooled {c['test']['pooled']['hr10']:.4f}/{c['test']['pooled']['ndcg10']:.4f} "
              f"(Arts {c['test']['Arts']['hr10']:.4f} Instr {c['test']['Instruments']['hr10']:.4f}) | {cu}")
    print("P1a:", json.dumps({n: {sp: round(v["hr10"], 4) for sp, v in x.items()} for n, x in out["P1a_configs"].items()}), "| selected pure lr", psel)
    print("selected on pooled validation:", json.dumps(out["selected"]))
    for k, r in out["comparisons"].items():
        print(f"TEST {k}: pooled HR {fm(r['pooled hr10'])} NDCG {fm(r['pooled ndcg10'])} | " + " | ".join(f"{D} HR {fm(r[D]['hr10'])} NDCG {fm(r[D]['ndcg10'])}" for D in DS))


if __name__ == "__main__":
    main()
