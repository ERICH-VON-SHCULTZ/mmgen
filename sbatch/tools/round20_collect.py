"""§20 Task 2 / Task 3 / Task 0-dense collection (docs/MFD_REPORT.md §20). CPU, inside the env, next to expl_top10.py.

Task 3: per dataset and seed, P1a plain ((ii), eval_ii ranked_test) and P1a+1c (mv_infer apply_test_ranks, 1c chosen on that
  seed's own validation); mean / std over seeds 42/43/44 next to C (MQL4GRec merged) and D (MACRec paper setting), whose
  per-seed numbers come from their results_ensemble_20.json; paired CIs per seed vs C merged of the SAME seed (users aligned
  by dataset index; 2,000 resamples, seed 0, 95% percentile).
Task 2: per dense run (mfd122_mv_geo<VAR>/seed42/dense_eval): every chosen test config, dense-only, best epoch, CIs vs
  C merged seed 42, vs P1a seed 42 (plain and +1c) and vs the same run at w_d = 0.
Task 0-dense: on C-hit / P2-at-w_d=0-miss users (and on the seed-42 C-hit / P1a-miss users), how many the selected dense
  config rescues, how many it loses, and the target's f rank (full-view log P(f)) / f rank by the selection term / dense
  rank for rescued users.
Usage: python round20_collect.py <D> [VAR ...]  -> mmgen_runs/geo/round20_<D>.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expl_top10 import R, rank_of  # noqa: E402
from plugin_paired import ndcg, paired  # noqa: E402

BIG = 10 ** 6
SEEDS = (42, 43, 44)


def c_lists_seed(D, seed, split="test", model="mql"):
    """C merged top-10 lists of any seed: expl_top10.c_lists' rule (that function hard-codes seed 42)."""
    ev = f"{R}/baselines/{D}/{model}/seed{seed}/eval/{split}"
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
    out = np.full((U, 10), -1, dtype=np.int64)
    for u in range(U):
        t, i = infos["seqrec"][u], infos["seqimage"][u]
        sc = {}
        for j in range(20):
            for lst in (t, i):
                if j < len(lst):
                    a, b = lst[j]
                    sc[a] = (b + sc[a]) / 2 + 1 if a in sc else b
        top = [a for a, _ in sorted(sc.items(), key=lambda p: p[1], reverse=True)][:10]
        out[u, :len(top)] = top
    return out


def met(r):
    return {"hr10": float((r < 10).mean()), "ndcg10": float(ndcg(r).mean())}


def ms(xs):
    a = np.array(xs)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "per_seed": [float(x) for x in a]}


def best_epoch(run):
    st = json.load(open(f"{run}/trainer_state.json"))
    step = int(st["best_model_checkpoint"].rsplit("-", 1)[1])
    ep = [h["epoch"] for h in st["log_history"] if h.get("step") == step and "eval_loss" in h]
    return {"best_step": step, "best_epoch": float(ep[0]) if ep else None, "stopped_epoch": float(st["epoch"]),
            "best_eval_loss": st["best_metric"]}


def main():
    D, VARS = sys.argv[1], sys.argv[2:]
    inter = json.load(open(f"{R}/data/{D}/{D}.inter.json"))
    tg = np.array([inter[u][-1] for u in inter])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(tg), size=(2000, len(tg)))
    out = {"dataset": D, "users": int(len(tg))}
    base = f"{R}/mfd/{D}/mfd122_mv_geoP1a"
    # ---------------- Task 3
    t3, ranks = {}, {}
    for s in SEEDS:
        b = f"{base}/seed{s}"
        E = torch.load(f"{b}/eval_ii/ranked_test.pt", weights_only=False)
        assert np.array_equal(E["targets"].numpy(), tg), "user order"
        m = E["joint"][0][:, :10] == E["targets"][:, None]
        plain = torch.where(m.any(1), m.int().argmax(1), torch.full_like(E["targets"], BIG)).numpy()
        rep = json.load(open(f"{b}/eval_ii/results_test.json"))["modes"]["joint"]
        assert abs(rep["hit@10"] - (plain < 10).mean()) < 1e-6
        A = json.load(open(f"{b}/mv_infer/apply_test.json"))
        c1 = torch.load(f"{b}/mv_infer/apply_test_ranks.pt", weights_only=False)["1c_separate"].numpy()
        assert abs(A["1c_separate"]["hr10"] - (c1 < 10).mean()) < 1e-6
        C = rank_of(c_lists_seed(D, s), tg)
        cr = json.load(open(f"{R}/baselines/{D}/mql/seed{s}/eval/test/results_ensemble_20.json"))
        assert abs(cr["hit@10"] - (C < 10).mean()) < 1e-6, "C lists"
        ranks[s] = {"plain": plain, "1c": c1, "C": C}
        t3[s] = {"P1a": met(plain), "P1a_1c": met(c1), "C": met(C), "1c_config": A["1c_separate"]["name"],
                 "1c_ms_per_user": A["1c_separate"]["ms_per_user"], "gpu": A["1c_separate"]["gpu"],
                 "P1a_vs_C": paired(plain, C, boot), "P1a_1c_vs_C": paired(c1, C, boot),
                 "P1a_1c_vs_P1a": paired(c1, plain, boot), "best_epoch": best_epoch(b)}
    agg = {}
    for k in ("P1a", "P1a_1c", "C"):
        agg[k] = {m_: ms([t3[s][k][m_] for s in SEEDS]) for m_ in ("hr10", "ndcg10")}
    Dm = [json.load(open(f"{R}/baselines/{D}/macrecpaper/seed{s}/eval/test/results_ensemble_20.json")) for s in SEEDS]
    agg["D"] = {"hr10": ms([d["hit@10"] for d in Dm]), "ndcg10": ms([d["ndcg@10"] for d in Dm])}
    out["task3"] = {"per_seed": {str(s): v for s, v in t3.items()}, "mean_std": agg}
    # ---------------- Task 2 + Task 0-dense
    t2 = {}
    p42, c42 = ranks[42]["plain"], ranks[42]["C"]
    for V in VARS:
        b = f"{R}/mfd/{D}/mfd122_mv_geo{V}/seed42"
        if not os.path.exists(f"{b}/dense_eval/apply_test_ranks.pt"):
            continue
        A = json.load(open(f"{b}/dense_eval/apply_test.json"))
        ch = json.load(open(f"{b}/dense_eval/chosen.json"))
        sw = json.load(open(f"{b}/dense_eval/sweep_valid.json"))
        X = torch.load(f"{b}/dense_eval/apply_test_ranks.pt", weights_only=False)
        rk = {k: v.numpy() for k, v in X["ranks"].items()}
        name = lambda c: f"{c['rule']}_wd{c['w_d']}_a{c['a_t']}-{c['a_i']}"
        sel = name(ch["overall"])
        zero = name(ch["plain"])
        res = {"best_epoch": best_epoch(b), "chosen_valid": {g: {k: ch[g][k] for k in ("rule", "w_d", "a_t", "a_i", "hr10", "ndcg10")} for g in ch},
               "dense_only_valid": sw["dense_only"], "dense_only_test": A["dense_only"], "test": {}}
        for k, r in rk.items():
            res["test"][k] = dict(A[k], vs_C=paired(r, c42, boot), vs_P1a=paired(r, p42, boot),
                                  vs_P1a_1c=paired(r, ranks[42]["1c"], boot), vs_same_run_wd0=paired(r, rk[zero], boot))
        # Task 0-dense
        dg = {a: b_.numpy() for a, b_ in X["diag"][sel].items()}
        rs, r0 = rk[sel], rk[zero]
        dec = {}
        for nm, setm in (("c_hit_wd0_miss", (c42 < 10) & (r0 >= 10)), ("c_hit_p1a_miss_seed42", (c42 < 10) & (p42 >= 10))):
            base_r = r0 if nm.startswith("c_hit_wd0") else p42
            resc = setm & (rs < 10)
            dec[nm] = {"users": int(setm.sum()), "rescued_by_selected": int(resc.sum()),
                       "lost_elsewhere_hit_before_miss_now": int(((base_r < 10) & (rs >= 10)).sum()),
                       "rescued_f_rank_full_median": float(np.median(dg["f_rank_full"][resc])) if resc.any() else None,
                       "rescued_f_rank_full_top1_share": float((dg["f_rank_full"][resc] == 0).mean()) if resc.any() else None,
                       "rescued_f_rank_term_eq_full_share": float((dg["f_rank_term"][resc] == dg["f_rank_full"][resc]).mean()) if resc.any() else None,
                       "rescued_dense_rank_lt10_share": float((dg["dense_rank"][resc] < 10).mean()) if resc.any() else None,
                       "set_dense_rank_lt10_share": float((dg["dense_rank"][setm] < 10).mean()),
                       "set_f_rank_term_up_vs_full_share": float((dg["f_rank_term"][setm] < dg["f_rank_full"][setm]).mean())}
        res["selected"], res["wd0"], res["task0_dense"] = sel, zero, dec
        t2[V] = res
    out["task2"] = t2
    json.dump(out, open(f"{R}/geo/round20_{D}.json", "w"), indent=1)
    print(json.dumps({"task3": agg, "task2": {V: {"selected": t2[V]["selected"], "test": {k: (v["hr10"], v["ndcg10"]) for k, v in t2[V]["test"].items()},
                                                  "task0_dense": t2[V]["task0_dense"]} for V in t2}}, indent=1))


if __name__ == "__main__":
    main()
