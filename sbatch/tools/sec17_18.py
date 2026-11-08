"""Tables and paired CIs for docs/MFD_REPORT.md §17 (V2-MFD follow-ups, Parts 1-3) and §18 Task 2 (LR schedule check). CPU, env.

Same sources and bootstrap as geo_sec15_16.py (2,000 resamples of the same users, seed 0, 95% percentile CI; HR@10, NDCG@10).
BEST per dataset = the higher VALIDATION HR@10 at (iii) of V2-MFD and P1a (tie NDCG@10); nothing chosen on test.
C lists are rebuilt from the saved beams with plugin_paired.per_user (asserted against the reported results).
Usage (inside the env): MACREC=/scratch/yw8866/mmgen/macrec python sec17_18.py -> mmgen_runs/geo/sec17_18.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/scratch/yw8866/mmgen/sbatch/tools")
import plugin_paired  # noqa: E402
from geo_sec15_16 import J, R, diag, f, mfd_ranks, paired, score, train  # noqa: E402
from geo_chain_paired import ranks as chain_ranks  # noqa: E402

M = lambda D, run: f"{R}/mfd/{D}/{run}/seed42"


def ms(base, sub):
    if sub == "iii":
        a = J(f"{base}/sweep/apply_test.json")["results"]
        c = J(f"{base}/sweep/chosen.json")["name"]
        a = a.get(c, a)
        return a["cost"]["ms_per_user"], a["cost"]["gpu"]
    c = J(f"{base}/{sub}/results_test.json")["cost"]
    return c["ms_per_user"], c["gpu"]


def mfd_entry(D, run, subs=("eval_kf20", "eval_ii", "iii")):
    b = M(D, run)
    e = {"train": train(b), "diag": diag(b), "chosen": (J(f"{b}/sweep/chosen.json") or {}).get("name"), "settings": {}}
    for s in subs:
        if s == "iii" and not e["chosen"]:
            continue
        x = {sp: mfd_ranks(b, s, sp) for sp in ("valid", "test")}
        m, g = ms(b, s)
        e["settings"][s] = {"valid": score(x["valid"]), "test": score(x["test"]), "ms": m, "gpu": g, "_r": x}
    return e


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "_r"}
    return o


def c_ranks(D, B="mql"):
    return {sp: np.minimum(plugin_paired.per_user(D, B, sp)[0]["merged"], 10) for sp in ("valid", "test")}


def main():
    res = {"part1": {}, "task2": {}, "part2": {}, "part3": {}}
    # Part 1 (Arts) + Task 2
    for v in ("V2", "P1a", "P1b", "P1c", "P1d", "P1e", "V2s", "P1a200"):
        res["part1"][v] = mfd_entry("Arts", f"mfd122_mv_geo{v}")
    P = res["part1"]
    t3 = lambda v, sp: P[v]["settings"]["iii"]["_r"][sp]
    for a, b in (("P1a", "V2"), ("V2s", "V2"), ("P1a200", "P1a"), ("P1a", "V2s"), ("P1a200", "V2")):
        res["task2"][f"{a} - {b}"] = {sp: paired(t3(a, sp), t3(b, sp)) for sp in ("valid", "test")}
    # Part 2: per dataset
    plugin_paired.IDX["../../geo/Arts/c_geo"] = plugin_paired.IDX["mql"]
    for D in ("Instruments", "Arts", "Games"):
        e = {"chain": {}, "mfd_control": None, "runs": {}, "refs": {}}
        cb = f"{R}/baselines/{D}/md122_a05_ep500_reassign/seed42"
        e["chain"] = {sp: score(chain_ranks(cb, sp)) for sp in ("valid", "test")}
        e["mfd_control"] = mfd_entry(D, "mfd122_mv")
        for v in ("V2", "P1a"):
            e["runs"][v] = P[v] if D == "Arts" else mfd_entry(D, f"mfd122_mv_geo{v}")
        key = lambda v: (e["runs"][v]["settings"]["iii"]["valid"]["hr10"], e["runs"][v]["settings"]["iii"]["valid"]["ndcg10"])
        best = max(("V2", "P1a"), key=key)
        e["best"] = best
        C = c_ranks(D)
        e["refs"]["C merged"] = {sp: score(C[sp]) for sp in ("valid", "test")}
        if D != "Games":
            pg = c_ranks(D, "plug122")
            e["refs"]["plug-in 1+2+2 merged"] = {sp: score(pg[sp]) for sp in ("valid", "test")}
        e["vs_C"] = {}
        for v in ("V2", "P1a"):
            r = e["runs"][v]["settings"]["iii"]["_r"]
            e["vs_C"][v] = {sp: paired(r[sp], C[sp]) for sp in ("valid", "test")}
        r = e["runs"]["P1a"]["settings"]["iii"]["_r"]; r2 = e["runs"]["V2"]["settings"]["iii"]["_r"]
        e["P1a_vs_V2"] = {sp: paired(r[sp], r2[sp]) for sp in ("valid", "test")}
        cs = e["mfd_control"]["settings"]
        cr = (cs.get("iii") or cs["eval_ii"])["_r"]
        e["best_vs_mfd_control"] = {sp: paired(e["runs"][best]["settings"]["iii"]["_r"][sp], cr[sp]) for sp in ("valid", "test")}
        if D == "Arts":
            G = c_ranks(D, "../../geo/Arts/c_geo")
            e["refs"]["C + codebook init merged"] = {sp: score(G[sp]) for sp in ("valid", "test")}
            res["part3"] = {"C_geo_vs_C": {sp: paired(G[sp], C[sp]) for sp in ("valid", "test")},
                            "best_vs_C_geo": {sp: paired(e["runs"][best]["settings"]["iii"]["_r"][sp], G[sp]) for sp in ("valid", "test")}}
        res["part2"][D] = e
    json.dump(strip(res), open(f"{R}/geo/sec17_18.json", "w"), indent=1)
    # print
    print("== Part 1 / Task 2 (Arts)")
    for v, e in P.items():
        t, d = e["train"], e["diag"]
        for s, st in e["settings"].items():
            print(f" {v:7s} ({s}) valid {st['valid']['hr1']:.4f} {st['valid']['hr10']:.4f} {st['valid']['ndcg10']:.4f} | test {st['test']['hr1']:.4f} "
                  f"{st['test']['hr10']:.4f} {st['test']['ndcg10']:.4f} | ms {st['ms']:.1f} {st['gpu'][-9:]}")
        print(f"   chosen {e['chosen']} ep {t['epochs']:.0f} best_ep {t['best_epoch']} steps {t['steps']} best_step {t['best_step']} vl {t['best_valid_loss']:.4f}"
              f" | f test {d['f_top1']:.4f}/{d['f_top20']:.4f} valid {d['valid_f_top1']:.4f}/{d['valid_f_top20']:.4f}")
    for k, v in res["task2"].items():
        print(f" {k}: valid HR@10 {f(v['valid']['hr10'])} | test HR@10 {f(v['test']['hr10'])} NDCG@10 {f(v['test']['ndcg10'])}")
    for D, e in res["part2"].items():
        print(f"== Part 2 {D}: BEST = {e['best']}")
        print(f"  chain test {e['chain']['test']['hr10']:.4f}/{e['chain']['test']['ndcg10']:.4f}")
        for n, x in [("mfd_control", e["mfd_control"])] + list(e["runs"].items()):
            for s, st in x["settings"].items():
                print(f"  {n} ({s}) valid {st['valid']['hr10']:.4f}/{st['valid']['ndcg10']:.4f} test {st['test']['hr1']:.4f} {st['test']['hr10']:.4f}/{st['test']['ndcg10']:.4f} ms {st['ms']:.1f} {st['gpu'][-9:]}")
            if n != "mfd_control":
                t, d = x["train"], x["diag"]
                print(f"    chosen {x['chosen']} ep {t['epochs']:.0f} best {t['best_epoch']} vl {t['best_valid_loss']:.4f} f test {d['f_top1']:.4f}/{d['f_top20']:.4f}")
        for n, x in e["refs"].items():
            print(f"  {n}: valid {x['valid']['hr10']:.4f}/{x['valid']['ndcg10']:.4f} test {x['test']['hr1']:.4f} {x['test']['hr10']:.4f}/{x['test']['ndcg10']:.4f}")
        for v, p in e["vs_C"].items():
            print(f"  {v} - C: test HR@10 {f(p['test']['hr10'])} NDCG@10 {f(p['test']['ndcg10'])} | valid HR@10 {f(p['valid']['hr10'])}")
        p = e["P1a_vs_V2"]; print(f"  P1a - V2: test {f(p['test']['hr10'])} / {f(p['test']['ndcg10'])} | valid {f(p['valid']['hr10'])}")
        p = e["best_vs_mfd_control"]; print(f"  BEST - MFD control: test {f(p['test']['hr10'])} / {f(p['test']['ndcg10'])} | valid {f(p['valid']['hr10'])}")
    for k, p in res["part3"].items():
        print(f" Part 3 {k}: test HR@10 {f(p['test']['hr10'])} NDCG@10 {f(p['test']['ndcg10'])} | valid HR@10 {f(p['valid']['hr10'])}")


if __name__ == "__main__":
    main()
