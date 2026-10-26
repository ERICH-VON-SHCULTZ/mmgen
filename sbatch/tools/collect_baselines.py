"""Collect every baseline run under mmgen_runs/baselines into one JSON (for docs/BASELINE_REPORT.md):
per run the five metrics per list and split, training stats, inference cost; per dataset the tokenizer
SID stats; per (dataset, baseline) the across-seed mean/std of the deciding list and the gap to target.
Usage: python3 sbatch/tools/collect_baselines.py [out.json]"""
import glob
import json
import os
import statistics as st
import sys

ROOT = "/scratch/yw8866/mmgen_runs/baselines"
M = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
TARGET = {  # HR@10, NDCG@10 (MACRec paper Table 2; TIGER-text row = TIGER in that table)
    "tiger_text": {"Instruments": (0.1221, 0.0950), "Arts": (0.1167, 0.0806), "Games": (0.0857, 0.0453)},
    "mql": {"Instruments": (0.1291, 0.0997), "Arts": (0.1254, 0.0898), "Games": (0.1007, 0.0538)},
    "macrecpaper": {"Instruments": (0.1363, 0.1046), "Arts": (0.1329, 0.0953), "Games": (0.1078, 0.0565)},
}
TARGET["macrec"] = TARGET["macrecpaper"]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def run_record(d):
    r = {"train": J(f"{d}/train_stats.json"), "config_eval": J(f"{d}/config_eval.json")}
    if r["train"]:
        r["train"].pop("eval_history", None)
    for split in ["valid", "test"]:
        e = f"{d}/eval/{split}"
        lists = {}
        for t, name in [("seqrec", "text"), ("seqimage", "image")]:
            res = J(f"{e}/results_{t}_20.json")
            if res:
                lists[name] = {"metrics": {k: res["mean_results"][k] for k in M}, "cost": J(f"{e}/cost_{t}.json")}
        ens = J(f"{e}/results_ensemble_20.json")
        if ens:
            lists["merged"] = {"metrics": {k: ens[k] for k in M}, "cost": J(f"{e}/cost_merge.json")}
        r[split] = lists
    return r


out = {"runs": {}, "tokenizers": {}, "summary": {}}
for d in sorted(glob.glob(f"{ROOT}/*/*/seed*")):
    D, B, S = d.split("/")[-3:]
    if not S[4:].isdigit():  # e.g. seed42_cancelled_20617403
        continue
    out["runs"][f"{D}/{B}/{S}"] = run_record(d)
for f in sorted(glob.glob(f"{ROOT}/*/tok/*/sid_stats*.json")):
    D, _, tag, name = f.split("/")[-4:]
    s = J(f)
    out["tokenizers"][f"{D}/{tag}/{name[:-5]}"] = {k: s.get(k) for k in
        ["items", "collision_rate_before", "max_group", "all_unique", "unique_final", "utilization_per_level"]}
groups = {}
for k, r in out["runs"].items():
    D, B, S = k.split("/")
    lst = "merged" if "merged" in r["test"] else "text"
    if lst in r["test"]:
        groups.setdefault((D, B), []).append((S, lst, r["test"][lst]["metrics"]))
for (D, B), v in sorted(groups.items()):
    g = {"seeds": [s for s, _, _ in v], "list": v[0][1]}
    for m in M:
        xs = [x[m] for _, _, x in v]
        g[m] = {"mean": st.mean(xs), "std": st.stdev(xs) if len(xs) > 1 else None, "per_seed": xs}
    t = TARGET.get(B, {}).get(D)
    if t:
        g["target"] = {"hit@10": t[0], "ndcg@10": t[1]}
        g["gap_rel"] = {"hit@10": g["hit@10"]["mean"] / t[0] - 1, "ndcg@10": g["ndcg@10"]["mean"] / t[1] - 1}
        g["matched_5pct"] = all(abs(x) <= 0.05 for x in g["gap_rel"].values())
    out["summary"][f"{D}/{B}"] = g
path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/baselines_all.json"
json.dump(out, open(path, "w"), indent=1)
for k, g in out["summary"].items():
    gap = g.get("gap_rel", {})
    print(f"{k:28s} n={len(g['seeds'])} {g['list']:6s} " + " ".join(
        f"{m}={g[m]['mean']:.4f}±{(g[m]['std'] or 0):.4f}" for m in M)
        + (f"  gap HR10 {gap['hit@10']:+.1%} NDCG10 {gap['ndcg@10']:+.1%} match={g['matched_5pct']}" if gap else ""))
print("written", path)
