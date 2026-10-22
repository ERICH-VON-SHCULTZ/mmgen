"""Collect one baseline run into <run>/run_summary.json: config, training stats, peak GPU memory
(nvidia-smi trace), valid/test metrics per list (text/seqrec, image/seqimage, merged), inference cost."""
import json
import os
import sys

run = sys.argv[1]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
s = {"run": run, "config": J(f"{run}/config.json"), "config_eval": J(f"{run}/config_eval.json"),
     "train": J(f"{run}/train_stats.json")}
trace = f"{run}/gpu_mem_trace.txt"
if os.path.exists(trace):
    vals = [int(x) for x in open(trace).read().split() if x.strip().isdigit()]
    s["peak_gpu_mem_mib_nvidia_smi"] = max(vals) if vals else None
for split in ["valid", "test"]:
    d = f"{run}/eval/{split}"
    r = {}
    for t in ["seqrec", "seqimage"]:
        res = J(f"{d}/results_{t}_20.json")
        if res:
            r[t] = {"metrics": res["mean_results"], "cost": J(f"{d}/cost_{t}.json")}
    ens = J(f"{d}/results_ensemble_20.json")
    if ens:
        r["merged"] = {"metrics": ens, "cost": J(f"{d}/cost_merge.json")}
    s[split] = r
json.dump(s, open(f"{run}/run_summary.json", "w"), indent=1)
for split in ["valid", "test"]:
    for k, v in s[split].items():
        m = v["metrics"]
        print(f"{split:5s} {k:8s} " + " ".join(f"{n}={m[n]:.4f}" for n in ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]))
