"""Paired bootstrap of the codebook-geometry chain runs vs the chain control (docs/MFD_REPORT.md §14), CPU.
Per-user rank of the target in each run's 20-beam list (eval/<split>/save_seqrec_20.json, users aligned by all_users);
2,000 resamples of the same users, seed 0, 95% percentile CI, HR@10 and NDCG@10, valid and test.
Usage: python3 geo_chain_paired.py [D] -> mmgen_runs/geo/<D>/chain_paired.json
"""
import json
import sys

import numpy as np

RUNS = "/scratch/yw8866/mmgen_runs"


def ranks(base, split):
    s = json.load(open(f"{base}/eval/{split}/save_seqrec_20.json"))
    B = len(s["all_outputs"]) // len(s["all_users"])
    r = np.full(len(s["all_users"]), 10)
    for k, u in enumerate(s["all_users"]):
        tgt = s["all_targets"][k].replace(" ", "")
        top = [o.replace(" ", "") for o in s["all_outputs"][k * B:k * B + 10]]
        r[u] = top.index(tgt) if tgt in top else 10
    return r


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "Instruments"
    runs = {"control": f"{RUNS}/baselines/{D}/md122_a05_ep500_reassign/seed42"} | \
           {v: f"{RUNS}/geo/{D}/chain122_{v}/seed42" for v in ("A", "C", "AB1", "AB2")}
    res = {}
    for split in ("valid", "test"):
        R = {k: ranks(b, split) for k, b in runs.items()}
        boot = np.random.default_rng(0).integers(0, len(R["control"]), size=(2000, len(R["control"])))
        for v in ("A", "C", "AB1", "AB2"):
            out = {}
            for m, f in (("hr10", lambda r: (r < 10).astype(float)), ("ndcg10", lambda r: np.where(r < 10, 1 / np.log2(r + 2), 0.0))):
                d = f(R[v]) - f(R["control"])
                bs = d[boot].mean(1)
                out[m] = dict(a=f(R[v]).mean(), b=f(R["control"]).mean(), diff=d.mean(),
                              ci95=[np.percentile(bs, 2.5), np.percentile(bs, 97.5)])
            res.setdefault(f"{v} vs control", {})[split] = out
            h = out["hr10"]
            print(f"{split} {v:4s} vs control: HR@10 {h['a']:.5f} vs {h['b']:.5f} diff {h['diff']:+.4f} "
                  f"[{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  NDCG@10 diff {out['ndcg10']['diff']:+.4f} "
                  f"[{out['ndcg10']['ci95'][0]:+.4f}, {out['ndcg10']['ci95'][1]:+.4f}]")
    json.dump(res, open(f"{RUNS}/geo/{D}/chain_paired.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
