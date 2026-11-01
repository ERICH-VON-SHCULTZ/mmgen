"""Paired comparison of the plug-in runs against C (MQL4GRec) seed 42 (docs/PLUGIN_REPORT.md §4). CPU, no model.

Per-user results are rebuilt from the saved beams with MACRec's own code, exactly as reported:
  single lists: evaluate.get_topk_results(..., all_items) (= test_ddp_save with --filter_items),
  merged list : ensemble.get_sort_results + get_topk_results_ensemble (= ensemble.py),
and checked against results_{seqrec,seqimage,ensemble}_20.json (assert). Users are aligned by dataset index
(all_users). Paired bootstrap: 2,000 resamples of the same users, seed 0, 95% percentile CI of plug-in - C for
HR@10 and NDCG@10. Lists compared: merged vs merged, best single list vs best single list (each run's best chosen
on VALIDATION HR@10, tie NDCG@10), text vs text, image vs image.
Usage: python plugin_paired.py [D:B ...]  (run inside the env with MACREC=<macrec dir>) -> mmgen_runs/plugin/paired.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get("MACREC", "/scratch/yw8866/mmgen/macrec"))
from ensemble import get_sort_results, get_topk_results_ensemble  # noqa: E402
from evaluate import get_metrics_results, get_topk_results  # noqa: E402

RD = "/scratch/yw8866/mmgen_runs"
IDX = {"mql": (".index_lemb_mql.json", ".index_vitemb_mql.json")}
METRICS = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
NB = 2000


def index_files(B):
    return IDX.get(B, (f".index_{B}_text.json", f".index_{B}_image.json"))


def first_hit(rows):
    return np.array([r.index(1) if 1 in r[:20] else 99 for r in rows])


def per_user(D, B, split):
    """{list: rank array indexed by dataset user}, plus recomputed metrics."""
    ev = f"{RD}/baselines/{D}/{B}/seed42/eval/{split}"
    out, met = {}, {}
    lookup = {}
    for task, fidx in (("seqrec", 0), ("seqimage", 1)):
        sv = json.load(open(f"{ev}/save_{task}_20.json"))
        idx = json.load(open(f"{RD}/data/{D}/{D}{index_files(B)[fidx]}"))
        all_items = {"".join(v) for v in idx.values()}
        rows = get_topk_results(list(sv["all_outputs"]), list(sv["all_scores"]), sv["all_targets"], 20, all_items=all_items)
        m = {k: v / len(rows) for k, v in get_metrics_results(rows, METRICS).items()}
        ref = json.load(open(f"{ev}/results_{task}_20.json"))["mean_results"]
        for k in METRICS:
            assert abs(m[k] - ref[k]) < 1e-9, (D, B, split, task, k, m[k], ref[k])
        r = np.full(len(rows), 99)
        r[np.array(sv["all_users"])] = first_hit(rows)
        out[task], met[task] = r, m
        i2s = defaultdict(list)
        for k, v in idx.items():
            i2s["".join(v)].append(int(k))
        lookup[task] = (sv, i2s)
    (ts, ti), (is_, ii) = lookup["seqrec"], lookup["seqimage"]
    tinfo = get_sort_results(ts["all_outputs"], ts["all_scores"], [ti[t] for t in ts["all_targets"]], ts["all_users"], 20, ti)
    iinfo = get_sort_results(is_["all_outputs"], is_["all_scores"], [ii[t] for t in is_["all_targets"]], is_["all_users"], 20, ii)
    rows = get_topk_results_ensemble(tinfo, iinfo)
    m = {k: v / len(rows) for k, v in get_metrics_results(rows, METRICS).items()}
    ref = json.load(open(f"{ev}/results_ensemble_20.json"))
    for k in METRICS:
        assert abs(m[k] - ref[k]) < 1e-9, (D, B, split, "merged", k, m[k], ref[k])
    r = np.full(len(rows), 99)
    r[np.array([x[0] for x in tinfo])] = np.array([x.index(1) if 1 in x[:10] else 99 for x in rows])
    out["merged"], met["merged"] = r, m
    return out, met


def ndcg(r):
    return np.where(r < 10, 1.0 / np.log2(r + 2.0), 0.0)


def paired(a, b, boot):
    res = {}
    for name, f in (("hr10", lambda r: (r < 10).astype(float)), ("ndcg10", ndcg)):
        d = f(a) - f(b)
        bs = d[boot].mean(1)
        res[name] = {"plugin": float(f(a).mean()), "C": float(f(b).mean()), "diff": float(d.mean()),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["won"], res["lost"] = int(((a < 10) & (b >= 10)).sum()), int(((a >= 10) & (b < 10)).sum())
    return res


def best_single(met_valid):
    return max(("seqrec", "seqimage"), key=lambda t: (met_valid[t]["hit@10"], met_valid[t]["ndcg@10"]))


def main():
    runs = sys.argv[1:] or ["Instruments:plug122", "Instruments:plug122s", "Instruments:plug133", "Instruments:plug133s",
                            "Arts:plug122", "Arts:plug133"]
    res = {"C_seed_spread": {}, "runs": {}}
    for D in sorted({r.split(":")[0] for r in runs}):
        seeds = {}
        for s in (42, 43, 44):
            f = f"{RD}/baselines/{D}/mql/seed{s}/run_summary.json"
            if os.path.exists(f):
                js = json.load(open(f))
                seeds[s] = {t: js["test"][t]["metrics"] for t in ("seqrec", "seqimage", "merged")}
        res["C_seed_spread"][D] = {"seeds": seeds, "summary": {
            t: {k: {"mean": float(np.mean([seeds[s][t][k] for s in seeds])), "std": float(np.std([seeds[s][t][k] for s in seeds], ddof=1)),
                    "min": float(min(seeds[s][t][k] for s in seeds)), "max": float(max(seeds[s][t][k] for s in seeds))}
                for k in ("hit@10", "ndcg@10")} for t in ("seqrec", "seqimage", "merged")}}
    cache = {}
    for r in runs + [f"{D}:mql" for D in sorted({x.split(':')[0] for x in runs})]:
        D, B = r.split(":")
        for split in ("valid", "test"):
            cache[(D, B, split)] = per_user(D, B, split)
    for r in runs:
        D, B = r.split(":")
        res["runs"][r] = {"metrics": {}, "paired": {}}
        bp, bc = best_single(cache[(D, B, "valid")][1]), best_single(cache[(D, "mql", "valid")][1])
        res["runs"][r]["best_single_valid"], res["runs"][r]["C_best_single_valid"] = bp, bc
        for split in ("test", "valid"):
            up, mp = cache[(D, B, split)]
            uc, _ = cache[(D, "mql", split)]
            boot = np.random.default_rng(0).integers(0, len(up["merged"]), size=(NB, len(up["merged"])))
            res["runs"][r]["metrics"][split] = mp
            res["runs"][r]["paired"][split] = {
                "merged": paired(up["merged"], uc["merged"], boot),
                "best_single": paired(up[bp], uc[bc], boot),
                "text": paired(up["seqrec"], uc["seqrec"], boot),
                "image": paired(up["seqimage"], uc["seqimage"], boot)}
            for k, v in res["runs"][r]["paired"][split].items():
                print(f"{r} {split} {k:11s}: HR@10 {v['hr10']['plugin']:.4f} vs C {v['hr10']['C']:.4f} "
                      f"diff {v['hr10']['diff']:+.4f} [{v['hr10']['ci95'][0]:+.4f}, {v['hr10']['ci95'][1]:+.4f}]  "
                      f"NDCG@10 diff {v['ndcg10']['diff']:+.4f} [{v['ndcg10']['ci95'][0]:+.4f}, {v['ndcg10']['ci95'][1]:+.4f}]  "
                      f"won/lost {v['won']}/{v['lost']}", flush=True)
    json.dump(res, open(f"{RD}/plugin/paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
