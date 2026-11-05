"""Tables and paired CIs for docs/MFD_REPORT.md §15 (disentangling AB2) and §16 (item-level encoder structure). CPU, env.

Every run's metrics come from its result files; per-user ranks from its saved lists (chain: eval/<split>/save_seqrec_20.json;
MFD: eval_kf20 / eval_ii ranked_<split>.pt, sweep apply_test_ranks / sweep_valid_ranks for (iii)). Paired bootstrap of X -
control of the same decoder and dataset: 2,000 resamples of the same users, seed 0, 95% percentile CI, HR@10 and NDCG@10.
Writes mmgen_runs/geo/sec15_16.json and prints the tables.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/scratch/yw8866/mmgen/sbatch/tools")
from geo_chain_paired import ranks as chain_ranks  # noqa: E402

R = "/scratch/yw8866/mmgen_runs"


def J(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def train(base):
    s = J(f"{base}/trainer_state.json")
    best = int(s["best_model_checkpoint"].split("-")[-1])
    be = next((h["epoch"] for h in s["log_history"] if "eval_loss" in h and h["step"] == best), None)
    return {"epochs": s["epoch"], "best_epoch": be, "steps": s["global_step"], "best_step": best, "best_valid_loss": s["best_metric"]}


def diag(base):
    g = J(f"{base}/geo_diag.json")
    if not g:
        return None
    t = g["splits"]["test"]
    return {"f_top1": t["f_top1"], "f_top20": t["f_top20"], "valid_f_top1": g["splits"]["valid"]["f_top1"],
            "valid_f_top20": g["splits"]["valid"]["f_top20"], "near_ratio": t["near_miss"]["ratio_to_mean_pair_dist"],
            "near_10nn": t["near_miss"]["top1_within_true_f_10nn"], "by_rank": t["hr10_by_f_rank"],
            "emb": {k: v["pearson"] for k, v in g["embedding_vs_codebook"].items()}}


def metr(ids, tg):
    m = ids[:, :10] == tg[:, None]
    return torch.where(m.any(1), m.int().argmax(1), torch.full_like(tg, 10)).numpy()


def mfd_ranks(base, sub, split):
    if sub == "iii":
        ch = J(f"{base}/sweep/chosen.json")
        if split == "test":
            return torch.load(f"{base}/sweep/apply_test_ranks.pt", weights_only=False)[ch["name"]].clamp(max=10).numpy()
        s = torch.load(f"{base}/sweep/sweep_valid_ranks.pt", weights_only=False)
        ci = next(i for i, c in enumerate(s["configs"]) if "mode" not in c and all(c[k] == ch[k] for k in ("k_f", "w_f", "w_t", "w_i")))
        return s["ranks"][ci].clamp(max=10).numpy()
    X = torch.load(f"{base}/{sub}/ranked_{split}.pt", weights_only=False)
    return metr(X["joint"][0], X["targets"])


def score(r):
    return {"hr1": float((r < 1).mean()), "hr5": float((r < 5).mean()), "hr10": float((r < 10).mean()),
            "ndcg10": float(np.where(r < 10, 1 / np.log2(r + 2.0), 0).mean())}


def paired(a, b):
    boot = np.random.default_rng(0).integers(0, len(a), size=(2000, len(a)))
    out = {}
    for n, f in (("hr10", lambda r: (r < 10).astype(float)), ("ndcg10", lambda r: np.where(r < 10, 1 / np.log2(r + 2.0), 0.))):
        d = f(a) - f(b)
        bs = d[boot].mean(1)
        out[n] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def f(p):
    return f"{p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"


def main():
    res = {"mfd": {}, "chain": {}, "struct": {}}
    runs = {"Arts": ("mfd122_mv", "mfd122_mv_geoAB2", "mfd122_mv_geoV1", "mfd122_mv_geoV2", "mfd122_mv_geoV3", "mfd122_mv_geoV4"),
            "Instruments": ("mfd122_mv", "mfd122_mv_geoAB2", "mfd122_mv_geoC", "mfd122_mv_geoV3"),
            "Games": ("mfd122_mv", "mfd122_mv_geoAB2", "mfd122_mv_geoV3")}
    for D, rs in runs.items():
        base0 = f"{R}/mfd/{D}/mfd122_mv/seed42"
        for run in rs:
            b = f"{R}/mfd/{D}/{run}/seed42"
            e = {"train": train(b), "diag": diag(b), "settings": {}}
            ch = J(f"{b}/sweep/chosen.json")
            e["chosen"] = ch["name"] if ch else None
            for sub, lab in (("eval_kf20", "i"), ("eval_ii", "ii"), ("iii", "iii")):
                if sub == "iii" and not ch:
                    continue
                st = {sp: score(mfd_ranks(b, sub, sp)) for sp in ("valid", "test")}
                if sub == "iii":
                    a = J(f"{b}/sweep/apply_test.json")["results"][ch["name"]]
                    st["ms"], st["gpu"], st["coverage"] = a["cost"]["ms_per_user"], a["cost"]["gpu"], a["coverage"]
                else:
                    c = J(f"{b}/{sub}/results_test.json")
                    st["ms"], st["gpu"], st["coverage"] = c["cost"]["ms_per_user"], c["cost"]["gpu"], c["coverage"]
                if run != "mfd122_mv":
                    ctl = "eval_ii" if sub == "iii" else sub               # (iii) vs the control's (ii), as §13
                    st["vs_control"] = {sp: paired(mfd_ranks(b, sub, sp), mfd_ranks(base0, ctl, sp)) for sp in ("valid", "test")}
                e["settings"][lab] = st
            res["mfd"].setdefault(D, {})[run] = e
    for D, vs in (("Arts", ("AB2", "V1", "V3")), ):
        c0 = f"{R}/baselines/{D}/md122_a05_ep500_reassign/seed42"
        res["chain"][D] = {"control": None}
        for name, b in [("control", c0)] + [(v, f"{R}/geo/{D}/chain122_{v}/seed42") for v in vs]:
            cost = J(f"{b}/eval/test/cost_seqrec.json")
            e = {"train": train(b), "diag": diag(b), "valid": score(chain_ranks(b, "valid")), "test": score(chain_ranks(b, "test")),
                 "passes": cost["decoder_forward_passes_per_user"], "ms": cost["ms_per_user"], "gpu": cost["gpu"]}
            if name != "control":
                e["vs_control"] = {sp: paired(chain_ranks(b, sp), chain_ranks(c0, sp)) for sp in ("valid", "test")}
            res["chain"][D][name] = e
    for D in ("Instruments", "Arts"):
        c0 = f"{R}/baselines/{D}/md122_a05_ep500_reassign/seed42"
        res["struct"][D] = {}
        for name, b in (("control", c0), ("E1", f"{R}/struct/{D}/chain122_E1/seed42"), ("E2", f"{R}/struct/{D}/chain122_E2/seed42")):
            cost = J(f"{b}/eval/test/cost_seqrec.json")
            e = {"train": train(b), "diag": diag(b), "valid": score(chain_ranks(b, "valid")), "test": score(chain_ranks(b, "test")),
                 "passes": cost["decoder_forward_passes_per_user"], "ms": cost["ms_per_user"], "gpu": cost["gpu"], "trunc_valid": {}}
            for h in (1, 5):
                m = J(f"{b}/eval/valid_his{h}/results_seqrec_20.json")["mean_results"]
                e["trunc_valid"][str(h)] = {"hr10": m["hit@10"], "ndcg10": m["ndcg@10"]}
            e["trunc_valid"]["20"] = {"hr10": e["valid"]["hr10"], "ndcg10": e["valid"]["ndcg10"]}
            if name != "control":
                e["vs_control"] = {sp: paired(chain_ranks(b, sp), chain_ranks(c0, sp)) for sp in ("valid", "test")}
            res["struct"][D][name] = e
    # variant vs variant (reading cases, §15) and the best run on VALIDATION vs C / plug-in merged
    import plugin_paired
    res["between"] = {}
    A = lambda run, sub="eval_ii": mfd_ranks(f"{R}/mfd/Arts/{run}/seed42", sub, "test")
    AV = lambda run, sub="eval_ii": mfd_ranks(f"{R}/mfd/Arts/{run}/seed42", sub, "valid")
    for a, b in (("mfd122_mv_geoAB2", "mfd122_mv_geoV1"), ("mfd122_mv_geoAB2", "mfd122_mv_geoV2"), ("mfd122_mv_geoAB2", "mfd122_mv_geoV3"),
                 ("mfd122_mv_geoV4", "mfd122_mv_geoAB2"), ("mfd122_mv_geoV2", "mfd122_mv_geoV1"), ("mfd122_mv_geoV1", "mfd122_mv_geoV3")):
        res["between"][f"Arts MFD (ii) {a[10:]} - {b[10:]}"] = {"test": paired(A(a), A(b)), "valid": paired(AV(a), AV(b))}
    C = lambda v, sp: chain_ranks(f"{R}/geo/Arts/chain122_{v}/seed42", sp)
    for a, b in (("AB2", "V1"), ("AB2", "V3"), ("V1", "V3")):
        res["between"][f"Arts chain {a} - {b}"] = {"test": paired(C(a, "test"), C(b, "test")), "valid": paired(C(a, "valid"), C(b, "valid"))}
    res["vs_reference"] = {}
    for D, run, sub in (("Instruments", "mfd122_mv_geoAB2", "eval_ii"), ("Arts", "mfd122_mv_geoV2", "iii"), ("Games", "mfd122_mv_geoAB2", "eval_ii")):
        x = {sp: mfd_ranks(f"{R}/mfd/{D}/{run}/seed42", sub, sp) for sp in ("valid", "test")}
        for B in ("mql", "plug122"):
            if D == "Games" and B == "plug122":
                continue
            r = {sp: np.minimum(plugin_paired.per_user(D, B, sp)[0]["merged"], 10) for sp in ("valid", "test")}
            res["vs_reference"][f"{D} {run[10:]} {sub} - {B} merged"] = {"run": score(x["test"]), "ref": score(r["test"]), "ref_valid": score(r["valid"]),
                                                                       "test": paired(x["test"], r["test"]), "valid": paired(x["valid"], r["valid"])}
    json.dump(res, open(f"{R}/geo/sec15_16.json", "w"), indent=1)
    for k, v in list(res["between"].items()) + list(res["vs_reference"].items()):
        print(f"{k}: test HR@10 {f(v['test']['hr10'])} NDCG@10 {f(v['test']['ndcg10'])} | valid HR@10 {f(v['valid']['hr10'])}" + (f" | run {v['run']['hr10']:.4f}/{v['run']['ndcg10']:.4f} ref {v['ref']['hr10']:.4f}/{v['ref']['ndcg10']:.4f}" if "run" in v else ""))
    for D, rr in res["mfd"].items():
        print("==", D, "MFD")
        for run, e in rr.items():
            d, t = e["diag"], e["train"]
            print(f" {run} chosen {e['chosen']} | ep {t['epochs']:.0f} best {t['best_epoch']} steps {t['steps']} vl {t['best_valid_loss']:.4f} | f {d['f_top1']:.4f}/{d['f_top20']:.4f} near {d['near_ratio']:.3f}/{d['near_10nn']:.3f} | emb s0 {d['emb']['s0']:.2f}")
            for lab, st in e["settings"].items():
                v, te = st["valid"], st["test"]
                x = st.get("vs_control")
                print(f"   ({lab}) valid {v['hr1']:.4f} {v['hr10']:.4f} {v['ndcg10']:.4f} | test {te['hr1']:.4f} {te['hr5']:.4f} {te['hr10']:.4f} {te['ndcg10']:.4f} | cov {st['coverage']:.3f} ms {st['ms']:.1f} {st['gpu'][-4:]}"
                      + (f" | test d {f(x['test']['hr10'])} n {f(x['test']['ndcg10'])} | valid d {f(x['valid']['hr10'])}" if x else ""))
            print("   by rank", " ".join(f"{k}:{v['users']}/{(v['hr10'] or 0):.3f}" for k, v in d["by_rank"].items()))
    for sec in ("chain", "struct"):
        for D, rr in res[sec].items():
            print("==", D, sec)
            for n, e in rr.items():
                if not e:
                    continue
                d, t = e["diag"], e["train"]
                x = e.get("vs_control")
                print(f" {n} valid {e['valid']['hr1']:.4f} {e['valid']['hr10']:.4f} {e['valid']['ndcg10']:.4f} | test {e['test']['hr1']:.4f} {e['test']['hr5']:.4f} {e['test']['hr10']:.4f} {e['test']['ndcg10']:.4f} | ep {t['epochs']:.0f} best {t['best_epoch']} steps {t['steps']} | f {d['f_top1']:.4f}/{d['f_top20']:.4f} near {d['near_ratio']:.3f}/{d['near_10nn']:.3f} | p {e['passes']} ms {e['ms']:.1f} {e['gpu'][-4:]}"
                      + (f" | valid d {f(x['valid']['hr10'])} test d {f(x['test']['hr10'])} n {f(x['test']['ndcg10'])}" if x else ""))
                if "trunc_valid" in e:
                    print("     trunc valid HR@10", {k: round(v["hr10"], 4) for k, v in e["trunc_valid"].items()})
    for D in ("Instruments", "Arts", "Games"):
        for b in ("mql", "plug122"):
            r = J(f"{R}/baselines/{D}/{b}/seed42/eval/test/results_ensemble_20.json")
            v = J(f"{R}/baselines/{D}/{b}/seed42/eval/valid/results_ensemble_20.json")
            print(D, b, "merged test", r and (round(r['hit@10'] if 'hit@10' in r else r['mean_results']['hit@10'], 4)), r and r.get("ndcg@10"), "valid", v and v.get("hit@10"))


if __name__ == "__main__":
    main()
