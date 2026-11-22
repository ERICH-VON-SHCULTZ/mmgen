"""mmgen addition (docs/MFD_REPORT.md §23): round-23 report (CPU, no model). Reads mmgen_runs/timing/r23/<D>/{prep,time_*,profile_*}.json.
 1 end-to-end: ms/user (mean +- std of 3 passes), users/s, peak GPU memory per method x batch x dataset; speedup = ms(C) / ms(method)
   and ms(C-fast) / ms(method) at the same dataset and batch size; HR@10 check (batch-64 lists vs the reported number); job anchors
 2 stage breakdown (profile, Arts, batch 64 and 1): ms/user and % of the instrumented total; profiler device (kernel) time
 3 hardware-independent counts and FLOPs per user (all datasets)
 4 accuracy-latency figure (x = batch-64 ms/user, y = test HR@10, 3-seed mean where available) for Arts and Instruments (+ Games):
   timing/r23/r23_acc_latency_<D>.png + timing/r23/r23_acc_latency.csv
 5 training cost from existing logs (optimizer steps and wall-clock to the best checkpoint, GPU-hours, GPU type)
Out: timing/r23/r23_report.json + printed tables.  Usage: python r23_report.py
"""
import glob
import json
import os
import re

import numpy as np

R = "/scratch/yw8866/mmgen_runs"
T = f"{R}/timing/r23"
DS = ("Arts", "Instruments", "Games")
BASE = {"C": "mql", "D": "macrecpaper", "TIGER": "tiger_text", "chain": "md122_a05_ep500_reassign"}
LABEL = {"P1a:20": "P1a K_f=20", "P1a:50": "P1a K_f=50", "P1a:100": "P1a K_f=100", "A01:50": "A01 K_f=50", "C": "C merged",
         "Cfast": "C-fast", "D": "D merged", "TIGER": "TIGER-text", "chain": "MD chain 1+2+2"}
OPT = {"Copt": "C-opt (forwards only)", "Coptk": "C-opt + top-k", "Dopt": "D-opt", "TIGERopt": "TIGER-opt", "chainopt": "chain-opt"}
ORDER = list(LABEL)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def e2e():
    rows, anchors, envs = {}, {}, {}
    for D in DS:
        for f in sorted(glob.glob(f"{T}/{D}/time_*.json")):
            d = load(f)
            tag = os.path.basename(f)[5:-5]
            if tag.startswith("copt"):                        # model-forwards-only lower bounds: separate same-job table (copt())
                continue
            envs[f"{D}/{tag}"] = {k: d["env"].get(k) for k in ("host", "gpu", "nvidia_smi", "torch", "cuda_torch", "cpu", "job")}
            if "anchor" in d:
                anchors[f"{D}/{tag}"] = {k: v["ms_per_user"] for k, v in d["anchor"].items()}
            for k, r in d["runs"].items():
                key = (D, r["method"], r["batch"])
                if key in rows:                               # repeats split over jobs (Games C / D / C-fast): pool the passes
                    q = rows[key]
                    q["seconds"] = q["seconds"] + r["seconds"]; q["ms_per_user"] = q["ms_per_user"] + r["ms_per_user"]
                    q["ms_mean"] = float(np.mean(q["ms_per_user"])); q["ms_std"] = float(np.std(q["ms_per_user"], ddof=1))
                    q["users_per_s"] = float(q["users"] / np.mean(q["seconds"]))
                    q["peak_alloc_gb"] = max(q["peak_alloc_gb"], r["peak_alloc_gb"])
                    q["hr10_match"] = q.get("hr10_match") and r.get("hr10_match") and q["hr10"] == r["hr10"]
                    q["jobs"] = q.get("jobs", 1) + 1
                else:
                    rows[key] = dict(r)
    out = {}
    for (D, m, b), r in rows.items():
        c, cf = rows.get((D, "C", b)), rows.get((D, "Cfast", b))
        out[f"{D} {m} {b}"] = dict(dataset=D, method=m, batch=b, batch_used=r["batch_used"], ms=r["ms_mean"], ms_std=r["ms_std"],
                                   users_per_s=r["users_per_s"], peak_gb=r["peak_alloc_gb"], users=r["users"],
                                   speedup_vs_C=c["ms_mean"] / r["ms_mean"] if c else None,
                                   speedup_vs_Cfast=cf["ms_mean"] / r["ms_mean"] if cf else None,
                                   hr10=r.get("hr10"), hr10_reported=r.get("hr10_reported"), hr10_match=r.get("hr10_match"),
                                   passes=len(r["ms_per_user"]), jobs=r.get("jobs", 1))
    return out, anchors, envs


def copt():
    """Same-job comparison: P1a K_f 50 vs the forwards-only lower bounds (time_copt.json per dataset), plus the speedups of the
    main e2e table's C / C-fast at the same batch for reference."""
    out = {}
    for D in DS:
        d = load(f"{T}/{D}/time_copt.json")
        if not d:
            continue
        runs = {(r["method"], r["batch"]): r for r in d["runs"].values()}
        for b in (1, 64):
            p = runs.get(("P1a:50", b))
            for m in OPT:
                r = runs.get((m, b))
                if not r:
                    continue
                out[f"{D} {m} {b}"] = dict(dataset=D, method=m, batch=b, ms=r["ms_mean"], ms_std=r["ms_std"], peak_gb=r["peak_alloc_gb"],
                                           steps=r.get("replay_steps_per_list"), p1a50_ms=p["ms_mean"] if p else None,
                                           p1a50_speedup_vs_opt=(r["ms_mean"] / p["ms_mean"]) if p else None)
        out[f"{D} flop_check"] = d.get("flop_check")
        out[f"{D} anchor"] = d.get("anchor")
    return out


def three_seed(D):
    """test HR@10 per method: mean over the seeds available (P1a from r23_prep's exact ranks)."""
    out = {}
    prep = load(f"{T}/{D}/prep.json") or {}
    ref = prep.get("p1a_ref", {})
    for K in (20, 50, 100):
        v = [ref[f"P1a:{K}:s{S}"]["hr10"] for S in (42, 43, 44) if f"P1a:{K}:s{S}" in ref]
        out[f"P1a:{K}"] = (float(np.mean(v)), len(v)) if v else None
    a = ref.get("A01:50:s42")
    out["A01:50"] = (a["hr10"], 1) if a else None
    for m, b in BASE.items():
        v = []
        for S in (42, 43, 44):
            e = f"{R}/baselines/{D}/{b}/seed{S}/eval/test"
            f = f"{e}/results_ensemble_20.json" if m in ("C", "D") else f"{e}/results_seqrec_20.json"
            d = load(f)
            if d:
                v.append(d.get("mean_results", d)["hit@10"])
        out[m] = (float(np.mean(v)), len(v)) if v else None
    out["Cfast"] = out["C"]
    return out


def figure(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for D in DS:
        hs = three_seed(D)
        pts = []
        for m in ORDER:
            r = tab.get(f"{D} {m} 64")
            if r and hs.get(m):
                pts.append((m, r["ms"], r["ms_std"], hs[m][0], hs[m][1]))
                rows.append(dict(dataset=D, method=m, label=LABEL[m], ms_per_user_b64=r["ms"], ms_std=r["ms_std"],
                                 test_hr10=hs[m][0], n_seeds=hs[m][1], hr10_seed42_batch64_lists=r["hr10"]))
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        p1 = [p for p in pts if p[0].startswith("P1a")]
        if p1:
            ax.plot([p[1] for p in p1], [p[3] for p in p1], "-o", color="#1f6fb4", label="P1a (K_f = 20 / 50 / 100)")
            for p in p1:
                ax.annotate(f"K_f={p[0].split(':')[1]}", (p[1], p[3]), textcoords="offset points", xytext=(5, -11), fontsize=8)
        mk = {"A01:50": ("#1f6fb4", "D"), "C": ("#c0392b", "s"), "Cfast": ("#e67e22", "s"), "D": ("#8e44ad", "^"),
              "TIGER": ("#7f8c8d", "v"), "chain": ("#27ae60", "P")}
        for p in pts:
            if p[0] in mk:
                ax.errorbar(p[1], p[3], xerr=p[2], fmt=mk[p[0]][1], color=mk[p[0]][0], label=LABEL[p[0]], ms=7, capsize=2)
        ax.set_xscale("log")
        ax.set_xlabel("ms / user (batch 64, H200, log scale)")
        ax.set_ylabel("test HR@10 (3-seed mean where available)")
        ax.set_title(f"{D}: accuracy vs latency")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{T}/r23_acc_latency_{D}.png", dpi=160)
        plt.close(fig)
    with open(f"{T}/r23_acc_latency.csv", "w") as f:
        keys = ["dataset", "method", "label", "ms_per_user_b64", "ms_std", "test_hr10", "n_seeds", "hr10_seed42_batch64_lists"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    return rows


def pack_count(job):
    """number of runs trained concurrently in the pack job (from its log), None if unknown."""
    for f in glob.glob(f"{R}/logs/*_{job}.out"):
        txt = open(f, errors="ignore").read()
        m = re.search(r"RUNS=([^\n]+)", txt)
        if m:
            return len(m.group(1).split())
        n = len(set(re.findall(r"== train (\S+ \S+ seed \d+)", txt)))
        if n:
            return n
    return None


def pack_of(token):
    """(job, runs in the pack) of the pack_train job whose RUNS= list contains D:B:S (runs without run_config.json)."""
    for f in sorted(glob.glob(f"{R}/logs/mmgen_pack_train_*.out")):
        with open(f, errors="ignore") as fh:
            head = fh.readline()
        m = re.search(r"RUNS=(.*?)\s+commit", head)
        if m and token in m.group(1).split():
            return re.search(r"job (\d+)", head).group(1), len(m.group(1).split())
    return None, None


def training_cost():
    out = {}
    for D in DS:
        runs = {"P1a": f"{R}/mfd/{D}/mfd122_mv_geoP1a/seed42", **{m: f"{R}/baselines/{D}/{b}/seed42" for m, b in BASE.items()}}
        for m, ck in runs.items():
            ts = load(f"{ck}/train_stats.json")
            rc = load(f"{ck}/run_config.json") or {}
            if not ts:
                continue
            best = int(re.search(r"checkpoint-(\d+)", ts["best_model_checkpoint"]).group(1))
            wall = ts["train_wall_seconds"]
            job = rc.get("job_id")
            npack = pack_count(job) if job else None
            if not job:
                job, npack = pack_of(f"{D}:{BASE[m]}:42")
            out[f"{D} {m}"] = {"optimizer_steps_total": ts["global_steps"], "optimizer_steps_to_best": best,
                               "epochs_run": ts["epochs_run"], "best_epoch": ts["best_epoch"], "wall_h_total": wall / 3600,
                               "wall_h_to_best_est": wall / 3600 * best / ts["global_steps"], "gpu": ts.get("gpu"),
                               "train_job": job, "runs_packed_on_the_gpu": npack,
                               "effective_batch": rc.get("effective_batch"), "examples_per_epoch": ts.get("train_examples_per_epoch")}
    return out


def main():
    tab, anchors, envs = e2e()
    res = {"e2e": tab, "anchors": anchors, "envs": envs, "prep": {}, "profile": {}, "train_cost": training_cost()}
    for D in DS:
        p = load(f"{T}/{D}/prep.json")
        if p:
            res["prep"][D] = {"all_pass": p.get("all_pass"), "cfast_match_rate_top10": p.get("cfast_match_rate_top10"),
                              "checks": {k: v["ok"] for k, v in p["checks"].items()}}
        for f in glob.glob(f"{T}/{D}/profile_*.json"):
            for m, v in load(f)["methods"].items():
                res["profile"][f"{D} {m}"] = v
    res["figure_rows"] = figure(tab)
    json.dump(res, open(f"{T}/r23_report.json", "w"), indent=1)

    print("== 1 end-to-end (ms/user mean +- std over 3 passes; speedup vs C / C-fast at the same batch) ==")
    for D in DS:
        for b in (1, 64, 256):
            for m in ORDER:
                r = tab.get(f"{D} {m} {b}")
                if not r:
                    continue
                sc = f"{r['speedup_vs_C']:.1f}x" if r["speedup_vs_C"] else "-"
                sf = f"{r['speedup_vs_Cfast']:.1f}x" if r["speedup_vs_Cfast"] else "-"
                chk = "" if b == 1 else f" HR@10 {r['hr10']:.4f} vs {r['hr10_reported']} match {r['hr10_match']}"
                print(f"{D:11s} b{b:<3d}{'(' + str(r['batch_used']) + ')' if r['batch_used'] != b else '':6s} {LABEL[m]:15s} "
                      f"{r['ms']:8.3f} +- {r['ms_std']:.3f} ms/user {r['users_per_s']:9.0f} users/s  vs C {sc:>7s} vs C-fast {sf:>7s} "
                      f"peak {r['peak_gb']:.2f} GB{chk}")
    res["copt"] = co = copt()
    print("\n== 1b model-forwards-only lower bounds (same job as P1a K_f 50; speedup = opt ms / P1a ms, < 1 = P1a slower) ==")
    for k, r in co.items():
        if k.endswith(("flop_check", "anchor")):
            print(f"  {k}: {json.dumps(r)[:600]}")
            continue
        print(f"  {r['dataset']:11s} b{r['batch']:<3d} {OPT[r['method']]:22s} {r['ms']:8.3f} +- {r['ms_std']:.3f} ms/user  P1a K_f=50 "
              f"{r['p1a50_ms']:.3f}  P1a speedup {r['p1a50_speedup_vs_opt']:.2f}x  steps/list {r['steps']}")
    json.dump(res, open(f"{T}/r23_report.json", "w"), indent=1)
    print("\n== anchors (ms/user; same measurement in every job) ==")
    for k, v in anchors.items():
        print(f"  {k:30s} {json.dumps({a: round(b, 3) for a, b in v.items()})}  {envs[k]['host']}")
    print("\n== 2 stage breakdown ==")
    for k, v in res["profile"].items():
        for b, s in v.get("stages", {}).items():
            tot = s["instrumented_total_ms_per_user"]
            print(f"-- {k} batch {b}: instrumented total {tot:.3f} ms/user")
            for st, ms in s["stages_ms_per_user"].items():
                dv = s.get("device_kernel_ms_per_user", {}).get(st)
                print(f"   {st:45s} {ms:8.3f} ms  {100 * ms / tot:5.1f}%" + (f"   device {dv:.3f} ms" if dv is not None else ""))
    print("\n== 3a hardware-independent summary per user (sequential decoder forwards, encoder / memory length, GFLOP measured) ==")
    for k, v in res["profile"].items():
        c, f = v["counts"], v["flops"]
        if "lists" in c:
            steps, mem = c["sequential_decoder_steps_total"], [round(c[t]["cross_attention_memory_len"], 1) for t in c if isinstance(c[t], dict)]
            rows = "20 beams x 1 new token"
        else:
            steps, mem, rows = c["sequential_decoder_steps"], [round(c["cross_attention_memory_len"], 1)], c["decoder_rows_per_step"]
        print(f"  {k:24s} seq. decoder forwards {steps}  memory len {mem}  rows/step {rows}  "
              f"GFLOP enc {f['encoder_gflop_per_user']:.3f} dec {f['decoder_incl_lm_head_gflop_per_user']:.3f} (lm_head {f['lm_head_gflop_per_user']:.3f})")
    print("\n== 3 counts / FLOPs per user ==")
    for k, v in res["profile"].items():
        print(f"  {k:22s} {json.dumps(v['counts'])[:400]}")
        print(f"  {'':22s} FLOPs {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v['flops'].items() if a != 'method'})}")
    print("\n== 5 training cost (seed 42) ==")
    for k, v in res["train_cost"].items():
        print(f"  {k:22s} {json.dumps(v)}")
    print("\nprep:", json.dumps(res["prep"]))


if __name__ == "__main__":
    main()
