"""Tables for MFD Parts A-C (docs/MFD_REPORT.md §7-§10), seed 42, from saved result files only.
Part B settings are chosen on VALIDATION (mfd_sweep.py chosen.json); C/D "best single list" on validation HR@10.
Usage: python3 mfd_parts_table.py  -> prints markdown, writes mmgen_runs/mfd/parts_tables.json
"""
import json
import os

RUNS = "/scratch/yw8866/mmgen_runs"
MFD = f"{RUNS}/mfd"
CHAIN = {"122": "md122_a05_ep500_reassign", "133": "md133_a06_ep500_reassign"}
OUT = {}


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def hn(m):
    return f"{m['hit@10']:.4f} / {m['ndcg@10']:.4f}"


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def mfd_eval(D, run):
    e = f"{MFD}/{D}/{run}/seed42/eval_kf20"
    return J(f"{e}/results_test.json"), J(f"{e}/results_valid.json")


def mfd_applied(D, run):
    """Part B setting through the real path: own sweep for the swept runs, else sweep_applied (fixed config)."""
    for sub in ("sweep", "sweep_applied"):
        t = J(f"{MFD}/{D}/{run}/seed42/{sub}/apply_test.json")
        if t:
            v = J(f"{MFD}/{D}/{run}/seed42/{sub}/apply_valid.json")
            if v is None:                                                # own sweep: valid numbers from the sweep
                s = J(f"{MFD}/{D}/{run}/seed42/{sub}/sweep_valid.json")
                v = {"results": {x["name"]: x for x in s["settings"] if "mode" not in x}}
            return t, v
    return None, None


def chain(D, b):
    s = J(f"{RUNS}/baselines/{D}/{b}/seed42/run_summary.json")
    return s


def chain_row(name, D, b):
    s = chain(D, b)
    c = s["test"]["seqrec"]["cost"]
    return [name, hn(s["test"]["seqrec"]["metrics"]), hn(s["valid"]["seqrec"]["metrics"]),
            int(c["decoder_forward_passes_per_user"]), f"{c['ms_per_user']:.1f} ({c['gpu'].replace('NVIDIA ', '')})"]


def cd_rows(D):
    rows = []
    for b, label in (("mql", "C MQL4GRec"), ("macrecpaper", "D MACRec")):
        s = chain(D, b)
        best = max(("seqrec", "seqimage"), key=lambda t: (s["valid"][t]["metrics"]["hit@10"], s["valid"][t]["metrics"]["ndcg@10"]))
        c = s["test"][best]["cost"]
        g = c["gpu"].replace("NVIDIA ", "")
        rows.append([f"{label}, best single list ({'text' if best == 'seqrec' else 'image'})",
                     hn(s["test"][best]["metrics"]), hn(s["valid"][best]["metrics"]), 5, f"{c['ms_per_user']:.1f} ({g})"])
        ms2 = sum(s["test"][t]["cost"]["ms_per_user"] for t in ("seqrec", "seqimage"))
        rows.append([f"{label}, merged", hn(s["test"]["merged"]["metrics"]), hn(s["valid"]["merged"]["metrics"]),
                     "2 × 5", f"{ms2:.1f} ({g})"])
    return rows


def part_a():
    print("### Part A: Instruments 1+3+3, K_f = 20\n")
    rows = []
    for run, cfg in (("mfd133_stage1", "1+3+3"), ("mfd122_stage1", "1+2+2")):
        t, v = mfd_eval("Instruments", run)
        ms = f"{t['cost']['ms_per_user']:.1f} (H200)"
        for mode in ("text", "image", "merge", "joint"):
            rows.append([f"MFD {cfg} ({mode})", hn(t["modes"][mode]), hn(v["modes"][mode]),
                         t["cost"]["decoder_passes_per_user"], ms])
        rows.append(chain_row(f"MD {cfg} chain", "Instruments", CHAIN[cfg.replace("+", "")]))
    table(["method", "test HR@10 / NDCG@10", "valid HR@10 / NDCG@10", "decoder passes / user", "ms / user"], rows)
    OUT["part_a"] = rows


def part_b(run):
    s = J(f"{MFD}/Instruments/{run}/seed42/sweep/sweep_valid.json")
    if s is None:
        print(f"(sweep for {run} not finished)\n")
        return
    cfg = "1+3+3" if "133" in run else "1+2+2"
    print(f"### Part B: Instruments {cfg} — validation sweep ({s['users']} users)\n")
    joint = [x for x in s["settings"] if "mode" not in x]
    rows = []
    for K in s["k_f_values"]:
        tm = s["timing"][str(K)]
        w1 = next(x for x in joint if x["k_f"] == K and (x["w_f"], x["w_t"], x["w_i"]) == (1.0, 1.0, 1.0))
        bk = max((x for x in joint if x["k_f"] == K), key=lambda x: (round(x["hit@10"], 10), round(x["ndcg@10"], 10)))
        for lab, x in (("1 / 1 / 1", w1), (f"{bk['w_f']} / {bk['w_t']} / {bk['w_i']} (best at this K_f)", bk)):
            rows.append([K, lab, hn(x), f"{x['coverage']:.3f}", f"{x['hit10_given_coverage']:.3f}",
                         f"{tm['ms_per_user']:.1f}", f"{tm['peak_gpu_mem_gb']:.1f}"])
        assert tm["real_vs_sweep_rank_mismatch_users"] == 0, (run, K)
    table(["K_f", "w_f / w_t / w_i", "valid HR@10 / NDCG@10", "coverage", "HR@10 among covered", "ms / user (H200)",
           "peak GPU mem (GB)"], rows)
    print("Text-only / image-only (w = 1) per K_f, valid HR@10: " + "; ".join(
        f"K_f={K}: text {next(x for x in s['settings'] if x.get('mode') == 'text' and x['k_f'] == K)['hit@10']:.4f}, "
        f"image {next(x for x in s['settings'] if x.get('mode') == 'image' and x['k_f'] == K)['hit@10']:.4f}"
        for K in s["k_f_values"]) + "\n")
    ch = s["chosen"]
    a = J(f"{MFD}/Instruments/{run}/seed42/sweep/apply_test.json")
    print(f"Chosen on validation: **{ch['name']}** (valid {ch['valid_hit@10']:.4f} / {ch['valid_ndcg@10']:.4f}).\n")
    # f-rank buckets: stage-1 setting and the chosen one, valid (sweep) and test (apply)
    names = {"stage1_reference": "K_f=20, w=1", ch["name"]: f"chosen {ch['name']}"}
    ref = next(x for x in joint if x["k_f"] == 20 and (x["w_f"], x["w_t"], x["w_i"]) == (1.0, 1.0, 1.0))
    chv = next(x for x in joint if x["name"] == ch["name"])
    buckets = list(ref["hit10_by_f_rank"])
    rows = []
    for lab, src in (("valid, K_f=20, w=1", ref), (f"valid, chosen", chv),
                     ("test, K_f=20, w=1", a["results"]["stage1_reference"] if a else None),
                     ("test, chosen", a["results"][ch["name"]] if a else None)):
        if src is None:
            continue
        rows.append([lab] + [f"{src['hit10_by_f_rank'][b]['hit10']:.3f} ({src['hit10_by_f_rank'][b]['users']})"
                             if src["hit10_by_f_rank"][b]["hit10"] is not None else "–" for b in buckets])
    print("HR@10 by the target f's rank (users in brackets):\n")
    table(["setting"] + buckets, rows)
    if a:
        rows = []
        for n in ("stage1_reference", ch["name"]):
            r = a["results"][n]
            rows.append([names[n], hn(r), f"{r['coverage']:.3f}", f"{r['hit10_given_coverage']:.3f}",
                         r["cost"]["decoder_passes_per_user"], f"{r['cost']['ms_per_user']:.1f}",
                         f"{r['cost']['peak_gpu_mem_gb']:.1f}"])
        print("Test (the chosen setting evaluated once):\n")
        table(["MFD setting", "test HR@10 / NDCG@10", "coverage", "HR@10 among covered", "passes", "ms / user (H200)",
               "peak GPU mem (GB)"], rows)
    OUT[f"part_b_{run}"] = dict(chosen=ch, test=a["results"] if a else None)


def part_c(D):
    print(f"### Part C: {D}, 1+2+2\n")
    rows = [chain_row("MD chain", D, CHAIN["122"]), chain_row("MD chain + multi-view aug", D, "mdmv122")]
    for run, lab in (("mfd122_stage1", "MFD"), ("mfd122_mv", "MFD + multi-view aug")):
        t, v = mfd_eval(D, run)
        if t is None:
            rows.append([f"{lab} (not evaluated yet)", "", "", "", ""]); continue
        ms = f"{t['cost']['ms_per_user']:.1f} (H200)"
        rows.append([f"{lab} joint, K_f=20, w=1", hn(t["modes"]["joint"]), hn(v["modes"]["joint"]), 3, ms])
        rows.append([f"{lab} merge, K_f=20", hn(t["modes"]["merge"]), hn(v["modes"]["merge"]), 3, ms])
        at, av = mfd_applied(D, run)
        if at:
            ch = at["chosen"]
            r, rv = at["results"][ch["name"]], av["results"][ch["name"]]
            rows.append([f"{lab} joint, Part B setting ({ch['name']})", hn(r), hn(rv), 3,
                         f"{r['cost']['ms_per_user']:.1f} (H200)"])
    rows += cd_rows(D)
    table(["method", "test HR@10 / NDCG@10", "valid HR@10 / NDCG@10", "decoder passes / user", "ms / user"], rows)
    OUT[f"part_c_{D}"] = rows
    # f accuracy +- augmentation
    rows = []
    for run, lab in (("mfd122_stage1", "MFD"), ("mfd122_mv", "MFD + aug")):
        t, v = mfd_eval(D, run)
        if t:
            rows.append([lab, f"{t['f_top1_acc']:.4f}", f"{t['f_top20_acc']:.4f}", f"{v['f_top1_acc']:.4f}",
                         f"{v['f_top20_acc']:.4f}"])
    for b, lab in ((CHAIN["122"], "chain"), ("mdmv122", "chain + aug")):
        t = J(f"{RUNS}/baselines/{D}/{b}/seed42/eval/chain_f_test.json")
        v = J(f"{RUNS}/baselines/{D}/{b}/seed42/eval/chain_f_valid.json")
        if t is None and b == CHAIN["122"]:                              # Instruments: from the MFD stage-1 eval
            e = mfd_eval(D, "mfd122_stage1")
            t, v = e[0]["chain_decoder_f"], e[1]["chain_decoder_f"]
        if t:
            rows.append([lab, f"{t['f_top1_acc']:.4f}", f"{t['f_top20_acc']:.4f}", f"{v['f_top1_acc']:.4f}",
                         f"{v['f_top20_acc']:.4f}"])
    print(f"f prediction ± augmentation ({D}):\n")
    table(["model", "test f top-1", "test f top-20", "valid f top-1", "valid f top-20"], rows)
    OUT[f"f_acc_{D}"] = rows


def main():
    part_a()
    part_b("mfd122_stage1")
    part_b("mfd133_stage1")
    for D in ("Instruments", "Arts"):
        part_c(D)
    json.dump(OUT, open(f"{MFD}/parts_tables.json", "w"), indent=1)


if __name__ == "__main__":
    main()
