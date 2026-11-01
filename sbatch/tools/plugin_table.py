"""Tables for docs/PLUGIN_REPORT.md from saved result files only (seed 42 unless noted).
Rows per dataset: C seed 42 and C 3-seed mean ± std, D seed 42 and 3-seed mean, the plug-in runs (text / image /
merged), the MD chain decoder (seqrec only) and the best MFD setting (chosen on VALIDATION HR@10 among the MFD
variants evaluated in docs/MFD_REPORT.md). Also the view-building and first-token tables and the paired CIs.
Usage: python3 plugin_table.py  -> prints markdown, writes mmgen_runs/plugin/tables.json
"""
import json
import os

import statistics as st

RD = "/scratch/yw8866/mmgen_runs"
M = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
PLUG = {"Instruments": ["plug122", "plug122s", "plug133", "plug133s"], "Arts": ["plug122", "plug133"]}
NAME = {"plug122": "plug-in 1+2+2 main", "plug122s": "plug-in 1+2+2 shared-f", "plug133": "plug-in 1+3+3 main",
        "plug133s": "plug-in 1+3+3 shared-f"}
LIST = {"seqrec": "text", "seqimage": "image", "merged": "merged"}
OUT = {}


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def summ(D, B, s=42):
    return J(f"{RD}/baselines/{D}/{B}/seed{s}/run_summary.json")


def gpu(c):
    return (c or {}).get("gpu", "?").replace("NVIDIA ", "").replace("-SXM4-80GB", "")


def row(label, test, valid, passes, ms, g):
    return [label] + [f"{test[k]:.4f}" for k in M] + [f"{valid['hit@10']:.4f} / {valid['ndcg@10']:.4f}", passes,
                                                     f"{ms:.1f} ({g})" if ms is not None else "–"]


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def two_list_rows(D, B, label, seed=42):
    s = summ(D, B, seed)
    rows = []
    for t in ("seqrec", "seqimage", "merged"):
        if t == "merged":
            c = [s["test"][x]["cost"] for x in ("seqrec", "seqimage")]
            rows.append(row(f"{label}, merged", s["test"][t]["metrics"], s["valid"][t]["metrics"], "2 × " +
                            str(round(c[0]["decoder_forward_passes_per_user"])),
                            sum(x["ms_per_user"] for x in c), gpu(c[0])))
        else:
            c = s["test"][t]["cost"]
            rows.append(row(f"{label}, {LIST[t]}", s["test"][t]["metrics"], s["valid"][t]["metrics"],
                            round(c["decoder_forward_passes_per_user"]), c["ms_per_user"], gpu(c)))
    return rows


def seed_mean_row(D, B, label):
    out = []
    for t in ("seqrec", "seqimage", "merged"):
        vals = {k: [summ(D, B, s)["test"][t]["metrics"][k] for s in (42, 43, 44)] for k in M}
        out.append([f"{label}, {LIST[t]} (3-seed mean ± std)"] +
                   [f"{st.mean(vals[k]):.4f} ± {st.stdev(vals[k]):.4f}" for k in M] + ["–", "", ""])
    return out


def best_mfd(D):
    cands = []
    for run in ("mfd122_stage1", "mfd122_mv", "mfd133_stage1"):
        base = f"{RD}/mfd/{D}/{run}/seed42"
        t, v = J(f"{base}/eval_kf20/results_test.json"), J(f"{base}/eval_kf20/results_valid.json")
        if t:
            cands.append((f"MFD {run} joint, K_f=20, w=1", t["modes"]["joint"], v["modes"]["joint"],
                          t["cost"]["decoder_passes_per_user"], t["cost"]["ms_per_user"], gpu(t["cost"])))
        for sub in ("sweep", "sweep_applied"):
            at = J(f"{base}/{sub}/apply_test.json")
            if not at:
                continue
            ch = at["chosen"]
            r = at["results"][ch["name"]]
            av = J(f"{base}/{sub}/apply_valid.json")
            if av:
                rv = av["results"][ch["name"]]
            else:
                sv = J(f"{base}/{sub}/sweep_valid.json")
                rv = next(x for x in sv["settings"] if x.get("name") == ch["name"] and "mode" not in x)
            cands.append((f"MFD {run}, Part B setting {ch['name']}", r, rv, r["cost"]["decoder_passes_per_user"],
                          r["cost"]["ms_per_user"], gpu(r["cost"])))
    best = max(cands, key=lambda c: (c[2]["hit@10"], c[2]["ndcg@10"]))
    return row(f"best MFD (valid-chosen): {best[0]}", best[1], best[2], best[3], best[4], best[5])


def results(D):
    print(f"### {D} (test; valid HR@10 / NDCG@10 next to it)\n")
    rows = two_list_rows(D, "mql", "C MQL4GRec seed 42") + seed_mean_row(D, "mql", "C MQL4GRec")
    rows += two_list_rows(D, "macrecpaper", "D MACRec seed 42") + seed_mean_row(D, "macrecpaper", "D MACRec")[2:]
    for B in PLUG[D]:
        if summ(D, B):
            rows += two_list_rows(D, B, NAME[B])
    for B, lab in (("md122_a05_ep500_reassign", "MD 1+2+2 chain decoder (seqrec only)"),
                   ("md133_a06_ep500_reassign", "MD 1+3+3 chain decoder (seqrec only)")):
        s = summ(D, B)
        c = s["test"]["seqrec"]["cost"]
        rows.append(row(lab, s["test"]["seqrec"]["metrics"], s["valid"]["seqrec"]["metrics"],
                        round(c["decoder_forward_passes_per_user"]), c["ms_per_user"], gpu(c)))
    rows.append(best_mfd(D))
    table(["method", "HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "valid HR@10 / NDCG@10", "decoder passes / list",
           "ms / user (GPU)"], rows)
    OUT[f"results_{D}"] = rows


def views():
    print("### Views\n")
    rows = []
    for D in ("Instruments", "Arts"):
        for c in ("122", "133"):
            v = J(f"{RD}/plugin/{D}/{c}/views.json")
            for name in ("text", "image"):
                x = v["views"][name]
                rows.append([D, v["cfg"], name, f"{x['items_in_collision_before']} ({100 * x['collision_rate_before']:.2f}%)",
                             x["groups_before"], x["max_group"], x["items_reassigned"], x["second_to_last_level_changed"],
                             x["passes"], v["files"][f"main/{name}"]["file"], v["files"][f"shared_f/{name}"]["file"],
                             v["v1_sha256"][:16] + "…"])
    table(["dataset", "SID", "view", "items colliding before", "groups", "max group", "items reassigned",
           "2nd-to-last level changed", "passes", "main file", "shared-f file", "v1 sha256"], rows)
    OUT["views"] = rows


def first_token():
    print("### First-token accuracy (test; valid in brackets)\n")
    rows = []
    for D in ("Instruments", "Arts"):
        for B in ["mql"] + PLUG[D]:
            t = J(f"{RD}/baselines/{D}/{B}/seed42/eval/first_token_test.json")
            v = J(f"{RD}/baselines/{D}/{B}/seed42/eval/first_token_valid.json")
            if not t:
                continue
            r = [D, "C MQL4GRec" if B == "mql" else NAME[B]]
            for task in ("seqrec", "seqimage"):
                r += [t[task]["first_tokens"], f"{t[task]['top1']:.4f} ({v[task]['top1']:.4f})",
                      f"{t[task]['top20']:.4f} ({v[task]['top20']:.4f})"]
            rows.append(r)
    table(["dataset", "run", "text: first tokens", "text top-1", "text top-20", "image: first tokens", "image top-1",
           "image top-20"], rows)
    OUT["first_token"] = rows


def paired_table():
    p = J(f"{RD}/plugin/paired.json")
    if not p:
        return
    print("### Paired vs C seed 42 (same users; 2,000 bootstrap resamples; 95% CI)\n")
    rows = []
    for r, x in p["runs"].items():
        D, B = r.split(":")
        if B == "mql":
            continue
        for lst in ("merged", "best_single"):
            for split in ("test", "valid"):
                y = x["paired"][split][lst]
                lab = lst if lst == "merged" else f"best single ({LIST[x['best_single_valid']]} vs C {LIST[x['C_best_single_valid']]})"
                rows.append([D, NAME[B], lab, split, f"{y['hr10']['plugin']:.4f} vs {y['hr10']['C']:.4f}",
                             f"{y['hr10']['diff']:+.4f} [{y['hr10']['ci95'][0]:+.4f}, {y['hr10']['ci95'][1]:+.4f}]",
                             f"{y['ndcg10']['diff']:+.4f} [{y['ndcg10']['ci95'][0]:+.4f}, {y['ndcg10']['ci95'][1]:+.4f}]",
                             f"{y['won']}/{y['lost']}"])
    table(["dataset", "run", "list", "split", "HR@10 plug-in vs C", "ΔHR@10 [95% CI]", "ΔNDCG@10 [95% CI]",
           "users won / lost"], rows)
    sp = p["C_seed_spread"]
    print("C seed spread (test, seeds 42/43/44): " + "; ".join(
        f"{D} {LIST[t]} HR@10 {sp[D]['summary'][t]['hit@10']['min']:.4f}–{sp[D]['summary'][t]['hit@10']['max']:.4f} "
        f"(std {sp[D]['summary'][t]['hit@10']['std']:.4f})" for D in sp for t in ("seqrec", "seqimage", "merged")) + "\n")
    OUT["paired"] = rows


def main():
    views()
    for D in ("Instruments", "Arts"):
        results(D)
    first_token()
    paired_table()
    json.dump(OUT, open(f"{RD}/plugin/tables.json", "w"), indent=1)


if __name__ == "__main__":
    main()
