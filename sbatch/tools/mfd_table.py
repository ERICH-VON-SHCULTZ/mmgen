"""Comparison table for MFD stage 1 (docs/MFD_REPORT.md): MFD modes vs the MD chain decoder and the
baselines, Instruments, seed 42, test with validation next to it. C/D "best single list" is chosen on
VALIDATION HR@10 (nothing is chosen on test). ms/user is the GPU time inside decoding (H200 unless noted).
Usage: python3 mfd_table.py [--dataset Instruments] [--k_f 20]  -> prints markdown, writes mfd_table.json
"""
import argparse
import json
import os

RUNS = "/scratch/yw8866/mmgen_runs"
M = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]


def summ(D, b, seed=42):
    f = f"{RUNS}/baselines/{D}/{b}/seed{seed}/run_summary.json"
    if b == "tiger_fnormj" and not os.path.exists(f):   # Instruments E-norm: the byte-identical final-epoch rerun
        f = f"{RUNS}/baselines/{D}/tiger_fnormj_ep500/seed{seed}/run_summary.json"
    return json.load(open(f))


def row(name, test, valid, passes, ms, gpu="H200"):
    return dict(name=name, test=[test.get(k) for k in M], valid_hr10=valid.get("hit@10"), valid_ndcg10=valid.get("ndcg@10"),
                passes=passes, ms=ms, gpu=gpu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Instruments")
    p.add_argument("--k_f", type=int, default=20)
    a = p.parse_args()
    D = a.dataset
    rows = []
    ev = f"{RUNS}/mfd/{D}/mfd122_stage1/seed42/eval_kf{a.k_f}"
    rt, rv = json.load(open(f"{ev}/results_test.json")), json.load(open(f"{ev}/results_valid.json"))
    ms, gpu = rt["cost"]["ms_per_user"], rt["cost"]["gpu"].replace("NVIDIA ", "")
    for mode, label in (("text", "MFD (a) text only"), ("image", "MFD (b) image only"),
                        ("merge", "MFD (c) C-style merge of (a)+(b)"), ("joint", "**MFD (d) joint**")):
        rows.append(row(label, rt["modes"][mode], rv["modes"][mode], 3, ms, gpu))
    for b, label in (("md122_a05_ep500_reassign", "MD 1+2+2 chain decoder"), ("tiger_text", "A TIGER-text"),
                     ("tiger_image", "B TIGER-image"), ("tiger_fraw", "E-raw"), ("tiger_fnormj", "E-norm")):
        s = summ(D, b)
        c = s["test"]["seqrec"]["cost"]
        rows.append(row(label, s["test"]["seqrec"]["metrics"], s["valid"]["seqrec"]["metrics"],
                        c["decoder_forward_passes_per_user"], c["ms_per_user"], c["gpu"].replace("NVIDIA ", "")))
    for b, label in (("mql", "C MQL4GRec"), ("macrecpaper", "D MACRec")):
        s = summ(D, b)
        best = max(("seqrec", "seqimage"), key=lambda t: s["valid"][t]["metrics"]["hit@10"])
        c = s["test"][best]["cost"]
        rows.append(row(f"{label}, best single list ({'text' if best == 'seqrec' else 'image'}, chosen on valid)",
                        s["test"][best]["metrics"], s["valid"][best]["metrics"], c["decoder_forward_passes_per_user"],
                        c["ms_per_user"], c["gpu"].replace("NVIDIA ", "")))
        ms2 = sum(s["test"][t]["cost"]["ms_per_user"] for t in ("seqrec", "seqimage"))
        rows.append(row(f"{label}, merged (two decodes)", s["test"]["merged"]["metrics"], s["valid"]["merged"]["metrics"],
                        "2 × 5", ms2, s["test"]["seqrec"]["cost"]["gpu"].replace("NVIDIA ", "")))
    print("| method | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | valid HR@10 / NDCG@10 | decoder passes / user | ms / user (GPU) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | " + " | ".join(f"{x:.4f}" for x in r["test"]) +
              f" | {r['valid_hr10']:.4f} / {r['valid_ndcg10']:.4f} | {r['passes']} | {r['ms']:.1f} ({r['gpu']}) |")
    json.dump(rows, open(f"{ev}/mfd_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
