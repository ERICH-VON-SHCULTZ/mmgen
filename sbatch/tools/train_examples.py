"""Training examples per epoch (per task) and optimizer steps for finished runs.
Runs from 2026-10-30 on record the counts in train_stats.json (finetune_ext.py); for older runs they
are recomputed here from the data with MACRec's rules (data.py: train split items[:-2], one example
per prefix for sequence tasks, one per item for item2image/image2item, two per item for mdtrans).
Examples seen = optimizer steps x effective batch.
Usage: python3 train_examples.py [--datasets Instruments Arts] [--seed 42]
"""
import argparse
import glob
import json
import os

RUNS = "/scratch/yw8866/mmgen_runs"


def counts(D):
    inters = json.load(open(f"{RUNS}/data/{D}/{D}.inter.json"))
    n_seq = sum(max(len(v[:-2]) - 1, 0) for v in inters.values())
    n_items = len(json.load(open(f"{RUNS}/data/{D}/{D}.index_lemb_tiger.json")))
    per = {"item2image": n_items, "image2item": n_items, "mdtrans": 2 * n_items}
    return lambda task: per.get(task, n_seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["Instruments", "Arts"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = {}
    for D in a.datasets:
        n = counts(D)
        for run in sorted(glob.glob(f"{RUNS}/baselines/{D}/*/seed{a.seed}")):
            cfg = json.load(open(f"{run}/config_eval.json")) if os.path.exists(f"{run}/config_eval.json") else None
            st = json.load(open(f"{run}/train_stats.json")) if os.path.exists(f"{run}/train_stats.json") else None
            if not cfg or not st:
                continue
            per_task = st.get("train_examples_per_task") or {t: n(t) for t in cfg["tasks"].split(",")}
            epoch = sum(per_task.values())
            r = {"tasks": per_task, "examples_per_epoch": epoch, "optimizer_steps": st["global_steps"],
                 "effective_batch": cfg["effective_batch"], "examples_seen": st["global_steps"] * cfg["effective_batch"],
                 "epochs_run": st["epochs_run"], "best_epoch": st["best_epoch"],
                 "recorded": "train_examples_per_task" in st}
            out[f"{D}/{cfg['baseline']}"] = r
            print(f"{D:12s} {cfg['baseline']:28s} ex/epoch {epoch:8d}  steps {r['optimizer_steps']:6d}  "
                  f"seen {r['examples_seen'] / 1e6:6.1f}M  epochs {r['epochs_run']:.0f} (best {r['best_epoch']:.0f})")
    json.dump(out, open(f"{RUNS}/baselines/train_examples_seed{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
