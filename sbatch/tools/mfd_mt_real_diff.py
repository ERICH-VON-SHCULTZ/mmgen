"""Real-vs-real check for MFD with C's task mix (docs/MFD_REPORT.md §13, sanity check 4b). Run inside the env.

The dry run (mfd_dryrun.py) does not reproduce real finetune_mfd.py runs in two TrainingArguments: every real MFD run
so far has dataloader_num_workers=0 and prediction_loss_only=True and no StatsTrainer fields in train_stats.json
(finetune_ext.main()'s in-function patches do not take effect there), the dry run has 2 / False. So once a task-mix
run has saved its first checkpoint, its training_args.bin, model config.json and the printed args line are compared
with mfd122_mv's REAL ones. Allowed: run-dir paths and the task mix (tasks). Anything else -> exit 1 (cancel the run).
Usage: python mfd_mt_real_diff.py <D> [run ...] -> mmgen_runs/mfd/mt_gate/<D>/real_diff.json
"""
import ast
import glob
import json
import sys

import torch

from plugin_config_diff import diff

RUNS = "/scratch/yw8866/mmgen_runs"
ALLOWED = {"output_dir", "logging_dir", "run_name", "stats_file", "run_config_file", "tasks"}


def args_line(path):
    return ast.literal_eval(next(l for l in open(path) if l.startswith("{'seed'")).strip())


def load(run_dir):
    ck = sorted(glob.glob(f"{run_dir}/checkpoint-*/training_args.bin")) or [f"{run_dir}/training_args.bin"]
    cfg = json.load(open(ck[0].replace("training_args.bin", "config.json")))
    return {"training_args": torch.load(ck[0], weights_only=False).to_dict(),
            "model_config": {k: v for k, v in cfg.items() if k not in ("transformers_version", "_name_or_path")},
            "args": args_line(f"{run_dir}/train.log")}


def main():
    D = sys.argv[1]
    runs = sys.argv[2:] or ["mfd122_mtA", "mfd122_mtB"]
    ref = load(f"{RUNS}/mfd/{D}/mfd122_mv/seed42")
    res, bad = {}, False
    for r in runs:
        new = load(f"{RUNS}/mfd/{D}/{r}/seed42")
        d = {part: diff(ref[part], new[part]) for part in ref}
        unintended = {f"{p}.{k}": [str(x) for x in v] for p in d for k, v in d[p].items() if k.split(".")[-1] not in ALLOWED}
        allowed = {f"{p}.{k}": [str(x)[:120] for x in v] for p in d for k, v in d[p].items() if k.split(".")[-1] in ALLOWED}
        ok = not unintended
        bad |= not ok
        res[r] = {"ok": ok, "unintended": unintended, "allowed": allowed}
        print(f"== {r} vs mfd122_mv (real vs real, {D}): {'OK' if ok else 'UNINTENDED DIFFERENCES'}")
        for k, v in unintended.items():
            print(f"   [UNINTENDED] {k}: mv={v[0]}  {r}={v[1]}")
        print(f"   allowed: {sorted(allowed)}")
    json.dump(res, open(f"{RUNS}/mfd/mt_gate/{D}/real_diff.json", "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
