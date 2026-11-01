"""Config diff of MFD with C's task mix (mfd122_mtA / mfd122_mtB) against mfd122_mv (docs/MFD_REPORT.md §13, sanity
check 4). Run inside the env, next to a copy of plugin_config_diff.py (flat / diff helpers).

Step 1 (reference): mfd122_mv's dry run (run_mfd_train.sh DRYRUN=1 AUG=1) must reproduce mfd122_mv's REAL seed-42 run:
  the args printed at the top of its train.log, its training_args.bin and its config.json. Allowed: run-dir paths.
Step 2: each task-mix dry run vs mfd122_mv's dry run (resolved args, TrainingArguments, model config incl. vocab size,
  model class, parameter count, examples). Allowed: the task mix (tasks, soft prompts, examples per task / epoch) and
  run-dir paths. Anything else is UNINTENDED -> exit 1 (do not train). The soft prompts used in training must equal
  C's construction as rebuilt by mfd_mt_data.c_soft_prompts (task_check_<run>.json from mfd_mt_check.py).
Usage: python mfd_mt_config_diff.py <D> [run ...]  -> mmgen_runs/mfd/mt_gate/<D>/config_diff.json
"""
import ast
import json
import sys

import torch

from plugin_config_diff import diff, flat

RUNS = "/scratch/yw8866/mmgen_runs"
PATHS = {"output_dir", "logging_dir", "run_name", "stats_file", "run_config_file"}
TASK_MIX = {"tasks", "soft_prompts", "train_examples_per_task", "train_examples_per_epoch"}
VOLATILE = {"_n_gpu", "local_rank", "deepspeed_plugin", "accelerator_config", "hub_token", "push_to_hub_token",
            "torch_compile_backend", "torch_compile_mode"}


def classify(d):
    out = {"task_mix": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
    for k, v in d.items():
        parts = k.split(".")
        cat = ("task_mix" if parts[0] in TASK_MIX or (len(parts) > 1 and parts[1] in TASK_MIX)
               else "paths" if parts[-1] in PATHS else "volatile" if parts[-1] in VOLATILE else "UNINTENDED")
        out[cat][k] = [str(v[0]), str(v[1])]
    return out


def comparable(dump):
    return {k: dump[k] for k in ("resolved_args", "training_args", "model_config", "model_class", "n_params",
                                 "train_examples_per_task", "train_examples_per_epoch", "valid_examples", "tokenizer_len")}


def args_line(path):
    return ast.literal_eval(next(l for l in open(path) if l.startswith("{'seed'")).strip())


def main():
    D = sys.argv[1]
    runs = sys.argv[2:] or ["mfd122_mtA", "mfd122_mtB"]
    G = f"{RUNS}/mfd/mt_gate"
    dry = lambda r: f"{G}/dryrun/{D}/{r}"
    res, bad = {"dataset": D, "reference": {}, "runs": {}}, False
    real = f"{RUNS}/mfd/{D}/mfd122_mv/seed42"
    mv = json.load(open(f"{dry('mfd122_mv')}/dryrun.json"))
    real_ta = torch.load(f"{real}/training_args.bin", weights_only=False).to_dict()
    real_cfg = json.load(open(f"{real}/config.json"))
    keys = set(real_cfg) - {"architectures", "transformers_version", "torch_dtype", "_name_or_path"}
    ref = {"args": classify(diff(args_line(f"{real}/train.log"), args_line(f"{dry('mfd122_mv')}/train.log"))),
           "training_args": classify(diff(real_ta, mv["training_args"])),
           "model_config": classify(diff({k: real_cfg[k] for k in keys}, {k: mv["model_config"].get(k) for k in keys}))}
    ref_bad = any(ref[x]["UNINTENDED"] or ref[x]["task_mix"] for x in ref)
    res["reference"] = ref | {"dry_run_reproduces_real_mfd122_mv": not ref_bad}
    print(f"== {D}: mfd122_mv dry run vs its real seed-42 run: {'OK' if not ref_bad else 'MISMATCH'}")
    for x in ref:
        for cat in ("task_mix", "UNINTENDED"):
            for k, v in ref[x][cat].items():
                print(f"   [{x}/{cat}] {k}: real={v[0]}  dry={v[1]}")
    bad |= ref_bad
    for r in runs:
        p = json.load(open(f"{dry(r)}/dryrun.json"))
        cl = classify(diff(comparable(mv), comparable(p)))
        tc = json.load(open(f"{G}/{D}/task_check_{r}.json"))
        prompts_ok = p["resolved_args"]["soft_prompts"] == tc["soft_prompts"]
        ok = not cl["UNINTENDED"] and prompts_ok and tc["all_pass"]
        bad |= not ok
        res["runs"][r] = cl | {"ok": ok, "soft_prompts_equal_c_construction": prompts_ok, "task_check_pass": tc["all_pass"],
                               "examples_per_task": p["train_examples_per_task"], "examples_per_epoch": p["train_examples_per_epoch"],
                               "mfd122_mv_examples_per_epoch": mv["train_examples_per_epoch"]}
        print(f"== {r} vs mfd122_mv ({D}): {'OK' if ok else 'NOT OK'}  (soft prompts = C's: {prompts_ok}, task check: {tc['all_pass']})")
        for cat in ("task_mix", "UNINTENDED"):
            for k, v in cl[cat].items():
                print(f"   [{cat}] {k}: mv={v[0][:160]}  {r}={v[1][:160]}")
        print(f"   (paths differing: {len(cl['paths'])}, volatile: {sorted(cl['volatile'])})")
    res["all_ok"] = not bad
    json.dump(res, open(f"{G}/{D}/config_diff.json", "w"), indent=1)
    print("ALL OK" if not bad else "CONFIG DIFF FAILED: do not train")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
