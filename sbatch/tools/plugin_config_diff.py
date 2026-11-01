"""Config diff of the plug-in runs against C (MQL4GRec) seed 42 (docs/PLUGIN_REPORT.md §2). Run inside the env.

Step 1 (reference): C's dry run (run_baseline.sh DRYRUN=1, B=mql) must reproduce C's REAL seed-42 run: the
  arguments printed at the top of its train.log, its training_args.bin (TrainingArguments) and its config.json
  (model config incl. vocab size). C's run predates run_config.json, so these are its resolved config.
Step 2: every plug-in dry run vs C's dry run, over resolved args, TrainingArguments, model config, model class and
  training examples per task. Allowed differences: the index files, the vocabulary (vocab size, added tokens,
  parameter count) and run-directory paths. Anything else is UNINTENDED -> exit code 1 (do not train).
Writes mmgen_runs/plugin/config_diff.json.
"""
import ast
import json
import sys

import torch

RUNS = "/scratch/yw8866/mmgen_runs"
DRY = f"{RUNS}/plugin/dryrun"
PATHS = {"output_dir", "logging_dir", "run_name", "stats_file", "run_config_file"}     # run-directory paths
INDEX = {"index_file", "image_index_file"}
VOCAB = {"vocab_size", "n_params", "tokenizer_len"}
# fields that only describe the software/hardware state of a run, not the training recipe
VOLATILE = {"_n_gpu", "local_rank", "deepspeed_plugin", "accelerator_config", "hub_token", "push_to_hub_token",
            "torch_compile_backend", "torch_compile_mode"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and k not in ("soft_prompts",):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return {k: (fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in sorted(set(fa) | set(fb))
            if str(fa.get(k, "<absent>")) != str(fb.get(k, "<absent>"))}


def comparable(dump):
    return {"resolved_args": dump["resolved_args"], "training_args": dump["training_args"],
            "model_config": dump["model_config"], "model_class": dump["model_class"], "n_params": dump["n_params"],
            "train_examples_per_task": dump["train_examples_per_task"], "valid_examples": dump["valid_examples"]}


def classify(d):
    out = {"index": {}, "vocab": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
    for k, v in d.items():
        leaf = k.split(".")[-1]
        cat = ("index" if leaf in INDEX else "vocab" if leaf in VOCAB else "paths" if leaf in PATHS
               else "volatile" if leaf in VOLATILE else "UNINTENDED")
        out[cat][k] = [str(v[0]), str(v[1])]
    return out


def main():
    runs = sys.argv[1:] or ["Instruments:plug122", "Instruments:plug122s", "Instruments:plug133", "Instruments:plug133s",
                            "Arts:plug122", "Arts:plug133"]
    res, bad = {"reference": {}, "runs": {}}, False
    for D in sorted({r.split(":")[0] for r in runs}):
        c_dry = json.load(open(f"{DRY}/{D}/mql/seed42/dryrun.json"))
        real = f"{RUNS}/baselines/{D}/mql/seed42"
        line = next(l for l in open(f"{real}/train.log") if l.startswith("{'seed'"))
        real_args = ast.literal_eval(line.strip())
        real_ta = torch.load(f"{real}/training_args.bin", weights_only=False).to_dict()
        real_cfg = json.load(open(f"{real}/config.json"))
        # compare like with like: the args line train() prints at start-up (before it sets args.deepspeed = None),
        # from C's real log and from the dry run's own log
        dline = next(l for l in open(f"{DRY}/{D}/mql/seed42/train.log") if l.startswith("{'seed'"))
        dry_args = ast.literal_eval(dline.strip())
        cfg_keys = set(real_cfg) - {"architectures", "transformers_version", "torch_dtype", "_name_or_path"}
        ref = {"args": classify(diff(real_args, dry_args)),
               "training_args": classify(diff(real_ta, c_dry["training_args"])),
               "model_config": classify(diff({k: real_cfg[k] for k in cfg_keys},
                                             {k: c_dry["model_config"].get(k) for k in cfg_keys}))}
        ref_bad = any(ref[x]["UNINTENDED"] or ref[x]["vocab"] or ref[x]["index"] for x in ref)
        res["reference"][D] = ref | {"dry_run_reproduces_real_C": not ref_bad}
        print(f"== {D}: C dry run vs C's real seed-42 run: {'OK' if not ref_bad else 'MISMATCH'}")
        for x in ref:
            for cat in ("index", "vocab", "UNINTENDED"):
                for k, v in ref[x][cat].items():
                    print(f"   [{x}/{cat}] {k}: real={v[0]}  dry={v[1]}")
        bad |= ref_bad
    for r in runs:
        D, B = r.split(":")
        p = json.load(open(f"{DRY}/{D}/{B}/seed42/dryrun.json"))
        c = json.load(open(f"{DRY}/{D}/mql/seed42/dryrun.json"))
        cl = classify(diff(comparable(c), comparable(p)))
        ok = not cl["UNINTENDED"]
        bad |= not ok
        res["runs"][r] = cl | {"ok": ok, "added_tokens": p["tokenizer_len"], "c_added_tokens": c["tokenizer_len"]}
        print(f"== {r} vs C {D}: {'OK' if ok else 'UNINTENDED DIFFERENCES'}")
        for cat in ("index", "vocab", "UNINTENDED"):
            for k, v in cl[cat].items():
                print(f"   [{cat}] {k}: C={v[0]}  plug-in={v[1]}")
        print(f"   (paths differing: {len(cl['paths'])}, volatile: {sorted(cl['volatile'])})")
    res["all_ok"] = not bad
    json.dump(res, open(f"{RUNS}/plugin/config_diff.json", "w"), indent=1)
    print("ALL OK" if not bad else "CONFIG DIFF FAILED: do not train")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
