"""§21 config gate (docs/MFD_REPORT.md §21): each round-21 dry run vs P1a's dry run made by the same gate, dry vs dry
(sbatch/r21/run_r21_train.sh DRYRUN=1 VAR=P1aref = the exact P1a command). Inside the env, next to plugin_config_diff.py.
Allowed differences: run-dir paths, volatile fields, and per variant
  grouped (G0, A*, B*): per-device batch 341 pairs (train + eval), gradient accumulation 1, train examples per epoch = pairs
                        (n instead of 3n; encoder sequences per epoch stay 3n), model_class (GroupedItemMFDT5 geo class)
  C*:                   train examples per task keyed seqrec_itemrand instead of seqrec_mv (same count 3n)
Anything else is UNINTENDED -> not ok. Usage: python r21_config_diff.py <D> VAR [VAR ...] -> geo/gate/<D>/r21_config_diff.json
"""
import json
import sys

from plugin_config_diff import diff

RUNS = "/scratch/yw8866/mmgen_runs"
PATHS = {"output_dir", "logging_dir", "run_name", "stats_file", "run_config_file"}
VOLATILE = {"_n_gpu", "local_rank", "deepspeed_plugin", "accelerator_config", "hub_token", "push_to_hub_token",
            "torch_compile_backend", "torch_compile_mode"}
GROUPED_OK = {"resolved_args.per_device_batch_size", "resolved_args.gradient_accumulation_steps",
              "training_args.per_device_train_batch_size", "training_args.per_device_eval_batch_size",
              "training_args.gradient_accumulation_steps", "train_examples_per_epoch", "model_class",
              "train_examples_per_task.seqrec_mv", "train_examples_per_task.seqrec_mv_grouped"}
C_OK = {"train_examples_per_task.seqrec_mv", "train_examples_per_task.seqrec_itemrand"}


def comparable(d):
    return {k: d[k] for k in ("resolved_args", "training_args", "model_config", "model_class", "n_params",
                              "train_examples_per_task", "train_examples_per_epoch", "valid_examples", "tokenizer_len")}


def main():
    D, vars_ = sys.argv[1], sys.argv[2:]
    G = f"{RUNS}/geo/gate/r21_dryrun/{D}"
    ref = json.load(open(f"{G}/P1aref/dryrun.json"))
    out = {"dataset": D, "reference": f"{G}/P1aref/dryrun.json", "runs": {}}
    allok = True
    for V in vars_:
        d = json.load(open(f"{G}/r21_{V}/dryrun.json"))
        dd = diff(comparable(ref), comparable(d))
        okset = GROUPED_OK if V[0] in "GAB" else C_OK
        cls = {"allowed": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
        for k, v in dd.items():
            last = k.split(".")[-1]
            c = "paths" if last in PATHS else "volatile" if last in VOLATILE else "allowed" if k in okset else "UNINTENDED"
            cls[c][k] = [str(v[0]), str(v[1])]
        ex = d["train_examples_per_epoch"], ref["train_examples_per_epoch"]
        exp_ok = (3 * ex[0] == ex[1]) if V[0] in "GAB" else (ex[0] == ex[1])
        bs_ok = True
        if V[0] in "GAB":
            ta = d["training_args"]
            bs_ok = ta["per_device_train_batch_size"] == 341 and ta["gradient_accumulation_steps"] == 1
        ok = not cls["UNINTENDED"] and exp_ok and bs_ok
        allok &= ok
        out["runs"][f"r21_{V}"] = {"ok": ok, "examples_per_epoch": ex, "batch_ok": bs_ok, **cls}
        print(f"r21_{V}: {'OK' if ok else 'NOT OK'}", json.dumps(cls["UNINTENDED"]), json.dumps(cls["allowed"]))
    json.dump(out, open(f"{RUNS}/geo/gate/{D}/r21_config_diff.json", "w"), indent=1)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
