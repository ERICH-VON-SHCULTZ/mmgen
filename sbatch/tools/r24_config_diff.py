"""§24 config gate: every round-24 dry run (sbatch/r24/run_r24_train.sh DRYRUN=1) vs round 24's P1aref dry run (P1a's exact command
through finetune_r24.py), dry vs dry; and r24 P1aref vs round 21's P1aref (the original P1a command): only the base-model path may
differ (the run script passes an edited copy of config/ckpt; for P1aref its values are config/ckpt's).
Allowed per VAR token (docs/MFD_REPORT.md §24):
  lr*  learning_rate (resolved args + training args)      wd01  weight_decay (both)        gn5  training_args.max_grad_norm
  do*  model_config.dropout_rate                           dm256 d_model, d_ff, num_heads (d_kv stays 64), n_params
  l6   num_layers, num_decoder_layers, n_params            B*    model_class, train examples per task / per epoch (+ r24_item)
Always allowed: run paths, volatile fields, resolved_args.base_model (path). Anything else is UNINTENDED -> not ok.
Usage: python r24_config_diff.py <D> VAR [VAR ...] -> geo/gate/<D>/r24_config_diff.json (merged)"""
import json
import sys

from plugin_config_diff import diff
from r21_config_diff import PATHS, VOLATILE, comparable

RUNS = "/scratch/yw8866/mmgen_runs"
LR = {"resolved_args.learning_rate", "training_args.learning_rate"}
OK = {"lr3e4": LR, "lr1e3": LR, "wd01": {"resolved_args.weight_decay", "training_args.weight_decay"},
      "gn5": {"training_args.max_grad_norm"}, "do02": {"model_config.dropout_rate"}, "do03": {"model_config.dropout_rate"},
      "dm256": {"model_config.d_model", "model_config.d_ff", "model_config.num_heads", "n_params"},
      "l6": {"model_config.num_layers", "model_config.num_decoder_layers", "n_params"},
      "B1": {"model_class", "train_examples_per_task.r24_item", "train_examples_per_epoch"},
      "B2": {"model_class", "train_examples_per_task.r24_item", "train_examples_per_epoch"}, "P1aref": set()}
ALWAYS = {"resolved_args.base_model"}


def classify(a, b, okset):
    cls = {"allowed": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
    for k, v in diff(comparable(a), comparable(b)).items():
        last = k.split(".")[-1]
        c = "paths" if last in PATHS else "volatile" if last in VOLATILE else "allowed" if (k in okset or k in ALWAYS) else "UNINTENDED"
        cls[c][k] = v
    return cls


def main():
    D, vars_ = sys.argv[1], sys.argv[2:]
    ref = json.load(open(f"{RUNS}/geo/gate/r24_dryrun/{D}/r24_P1aref/dryrun.json"))
    path = f"{RUNS}/geo/gate/{D}/r24_config_diff.json"
    try:
        out = json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        out = {"runs": {}}
    r21 = json.load(open(f"{RUNS}/geo/gate/r21_dryrun/{D}/P1aref/dryrun.json"))
    c = classify(r21, ref, set())
    out["r24_P1aref_vs_r21_P1aref"] = {"ok": not c["UNINTENDED"], **c}
    allok = not c["UNINTENDED"]
    print("r24 P1aref vs r21 P1aref:", "OK" if allok else "NOT OK", json.dumps(c["UNINTENDED"]))
    for V in vars_:
        okset = set().union(*(OK[t] for t in V.split("_")))
        cls = classify(ref, json.load(open(f"{RUNS}/geo/gate/r24_dryrun/{D}/r24_{V}/dryrun.json")), okset)
        # training_args.max_grad_norm (gn5) is set by a TrainingArguments wrapper, which dry runs never pass through (as
        # finetune_mfd's prediction_loss_only, §13.2): it is checked in the 5-step smoke run of the exact command instead
        expect = set().union(*(OK[t] for t in V.split("_"))) - {"n_params", "model_class", "training_args.max_grad_norm"}
        missing = sorted(k for k in expect if not any(k == x or x.startswith(k) for x in cls["allowed"]))
        smoke_gn = None
        if "gn5" in V.split("_"):
            try:
                smoke_gn = json.load(open(f"{RUNS}/geo/gate/r24_smoke/{D}/r24_{V}/smoke.json"))["max_grad_norm"]
            except (FileNotFoundError, KeyError):
                smoke_gn = "pending (checked by the smoke run: smoke.json ok requires max_grad_norm == 5)"
        ok = not cls["UNINTENDED"] and (not missing or V == "P1aref") and smoke_gn in (None, 5.0) or \
            (not cls["UNINTENDED"] and not missing and isinstance(smoke_gn, str))
        allok &= ok
        out["runs"][f"r24_{V}"] = {"ok": ok, "expected_but_unchanged": missing, "smoke_max_grad_norm": smoke_gn, **cls}
        print(f"r24_{V}: {'OK' if ok else 'NOT OK'}", json.dumps(cls["UNINTENDED"]), json.dumps(cls["allowed"])[:300], missing)
    json.dump(out, open(path, "w"), indent=1)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
