"""§22 config gate: each round-22 dry run (sbatch/r22/run_r22_train.sh DRYRUN=1) vs round 21's A01 dry run (the base), dry vs dry.
Allowed: run-dir paths, volatile fields and per variant
  E2g*: the aux heads (model_class, n_params, model_config.r22_aux / r22_aux_nt / r22_aux_ni)
  E1:   the 1+3+3 SID set (index_file / image_index_file, vocabulary: model_config.vocab_size, tokenizer_len, n_params)
Anything else is UNINTENDED -> not ok.  Usage: python r22_config_diff.py <D> VAR [VAR ...] -> geo/gate/<D>/r22_config_diff.json (merged)
"""
import json
import sys

from plugin_config_diff import diff
from r21_config_diff import PATHS, VOLATILE, comparable

RUNS = "/scratch/yw8866/mmgen_runs"
OK = {"E2": {"model_class", "n_params", "model_config.r22_aux", "model_config.r22_aux_nt", "model_config.r22_aux_ni"},
      "E1": {"resolved_args.index_file", "resolved_args.image_index_file", "model_config.vocab_size", "tokenizer_len", "n_params"}}


def main():
    D, vars_ = sys.argv[1], sys.argv[2:]
    ref = json.load(open(f"{RUNS}/geo/gate/r21_dryrun/{D}/r21_A01/dryrun.json"))
    path = f"{RUNS}/geo/gate/{D}/r22_config_diff.json"
    try:
        out = json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        out = {"runs": {}}
    out.update({"dataset": D, "reference": f"{RUNS}/geo/gate/r21_dryrun/{D}/r21_A01/dryrun.json"})
    allok = True
    for V in vars_:
        d = json.load(open(f"{RUNS}/geo/gate/r22_dryrun/{D}/r22_{V}/dryrun.json"))
        okset = OK["E2" if V.startswith("E2") else "E1"]
        cls = {"allowed": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
        for k, v in diff(comparable(ref), comparable(d)).items():
            last = k.split(".")[-1]
            c = "paths" if last in PATHS else "volatile" if last in VOLATILE else "allowed" if k in okset else "UNINTENDED"
            cls[c][k] = [str(v[0]), str(v[1])]
        ok = not cls["UNINTENDED"]
        allok &= ok
        out["runs"][f"r22_{V}"] = {"ok": ok, **cls}
        print(f"r22_{V}: {'OK' if ok else 'NOT OK'}", json.dumps(cls["UNINTENDED"]), json.dumps(cls["allowed"])[:400])
    json.dump(out, open(path, "w"), indent=1)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
