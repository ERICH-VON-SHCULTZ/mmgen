"""Config diff of the encoder-structure runs (docs/MFD_REPORT.md §16, sanity check 1) against the chain control,
dry run vs dry run (resolved args, TrainingArguments, model config, examples, tokenizer size). Allowed: run-dir paths and
the structure itself (model_class StructT5, n_params = + slot / recency embeddings, model_config.struct_mode); the
dry run's struct_init.json must state the intended mode. Anything else is UNINTENDED -> not ok.
Run inside the env next to copies of geo_config_diff.py and plugin_config_diff.py.
Usage: python struct_config_diff.py <D> run [run ...] -> mmgen_runs/struct/gate/<D>/struct_config_diff.json
"""
import json
import sys

from geo_config_diff import classify, comparable
from plugin_config_diff import diff

RUNS = "/scratch/yw8866/mmgen_runs"


def main():
    D, runs = sys.argv[1], sys.argv[2:]
    G = f"{RUNS}/struct/gate"
    ctl = json.load(open(f"{RUNS}/plugin/dryrun/{D}/md122_a05_ep500_reassign/seed42/dryrun.json"))
    res = {"dataset": D, "control": "chain md122_a05_ep500_reassign dry run", "runs": {}}
    for r in runs:
        p = json.load(open(f"{G}/dryrun/{D}/{r}/dryrun.json"))
        si = json.load(open(f"{G}/dryrun/{D}/{r}/struct_init.json"))
        cl = classify(diff(comparable(ctl), comparable(p)))
        for k in [k for k in cl["UNINTENDED"] if k == "model_config.struct_mode"]:
            cl["geometry"][k] = cl["UNINTENDED"].pop(k)              # the structure itself (category name kept from geo)
        mode_ok = si["struct_mode"] == r.split("_")[1] and p["model_config"].get("struct_mode") == si["struct_mode"]
        ok = not cl["UNINTENDED"] and mode_ok
        res["runs"][r] = {"structure": cl["geometry"], "paths": cl["paths"], "volatile": cl["volatile"],
                          "UNINTENDED": cl["UNINTENDED"], "ok": ok, "struct_init": si}
        print(f"== {r} vs chain control: {'OK' if ok else 'NOT OK'} (mode {si['struct_mode']})")
        for cat in ("geometry", "UNINTENDED"):
            for k, v in cl[cat].items():
                print(f"   [{'structure' if cat == 'geometry' else cat}] {k}: control={v[0][:100]}  {r}={v[1][:100]}")
    res["all_ok"] = all(v["ok"] for v in res["runs"].values())
    json.dump(res, open(f"{G}/{D}/struct_config_diff.json", "w"), indent=1)
    print("ALL RUNS OK" if res["all_ok"] else "SOME RUNS NOT OK")
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
