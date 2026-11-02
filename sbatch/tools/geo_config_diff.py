"""Config diff of the codebook-geometry runs against their controls (docs/MFD_REPORT.md §14, sanity check 3).
Run inside the env, next to a copy of plugin_config_diff.py (flat / diff helpers).

Reference (recorded, informational): the chain control's dry run (run_baseline.sh DRYRUN=1 B=md122_a05_ep500_reassign)
  vs its REAL seed-42 run (printed args, training_args.bin, config.json).
Runs: each geometry dry run (run_geo_train.sh DRYRUN=1) vs its control's dry run, dry vs dry:
  chain122_<VAR>     vs the chain control (plugin/dryrun/<D>/md122_a05_ep500_reassign/seed42)
  mfd122_mv_geo<VAR> vs mfd122_mv (mfd/mt_gate/dryrun/<D>/mfd122_mv)
  (resolved args, TrainingArguments, model config incl. vocab, examples, tokenizer size). Allowed: run-dir paths and
  the embedding construction itself (model_class Geo*, n_params = + P_level + delta). The f loss is not an argument;
  geo_init.json of the dry run must state the intended construction (geo, eps, tied head, codebook tensor).
  Anything else is UNINTENDED -> not ok.
Usage: python geo_config_diff.py <D> run [run ...] -> mmgen_runs/geo/gate/<D>/geo_config_diff.json (merged with runs
recorded earlier); exit status = the run checks.
"""
import ast
import json
import sys

import torch

from plugin_config_diff import diff

RUNS = "/scratch/yw8866/mmgen_runs"
PATHS = {"output_dir", "logging_dir", "run_name", "stats_file", "run_config_file"}
GEOMETRY = {"model_class", "n_params"}
VOLATILE = {"_n_gpu", "local_rank", "deepspeed_plugin", "accelerator_config", "hub_token", "push_to_hub_token",
            "torch_compile_backend", "torch_compile_mode"}
WANT = {"A": ("codebook", 0.0), "C": ("shuffled", 0.0), "AB1": ("codebook", 0.1), "AB2": ("codebook", 0.2)}


def classify(d):
    out = {"geometry": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
    for k, v in d.items():
        parts = k.split(".")
        cat = ("geometry" if parts[0] in GEOMETRY else "paths" if parts[-1] in PATHS
               else "volatile" if parts[-1] in VOLATILE else "UNINTENDED")
        out[cat][k] = [str(v[0]), str(v[1])]
    return out


def comparable(dump):
    return {k: dump[k] for k in ("resolved_args", "training_args", "model_config", "model_class", "n_params",
                                 "train_examples_per_task", "train_examples_per_epoch", "valid_examples", "tokenizer_len")}


def args_line(path):
    return ast.literal_eval(next(l for l in open(path) if l.startswith("{'seed'")).strip())


def main():
    D, runs = sys.argv[1], sys.argv[2:]
    G = f"{RUNS}/geo/gate"
    chain_dry = f"{RUNS}/plugin/dryrun/{D}/md122_a05_ep500_reassign/seed42"
    real = f"{RUNS}/baselines/{D}/md122_a05_ep500_reassign/seed42"
    cd = json.load(open(f"{chain_dry}/dryrun.json"))
    real_cfg = json.load(open(f"{real}/config.json"))
    keys = set(real_cfg) - {"architectures", "transformers_version", "torch_dtype", "_name_or_path"}
    ref = {"args": classify(diff(args_line(f"{real}/train.log"), args_line(f"{chain_dry}/train.log"))),
           "training_args": classify(diff(torch.load(f"{real}/training_args.bin", weights_only=False).to_dict(), cd["training_args"])),
           "model_config": classify(diff({k: real_cfg[k] for k in keys}, {k: cd["model_config"].get(k) for k in keys}))}
    res = {"dataset": D, "reference": ref | {"chain_dry_reproduces_real_control": not any(ref[x]["UNINTENDED"] for x in ref)},
           "runs": {}}
    print(f"== {D}: chain control dry run vs its real seed-42 run: {'OK' if res['reference']['chain_dry_reproduces_real_control'] else 'MISMATCH'}")
    for x in ref:
        for k, v in ref[x]["UNINTENDED"].items():
            print(f"   [{x}] {k}: real={v[0]}  dry={v[1]}")
    for r in runs:
        ctl = cd if r.startswith("chain122_") else json.load(open(f"{RUNS}/mfd/mt_gate/dryrun/{D}/mfd122_mv/dryrun.json"))
        p = json.load(open(f"{G}/dryrun/{D}/{r}/dryrun.json"))
        gi = json.load(open(f"{G}/dryrun/{D}/{r}/geo_init.json"))
        var = r.split("_")[1] if r.startswith("chain122_") else r.split("_geo")[1]
        geo, eps = WANT[var]
        geo_ok = (gi["geo"] == geo and gi.get("soft_f", {}).get("eps", 0.0) == eps and gi["tied_head"]
                  and gi["codebook_tensor"] == "<chain>_rq.vq_layers.<l>.embedding.weight")
        cl = classify(diff(comparable(ctl), comparable(p)))
        ok = not cl["UNINTENDED"] and geo_ok
        res["runs"][r] = cl | {"ok": ok, "geo_init_as_intended": geo_ok, "geo_init": gi,
                               "control": "chain md122_a05_ep500_reassign" if r.startswith("chain122_") else "mfd122_mv"}
        print(f"== {r} vs its control: {'OK' if ok else 'NOT OK'}  (geo={gi['geo']}, eps={gi.get('soft_f', {}).get('eps', 0.0)}, "
              f"intended: {geo_ok})")
        for cat in ("geometry", "UNINTENDED"):
            for k, v in cl[cat].items():
                print(f"   [{cat}] {k}: control={v[0][:120]}  {r}={v[1][:120]}")
        print(f"   (paths differing: {len(cl['paths'])}, volatile: {sorted(cl['volatile'])})")
    out = f"{G}/{D}/geo_config_diff.json"
    try:
        old = json.load(open(out))
        res["runs"] = {k: v for k, v in old.get("runs", {}).items() if k not in res["runs"]} | res["runs"]
    except FileNotFoundError:
        pass
    res["all_ok"] = all(v["ok"] for v in res["runs"].values())
    json.dump(res, open(out, "w"), indent=1)
    print("ALL RUNS OK" if res["all_ok"] else "SOME RUNS NOT OK")
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
