"""§24.4 gate: a tuned baseline's dry run vs the same baseline's dry run at its own settings (TAG ref), plus its 5-step smoke run.
Allowed differences: the learning rate (lr tokens) and the batch size (b256); anything else is UNINTENDED. The smoke run (exact training
command) must have finite losses and the requested learning rate / batch. -> geo/gate/r24_base/<D>/diff_<B>_<TAG>.json
Usage: python r24_base_diff.py D B TAG"""
import json
import sys

from plugin_config_diff import diff
from r21_config_diff import PATHS, VOLATILE, comparable

D, B, TAG = sys.argv[1:]
G = f"/scratch/yw8866/mmgen_runs/geo/gate/r24_base/{D}"
LRV = {"lr3e4": 3e-4, "lr1e3": 1e-3, "lr2e3": 2e-3, "lr3e3": 3e-3, "lr4e3": 4e-3}
toks = TAG.split("_")
ok_keys = {"resolved_args.learning_rate", "training_args.learning_rate"} if any(t in LRV for t in toks) else set()
if "b256" in toks:
    ok_keys |= {"resolved_args.per_device_batch_size", "training_args.per_device_train_batch_size", "training_args.per_device_eval_batch_size"}
ref = json.load(open(f"{G}/{B}_ref/dryrun/dryrun.json"))
new = json.load(open(f"{G}/{B}_{TAG}/dryrun/dryrun.json"))
cls = {"allowed": {}, "paths": {}, "volatile": {}, "UNINTENDED": {}}
for k, v in diff(comparable(ref), comparable(new)).items():
    last = k.split(".")[-1]
    cls["paths" if last in PATHS else "volatile" if last in VOLATILE else "allowed" if k in ok_keys else "UNINTENDED"][k] = v
sm = json.load(open(f"{G}/{B}_{TAG}/smoke/seed42/smoke.json"))
lr = next((LRV[t] for t in toks if t in LRV), ref["training_args"]["learning_rate"])
bs = 256 if "b256" in toks else ref["training_args"]["per_device_train_batch_size"]
sm_ok = sm["ok"] and abs(sm["learning_rate"] - lr) < 1e-12 and sm["per_device_train_batch_size"] == bs and sm["gradient_accumulation_steps"] == 4
new_ok = new["training_args"]["learning_rate"] == lr and new["training_args"]["per_device_train_batch_size"] == bs
missing = [] if TAG == "ref" else sorted(k for k in ok_keys if k.startswith("training_args.learning") and lr != ref["training_args"]["learning_rate"] and k not in cls["allowed"])
out = {"ok": not cls["UNINTENDED"] and sm_ok and new_ok and not missing, "smoke_ok": sm_ok, "dry_settings_ok": new_ok, "missing": missing,
       "smoke": {k: sm[k] for k in ("learning_rate", "per_device_train_batch_size", "steps", "losses", "max_steps")}, **cls}
json.dump(out, open(f"{G}/diff_{B}_{TAG}.json", "w"), indent=1)
print(f"{D} {B} {TAG}: {'OK' if out['ok'] else 'NOT OK'} lr {sm['learning_rate']} batch {bs} x 4 smoke {sm_ok} UNINTENDED {json.dumps(cls['UNINTENDED'])} "
      f"allowed {sorted(cls['allowed'])} losses {[round(x, 3) for x in sm['losses'][:6]]}")
sys.exit(0 if out["ok"] else 1)
