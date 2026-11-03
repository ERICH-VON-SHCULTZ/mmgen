"""mmgen addition: chain decoder with item-level encoder structure (docs/MFD_REPORT.md §16, macrec/enc_struct.py).

Same path and arguments as the chain control (finetune_ext.main(), run_baseline.sh md*), plus
  --struct E1|E2     E1 slot + recency embeddings on every history token (zero-init); E2 one position per item
  --struct_report P  JSON written at install (slot table sizes, mode)
The model class becomes enc_struct.StructT5 (a finetune_ext.PlainT5 subclass); the structure is installed when the
Trainer is built (after resize_token_embeddings, before the optimizer exists). The saved model keeps
config.struct_mode, so evaluation loads it with StructT5 (eval_struct.py).
"""
import json
import os
import sys


def pop_arg(name, default=None):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


MODE = pop_arg("--struct")
REPORT = pop_arg("--struct_report")
assert MODE in ("E1", "E2")


def setup():
    import enc_struct
    import finetune_ext
    finetune_ext.PlainT5 = enc_struct.StructT5
    _Base = finetune_ext.StatsTrainer

    class StructTrainer(_Base):
        def __init__(self, *a, **k):
            info = k["model"].install_struct(k["tokenizer"], MODE)
            info["n_params"] = sum(p.numel() for p in k["model"].parameters())
            if REPORT and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                json.dump(info, open(REPORT, "w"), indent=1)
            print("[finetune_struct]", json.dumps(info))
            super().__init__(*a, **k)

    finetune_ext.StatsTrainer = StructTrainer
    return finetune_ext


if __name__ == "__main__":
    setup().main()
