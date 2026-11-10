"""mmgen addition (docs/MFD_REPORT.md §21): P1a training (finetune_geo.py --mfd --geo codebook --item_enc, unchanged) with
one of the r21_views.py methods swapped in:
  --r21_mode G0|A|B   grouped views (GroupedMVDataset + GroupedCollator, model r21_views.GroupedItemMFDT5)
  --r21_beta BETA     A / B: weight of the KL term
  --r21_mode C        item-level random views (ItemRandViewDataset), --r21_p_full P; model and loss = P1a
  --dump_file PATH    dry run: resolved config dumped at trainer.train() (config_dryrun), no training
Every other argument is finetune_geo.py's (the exact P1a command; batch / accumulation set by the caller to match
compute: grouped runs use 341 pairs x 3 views = 1,023 sequences per optimizer step, accumulation 1).
After training: best checkpoint baked and saved as P1a's (finetune_geo.bake_and_save), stats get the r21 settings and
optimizer steps per epoch.
"""
import json
import os
import runpy
import sys

from torch.utils.data import ConcatDataset


def pop(name, default=None):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


MODE = pop("--r21_mode")
BETA = float(pop("--r21_beta", 0.0))
P_FULL = float(pop("--r21_p_full", 0.0))
assert MODE in ("G0", "A", "B", "C"), MODE
assert (MODE in ("A", "B")) == (BETA > 0) and (MODE == "C") == (P_FULL > 0)
assert "--item_enc" in sys.argv and "--mfd" in sys.argv and "--dense_lambda" not in sys.argv, "r21 runs are P1a + one change"

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if "--dump_file" in sys.argv:
        import config_dryrun
        config_dryrun.install()
    import enc_struct
    import finetune_ext
    import mfd_data
    import r21_views
    grouped = MODE in ("G0", "A", "B")
    if grouped:
        enc_struct.ItemMFDT5 = r21_views.GroupedItemMFDT5          # finetune_geo.setup() reads it at call time
        mfd_data.MFDCollator = r21_views.GroupedCollator            # finetune_mfd.py imports it at run time

    def load_r21(args):
        train = r21_views.GroupedMVDataset(args, "train") if grouped else r21_views.ItemRandViewDataset(args, "train", P_FULL)
        return ConcatDataset([train]), ConcatDataset([mfd_data.MFDDataset(args, "valid")])
    mfd_data.load_mfd_datasets = load_r21

    import finetune_geo                                             # pops its own arguments at import
    state = finetune_geo.setup()
    info = {"r21_mode": MODE, "r21_beta": BETA, "r21_p_full": P_FULL}
    _Geo = finetune_ext.StatsTrainer

    class R21Trainer(_Geo):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if grouped:
                info.update(k["model"].install_r21(k["tokenizer"], MODE, BETA))
            print("[finetune_r21]", json.dumps(info), flush=True)

    finetune_ext.StatsTrainer = R21Trainer
    stats = sys.argv[sys.argv.index("--stats_file") + 1] if "--stats_file" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--output_dir") + 1]
    runpy.run_path(os.path.join(here, "finetune_mfd.py"), run_name="__main__")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and "model" in state:
        diff = finetune_geo.bake_and_save(state, out_dir)
        if stats:
            s = json.load(open(stats))
            s["geo"] = state["info"] | {"bake_max_logit_diff": diff}
            s["r21"] = info | {"steps_per_epoch": s["global_steps"] / s["epochs_run"],
                               "encoder_sequences_per_epoch": 3 * s["train_examples_per_epoch"] if grouped else s["train_examples_per_epoch"]}
            json.dump(s, open(stats, "w"), indent=1)
