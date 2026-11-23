"""mmgen addition (docs/MFD_REPORT.md §24): P1a training (finetune_geo.py --mfd --geo codebook --item_enc, P1a's exact command)
with round-24 changes:
  Track A (tuning / capacity): learning rate and weight decay are P1a's own flags; dropout, d_model / d_ff / heads and layer counts
    come from --base_model (a copy of config/ckpt with the edited T5 config, made by sbatch/r24/run_r24_train.sh);
    --r24_max_grad_norm X   gradient clipping threshold (P1a: the HF default 1.0)
  Track B (item-level semantic tasks, macrec/r24_items.py):
    --r24_item W            add one B-text and one B-image example per catalog item per epoch (input = one item position
                            without E(f), target = the full SID), loss weight W (B1 1.0, B2 0.3)
  --smoke_steps N           gate: N real optimizer steps of the exact command -> <output_dir>/smoke.json, then exit
  --dump_file PATH          dry run (config_dryrun): resolved config dumped at trainer.train(), no training
Every run logs the pre-clipping gradient norm of every optimizer step (grad_norms.json, stats r24.clip) as round 21.
After training the best checkpoint is baked and saved in P1a's layout (finetune_geo.bake_and_save).
"""
import json
import math
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


GN = pop("--r24_max_grad_norm")
GN = None if GN is None else float(GN)
ITEM_W = pop("--r24_item")
ITEM_W = None if ITEM_W is None else float(ITEM_W)
SMOKE = int(pop("--smoke_steps", 0))
assert ITEM_W is None or ITEM_W > 0
assert "--item_enc" in sys.argv and "--mfd" in sys.argv and "--dense_lambda" not in sys.argv, "r24 runs are P1a + changes"

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if "--dump_file" in sys.argv:
        import config_dryrun
        config_dryrun.install()
    import transformers

    import finetune_ext
    import mfd_data
    if GN is not None:                                       # finetune_contrastive builds transformers.TrainingArguments(...)
        _TA = transformers.TrainingArguments

        def _ta(*a, **k):
            k["max_grad_norm"] = GN
            return _TA(*a, **k)
        transformers.TrainingArguments = _ta
    if ITEM_W is not None:
        import r24_items
        r24_items.install()
        _load = mfd_data.load_mfd_datasets

        def load_r24(args):
            train, valid = _load(args)
            assert len(train.datasets) == 1, "P1a's training data is one MVMFDDataset"
            return ConcatDataset([train.datasets[0], r24_items.ItemSemDataset(args, ITEM_W)]), valid
        mfd_data.load_mfd_datasets = load_r24

    import finetune_geo                                     # pops its own arguments at import
    state = finetune_geo.setup()
    info = {"r24_max_grad_norm": GN, "r24_item_weight": ITEM_W}
    NORMS, CLIP, LOSSES = [], {}, []
    _Geo = finetune_ext.StatsTrainer

    class R24Trainer(_Geo):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            m = k["model"]
            info["model_class"] = type(m).__name__
            info["t5"] = {x: getattr(m.config, x) for x in ("d_model", "d_ff", "d_kv", "num_heads", "num_layers",
                                                              "num_decoder_layers", "dropout_rate")}
            info["n_params"] = sum(p.numel() for p in m.parameters())
            info["max_grad_norm"] = self.args.max_grad_norm
            info["learning_rate"], info["weight_decay"] = self.args.learning_rate, self.args.weight_decay
            b = self.data_collator([self.train_dataset[0], self.train_dataset[len(self.train_dataset) - 1]])
            if ITEM_W is not None:                          # last example = an item example (ConcatDataset order)
                assert "example_weight" in b and float(b["example_weight"][-1]) == ITEM_W, b.get("example_weight")
                info["item_examples_per_epoch"] = len(self.train_dataset.datasets[1])
                info["sequence_examples_per_epoch"] = len(self.train_dataset.datasets[0])
            clip = self.accelerator.clip_grad_norm_

            def logged_clip(params, max_norm, *ca, **ck):
                n = clip(params, max_norm, *ca, **ck)
                NORMS.append(float(n))
                CLIP["max_norm"] = float(max_norm)
                return n
            self.accelerator.clip_grad_norm_ = logged_clip
            if SMOKE:
                self.add_callback(SmokeStop(self))
            print("[finetune_r24]", json.dumps(info), flush=True)

        def training_step(self, *a, **k):
            loss = super().training_step(*a, **k)
            if SMOKE:
                LOSSES.append(float(loss.detach().float()))
            return loss

    from transformers import TrainerCallback

    class SmokeStop(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, st, control, **kw):
            if st.global_step < SMOKE:
                return
            res = {"steps": st.global_step, "losses": LOSSES, "grad_norms": NORMS, "info": info,
                   "per_device_train_batch_size": args.per_device_train_batch_size,
                   "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_grad_norm": args.max_grad_norm,
                   "learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
            res["ok"] = len(LOSSES) >= SMOKE and all(math.isfinite(x) for x in LOSSES + NORMS) and (GN is None or args.max_grad_norm == GN)
            json.dump(res, open(os.path.join(args.output_dir, "smoke.json"), "w"), indent=1)
            print("[finetune_r24] smoke", json.dumps(res), flush=True)
            os._exit(0 if res["ok"] else 4)

    finetune_ext.StatsTrainer = R24Trainer
    stats = sys.argv[sys.argv.index("--stats_file") + 1] if "--stats_file" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--output_dir") + 1]
    runpy.run_path(os.path.join(here, "finetune_mfd.py"), run_name="__main__")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and "model" in state:
        diff = finetune_geo.bake_and_save(state, out_dir)
        if NORMS:
            json.dump(NORMS, open(os.path.join(out_dir, "grad_norms.json"), "w"))
            srt = sorted(NORMS)
            q = lambda x: srt[min(len(srt) - 1, int(x * len(srt)))]
            info["clip"] = {"steps": len(NORMS), "max_grad_norm": CLIP.get("max_norm"),
                            "clipped_share": sum(n > CLIP.get("max_norm", float("inf")) for n in NORMS) / len(NORMS),
                            "norm_median": q(0.5), "norm_p10": q(0.1), "norm_p90": q(0.9), "norm_max": srt[-1]}
        if stats:
            s = json.load(open(stats))
            s["geo"] = state["info"] | {"bake_max_logit_diff": diff}
            s["r24"] = info | {"steps_per_epoch": s["global_steps"] / s["epochs_run"]}
            json.dump(s, open(stats, "w"), indent=1)
