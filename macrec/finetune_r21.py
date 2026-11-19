"""mmgen addition (docs/MFD_REPORT.md §21): P1a training (finetune_geo.py --mfd --geo codebook --item_enc, unchanged) with
one of the r21_views.py methods swapped in:
  --r21_mode G0|A|B   grouped views (GroupedMVDataset + GroupedCollator, model r21_views.GroupedItemMFDT5)
  --r21_beta BETA     A / B: weight of the KL term
  --r21_mode C        item-level random views (ItemRandViewDataset), --r21_p_full P; model and loss = P1a
  --r21_mode AC       §21.4: grouped [full, random, random] rows (GroupedRandDataset, --r21_p_full P in [0, 1)) + A's KL (--r21_beta)
  --r21_mode E10c     §22.10 E10c: A01's teacher KL WITHOUT grouping, on P1a's exact data layout / optimisation path
                      (E10cDataset = MVMFDDataset, 256 rows x 4 micro-batches as P1a; KL on full-view rows only, weight 3 x --r21_beta,
                      two extra no-grad forwards for p_text / p_image; r22_e10c.py). --r21_beta >= 0 (0 only for the gate)
  --r22_aux_gamma G   §22 E2 (grouped modes only): + G x [CE(t1) + CE(i1)] from aux heads at the f position (r22_aux.py)
  --r22_e7 a|b        §22.7 E7 (grouped modes only): zero-initialised linear map of the item's f / text / image code sums (a) or of
                      its f embedding only (b) added to the item vector (r22_e7.py)
  --r22_e8 a|b        §22.8 E8 (grouped modes only): f embedding E'(f) = E(f) + W c(f), c(f) = mean child-code embedding sum
                      under f in the SID tree (a: t1 t2 i1 i2, b: t1 i1), W zero-initialised (r22_e8.py; folded into the table at bake)
  --r22_e9_k K        §22.9 E9 (grouped modes only): mixture of K softmaxes at the f step (r22_e9.py)
  --smoke_steps N     gate: run N real optimizer steps of the exact command (data loading, loss, backward, optimizer, clipping),
                      write <output_dir>/smoke.json (per-step loss + grad norm, E7 weight norm) and exit; no evaluation, no save
  --dump_file PATH    dry run: resolved config dumped at trainer.train() (config_dryrun), no training
Every other argument is finetune_geo.py's (the exact P1a command; batch / accumulation set by the caller to match
compute: grouped runs use 341 pairs x 3 views = 1,023 sequences per optimizer step, accumulation 1).
After training: best checkpoint baked and saved as P1a's (finetune_geo.bake_and_save), stats get the r21 settings and
optimizer steps per epoch.
Every run (also C) logs the gradient norm of EVERY optimizer step (the value accelerate's clip_grad_norm_ returns, before
clipping at max_grad_norm) -> <output_dir>/grad_norms.json and stats r21.clip (steps, clipped share, quantiles).
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
AUX = float(pop("--r22_aux_gamma", 0.0))
E7 = pop("--r22_e7")
E8 = pop("--r22_e8")
E9 = int(pop("--r22_e9_k", 0))
SMOKE = int(pop("--smoke_steps", 0))
BETA = float(pop("--r21_beta", 0.0))
P_FULL = pop("--r21_p_full")
P_FULL = None if P_FULL is None else float(P_FULL)
assert MODE in ("G0", "A", "B", "C", "AC", "E10c"), MODE
assert (MODE in ("A", "B", "AC")) == (BETA > 0) or (MODE == "E10c" and BETA >= 0)
assert (MODE == "C" and P_FULL is not None and 0 < P_FULL < 1) or (MODE == "AC" and P_FULL is not None and 0 <= P_FULL < 1) \
    or (MODE in ("G0", "A", "B", "E10c") and P_FULL is None)
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
    grouped = MODE in ("G0", "A", "B", "AC")
    model_mode = "A" if MODE == "AC" else MODE                     # AC = A's KL on [full, random, random] rows
    assert AUX == 0 or grouped, "--r22_aux_gamma needs a grouped mode"
    assert E7 is None or (grouped and AUX == 0 and E7 in ("a", "b")), "--r22_e7 a|b needs a grouped mode, no aux heads"
    if grouped:
        enc_struct.ItemMFDT5 = r21_views.GroupedItemMFDT5          # finetune_geo.setup() reads it at call time
        mfd_data.MFDCollator = r21_views.GroupedCollator            # finetune_mfd.py imports it at run time
    if MODE == "E10c":
        import r22_e10c
        enc_struct.ItemMFDT5 = r22_e10c.E10cItemMFDT5
        mfd_data.MFDCollator = r22_e10c.E10cCollator
    assert MODE != "E10c" or not (AUX or E7 or E8 or E9), "E10c is P1a + the KL only"
    if AUX > 0:
        import r22_aux
        enc_struct.ItemMFDT5 = r22_aux.AuxGroupedItemMFDT5
    assert sum(map(bool, (AUX > 0, E7, E8, E9))) <= 1, "one round-22 change at a time"
    assert (E8 is None or (grouped and E8 in ("a", "b"))) and (E9 == 0 or (grouped and E9 >= 2)), "--r22_e8 / --r22_e9_k need a grouped mode"
    if E7:
        import r22_e7
        enc_struct.ItemMFDT5 = r22_e7.E7GroupedItemMFDT5
    if E8:
        import r22_e8
        enc_struct.ItemMFDT5 = r22_e8.E8GroupedItemMFDT5
    if E9:
        import r22_e9
        enc_struct.ItemMFDT5 = r22_e9.E9GroupedItemMFDT5

    def load_r21(args):
        if MODE == "AC":
            train = r21_views.GroupedRandDataset(args, "train", P_FULL)
        elif grouped:
            train = r21_views.GroupedMVDataset(args, "train")
        elif MODE == "E10c":
            train = r22_e10c.E10cDataset(args, "train")
        else:
            train = r21_views.ItemRandViewDataset(args, "train", P_FULL)
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
                info.update(k["model"].install_r21(k["tokenizer"], model_mode, BETA))
            if MODE == "E10c":
                info.update(k["model"].install_e10c(k["tokenizer"], BETA))
            if AUX > 0:
                info.update(k["model"].install_aux(k["tokenizer"], AUX))
            if E7:
                info.update(k["model"].install_e7(E7))
            if E8:
                arg = lambda n: sys.argv[sys.argv.index("--" + n) + 1]
                info["e8"] = k["model"].install_e8(k["tokenizer"], os.path.join(arg("data_path"), arg("dataset"), arg("dataset") + arg("index_file")), E8)
            if E9:
                info["e9"] = k["model"].install_e9(E9)
            b = self.data_collator([self.train_dataset[0]])             # collator / dataset mismatch fails here, also in
            assert not grouped or b["input_ids"].shape[0] == 3, b["input_ids"].shape   # the gate's dry run (23398867)
            assert MODE != "E10c" or ("e10_rows" in b and b["e10_text_ids"].shape[0] == 1), list(b)   # row 0 is a full-view row
            clip = self.accelerator.clip_grad_norm_

            def logged_clip(params, max_norm, *ca, **ck):
                n = clip(params, max_norm, *ca, **ck)
                NORMS.append(float(n))
                CLIP["max_norm"] = float(max_norm)
                return n
            self.accelerator.clip_grad_norm_ = logged_clip
            if SMOKE:
                self.add_callback(SmokeStop(self))
            info["max_grad_norm"] = self.args.max_grad_norm
            print("[finetune_r21]", json.dumps(info), flush=True)

        def training_step(self, *a, **k):
            loss = super().training_step(*a, **k)
            if SMOKE:
                LOSSES.append(float(loss.detach().float()))
            return loss

    NORMS, CLIP = [], {}
    LOSSES = []

    from transformers import TrainerCallback

    class SmokeStop(TrainerCallback):
        """--smoke_steps: after N optimizer steps write smoke.json and exit (the gate's real-training check)."""

        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, state, control, **kw):
            if state.global_step < SMOKE:
                return
            import math
            m = self.trainer.model
            e7 = getattr(m, "e7", None)
            e8 = m.e8_w_norm() if E8 else None
            e9 = m.e9_stats() if E9 else None
            e10 = m.e10c_stats() if MODE == "E10c" else None
            res = {"steps": state.global_step, "losses": LOSSES, "grad_norms": NORMS,
                   "e7_weight_norm": None if e7 is None else float(e7.weight.norm()), "e8_weight_norm": e8, "e9": e9, "e10c": e10,
                   "r21_mode": MODE, "r21_beta": BETA, "per_device_train_batch_size": args.per_device_train_batch_size,
                   "gradient_accumulation_steps": args.gradient_accumulation_steps,
                   "trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad)}
            res["ok"] = (len(LOSSES) >= SMOKE and all(math.isfinite(x) for x in LOSSES + NORMS)
                         and (e7 is None or res["e7_weight_norm"] > 0) and (not E8 or e8 > 0)
                         and (not E9 or e9["W_pi_norm"] > 0)
                         and (e10 is None or (e10["full_row_share"] > 0.2 and math.isfinite(e10["mean_kl_full_rows"])
                                              and (BETA == 0 or e10["mean_term"] > 0))))   # the KL really ran on the full rows
            json.dump(res, open(os.path.join(args.output_dir, "smoke.json"), "w"), indent=1)
            print("[finetune_r21] smoke", json.dumps(res), flush=True)
            os._exit(0 if res["ok"] else 4)

    finetune_ext.StatsTrainer = R21Trainer
    stats = sys.argv[sys.argv.index("--stats_file") + 1] if "--stats_file" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--output_dir") + 1]
    runpy.run_path(os.path.join(here, "finetune_mfd.py"), run_name="__main__")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and "model" in state:
        if E8:
            info["e8"]["w_norm_final"] = state["model"].e8_w_norm()    # W is folded into the table by the bake
        if E9:
            info["e9"]["final"] = state["model"].e9_stats()
        if MODE == "E10c":
            info["e10c"] = state["model"].e10c_stats()
        diff = finetune_geo.bake_and_save(state, out_dir)
        if NORMS:
            json.dump(NORMS, open(os.path.join(out_dir, "grad_norms.json"), "w"))
            srt = sorted(NORMS)
            q = lambda x: srt[min(len(srt) - 1, int(x * len(srt)))]
            info["clip"] = {"steps": len(NORMS), "max_grad_norm": CLIP.get("max_norm"),
                            "clipped_share": sum(n > CLIP.get("max_norm", float("inf")) for n in NORMS) / len(NORMS),
                            "norm_min": srt[0], "norm_median": q(0.5), "norm_p10": q(0.1), "norm_p90": q(0.9), "norm_max": srt[-1]}
            print("[finetune_r21] clipping", json.dumps(info["clip"]), flush=True)
        if stats:
            s = json.load(open(stats))
            s["geo"] = state["info"] | {"bake_max_logit_diff": diff}
            s["r21"] = info | {"steps_per_epoch": s["global_steps"] / s["epochs_run"],
                               "encoder_sequences_per_epoch": 3 * s["train_examples_per_epoch"] if grouped else s["train_examples_per_epoch"]}
            json.dump(s, open(stats, "w"), indent=1)
