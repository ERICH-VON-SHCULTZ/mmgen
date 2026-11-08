"""mmgen addition: codebook-geometry-aware training (docs/MFD_REPORT.md §14, macrec/geo_embed.py).

Same path as the controls: finetune_ext.main() (chain decoder, as run_baseline.sh md*) or finetune_mfd.py (MFD, as
run_mfd_train.sh), unchanged arguments, plus:
  --geo none|codebook|shuffled   A (codebook-initialised code embeddings) or C (shuffled control); none = off
  --geo_ckpt PATH                MD-RQ-VAE tokenizer checkpoint that produced the SID (v1: md/sids/v1/<D>/1+2+2/tokenizer.pth)
  --soft_f_eps EPS               B: soft labels for f (0 = one-hot, the control's loss)
  --geo_perm_seed S              permutation seed of the shuffled control (default 0)
  --soft_f_mode geometric|uniform   B's targets: geometric (codebook distances) or plain label smoothing on f (§15 V3)
  --soft_f_nn_mass M             geometric targets with the T that puts share M of the eps mass on the 10 nearest f (§15 V4)
  --f_loss_weight W              §17: f cross-entropy x W in training (validation loss unchanged)
  --offset_lambda L              §17: + L * mean ||delta||^2 in training
  --item_enc                     §17 P1a (with --mfd): one encoder position per history item (enc_struct.ItemMFDT5)
  --dense_lambda L               §20 Task 2 (with --item_enc): + L * dense item-level CE (dense_head.DenseItemMFDT5), training only
  --geo_sid md|c                 c = baseline C's tokens (<a_k>.. / <A_k>..), §17 Part 3; --geo_ckpt = text tokenizer,
  --geo_ckpt_image PATH          C's image tokenizer
  --mfd                          run finetune_mfd.py's path (MFDT5, MFD datasets / collator) instead of the chain's
  --geo_report PATH              JSON written at install: codebook tensor, tied head, norms, T, ...
The model class becomes geo_embed.make_geo_class(PlainT5 | MFDT5); the geometry is installed when the Trainer is
built (after resize_token_embeddings, before the optimizer exists). After training (best checkpoint loaded) the model
is baked to a plain tied T5 and saved over output_dir, so evaluation loads it like any control.
"""
import json
import os
import runpy
import sys

import torch


def pop_arg(name, default=None, flag=False):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if flag:
        del sys.argv[i]
        return True
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


GEO = pop_arg("--geo", "none")
CKPT = pop_arg("--geo_ckpt")
EPS = float(pop_arg("--soft_f_eps", 0.0))
PERM_SEED = int(pop_arg("--geo_perm_seed", 0))
SOFT_MODE = pop_arg("--soft_f_mode", "geometric")
NN_MASS = pop_arg("--soft_f_nn_mass")
NN_MASS = float(NN_MASS) if NN_MASS is not None else None
F_WEIGHT = float(pop_arg("--f_loss_weight", 1.0))
OFFSET = float(pop_arg("--offset_lambda", 0.0))
ITEM = pop_arg("--item_enc", False, flag=True)
DENSE = float(pop_arg("--dense_lambda", 0.0))
SID = pop_arg("--geo_sid", "md")
CKPT_IMAGE = pop_arg("--geo_ckpt_image")
MFD = pop_arg("--mfd", False, flag=True)
REPORT = pop_arg("--geo_report")
assert DENSE == 0 or ITEM, "--dense_lambda needs --item_enc"
assert GEO in ("none", "codebook", "shuffled") and (GEO != "none" or EPS > 0) and CKPT and SOFT_MODE in ("geometric", "uniform")


def setup():
    """Patch finetune_ext so that its model is the Geo class and the Trainer installs the geometry."""
    import finetune_ext
    import geo_embed
    if MFD:                                     # finetune_mfd.py does `from mfd_model import MFDT5; finetune_ext.PlainT5 = MFDT5`
        import mfd_model
        if ITEM and DENSE > 0:
            import dense_head
            mfd_model.MFDT5 = geo_embed.make_geo_class(dense_head.DenseItemMFDT5)
        elif ITEM:
            import enc_struct
            mfd_model.MFDT5 = geo_embed.make_geo_class(enc_struct.ItemMFDT5)
        else:
            mfd_model.MFDT5 = geo_embed.make_geo_class(mfd_model.MFDT5)
    else:
        finetune_ext.PlainT5 = geo_embed.make_geo_class(finetune_ext.PlainT5)
    state = {}
    _Base = finetune_ext.StatsTrainer

    class GeoTrainer(_Base):
        def __init__(self, *a, **k):
            model, tok = k["model"], k["tokenizer"]
            info = model.install_geometry(tok, CKPT, GEO, EPS, PERM_SEED, SOFT_MODE, NN_MASS, F_WEIGHT, OFFSET, SID, CKPT_IMAGE)
            if ITEM:
                info["item_enc"] = model.install_struct(tok, "E2")
            if DENSE > 0:
                import argparse
                import mfd_data
                data_ns = argparse.Namespace(**{k: sys.argv[sys.argv.index("--" + k) + 1] for k in ("data_path", "dataset", "index_file")})
                info["dense"] = model.install_dense(mfd_data.item_token_table(data_ns, tok), DENSE)
            info["n_params"] = sum(p.numel() for p in model.parameters())
            state["model"], state["info"] = model, info
            if REPORT and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                json.dump(info, open(REPORT, "w"), indent=1)
            print("[finetune_geo]", json.dumps(info))
            super().__init__(*a, **k)

    finetune_ext.StatsTrainer = GeoTrainer
    return state


def bake_and_save(state, out_dir):
    """Replace the geometry by a plain embedding table; check identical logits; save over out_dir."""
    m = state["model"]
    m.eval()
    dev = next(m.parameters()).device
    V = m.config.vocab_size
    x = torch.randint(32100, V, (4, 12), device=dev)               # SID code tokens (+ the few other added tokens)
    if getattr(m, "struct_mode", None):                            # item-level encoder: items start at <s0_..>
        s0 = (m.struct.slot_of == 0).nonzero().squeeze(1)
        x[:, ::4] = s0[torch.randint(len(s0), x[:, ::4].shape, device=dev)]
    y = torch.randint(32100, V, (4, 3), device=dev)
    y = y.unsqueeze(1).repeat(1, 2, 1) if MFD else y
    with torch.no_grad():
        before = m(input_ids=x, attention_mask=torch.ones_like(x), decoder_input_ids=y).logits
        m.bake()
        after = m(input_ids=x, attention_mask=torch.ones_like(x), decoder_input_ids=y).logits
    diff = (before - after).abs().max().item()
    assert diff < 1e-4, f"bake changed the logits by {diff}"
    m.__class__ = type(m).__mro__[2]             # back to the control's class (PlainT5 / MFDT5): same saved layout
    m.save_pretrained(out_dir)
    print(f"[finetune_geo] baked (max logit diff {diff:.2e}) and saved to {out_dir}")
    return diff


if __name__ == "__main__":
    state = setup()
    stats = sys.argv[sys.argv.index("--stats_file") + 1] if "--stats_file" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--output_dir") + 1]
    if MFD:
        runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_mfd.py"), run_name="__main__")
    else:
        import finetune_ext
        finetune_ext.main()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and "model" in state:
        diff = bake_and_save(state, out_dir)
        if stats:
            s = json.load(open(stats))
            s["geo"] = state["info"] | {"bake_max_logit_diff": diff}
            json.dump(s, open(stats, "w"), indent=1)
