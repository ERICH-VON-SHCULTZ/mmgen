"""mmgen addition (baseline E, early-fusion TIGER): one fused vector per item from the SAME text and
image features used by A/B (<D>.emb-llama-td.npy, <D>.emb-ViT-L-14.npy), rows in item-id order.

  norm: s * concat(text / ||text||, image / ||image||)  -> <D>.emb-fused-norm.npy
        s = mean ||text|| over the dataset's items, so each half has the scale of A's text input.
        (Unscaled, s = 1, the shared RQ-VAE settings collapsed the codebooks: Instruments levels used
        [49, 30, 14] of 256, 42.9% collisions; decision 2026-10-26: rescale. --scale none keeps s = 1.)
  normj: c * concat(text / ||text||, image / ||image||) -> <D>.emb-fused-normj.npy   (E-norm, official)
        the joined unit-normed vector has length sqrt(2), so c = mean ||text|| / sqrt(2) makes the
        joined vector's length equal A's mean text length. The mean is over TRAINING items (items in
        items[:-2] of any user in <D>.inter.json), computed once and recorded (decision 2026-10-27).
  raw:  concat(text, image)                              -> <D>.emb-fused-raw.npy

Writes <D>.emb-fused.stats.json with dims and per-modality norm statistics.
Usage: python fuse_features.py --dataset Instruments --data_root /scratch/yw8866/mmgen_runs/data
"""
import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--scale", default="mean_text_norm", choices=["mean_text_norm", "none"])
    a = p.parse_args()
    d = f"{a.data_root}/{a.dataset}/{a.dataset}"
    t = np.load(f"{d}.emb-llama-td.npy").astype(np.float32)
    i = np.load(f"{d}.emb-ViT-L-14.npy").astype(np.float32)
    assert t.shape[0] == i.shape[0], (t.shape, i.shape)
    tn, im = np.linalg.norm(t, axis=1, keepdims=True), np.linalg.norm(i, axis=1, keepdims=True)
    assert (tn > 0).all() and (im > 0).all()
    s = float(tn.mean()) if a.scale == "mean_text_norm" else 1.0
    inter = json.load(open(f"{d}.inter.json"))
    train_items = sorted({it for seq in inter.values() for it in seq[:-2]})
    cj = float(tn[train_items].mean()) / np.sqrt(2.0)
    np.save(f"{d}.emb-fused-normj.npy", (cj * np.concatenate([t / tn, i / im], axis=1)).astype(np.float32))
    np.save(f"{d}.emb-fused-norm.npy", (s * np.concatenate([t / tn, i / im], axis=1)).astype(np.float32))
    np.save(f"{d}.emb-fused-raw.npy", np.concatenate([t, i], axis=1))
    q = lambda x: {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    stats = {"items": int(t.shape[0]), "text_dim": int(t.shape[1]), "image_dim": int(i.shape[1]),
             "fused_dim": int(t.shape[1] + i.shape[1]), "text_norm": q(tn), "image_norm": q(im),
             "norm_scale": a.scale, "norm_scale_value": s,
             "normj_train_items": len(train_items), "normj_mean_text_norm_train": cj * float(np.sqrt(2.0)),
             "normj_constant_c": cj}
    json.dump(stats, open(f"{d}.emb-fused.stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
