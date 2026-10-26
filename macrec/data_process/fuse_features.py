"""mmgen addition (baseline E, early-fusion TIGER): one fused vector per item from the SAME text and
image features used by A/B (<D>.emb-llama-td.npy, <D>.emb-ViT-L-14.npy), rows in item-id order.

  norm: concat(text / ||text||, image / ||image||)   -> <D>.emb-fused-norm.npy
  raw:  concat(text, image)                           -> <D>.emb-fused-raw.npy

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
    a = p.parse_args()
    d = f"{a.data_root}/{a.dataset}/{a.dataset}"
    t = np.load(f"{d}.emb-llama-td.npy").astype(np.float32)
    i = np.load(f"{d}.emb-ViT-L-14.npy").astype(np.float32)
    assert t.shape[0] == i.shape[0], (t.shape, i.shape)
    tn, im = np.linalg.norm(t, axis=1, keepdims=True), np.linalg.norm(i, axis=1, keepdims=True)
    assert (tn > 0).all() and (im > 0).all()
    np.save(f"{d}.emb-fused-norm.npy", np.concatenate([t / tn, i / im], axis=1))
    np.save(f"{d}.emb-fused-raw.npy", np.concatenate([t, i], axis=1))
    q = lambda x: {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    stats = {"items": int(t.shape[0]), "text_dim": int(t.shape[1]), "image_dim": int(i.shape[1]),
             "fused_dim": int(t.shape[1] + i.shape[1]), "text_norm": q(tn), "image_norm": q(im)}
    json.dump(stats, open(f"{d}.emb-fused.stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
