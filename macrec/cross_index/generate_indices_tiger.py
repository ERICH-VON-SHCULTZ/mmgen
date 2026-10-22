"""TIGER-style SID assignment (mmgen addition, baselines A/B).

Loads a single-modality RQVAE checkpoint trained by main_single.py (e.g. 3 levels x 256), takes the
argmin code at every level (no Sinkhorn), and appends ONE extra token: the item's index within its
group of identical code tuples (0 for the first item of a tuple, 1 for the next, ...), as in
TIGER (Rajput et al., 2023). No codes are re-assigned. Output format is MACRec's index JSON
({item_idx: ["<a_12>", "<b_3>", "<c_200>", "<d_0>"]}), so data.py / test_ddp_save.py consume it
unchanged.
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--content", default="text", choices=["text", "image"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stats_file", default=None)
    a = p.parse_args()

    ckpt = torch.load(a.ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
                  layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
                  loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(a.device).eval()

    codes = []
    for d in DataLoader(data, batch_size=1024, shuffle=False):
        idx, _ = model.get_indices(d.to(a.device), use_sk=False)
        codes.append(idx.view(-1, idx.shape[-1]).cpu().numpy())
    codes = np.concatenate(codes).astype(int)                     # (N, L) in item-id order
    N, L = codes.shape

    prefix = (["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>", "<E_{}>"] if a.content == "image"
              else ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"])
    seen = collections.Counter()
    index = {}
    for item in range(N):
        t = tuple(codes[item].tolist())
        dup = seen[t]
        seen[t] += 1
        index[str(item)] = [prefix[l].format(c) for l, c in enumerate(t)] + [prefix[L].format(dup)]

    n_unique_prefix = len(seen)
    stats = {
        "items": N, "levels_rq": L,
        "collision_rate_before": (N - n_unique_prefix) / N,          # same definition as MACRec's logs
        "items_in_collision": int(sum(c for c in seen.values() if c > 1)),
        "max_group": int(max(seen.values())),                       # = vocab needed by the extra level
        "unique_final": len({"".join(v) for v in index.values()}),
        "utilization_per_level": [int(len(np.unique(codes[:, l]))) for l in range(L)],
        "codebook_sizes": list(args.num_emb_list),
    }
    stats["all_unique"] = stats["unique_final"] == N
    assert stats["all_unique"]
    os.makedirs(os.path.dirname(os.path.abspath(a.output_file)), exist_ok=True)
    with open(a.output_file, "w") as f:
        json.dump(index, f, indent=4)
    print(json.dumps(stats))
    if a.stats_file:
        with open(a.stats_file, "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
