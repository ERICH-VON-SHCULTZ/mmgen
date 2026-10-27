"""mmgen addition: MD-RQ-VAE SIDs -> MACRec index JSON for the chain decoder.

Token layout (one sequence, serialized [shared, text chain, image chain]):
  shared levels  <s{j}_k>   j = 0..S-1
  text chain     <t{j}_k>   j = 0..L-1
  image chain    <v{j}_k>   j = 0..L-1
  collision code <x_k>      appended only when --dedup append and the full SID is not unique:
                            the item's index within its group of identical SIDs (0 for the first),
                            as in TIGER (A/B/E). The shared code is never changed.
Writes <D>.index_<tag>.json, <out_dir>/codes.npy (N, S+2L, before dedup) and <out_dir>/sid_stats.json.
"""
import argparse
import collections
import json
import os

import numpy as np
import torch

from main_mdrqvae import prep
from models.mdrqvae import MDRQVAE


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = MDRQVAE(c["text_dim"], c["image_dim"], n_shared=c["n_shared"], n_modal=c["n_modal"],
                codebook_size=c["codebook_size"], e_dim=c["e_dim"], layers=c["layers"], fusion=c["fusion"],
                info_nce_weight=c["info_nce_weight"], quant_loss_weight=c["quant_loss_weight"])
    for rq in (m.shared_rq, m.text_rq, m.image_rq):
        for vq in rq.vq_layers:
            vq.initted = True
    m.load_state_dict(ck["state_dict"])
    return m.eval(), c, ck.get("epoch")


@torch.no_grad()
def compute_codes(model, c, device="cuda"):
    d = f"{c['data_root']}/{c['dataset']}/{c['dataset']}"
    xt = prep(np.load(f"{d}.emb-llama-td.npy"), c["input_mode"]).to(device)
    xi = prep(np.load(f"{d}.emb-ViT-L-14.npy"), c["input_mode"]).to(device)
    model = model.to(device)
    out = [model.encode(xt[s:s + 4096], xi[s:s + 4096])["codes"] for s in range(0, xt.shape[0], 4096)]
    return torch.cat(out).cpu().numpy(), xt, xi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True, help="index file tag, e.g. md122")
    p.add_argument("--dedup", default="append", choices=["append", "none"])
    a = p.parse_args()
    model, c, epoch = load(a.ckpt)
    codes, _, _ = compute_codes(model, c)
    out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    np.save(f"{out_dir}/codes.npy", codes)
    S, L = c["n_shared"], c["n_modal"]
    names = [f"s{j}" for j in range(S)] + [f"t{j}" for j in range(L)] + [f"v{j}" for j in range(L)]
    groups = collections.Counter(tuple(r) for r in codes.tolist())
    seen = collections.Counter()
    index = {}
    for item, r in enumerate(codes.tolist()):
        toks = [f"<{n}_{k}>" for n, k in zip(names, r)]
        if a.dedup == "append" and len(groups) < len(codes):
            toks.append(f"<x_{seen[tuple(r)]}>")
            seen[tuple(r)] += 1
        index[str(item)] = toks
    N = len(codes)
    stats = {"ckpt": a.ckpt, "epoch": epoch, "items": N, "S": S, "L": L, "sid_len": len(index["0"]),
             "full_sid_collision_before": (N - len(groups)) / N, "max_group": max(groups.values()),
             "dedup": a.dedup, "unique_final": len({"".join(v) for v in index.values()}),
             "utilization_per_level": dict(zip(names, [int(len(np.unique(codes[:, j]))) for j in range(codes.shape[1])]))}
    stats["all_unique"] = stats["unique_final"] == N
    path = f"{c['data_root']}/{c['dataset']}/{c['dataset']}.index_{a.tag}.json"
    json.dump(index, open(path, "w"), indent=1)
    json.dump(stats, open(f"{out_dir}/sid_stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
