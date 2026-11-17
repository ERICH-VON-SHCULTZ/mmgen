"""mmgen addition (docs/MFD_REPORT.md §22.5, E5): coarse item partitions ("regions") for the region-mass score. Per dataset:
  oracle_ctext / oracle_cimage   C (MQL4GRec) text / image SID level-1 code (diagnostic only, never a method)
  mdlat_G{32,64,128}             (a) k-means on the MD-RQ-VAE pre-quantization fused latent z_f (locked v1 1+2+2 tokenizer)
  behav_G{32,64}                 (b) groups of the 256 f codes from training histories only: f -> next-f transition counts over
                                 consecutive pairs inside items[:-2] of every user (the training part of the leave-one-out split),
                                 + 0.01 per cell, row-normalised, square root (Hellinger), k-means on the rows; region = group of f
  rawtext_G{32,64}               (d) "uses raw features": k-means on the item text features the tokenizer consumes (same prep)
Fitted on catalog items / training sequences only (no validation / test interactions). k-means: sklearn, n_init 10, seed 0.
(c) A01's learned f embeddings is per checkpoint and built in r22_e5_eval.py.
Out: geo/r22/<D>/e5_regions.pt {name: LongTensor(n_items) region id by item id} and e5_regions.json (G, sizes, NMI vs C-text /
C-image level 1 on catalog items).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_index"))
from generate_indices_md import load  # noqa: E402
from main_mdrqvae import prep  # noqa: E402

MD = ".index_md122_a05_ep500_reassign.json"


def km(x, g):
    return KMeans(g, n_init=10, random_state=0).fit_predict(np.asarray(x, dtype=np.float64))


def first_codes(path, n):
    idx = json.load(open(path))
    assert len(idx) == n and all(int(k) < n for k in idx)
    toks = [idx[str(i)][0] for i in range(n)]
    u = {t: j for j, t in enumerate(sorted(set(toks)))}
    return np.array([u[t] for t in toks])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    d = f"{a.data_path}/{a.dataset}/{a.dataset}"
    n = len(json.load(open(d + MD)))
    f = first_codes(d + MD, n)
    reg = {"oracle_ctext": first_codes(d + ".index_lemb_mql.json", n), "oracle_cimage": first_codes(d + ".index_vitemb_mql.json", n)}
    # (a) MD latent z_f
    m, c, _ = load(a.tokenizer_ckpt)
    xt = prep(np.load(f"{d}.emb-llama-td.npy"), c["input_mode"])
    xi = prep(np.load(f"{d}.emb-ViT-L-14.npy"), c["input_mode"])
    assert xt.shape[0] == n
    zf = torch.cat([m.encode(xt[s:s + 4096], xi[s:s + 4096])["z_f"] for s in range(0, n, 4096)]).numpy()
    for g in (32, 64, 128):
        reg[f"mdlat_G{g}"] = km(zf, g)
    # (b) behavioural f groups from training histories
    nF = f.max() + 1
    T = np.zeros((nF, nF))
    for items in json.load(open(d + ".inter.json")).values():
        tr = items[:-2]
        for x, y in zip(tr[:-1], tr[1:]):
            T[f[x], f[y]] += 1
    H = np.sqrt((T + 0.01) / (T + 0.01).sum(1, keepdims=True))
    for g in (32, 64):
        reg[f"behav_G{g}"] = km(H, g)[f]
    # (d) raw text features
    for g in (32, 64):
        reg[f"rawtext_G{g}"] = km(xt.numpy(), g)
    info = {"items": n, "f_codes": int(nF), "train_transitions": int(T.sum())}
    for k, r in reg.items():
        info[k] = {"G": int(r.max() + 1), "size_median": float(np.median(np.bincount(r))), "size_max": int(np.bincount(r).max()),
                   "nmi_ctext": float(nmi(reg["oracle_ctext"], r)), "nmi_cimage": float(nmi(reg["oracle_cimage"], r)),
                   "nmi_f": float(nmi(f, r))}
        print(k, json.dumps(info[k]), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.save({k: torch.from_numpy(np.asarray(v)).long() for k, v in reg.items()}, f"{a.out_dir}/e5_regions.pt")
    json.dump(info, open(f"{a.out_dir}/e5_regions.json", "w"), indent=1)


if __name__ == "__main__":
    main()
