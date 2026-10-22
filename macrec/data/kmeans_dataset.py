"""mmgen addition: data/kmeans.py with the dataset and data root as arguments (the original hard-codes
dataset="Games" and relative paths). Identical computation: KMeans(n_clusters=512, random_state=42,
n_init=10) on each raw embedding; writes <D>.index_{lemb,vitemb}_kmeans512.json = {idx: ["<a_label>"]}.
"""
import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data_root", required=True)
p.add_argument("--n_clusters", type=int, default=512)
a = p.parse_args()

d = os.path.join(a.data_root, a.dataset)
for emb_path in [f"{d}/{a.dataset}.emb-llama-td.npy", f"{d}/{a.dataset}.emb-ViT-L-14.npy"]:
    embeddings = np.load(emb_path)
    print("Embedding shape:", embeddings.shape)
    kmeans = KMeans(n_clusters=a.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    item2cluster = {f"{idx}": [f"<a_{label}>"] for idx, label in enumerate(labels)}
    tag = "lemb" if "llama" in emb_path else "vitemb"
    out_path = f"{d}/{a.dataset}.index_{tag}_kmeans{a.n_clusters}.json"
    with open(out_path, "w") as f:
        json.dump(item2cluster, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")
