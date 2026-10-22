"""mmgen addition: clip_feature.py's computation (same CLIP load, preprocess, encode_image, same output
file and dtype), but items whose image could not be downloaded do not crash the run
(clip_feature.py indexes images_info[item][0] outside its try block).

Rows of items without an image are filled, and listed in <D>.emb-ViT-L-14.missing.json:
  --ref_npy given (same shape): the row from that file (MQL4GRec's published features, Instruments),
  otherwise: the mean of all computed image features of the dataset.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from clip import clip
from clip_feature import get_id2item_dict, load_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--save_root", required=True)
    p.add_argument("--backbone", default="ViT-L/14")
    p.add_argument("--model_cache_dir", required=True)
    p.add_argument("--ref_npy", default=None)
    a = p.parse_args()

    save_path = f"{a.save_root}/{a.dataset}"
    id2item = get_id2item_dict(os.path.join(save_path, f"{a.dataset}.item2id"))
    images_info = load_json(os.path.join(a.image_root, f"{a.dataset}_images_info.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(a.backbone, device=device, download_root=a.model_cache_dir)

    rows, missing = [], []
    with torch.no_grad():
        for i in tqdm(range(len(id2item))):
            item = id2item[str(i)]
            names = images_info.get(item, [])
            if not names:
                rows.append(None); missing.append({"idx": i, "asin": item}); continue
            image = preprocess(Image.open(os.path.join(a.image_root, a.dataset, names[0]))).unsqueeze(0).to(device)
            rows.append(model.encode_image(image)[0].cpu())

    computed = torch.stack([r for r in rows if r is not None]).numpy()
    ref = np.load(a.ref_npy) if a.ref_npy else None
    if ref is not None and ref.shape != (len(rows), computed.shape[1]):
        ref = None
    fill_src = "ref_npy" if ref is not None else "dataset_mean"
    mean = computed.astype(np.float32).mean(0).astype(computed.dtype)
    out = np.stack([r.numpy() if r is not None else (ref[i].astype(computed.dtype) if ref is not None else mean)
                    for i, r in enumerate(rows)])
    name = a.backbone.replace("/", "-")
    np.save(os.path.join(save_path, f"{a.dataset}.emb-{name}.npy"), out)
    json.dump({"fill": fill_src, "missing": missing},
              open(os.path.join(save_path, f"{a.dataset}.emb-{name}.missing.json"), "w"), indent=1)
    print("Embeddings shape:", out.shape, out.dtype, "| missing images:", len(missing), "filled from", fill_src)


if __name__ == "__main__":
    main()
