"""Download the main product image for every item of a processed dataset (mmgen addition).

Same selection rule as MACRec's load_all_figures.py (first URL in `imageURLHighRes` that downloads
as a valid JPEG; it re-uses that file's download_image / is_valid_jpg), but restricted to the
items in <D>.item2id instead of every item in the category meta file, and parallelised.
Writes <image_root>/<D>/<file>.jpg and <image_root>/<D>_images_info.json ({asin: [file]} or
{asin: []} when no image could be obtained), the layout clip_feature.py expects.
"""
import argparse
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor

from load_all_figures import download_image, is_valid_jpg
from utils import amazon18_dataset2fullname


def read_item2id(path):
    with open(path) as f:
        return dict(line.rstrip("\n").split("\t") for line in f)


def read_meta_urls(meta_file, asins):
    urls = {}
    with gzip.open(meta_file, "r") as fp:
        for line in fp:
            d = json.loads(line)
            if d["asin"] in asins and d["asin"] not in urls:
                urls[d["asin"]] = d.get("imageURLHighRes", []) or []
    return urls


def fetch_one(asin, urls, out_dir):
    for rank, url in enumerate(urls):
        name = os.path.basename(url)
        path = os.path.join(out_dir, name)
        ok = os.path.exists(path) or download_image(url, path)
        if ok and is_valid_jpg(path):
            return asin, [name], rank
    return asin, [], -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", required=True, help="dir holding <D>/<D>.item2id")
    p.add_argument("--meta_data_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    item2id = read_item2id(os.path.join(a.data_root, a.dataset, f"{a.dataset}.item2id"))
    meta = os.path.join(a.meta_data_path, f"meta_{amazon18_dataset2fullname[a.dataset]}.json.gz")
    urls = read_meta_urls(meta, set(item2id))
    out_dir = os.path.join(a.image_root, a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda s: fetch_one(s, urls.get(s, []), out_dir), item2id))
    info = {asin: names for asin, names, _ in res}
    with open(os.path.join(a.image_root, f"{a.dataset}_images_info.json"), "w") as f:
        json.dump(info, f, indent=1)

    n = len(item2id)
    no_meta = sum(1 for s in item2id if s not in urls)
    no_url = sum(1 for s in item2id if s in urls and not urls[s])
    failed = sum(1 for _, names, _ in res if not names)
    not_first = sum(1 for _, _, r in res if r > 0)
    print(f"[{a.dataset}] items {n}  missing from meta {no_meta}  no imageURLHighRes {no_url}  "
          f"no image obtained {failed}  image not the first URL {not_first}")


if __name__ == "__main__":
    main()
