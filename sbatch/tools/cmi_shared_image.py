"""First-level CMI(t1; i1 | f) of an MD-RQ-VAE SID set, on all items and after excluding every item
whose image file is shared with any other item (identical bytes, SHA-256; this includes the Games
banner image). Same estimator and permutation null as md_diagnostics.cmi_test (100 reps, seed 0).
Observation only: the tokenizer is not changed based on this.
Usage: python3 cmi_shared_image.py [--datasets Instruments Arts Games] [--variant ep500]
       -> /scratch/yw8866/mmgen_runs/md/cmi_shared_image[_<variant>].json
--variant ep500 = the final-epoch sets (official from 2026-10-27; acceptance: excluded-items
corrected CMI <= ~2% of H(t1|f)); no variant = the superseded best-collision sets.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../macrec/cross_index"))
from md_diagnostics import cmi_test  # noqa: E402

RUNS = "/scratch/yw8866/mmgen_runs"
IMG = f"{RUNS}/raw/amazon18/Images"
SETS = {"1+2+2": ("a05_122", "md122_a05_reassign", 2), "1+3+3": ("a06_133", "md133_a06_reassign", 3)}


def shared_image_items(D):
    """Item ids whose image file has identical bytes to another item's image file."""
    info = json.load(open(f"{IMG}/{D}_images_info.json"))
    item2id = dict(l.split("\t") for l in open(f"{RUNS}/data/{D}/{D}.item2id").read().split("\n") if l)
    cache, by_hash, missing = {}, collections.defaultdict(list), 0
    for asin, files in info.items():
        if not files or not os.path.exists(f"{IMG}/{D}/{files[0]}"):
            missing += 1
            continue
        f = files[0]
        if f not in cache:
            cache[f] = hashlib.sha256(open(f"{IMG}/{D}/{f}", "rb").read()).hexdigest()
        by_hash[cache[f]].append(int(item2id[asin]))
    shared = sorted(i for g in by_hash.values() if len(g) > 1 for i in g)
    biggest = max(by_hash.values(), key=len)
    return shared, {"items_with_image": len(info) - missing, "items_without_image_file": missing,
                    "shared_image_items": len(shared), "shared_image_groups": sum(len(g) > 1 for g in by_hash.values()),
                    "largest_group": len(biggest)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["Instruments", "Arts", "Games"])
    p.add_argument("--variant", default="", help="ep500 -> md1X2_a0X_ep500_reassign")
    a = p.parse_args()
    sets = {cfg: (n, t.replace("_reassign", f"_{a.variant}_reassign") if a.variant else t, L) for cfg, (n, t, L) in SETS.items()}
    out = {}
    for D in a.datasets:
        shared, st = shared_image_items(D)
        out[D] = {"image_sharing": st}
        for cfg, (name, tag, L) in sets.items():
            codes = np.load(f"{RUNS}/md/{D}/tok/{name}/codes_{tag}.npy")
            keep = np.setdiff1d(np.arange(len(codes)), shared)
            res = {}
            for label, rows in (("all_items", np.arange(len(codes))), ("excl_shared_image", keep)):
                c = codes[rows]
                r = cmi_test(c[:, 1].tolist(), c[:, 1 + L].tolist(), c[:, 0].tolist())
                r["items"] = int(len(rows))
                res[label] = r
            out[D][cfg] = res
            print(D, cfg, {k: (v["items"], round(v["cmi_corrected_bits"], 4), round(v["frac_corrected"] * 100, 2))
                           for k, v in res.items()}, st, flush=True)
    json.dump(out, open(f"{RUNS}/md/cmi_shared_image{'_' + a.variant if a.variant else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
