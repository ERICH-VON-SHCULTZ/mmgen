"""Modality dominance of early-fusion codes (baseline E): adjusted mutual information between the
first-level SID codes of E and those of A (text TIGER) and B (image TIGER), over the same items.
A vs B is printed as the reference for two unrelated tokenizers on the same items.
Run inside the container (needs sklearn). Usage: python ami_first_level.py --dataset Instruments
Writes <runs>/baselines/<D>/tok/ami_first_level.json."""
import argparse
import itertools
import json
import os

from sklearn.metrics import adjusted_mutual_info_score

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data_root", default="/scratch/yw8866/mmgen_runs/data")
p.add_argument("--out_root", default="/scratch/yw8866/mmgen_runs/baselines")
a = p.parse_args()
D = a.dataset
# E_norm = the official rescaled E-norm (fnormj: joint length = A's mean text norm); E_norm_x53 = the
# superseded variant (each half x mean text norm), kept as a record.
files = {"A_text": "lemb_tiger", "B_image": "vitemb_tiger", "E_norm": "fnormj_tiger", "E_norm_x53": "fnorm_tiger",
         "E_raw": "fraw_tiger"}
codes = {}
for k, tag in files.items():
    f = f"{a.data_root}/{D}/{D}.index_{tag}.json"
    if os.path.exists(f):
        idx = json.load(open(f))
        codes[k] = [idx[str(i)][0] for i in range(len(idx))]   # first-level token, item-id order
n = {len(v) for v in codes.values()}
assert len(n) == 1, n
out = {"dataset": D, "items": n.pop(), "level0_used": {k: len(set(v)) for k, v in codes.items()}, "ami": {}}
for x, y in itertools.combinations(codes, 2):
    out["ami"][f"{x}~{y}"] = adjusted_mutual_info_score(codes[x], codes[y])
os.makedirs(f"{a.out_root}/{D}/tok", exist_ok=True)
json.dump(out, open(f"{a.out_root}/{D}/tok/ami_first_level.json", "w"), indent=1)
print(json.dumps(out))
