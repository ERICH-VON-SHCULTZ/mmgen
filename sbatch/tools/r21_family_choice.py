"""§21: best run per family (G0, A, B, C) by Arts valid HR@10, plain inference (tie NDCG@10) -> geo/r21_family_choice.json/.txt."""
import json
R = "/scratch/yw8866/mmgen_runs"
X = json.load(open(f"{R}/geo/r21_Arts.json"))["runs"]
fam = {"G0": ["r21_G0"], "A": ["r21_A01", "r21_A05"], "B": ["r21_B01", "r21_B05"], "C": ["r21_C033", "r21_C060"]}
ch = {}
for f, rs in fam.items():
    ok = [r for r in rs if r in X and not X[r].get("missing")]
    if ok:
        best = max(ok, key=lambda r: (X[r]["valid"]["plain"]["hr10"], X[r]["valid"]["plain"]["ndcg10"]))
        ch[f] = {"run": best, "valid_plain": {k: X[best]["valid"]["plain"][k] for k in ("hr10", "ndcg10")},
                 "candidates": {r: X[r]["valid"]["plain"]["hr10"] for r in ok}}
json.dump({"rule": "max Arts valid HR@10, plain inference K_f=50 (tie NDCG@10), per family", "chosen": ch},
          open(f"{R}/geo/r21_family_choice.json", "w"), indent=1)
open(f"{R}/geo/r21_family_choice.txt", "w").write(" ".join(v["run"][4:] for v in ch.values()))
print(json.dumps(ch, indent=1))
