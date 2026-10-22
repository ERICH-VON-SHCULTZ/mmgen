"""SID diagnostics for an index JSON: uniqueness and per-level codebook utilization, plus the
pre-reassignment collision rate parsed from the generator log (its first 'Collision Rate' line)."""
import argparse
import json
import re

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--gen_log", default=None)
p.add_argument("--out", required=True)
p.add_argument("--codebook_size", type=int, default=256)
a = p.parse_args()

idx = json.load(open(a.index))
codes = list(idx.values())
L = len(codes[0])
uniq = len({"".join(c) for c in codes})
st = {
    "index": a.index, "items": len(codes), "levels": L, "unique_final": uniq,
    "all_unique": uniq == len(codes),
    "utilization_per_level": [len({c[l] for c in codes}) for l in range(L)],
    "codebook_size": a.codebook_size,
}
if a.gen_log:
    rates = re.findall(r"Collision Rate\s+([0-9.eE-]+)", open(a.gen_log, errors="ignore").read())
    if rates:
        st["collision_rate_before"] = float(rates[0])
        st["collision_rate_after"] = float(rates[-1])
json.dump(st, open(a.out, "w"), indent=1)
print(json.dumps(st))
