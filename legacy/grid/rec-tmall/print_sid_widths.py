"""Print the actual per-level codebook widths of a semantic_ids.pt file (max code + 1 per level),
so the run header reports the true widths instead of assuming them."""
import sys

import torch

p = sys.argv[1]
o = torch.load(p, map_location="cpu", weights_only=False)
t = o
if isinstance(o, dict):
    for k, v in o.items():
        if torch.is_tensor(v) and v.dim() == 2:
            t = v
            print(f"  semantic_ids key: {k}")
            break
axis = 0 if t.shape[0] < t.shape[1] else 1        # levels are the SHORT axis
L = t.shape[axis]
widths = [int(t.select(axis, l).max()) + 1 for l in range(L)]
n_items = t.shape[1 - axis]
print(f"  tensor shape {tuple(t.shape)}  -> {L} levels x {n_items:,} items")
print(f"  per-level codebook widths (max code + 1): {widths}")