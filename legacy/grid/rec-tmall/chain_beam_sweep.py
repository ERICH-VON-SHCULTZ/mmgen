"""Pick chain's BEST beam allocation at a fixed 808-decoder-state budget.

Chain's cost is 1 + 3 + nb*(1 + B1 + B2) states, so at nb=4 the budget fixes B1 + B2 = 200; how it is
SPLIT across levels is free. Chain has no equivalent to MFD's free K_t/K_i widening (its later levels
are conditioned on earlier ones, so extra candidates there cost extra forwards), so the fair
comparison is to give chain its best split rather than assume the uniform B1=B2=100.

Reads the small VAL sweep dumps (generated without encoder memory -- only coverage is needed),
reports coverage@pool and the achieved state count for each allocation, and writes the winner.
Selection is on VAL, never TEST.
"""

import argparse
import json
import os

import torch


def coverage(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    N, nb, Kp = c["pool_f"].shape
    fl = lambda x: x.reshape(N, nb * Kp)
    br = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1)
    gt = ((br == c["gt_b"].long()[:, None])
          & (fl(c["pool_f"]).long() == c["gt_f"].long()[:, None])
          & (fl(c["pool_t"]).long() == c["gt_t"].long()[:, None])
          & (fl(c["pool_i"]).long() == c["gt_i"].long()[:, None])
          & (fl(c["pool_sidscore"]).float() > -1e30))
    return float(gt.any(1).float().mean()), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs=2, action="append", metavar=("SPEC", "PATH"), required=True,
                    help='SPEC like "150x50" (B1xB2)')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("=" * 78)
    print("chain beam-ALLOCATION sweep at fixed 808 decoder states (B1 + B2 = 200), selected on VAL")
    print("=" * 78)
    print(f"  {'B1xB2':<10}{'states/item':>13}{'VAL coverage@pool':>20}")
    best = None
    for spec, path in args.dump:
        if not os.path.exists(path):
            print(f"  {spec:<10}  MISSING {path}")
            continue
        cov, meta = coverage(path)
        st = meta.get("fwd_states_per_item", "?")
        flag = "" if st == 808 else "  ** STATE MISMATCH -- excluded **"
        print(f"  {spec:<10}{str(st):>13}{cov:>20.4f}{flag}")
        if st == 808 and (best is None or cov > best[0]):
            best = (cov, spec, meta.get("beam1"), meta.get("beam2"))
    if best is None:
        raise SystemExit("** no valid 808-state chain allocation -- stop. **")
    print(f"\n  WINNER (VAL coverage): B1={best[2]} B2={best[3]}  coverage={best[0]:.4f}")
    json.dump({"beam1": int(best[2]), "beam2": int(best[3]), "val_coverage": best[0], "spec": best[1]},
              open(args.out, "w"), indent=1)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
