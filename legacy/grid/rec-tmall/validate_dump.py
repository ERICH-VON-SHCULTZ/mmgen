"""Validate a pool dump IMMEDIATELY after generation, so a truncated dump fails at generation time
instead of poisoning everything downstream.

Root cause this guards against: `+trainer.limit_test_batches=N` multiplies with
`batch_size_per_device=B`, so the dump silently truncates to N*B events. With the default B=64 and
N=6000 that is 384,000 (never binding), but lowering B to 4 for the dense-compose pool made it
6000*4 = 24,000 -- the P2/P3 pools were truncated and only failed much later, at load time, after
three full generation runs had already been paid for.

Checks: total event count, optional per-behavior counts, decoder-states/item, and coverage@pool.
Exits non-zero with a loud message on any mismatch.
"""

import argparse
import sys

import torch

BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--expect_n", type=int, required=True)
    ap.add_argument("--expect_perbeh", default="", help='e.g. "77833,2042,1554,1576" (test split only)')
    ap.add_argument("--expect_states", type=int, default=808)
    ap.add_argument("--min_cov", type=float, default=None,
                    help="FAIL if coverage@pool < this (a widened pool that does not widen coverage is a failed dump)")
    ap.add_argument("--max_mem_gb", type=float, default=None, help="optional sanity bound on peak GPU memory")
    ap.add_argument("--expect_cov", default="",
                    help='coverage of the pool TRUNCATED to the top-K slots must match: "K:value:tol,..." '
                         'e.g. "500:0.2618:0.003,2000:0.2991:0.02,5000:0.3346:0.02" (pool slots are score-sorted)')
    args = ap.parse_args()

    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    c, meta = d["cols"], d["meta"]
    N, nb, Kp = c["pool_f"].shape
    gt_b = c["gt_b"].long()
    per = {b: int((gt_b == b).sum()) for b in range(nb)}
    states = meta.get("fwd_states_per_item", None)

    fl = lambda x: x.reshape(N, nb * Kp)
    br = torch.arange(nb).repeat_interleave(Kp).view(1, -1).expand(N, -1)
    cov = float((((br == gt_b[:, None])
                  & (fl(c["pool_f"]).long() == c["gt_f"].long()[:, None])
                  & (fl(c["pool_t"]).long() == c["gt_t"].long()[:, None])
                  & (fl(c["pool_i"]).long() == c["gt_i"].long()[:, None])
                  & (fl(c["pool_sidscore"]).float() > -1e30)).any(1)).float().mean())

    print(f"[validate] {args.dump}")
    print(f"           N={N:,} (expect {args.expect_n:,})  K_pool={Kp}  "
          f"decoder-states/item={states} (expect {args.expect_states})  coverage@pool={cov:.4f}")
    print(f"           per-behavior: " + "  ".join(f"{BEH.get(b,b)}={per.get(b,0):,}" for b in range(nb)))
    if "peak_mem_gb" in meta:
        print(f"           peak GPU memory during generation: {meta['peak_mem_gb']:.2f} GB")
    if meta.get("eff"):
        print(f"           composition-stage effective values (recorded INSIDE _audit_forward): {meta['eff']}")
    if "gt_comp_rank" in c:
        # coverage@K for ANY K, from the GT's rank among all composed candidates (before the K_pool cut)
        r = c["gt_comp_rank"].long()
        ladder = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        print("           coverage@K if K_pool were K (from gt_comp_rank): " + "  ".join(
            f"K={K}:{float(((r >= 0) & (r < K)).float().mean()):.4f}" for K in ladder)
              + f"  composed-at-all:{float((r >= 0).float().mean()):.4f}")
        ip = float(c["gt_in_pool"].float().mean()) if "gt_in_pool" in c else float("nan")
        print(f"           in-model gt_in_pool={ip:.4f} vs recomputed coverage={cov:.4f} -> "
              f"{'consistent' if abs(ip - cov) < 1e-4 else 'INCONSISTENT'}")

    bad = []
    if N != args.expect_n:
        bad.append(f"event count {N:,} != expected {args.expect_n:,} "
                   f"(check limit_test_batches * batch_size_per_device >= expected)")
    if args.expect_perbeh:
        exp = [int(x) for x in args.expect_perbeh.split(",")]
        got = [per.get(b, 0) for b in range(len(exp))]
        if got != exp:
            bad.append(f"per-behavior {got} != expected {exp}")
    if states is not None and args.expect_states and states != args.expect_states:
        bad.append(f"decoder-states/item {states} != {args.expect_states} -- the equal-compute match is BROKEN")
    if args.expect_cov:
        # GT position inside its own behavior's score-sorted pool slice -> coverage@K = P(pos < K)
        ar = torch.arange(N)
        sf = c["pool_f"][ar, gt_b].long(); st = c["pool_t"][ar, gt_b].long(); si = c["pool_i"][ar, gt_b].long()
        ss = c["pool_sidscore"][ar, gt_b].float()
        mt = ((sf == c["gt_f"].long()[:, None]) & (st == c["gt_t"].long()[:, None])
              & (si == c["gt_i"].long()[:, None]) & (ss > -1e30))
        pos = torch.where(mt.any(1), mt.float().argmax(1), torch.full((N,), -1, dtype=torch.long))
        for spec in args.expect_cov.split(","):
            K, val, tol = spec.split(":"); K, val, tol = int(K), float(val), float(tol)
            got = float(((pos >= 0) & (pos < K)).float().mean())
            ok = abs(got - val) <= tol
            print(f"           coverage@top-{K}: {got:.4f}  expected {val:.4f} +- {tol:.3f}  -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append(f"coverage@top-{K} {got:.4f} not within {tol} of the ladder value {val:.4f}")
    if args.min_cov is not None and cov < args.min_cov:
        bad.append(f"coverage@pool {cov:.4f} < required {args.min_cov:.2f} -- the pool did NOT widen; "
                   f"K_pool={Kp} is the binding cut (see coverage@K ladder above)")
    if args.max_mem_gb is not None and meta.get("peak_mem_gb", 0) > args.max_mem_gb:
        bad.append(f"peak GPU memory {meta['peak_mem_gb']:.2f} GB > {args.max_mem_gb}")
    if bad:
        print("** DUMP VALIDATION FAILED **")
        for b in bad:
            print("   - " + b)
        sys.exit(1)
    print("           OK")


if __name__ == "__main__":
    main()
