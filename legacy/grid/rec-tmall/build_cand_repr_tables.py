"""Build the candidate-representation cache for the reranker's tokenizer-geometry ablation.

Each reranker candidate is an MD-RQ-VAE tuple (f,t,i). To represent it with an ALTERNATIVE
tokenizer's geometry we map: MD tuple -> a representative catalog item -> that tokenizer's codes
-> its codebook centroid(s). All three tokenizers write semantic_ids.pt with the SAME item-column
layout (shape (num_levels, N_items); §13), so item column j is the same physical item in every
file -> the mapping is exact.

Produces a compact cache keyed by the UNIQUE MD catalog tuples (sorted packed keys, = the reranker's
searchsorted index space):
  keys        (M,)   int64   sorted unique packed MD keys  ((f*k1+t)*k2+i)
  mbgen1_id   (M,)   int16   MBGen level-1 (QAE) code of the representative item
  mbgen_cb    (2048, d_qae)  MBGen QAE codebook (level-1 geometry)
  tiger_l     (M, 3) int16   TIGER (chain RQ-VAE) 3 codes of the representative item
  tiger_cb    list of 3 (1024, d_tiger)  TIGER residual codebooks (level geometry; sum is additive)
Only codebooks + per-tuple code INDICES are stored (small); the reranker builds the lifted
(M, d_model) view tables at load time.

Heavy diagnostics on purpose (discovered checkpoint keys, map coverage, shapes) so a first run
self-validates the cross-tokenizer alignment. Run in-container (needs torch); no GPU required.
"""

from __future__ import annotations

import argparse
import os
import sys

# The TIGER tokenizer checkpoint is a Lightning ckpt pickled with src.* class references; loading
# it needs the GRID package importable (and warmed to avoid the interfaces<->utils circular import).
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import torch


MD_K0, MD_K1, MD_K2 = 2048, 1024, 1024   # MD-RQ-VAE codebook widths (shared/text/image)


def load_sid(path):
    """semantic_ids.pt -> (num_levels, N_items) long tensor."""
    t = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(t):
        raise ValueError(f"{path}: expected a tensor, got {type(t)}")
    t = t.long()
    if t.shape[0] > t.shape[1]:                       # guard: must be (levels, items), levels small
        raise ValueError(f"{path}: shape {tuple(t.shape)} looks transposed (levels should be dim0)")
    print(f"[sid] {os.path.basename(os.path.dirname(os.path.dirname(path)))}: shape={tuple(t.shape)} "
          f"levels={t.shape[0]} items={t.shape[1]:,} "
          f"level-ranges={[ (int(t[l].min()), int(t[l].max())) for l in range(t.shape[0]) ]}", flush=True)
    return t


def discover_centroids(state_dict, expect_k, want=3):
    """Find `want` codebook centroid tensors of shape (expect_k, d) in a tokenizer state_dict.
    Robust to unknown key names: scan for keys containing 'centroid'/'codebook'/'embed' whose
    first dim == expect_k. Returns list of (key, tensor) sorted by key. Prints all candidates."""
    cands = []
    for k, v in state_dict.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            continue
        kl = k.lower()
        if ("centroid" in kl or "codebook" in kl or "embed" in kl) and v.shape[0] == expect_k:
            cands.append((k, v))
    print(f"[tiger] centroid-like keys with dim0=={expect_k}:", flush=True)
    for k, v in cands:
        print(f"          {k:60s} {tuple(v.shape)}", flush=True)
    if len(cands) < want:
        raise ValueError(f"expected >={want} codebooks of width {expect_k}, found {len(cands)}. "
                         f"Inspect the printed keys and set them explicitly.")
    cands = sorted(cands, key=lambda kv: kv[0])[:want]
    print(f"[tiger] using (sorted): {[k for k, _ in cands]}", flush=True)
    return [v.float() for _, v in cands]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_sid", default="/scratch/yw8866/logs/inference/runs/2026-05-06/2048_1024/pickle/semantic_ids.pt")
    ap.add_argument("--mbgen_sid", default="/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/semantic_ids.pt")
    ap.add_argument("--mbgen_artifacts", default="/scratch/yw8866/logs/inference/runs/mbgen_sid/2048_2048/mbgen_sid_artifacts.pt")
    ap.add_argument("--tiger_sid", default="/scratch/yw8866/logs/inference/runs/2026-04-29/tiger_1024/pickle/semantic_ids.pt")
    ap.add_argument("--tiger_ckpt", default="/scratch/yw8866/logs/train/runs/2026-04-29/tiger_1024/checkpoints/checkpoint_000_010000.ckpt")
    ap.add_argument("--tiger_k", type=int, default=1024, help="TIGER per-level codebook width")
    ap.add_argument("--out", default="/scratch/yw8866/rec-tmall/cand_repr_tables.pt")
    args = ap.parse_args()

    # ---- MD tuples -> unique sorted catalog keys + representative item per key ----
    md = load_sid(args.md_sid)                                   # (3, N)
    f, t, i = md[0], md[1], md[2]
    N = md.shape[1]
    packed = (f * MD_K1 + t) * MD_K2 + i                        # (N,)  per-item MD key
    # sorted unique keys + representative item (lowest item id) per key, via stable argsort +
    # first-occurrence (version-robust; no scatter_reduce). Ties keep ascending item order.
    srt = torch.argsort(packed, stable=True)
    pk = packed[srt]
    first = torch.ones(pk.numel(), dtype=torch.bool)
    first[1:] = pk[1:] != pk[:-1]
    keys = pk[first].contiguous()                              # (M,) sorted unique keys
    rep_item = srt[first].contiguous()                         # (M,) representative item per key
    M = keys.numel()
    print(f"[md] unique catalog tuples M={M:,} of {N:,} items "
          f"(collision {1 - M / N:.3f}); rep-item map built.", flush=True)

    out = {"keys": keys, "md_k": [MD_K0, MD_K1, MD_K2]}

    # ---- MBGen level-1 (QAE) geometry ----
    try:
        mb = load_sid(args.mbgen_sid)                           # (>=1, N)
        art = torch.load(args.mbgen_artifacts, map_location="cpu", weights_only=False)
        mbgen_cb = art["codebook"].float()                     # QAE codebook (may be (d_qae,num_codes))
        mbgen1_item = mb[0]                                     # level-1 code per item
        mbgen1_id = mbgen1_item[rep_item].to(torch.int16)      # (M,) per unique tuple
        if mbgen_cb.shape[0] <= int(mbgen1_id.long().max()):   # orient to (num_codes, d_qae)
            mbgen_cb = mbgen_cb.t().contiguous()
        bad = int((mbgen1_id.long() >= mbgen_cb.shape[0]).sum())
        print(f"[mbgen] codebook {tuple(mbgen_cb.shape)}  id1 range "
              f"[{int(mbgen1_id.min())},{int(mbgen1_id.max())}]  out-of-range={bad}", flush=True)
        out["mbgen1_id"] = mbgen1_id
        out["mbgen_cb"] = mbgen_cb
    except Exception as e:
        print(f"[mbgen] SKIP ({e})", flush=True)

    # ---- TIGER (chain residual RQ-VAE) geometry ----
    try:
        tg = load_sid(args.tiger_sid)                          # (3, N)
        if tg.shape[0] < 3:
            raise ValueError(f"TIGER sid has {tg.shape[0]} levels, need 3")
        tiger_l = tg[:3, rep_item].t().contiguous().to(torch.int16)   # (M, 3)
        print(f"[tiger] loading centroids from {args.tiger_ckpt} (one-time ~50GB) ...", flush=True)
        try:
            import src.utils.utils  # noqa: F401  (warm GRID's graph before unpickling the ckpt)
        except Exception as ex:
            print(f"[tiger][warn] GRID warm-import failed ({ex}); attempting load anyway", flush=True)
        ck = torch.load(args.tiger_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        tiger_cb = discover_centroids(sd, expect_k=args.tiger_k, want=3)   # 3x (1024, d_tiger)
        d_t = tiger_cb[0].shape[1]
        for lvl in range(3):
            hi = int(tiger_l[:, lvl].max())
            if hi >= tiger_cb[lvl].shape[0]:
                raise ValueError(f"TIGER level {lvl} code {hi} >= codebook {tiger_cb[lvl].shape[0]}")
        print(f"[tiger] codebooks 3x{tuple(tiger_cb[0].shape)} d_tiger={d_t}; per-tuple codes built.",
              flush=True)
        out["tiger_l"] = tiger_l
        out["tiger_cb"] = tiger_cb
    except Exception as e:
        print(f"[tiger] SKIP ({e})", flush=True)

    torch.save(out, args.out)
    have = [k for k in ("mbgen1_id", "tiger_l") if k in out]
    print(f"[done] wrote {args.out}  (M={M:,}; variants available: "
          f"{'mbgen1 ' if 'mbgen1_id' in out else ''}{'tiger1/tiger3 ' if 'tiger_l' in out else ''})",
          flush=True)


if __name__ == "__main__":
    main()