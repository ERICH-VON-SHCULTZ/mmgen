"""mmgen addition (docs/EXPLORATION.md Part 3): inference-only re-scoring checks on a trained MFD model, VALIDATION only.

One exact scoring pass with K_f = all catalog f (mfd_eval.score_batch, as mfd_sweep), so every item has
s_f = log P(f), s_t, s_i (full-vocabulary log-probs of its text / image branch codes given f). Settings, all ranked with
mfd_sweep's tie rule (score, then item id); an item can only be ranked if its f is kept:
  base        keep the top-K f by log P(f); item score w_f s_f + w_t s_t + w_i s_i   (K=100, w=1/.5/.5 = setting (ii))
  sum         keep the top-K f by  log P(f) + log sum_{items under f} exp(b),  b = s_t + s_i  (total branch probability
              of the catalog items under f); items scored as base
  sum_w       same with b = w_t s_t + w_i s_i
  max         keep by log P(f) + max_{items under f} b
  renorm      branch scores renormalised over the catalog items under the same f: s_t' = s_t - logsumexp_f(s_t),
              s_i' likewise (P(branch path | f, catalog)); keep by log P(f); score w_f s_f + w_t s_t' + w_i s_i'
  tta_f       test-time view ensemble of the f distribution: log P(f) = mean of log P(f) over the full, text and image
              views of the history (the views mfd122_mv was trained on); branches from the full view; keep + score as base
  tta_all     every term (s_f, s_t, s_i) averaged over the three views
for K in {20, 50, 100}. Reports HR@1/5/10, NDCG@5/10 and coverage (target f kept).
Usage: python expl_infer.py --ckpt_path <mfd run> --dataset D --index_file ... --image_index_file ... --data_path ...
       --out_file <json>
"""
import argparse
import json

import torch

from mfd_data import Catalog, MFDCollator, MFDDataset, view_maps
from mfd_eval import score_batch
from mfd_model import BRANCH_TOKENS, MFDT5
from utils import parse_dataset_args, parse_global_args, parse_test_args

KS = (20, 50, 100)
W = (1.0, 0.5, 0.5)


def full_scores(model, cat, ids, mask, T, I):
    out = score_batch(model, cat, ids, mask, cat.nF, T, I)
    B = ids.shape[0]
    S = torch.zeros(3, B, cat.N, device=ids.device)
    S[0, out["user"], out["item"]] = out["s_f"]
    S[1, out["user"], out["item"]] = out["s_t"]
    S[2, out["user"], out["item"]] = out["s_i"]
    return S, out["lpf_all"]


def lse_by_f(x, fi, nF):
    """x (B, N) -> (B, nF): logsumexp over the items of each f."""
    B = x.shape[0]
    idx = fi[None].expand(B, -1)
    m = torch.full((B, nF), float("-inf"), device=x.device).scatter_reduce(1, idx, x, "amax")
    s = torch.zeros(B, nF, device=x.device).scatter_add(1, idx, (x - m.gather(1, idx)).exp())
    return m + s.log()


def max_by_f(x, fi, nF):
    B = x.shape[0]
    return torch.full((B, nF), float("-inf"), device=x.device).scatter_reduce(1, fi[None].expand(B, -1), x, "amax")


def ranks(score, fkeep_score, fi, tgt, K):
    """Keep the top-K f by fkeep_score (B, nF); rank target among kept items; BIG if target f not kept."""
    B, N = score.shape
    kth = fkeep_score.topk(K, dim=1).values[:, -1:]
    keep_f = fkeep_score >= kth                                        # (B, nF)  (ties at the boundary kept)
    keep = keep_f.gather(1, fi[None].expand(B, -1))
    sc = torch.where(keep, score, torch.full_like(score, float("-inf")))
    st = sc.gather(1, tgt[:, None])
    ids = torch.arange(N, device=score.device)
    r = (sc > st).sum(1) + ((sc == st) & (ids[None] < tgt[:, None])).sum(1)
    cov = keep.gather(1, tgt[:, None]).squeeze(1)
    return torch.where(cov, r, torch.full_like(r, 10 ** 6)), cov


def metrics(r, cov):
    rf = r.float()
    out = {f"hit@{k}": (r < k).float().mean().item() for k in (1, 5, 10)}
    for k in (5, 10):
        out[f"ndcg@{k}"] = torch.where(r < k, 1 / torch.log2(rf + 2), torch.zeros_like(rf)).mean().item()
    out["coverage"] = cov.float().mean().item()
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--split", default="valid")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    assert args.split == "valid", "Part 3 checks are validation-only"
    from transformers import T5Tokenizer
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = MFDT5.from_pretrained(args.ckpt_path).cuda().eval()
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    ds = MFDDataset(args, args.split)
    vmap = view_maps(ds.indices)
    coll = MFDCollator(args, tok)
    fi = cat.code[:, 0]
    nF = cat.nF
    names = ["base", "sum", "sum_w", "max", "renorm", "tta_f", "tta_all"]
    R = {(n, K): [] for n in names for K in KS}
    C = {(n, K): [] for n in names for K in KS}
    bs = args.test_batch_size
    for s in range(0, len(ds), bs):
        js = range(s, min(s + bs, len(ds)))
        views = {}
        for v in ("full", "text", "image"):
            batch = [dict(input_ids="".join(vmap[v][h] for h in ds.inter_data[j]["inters"]), labels=ds[j]["labels"]) for j in js]
            if v == "full":
                assert all(b["input_ids"] == ds[j]["input_ids"] for b, j in zip(batch, js))
            inp = coll(batch)
            views[v] = full_scores(model, cat, inp["input_ids"].cuda(), inp["attention_mask"].cuda(), T, I)
        tgt = torch.tensor([ds[j]["labels"] for j in js], device="cuda")
        S, lpf = views["full"]
        base = W[0] * S[0] + W[1] * S[1] + W[2] * S[2]
        b = S[1] + S[2]
        bw = W[1] * S[1] + W[2] * S[2]
        st_n = S[1] - lse_by_f(S[1], fi, nF).gather(1, fi[None].expand(len(js), -1))
        si_n = S[2] - lse_by_f(S[2], fi, nF).gather(1, fi[None].expand(len(js), -1))
        lpf_tta = torch.stack([views[v][1] for v in views]).mean(0)
        S_tta = torch.stack([views[v][0] for v in views]).mean(0)
        variants = {
            "base": (base, lpf),
            "sum": (base, lpf + lse_by_f(b, fi, nF)),
            "sum_w": (base, lpf + lse_by_f(bw, fi, nF)),
            "max": (base, lpf + max_by_f(b, fi, nF)),
            "renorm": (W[0] * S[0] + W[1] * st_n + W[2] * si_n, lpf),
            "tta_f": (W[0] * lpf_tta.gather(1, fi[None].expand(len(js), -1)) + W[1] * S[1] + W[2] * S[2], lpf_tta),
            "tta_all": (W[0] * S_tta[0] + W[1] * S_tta[1] + W[2] * S_tta[2], lpf_tta),
        }
        for n, (sc, fk) in variants.items():
            for K in KS:
                r, cov = ranks(sc, fk, fi, tgt, K)
                R[(n, K)].append(r.cpu()); C[(n, K)].append(cov.cpu())
        if s % (bs * 50) == 0:
            print(s, flush=True)
    res = {"split": args.split, "ckpt": args.ckpt_path, "weights": W, "users": len(ds), "results": {}}
    for n in names:
        for K in KS:
            m = metrics(torch.cat(R[(n, K)]), torch.cat(C[(n, K)]))
            res["results"][f"{n}_K{K}"] = m
            print(f"{n:8s} K={K:3d}  HR@10 {m['hit@10']:.4f}  NDCG@10 {m['ndcg@10']:.4f}  HR@1 {m['hit@1']:.4f}  coverage {m['coverage']:.4f}", flush=True)
    torch.save({k: torch.cat(v) for k, v in R.items()}, args.out_file.replace(".json", "_ranks.pt"))
    json.dump(res, open(args.out_file, "w"), indent=1)


if __name__ == "__main__":
    main()
