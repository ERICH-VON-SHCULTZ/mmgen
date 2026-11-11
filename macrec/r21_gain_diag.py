"""mmgen addition (docs/MFD_REPORT.md §21.6): where do the round-21 gains come from (inference only, TEST, plain K_f = 50).

Per test user of one run (full view, setting (ii) at K_f = 50, the deployment scoring of r21_eval.py):
  lp_true     log P(true f) (softmax over the shared codes, as inference), lp_top = max_f log P(f), margin = lp_true - lp_top
  f_rank      rank of the true f by log P(f) (0 = top); kept = f_rank < K_f
  rank        rank of the target among the scored candidates (BIG if its f is not kept)
  n_other     candidates from OTHER f buckets ranked above the target (score higher, ties by item id)
  n_within    candidates from the target's OWN f bucket ranked above it;  rank = n_other + n_within when kept
Sanity: rank equals the run's r21_eval test plain ranks (share > 0.999).
Usage: python r21_gain_diag.py --ckpt_path <run>/seed<S> --out_file X.pt (+ dataset args)
"""
import argparse
import os

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import BIG, branch_scores, encode, f_logprobs, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args

K = 50


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    out = {k: [] for k in ("lp_true", "lp_top", "f_rank", "rank", "n_other", "n_within", "target")}
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda()
            B = len(tgt)
            ar = torch.arange(B, device="cuda")
            enc, mask = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            lpf = f_logprobs(model, cat, enc, mask)
            tf = cat.code[tgt, 0]
            lt = lpf[ar, tf]
            fr = (lpf > lt[:, None]).sum(1)
            fsel = lpf.topk(K, dim=1).indices
            user, item, frank, st, si = branch_scores(model, cat, enc, mask, fsel, T, I)
            fi = cat.code[item, 0]
            score = lpf[user, fi] + 0.5 * st + 0.5 * si
            hit = item == tgt[user]
            ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
            better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
            own = fi == tf[user]
            kept = torch.isfinite(ts)
            r = torch.bincount(user[better], minlength=B)
            no = torch.bincount(user[better & ~own], minlength=B)
            nw = torch.bincount(user[better & own], minlength=B)
            out["lp_true"].append(lt.cpu()); out["lp_top"].append(lpf.max(1).values.cpu()); out["f_rank"].append(fr.cpu())
            out["rank"].append(torch.where(kept, r, torch.full_like(r, BIG)).cpu())
            out["n_other"].append(torch.where(kept, no, torch.full_like(no, -1)).cpu())
            out["n_within"].append(torch.where(kept, nw, torch.full_like(nw, -1)).cpu())
            out["target"].append(tgt.cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    ref = torch.load(os.path.join(args.ckpt_path, "r21_eval", "ranks.pt"), weights_only=False)["test"]["plain"]
    same = (out["rank"].clamp(max=10) == ref.clamp(max=10)).float().mean().item()
    out["sanity_share_same_rank_as_r21_eval"] = same
    print(f"[r21_gain_diag] {args.ckpt_path}: users {len(out['rank'])}, same rank as r21_eval {same:.5f}, HR@10 {(out['rank'] < 10).float().mean():.4f}")
    assert same > 0.999, same
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    torch.save(out, args.out_file)


if __name__ == "__main__":
    main()
