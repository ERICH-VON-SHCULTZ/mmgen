"""mmgen addition (docs/MFD_REPORT.md §20 Task 0): per-user error decomposition of an MFD run at one setting.

Scores EVERY catalog item (score_batch with K = all shared codes; same exact scoring as mfd_eval) once, then derives:
  f_rank        rank of the target's f among all f codes by log P(f) (0 = top)
  rank          target's rank in the normal list = items under the top-K_f codes (10**6 if its f is not kept);
                checked against the run's reported ranks (assert)
  rank_oracle_keep   rank if the target's f were always kept (candidate set = top-K_f codes + the target's f)
  rank_oracle_top    rank if the target's f had the top-ranked f's log P(f): every item under the target's f moves up
                by w_f * (max log P(f) - log P(f*)), f* is kept, the other kept codes stay (the K_f-th code drops out
                when f* was outside the top K_f)
  rank_in_bucket     target's rank among the items that share its f (branch terms only decide this order)
  bucket        number of catalog items under the target's f
Out: torch .pt {setting, split: {...per-user tensors..., targets}} for the CPU analysis (sbatch/tools/expl_decomp_report.py).
"""
import argparse

import torch
from transformers import T5Tokenizer

import mfd_eval
from mfd_data import Catalog
from mfd_eval import load_mfd_model, load_split, score_batch
from mfd_model import BRANCH_TOKENS
from utils import parse_dataset_args, parse_global_args, parse_test_args

BIG = 10 ** 6


def better(sc, user, ts, item, tgt):
    return (sc > ts[user]) | ((sc == ts[user]) & (item < tgt[user]))


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--k_f", type=int, required=True)
    p.add_argument("--w_f", type=float, required=True)
    p.add_argument("--w_t", type=float, required=True)
    p.add_argument("--w_i", type=float, required=True)
    p.add_argument("--splits", default="test")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    K = min(args.k_f, cat.nF)
    out = {"setting": dict(k_f=K, w_f=args.w_f, w_t=args.w_t, w_i=args.w_i), "n_f": cat.nF,
           "bucket_sizes": cat.f_item_cnt.cpu()}
    for split in args.splits.split(","):
        ds, coll = load_split(args, split, tok)
        acc = {k: [] for k in ("f_rank", "rank", "rank_oracle_keep", "rank_oracle_top", "rank_in_bucket", "bucket", "targets")}
        with torch.no_grad():
            for s in range(0, len(ds), args.test_batch_size):
                batch = [ds[j] for j in range(s, min(s + args.test_batch_size, len(ds)))]
                inp = coll(batch)
                ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
                B = ids.shape[0]
                o = score_batch(model, cat, ids, mask, cat.nF, T, I)          # every f kept: every catalog item scored
                u, it = o["user"], o["item"]
                sc = args.w_f * o["s_f"] + args.w_t * o["s_t"] + args.w_i * o["s_i"]
                tgt = torch.tensor([d["labels"] for d in batch], device="cuda")
                lpf = o["lpf_all"]                                            # (B, nF)
                tf = cat.code[tgt, 0]                                         # target's f (local index)
                lt = lpf[torch.arange(B, device="cuda"), tf]
                f_rank = (lpf > lt[:, None]).sum(1)
                kept = o["frank"] < K                                         # frank = rank of the item's f by log P(f)
                hit = it == tgt[u]
                assert int(hit.sum()) == B
                ts = torch.zeros(B, device="cuda"); ts[u[hit]] = sc[hit]
                bt = better(sc, u, ts, it, tgt)
                own = cat.code[it, 0] == tf[u]
                r_norm = torch.bincount(u[bt & kept], minlength=B)
                r_norm = torch.where(f_rank < K, r_norm, torch.full_like(r_norm, BIG))
                r_keep = torch.bincount(u[bt & (kept | own)], minlength=B)
                shift = args.w_f * (lpf.max(1).values - lt)                    # >= 0
                sc2 = torch.where(own, sc + shift[u], sc)
                ts2 = ts + shift
                kept2 = (o["frank"] < K - 1) | ((o["frank"] == K - 1) & (f_rank < K)[u]) | own
                r_top = torch.bincount(u[better(sc2, u, ts2, it, tgt) & kept2 & (it != tgt[u])], minlength=B)
                r_bucket = torch.bincount(u[bt & own], minlength=B)
                for k, v in (("f_rank", f_rank), ("rank", r_norm), ("rank_oracle_keep", r_keep), ("rank_oracle_top", r_top),
                             ("rank_in_bucket", r_bucket), ("bucket", cat.f_item_cnt[tf]), ("targets", tgt)):
                    acc[k].append(v.cpu())
        out[split] = {k: torch.cat(v) for k, v in acc.items()}
        r = out[split]["rank"]
        print(split, "HR@10", (r < 10).float().mean().item(), "oracle keep", (out[split]["rank_oracle_keep"] < 10).float().mean().item(),
              "oracle top", (out[split]["rank_oracle_top"] < 10).float().mean().item(), flush=True)
    torch.save(out, args.out_file)


if __name__ == "__main__":
    main()
