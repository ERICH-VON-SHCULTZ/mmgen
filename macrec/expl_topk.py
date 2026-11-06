"""mmgen addition (docs/MFD_REPORT.md §18): save each user's top-N list of a trained MFD model at one setting, and the
target's rank in the full scored list. Same scoring as mfd_eval (score_batch, w_f s_f + w_t s_t + w_i s_i over every
catalog item under the kept K_f codes, ties by item id). Out: {split: {ids (U, N), scores (U, N), rank (U; 0 = top, 10^6 =
target's f not kept), targets (U)}} in dataset order. N (default 300) is deep enough for the per-f cap / penalty of §18 1b.
"""
import argparse

import torch
from transformers import T5Tokenizer

import mfd_eval
from mfd_data import Catalog
from mfd_eval import load_mfd_model, load_split, rank_lists, score_batch
from mfd_model import BRANCH_TOKENS
from utils import parse_dataset_args, parse_global_args, parse_test_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--k_f", type=int, required=True)
    p.add_argument("--w_f", type=float, required=True)
    p.add_argument("--w_t", type=float, required=True)
    p.add_argument("--w_i", type=float, required=True)
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    mfd_eval.TOPN = args.top_n - 1                     # rank_lists keeps TOPN + 1
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    K = min(args.k_f, cat.nF)
    out = {"setting": dict(k_f=K, w_f=args.w_f, w_t=args.w_t, w_i=args.w_i), "item_f": cat.code[:, 0].cpu()}
    for split in args.splits.split(","):
        ds, coll = load_split(args, split, tok)
        I_, S_, R_, G_ = [], [], [], []
        for s in range(0, len(ds), args.test_batch_size):
            batch = [ds[j] for j in range(s, min(s + args.test_batch_size, len(ds)))]
            inp = coll(batch)
            ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
            B = ids.shape[0]
            o = score_batch(model, cat, ids, mask, K, T, I)
            sc = args.w_f * o["s_f"] + args.w_t * o["s_t"] + args.w_i * o["s_i"]
            li, ls = rank_lists(o["user"], o["item"], sc, B, cat.N)
            tgt = torch.tensor([d["labels"] for d in batch], device="cuda")
            hit = o["item"] == tgt[o["user"]]
            ts = torch.full((B,), float("-inf"), device="cuda")
            ts[o["user"][hit]] = sc[hit]
            better = (sc > ts[o["user"]]) | ((sc == ts[o["user"]]) & (o["item"] < tgt[o["user"]]))
            r = torch.bincount(o["user"][better], minlength=B)
            r = torch.where(torch.isfinite(ts), r, torch.full_like(r, 10 ** 6))
            I_.append(li.cpu()); S_.append(ls.cpu()); R_.append(r.cpu()); G_.append(tgt.cpu())
        n = min(x.shape[1] for x in I_)
        out[split] = dict(ids=torch.cat([x[:, :n] for x in I_]), scores=torch.cat([x[:, :n] for x in S_]),
                          rank=torch.cat(R_), targets=torch.cat(G_))
        r = out[split]["rank"].float()
        print(split, "HR@10", (r < 10).float().mean().item(), "list depth", n, flush=True)
    torch.save(out, args.out_file)


if __name__ == "__main__":
    main()
