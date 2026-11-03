"""mmgen addition: diagnostics of codebook-geometry-aware training (docs/MFD_REPORT.md §14), for any trained chain
(plain T5) or MFD (baked / plain MFDT5) checkpoint on the 1+2+2 MD SID, controls included.

f distribution: the first decoder step ([start] -> logits over the f tokens <s0_k>); for the chain it is the first
beam-search step, for MFD position 0 (identical in both rows). Inputs = test_ddp_save's (SeqRecDataset + TestCollatorSave,
full-SID history), which are the MFD eval inputs for non-task-mix runs (checked: f top-1 equals mfd_eval's).
Per split:
  f top-1 / top-20, rank of the true f (0 = top-1);
  near-miss: users whose true f is not the top-1: squared codebook distance between the top-1 f and the true f
    (vs the mean over all f pairs), the true f's rank, and the top-1's neighbour rank around the true f;
  HR@10 by the true f's rank, buckets 1, 2-5, 6-20, 21-50, 51-100, 101-256, from the run's own final list:
    chain eval/<split>/save_seqrec_20.json (20 beams; per-user order by all_users), MFD <list>/ranked_<split>.pt (joint).
Embedding geometry: per level (s0, t0, t1, v0, v1), Pearson and Spearman correlation over code pairs between the
model's embedding-row cosines and the codebook-vector cosines.
Writes <ckpt>/geo_diag.json.
"""
import argparse
import json
import os
import re

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

import geo_embed as ge
from collator import TestCollatorSave
from data import SeqRecDataset
from utils import parse_dataset_args, parse_global_args, parse_test_args

BUCKETS = [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 256)]


def bname(a, b):
    return str(b) if b - a == 1 else f"{a + 1}-{b}"


def spearman(x, y):
    rx, ry = x.argsort().argsort().float(), y.argsort().argsort().float()
    return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()


def cos(x):
    x = torch.nn.functional.normalize(x.float(), dim=1)
    return x @ x.t()


def per_user_hit10(args, split, n, items):
    """rank < 10 of the target in the run's final list, users in dataset order."""
    if args.kind == "chain":
        s = json.load(open(os.path.join(args.ckpt_path, "eval", split, "save_seqrec_20.json")))
        B = len(s["all_outputs"]) // len(s["all_users"])
        hit = torch.zeros(n, dtype=torch.bool)
        for k, u in enumerate(s["all_users"]):
            tgt = s["all_targets"][k].replace(" ", "")
            top = [o.replace(" ", "") for o in s["all_outputs"][k * B:(k + 1) * B][:10]]
            hit[u] = tgt in top
        return hit
    R = torch.load(os.path.join(args.ckpt_path, args.mfd_list, f"ranked_{split}.pt"))
    ids, tg = R["joint"][0][:, :10], R["targets"]
    assert torch.equal(tg, items), "MFD ranked targets != dataset targets"
    return (ids == tg[:, None]).any(1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--kind", choices=["chain", "mfd"], required=True)
    p.add_argument("--mfd_list", default="eval_ii", help="MFD list for HR@10 by f rank (eval_kf20 | eval_ii)")
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--splits", default="valid,test")
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    cls = T5ForConditionalGeneration
    if json.load(open(os.path.join(args.ckpt_path, "config.json"))).get("struct_mode"):   # §16 encoder-structure runs
        from enc_struct import StructT5 as cls
    model = cls.from_pretrained(args.ckpt_path).cuda().eval()
    books = ge.load_codebooks(args.tokenizer_ckpt)
    codes = ge.code_tokens(tok)
    f_ids = torch.tensor([i for i, lv, _ in codes if lv == "s0"])
    f_code = torch.tensor([c for _, lv, c in codes if lv == "s0"])
    cf = books["s0"][f_code]
    d2 = torch.cdist(cf, cf).pow(2)
    off = ~torch.eye(len(f_ids), dtype=torch.bool)
    pos = {int(t): j for j, t in enumerate(f_ids)}
    idx = json.load(open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)))
    sid2item = {"".join(v): int(k) for k, v in idx.items()}
    start = model.config.decoder_start_token_id
    out = {"ckpt": args.ckpt_path, "kind": args.kind, "f_tokens": len(f_ids),
           "mean_sq_dist_all_f_pairs": d2[off].mean().item(), "splits": {}}
    for split in args.splits.split(","):
        ds = SeqRecDataset(args, task="seqrec", mode=split)
        ds.set_prompt(0)
        ranks, top1, items = [], [], []
        for inputs, targets, _ in DataLoader(ds, batch_size=args.test_batch_size, collate_fn=TestCollatorSave(args, tok)):
            ids, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            lg = model(input_ids=ids, attention_mask=mask,
                       decoder_input_ids=torch.full((ids.shape[0], 1), start, device="cuda")).logits[:, -1][:, f_ids.cuda()].float().cpu()
            tf = torch.tensor([pos[tok.convert_tokens_to_ids(re.findall(r"<[^>]+>", t)[0])] for t in targets])   # target's f
            ranks.append((lg > lg.gather(1, tf[:, None])).sum(1))
            top1.append(lg.argmax(1))
            items.append(torch.tensor([sid2item[t.replace(" ", "")] for t in targets]))
        r, t1, it = torch.cat(ranks), torch.cat(top1), torch.cat(items)
        tf = torch.tensor([pos[tok.convert_tokens_to_ids(idx[str(int(i))][0])] for i in it])   # true f per user
        miss = r > 0
        dist = d2[t1[miss], tf[miss]]
        nb_rank = (d2[tf[miss]] < dist[:, None]).sum(1) - 1       # neighbours of the true f closer than the top-1 (excl. itself)
        hit = per_user_hit10(args, split, len(it), it)
        res = {"users": len(r), "f_top1": (r == 0).float().mean().item(), "f_top20": (r < 20).float().mean().item(),
               "f_top100": (r < 100).float().mean().item(), "hr10": hit.float().mean().item(),
               "near_miss": {"users": int(miss.sum()), "mean_sq_dist_top1_to_true": dist.mean().item(),
                             "median_sq_dist_top1_to_true": dist.median().item(),
                             "ratio_to_mean_pair_dist": dist.mean().item() / d2[off].mean().item(),
                             "top1_within_true_f_10nn": (nb_rank < 10).float().mean().item(),
                             "median_true_f_rank": (r[miss] + 1).float().median().item(),
                             "mean_true_f_rank": (r[miss] + 1).float().mean().item()},
               "hr10_by_f_rank": {bname(a, b): {"users": int(((r >= a) & (r < b)).sum()),
                                                "hr10": hit[(r >= a) & (r < b)].float().mean().item() if ((r >= a) & (r < b)).any() else None}
                                  for a, b in BUCKETS}}
        out["splits"][split] = res
        print(split, json.dumps(res), flush=True)
    W = model.shared.weight.detach().float().cpu()
    out["embedding_vs_codebook"] = {}
    for lv in sorted(books):
        sel = [(i, c) for i, l, c in codes if l == lv]
        e = W[torch.tensor([i for i, _ in sel])]
        cv = books[lv][torch.tensor([c for _, c in sel])]
        iu = torch.triu_indices(len(sel), len(sel), 1)
        a, b = cos(e)[iu[0], iu[1]], cos(cv)[iu[0], iu[1]]
        out["embedding_vs_codebook"][lv] = {"pearson": torch.corrcoef(torch.stack([a, b]))[0, 1].item(), "spearman": spearman(a, b)}
    print("embedding vs codebook cosine:", json.dumps(out["embedding_vs_codebook"]))
    json.dump(out, open(os.path.join(args.ckpt_path, "geo_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
