"""mmgen addition (docs/MFD_REPORT.md §20 Task 2): evaluation of P1a + dense head (dense_head.DenseItemMFDT5) with the fused score

  score(item) = w_f log P(f) + w_t log P(text | f) + w_i log P(image | f) + w_d * log_softmax(s)(item)   (s over the catalog)
at setting (ii) (K_f = 100, w = 1 / .5 / .5). Two f-selection rules:
  A  keep the top-K_f f by w_f log P(f) (as P1a)
  B  keep the top-K_f f by w_f log P(f) + w_d * logsumexp_{item in f} log_softmax(s)(item)
--mode sweep (VALID): every item under every f scored once; exact target ranks for rule x w_d in W_D; dense-only retrieval
  HR@10 (rank of the target by s alone over the catalog); chosen w_d per rule = max valid HR@10, tie NDCG@10.
--mode apply (TEST): the chosen w_d of each rule, deployment path (K_f codes only), timed (ms/user, batch 64).
"""
import argparse
import json
import os
import time

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import branch_scores, encode, f_logprobs, metrics
from utils import parse_dataset_args, parse_global_args, parse_test_args

W_D = (0.0, 0.25, 0.5, 1.0, 2.0)
BIG = 10 ** 6


def dense_lp(model, enc, mask):
    return model.dense_logits(model.start_state(enc, mask)).float().log_softmax(-1)          # (B, N)


def f_lse(dlp, cat):
    """logsumexp of the dense log-probs over the items of each f: (B, nF)."""
    fi = cat.code[torch.arange(cat.N, device=dlp.device), 0]
    s = torch.zeros(dlp.shape[0], cat.nF, device=dlp.device).index_add_(1, fi, dlp.exp())
    return s.clamp_min(1e-30).log()


def rank_of(score, user, item, tgt, kept, B, tf_kept):
    ts = torch.full((B,), float("-inf"), device=score.device)
    hit = item == tgt[user]
    ts[user[hit]] = score[hit]
    better = ((score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))) & kept
    r = torch.bincount(user[better], minlength=B)
    return torch.where(tf_kept, r, torch.full_like(r, BIG))


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--mode", choices=("sweep", "apply"), required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_f", type=int, default=100)
    p.add_argument("--w_f", type=float, default=1.0)
    p.add_argument("--w_t", type=float, default=0.5)
    p.add_argument("--w_i", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0, help="smoke test only: first N users")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    assert type(model).__name__ == "DenseItemMFDT5", type(model).__name__
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    K = min(args.k_f, cat.nF)
    split = "valid" if args.mode == "sweep" else "test"
    ds, coll = load_split(args, split, tok)
    n = min(len(ds), args.limit) if args.limit else len(ds)
    rules = [(r, w) for r in ("A", "B") for w in W_D]
    if args.mode == "apply":
        chosen = json.load(open(f"{args.out_dir}/chosen.json"))
        rules = [(r, chosen[r]["w_d"]) for r in ("A", "B")]
    R = {k: [] for k in rules}
    R_dense, t_gpu = [], {k: 0.0 for k in rules}
    allf = torch.arange(cat.nF, device="cuda")
    with torch.no_grad():
        for s in range(0, n, args.test_batch_size):
            batch = [ds[j] for j in range(s, min(s + args.test_batch_size, n))]
            inp = coll(batch)
            ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
            tgt = torch.tensor([d["labels"] for d in batch], device="cuda")
            B = len(tgt)
            ar = torch.arange(B, device="cuda")
            tf = cat.code[tgt, 0]
            if args.mode == "sweep":
                enc, m = encode(model, ids, mask)
                lpf = f_logprobs(model, cat, enc, m)
                dlp = dense_lp(model, enc, m)
                lse = f_lse(dlp, cat)
                user, item, _, st, si = branch_scores(model, cat, enc, m, allf[None].expand(B, -1), T, I)
                fi = cat.code[item, 0]
                base = args.w_f * lpf[user, fi] + args.w_t * st + args.w_i * si
                dl = dlp[user, item]
                td = dlp[ar, tgt]
                R_dense.append((dlp > td[:, None]).sum(1).cpu())
                for (rule, wd) in rules:
                    fterm = args.w_f * lpf + (wd * lse if rule == "B" else 0)
                    keptF = fterm >= fterm.topk(K, dim=1).values[:, -1:]
                    R[(rule, wd)].append(rank_of(base + wd * dl, user, item, tgt, keptF[user, fi], B, keptF[ar, tf]).cpu())
            else:
                for (rule, wd) in rules:
                    torch.cuda.synchronize(); t0 = time.time()
                    enc, m = encode(model, ids, mask)
                    lpf = f_logprobs(model, cat, enc, m)
                    dlp = dense_lp(model, enc, m)
                    fterm = args.w_f * lpf + (wd * f_lse(dlp, cat) if rule == "B" else 0)
                    fsel = fterm.topk(K, dim=1).indices
                    user, item, frank, st, si = branch_scores(model, cat, enc, m, fsel, T, I)
                    fi = cat.code[item, 0]
                    score = args.w_f * lpf[user, fi] + args.w_t * st + args.w_i * si + wd * dlp[user, item]
                    torch.cuda.synchronize(); t_gpu[(rule, wd)] += time.time() - t0
                    keptF = torch.zeros(B, cat.nF, dtype=torch.bool, device="cuda"); keptF[ar[:, None], fsel] = True
                    R[(rule, wd)].append(rank_of(score, user, item, tgt, torch.ones_like(user, dtype=torch.bool), B, keptF[ar, tf]).cpu())
                R_dense.append((dlp > dlp[ar, tgt][:, None]).sum(1).cpu())
    res = {}
    for (rule, wd), v in R.items():
        r = torch.cat(v)
        key = f"{rule}_wd{wd}"
        res[key] = dict(rule=rule, w_d=wd, k_f=K, w_f=args.w_f, w_t=args.w_t, w_i=args.w_i, **metrics(r))
        if args.mode == "apply":
            res[key].update(ms_per_user=1000 * t_gpu[(rule, wd)] / n, gpu=torch.cuda.get_device_name(0), batch_size=args.test_batch_size)
        R[(rule, wd)] = r
    if args.mode == "sweep":
        rd = torch.cat(R_dense)
        out = {"results": res, "dense_only": metrics(rd)}
        chosen = {}
        for rule in ("A", "B"):
            ks = [k for k in res if res[k]["rule"] == rule]
            chosen[rule] = res[max(ks, key=lambda k: (res[k]["hr10"], res[k]["ndcg10"]))]
        both = max(chosen.values(), key=lambda c: (c["hr10"], c["ndcg10"]))
        chosen["overall"] = both
        json.dump(out, open(f"{args.out_dir}/sweep_valid.json", "w"), indent=1)
        json.dump(chosen, open(f"{args.out_dir}/chosen.json", "w"), indent=1)
        torch.save({k: v for k, v in R.items()} | {"dense_only": rd}, f"{args.out_dir}/sweep_valid_ranks.pt")
        print("[dense only valid]", json.dumps(out["dense_only"])); print("[chosen]", json.dumps(chosen))
    else:
        res["dense_only"] = metrics(torch.cat(R_dense))                  # diagnostic: rank by s alone over the catalog
        json.dump(res, open(f"{args.out_dir}/apply_test.json", "w"), indent=1)
        torch.save({f"{r}_wd{w}": v for (r, w), v in R.items()}, f"{args.out_dir}/apply_test_ranks.pt")
        print(json.dumps(res))


if __name__ == "__main__":
    main()
