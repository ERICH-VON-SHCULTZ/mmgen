"""mmgen addition: MFD wider search (docs/MFD_REPORT.md Part B): K_f sweep x score-weight grid, inference only.

score = w_f * log P(f) + w_t * (text-chain terms) + w_i * (image-chain terms), over every catalog item whose f
is among the kept K_f (exact scoring, mfd_eval.score_batch). Ties broken by item id. Metrics as in
evaluate.get_metrics_results: hit@k = [rank < k], ndcg@k = 1 / log2(rank + 2).

--mode sweep  (VALIDATION only): one exact pass with K_f = all catalog f, so every item has (s_f, s_t, s_i) and
              the rank of its f; each setting (K_f, w) is then ranked by masking items whose f rank >= K_f.
              This equals running score_batch at that K_f (the same first K_f roots, the same exact scores).
              Then the timing of the real path (score_batch at K_f + ranking) for each K_f.
              Picks ONE setting: max valid HR@10, tie-break NDCG@10 -> chosen.json. Nothing is computed on test.
--mode apply  --config chosen.json: evaluates exactly that setting on --split (and the stage-1 reference
              K_f=20, w=1) through the real K_f path; HR@10 by target-f rank.
Outputs in --out_dir: sweep_valid.json, sweep_valid_ranks.pt, chosen.json; apply_<split>.json, apply_<split>_ranks.pt
"""
import argparse
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from transformers import T5Tokenizer

from mfd_data import Catalog
from mfd_eval import load_split, rank_lists, score_batch
from mfd_model import BRANCH_TOKENS, MFDT5
from utils import parse_dataset_args, parse_global_args, parse_test_args

W_F = (0.5, 0.75, 1.0)
W_B = (0.5, 1.0, 1.5)
BUCKETS = ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 256))
BIG = 10 ** 6


def bucket_name(a, b):
    return f"{a + 1}" if b == a + 1 else f"{a + 1}-{b}"


def metrics(rank):
    r = rank.float()
    out = {f"hit@{k}": (rank < k).float().mean().item() for k in (1, 5, 10)}
    for k in (5, 10):
        out[f"ndcg@{k}"] = torch.where(rank < k, 1.0 / torch.log2(r + 2), torch.zeros_like(r)).mean().item()
    return out


def summarize(rank, tf_rank, K):
    m = metrics(rank)
    cov = tf_rank < K
    m["coverage"] = cov.float().mean().item()
    m["hit10_given_coverage"] = ((rank < 10) & cov).float().sum().item() / max(cov.sum().item(), 1)
    m["hit10_by_f_rank"] = {bucket_name(a, b): dict(users=int(((tf_rank >= a) & (tf_rank < b)).sum()),
                                                    hit10=(rank[(tf_rank >= a) & (tf_rank < b)] < 10).float().mean().item()
                                                    if ((tf_rank >= a) & (tf_rank < b)).any() else None)
                            for a, b in BUCKETS}
    return m


def setting_name(c):
    return f"K{c['k_f']}_wf{c['w_f']}_wt{c['w_t']}_wi{c['w_i']}"


@torch.no_grad()
def timing(args, model, tokenizer, cat, split, K, T, I, w=(1.0, 1.0, 1.0)):
    """Real inference path at K: score_batch + joint ranking; GPU time per user and peak memory."""
    ds, coll = load_split(args, split, tokenizer)
    dev = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    t_gpu, n_seq, n_cand, U = 0.0, 0, 0, len(ds)
    ranks, tf_rank = [], []
    for s in range(0, U, args.test_batch_size):
        batch = [ds[j] for j in range(s, min(s + args.test_batch_size, U))]
        inp = coll(batch)
        ids, mask = inp["input_ids"].to(dev), inp["attention_mask"].to(dev)
        B = ids.shape[0]
        torch.cuda.synchronize(); t0 = time.time()
        out = score_batch(model, cat, ids, mask, K, T, I)
        lid, _ = rank_lists(out["user"], out["item"], w[0] * out["s_f"] + w[1] * out["s_t"] + w[2] * out["s_i"], B, cat.N)
        torch.cuda.synchronize(); t_gpu += time.time() - t0
        n_seq += out["n_seq"]; n_cand += len(out["item"])
        tgt = torch.tensor([d["labels"] for d in batch], device=dev)
        m = lid == tgt[:, None]
        ranks.append(torch.where(m.any(1), m.int().argmax(1), torch.full_like(tgt, BIG)).cpu())
        tf = cat.code[tgt, 0]
        tf_rank.append((out["lpf_all"] > out["lpf_all"].gather(1, tf[:, None])).sum(1).cpu())
        passes = out["passes"]
    cost = {"k_f": K, "decoder_passes_per_user": passes, "decoder_sequences_per_user": n_seq / U,
            "scored_items_per_user": n_cand / U, "ms_per_user": 1000 * t_gpu / U,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0),
            "batch_size": args.test_batch_size}
    return cost, torch.cat(ranks), torch.cat(tf_rank)


@torch.no_grad()
def sweep(args, model, tokenizer, cat, T, I):
    split = "valid"
    nF = cat.nF
    kfs = [k for k in (20, 50, 100) if k < nF] + [nF]
    grid = [dict(k_f=k, w_f=a, w_t=b, w_i=c) for k in kfs for a, b, c in itertools.product(W_F, W_B, W_B)]
    extra = [dict(k_f=k, w_f=1.0, w_t=1.0, w_i=0.0, mode="text") for k in kfs] + \
            [dict(k_f=k, w_f=1.0, w_t=0.0, w_i=1.0, mode="image") for k in kfs]
    configs = grid + extra
    W = torch.tensor([[c["w_f"], c["w_t"], c["w_i"]] for c in configs], device="cuda")
    Kc = torch.tensor([c["k_f"] for c in configs], device="cuda")
    ds, coll = load_split(args, split, tokenizer)
    dev = next(model.parameters()).device
    U = len(ds)
    all_rank, tf_ranks = [], []
    item_ids = torch.arange(cat.N, device=dev)
    for s in range(0, U, args.test_batch_size):
        batch = [ds[j] for j in range(s, min(s + args.test_batch_size, U))]
        inp = coll(batch)
        ids, mask = inp["input_ids"].to(dev), inp["attention_mask"].to(dev)
        B = ids.shape[0]
        out = score_batch(model, cat, ids, mask, nF, T, I)
        assert len(out["item"]) == B * cat.N, "K_f = all must score every catalog item"
        S = torch.zeros(3, B, cat.N, device=dev)
        FR = torch.zeros(B, cat.N, dtype=torch.long, device=dev)
        S[0, out["user"], out["item"]] = out["s_f"]
        S[1, out["user"], out["item"]] = out["s_t"]
        S[2, out["user"], out["item"]] = out["s_i"]
        FR[out["user"], out["item"]] = out["frank"]
        tgt = torch.tensor([d["labels"] for d in batch], device=dev)
        tfr = FR.gather(1, tgt[:, None]).squeeze(1)                       # target's f rank (0 = top-1)
        tf_ranks.append(tfr.cpu())
        r_b = torch.empty(len(configs), B, dtype=torch.long, device=dev)
        for ci in range(len(configs)):
            sc = (W[ci][:, None, None] * S).sum(0)
            sc = torch.where(FR < Kc[ci], sc, torch.full_like(sc, float("-inf")))
            st = sc.gather(1, tgt[:, None])
            r = (sc > st).sum(1) + ((sc == st) & (item_ids[None] < tgt[:, None])).sum(1)
            r_b[ci] = torch.where(tfr < Kc[ci], r, torch.full_like(r, BIG))
        all_rank.append(r_b.cpu())
    ranks = torch.cat(all_rank, 1)                                        # (configs, U)
    tf_rank = torch.cat(tf_ranks)
    res = {"split": split, "users": U, "n_f": nF, "k_f_values": kfs, "settings": []}
    for ci, c in enumerate(configs):
        res["settings"].append(c | {"name": setting_name(c)} | summarize(ranks[ci], tf_rank, c["k_f"]))
    # choose ONE joint setting on validation: HR@10, tie-break NDCG@10
    joint = [x for x in res["settings"] if "mode" not in x]
    best = max(joint, key=lambda x: (round(x["hit@10"], 10), round(x["ndcg@10"], 10)))
    chosen = {k: best[k] for k in ("k_f", "w_f", "w_t", "w_i", "name")} | \
             {"valid_hit@10": best["hit@10"], "valid_ndcg@10": best["ndcg@10"], "rule": "max valid HR@10, tie NDCG@10"}
    res["chosen"] = chosen
    # real-path timing per K_f (weights do not change the cost)
    res["timing"] = {}
    for K in kfs:
        cost, r_real, _ = timing(args, model, tokenizer, cat, split, K, T, I)
        ci = next(i for i, c in enumerate(configs) if c["k_f"] == K and c["w_f"] == 1.0 and c["w_t"] == 1.0
                  and c["w_i"] == 1.0 and "mode" not in c)
        cost["real_path_hit10"] = (r_real < 10).float().mean().item()
        cost["sweep_hit10"] = (ranks[ci] < 10).float().mean().item()
        cost["real_vs_sweep_rank_mismatch_users"] = int((r_real.clamp(max=20) != ranks[ci].clamp(max=20)).sum())
        res["timing"][str(K)] = cost
        print("[timing]", json.dumps(cost), flush=True)
    torch.save({"configs": configs, "ranks": ranks, "target_f_rank": tf_rank},
               os.path.join(args.out_dir, "sweep_valid_ranks.pt"))
    json.dump(res, open(os.path.join(args.out_dir, "sweep_valid.json"), "w"), indent=1)
    json.dump(chosen, open(os.path.join(args.out_dir, "chosen.json"), "w"), indent=1)
    print("[chosen]", json.dumps(chosen))


def apply(args, model, tokenizer, cat, T, I):
    chosen = json.load(open(args.config))
    ref = dict(k_f=20, w_f=1.0, w_t=1.0, w_i=1.0, name="stage1_reference")
    res = {"split": args.split, "chosen": chosen, "results": {}}
    saved = {}
    for c in (ref, chosen):
        K = min(int(c["k_f"]), cat.nF)
        cost, r, tf_rank = timing(args, model, tokenizer, cat, args.split, K, T, I, w=(c["w_f"], c["w_t"], c["w_i"]))
        res["results"][c["name"]] = {"setting": {k: c[k] for k in ("k_f", "w_f", "w_t", "w_i")},
                                     "cost": cost} | summarize(r, tf_rank, K)
        saved[c["name"]] = r
        print(f"[{c['name']}]", json.dumps(res["results"][c["name"]]), flush=True)
    torch.save(saved | {"target_f_rank": tf_rank}, os.path.join(args.out_dir, f"apply_{args.split}_ranks.pt"))
    json.dump(res, open(os.path.join(args.out_dir, f"apply_{args.split}.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--mode", choices=["sweep", "apply"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test", help="apply mode only")
    p.add_argument("--config", default=None, help="apply mode: chosen.json")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = MFDT5.from_pretrained(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tokenizer, "cuda")
    T, I = tokenizer.convert_tokens_to_ids(list(BRANCH_TOKENS))
    if args.mode == "sweep":
        sweep(args, model, tokenizer, cat, T, I)
    else:
        apply(args, model, tokenizer, cat, T, I)


if __name__ == "__main__":
    main()
