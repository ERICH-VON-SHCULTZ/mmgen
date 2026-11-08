"""mmgen addition (docs/MFD_REPORT.md §20 Task 2): evaluation of P1a + dense head (dense_head.DenseItemMFDT5), alone and with 1c

  fterm(f)    = w_f * sum_v a_v log P_v(f)  [+ w_d * logsumexp_{item in f} log_softmax(s)(item)   rule B]
  score(item) = w_f * sum_v a_v log P_v(f) + w_t log P(text | f) + w_i log P(image | f) + w_d * log_softmax(s)(item)
v in {full, text, image}, a_full = 1 (a_text, a_image > 0 = the 1c view ensemble in the f term; branch terms and the dense
logits s use the full view), K_f = 100, w = 1 / .5 / .5 (setting (ii)). Candidate f = top-K_f by fterm (rule A: no dense term
in the selection). Grid: rule x W_D x A_GRID^2.
--mode sweep (VALID): every item under every f scored once; exact target ranks for every config; dense-only retrieval; chosen
  per group (max valid HR@10, tie NDCG@10): plain (w_d = 0, a = 0), 1c (w_d = 0), dense_A / dense_B (a = 0),
  dense_1c_A / dense_1c_B (joint), overall.
--mode apply (TEST): every chosen config once, deployment path, timed; per-user diagnostics for the Task 0 rerun (rank,
  target f rank by fterm and by the full-view log P(f), dense rank of the target).
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
from mfd_data import view_maps
from mfd_mv_infer import VIEWS, branch_scores, encode, f_logprobs, metrics, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args

W_D = (0.0, 0.25, 0.5, 1.0, 2.0)
A_GRID = (0.0, 0.25, 0.5)
GROUPS = ("plain", "1c", "dense_A", "dense_B", "dense_1c_A", "dense_1c_B")
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
    cfgs = [dict(rule=r, w_d=w, a_t=at, a_i=ai) for r in ("A", "B") for w in W_D for at in A_GRID for ai in A_GRID
            if not (r == "B" and w == 0)]                                    # rule B with w_d = 0 = rule A
    if args.mode == "apply":
        chosen = json.load(open(f"{args.out_dir}/chosen.json"))
        cfgs, seen = [], set()
        for g in GROUPS:
            c = {k: chosen[g][k] for k in ("rule", "w_d", "a_t", "a_i")}
            if tuple(c.values()) not in seen:
                seen.add(tuple(c.values())); cfgs.append(c)
    name = lambda c: f"{c['rule']}_wd{c['w_d']}_a{c['a_t']}-{c['a_i']}"
    R = {name(c): [] for c in cfgs}
    diag = {name(c): {"f_rank_term": [], "f_rank_full": [], "dense_rank": []} for c in cfgs}
    R_dense, t_gpu = [], {name(c): 0.0 for c in cfgs}
    allf = torch.arange(cat.nF, device="cuda")
    vmap = view_maps(ds.indices)
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size, args.limit):
            tgt = tgt.cuda()
            B = len(tgt)
            ar = torch.arange(B, device="cuda")
            tf = cat.code[tgt, 0]
            if args.mode == "sweep":
                lpf, encs = [], {}
                for v in VIEWS:
                    enc, m = encode(model, vb[v]["input_ids"].cuda(), vb[v]["attention_mask"].cuda())
                    encs[v] = (enc, m)
                    lpf.append(f_logprobs(model, cat, enc, m))
                enc, m = encs["full"]
                dlp = dense_lp(model, enc, m)
                lse = f_lse(dlp, cat)
                user, item, _, st, si = branch_scores(model, cat, enc, m, allf[None].expand(B, -1), T, I)
                fi = cat.code[item, 0]
                br = args.w_t * st + args.w_i * si
                dl = dlp[user, item]
                R_dense.append((dlp > dlp[ar, tgt][:, None]).sum(1).cpu())
                for c in cfgs:
                    fv = args.w_f * (lpf[0] + c["a_t"] * lpf[1] + c["a_i"] * lpf[2])
                    fterm = fv + (c["w_d"] * lse if c["rule"] == "B" else 0)
                    keptF = fterm >= fterm.topk(K, dim=1).values[:, -1:]
                    R[name(c)].append(rank_of(fv[user, fi] + br + c["w_d"] * dl, user, item, tgt, keptF[user, fi], B, keptF[ar, tf]).cpu())
            else:
                for c in cfgs:
                    need = ["full"] + [v for v, a in (("text", c["a_t"]), ("image", c["a_i"])) if a > 0]
                    torch.cuda.synchronize(); t0 = time.time()
                    encs = {v: encode(model, vb[v]["input_ids"].cuda(), vb[v]["attention_mask"].cuda()) for v in need}
                    lp = {v: f_logprobs(model, cat, *encs[v]) for v in need}
                    fv = args.w_f * (lp["full"] + sum({"text": c["a_t"], "image": c["a_i"]}[v] * lp[v] for v in need[1:]))
                    dlp = dense_lp(model, *encs["full"]) if c["w_d"] > 0 else None
                    fterm = fv + (c["w_d"] * f_lse(dlp, cat) if c["rule"] == "B" and dlp is not None else 0)
                    fsel = fterm.topk(K, dim=1).indices
                    user, item, frank, st, si = branch_scores(model, cat, *encs["full"], fsel, T, I)
                    fi = cat.code[item, 0]
                    score = fv[user, fi] + args.w_t * st + args.w_i * si + (c["w_d"] * dlp[user, item] if dlp is not None else 0)
                    torch.cuda.synchronize(); t_gpu[name(c)] += time.time() - t0
                    keptF = torch.zeros(B, cat.nF, dtype=torch.bool, device="cuda"); keptF[ar[:, None], fsel] = True
                    R[name(c)].append(rank_of(score, user, item, tgt, torch.ones_like(user, dtype=torch.bool), B, keptF[ar, tf]).cpu())
                    d = diag[name(c)]
                    d["f_rank_term"].append((fterm > fterm[ar, tf][:, None]).sum(1).cpu())
                    d["f_rank_full"].append((lp["full"] > lp["full"][ar, tf][:, None]).sum(1).cpu())
                    dd = dlp if dlp is not None else dense_lp(model, *encs["full"])
                    d["dense_rank"].append((dd > dd[ar, tgt][:, None]).sum(1).cpu())
                R_dense.append(d["dense_rank"][-1])
    res = {}
    n = sum(len(x) for x in R[name(cfgs[0])])
    for c in cfgs:
        k = name(c)
        r = torch.cat(R[k])
        res[k] = dict(c, k_f=K, w_f=args.w_f, w_t=args.w_t, w_i=args.w_i, **metrics(r))
        if args.mode == "apply":
            res[k].update(ms_per_user=1000 * t_gpu[k] / n, gpu=torch.cuda.get_device_name(0), batch_size=args.test_batch_size)
        R[k] = r
    if args.mode == "sweep":
        rd = torch.cat(R_dense)
        sel = {"plain": lambda c: c["w_d"] == 0 and c["a_t"] == 0 and c["a_i"] == 0, "1c": lambda c: c["w_d"] == 0,
               "dense_A": lambda c: c["rule"] == "A" and c["a_t"] == 0 and c["a_i"] == 0,
               "dense_B": lambda c: (c["rule"] == "B" or c["w_d"] == 0) and c["a_t"] == 0 and c["a_i"] == 0,
               "dense_1c_A": lambda c: c["rule"] == "A", "dense_1c_B": lambda c: c["rule"] == "B" or c["w_d"] == 0,
               "overall": lambda c: True}
        chosen = {g: res[max([k for k in res if f(res[k])], key=lambda k: (res[k]["hr10"], res[k]["ndcg10"]))] for g, f in sel.items()}
        json.dump({"results": res, "dense_only": metrics(rd)}, open(f"{args.out_dir}/sweep_valid.json", "w"), indent=1)
        json.dump(chosen, open(f"{args.out_dir}/chosen.json", "w"), indent=1)
        torch.save(R | {"dense_only": rd}, f"{args.out_dir}/sweep_valid_ranks.pt")
        print("[dense only valid]", json.dumps(metrics(rd))); print("[chosen]", json.dumps(chosen))
    else:
        res["dense_only"] = metrics(torch.cat(R_dense))                  # diagnostic: rank by s alone over the catalog
        json.dump(res, open(f"{args.out_dir}/apply_test.json", "w"), indent=1)
        torch.save({"ranks": R, "diag": {k: {a: torch.cat(b) for a, b in v.items()} for k, v in diag.items()}},
                   f"{args.out_dir}/apply_test_ranks.pt")
        print(json.dumps(res))

if __name__ == "__main__":
    main()
