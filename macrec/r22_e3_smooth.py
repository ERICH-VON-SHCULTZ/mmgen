"""mmgen addition (docs/MFD_REPORT.md §22.1, E3): codebook-smoothed P(f) at inference, no training.

c_f = the shared-level MD-RQ-VAE codebook vector of code f (shared_rq.vq_layers.0 of the locked v1 tokenizer, row = the
code number in <s0_k>). K(f', f) = softmax_f(-||c_f' - c_f||^2 / tau) (each row sums to 1; the f' = f term is included).
P'(f) = (1 - lam) P(f) + lam sum_f' P(f') K(f', f), P = the model's full-view softmax over the shared codes. log P' replaces
log P everywhere: the f term of the score AND the choice of the K_f = 50 kept codes; branch terms unchanged (setting (ii),
w = 1 / .5 / .5). tau = m x median pairwise squared codebook distance, m in TAU_MULT; lam in LAMS (lam = 0 = plain).
VALID: every catalog item under every f scored once; exact ranks of every (tau, lam) (mfd_mv_infer.ranks_from_terms);
chosen = max valid HR@10 (tie NDCG@10). Sanity: lam = 0 reproduces the run's r21_eval valid ranks.
TEST: the chosen config and lam = 0 once each, deployment path, timed (H200, batch 64).
Also saved per TEST user for the C-hit / A01-miss diagnostic (tools/r22_e3_report.py): true f, its rank by P, the 64 top f
codes by P; and the 5 nearest codebook neighbours of every f.  Out: <out_dir>/e3.json, e3_ranks.pt
"""
import argparse
import json
import os
import time

import torch
from transformers import T5Tokenizer

from geo_embed import load_codebooks
from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import BIG, branch_scores, encode, f_logprobs, metrics, ranks_from_terms, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args

K = 50
TAU_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMS = (0.0, 0.1, 0.25, 0.5)


def kernel(cat, tok, books, mult):
    codes = [int(t[4:-1]) for t in tok.convert_ids_to_tokens(cat.tok[0].tolist())]      # "<s0_k>" -> k, local f order
    c = books["s0"][torch.tensor(codes)].to(cat.tok[0].device)
    d2 = torch.cdist(c, c).pow(2)
    off = d2[~torch.eye(len(c), dtype=torch.bool, device=c.device)]
    med = off.median()
    return (-d2 / (mult * med)).softmax(1), d2, med.item()


def smooth(lp, Km, lam):
    if lam == 0:
        return lp
    P = lp.exp()
    return ((1 - lam) * P + lam * P @ Km).clamp_min(1e-30).log()


def deploy(model, cat, ds, coll, vmap, Km, lam, T, I, bs, keep_diag=False):
    t_gpu, R, diag = 0.0, [], {"tf": [], "f_rank": [], "top64": []}
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, bs):
            tgt = tgt.cuda()
            torch.cuda.synchronize(); t0 = time.time()
            enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            lp = f_logprobs(model, cat, enc, m)
            lps = smooth(lp, Km, lam)
            fsel = lps.topk(K, dim=1).indices
            user, item, frank, st, si = branch_scores(model, cat, enc, m, fsel, T, I)
            score = lps.gather(1, fsel).reshape(-1)[user * K + frank] + 0.5 * st + 0.5 * si
            torch.cuda.synchronize(); t_gpu += time.time() - t0
            B = len(tgt)
            hit = item == tgt[user]
            ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
            better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
            r = torch.bincount(user[better], minlength=B)
            R.append(torch.where(torch.isfinite(ts), r, torch.full_like(r, BIG)).cpu())
            if keep_diag:
                tf = cat.code[tgt, 0]
                diag["tf"].append(tf.cpu()); diag["f_rank"].append((lp > lp.gather(1, tf[:, None])).sum(1).cpu())
                diag["top64"].append(lp.topk(64, dim=1).indices.cpu())
    R = torch.cat(R)
    return R, 1000 * t_gpu / len(R), {k: torch.cat(v) for k, v in diag.items()} if keep_diag else None


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--tokenizer_ckpt", required=True, help="locked v1 MD-RQ-VAE tokenizer.pth of this SID set")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    books = load_codebooks(args.tokenizer_ckpt)
    kern = {m: kernel(cat, tok, books, m) for m in TAU_MULT}
    d2, med = kern[1.0][1], kern[1.0][2]
    nn5 = (d2 + torch.eye(len(d2), device=d2.device) * 1e9).topk(5, dim=1, largest=False).indices.cpu()
    cfgs = [(m, 0.0) for m in TAU_MULT[:1]] + [(m, lam) for m in TAU_MULT for lam in LAMS if lam > 0]
    # ---- VALID exact ranks of every config
    ds, coll = load_split(args, "valid", tok)
    vmap = view_maps(ds.indices)
    allf = torch.arange(cat.nF, device="cuda")
    base = dict(variant="e3", K=K, w_t=0.5, w_i=0.5, a=(1,), b=(1,))
    R = {c: [] for c in cfgs}
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda()
            enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            lp = f_logprobs(model, cat, enc, m)
            B = enc.shape[0]
            user, item, _, s_t, s_i = branch_scores(model, cat, enc, m, allf[None].expand(B, -1), T, I)
            for c in cfgs:
                lps = smooth(lp, kern[c[0]][0], c[1])
                R[c].append(ranks_from_terms([base], lps[None], user, item, s_t[None], s_i[None], tgt, cat)[0])
    res = {f"tau{c[0]}_lam{c[1]}": dict(tau_mult=c[0], lam=c[1], **metrics(torch.cat(R[c]))) for c in cfgs}
    best = max(res, key=lambda k: (res[k]["hr10"], res[k]["ndcg10"]))
    ref = torch.load(os.path.join(args.ckpt_path, "r21_eval", "ranks.pt"), weights_only=False)["valid"]["plain"]
    same = (torch.cat(R[cfgs[0]]).clamp(max=10) == ref.clamp(max=10)).float().mean().item()
    assert same > 0.999, same
    # ---- TEST once: chosen and plain (lam = 0), deployment path, timed
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    ch = res[best]
    r_ch, ms_ch, _ = deploy(model, cat, ds, coll, vmap, kern[ch["tau_mult"]][0], ch["lam"], T, I, args.test_batch_size)
    r_pl, ms_pl, dg = deploy(model, cat, ds, coll, vmap, kern[1.0][0], 0.0, T, I, args.test_batch_size, keep_diag=True)
    out = {"median_sq_dist": med, "valid": res, "chosen": best, "sanity_lam0_valid_same_rank_as_r21_eval": same,
           "test": {"chosen": dict(ch, **metrics(r_ch), ms_per_user=ms_ch), "plain": dict(metrics(r_pl), ms_per_user=ms_pl)},
           "gpu": torch.cuda.get_device_name(0), "batch_size": args.test_batch_size}
    json.dump(out, open(f"{args.out_dir}/e3.json", "w"), indent=1)
    torch.save({"test_chosen": r_ch, "test_plain": r_pl, "valid": {k: torch.cat(v) for k, v in R.items()}, "nn5": nn5,
                "f_codes_local_to_token": cat.tok[0].cpu(), **dg}, f"{args.out_dir}/e3_ranks.pt")
    print(json.dumps({"chosen": best, "test": out["test"], "lam0_same": same}))


if __name__ == "__main__":
    main()
