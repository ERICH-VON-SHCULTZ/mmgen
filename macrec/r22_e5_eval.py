"""mmgen addition (docs/MFD_REPORT.md §22.5, E5): coarse-region mass on top of A01's score (inference only, no training).

Candidates: every catalog item under the top-K_f f codes by P(f) (unchanged). s(item) = A01 score (log P(f) + .5 log P(text|f) +
.5 log P(image|f)). p(item) = softmax of s over the user's candidates; P(r) = sum of p over the candidates in region r;
s'(item) = s(item) + beta log P(r(item)). Partitions: geo/r22/<D>/e5_regions.pt (r22_e5_regions.py) + (c) femb_G{32,64} = k-means
(sklearn, n_init 10, seed 0) on THIS checkpoint's trained f input-embedding rows (codebook projection + offset, baked); region of an
item = group of its f.
VALID: every item under every f scored once; exact ranks of every partition x beta in BETAS x K_f in KFS; chosen per partition =
max valid HR@10 (tie NDCG@10). Sanity: beta = 0, K_f = 50 reproduces the run's r21_eval valid plain ranks.
TEST once per chosen config (+ plain beta = 0, K_f = 50): one timed deployment pass per K_f (P(f), top-K_f, branch scores), the
region step timed separately per partition; ms/user = pass + region step. Per user and partition: rank and the items above the
target split into same f / other f in the target's region / other region (chosen config and plain, w.r.t. that partition).
Out: <out_dir>/e5.json, e5_ranks.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import BIG, branch_scores, encode, f_logprobs, metrics, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args

BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
KFS = (50, 100)


def region_logmass(s, u, r, G, B):
    """log P(r(item)) per candidate: softmax of s over each user's candidates, summed per region."""
    mx = torch.full((B,), float("-inf"), device=s.device).scatter_reduce(0, u, s, "amax")
    e = (s - mx[u]).exp()
    z = torch.zeros(B, device=s.device).index_add_(0, u, e)
    pm = torch.zeros(B * G, device=s.device).index_add_(0, u * G + r, e / z[u])
    return pm[u * G + r].clamp_min(1e-30).log()


def ranks(score, u, it, fi, r, tgt, tf, tr, B, split=False):
    hit = it == tgt[u]
    ts = torch.full((B,), float("-inf"), device=score.device); ts[u[hit]] = score[hit]
    better = (score > ts[u]) | ((score == ts[u]) & (it < tgt[u]))
    kept = torch.isfinite(ts)
    fix = lambda x, v: torch.where(kept, x, torch.full_like(x, v))
    rk = fix(torch.bincount(u[better], minlength=B), BIG)
    if not split:
        return rk
    same_f = fi == tf[u]
    same_r = r == tr[u]
    cnt = lambda m: fix(torch.bincount(u[better & m], minlength=B), -1)
    return rk, cnt(same_f), cnt(~same_f & same_r), cnt(~same_r)


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--regions", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    fcode = cat.code[:, 0]
    regs = {k: v.cuda() for k, v in torch.load(args.regions, weights_only=False).items()}
    emb = model.get_input_embeddings().weight[cat.tok[0]].detach().float().cpu().numpy()
    info = {}
    for g in (32, 64):
        grp = torch.from_numpy(KMeans(g, n_init=10, random_state=0).fit_predict(emb.astype(np.float64))).long().cuda()
        regs[f"femb_G{g}"] = grp[fcode]
    for k, v in regs.items():
        assert v.shape[0] == cat.N, (k, v.shape, cat.N)
        info[k] = {"G": int(v.max()) + 1, "nmi_ctext": float(nmi(regs["oracle_ctext"].cpu(), v.cpu())),
                   "nmi_cimage": float(nmi(regs["oracle_cimage"].cpu(), v.cpu()))}
    Gs = {k: int(v.max()) + 1 for k, v in regs.items()}
    allf = torch.arange(cat.nF, device="cuda")
    # ---- VALID: exact ranks for every partition x K_f x beta
    ds, coll = load_split(args, "valid", tok)
    vmap = view_maps(ds.indices)
    R = {"plain": {K: [] for K in KFS}}
    R.update({(k, K, b): [] for k in regs for K in KFS for b in BETAS if b > 0})
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda(); B = len(tgt)
            enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            lp = f_logprobs(model, cat, enc, m)
            user, item, _, st, si = branch_scores(model, cat, enc, m, allf[None].expand(B, -1), T, I)
            fi = fcode[item]
            base = lp[user, fi] + 0.5 * st + 0.5 * si
            tf = fcode[tgt]
            for K in KFS:
                kept = (lp >= lp.topk(K, dim=1).values[:, -1:])[user, fi]
                s, u, it, f2 = base[kept], user[kept], item[kept], fi[kept]
                R["plain"][K].append(ranks(s, u, it, f2, f2, tgt, tf, tf, B).cpu())
                for k, reg in regs.items():
                    r = reg[it]
                    lpr = region_logmass(s, u, r, Gs[k], B)
                    for b in BETAS:
                        if b > 0:
                            R[(k, K, b)].append(ranks(s + b * lpr, u, it, f2, r, tgt, tf, reg[tgt], B).cpu())
    plain = {K: torch.cat(R["plain"][K]) for K in KFS}
    ref = torch.load(os.path.join(args.ckpt_path, "r21_eval", "ranks.pt"), weights_only=False)["valid"]["plain"]
    same = (plain[50].clamp(max=10) == ref.clamp(max=10)).float().mean().item()
    assert same > 0.999, same
    valid, chosen = {}, {}
    for k in regs:
        rows = {f"K{K}_b0.0": dict(K=K, beta=0.0, **metrics(plain[K])) for K in KFS}
        rows.update({f"K{K}_b{b}": dict(K=K, beta=b, **metrics(torch.cat(R[(k, K, b)]))) for K in KFS for b in BETAS if b > 0})
        valid[k] = rows
        best = max(rows, key=lambda x: (rows[x]["hr10"], rows[x]["ndcg10"]))
        chosen[k] = (rows[best]["K"], rows[best]["beta"])
    # ---- TEST once per chosen config (+ plain K_f = 50)
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    need = {K for K, _ in chosen.values()} | {50}
    out_r = {k: {"rank": [], "same_f": [], "other_f_same_region": [], "other_region": [],
                 "plain_same_f": [], "plain_other_f_same_region": [], "plain_other_region": []} for k in regs}
    plain_r, t_pass, t_reg, nU = [], {K: 0.0 for K in need}, {k: 0.0 for k in regs}, 0
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda(); B = len(tgt); nU += B
            tf = fcode[tgt]
            for K in sorted(need):
                torch.cuda.synchronize(); t0 = time.time()
                enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
                lp = f_logprobs(model, cat, enc, m)
                fsel = lp.topk(K, dim=1).indices
                user, item, frank, st, si = branch_scores(model, cat, enc, m, fsel, T, I)
                s = lp.gather(1, fsel).reshape(-1)[user * K + frank] + 0.5 * st + 0.5 * si
                torch.cuda.synchronize(); t_pass[K] += time.time() - t0
                fi = fcode[item]
                if K == 50:
                    plain_r.append(ranks(s, user, item, fi, fi, tgt, tf, tf, B).cpu())
                for k, reg in regs.items():
                    r = reg[item]
                    if K == 50:
                        _, a, b_, c = ranks(s, user, item, fi, r, tgt, tf, reg[tgt], B, split=True)
                        for n, x in (("plain_same_f", a), ("plain_other_f_same_region", b_), ("plain_other_region", c)):
                            out_r[k][n].append(x.cpu())
                    if chosen[k][0] != K:
                        continue
                    torch.cuda.synchronize(); t0 = time.time()
                    s2 = s + chosen[k][1] * region_logmass(s, user, r, Gs[k], B) if chosen[k][1] > 0 else s
                    torch.cuda.synchronize(); t_reg[k] += time.time() - t0
                    rk, a, b_, c = ranks(s2, user, item, fi, r, tgt, tf, reg[tgt], B, split=True)
                    for n, x in (("rank", rk), ("same_f", a), ("other_f_same_region", b_), ("other_region", c)):
                        out_r[k][n].append(x.cpu())
    out_r = {k: {n: torch.cat(x) for n, x in v.items()} for k, v in out_r.items()}
    plain_t = torch.cat(plain_r)
    test = {k: dict(K=chosen[k][0], beta=chosen[k][1], **metrics(out_r[k]["rank"]),
                    ms_per_user=1000 * (t_pass[chosen[k][0]] + t_reg[k]) / nU) for k in regs}
    out = {"partitions": info, "valid": valid, "chosen": {k: {"K": v[0], "beta": v[1]} for k, v in chosen.items()},
           "sanity_plain_valid_same_rank_as_r21_eval": same, "test": test,
           "test_plain": dict(K=50, beta=0.0, **metrics(plain_t), ms_per_user=1000 * t_pass[50] / nU),
           "gpu": torch.cuda.get_device_name(0), "batch_size": args.test_batch_size}
    json.dump(out, open(f"{args.out_dir}/e5.json", "w"), indent=1)
    torch.save({"plain": plain_t, "partitions": out_r, "valid_plain": plain}, f"{args.out_dir}/e5_ranks.pt")
    print(json.dumps({"chosen": out["chosen"], "test": {k: round(v["hr10"], 4) for k, v in test.items()},
                      "plain": round(out["test_plain"]["hr10"], 4)}))


if __name__ == "__main__":
    main()
