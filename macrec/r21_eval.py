"""mmgen addition (docs/MFD_REPORT.md §21): evaluation of a P1a-family MFD run at the main setting K_f = 50, inference only.

  plain  setting (ii) at K_f = 50: w = 1 / .5 / .5, full view only (deployment path, timed)
  1c     f term = log P_full(f) + a_text log P_text(f) + a_image log P_image(f) (branch terms full view), chosen on VALIDATION
         from w_t, w_i in mfd_mv_infer.GRID_W, a in GRID_A^2 \\ {(0, 0)}, K_f = 50 (round 20's grid with K_f fixed at the
         main setting); applied once to TEST (timed, views in separate calls)
  f diagnostics (valid, test; full view): top-1 accuracy of the true f, mean / median rank of the true f among all
         shared codes (0 = top), true-f recall@10 and @50 (= K_f).
Valid: every item under every f scored once (full view) + the f distributions of the 3 views -> exact ranks of every config
(mfd_mv_infer.ranks_from_terms); sanity: the plain config's valid ranks from the deployment path match the exact ones.
Out: <out_dir>/eval.json, ranks.pt ({split: {"plain", "1c", "f_rank"}}).
"""
import argparse
import itertools
import json
import os
import time

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import GRID_A, GRID_W, BIG, branch_scores, encode, f_logprobs, infer, metrics, ranks_from_terms, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args

K_MAIN = 50


def cfg_name(c):
    return f"{c['variant']}_K{c['K']}_wt{c['w_t']}_wi{c['w_i']}_a{c['a'][1]}-{c['a'][2]}"


def grid():
    out = [dict(variant="plain", K=K_MAIN, w_t=0.5, w_i=0.5, a=(1, 0, 0), b=(1, 0, 0))]
    for wt, wi, at, ai in itertools.product(GRID_W, GRID_W, GRID_A, GRID_A):
        if at == ai == 0:
            continue
        out.append(dict(variant="1c", K=K_MAIN, w_t=wt, w_i=wi, a=(1, at, ai), b=(1, 0, 0)))
    for c in out:
        c["name"] = cfg_name(c)
    return out


def f_diag(fr):
    fr = fr.float()
    return {"f_top1": (fr == 0).float().mean().item(), "f_rank_mean": fr.mean().item(), "f_rank_median": fr.median().item(),
            "f_recall10": (fr < 10).float().mean().item(), "f_recall50": (fr < K_MAIN).float().mean().item()}


def run_deploy(model, cat, ds, coll, vmap, cfg, T, I, bs, limit):
    t_gpu, R, FR = 0.0, [], []
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, bs, limit):
            tgt = tgt.cuda()
            torch.cuda.synchronize(); t0 = time.time()
            user, item, score = infer(model, cat, vb, cfg, T, I, False)
            torch.cuda.synchronize(); t_gpu += time.time() - t0
            B = len(tgt)
            hit = item == tgt[user]
            ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
            better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
            r = torch.bincount(user[better], minlength=B)
            R.append(torch.where(torch.isfinite(ts), r, torch.full_like(r, BIG)).cpu())
    R = torch.cat(R)
    return R, 1000 * t_gpu / len(R)


def f_ranks(model, cat, ds, coll, vmap, bs, limit):
    out = []
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, bs, limit):
            tgt = tgt.cuda()
            lpf = f_logprobs(model, cat, *encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda()))
            tf = cat.code[tgt, 0]
            out.append((lpf > lpf.gather(1, tf[:, None])).sum(1).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke test only: first N users")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    cfgs = grid()
    res, ranks = {"model_class": type(model).__name__, "k_f": K_MAIN, "gpu": torch.cuda.get_device_name(0),
                  "batch_size": args.test_batch_size}, {}
    # ---- validation: exact ranks of every config
    ds, coll = load_split(args, "valid", tok)
    vmap = view_maps(ds.indices)
    allf = torch.arange(cat.nF, device="cuda")
    R = []
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size, args.limit):
            tgt = tgt.cuda()
            lpf = []
            for v in ("full", "text", "image"):
                enc, mask = encode(model, vb[v]["input_ids"].cuda(), vb[v]["attention_mask"].cuda())
                lpf.append(f_logprobs(model, cat, enc, mask))
                if v == "full":
                    B = enc.shape[0]
                    user, item, _, s_t, s_i = branch_scores(model, cat, enc, mask, allf[None].expand(B, -1), T, I)
            z = torch.zeros_like(s_t)
            R.append(ranks_from_terms(cfgs, torch.stack(lpf), user, item, torch.stack([s_t, z, z]), torch.stack([s_i, z, z]), tgt, cat))
    R = torch.cat(R, 1)
    sweep = {c["name"]: dict(c, **metrics(R[j])) for j, c in enumerate(cfgs)}
    best1c = max((n for n in sweep if sweep[n]["variant"] == "1c"), key=lambda n: (sweep[n]["hr10"], sweep[n]["ndcg10"]))
    chosen = {"plain": cfgs[0], "1c": next(c for c in cfgs if c["name"] == best1c)}
    json.dump(sweep, open(f"{args.out_dir}/sweep_valid.json", "w"), indent=1)
    rv_plain, ms_v = run_deploy(model, cat, ds, coll, vmap, chosen["plain"], T, I, args.test_batch_size, args.limit)
    same = (rv_plain.clamp(max=10) == R[0].clamp(max=10)).float().mean().item()
    res["sanity_plain_valid_deploy_vs_exact_share_same_rank"] = same
    assert same > 0.999, same
    frv = f_ranks(model, cat, ds, coll, vmap, args.test_batch_size, args.limit)
    res["valid"] = {"plain": dict(chosen["plain"], **metrics(R[0]), ms_per_user=ms_v),
                    "1c": dict(chosen["1c"], **metrics(R[[c["name"] for c in cfgs].index(best1c)])), "f": f_diag(frv)}
    ranks["valid"] = {"plain": R[0], "1c": R[[c["name"] for c in cfgs].index(best1c)], "f_rank": frv}
    # ---- test: the chosen configs once, deployment path, timed
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    res["test"], ranks["test"] = {}, {}
    for k in ("plain", "1c"):
        r, ms = run_deploy(model, cat, ds, coll, vmap, chosen[k], T, I, args.test_batch_size, args.limit)
        res["test"][k] = dict(chosen[k], **metrics(r), ms_per_user=ms)
        ranks["test"][k] = r
    frt = f_ranks(model, cat, ds, coll, vmap, args.test_batch_size, args.limit)
    res["test"]["f"] = f_diag(frt)
    ranks["test"]["f_rank"] = frt
    json.dump(res, open(f"{args.out_dir}/eval.json", "w"), indent=1, default=str)
    torch.save(ranks, f"{args.out_dir}/ranks.pt")
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
