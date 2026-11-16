"""mmgen addition (docs/MFD_REPORT.md §22.2, E2 step 2): evaluation of A01 + aux t1 / i1 heads (r22_aux.py), K_f = 50.

score(item) = log P(f) + .5 log P(text | f) + .5 log P(image | f) + w [log P_aux(t1(item)) + log P_aux(i1(item))]
candidates = every catalog item under the top-K_f f codes by P(f) (unchanged; the aux term only re-orders), w in W_GRID.
VALID: exact ranks for every w (every item under every f scored once); chosen w = max valid HR@10 (tie NDCG@10). w = 0 is the
run's own plain A01-style score (sanity: equals its r21_eval valid plain ranks). TEST once: chosen w and w = 0, deployment
path, timed; f diagnostics (r21_eval.f_diag); per user rank split n_other / n_within (as r21_gain_diag) for the chosen w.
Out: <out_dir>/e2_eval.json, e2_ranks.pt
"""
import argparse
import json
import os
import time

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import BIG, branch_scores, encode, f_logprobs, metrics, view_batches
from r21_eval import f_diag
from utils import parse_dataset_args, parse_global_args, parse_test_args

K = 50
W_GRID = (0.0, 0.25, 0.5, 1.0)


def aux_maps(model, cat):
    """local catalog code index of levels t1 / i1 -> index in the aux heads' code lists (via token ids)."""
    out = []
    for lv, toks in ((cat.levels["t"][0], model.aux_t_tok), (cat.levels["i"][0], model.aux_i_tok)):
        pos = {int(t): j for j, t in enumerate(toks.tolist())}
        out.append(torch.tensor([pos[int(t)] for t in cat.tok[lv].tolist()], device="cuda"))
    return out


def rank_split(score, user, item, fi, tgt, tf, B):
    hit = item == tgt[user]
    ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
    better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
    kept = torch.isfinite(ts)
    own = fi == tf[user]
    r = torch.bincount(user[better], minlength=B)
    no = torch.bincount(user[better & ~own], minlength=B)
    nw = torch.bincount(user[better & own], minlength=B)
    big = lambda x, v: torch.where(kept, x, torch.full_like(x, v))
    return big(r, BIG), big(no, -1), big(nw, -1)


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    assert type(model).__name__ == "AuxGroupedItemMFDT5", type(model).__name__
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    mt, mi = aux_maps(model, cat)
    lt, li = cat.levels["t"][0], cat.levels["i"][0]
    allf = torch.arange(cat.nF, device="cuda")
    # ---- VALID
    ds, coll = load_split(args, "valid", tok)
    vmap = view_maps(ds.indices)
    R = {w: [] for w in W_GRID}
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda()
            B = len(tgt)
            enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            lp = f_logprobs(model, cat, enc, m)
            at, ai = model.aux_logprobs(enc, m)
            user, item, _, st, si = branch_scores(model, cat, enc, m, allf[None].expand(B, -1), T, I)
            fi = cat.code[item, 0]
            base = lp[user, fi] + 0.5 * st + 0.5 * si
            extra = at[user, mt[cat.code[item, lt]]] + ai[user, mi[cat.code[item, li]]]
            keptF = lp >= lp.topk(K, dim=1).values[:, -1:]
            kept = keptF[user, fi]
            tf = cat.code[tgt, 0]
            for w in W_GRID:
                r, _, _ = rank_split((base + w * extra)[kept], user[kept], item[kept], fi[kept], tgt, tf, B)
                R[w].append(r.cpu())
    res = {f"w{w}": dict(w=w, **metrics(torch.cat(R[w]))) for w in W_GRID}
    best = max(res, key=lambda k: (res[k]["hr10"], res[k]["ndcg10"]))
    wbest = res[best]["w"]
    ref = torch.load(os.path.join(args.ckpt_path, "r21_eval", "ranks.pt"), weights_only=False)["valid"]["plain"]
    same = (torch.cat(R[0.0]).clamp(max=10) == ref.clamp(max=10)).float().mean().item()
    assert same > 0.999, same
    # ---- TEST once: chosen w and w = 0
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    out_r, tim = {}, {}
    for w in sorted({wbest, 0.0}):
        Rr, No, Nw, FR, t_gpu = [], [], [], [], 0.0
        with torch.no_grad():
            for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
                tgt = tgt.cuda()
                B = len(tgt)
                torch.cuda.synchronize(); t0 = time.time()
                enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
                lp = f_logprobs(model, cat, enc, m)
                fsel = lp.topk(K, dim=1).indices
                user, item, frank, st, si = branch_scores(model, cat, enc, m, fsel, T, I)
                score = lp.gather(1, fsel).reshape(-1)[user * K + frank] + 0.5 * st + 0.5 * si
                if w > 0:
                    at, ai = model.aux_logprobs(enc, m)
                    score = score + w * (at[user, mt[cat.code[item, lt]]] + ai[user, mi[cat.code[item, li]]])
                torch.cuda.synchronize(); t_gpu += time.time() - t0
                tf = cat.code[tgt, 0]
                r, no, nw = rank_split(score, user, item, cat.code[item, 0], tgt, tf, B)
                Rr.append(r.cpu()); No.append(no.cpu()); Nw.append(nw.cpu())
                FR.append((lp > lp.gather(1, tf[:, None])).sum(1).cpu())
        out_r[w] = {"rank": torch.cat(Rr), "n_other": torch.cat(No), "n_within": torch.cat(Nw), "f_rank": torch.cat(FR)}
        tim[w] = 1000 * t_gpu / len(out_r[w]["rank"])
    out = {"valid": res, "chosen_w": wbest, "sanity_w0_valid_same_rank_as_r21_eval": same,
           "test": {f"w{w}": dict(w=w, **metrics(out_r[w]["rank"]), ms_per_user=tim[w]) for w in out_r},
           "test_f": f_diag(out_r[0.0]["f_rank"]), "gpu": torch.cuda.get_device_name(0), "batch_size": args.test_batch_size}
    json.dump(out, open(f"{args.out_dir}/e2_eval.json", "w"), indent=1)
    torch.save({f"w{w}": v for w, v in out_r.items()} | {"valid": {f"w{w}": torch.cat(R[w]) for w in W_GRID}}, f"{args.out_dir}/e2_ranks.pt")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
