"""mmgen addition (docs/MFD_REPORT.md §20 Task 1): test-time multi-view ensembling for MFD (P1a), inference only.

For each view v in {full, text, image} the history is encoded in that view (the same strings as training's seqrec_mv;
for P1a the encoder length = number of items in every view) and
  score(item) = fterm(f) + sum_v b_v * (w_t * log P_v(text | f) + w_i * log P_v(image | f)),
  fterm(f)    = sum_v a_v * log P_v(f)            (w_f = 1, a_full = b_full = 1)
Candidate f set = top-K_f by fterm; every catalog item under those f is scored exactly (as mfd_eval.score_batch).
Variants: 1a a = b = (1, 0, 0), w_t, w_i in GRID_W, K_f in GRID_K;   1b a = b = (1, a_text, a_image), a in GRID_A;
          1c a = (1, a_text, a_image), b = (1, 0, 0)  (views only in the f term).
--mode sweep (VALID only): every item under every f is scored once per view; every config's target rank is computed exactly
  from those terms; chosen config per variant = max valid HR@10, tie NDCG@10 -> <out>/sweep_valid.json, chosen.json.
  Sanity: 1a at w = .5/.5, K = 100 must reproduce --check_ranks (the run's reported (ii) valid ranks).
--mode apply (TEST, chosen configs only): the inference path a deployment would run (only the views and K_f the config
  needs), timed twice: views in separate calls and views stacked into one batch -> <out>/apply_test.json + ranks.
  --variants 1c limits apply to the listed variants (seeds 43/44).
--mode kf: the chosen 1c config at K_f in KF_GRID on VALID and TEST (stacked views, timed) -> <out>/kf.json + ranks (§20).
"""
import argparse
import itertools
import json
import os
import time

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import excl_cumsum, load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from utils import parse_dataset_args, parse_global_args, parse_test_args

VIEWS = ("full", "text", "image")
GRID_W = (0.25, 0.5, 0.75, 1.0)
GRID_K = (100, 200)
GRID_A = (0.0, 0.25, 0.5, 1.0)
KF_GRID = (20, 50, 100)
BIG = 10 ** 6


def configs():
    out = []
    for K, wt, wi in itertools.product(GRID_K, GRID_W, GRID_W):
        out.append(dict(variant="1a", K=K, w_t=wt, w_i=wi, a=(1, 0, 0), b=(1, 0, 0)))
        for at, ai in itertools.product(GRID_A, GRID_A):
            if at == ai == 0:
                continue
            out.append(dict(variant="1b", K=K, w_t=wt, w_i=wi, a=(1, at, ai), b=(1, at, ai)))
            out.append(dict(variant="1c", K=K, w_t=wt, w_i=wi, a=(1, at, ai), b=(1, 0, 0)))
    for c in out:
        c["name"] = f"{c['variant']}_K{c['K']}_wt{c['w_t']}_wi{c['w_i']}_a{c['a'][1]}-{c['a'][2]}"
    return out


def encode(model, ids, mask):
    if getattr(model, "struct_mode", None):
        enc, mask = model.struct_encode(ids, mask)
        return enc.last_hidden_state, mask
    return model.encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state, mask


def f_logprobs(model, cat, enc, mask):
    B = enc.shape[0]
    start = model.config.decoder_start_token_id
    return model.last_logprobs(enc, mask, torch.arange(B, device=enc.device), torch.full((B, 1), start, device=enc.device),
                               cols=[cat.tok[0]])[0]


def branch_scores(model, cat, enc, mask, fsel, T, I):
    """log P(text | f), log P(image | f) of every catalog item under the given f codes (passes 2 .. 1 + L of
    mfd_eval.score_batch, unchanged). fsel (B, K) local f. Returns per-candidate user, item, f slot, s_t, s_i."""
    dev = enc.device
    B, K = fsel.shape
    L = cat.L
    start = model.config.decoder_start_token_id
    ar = torch.arange(B, device=dev)
    BK = B * K
    fflat = fsel.reshape(-1)
    rows = ar.repeat_interleave(K)
    ftok = cat.tok[0][fflat]
    z = torch.full_like(ftok, start)
    lv_t, lv_i = cat.levels["t"], cat.levels["i"]
    dec2 = torch.cat([torch.stack([z, ftok, torch.full_like(ftok, T)], 1), torch.stack([z, ftok, torch.full_like(ftok, I)], 1)])
    lp2t, lp2i = model.last_logprobs(enc, mask, torch.cat([rows, rows]), dec2, cols=[cat.tok[lv_t[0]], cat.tok[lv_i[0]]])
    lp = {("t", 0): lp2t[:BK], ("i", 0): lp2i[BK:]}
    meta = {}
    for d in range(1, L):
        for name, branch_tok in (("t", T), ("i", I)):
            lv = cat.levels[name]
            P = cat.pref[(name, d)]
            cnt = P["cnt"][fflat]
            slot = torch.arange(BK, device=dev).repeat_interleave(cnt)
            st = excl_cumsum(cnt)
            local = torch.arange(len(slot), device=dev) - st[slot]
            codes = P["code"][P["ptr"][fflat[slot]] + local]
            toks = torch.stack([cat.tok[lv[j]][codes[:, j]] for j in range(d)], 1)
            dec = torch.cat([torch.full_like(slot, start)[:, None], ftok[slot][:, None],
                             torch.full_like(slot, branch_tok)[:, None], toks], 1)
            lp[(name, d)] = model.last_logprobs(enc, mask, rows[slot], dec, cols=[cat.tok[lv[d]]])[0]
            meta[(name, d)] = st
    cnt_c = cat.f_item_cnt[fflat]
    slot_c = torch.arange(BK, device=dev).repeat_interleave(cnt_c)
    local_c = torch.arange(len(slot_c), device=dev) - excl_cumsum(cnt_c)[slot_c]
    item = cat.items_by_f[cat.f_item_ptr[fflat[slot_c]] + local_c]
    c = cat.code[item]
    s = {}
    for name in ("t", "i"):
        lv = cat.levels[name]
        v = lp[(name, 0)][slot_c, c[:, lv[0]]]
        for d in range(1, L):
            row = meta[(name, d)][slot_c] + cat.pref[(name, d)]["item_local"][item]
            v = v + lp[(name, d)][row, c[:, lv[d]]]
        s[name] = v
    return slot_c // K, item, slot_c % K, s["t"], s["i"]


def view_batches(ds, coll, vmap, bs, limit=0):
    n = min(len(ds), limit) if limit else len(ds)
    for s in range(0, n, bs):
        idx = range(s, min(s + bs, n))
        out = {}
        for v in VIEWS:
            b = []
            for j in idx:
                d = ds.inter_data[j]
                b.append(dict(input_ids="".join(vmap[v][h] for h in d["inters"]), labels=ds.sid2item[d["item"]]))
            out[v] = coll(b)
        yield out, torch.tensor([ds.sid2item[ds.inter_data[j]["item"]] for j in idx])


def ranks_from_terms(cfgs, lpf, user, item, st, si, tgt, cat):
    """Exact target rank for every config. lpf (V, B, nF); user/item (M,); st/si (V, M) for all items of all f."""
    dev = lpf.device
    B = lpf.shape[1]
    fi = cat.code[item, 0]
    lpfc = lpf[:, user, fi]                                                    # (V, M)
    ti = (item == tgt[user]).nonzero().squeeze(1)                              # the target's candidate, one per user
    assert len(ti) == B
    tu = user[ti]
    out = torch.empty(len(cfgs), B, dtype=torch.long)
    for c0 in range(0, len(cfgs), 16):
        cs = cfgs[c0:c0 + 16]
        a = torch.tensor([c["a"] for c in cs], device=dev, dtype=lpf.dtype)       # (C, V)
        bt = torch.tensor([[x * c["w_t"] for x in c["b"]] for c in cs], device=dev, dtype=lpf.dtype)
        bi = torch.tensor([[x * c["w_i"] for x in c["b"]] for c in cs], device=dev, dtype=lpf.dtype)
        fterm = torch.einsum("cv,vbf->cbf", a, lpf)                               # (C, B, nF)
        sc = a @ lpfc + bt @ st + bi @ si                                          # (C, M)
        for j, c in enumerate(cs):
            K = min(c["K"], cat.nF)
            kth = fterm[j].topk(K, dim=1).values[:, -1:]                           # (B, 1)
            keptF = fterm[j] >= kth                                                # ties at the K-th code: kept
            kept = keptF[user, fi]
            ts = torch.empty(B, device=dev, dtype=sc.dtype); ts[tu] = sc[j, ti]
            better = ((sc[j] > ts[user]) | ((sc[j] == ts[user]) & (item < tgt[user]))) & kept
            r = torch.bincount(user[better], minlength=B)
            r = torch.where(keptF[torch.arange(B, device=dev), cat.code[tgt, 0]], r, torch.full_like(r, BIG))
            out[c0 + j] = r.cpu()
    return out


def metrics(r):
    r = r.float()
    nd = torch.where(r < 10, 1 / torch.log2(r + 2), torch.zeros_like(r))
    nd5 = torch.where(r < 5, 1 / torch.log2(r + 2), torch.zeros_like(r))
    return {"hr1": (r < 1).float().mean().item(), "hr5": (r < 5).float().mean().item(), "hr10": (r < 10).float().mean().item(),
            "ndcg5": nd5.mean().item(), "ndcg10": nd.mean().item()}


def infer(model, cat, vb, cfg, T, I, stacked):
    """Deployment path for one config: encode the views the config needs, f term, top-K_f, branch terms of the views with
    b_v > 0, exact scores; returns (user, item, score) of every candidate."""
    need_f = [v for v in range(3) if cfg["a"][v] > 0]
    need_b = [v for v in range(3) if cfg["b"][v] > 0]
    need = sorted(set(need_f) | set(need_b))
    dev = next(model.parameters()).device
    ids = [vb[VIEWS[v]]["input_ids"].to(dev) for v in need]
    masks = [vb[VIEWS[v]]["attention_mask"].to(dev) for v in need]
    B = ids[0].shape[0]
    if stacked and len(need) > 1:
        Lm = max(x.shape[1] for x in ids)
        pad = lambda x: torch.nn.functional.pad(x, (0, Lm - x.shape[1]), value=0)
        enc, mask = encode(model, torch.cat([pad(x) for x in ids]), torch.cat([pad(x) for x in masks]))
        encs = {v: (enc[k * B:(k + 1) * B], mask[k * B:(k + 1) * B]) for k, v in enumerate(need)}
        lpf_all = f_logprobs(model, cat, enc, mask)
        lpf = {v: lpf_all[k * B:(k + 1) * B] for k, v in enumerate(need)}
    else:
        encs = {v: encode(model, x, m) for v, x, m in zip(need, ids, masks)}
        lpf = {v: f_logprobs(model, cat, *encs[v]) for v in need}
    fterm = sum(cfg["a"][v] * lpf[v] for v in need_f)
    K = min(cfg["K"], cat.nF)
    fsel = fterm.topk(K, dim=1).indices
    if stacked and len(need_b) > 1:
        e = torch.cat([encs[v][0] for v in need_b]); m = torch.cat([encs[v][1] for v in need_b])
        u, it, fr, st, si = branch_scores(model, cat, e, m, fsel.repeat(len(need_b), 1), T, I)
        n = len(u) // len(need_b)
        user, item, frank = u[:n], it[:n], fr[:n]
        stv = {v: st[k * n:(k + 1) * n] for k, v in enumerate(need_b)}
        siv = {v: si[k * n:(k + 1) * n] for k, v in enumerate(need_b)}
    else:
        stv, siv = {}, {}
        for v in need_b:
            user, item, frank, stv[v], siv[v] = branch_scores(model, cat, *encs[v], fsel, T, I)
    fv = fterm.gather(1, fsel).reshape(-1)[user * K + frank]
    score = fv + sum(cfg["b"][v] * (cfg["w_t"] * stv[v] + cfg["w_i"] * siv[v]) for v in need_b)
    return user, item, score


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--mode", choices=("sweep", "apply", "kf"), required=True)
    p.add_argument("--variants", default="1a,1b,1c")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--check_ranks", default=None, help="sweep: reported (ii) valid ranks (eval_ii/ranked_valid.pt)")
    p.add_argument("--limit", type=int, default=0, help="smoke test only: first N users")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    cfgs = configs()
    if args.mode == "sweep":
        ds, coll = load_split(args, "valid", tok)
        vmap = view_maps(ds.indices)
        R = []
        allf = torch.arange(cat.nF, device="cuda")
        with torch.no_grad():
            for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size, args.limit):
                tgt = tgt.cuda()
                lpf, st, si = [], [], []
                for v in VIEWS:
                    enc, mask = encode(model, vb[v]["input_ids"].cuda(), vb[v]["attention_mask"].cuda())
                    lpf.append(f_logprobs(model, cat, enc, mask))
                    B = enc.shape[0]
                    user, item, _, s_t, s_i = branch_scores(model, cat, enc, mask, allf[None].expand(B, -1), T, I)
                    st.append(s_t); si.append(s_i)
                R.append(ranks_from_terms(cfgs, torch.stack(lpf), user, item, torch.stack(st), torch.stack(si), tgt, cat))
        R = torch.cat(R, 1)                                                     # (n configs, users)
        res = {c["name"]: dict(c, **metrics(R[j])) for j, c in enumerate(cfgs)}
        base = next(j for j, c in enumerate(cfgs) if c["variant"] == "1a" and c["K"] == 100 and c["w_t"] == 0.5 and c["w_i"] == 0.5)
        sanity = {"config": cfgs[base]["name"], "valid": metrics(R[base])}
        if args.check_ranks:
            X = torch.load(args.check_ranks, weights_only=False)
            ids = X["joint"][0]; tg = X["targets"]
            m = ids[:, :10] == tg[:, None]
            rep = torch.where(m.any(1), m.int().argmax(1), torch.full_like(tg, 10))
            rep = rep[:R.shape[1]]
            same = (R[base].clamp(max=10) == rep).float().mean().item()
            sanity.update(reported_hr10=(rep < 10).float().mean().item(), share_users_same_rank=same, ok=same > 0.999)
        chosen = {}
        for var in ("1a", "1b", "1c"):
            cand = [n for n, c in res.items() if c["variant"] == var]
            chosen[var] = max(cand, key=lambda n: (res[n]["hr10"], res[n]["ndcg10"]))
        json.dump({"sanity": sanity, "results": res}, open(f"{args.out_dir}/sweep_valid.json", "w"), indent=1)
        json.dump({v: res[n] for v, n in chosen.items()}, open(f"{args.out_dir}/chosen.json", "w"), indent=1)
        torch.save({"configs": cfgs, "ranks": R}, f"{args.out_dir}/sweep_valid_ranks.pt")
        print("[sanity]", json.dumps(sanity)); print("[chosen]", json.dumps({v: res[n] for v, n in chosen.items()}))
    elif args.mode == "kf":
        base = json.load(open(f"{args.out_dir}/chosen.json"))["1c"]
        base["a"], base["b"] = tuple(base["a"]), tuple(base["b"])
        out, ranks = {}, {}
        for split in ("valid", "test"):
            ds, coll = load_split(args, split, tok)
            vmap = view_maps(ds.indices)
            for K in KF_GRID:
                cfg = dict(base, K=K)
                t_gpu, R = 0.0, []
                with torch.no_grad():
                    for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size, args.limit):
                        tgt = tgt.cuda()
                        torch.cuda.synchronize(); t0 = time.time()
                        user, item, score = infer(model, cat, vb, cfg, T, I, True)
                        torch.cuda.synchronize(); t_gpu += time.time() - t0
                        B = len(tgt)
                        hit = item == tgt[user]
                        ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
                        better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
                        r = torch.bincount(user[better], minlength=B)
                        R.append(torch.where(torch.isfinite(ts), r, torch.full_like(r, BIG)).cpu())
                R = torch.cat(R)
                key = f"{split}_K{K}"
                out[key] = dict(cfg, split=split, **metrics(R), ms_per_user=1000 * t_gpu / len(R), gpu=torch.cuda.get_device_name(0),
                                batch_size=args.test_batch_size)
                ranks[key] = R
                print(key, json.dumps(out[key]), flush=True)
        json.dump(out, open(f"{args.out_dir}/kf.json", "w"), indent=1)
        torch.save(ranks, f"{args.out_dir}/kf_ranks.pt")
    else:
        chosen = json.load(open(f"{args.out_dir}/chosen.json"))
        chosen = {k: v for k, v in chosen.items() if k in args.variants.split(",")}
        ds, coll = load_split(args, "test", tok)
        vmap = view_maps(ds.indices)
        out, ranks = {}, {}
        for var, cfg in chosen.items():
            cfg["a"], cfg["b"] = tuple(cfg["a"]), tuple(cfg["b"])
            for stacked in (False, True):
                t_gpu, R = 0.0, []
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size, args.limit):
                        tgt = tgt.cuda()
                        torch.cuda.synchronize(); t0 = time.time()
                        user, item, score = infer(model, cat, vb, cfg, T, I, stacked)
                        torch.cuda.synchronize(); t_gpu += time.time() - t0
                        B = len(tgt)
                        hit = item == tgt[user]
                        ts = torch.full((B,), float("-inf"), device="cuda"); ts[user[hit]] = score[hit]
                        better = (score > ts[user]) | ((score == ts[user]) & (item < tgt[user]))
                        r = torch.bincount(user[better], minlength=B)
                        R.append(torch.where(torch.isfinite(ts), r, torch.full_like(r, BIG)).cpu())
                R = torch.cat(R)
                key = f"{var}_{'stacked' if stacked else 'separate'}"
                out[key] = dict(cfg, **metrics(R), ms_per_user=1000 * t_gpu / len(R), gpu=torch.cuda.get_device_name(0),
                                batch_size=args.test_batch_size, peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
                ranks[key] = R
                print(key, json.dumps(out[key]), flush=True)
            a, b = ranks[f"{var}_separate"].clamp(max=10), ranks[f"{var}_stacked"].clamp(max=10)
            out[f"{var}_stacked"]["share_users_same_rank_as_separate"] = (a == b).float().mean().item()
        json.dump(out, open(f"{args.out_dir}/apply_test.json", "w"), indent=1)
        torch.save(ranks, f"{args.out_dir}/apply_test_ranks.pt")


if __name__ == "__main__":
    main()
