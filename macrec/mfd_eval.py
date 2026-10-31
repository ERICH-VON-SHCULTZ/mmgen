"""mmgen addition: MFD stage-1 inference, evaluation, diagnostics and sanity checks.

Per user, three batched decoder passes (no KV cache; each pass re-runs its short sequences):
  pass 1  [start]                 -> log P(f)            ; keep the top K_f catalog shared codes
  pass 2  [start, f, <T>] / <I>   -> log P(t1|f), log P(i1|f)        for each kept f (2 K_f sequences)
  pass 3  [start, f, <T>, t1] / [start, f, <I>, i1] -> log P(t2|f,t1), log P(i2|f,i1)
          for every (f, t1) / (f, i1) prefix that exists in the catalog under a kept f
All log-probabilities are full-vocabulary log-softmax values (as in the chain decoder's beam scores).
Every catalog item whose f is kept is scored exactly:
  (d) joint  : log P(f) + [log P(t1|f) + log P(t2|f,t1)] + [log P(i1|f) + log P(i2|f,i1)]   (w_t = w_i = 1)
  (a) text   : log P(f) + text terms        (b) image : log P(f) + image terms
  (c) merge  : top-20 lists of (a) and (b) merged with MQL4GRec's rule (ensemble.get_topk_results_ensemble)
Ties (equal scores) are broken by item id. Metrics: evaluate.get_metrics_results on the ranked lists.

Usage:
  python mfd_eval.py --ckpt_path <run> --dataset Instruments --index_file .index_md122_a05_ep500_reassign.json \
      --image_index_file <same> --data_path <data> --out_dir <run>/mfd_eval [--k_f 20] [--sanity]
  python mfd_eval.py --random_init ... --sanity_only      (sanity checks on an untrained model)
  python mfd_eval.py --chain_ckpt <chain run> ...           (also: f accuracy of the chain decoder)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from ensemble import get_topk_results_ensemble
from evaluate import get_metrics_results
from mfd_data import Catalog, MFDCollator, MFDDataset, branch_tensors
from mfd_model import BRANCH_TOKENS, MFDT5
from utils import parse_dataset_args, parse_global_args, parse_test_args

METRICS = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
TOPN = 20


def excl_cumsum(x):
    return torch.cumsum(x, 0) - x


@torch.no_grad()
def score_batch(model, cat, ids, mask, K, T, I):
    """Exact MFD scores for one user batch. Returns per-candidate tensors (flat over users)."""
    dev = ids.device
    B = ids.shape[0]
    enc = model.encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
    start = model.config.decoder_start_token_id
    ar = torch.arange(B, device=dev)
    # pass 1: log P(f), restricted to the shared codes that exist in the catalog
    lpf_all = model.last_logprobs(enc, mask, ar, torch.full((B, 1), start, device=dev), cols=[cat.tok[0]])[0]
    lpf_top, fsel = lpf_all.topk(K, dim=1)                                   # (B, K) local f index
    BK = B * K
    fflat = fsel.reshape(-1)
    rows = ar.repeat_interleave(K)
    ftok = cat.tok[0][fflat]
    z = torch.full_like(ftok, start)
    # pass 2: <T> and <I> after each kept f
    dec2 = torch.cat([torch.stack([z, ftok, torch.full_like(ftok, T)], 1),
                      torch.stack([z, ftok, torch.full_like(ftok, I)], 1)])
    lp2t, lp2i = model.last_logprobs(enc, mask, torch.cat([rows, rows]), dec2, cols=[cat.tok[1], cat.tok[3]])
    lpt1, lpi1 = lp2t[:BK], lp2i[BK:]                                       # (BK, n_t1), (BK, n_i1)
    # pass 3: every catalog prefix (f, t1) / (f, i1) under the kept f
    seqs, meta = [], {}
    for name, branch_tok, lvl in (("t", T, 1), ("i", I, 3)):
        P = cat.pref[name]
        cnt = P["cnt"][fflat]
        slot = torch.arange(BK, device=dev).repeat_interleave(cnt)
        st = excl_cumsum(cnt)
        local = torch.arange(len(slot), device=dev) - st[slot]
        code1 = P["code"][P["ptr"][fflat[slot]] + local]
        dec3 = torch.stack([torch.full_like(slot, start), ftok[slot], torch.full_like(slot, branch_tok),
                            cat.tok[lvl][code1]], 1)
        seqs.append((rows[slot], dec3))
        meta[name] = dict(start=st, n=len(slot))
    lp3t = model.last_logprobs(enc, mask, seqs[0][0], seqs[0][1], cols=[cat.tok[2]])[0]
    lp3i = model.last_logprobs(enc, mask, seqs[1][0], seqs[1][1], cols=[cat.tok[4]])[0]
    # candidates: every item whose f is kept
    cnt_c = cat.f_item_cnt[fflat]
    slot_c = torch.arange(BK, device=dev).repeat_interleave(cnt_c)
    local_c = torch.arange(len(slot_c), device=dev) - excl_cumsum(cnt_c)[slot_c]
    item = cat.items_by_f[cat.f_item_ptr[fflat[slot_c]] + local_c]
    c = cat.code[item]
    row_t = meta["t"]["start"][slot_c] + cat.pref["t"]["item_local"][item]
    row_i = meta["i"]["start"][slot_c] + cat.pref["i"]["item_local"][item]
    s_f = lpf_top.reshape(-1)[slot_c]
    s_t = lpt1[slot_c, c[:, 1]] + lp3t[row_t, c[:, 2]]
    s_i = lpi1[slot_c, c[:, 3]] + lp3i[row_i, c[:, 4]]
    return dict(user=slot_c // K, item=item, s_f=s_f, s_t=s_t, s_i=s_i, fsel=fsel, lpf_all=lpf_all,
                n_seq=B * (1 + 2 * K) + meta["t"]["n"] + meta["i"]["n"],
                n_pref=(meta["t"]["n"], meta["i"]["n"]))


def rank_lists(user, item, score, B, N):
    """Top-TOPN items and scores per user; ties broken by item id. Returns (B, TOPN+1) ids / scores."""
    dev = score.device
    cnt = torch.bincount(user, minlength=B)
    pos = torch.arange(len(user), device=dev) - excl_cumsum(cnt)[user]
    C = int(cnt.max())
    ids = torch.full((B, C), N, dtype=torch.long, device=dev)
    sc = torch.full((B, C), float("-inf"), device=dev)
    ids[user, pos] = item
    sc[user, pos] = score
    o = torch.argsort(ids, dim=1)                                    # item-id order first ...
    ids, sc = ids.gather(1, o), sc.gather(1, o)
    s2, o2 = torch.sort(sc, dim=1, descending=True, stable=True)     # ... then stable sort by score
    n = min(TOPN + 1, C)
    return ids.gather(1, o2)[:, :n], s2[:, :n]


def load_split(args, split, tokenizer):
    ds = MFDDataset(args, split)
    coll = MFDCollator(args, tokenizer)
    return ds, coll


def run_split(args, model, tokenizer, cat, split, T, I):
    ds, coll = load_split(args, split, tokenizer)
    dev = next(model.parameters()).device
    lists = {m: ([], []) for m in ("text", "image", "joint")}
    targets, fsel_all, lpf_rank, n_seq, n_pref = [], [], [], 0, [0, 0]
    t_gpu = 0.0
    torch.cuda.reset_peak_memory_stats()
    bs = args.test_batch_size
    for s in range(0, len(ds), bs):
        batch = [ds[j] for j in range(s, min(s + bs, len(ds)))]
        inp = coll(batch)
        ids, mask = inp["input_ids"].to(dev), inp["attention_mask"].to(dev)
        B = ids.shape[0]
        torch.cuda.synchronize(); t0 = time.time()
        out = score_batch(model, cat, ids, mask, args.k_f, T, I)
        res = {"joint": rank_lists(out["user"], out["item"], out["s_f"] + out["s_t"] + out["s_i"], B, cat.N)}
        torch.cuda.synchronize(); t_gpu += time.time() - t0      # cost = MFD (joint) inference only
        res["text"] = rank_lists(out["user"], out["item"], out["s_f"] + out["s_t"], B, cat.N)
        res["image"] = rank_lists(out["user"], out["item"], out["s_f"] + out["s_i"], B, cat.N)
        for m, (i_, s_) in res.items():
            lists[m][0].append(i_.cpu()); lists[m][1].append(s_.cpu())
        tgt = torch.tensor([d["labels"] for d in batch], device=dev)
        targets.append(tgt.cpu())
        fsel_all.append(out["fsel"].cpu())
        tf = cat.code[tgt, 0]
        lpf_rank.append((out["lpf_all"] > out["lpf_all"].gather(1, tf[:, None])).sum(1).cpu())   # 0 = top-1
        n_seq += out["n_seq"]; n_pref[0] += out["n_pref"][0]; n_pref[1] += out["n_pref"][1]
    U = len(ds)
    targets = torch.cat(targets)
    tf = cat.code[targets.to(cat.code.device), 0].cpu()
    fsel = torch.cat(fsel_all)
    covered = (fsel == tf[:, None]).any(1)
    frank = torch.cat(lpf_rank)
    out = {"users": U, "k_f": args.k_f,
           "coverage": covered.float().mean().item(),
           "f_top1_acc": (frank == 0).float().mean().item(),
           f"f_top{args.k_f}_acc": (frank < args.k_f).float().mean().item(),
           "cost": {"decoder_passes_per_user": 3, "decoder_sequences_per_user": n_seq / U,
                    "pass3_text_prefixes_per_user": n_pref[0] / U, "pass3_image_prefixes_per_user": n_pref[1] / U,
                    "ms_per_user": 1000 * t_gpu / U, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu": torch.cuda.get_device_name(0), "batch_size": args.test_batch_size},
           "modes": {}}
    ranked = {}
    for m in lists:
        ids = torch.cat(lists[m][0]); sc = torch.cat(lists[m][1])
        ranked[m] = (ids, sc)
        hits = (ids[:, :TOPN] == targets[:, None]).int().tolist()
        met = {k: v / U for k, v in get_metrics_results(hits, METRICS).items()}
        h10 = (ids[:, :10] == targets[:, None]).any(1)
        met["coverage"] = out["coverage"]
        met["hit10_given_coverage"] = (h10 & covered).float().sum().item() / max(covered.sum().item(), 1)
        if m != "joint":   # equal scores among ranks 1..11 (inside the top 10 or at its boundary)
            s11 = sc[:, :11]
            met["users_with_tie_in_top10"] = int(((s11[:, 1:] == s11[:, :-1]) & torch.isfinite(s11[:, 1:])).any(1).sum())
        out["modes"][m] = met
    # (c) C-style merge of the top-20 lists of (a) and (b), MQL4GRec's rule, via MACRec's ensemble code
    def info(m):
        ids, sc = ranked[m]
        return [(u, [int(targets[u])], [[int(x)] if x < cat.N else [-1] for x in ids[u, :TOPN]],
                 [float(x) for x in sc[u, :TOPN]]) for u in range(U)]
    t0 = time.time()
    rows = get_topk_results_ensemble(info("text"), info("image"))
    merge_s = time.time() - t0
    met = {k: v / U for k, v in get_metrics_results(rows, METRICS).items()}
    met["coverage"] = out["coverage"]
    met["hit10_given_coverage"] = sum(sum(r[:10]) > 0 for r in rows) / max(covered.sum().item(), 1)
    met["merge_ms_per_user_cpu"] = 1000 * merge_s / U
    out["modes"]["merge"] = met
    return out, ranked, targets


@torch.no_grad()
def chain_f_accuracy(args, split, K):
    """f prediction of the chain decoder (first decoding step), same restriction to catalog f codes."""
    tok = T5Tokenizer.from_pretrained(args.chain_ckpt)
    model = T5ForConditionalGeneration.from_pretrained(args.chain_ckpt).cuda().eval()
    cat = Catalog(args, tok, "cuda")
    ds = MFDDataset(args, split)
    coll = MFDCollator.__new__(MFDCollator); coll.tokenizer = tok
    from mfd_data import item_token_table
    coll.tab = item_token_table(args, tok); coll.T = coll.I = tok.pad_token_id
    ranks = []
    for s in range(0, len(ds), args.test_batch_size):
        batch = [ds[j] for j in range(s, min(s + args.test_batch_size, len(ds)))]
        inp = coll(batch)
        ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
        o = model(input_ids=ids, attention_mask=mask,
                  decoder_input_ids=torch.full((ids.shape[0], 1), model.config.decoder_start_token_id, device="cuda"))
        lp = o.logits[:, -1].float().log_softmax(-1)[:, cat.tok[0]]
        tf = cat.code[torch.tensor([d["labels"] for d in batch], device="cuda"), 0]
        ranks.append((lp > lp.gather(1, tf[:, None])).sum(1).cpu())
    r = torch.cat(ranks)
    res = {"f_top1_acc": (r == 0).float().mean().item(), f"f_top{K}_acc": (r < K).float().mean().item()}
    # f of the 20 final beams (what the chain's ranked list can contain)
    save = os.path.join(args.chain_ckpt, "eval", split, "save_seqrec_20.json")
    if os.path.exists(save):
        sv = json.load(open(save))
        outs, tg = sv["all_outputs"], sv["all_targets"]
        f_of = lambda x: x.strip().replace(" ", "").split(">")[0] + ">"
        cov, hit_cov = 0, 0
        for u in range(len(tg)):
            beams = outs[u * 20:(u + 1) * 20]
            if f_of(tg[u]) in {f_of(b) for b in beams}:
                cov += 1
                sc = sv["all_scores"][u * 20:(u + 1) * 20]
                top10 = [b.strip().replace(" ", "") for _, b in sorted(zip(sc, beams), key=lambda x: -x[0])[:10]]
                hit_cov += tg[u] in top10
        res["target_f_among_20_beams"] = cov / len(tg)
        res["hit10_given_target_f_among_beams"] = hit_cov / max(cov, 1)
    return res


@torch.no_grad()
def sanity(args, model, tokenizer, cat, T, I):
    """1) branch independence, 2) exact scoring vs manual computation, 3) f loss counted once."""
    dev = next(model.parameters()).device
    ds, coll = load_split(args, "test", tokenizer)
    rng = random.Random(0)
    batch = [ds[j] for j in rng.sample(range(len(ds)), 8)]
    inp = {k: v.to(dev) for k, v in coll(batch).items()}
    rep = {}
    # 1) changing one branch's codes must leave every logit of the other branch unchanged
    base = model(**inp).logits
    for b_mod, b_chk, pos in ((0, 1, 3), (1, 0, 3)):
        x = {k: v.clone() for k, v in inp.items()}
        lvl = 1 if b_mod == 0 else 3
        other = cat.tok[lvl][torch.randint(len(cat.tok[lvl]), (8,), generator=torch.Generator().manual_seed(1)).to(dev)]
        x["decoder_input_ids"][:, b_mod, pos] = other                 # t1 (text) or i1 (image) input
        x["labels"][:, b_mod, 2] = other
        new = model(**x).logits
        name = "text->image" if b_mod == 0 else "image->text"
        rep[f"independence_{name}"] = {
            "other_branch_identical": bool(torch.equal(base[:, b_chk], new[:, b_chk])),
            "other_branch_max_abs_diff": float((base[:, b_chk] - new[:, b_chk]).abs().max()),
            "modified_branch_changed": bool(not torch.equal(base[:, b_mod], new[:, b_mod]))}
    # 3) f counted once: model loss == mean CE over exactly 5 targets per example (f, t1, t2, i1, i2)
    out = model(**inp)
    lab = inp["labels"]
    lp = out.logits.float().log_softmax(-1)
    tok_ce = []
    for bi in range(lab.shape[0]):
        for br in range(2):
            for p in range(4):
                if lab[bi, br, p] != -100:
                    tok_ce.append(-lp[bi, br, p, lab[bi, br, p]])
    manual = torch.stack(tok_ce).mean()
    rep["f_loss_once"] = {"targets_per_example": len(tok_ce) / lab.shape[0],
                          "image_branch_f_is_ignored": bool((lab[:, 1, 0] == -100).all()),
                          "text_branch_f_target_equals_image_f_input": bool((lab[:, 0, 0] == inp["decoder_input_ids"][:, 1, 1]).all()),
                          "model_loss": float(out.loss), "manual_loss": float(manual),
                          "abs_diff": float((out.loss - manual).abs())}
    # 2) exact scoring: vectorized joint score vs manual forward of the two branch sequences
    ids, mask = inp["input_ids"], inp["attention_mask"]
    sb = score_batch(model, cat, ids, mask, args.k_f, T, I)
    diffs = []
    for u in range(4):
        cand = (sb["user"] == u).nonzero().squeeze(1).tolist()
        picks = rng.sample(cand, min(5, len(cand)))
        tgt = batch[u]["labels"]
        hit = [j for j in cand if int(sb["item"][j]) == tgt]
        for j in picks + hit:
            it = int(sb["item"][j])
            dec, _ = branch_tensors(cat.item_tok.cpu(), torch.tensor([it]), T, I)
            lo = model(input_ids=ids[u:u + 1], attention_mask=mask[u:u + 1], decoder_input_ids=dec.to(dev)).logits[0].float().log_softmax(-1)
            f, t1, t2, i1, i2 = cat.item_tok[it].tolist()
            man = lo[0, 0, f] + lo[0, 2, t1] + lo[0, 3, t2] + lo[1, 2, i1] + lo[1, 3, i2]
            vec = sb["s_f"][j] + sb["s_t"][j] + sb["s_i"][j]
            diffs.append(float((man - vec).abs()))
    rep["exact_scoring"] = {"pairs_checked": len(diffs), "max_abs_diff": max(diffs), "tolerance": 1e-4,
                            "pass": max(diffs) < 1e-4}
    rep["all_pass"] = (all(v["other_branch_identical"] and v["modified_branch_changed"]
                           for k, v in rep.items() if k.startswith("independence"))
                       and rep["f_loss_once"]["targets_per_example"] == 5
                       and rep["f_loss_once"]["image_branch_f_is_ignored"]
                       and rep["f_loss_once"]["abs_diff"] < 1e-4 and rep["exact_scoring"]["pass"])
    return rep


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--k_f", type=int, default=20)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--sanity_only", action="store_true")
    p.add_argument("--random_init", action="store_true", help="untrained MFDT5 (sanity checks before training)")
    p.add_argument("--chain_ckpt", default=None)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.random_init:
        tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
        tokenizer.add_tokens(MFDDataset(args, "valid").get_new_tokens())
        config = T5Config.from_pretrained(args.base_model)
        model = MFDT5(config); model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
        model = MFDT5.from_pretrained(args.ckpt_path)
    model = model.cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tokenizer, "cuda")
    T, I = tokenizer.convert_tokens_to_ids(list(BRANCH_TOKENS))
    if args.sanity or args.sanity_only:
        rep = sanity(args, model, tokenizer, cat, T, I)
        json.dump(rep, open(os.path.join(args.out_dir, "sanity.json"), "w"), indent=1)
        print("[sanity]", json.dumps(rep))
        if not rep["all_pass"]:
            raise SystemExit("SANITY CHECK FAILED; stopping")
        if args.sanity_only:
            return
    for split in args.splits.split(","):
        res, ranked, targets = run_split(args, model, tokenizer, cat, split, T, I)
        if args.chain_ckpt:
            res["chain_decoder_f"] = chain_f_accuracy(args, split, args.k_f)
        json.dump(res, open(os.path.join(args.out_dir, f"results_{split}.json"), "w"), indent=1)
        torch.save({m: v for m, v in ranked.items()} | {"targets": targets},
                   os.path.join(args.out_dir, f"ranked_{split}.pt"))
        print(f"[{split}]", json.dumps(res))


if __name__ == "__main__":
    main()
