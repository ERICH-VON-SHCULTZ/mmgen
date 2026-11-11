"""mmgen addition (docs/MFD_REPORT.md §21): sanity checks of the round-21 training methods (r21_views.py), GPU node.

 1 grouped layout: GroupedMVDataset has one example per pair; GroupedCollator puts the full / text / image view of pair
   k in rows 3k..3k+2 with the same decoder rows; view strings = mfd_data.view_maps (5 / 3 / 3 code tokens per item)
 2 G0 loss = P1a's loss (ItemMFDT5) on the same 3P rows (mean over views = P1a's scale); model.eval() loss = P1a's (validation loss / early stopping)
 3 A / B loss - G0 loss = beta x the KL term recomputed independently from the eval-mode logits (dropout off)
 4 stop-gradient: A's KL sends no gradient to the text / image views' f logits, B's sends gradient to all three
 5 f distribution = softmax over the 256 <s0_*> codes at the text branch's position 0 (the f loss's logits)
 6 C: item-level view rates ~ p_full / (1 - p_full) / 2 each over 20k item draws; redrawn per build; target = full SID;
   validation stays the full view (MFDDataset)
 9 AC (§21.4): GroupedRandDataset rows = [full view, item-random, item-random] with the right per-item view rates (p_full
   1/3 and 0), redrawn per build; mode A on an AC batch: loss - G0 loss = beta x KL(stopgrad(pbar) || p_row0)
 7b acc layout (G0acc / A01acc / AC*): 85 pairs per micro-batch x 4: optimizer steps per epoch vs P1a's (tolerance 2)
 8 clipping: HF's training loop clips through self.accelerator.clip_grad_norm_ (the call finetune_r21.py wraps to log every
   step's norm) and P1a's dry run has max_grad_norm 1.0
 7 matched compute: optimizer steps per epoch of grouped (341 pairs, accumulation 1) and C (256 x 4) vs P1a's train_stats
Model checks use P1a seed 42's trained checkpoint. C uses P1a's exact layout, so its steps equal P1a's.
Usage: python r21_check.py --data_path .. --dataset D --p1a_ckpt .. --out_file ..
"""
import argparse
import json
import math

import torch
import torch.nn as nn
from transformers import T5Tokenizer

import mfd_data
import r21_views
from enc_struct import ItemMFDT5
from utils import parse_dataset_args, parse_global_args

IDX = ".index_md122_a05_ep500_reassign.json"
PAIRS_PER_STEP = 341


def no_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--p1a_ckpt", required=True); q.add_argument("--out_file", required=True)
    q.add_argument("--data_path", required=True); q.add_argument("--dataset", required=True)
    a, _ = q.parse_known_args()
    p = parse_dataset_args(parse_global_args(argparse.ArgumentParser()))
    args = p.parse_args(["--data_path", a.data_path, "--dataset", a.dataset, "--index_file", IDX, "--image_index_file", IDX,
                         "--max_his_len", "20", "--tasks", "seqrec_mv", "--prompt_num", "4"])
    tok = T5Tokenizer.from_pretrained(a.p1a_ckpt)
    res, ok = {}, True

    def check(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str), flush=True)

    # 1 layout
    G = r21_views.GroupedMVDataset(args, "train")
    MV = mfd_data.MVMFDDataset(args, "train")
    coll = r21_views.GroupedCollator(args, tok)
    base = mfd_data.MFDCollator(args, tok)
    idx = list(range(0, len(G), max(1, len(G) // 8)))[:8]
    b = coll([G[i] for i in idx])
    n = len(G.inter_data)
    ref = base([MV[i + j * n] for i in idx for j in range(3)])                 # MVMFDDataset: view j of pair i = index i + j n
    lay = torch.equal(b["decoder_input_ids"], ref["decoder_input_ids"]) and torch.equal(b["labels"], ref["labels"]) \
        and torch.equal(b["input_ids"], ref["input_ids"])
    per_item = []
    for v, h in zip(G[idx[1]]["input_ids"], [5, 3, 3]):
        per_item.append(len(tok.tokenize(v)) / len(G.inter_data[idx[1]]["inters"]) == h)
    check("1_grouped_layout", lay and all(per_item) and len(G) == n and len(MV) == 3 * n,
          pairs=n, mv_examples=len(MV), tokens_per_item_ok=per_item)
    # 2-5 model
    dev = "cuda"
    model = r21_views.GroupedItemMFDT5.from_pretrained(a.p1a_ckpt).to(dev)
    p1a = ItemMFDT5.from_pretrained(a.p1a_ckpt).to(dev)
    for m in (model, p1a):
        no_dropout(m)
        m.train()
    bb = {k: v.to(dev) for k, v in b.items()}
    info = model.install_r21(tok, "G0", 0.0)
    fcols = model.r21_fcols
    names = tok.convert_ids_to_tokens(fcols.tolist())
    check("5_f_codes", len(fcols) == 256 and all(x.startswith("<s0_") for x in names), n=len(fcols), info=info)
    with torch.no_grad():
        l_g0 = model(**bb).loss.item()
        l_p1a = p1a(**bb).loss.item()
        model.eval(); p1a.eval()
        l_eval, l_eval_p1a = model(**bb).loss.item(), p1a(**bb).loss.item()
        logits = model(**bb).logits
        model.train()
    check("2_G0_is_P1a_loss", abs(l_g0 - l_p1a) < 1e-5 * max(1, l_g0), g0=l_g0, p1a=l_p1a)
    check("2_eval_loss_is_P1a", abs(l_eval - l_eval_p1a) < 1e-5, eval=l_eval, p1a=l_eval_p1a)
    lp = logits[:, 0, 0, fcols].float().log_softmax(-1).view(-1, 3, len(fcols))
    P = lp.exp()
    pbar = P.mean(1)
    kl_a = (pbar * (pbar.log() - lp[:, 0])).sum(-1).mean().item()
    def klr(x, y):
        return (x.exp() * (x - y)).sum(-1)
    kl_b = torch.stack([(klr(lp[:, u], lp[:, v]) + klr(lp[:, v], lp[:, u])) / 2 for u, v in ((0, 1), (0, 2), (1, 2))], 1).mean().item()
    for mode, beta, ref_kl in (("A", 0.5, kl_a), ("B", 0.5, kl_b)):
        model.install_r21(tok, mode, beta)
        with torch.no_grad():
            l = model(**bb).loss.item()
        check(f"3_{mode}_loss_is_G0_plus_beta_KL", abs(l - l_g0 - beta * ref_kl) < 1e-4 * max(1, l), loss=l, g0=l_g0, kl=ref_kl, beta=beta)
    for mode in ("A", "B"):
        x = lp.detach().clone().requires_grad_(True)
        r21_views.aux_loss(x.log_softmax(-1), mode).backward()
        gn = x.grad.abs().sum((0, 2))
        cond = (gn[0] > 0 and gn[1] == 0 and gn[2] == 0) if mode == "A" else bool((gn > 0).all())
        check(f"4_{mode}_gradient_to_views", cond, grad_abs_sum_full_text_image=gn.tolist())
    # 6 C
    stats = {}
    for pf in (1 / 3, 0.6):
        C = r21_views.ItemRandViewDataset(args, "train", pf)
        draws = C.draw(20000)
        share = {v: draws.count(v) / len(draws) for v in mfd_data.VIEWS}
        j = max(range(n), key=lambda i: len(C.inter_data[i]["inters"]))           # longest history
        builds = {C[j]["input_ids"] for _ in range(20)}
        tgt_ok = C[j]["labels"] == C.sid2item[C.inter_data[j]["item"]] and C[j + n]["labels"] == C[j]["labels"]
        hs = C.inter_data[j]["inters"]
        s = C[j]["input_ids"]
        toks = tok.tokenize(s)
        items_ok = sum(1 for t in toks if t.startswith("<s0_")) == len(hs)
        stats[f"{pf:.3f}"] = dict(share=share, distinct_builds_of_20=len(builds), target_ok=tgt_ok, items_ok=items_ok, len=len(C))
        cond = abs(share["full"] - pf) < 0.02 and abs(share["text"] - share["image"]) < 0.03 and len(builds) > 1 and tgt_ok \
            and items_ok and len(C) == 3 * n
        check(f"6_C_item_views_p{pf:.2f}", cond, **stats[f"{pf:.3f}"])
    V = mfd_data.MFDDataset(args, "valid")
    vs = V[0]["input_ids"]
    check("6_valid_is_full_view", len(tok.tokenize(vs)) == 5 * len(V.inter_data[0]["inters"]))
    # 7 compute
    st = json.load(open(f"{a.p1a_ckpt}/train_stats.json"))
    p1a_spe = st["global_steps"] / st["epochs_run"]
    g_spe = math.ceil(n / PAIRS_PER_STEP)
    check("7_steps_per_epoch", abs(g_spe - p1a_spe) <= 1, p1a=p1a_spe, grouped=g_spe, grouped_minus_p1a=g_spe - p1a_spe,
          C="= P1a by construction (same 3n examples, 256 x 4; checked in 6: len == 3n)",
          grouped_sequences_per_step=3 * PAIRS_PER_STEP, C_sequences_per_step=1024)
    acc_micro = math.ceil(n / 85)
    check("7b_steps_per_epoch_acc", min(abs(math.ceil(acc_micro / 4) - p1a_spe), abs(acc_micro // 4 - p1a_spe)) <= 2, p1a=p1a_spe,
          micro_batches=acc_micro, steps_floor=acc_micro // 4, steps_ceil=math.ceil(acc_micro / 4), rows_per_step=4 * 85 * 3)
    # 9 AC
    for pf in (1 / 3, 0.0):
        A = r21_views.GroupedRandDataset(args, "train", pf)
        j = max(range(n), key=lambda i: len(A.inter_data[i]["inters"]))
        h = A.inter_data[j]["inters"]
        ex = A[j]
        full_ok = ex["input_ids"][0] == "".join(A.vmap["full"][x] for x in h)
        cnt = {"full": 0, "text": 0, "image": 0}
        builds = set()
        for _ in range(300):
            e = A[j]
            builds.add(tuple(e["input_ids"][1:]))
            for row in e["input_ids"][1:]:
                toks = tok.tokenize(row)
                k = 0
                for x in h:                                         # walk the row item by item (5 or 3 tokens)
                    full5 = tok.tokenize(A.vmap["full"][x])
                    if toks[k:k + 5] == full5:
                        cnt["full"] += 1; k += 5
                    elif toks[k:k + 3] == tok.tokenize(A.vmap["text"][x]):
                        cnt["text"] += 1; k += 3
                    else:
                        assert toks[k:k + 3] == tok.tokenize(A.vmap["image"][x]), (row, x)
                        cnt["image"] += 1; k += 3
                assert k == len(toks)
        tot = sum(cnt.values())
        share = {k: v / tot for k, v in cnt.items()}
        cond = full_ok and abs(share["full"] - pf) < 0.03 and abs(share["text"] - share["image"]) < 0.05 and len(builds) > 100 \
            and ex["labels"] == A.sid2item[A.inter_data[j]["item"]] and len(A) == n
        check(f"9_AC_rows_p{pf:.2f}", cond, row0_full=full_ok, share=share, distinct_builds_of_300=len(builds), items=len(h))
    acb = {k: v.to(dev) for k, v in coll([r21_views.GroupedRandDataset(args, "train", 1 / 3)[i] for i in idx]).items()}
    model.install_r21(tok, "G0", 0.0)
    with torch.no_grad():
        g0 = model(**acb).loss.item()
        model.eval(); lg = model(**acb).logits; model.train()
        model.install_r21(tok, "A", 0.1)
        la = model(**acb).loss.item()
    lpa = lg[:, 0, 0, fcols].float().log_softmax(-1).view(-1, 3, len(fcols))
    pb = lpa.exp().mean(1)
    kla = (pb * (pb.log() - lpa[:, 0])).sum(-1).mean().item()
    check("9_AC_loss_is_G0_plus_beta_KL", abs(la - g0 - 0.1 * kla) < 1e-4 * max(1, la), loss=la, g0=g0, kl=kla, beta=0.1)
    # 8 clipping hook
    import inspect
    import transformers
    src = inspect.getsource(transformers.Trainer._inner_training_loop)
    ta = json.load(open(f"{a.p1a_ckpt}/../../../../geo/gate/r21_dryrun/{a.dataset}/P1aref/dryrun.json"))["training_args"]
    check("8_clip_hook", "self.accelerator.clip_grad_norm_(" in src and ta["max_grad_norm"] == 1.0,
          transformers=transformers.__version__, p1a_max_grad_norm=ta["max_grad_norm"])
    res["all_pass"] = ok
    json.dump(res, open(a.out_file, "w"), indent=1, default=str)
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
