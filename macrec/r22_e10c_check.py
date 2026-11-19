"""mmgen addition (docs/MFD_REPORT.md §22.10, E10c): gate checks of A01's KL without grouping (r22_e10c.py), GPU, P1a seed 42's checkpoint.
Batch: 24 rows of P1a's MVMFDDataset (24 pairs, views cycling full / text / image -> 8 full rows), as P1a's sampler would draw them.
 0 data layout: E10cDataset has P1a's length; every row's main history string = P1a's row; the collated input_ids / attention_mask /
   decoder_input_ids / labels = P1a's MFDCollator batch; e10_rows = the full rows; their extra text / image strings = P1a's own
   text-view / image-view rows of the same pair (index + n, + 2n); a validation batch has no e10_* keys
 1 beta = 0 equals P1a: train mode WITH dropout, same RNG seed: loss and every parameter gradient equal P1a's (ItemMFDT5)
 2 no gradient through the extra forwards: p_text / p_image carry no autograd graph; the gradient of the E10c loss (beta 0.1, dropout
   off) = the gradient of P1a's CE + 3 beta / R * sum KL(pbar_const || p_full) built with pbar_const from DETACHED tensors of an
   independent model copy
 3 KL term = independent recomputation: p_full / p_text / p_image from the plain P1a model on P1a's own full / text / image rows of the
   same pairs (full decoder input), float64 KL, 3 beta / R weighting; loss - CE = term
 4 training class path: Geo(E10cItemMFDT5) with the geometry installed as in training: the e10_* inputs reach the KL through
   Geo.forward (term > 0, loss = CE + term); bake / save / reload through mfd_eval.load_mfd_model -> ItemMFDT5, identical logits
 5 cost (info): 20 train steps (forward + backward) of a 256-row micro-batch, P1a vs E10c (beta 0.1), same GPU, dropout on
Usage: python r22_e10c_check.py --data_path .. --dataset Arts --p1a_ckpt .. --tok_ckpt <v1 tokenizer.pth> --out_file ..
"""
import argparse
import json
import math
import os
import tempfile
import time

import torch
from transformers import T5Tokenizer

import mfd_data
from enc_struct import ItemMFDT5
from geo_embed import make_geo_class
from mfd_eval import load_mfd_model
from r22_check import dargs, no_dropout
from r22_e10c import E10cCollator, E10cDataset, E10cItemMFDT5

I122 = ".index_md122_a05_ep500_reassign.json"
BETA = 0.1


def main():
    q = argparse.ArgumentParser()
    for k in ("data_path", "dataset", "p1a_ckpt", "tok_ckpt", "out_file"):
        q.add_argument("--" + k, required=True)
    a = q.parse_args()
    res, ok = {}, True

    def check(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str), flush=True)

    args = dargs(a, I122)
    tok = T5Tokenizer.from_pretrained(a.p1a_ckpt)
    P = mfd_data.MVMFDDataset(args, "train")
    E = E10cDataset(args, "train")
    n = len(P.inter_data)
    pairs = list(range(0, n, n // 24))[:24]
    rows = [p + (j % 3) * n for j, p in enumerate(pairs)]
    pc, ec = mfd_data.MFDCollator(args, tok), E10cCollator(args, tok)
    bp = {k: v.cuda() for k, v in pc([P[i] for i in rows]).items()}
    be = {k: v.cuda() for k, v in ec([E[i] for i in rows]).items()}
    full_pos = [j for j, i in enumerate(rows) if i < n]
    fr = [rows[j] for j in full_pos]

    # 0 data layout
    same_str = all((E[i]["input_ids"][0] if isinstance(E[i]["input_ids"], list) else E[i]["input_ids"]) == P[i]["input_ids"]
                   and E[i]["labels"] == P[i]["labels"] for i in rows)
    same_t = all(torch.equal(bp[k], be[k]) for k in bp)
    extra_ok = all(E[i]["input_ids"][1] == P[i + n]["input_ids"] and E[i]["input_ids"][2] == P[i + 2 * n]["input_ids"] for i in fr)
    vb = ec([mfd_data.MFDDataset(args, "valid")[j] for j in range(4)])
    check("0_data_layout_is_P1a", len(E) == len(P) and same_str and same_t and extra_ok and be["e10_rows"].tolist() == full_pos
          and not any(k.startswith("e10") for k in vb), len=len(E), rows=len(rows), full_rows=len(full_pos),
          same_strings=same_str, same_tensors=same_t, extra_views_are_P1a_rows=extra_ok, valid_keys=sorted(vb))

    # 1 beta = 0 equals P1a (dropout on, same RNG)
    ref = ItemMFDT5.from_pretrained(a.p1a_ckpt).cuda().train()
    m0 = E10cItemMFDT5.from_pretrained(a.p1a_ckpt).cuda().train()
    m0.install_e10c(tok, 0.0)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    lr_ = ref(**bp).loss; lr_.backward()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    l0 = m0(**be).loss; l0.backward()
    gr = dict(ref.named_parameters())
    gd = max(float((p.grad - gr[k].grad).abs().max()) for k, p in m0.named_parameters() if p.grad is not None)
    check("1_beta0_equals_P1a", abs(l0.item() - lr_.item()) < 1e-6 and gd < 1e-6 and m0.e10_last is not None,
          loss=l0.item(), p1a_loss=lr_.item(), max_grad_diff=gd, kl_computed=m0.e10_last is not None)

    # 2 + 3 at beta 0.1, dropout off
    ref = ItemMFDT5.from_pretrained(a.p1a_ckpt).cuda().train(); no_dropout(ref)
    m = E10cItemMFDT5.from_pretrained(a.p1a_ckpt).cuda().train(); no_dropout(m)
    m.install_e10c(tok, BETA)
    loss = m(**be).loss
    L = m.e10_last
    loss.backward()
    fcols = m.e10_fcols
    R = len(rows)
    ce = ref(**bp)
    lpf_ref = ce.logits[full_pos, 0, 0][:, fcols].float().log_softmax(-1)
    with torch.no_grad():
        side = {}
        for v, off in (("text", n), ("image", 2 * n)):
            b = {k: x.cuda() for k, x in pc([P[i + off] for i in fr]).items()}
            side[v] = ref(**b).logits[:, 0, 0][:, fcols].double().log_softmax(-1)
        pf = lpf_ref.detach().double().exp()
        pbar = (pf + side["text"].exp() + side["image"].exp()) / 3
        kl64 = (pbar * (pbar.log() - lpf_ref.detach().double())).sum(-1)
        term64 = 3 * BETA * kl64.sum().item() / R
    check("3_kl_matches_independent", abs(L["term"].item() - term64) < 1e-5 * max(1, term64)
          and abs((loss - ce.loss).item() - term64) < 1e-5 * max(1, term64) and abs(L["R"] - R) == 0,
          term=L["term"].item(), term_independent=term64, loss_minus_ce=(loss - ce.loss).item(), kl_rows=L["kl_rows"].tolist(),
          kl_rows_independent=kl64.tolist(), R=R)
    tgt = ce.loss + 3 * BETA / R * (pbar.float() * (pbar.float().log() - lpf_ref)).sum()      # pbar constant: no graph
    tgt.backward()
    gr = dict(ref.named_parameters())
    gdiff = max(float((p.grad - gr[k].grad).abs().max()) for k, p in m.named_parameters() if p.grad is not None)
    gscale = max(float(p.grad.abs().max()) for p in m.parameters() if p.grad is not None)
    check("2_no_gradient_through_extra_forwards", not L["lpt"].requires_grad and not L["lpi"].requires_grad
          and L["lpt"].grad_fn is None and L["lpi"].grad_fn is None and gdiff < 1e-5 * max(1, gscale),
          p_text_requires_grad=L["lpt"].requires_grad, p_image_requires_grad=L["lpi"].requires_grad, max_grad_diff=gdiff, grad_scale=gscale)

    # 4 training class path + bake / save / reload
    Geo = make_geo_class(E10cItemMFDT5)
    g = Geo.from_pretrained(a.p1a_ckpt).cuda()
    g.install_geometry(tok, a.tok_ckpt, "codebook")
    g.cuda()                                           # training installs the geometry before the Trainer moves the model
    g.install_e10c(tok, BETA)
    no_dropout(g)
    g.train()
    with torch.no_grad():
        lg = g(**be).loss.item()
        tg = g.e10_last["term"].item()
        lce = g(**bp).loss.item()                      # P1a batch (no e10_* inputs): CE only
        g.eval()
        before = g(**bp).logits
        g.bake()
        g.__class__ = type(g).__mro__[2]
        d_bake = (g(**bp).logits - before).abs().max().item()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"), ignore_cleanup_errors=True) as t:
        g.save_pretrained(t)
        tok.save_pretrained(t)
        re = load_mfd_model(t).cuda().eval()
        with torch.no_grad():
            d_re = (re(**bp).logits - before).abs().max().item()
        cls = type(re).__name__
    check("4_geo_path_and_bake_reload", tg > 0 and abs(lg - (lce + tg)) < 1e-5 and d_bake < 1e-4 and d_re < 1e-4 and cls == "ItemMFDT5",
          geo_loss=lg, geo_ce=lce, geo_term=tg, bake_max_logit_diff=d_bake, reload_max_logit_diff=d_re, reloaded_class=cls)

    # 5 cost (info only)
    big = list(torch.randperm(len(E), generator=torch.Generator().manual_seed(0))[:256].tolist())
    bp2 = {k: v.cuda() for k, v in pc([P[i] for i in big]).items()}
    be2 = {k: v.cuda() for k, v in ec([E[i] for i in big]).items()}
    t = {}
    for name, mod, b in (("P1a", ItemMFDT5.from_pretrained(a.p1a_ckpt).cuda().train(), bp2), ("E10c", m, be2)):
        mod.train()
        for it in range(25):
            if it == 5:
                torch.cuda.synchronize(); s = time.time()
            mod.zero_grad(set_to_none=True)
            mod(**b).loss.backward()
        torch.cuda.synchronize()
        t[name] = (time.time() - s) / 20
    mb_epoch = math.ceil(len(E) / 256)
    check("5_cost_info", True, ms_per_microbatch=t, ratio=t["E10c"] / t["P1a"], full_rows_in_batch=int(be2["e10_rows"].numel()),
          micro_batches_per_epoch=mb_epoch, extra_seconds_per_epoch_compute_only=(t["E10c"] - t["P1a"]) * mb_epoch)
    res["all_pass"] = ok
    json.dump(res, open(a.out_file, "w"), indent=1, default=str)
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
