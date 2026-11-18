"""mmgen addition (docs/MFD_REPORT.md §22.7): gate checks of E7 (r22_e7.E7GroupedItemMFDT5) on A01 seed 42's checkpoint, GPU.
For each mode (a: W [E(f); E(t1)+E(t2); E(i1)+E(i2)], b: W_f E(f)), on 8 grouped training pairs (24 rows), dropout off:
 1 step 0 = A01: with W = 0 the logits (eval) and the training loss (A01's grouped loss, beta 0.1) equal A01's (GroupedItemMFDT5)
 2 W receives gradient: after backward of the training loss, |grad W| > 0
 3 view parts: full-view rows have non-zero f / text / image parts for every item; text-view rows have a zero image part and
   image-view rows a zero text part (absent part = zero vector, no view marker); f + text + image = A01's item sum
 4 save / reload with W != 0: mfd_eval.load_mfd_model rebuilds E7GroupedItemMFDT5 with the same mode and identical logits
Usage: python r22_e7_check.py --data_path .. --dataset Arts --a01_ckpt .. --out_file ..
"""
import argparse
import json
import os
import tempfile

import torch
from transformers import T5Tokenizer

import r21_views
from mfd_eval import load_mfd_model
from r22_check import dargs, no_dropout
from r22_e7 import E7GroupedItemMFDT5

I122 = ".index_md122_a05_ep500_reassign.json"


def main():
    q = argparse.ArgumentParser()
    for k in ("data_path", "dataset", "a01_ckpt", "out_file"):
        q.add_argument("--" + k, required=True)
    a = q.parse_args()
    res, ok = {}, True

    def check(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str), flush=True)

    args = dargs(a, I122)
    tok = T5Tokenizer.from_pretrained(a.a01_ckpt)
    G = r21_views.GroupedMVDataset(args, "train")
    coll = r21_views.GroupedCollator(args, tok)
    idx = list(range(0, len(G), max(1, len(G) // 8)))[:8]
    b = {k: v.cuda() for k, v in coll([G[i] for i in idx]).items()}
    ref = r21_views.GroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
    ref.install_r21(tok, "A", 0.1)
    no_dropout(ref)
    ref.eval()
    with torch.no_grad():
        lg_ref = ref(**b).logits
        ref.train()
        loss_ref = ref(**b).loss.item()
        enc_ref, _ = ref.struct_encode(b["input_ids"], b["attention_mask"])
    for mode in ("a", "b"):
        m = E7GroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
        m.install_r21(tok, "A", 0.1)
        info = m.install_e7(mode)
        no_dropout(m)
        m.eval()
        with torch.no_grad():
            d_lg = (m(**b).logits - lg_ref).abs().max().item()
            enc, _ = m.struct_encode(b["input_ids"], b["attention_mask"])
            d_enc = (enc.last_hidden_state - enc_ref.last_hidden_state).abs().max().item()
        m.train()
        loss = m(**b).loss
        check(f"{mode}_1_step0_equals_A01", d_lg < 1e-4 and d_enc < 1e-4 and abs(loss.item() - loss_ref) < 1e-5 * max(1, loss_ref),
              max_logit_diff=d_lg, max_encoder_diff=d_enc, loss=loss.item(), a01_loss=loss_ref, info=info)
        loss.backward()
        g = m.e7.weight.grad
        check(f"{mode}_2_W_gets_gradient", g is not None and g.abs().sum().item() > 0,
              grad_abs_sum=None if g is None else g.abs().sum().item(), w_shape=list(m.e7.weight.shape))
        m.eval()
        with torch.no_grad():
            p, mask, n, _ = m.item_parts(b["input_ids"])
            nz = lambda x: x.abs().sum(-1) > 0                         # (rows, items)
            real = mask.bool()
            full, text, image = slice(0, None, 3), slice(1, None, 3), slice(2, None, 3)
            c_full = all(bool((nz(p[k][full]) == real[full]).all()) for k in ("f", "t", "i"))
            c_text = bool((nz(p["f"][text]) == real[text]).all() and (nz(p["t"][text]) == real[text]).all() and not nz(p["i"][text]).any())
            c_img = bool((nz(p["f"][image]) == real[image]).all() and (nz(p["i"][image]) == real[image]).all() and not nz(p["t"][image]).any())
            emb = m.shared(b["input_ids"])
            s, code, item, _, _ = m.struct.structure(b["input_ids"])
            x = torch.zeros_like(p["f"]).scatter_add(1, item.clamp(min=0)[..., None].expand_as(emb), emb * code[..., None].to(emb.dtype))
            d_sum = (p["f"] + p["t"] + p["i"] - x).abs().max().item()
        check(f"{mode}_3_view_parts", c_full and c_text and c_img and d_sum < 1e-5, full=c_full, text_view=c_text, image_view=c_img,
              parts_minus_a01_sum=d_sum)
        with torch.no_grad():
            m.e7.weight.normal_(0, 0.02)
            lg = m(**b).logits
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
            m.save_pretrained(t)
            tok.save_pretrained(t)
            re = load_mfd_model(t).cuda().eval()
            no_dropout(re)
            with torch.no_grad():
                d = (re(**b).logits - lg).abs().max().item()
            check(f"{mode}_4_save_reload", type(re).__name__ == "E7GroupedItemMFDT5" and re.r22_e7 == mode and d < 1e-5,
                  cls=type(re).__name__, mode=re.r22_e7, max_logit_diff=d)
    res["all_pass"] = ok
    json.dump(res, open(a.out_file, "w"), indent=1, default=str)
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
