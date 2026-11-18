"""mmgen addition (docs/MFD_REPORT.md §22.8 / §22.9): gate checks of E8 (r22_e8) and E9 (r22_e9) on A01 seed 42's checkpoint, GPU.
On 8 grouped training pairs (24 rows), dropout off, A01 reference = r21_views.GroupedItemMFDT5 with A's KL (beta 0.1).
E8 (modes a, b):
 1 step 0 = A01: with W = 0 the logits (eval) and the training loss equal A01's
 2 W receives gradient
 3 c(f): finite (no NaN / inf); bucket sizes (catalog items per f, f tokens without items) reported
 4 bake / save / reload: geometry installed for real (finetune_geo path), W != 0; geo_embed bake() folds E'(f) into the plain table
   with identical logits; the saved model reloads through mfd_eval.load_mfd_model with identical logits
E9 (K = 2, 4):
 1 initial loss close to A01's (W_k = I + N(0, 0.01^2), pi uniform): |loss - A01 loss| / A01 loss < 0.02; max |log P(f) - A01| reported
 2 pi (W_pi) and every W_k receive gradient
 3 inference = training: full-vocabulary log P at the f position from the training forward (full-view rows) = MFDT5.last_logprobs
   at [start] (inference path); and a later position read through last_logprobs (mixture skipped there) = the training forward
 4 bake / save / reload as E8 (W_pi != 0): identical logits, class E9GroupedItemMFDT5 with the same K
Usage: python r22_e89_check.py --data_path .. --dataset Arts --a01_ckpt .. --tok_ckpt <v1 tokenizer.pth> --out_file ..
"""
import argparse
import json
import os
import tempfile

import torch
from transformers import T5Tokenizer

import r21_views
from geo_embed import make_geo_class
from mfd_eval import load_mfd_model
from r22_check import dargs, no_dropout
from r22_e8 import E8GroupedItemMFDT5
from r22_e9 import E9GroupedItemMFDT5

I122 = ".index_md122_a05_ep500_reassign.json"


def main():
    q = argparse.ArgumentParser()
    for k in ("data_path", "dataset", "a01_ckpt", "tok_ckpt", "out_file"):
        q.add_argument("--" + k, required=True)
    q.add_argument("--which", default="E8,E9")
    a = q.parse_args()
    res, ok = {}, True

    def check(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str), flush=True)

    args = dargs(a, I122)
    index_path = os.path.join(a.data_path, a.dataset, a.dataset + I122)
    tok = T5Tokenizer.from_pretrained(a.a01_ckpt)
    G = r21_views.GroupedMVDataset(args, "train")
    coll = r21_views.GroupedCollator(args, tok)
    idx = list(range(0, len(G), max(1, len(G) // 8)))[:8]
    b = {k: v.cuda() for k, v in coll([G[i] for i in idx]).items()}
    ref = r21_views.GroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
    ref.install_r21(tok, "A", 0.1)
    no_dropout(ref)
    fcols = ref.r21_fcols
    ref.eval()
    with torch.no_grad():
        lg_ref = ref(**b).logits
        ref.train()
        loss_ref = ref(**b).loss.item()

    def bake_reload(cls, install, perturb, name, extra_ok):
        Geo = make_geo_class(cls)
        g = Geo.from_pretrained(a.a01_ckpt).cuda()
        g.install_r21(tok, "A", 0.1)
        g.install_geometry(tok, a.tok_ckpt, "codebook")
        install(g)
        no_dropout(g)
        with torch.no_grad():
            perturb(g)
        g.eval()
        with torch.no_grad():
            before = g(**b).logits
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:   # mid-training checkpoint (Trainer: safetensors)
            g.save_pretrained(t, safe_serialization=True)
            from safetensors.torch import load_file
            sd = load_file(os.path.join(t, "model.safetensors"))
            new_keys = sorted(k for k in sd if k.endswith(".W.weight") or k.startswith("e9_"))
            with torch.no_grad():
                for p in g.parameters():
                    p.add_(0.01)
                g.load_state_dict(sd, strict=False)                              # as Trainer._load_best_model
                d_ck = (g(**b).logits - before).abs().max().item()
        check(name.replace("_4_bake", "_4a_checkpoint"), d_ck < 1e-4 and len(new_keys) > 0, checkpoint_reload_max_logit_diff=d_ck,
              saved_new_param_keys=new_keys)
        with torch.no_grad():
            g.bake()
            g.__class__ = type(g).__mro__[2]
            after = g(**b).logits
        d_bake = (before - after).abs().max().item()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
            g.save_pretrained(t)
            tok.save_pretrained(t)
            re = load_mfd_model(t).cuda().eval()
            no_dropout(re)
            with torch.no_grad():
                d_re = (re(**b).logits - before).abs().max().item()
        check(name, d_bake < 1e-4 and d_re < 1e-4 and extra_ok(re), bake_max_logit_diff=d_bake, reload_max_logit_diff=d_re,
              baked_class=type(g).__name__, reloaded_class=type(re).__name__)

    if "E8" in a.which:
        for mode in ("a", "b"):
            m = E8GroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
            m.install_r21(tok, "A", 0.1)
            info = m.install_e8(tok, index_path, mode)
            no_dropout(m)
            m.eval()
            with torch.no_grad():
                d_lg = (m(**b).logits - lg_ref).abs().max().item()
            m.train()
            loss = m(**b).loss
            check(f"E8{mode}_1_step0_equals_A01", d_lg < 1e-4 and abs(loss.item() - loss_ref) < 1e-5 * max(1, loss_ref),
                  max_logit_diff=d_lg, loss=loss.item(), a01_loss=loss_ref)
            loss.backward()
            gW = m.shared.W.weight.grad
            check(f"E8{mode}_2_W_gets_gradient", gW is not None and gW.abs().sum().item() > 0,
                  grad_abs_sum=None if gW is None else gW.abs().sum().item())
            check(f"E8{mode}_3_children_finite", info["c_finite"], **{k: info[k] for k in
                  ("f_tokens", "f_with_items", "f_without_items", "items", "items_per_f", "child_tokens", "children_per_item", "c_norm_mean")})
            bake_reload(E8GroupedItemMFDT5, lambda g, mode=mode: g.install_e8(tok, index_path, mode),
                        lambda g: g.shared.W.weight.normal_(0, 0.05), f"E8{mode}_4_bake_save_reload",
                        lambda re: type(re).__name__ == "ItemMFDT5")
    if "E9" in a.which:
        for K in (2, 4):
            torch.manual_seed(0)
            m = E9GroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
            m.install_r21(tok, "A", 0.1)
            m.install_e9(K)
            no_dropout(m)
            m.eval()
            with torch.no_grad():
                lg = m(**b).logits
                d_f = (lg[:, 0, 0].float().log_softmax(-1)[:, fcols] - lg_ref[:, 0, 0].float().log_softmax(-1)[:, fcols]).abs().max().item()
                d_other = (lg[:, :, 1:] - lg_ref[:, :, 1:]).abs().max().item()
            m.train()
            loss = m(**b).loss
            rel = abs(loss.item() - loss_ref) / loss_ref
            check(f"E9k{K}_1_init_loss_close_to_A01", rel < 0.02 and d_other < 1e-4, loss=loss.item(), a01_loss=loss_ref,
                  loss_diff=loss.item() - loss_ref, rel_diff=rel, max_abs_logPf_diff=d_f, other_positions_max_logit_diff=d_other)
            loss.backward()
            gpi = m.e9_pi.weight.grad
            gw = [float(m.e9_W.grad[k].abs().sum()) for k in range(K)] if m.e9_W.grad is not None else None
            check(f"E9k{K}_2_gradients", gpi is not None and gpi.abs().sum().item() > 0 and gw is not None and min(gw) > 0,
                  pi_grad_abs_sum=None if gpi is None else gpi.abs().sum().item(), W_k_grad_abs_sum=gw)
            m.eval()
            with torch.no_grad():
                with torch.no_grad():
                    m.e9_pi.weight.normal_(0, 0.5)                          # non-uniform pi for the comparison
                lg = m(**b).logits
                full = slice(0, None, 3)
                ids, mk = b["input_ids"][full], b["attention_mask"][full]
                enc, em = m.struct_encode(ids, mk)
                B = ids.shape[0]
                ar = torch.arange(B, device=ids.device)
                start = torch.full((B, 1), m.config.decoder_start_token_id, device=ids.device)
                lp_inf = m.last_logprobs(enc.last_hidden_state, em, ar, start, cols=[fcols])[0]
                lp_tr = lg[full][:, 0, 0].float().log_softmax(-1)[:, fcols]
                d_inf = (lp_inf - lp_tr).abs().max().item()
                dec3 = b["decoder_input_ids"][full][:, 0, :3]                  # [start, f, <T>] -> t1 position
                lp3 = m.last_logprobs(enc.last_hidden_state, em, ar, dec3)
                d3 = (lp3 - lg[full][:, 0, 2].float().log_softmax(-1)).abs().max().item()
            check(f"E9k{K}_3_inference_equals_training", d_inf < 1e-4 and d3 < 1e-4, f_position_max_diff=d_inf, later_position_max_diff=d3)
            bake_reload(E9GroupedItemMFDT5, lambda g, K=K: g.install_e9(K),
                        lambda g: g.e9_pi.weight.normal_(0, 0.5), f"E9k{K}_4_bake_save_reload",
                        lambda re, K=K: type(re).__name__ == "E9GroupedItemMFDT5" and re.r22_e9 == K)
    res["all_pass"] = ok
    json.dump(res, open(a.out_file, "w"), indent=1, default=str)
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
