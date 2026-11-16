"""mmgen addition (docs/MFD_REPORT.md §22): sanity checks of the round-22 training changes, GPU node.

E2 (aux heads, r22_aux.AuxGroupedItemMFDT5, on A01 seed 42's trained checkpoint + freshly installed heads):
 1 aux targets = the target item's t1 / i1 tokens (from the decoder label rows) mapped into the aux code lists
 2 identity init: aux logits at the start position = the main (tied) head's logits there, restricted to the level's codes
 3 training loss = A01 loss + gamma x [CE(t1) + CE(i1)], recomputed independently (dropout off)
 4 eval-mode loss = the plain MFD loss (early stopping unchanged)
 5 save_pretrained -> mfd_eval.load_mfd_model rebuilds AuxGroupedItemMFDT5 with identical aux log-probs
E1 (locked 1+3+3 SIDs):
 6 grouped view strings carry 7 / 4 / 4 code tokens per item; the item-level encoder counts every one of them as a code
   (t2 / v2 included) and one position per item; on the 1+2+2 tokenizer the encoder structure is unchanged
 7 decoder rows have length L + 2 = 5 (two 3-step branches); codebook-init levels from the 1+3+3 tokenizer: s0 t0 t1 t2 v0 v1 v2
Usage: python r22_check.py --data_path .. --dataset Arts --a01_ckpt .. --v1_dir .. --out_file ..
"""
import argparse
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer

import enc_struct
import mfd_data
import r21_views
from geo_embed import code_tokens, load_codebooks
from mfd_eval import load_mfd_model
from r22_aux import AuxGroupedItemMFDT5
from utils import parse_dataset_args, parse_global_args

I122 = ".index_md122_a05_ep500_reassign.json"
I133 = ".index_md133_a06_ep500_reassign.json"


def no_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0


def dargs(a, idx):
    p = parse_dataset_args(parse_global_args(argparse.ArgumentParser()))
    return p.parse_args(["--data_path", a.data_path, "--dataset", a.dataset, "--index_file", idx, "--image_index_file", idx,
                         "--max_his_len", "20", "--tasks", "seqrec_mv", "--prompt_num", "4"])


def main():
    q = argparse.ArgumentParser()
    for k in ("data_path", "dataset", "a01_ckpt", "v1_dir", "out_file", "base_model"):
        q.add_argument("--" + k, required=k != "base_model", default="./config/ckpt" if k == "base_model" else None)
    a = q.parse_args()
    res, ok = {}, True

    def check(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str), flush=True)

    # ---------------- E2
    args = dargs(a, I122)
    tok = T5Tokenizer.from_pretrained(a.a01_ckpt)
    G = r21_views.GroupedMVDataset(args, "train")
    coll = r21_views.GroupedCollator(args, tok)
    idx = list(range(0, len(G), max(1, len(G) // 8)))[:8]
    b = {k: v.cuda() for k, v in coll([G[i] for i in idx]).items()}
    model = AuxGroupedItemMFDT5.from_pretrained(a.a01_ckpt).cuda()
    info = model.install_r21(tok, "A", 0.1)
    info.update(model.install_aux(tok, 0.5))
    no_dropout(model)
    yt, yi = model.aux_targets(b["labels"])
    check("1_aux_targets", bool((model.aux_t_tok[yt] == b["labels"][:, 0, 2]).all()) and bool((model.aux_i_tok[yi] == b["labels"][:, 1, 2]).all()),
          n_t1=len(model.aux_t_tok), n_i1=len(model.aux_i_tok), info=info)
    model.eval()
    with torch.no_grad():
        enc, m = model.struct_encode(b["input_ids"], b["attention_mask"])
        lt, li = model.aux_logits(model.start_state(enc.last_hidden_state, m))
        main = model(**b).logits[:, 0, 0]                               # text row, position 0 (the f position)
    d = max((lt - main[:, model.aux_t_tok]).abs().max().item(), (li - main[:, model.aux_i_tok]).abs().max().item())
    check("2_identity_init_equals_main_head", d < 1e-3, max_abs_diff=d)
    with torch.no_grad():
        l_eval = model(**b).loss.item()
        p1 = enc_struct.ItemMFDT5.from_pretrained(a.a01_ckpt).cuda().eval()
        l_plain = p1(**b).loss.item()
        model.train()
        l_full = model(**b).loss.item()
        g = model.r22_gamma; model.r22_gamma = 0.0
        l_a01 = model(**b).loss.item()
        model.r22_gamma = g
        ref_aux = (F.cross_entropy(lt.float(), yt) + F.cross_entropy(li.float(), yi)).item()
    check("3_loss_is_A01_plus_gamma_aux", abs(l_full - l_a01 - g * ref_aux) < 1e-4 * max(1, l_full), loss=l_full, a01=l_a01, aux=ref_aux, gamma=g)
    check("4_eval_loss_is_plain", abs(l_eval - l_plain) < 1e-5, eval=l_eval, plain=l_plain)
    model.eval()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        model.save_pretrained(t)
        tok.save_pretrained(t)
        re = load_mfd_model(t).cuda().eval()
        with torch.no_grad():
            x = model.aux_logprobs(enc.last_hidden_state, m)
            y = re.aux_logprobs(enc.last_hidden_state, m)
        dd = max((x[0] - y[0]).abs().max().item(), (x[1] - y[1]).abs().max().item())
        check("5_save_reload_aux", type(re).__name__ == "AuxGroupedItemMFDT5" and dd < 1e-6, cls=type(re).__name__, max_abs_diff=dd)
    # ---------------- E1
    a3 = dargs(a, I133)
    G3 = r21_views.GroupedMVDataset(a3, "train")
    new = sorted(set(G3.get_new_tokens()))
    t3 = T5Tokenizer.from_pretrained(a.base_model)
    t3.add_tokens(new)
    st = enc_struct.StructInput(len(t3), 8)
    st.set_table(t3)
    j = max(range(len(G3)), key=lambda i: len(G3.inter_data[i]["inters"]))
    n_items = len(G3.inter_data[j]["inters"])
    per, codes = [], []
    for v in G3[j]["input_ids"]:
        ids = torch.tensor([t3.convert_tokens_to_ids(t3.tokenize(v))])
        s, code, item, rec, n = st.structure(ids)
        per.append(ids.shape[1] / n_items)
        codes.append((int(code.sum()), int(n)))
    cond = per == [7, 4, 4] and codes == [(7 * n_items, n_items), (4 * n_items, n_items), (4 * n_items, n_items)]
    st2 = enc_struct.StructInput(len(tok), 8); st2.set_table(tok)
    ids2 = torch.tensor([tok.convert_tokens_to_ids(tok.tokenize(G[idx[1]]["input_ids"][0]))])
    s2, c2, _, _, n2 = st2.structure(ids2)
    cond2 = bool((c2 == (s2 < enc_struct.OTHER)).all()) and int(c2.sum()) == 5 * int(n2)
    check("6_1+3+3_encoder_counts_all_codes", cond and cond2, tokens_per_item=per, codes_and_items=codes, items=n_items,
          one_plus_two_plus_two_unchanged=cond2)
    tab = mfd_data.item_token_table(a3, t3)
    dec, lab = mfd_data.branch_tensors(tab, torch.tensor([0, 1]), 1, 2)
    books = load_codebooks(os.path.join(a.v1_dir.replace("1+2+2", "1+3+3"), "tokenizer.pth"))
    lv = sorted({l for _, l, _ in code_tokens(t3)})
    check("7_1+3+3_decoder_rows_and_levels", dec.shape[-1] == 5 and lab.shape[-1] == 5 and lv == sorted(books) == ["s0", "t0", "t1", "t2", "v0", "v1", "v2"],
          row_len=dec.shape[-1], token_levels=lv, codebook_levels=sorted(books))
    res["all_pass"] = ok
    json.dump(res, open(a.out_file, "w"), indent=1, default=str)
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
