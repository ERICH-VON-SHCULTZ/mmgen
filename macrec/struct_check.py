"""mmgen addition: sanity checks of the item-level encoder structure before training (docs/MFD_REPORT.md §16).

4. E1 at initialisation equals the control: a PlainT5 and a StructT5 with the same weights (struct embeddings zero)
   give identical logits on a real validation batch and identical beam-search outputs.
5. E2: the encoder sequence length equals the number of history items for every example of the batch; one encoded
   example is printed (tokens -> items -> recency); a forward pass with labels and a beam search run.
Uses the training tokenizer construction (base T5 tokenizer + the dataset's SID tokens). Writes --out_file; exit 1 on
any failure.
"""
import argparse
import json
import sys

import torch
from transformers import T5Config, T5Tokenizer

from collator import Collator
from data import SeqRecDataset
from enc_struct import SLOTS, StructT5
from finetune_ext import PlainT5
from utils import parse_dataset_args, parse_global_args, parse_train_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_train_args(p); p = parse_dataset_args(p)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    rep, fails = {"checks": {}}, []

    def ok(name, cond, **kw):
        rep["checks"][name] = {"ok": bool(cond), **kw}
        if not cond:
            fails.append(name)

    tok = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    ds = SeqRecDataset(args, task="seqrec", mode="valid")
    tok.add_tokens(ds.get_new_tokens())
    batch = [ds[j] for j in range(0, 800, 100)]
    inp = Collator(args, tok)(batch)
    ids, mask, lab = inp["input_ids"].cuda(), inp["attention_mask"].cuda(), inp["labels"].cuda()
    torch.manual_seed(0)
    ctl = PlainT5(T5Config.from_pretrained(args.base_model))
    ctl.resize_token_embeddings(len(tok))
    with torch.no_grad():                                   # distinct rows for the added tokens (HF mean-resizing makes them equal)
        w = ctl.shared.weight
        w[32100:] = torch.randn_like(w[32100:]) * w[:32100].std()
    ctl = ctl.cuda().eval()
    gen = dict(max_new_tokens=6, num_beams=4, num_return_sequences=4, early_stopping=True)
    for mode in ("E1", "E2"):
        m = StructT5(ctl.config)
        m.resize_token_embeddings(len(tok))
        res = m.load_state_dict(ctl.state_dict(), strict=False)
        info = m.install_struct(tok, mode)
        m = m.cuda().eval()
        rep[mode] = info | {"missing_keys": res.missing_keys}
        ok(f"{mode}_only_struct_params_new", all(k.startswith("struct.") for k in res.missing_keys) and not res.unexpected_keys)
        ok(f"{mode}_slot_table", all(info["slot_tokens"][s] > 0 for s in SLOTS), slot_tokens=info["slot_tokens"])
        out = m(input_ids=ids, attention_mask=mask, labels=lab)
        g = m.generate(input_ids=ids, attention_mask=mask, **gen)
        if mode == "E1":
            ref = ctl(input_ids=ids, attention_mask=mask, labels=lab)
            ok("E1_zero_init_logits_equal_control", torch.equal(out.logits, ref.logits),
               max_abs_diff=float((out.logits - ref.logits).abs().max()))
            ok("E1_zero_init_loss_equal_control", abs(out.loss.item() - ref.loss.item()) < 1e-6,
               loss=out.loss.item(), control_loss=ref.loss.item())
            ok("E1_zero_init_generate_equal_control", torch.equal(g, ctl.generate(input_ids=ids, attention_mask=mask, **gen)))
        else:
            enc, emask = m.struct_encode(ids, mask)
            s, code, item, rec, n = m.struct.structure(ids)
            hist = torch.tensor([len(ds.inter_data[j]["inters"]) for j in range(0, 800, 100)])
            ok("E2_encoder_length_is_items", torch.equal(emask.sum(1).cpu(), hist) and enc.last_hidden_state.shape[1] == int(hist.max()),
               items=hist.tolist(), encoder_lengths=emask.sum(1).tolist(), padded_length=int(enc.last_hidden_state.shape[1]),
               token_lengths=mask.sum(1).tolist())
            ok("E2_loss_finite", bool(torch.isfinite(out.loss)), loss=out.loss.item())
            ok("E2_generate_runs", g.shape[0] == len(batch) * 4, shape=list(g.shape))
            u = 0
            toks = tok.convert_ids_to_tokens(ids[u][mask[u].bool()].tolist())
            rep["E2_example"] = {"history_items": int(hist[u]), "encoder_positions": int(emask[u].sum()),
                                 "tokens": [(t, int(it), int(r)) for t, it, r, c in
                                            zip(toks, item[u].tolist(), rec[u].tolist(), code[u].tolist())][:40]}
            print("E2 example: history items", int(hist[u]), "-> encoder positions", int(emask[u].sum()))
            for t, it, r in rep["E2_example"]["tokens"][:25]:
                print(f"   {t:12s} item {it:2d} recency {r:2d}")
    rep["all_pass"] = not fails
    rep["failures"] = fails
    json.dump(rep, open(args.out_file, "w"), indent=1, default=str)
    for k, v in rep["checks"].items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k} {({a: b for a, b in v.items() if a != 'ok'})}")
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
