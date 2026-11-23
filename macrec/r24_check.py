"""mmgen addition (docs/MFD_REPORT.md §24): gate checks for the round-24 Track B item examples, on the trained P1a seed-42
checkpoint of one dataset (no training). Out: geo/<D>/r24_check.json (replaced only when every check passes).
  1 structure: R24StructInput == P1a's StructInput (all 5 outputs) on real P1a training rows (full / text / image views) and
    validation rows -> P1a's encoder input is unchanged for every row that contains an f token
  2 item input: B-text / B-image input ids contain no f token; one item per row; the encoder input vector equals
    E(t1) + E(t2) (+ recency 0) exactly (resp. E(i1) + E(i2)), and differs from the same row with E(f) added
  3 targets: the collated item rows carry the full SID (f, both branches) = branch_tensors of the item; example_weight = w on
    item rows, 1 on sequence rows
  4 loss: all weights 1 -> R24ItemMFDT5 loss == ItemMFDT5 loss; weights w -> == sum(w * token CE) / sum(w) (independent recompute)
  5 counts: examples per epoch, P1a sequence examples vs item examples (2 x catalog) vs C (optimizer steps / epoch x 1,024)
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import enc_struct
import r23_timing as RT
import r24_items
from mfd_data import MFDCollator, MVMFDDataset, branch_tensors
from mfd_eval import load_mfd_model
from transformers import T5Tokenizer

R = "/scratch/yw8866/mmgen_runs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--n", type=int, default=2048)
    a = p.parse_args()
    D = a.dataset
    torch.manual_seed(0)
    res, ok = {}, True
    ck = f"{R}/mfd/{D}/mfd122_mv_geoP1a/seed42"
    tok = T5Tokenizer.from_pretrained(ck)
    model = load_mfd_model(ck).cuda().eval()
    args = RT.mk_args(D, RT.I122, RT.I122, ["--max_his_len", "20", "--tasks", "seqrec_mv", "--prompt_num", "4"])
    old = model.struct
    new = r24_items.R24StructInput(len(tok), model.config.d_model).cuda()
    new.load_state_dict(old.state_dict())
    train = MVMFDDataset(args, "train")
    g = torch.Generator().manual_seed(0)
    pick = torch.randint(len(train), (a.n,), generator=g).tolist()
    coll = MFDCollator(args, tok)

    # 1 structure identical on P1a rows (train: all three views; valid rows via the eval dataset)
    from mfd_eval import load_split
    ds_v, coll_v = load_split(args, "valid", tok)
    same, rows = True, 0
    for s0 in range(0, a.n, 256):
        b = coll([train[i] for i in pick[s0:s0 + 256]])
        ids = b["input_ids"].cuda()
        same &= all(torch.equal(x, y) for x, y in zip(old.structure(ids), new.structure(ids)))
        rows += ids.shape[0]
    vb = coll_v([ds_v[i] for i in range(min(1024, len(ds_v)))])
    ids = vb["input_ids"].cuda()
    same &= all(torch.equal(x, y) for x, y in zip(old.structure(ids), new.structure(ids)))
    rows += ids.shape[0]
    views = sorted({(i // len(train.inter_data)) for i in pick})
    res["1_structure_unchanged_on_p1a_rows"] = {"ok": bool(same), "rows": rows, "train_views_sampled": views}
    ok &= bool(same)

    # 2 item input
    items = r24_items.ItemSemDataset(args, 0.3)
    f_ids = set(i for t, i in tok.get_vocab().items() if t.startswith("<s0_"))
    sel = list(range(0, len(items), max(1, len(items) // 1000)))[:1000]
    rc = r24_items.R24Collator(args, tok)
    b = rc([items[i] for i in sel])
    ids = b["input_ids"].cuda()
    no_f = not any(int(x) in f_ids for x in ids.flatten().tolist())
    s, code, item, rec, n = new.structure(ids)
    one_item = bool((n == 1).all()) and bool((item[code] == 0).all()) and bool((rec[code] == 0).all())
    emb = model.shared(ids)
    x = (emb * code[..., None].float()).sum(1) + new.rec.weight[0]          # E2's position = sum of code embeddings + rec(0)
    exp = []
    from mfd_data import item_token_table
    T = item_token_table(args, tok).cuda()
    for k, i in enumerate(sel):
        it, _, mod, _ = items[i]["labels"]                               # (item, "item", modality, weight)
        row = T[it]
        L = (T.shape[1] - 1) // 2
        cols = row[1:1 + L] if mod == "text" else row[1 + L:]
        exp.append(model.shared(cols).sum(0) + new.rec.weight[0])
    exp = torch.stack(exp)
    eq = float((x - exp).abs().max())
    with_f = torch.stack([model.shared(T[items[i]["labels"][0], 0]) for i in sel]) + exp
    differs = float((x - with_f).abs().max())
    # the real struct_encode path builds the same vector (capture the encoder's inputs_embeds)
    cap = {}
    h = model.encoder.register_forward_pre_hook(lambda m, a, k: cap.__setitem__("x", k["inputs_embeds"]), with_kwargs=True)
    model.struct = new
    with torch.no_grad():
        model.struct_encode(ids, b["attention_mask"].cuda())
    h.remove()
    eq_path = float((cap["x"][:, 0] - exp).abs().max()) if cap["x"].shape[1] == 1 else float("inf")
    c2 = no_f and one_item and eq < 1e-5 and eq_path < 1e-5 and differs > 1e-3
    res["2_item_input_no_f"] = {"ok": c2, "rows": len(sel), "no_f_token": no_f, "one_item_rec0": one_item,
                                "max_abs_vs_Et1+Et2": eq, "struct_encode_path_max_abs": eq_path, "max_abs_vs_with_Ef": differs}
    ok &= c2

    # 3 targets and weights
    dec_ref, lab_ref = branch_tensors(rc.tab, torch.tensor([items[i]["labels"][0] for i in sel]), rc.T, rc.I)
    seq = rc([train[i] for i in pick[:64]])
    c3 = torch.equal(b["labels"], lab_ref) and torch.equal(b["decoder_input_ids"], dec_ref) \
        and bool((b["example_weight"] == 0.3).all()) and bool((seq["example_weight"] == 1).all()) \
        and torch.equal(seq["labels"], coll([train[i] for i in pick[:64]])["labels"])
    res["3_item_targets_full_sid_and_weights"] = {"ok": bool(c3), "f_target_rows": int((b["labels"][:, 0, 0] != -100).sum()),
                                                  "rows": len(sel)}
    ok &= bool(c3)

    # 4 loss
    mix = [train[i] for i in pick[:96]] + [items[i] for i in sel[:32]]
    mb = {k: v.cuda() for k, v in rc(mix).items()}
    model.__class__ = r24_items.R24ItemMFDT5
    with torch.no_grad():
        ones = model(**{k: v for k, v in mb.items() if k != "example_weight"}, example_weight=torch.ones_like(mb["example_weight"])).loss
        base = enc_struct.ItemMFDT5.forward(model, input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                                            decoder_input_ids=mb["decoder_input_ids"], labels=mb["labels"]).loss
        wl = model(**mb).loss
        lg = enc_struct.ItemMFDT5.forward(model, input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                                          decoder_input_ids=mb["decoder_input_ids"]).logits.double()
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), mb["labels"].reshape(-1), ignore_index=-100, reduction="none")
        v = (mb["labels"].reshape(-1) != -100).double()
        w = mb["example_weight"].double()[:, None, None].expand_as(mb["labels"]).reshape(-1) * v
        manual = float((w * ce).sum() / w.sum())
    c4 = abs(float(ones) - float(base)) < 1e-5 and abs(float(wl) - manual) < 1e-4
    res["4_weighted_loss"] = {"ok": c4, "loss_w1": float(ones), "loss_p1a": float(base), "loss_w": float(wl), "manual_float64": manual}
    ok &= c4

    # 5 counts
    cs = json.load(open(f"{R}/baselines/{D}/mql/seed42/train_stats.json"))
    res["5_examples_per_epoch"] = {"ok": True, "p1a_sequence_examples": len(train), "item_examples": len(items),
                                   "catalog_items": len(items) // 2,
                                   "C_examples_approx": round(cs["global_steps"] / cs["epochs_run"]) * 1024,
                                   "C_note": "C's optimizer steps per epoch x effective batch 1,024 (its stats do not record examples)"}
    res["all_pass"] = bool(ok)
    print(json.dumps(res, indent=1))
    if ok:
        json.dump(res, open(a.out_file, "w"), indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
