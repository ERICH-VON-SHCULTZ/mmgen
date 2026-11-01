"""mmgen addition: sanity checks 2-3 for MFD with C's task mix (docs/MFD_REPORT.md §13), before training. CPU.

2. Inputs and targets of every task match the mapping (mfd_mt_data.py): one printed sample per task, and on
   --n_check random examples per task (all of them for the item tasks): the input is [prompt +] the right view of
   the history (or of the item itself), and the collated labels carry loss on exactly the targeted branch codes
   plus f once.
3. Examples per epoch per task: seqrec / seqimage / fusionseqrec = C's seqrec count (SeqRecDataset train split),
   item2image / image2item = number of catalog items (C's ItemImageDataset).
Uses the same tokenizer construction as training (base T5 tokenizer + added SID / branch tokens) and C's soft
prompts. Writes --out_file (JSON) and exits 1 on any failure.
"""
import argparse
import json
import random
import sys

from transformers import T5Tokenizer

import mfd_mt_data as mt
from data import SeqRecDataset
from mfd_data import MFDCollator, load_mfd_datasets, view_maps
from utils import parse_dataset_args, parse_global_args


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p)
    p.add_argument("--out_file", required=True)
    p.add_argument("--n_check", type=int, default=5000)
    args = p.parse_args()
    random.seed(0)
    tok = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    args.soft_prompts = mt.c_soft_prompts(args.tasks, tok, args.prompt_num)
    train, valid = load_mfd_datasets(args)
    for d in train.datasets:
        tok.add_tokens(d.get_new_tokens())
    coll = MFDCollator(args, tok)
    idx = train.datasets[0].indices
    vmap = view_maps(idx)
    sid = {int(k): "".join(v) for k, v in idx.items()}
    n_seq = len(SeqRecDataset(args, task="seqrec", mode="train"))
    rep = {"tasks": args.tasks, "soft_prompts": args.soft_prompts, "examples_per_task": {}, "expected": {},
           "samples": {}, "checked": {}, "failures": []}

    def fail(msg):
        rep["failures"].append(msg)

    for ds in list(train.datasets) + [valid.datasets[0]]:
        name = ds.task if ds in train.datasets else f"valid:{ds.task}"
        is_cross = ds.task in mt.CROSS
        is_seq = ds.task in mt.SEQ or is_cross
        view, branch = (mt.SEQ[ds.task][:2] if ds.task in mt.SEQ else ("?", "?") if is_cross else mt.ITEM[ds.task])
        prompt = args.soft_prompts[ds.task] if (not is_seq or is_cross or mt.SEQ[ds.task][2]) else ""
        n_dir = {"text": 0, "image": 0}
        if ds in train.datasets:
            rep["examples_per_task"][ds.task] = len(ds)
            rep["expected"][ds.task] = n_seq if is_seq else len(idx)
            if len(ds) != rep["expected"][ds.task]:
                fail(f"{ds.task}: {len(ds)} examples, expected {rep['expected'][ds.task]}")
        picks = list(range(len(ds))) if (not is_seq or len(ds) <= args.n_check) else random.sample(range(len(ds)), args.n_check)
        for j in picks:
            ex = ds[j]
            lab = ex["labels"]
            item, br = (lab, "both") if not isinstance(lab, (tuple, list)) else lab
            if is_cross:                       # coin flip: text history -> image branch, image history -> text branch
                view, branch = dict((b, v) for v, b in mt.CROSS[ds.task])[br], br
                n_dir[br] += 1
            if br != branch:
                fail(f"{name}[{j}]: branch {br} != {branch}"); break
            if not ex["input_ids"].startswith(prompt):
                fail(f"{name}[{j}]: prompt missing"); break
            body = ex["input_ids"][len(prompt):]
            if is_seq:
                d = ds.inter_data[j]
                if sid[item] != d["item"] or body != "".join(vmap[view][h] for h in d["inters"]):
                    fail(f"{name}[{j}]: input/target mismatch"); break
            elif body != vmap[view][sid[item]]:
                fail(f"{name}[{j}]: item input is not the target item's {view} view"); break
            allowed = {"full": ("<s", "<t", "<v"), "text": ("<s", "<t"), "image": ("<s", "<v")}[view]
            toks = body.replace(">", ">\n").split()
            if not all(x.startswith(allowed) for x in toks):
                fail(f"{name}[{j}]: tokens outside the {view} view"); break
        rep["checked"][name] = len(picks)
        if is_cross:
            share = n_dir["image"] / max(len(picks), 1)
            rep.setdefault("cross_direction_share", {})[name] = {"text_history_to_image": share, "n": len(picks)}
            if abs(share - 0.5) > 0.05:
                fail(f"{name}: coin flip share {share:.3f}")
        # one printed sample + its collated decoder rows
        ex = ds[picks[0]]
        if is_cross:
            branch = ex["labels"][1]
        b = coll([ex])
        conv = lambda row: [("-" if x == -100 else tok.convert_ids_to_tokens(int(x))) for x in row]
        lab = b["labels"][0]
        rep["samples"][name] = {"input": ex["input_ids"][:400] + ("…" if len(ex["input_ids"]) > 400 else ""),
                                "target_item_sid": sid[ex["labels"] if not isinstance(ex["labels"], (tuple, list)) else ex["labels"][0]],
                                "branches": branch,
                                "dec_text": conv(b["decoder_input_ids"][0, 0]), "lab_text": conv(lab[0]),
                                "dec_image": conv(b["decoder_input_ids"][0, 1]), "lab_image": conv(lab[1])}
        # loss pattern: f once (text row, position 0) + the targeted branch's codes only
        L = (lab.shape[1] - 2)
        n_t = int((lab[0, 2:] != -100).sum()); n_i = int((lab[1] != -100).sum())
        want = {"both": (L, L), "text": (L, 0), "image": (0, L)}[branch]
        if int(lab[0, 0] != -100) != 1 or (n_t, n_i) != want or int(lab[1, :2].ne(-100).sum()) != 0:
            fail(f"{name}: label pattern f={int(lab[0, 0] != -100)} text={n_t} image={n_i}, want f=1 {want}")
        print(f"== {name}  ({len(ds)} examples, branches={branch})")
        for k, v in rep["samples"][name].items():
            print(f"   {k}: {v}")
    rep["all_pass"] = not rep["failures"]
    json.dump(rep, open(args.out_file, "w"), indent=1)
    print("examples per task:", rep["examples_per_task"], "expected:", rep["expected"])
    print("ALL PASS" if rep["all_pass"] else f"FAILED: {rep['failures']}")
    sys.exit(0 if rep["all_pass"] else 1)


if __name__ == "__main__":
    main()
