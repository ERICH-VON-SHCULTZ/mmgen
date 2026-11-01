"""mmgen addition: first-token accuracy of a trained MACRec/MQL4GRec-style model (docs/PLUGIN_REPORT.md §5).

For each list (seqrec = text SID, seqimage = image SID) and split: the model's first decoding step on exactly the
inputs of test_ddp_save.py (SeqRecDataset + TestCollatorSave, history only), restricted to the tokens the trie
allows at the first step (first token of every catalog SID, as in test_ddp_save's Trie of [0] + encode(SID)).
Reports how often the target's first token (f for the plug-in runs; C's first code level for C) is the top-1 and
within the top 20, plus the number of distinct first tokens (the size of that choice).
Writes <ckpt>/eval/first_token_<split>.json.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from collator import TestCollatorSave
from data import SeqRecDataset
from utils import parse_dataset_args, parse_global_args, parse_test_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).cuda().eval()
    start = model.config.decoder_start_token_id
    for split in args.splits.split(","):
        out = {}
        for task in ("seqrec", "seqimage"):
            ds = SeqRecDataset(args, task=task, mode=split)
            ds.set_prompt(0)
            first = sorted({tok.encode(c)[0] for c in ds.get_all_items()})
            allowed = torch.tensor(first, device="cuda")
            pos = {t: i for i, t in enumerate(first)}
            ranks = []
            for inputs, targets, _ in DataLoader(ds, batch_size=args.test_batch_size, collate_fn=TestCollatorSave(args, tok)):
                ids, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
                logits = model(input_ids=ids, attention_mask=mask,
                               decoder_input_ids=torch.full((ids.shape[0], 1), start, device="cuda")).logits[:, -1]
                lg = logits[:, allowed].float()
                tgt = torch.tensor([pos[tok.encode(t)[0]] for t in targets], device="cuda")
                ranks.append((lg > lg.gather(1, tgt[:, None])).sum(1).cpu())
            r = torch.cat(ranks)
            out[task] = {"users": len(r), "first_tokens": len(first), "top1": (r == 0).float().mean().item(),
                         f"top{args.k}": (r < args.k).float().mean().item(), "mean_rank": r.float().mean().item()}
            print(split, task, json.dumps(out[task]), flush=True)
        os.makedirs(os.path.join(args.ckpt_path, "eval"), exist_ok=True)
        json.dump(out, open(os.path.join(args.ckpt_path, "eval", f"first_token_{split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
