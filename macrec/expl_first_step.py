"""mmgen addition (docs/EXPLORATION.md Part 1): per-user rank of the target's first SID token at the first decoding
step, for any MACRec-style checkpoint (C / plug-in: seqrec + seqimage; MD chain / MFD on the MD index: seqrec, whose
first token is f). Same inputs and first-step restriction as first_token_acc.py (SeqRecDataset + TestCollatorSave;
allowed = first token of every catalog SID, as test_ddp_save's trie). Users in dataset order.
Writes --out_file (torch): {task: {"rank": LongTensor (0 = top-1), "first_tokens": n}}.
"""
import argparse

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
    p.add_argument("--split", default="test")
    p.add_argument("--first_tasks", default="seqrec,seqimage")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).cuda().eval()
    start = model.config.decoder_start_token_id
    out = {}
    for task in args.first_tasks.split(","):
        ds = SeqRecDataset(args, task=task, mode=args.split)
        ds.set_prompt(0)
        first = sorted({tok.encode(c)[0] for c in ds.get_all_items()})
        allowed = torch.tensor(first, device="cuda")
        pos = {t: i for i, t in enumerate(first)}
        ranks = []
        for inputs, targets, _ in DataLoader(ds, batch_size=args.test_batch_size, collate_fn=TestCollatorSave(args, tok)):
            ids, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            lg = model(input_ids=ids, attention_mask=mask,
                       decoder_input_ids=torch.full((ids.shape[0], 1), start, device="cuda")).logits[:, -1][:, allowed].float()
            tgt = torch.tensor([pos[tok.encode(t)[0]] for t in targets], device="cuda")
            ranks.append((lg > lg.gather(1, tgt[:, None])).sum(1).cpu())
        r = torch.cat(ranks)
        out[task] = {"rank": r, "first_tokens": len(first)}
        print(task, len(r), "first tokens", len(first), "top1", (r == 0).float().mean().item(),
              "top20", (r < 20).float().mean().item(), flush=True)
    torch.save(out, args.out_file)


if __name__ == "__main__":
    main()
