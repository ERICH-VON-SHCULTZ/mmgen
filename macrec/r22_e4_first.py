"""mmgen addition (docs/MFD_REPORT.md §22.4, E4): first-step (level-1) distribution of a trained model over its own
level-1 codebook, per TEST user, teacher forcing at the first SID position. No training, no decoding beyond step 1.

--kind chain  MACRec / MQL4GRec / TIGER checkpoint (T5ForConditionalGeneration): for each task in --first_tasks
              (seqrec = text SID, seqimage = image SID), exactly first_token_acc.py's inputs (SeqRecDataset, prompt 0,
              TestCollatorSave) and allowed set (first token of every catalog SID, as test_ddp_save's trie).
--kind mfd    MFD checkpoint (A01): log P(f) from the full-view encoding at the f position (mfd_mv_infer.f_logprobs,
              the deployment f distribution), over the catalog f codes.
Both: the full-vocabulary log-softmax restricted to the level-1 codes is RENORMALISED over those codes (log-softmax over
the allowed set), so NLLs of different partitions are proper distributions over their own codebook.
Users in dataset order (= inter.json order, as every other per-user tool here).
Out (torch): {name: {"lp": (U, n) float32 log-probs, "tokens": [n level-1 token strings]}}.
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils import parse_dataset_args, parse_global_args, parse_test_args


@torch.no_grad()
def chain(args, tok):
    from collator import TestCollatorSave
    from data import SeqRecDataset
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).cuda().eval()
    start = model.config.decoder_start_token_id
    out = {}
    for task in args.first_tasks.split(","):
        ds = SeqRecDataset(args, task=task, mode="test")
        ds.set_prompt(0)
        first = sorted({tok.encode(c)[0] for c in ds.get_all_items()})
        allowed = torch.tensor(first, device="cuda")
        lps = []
        for inputs, _, _ in DataLoader(ds, batch_size=args.test_batch_size, collate_fn=TestCollatorSave(args, tok)):
            ids, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
            lg = model(input_ids=ids, attention_mask=mask,
                       decoder_input_ids=torch.full((ids.shape[0], 1), start, device="cuda")).logits[:, -1].float()
            lps.append(lg[:, allowed].log_softmax(-1).cpu())
        out[task] = {"lp": torch.cat(lps), "tokens": tok.convert_ids_to_tokens(first)}
        print(task, out[task]["lp"].shape, flush=True)
    return out


@torch.no_grad()
def mfd(args, tok):
    from mfd_data import Catalog, view_maps
    from mfd_eval import load_mfd_model, load_split
    from mfd_mv_infer import encode, f_logprobs, view_batches
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    lps = []
    for vb, _ in view_batches(ds, coll, vmap, args.test_batch_size):
        enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
        lps.append(f_logprobs(model, cat, enc, m).log_softmax(-1).cpu())     # renormalise over the catalog f codes
    out = {"f": {"lp": torch.cat(lps), "tokens": tok.convert_ids_to_tokens(cat.tok[0].tolist())}}
    print("f", out["f"]["lp"].shape, flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--kind", choices=("chain", "mfd"), required=True)
    p.add_argument("--first_tasks", default="seqrec,seqimage")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    torch.save(chain(args, tok) if args.kind == "chain" else mfd(args, tok), args.out_file)


if __name__ == "__main__":
    main()
