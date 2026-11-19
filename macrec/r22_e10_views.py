"""mmgen addition (docs/MFD_REPORT.md §22.10, E10a): per-view f distributions of a trained MFD checkpoint (P1a, G0, A01, ..).
For every TEST user (dataset order) and every view v in (full, text, image): log P(f | view-v history) at the f position
(mfd_mv_infer.f_logprobs, the deployment f path; teacher forcing at [start]), restricted to the catalog f codes and
RENORMALISED over them (as r22_e4_first --kind mfd; on the locked 1+2+2 SIDs every <s0_*> code has catalog items, so this is
the same set as A01's training-time teacher r21_views.f_logprobs_train). No training, no decoding beyond the f step.
Out (torch): {"full"|"text"|"image": (U, nF) float32 log-probs, "tokens": [nF f token strings]}.
Usage: python r22_e10_views.py --ckpt_path .. --data_path .. --dataset .. --index_file .. --image_index_file .. --tasks seqrec_mv
       --prompt_num 4 --max_his_len 20 --test_batch_size 64 --out_file ..
"""
import argparse

import torch
from transformers import T5Tokenizer

from utils import parse_dataset_args, parse_global_args, parse_test_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    from mfd_data import Catalog, view_maps
    from mfd_eval import load_mfd_model, load_split
    from mfd_mv_infer import VIEWS, encode, f_logprobs, view_batches
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    lps = {v: [] for v in VIEWS}
    for vb, _ in view_batches(ds, coll, vmap, args.test_batch_size):
        for v in VIEWS:
            enc, m = encode(model, vb[v]["input_ids"].cuda(), vb[v]["attention_mask"].cuda())
            lps[v].append(f_logprobs(model, cat, enc, m).float().log_softmax(-1).cpu())
    out = {v: torch.cat(lps[v]) for v in VIEWS}
    out["tokens"] = tok.convert_ids_to_tokens(cat.tok[0].tolist())
    print({v: tuple(out[v].shape) for v in VIEWS}, flush=True)
    torch.save(out, args.out_file)


if __name__ == "__main__":
    main()
