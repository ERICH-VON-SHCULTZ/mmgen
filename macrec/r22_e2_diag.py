"""mmgen addition (docs/MFD_REPORT.md §22.2, E2 step 1): per TEST user, the rank of the target's first text code t1 and first
image code i1 under the model's branch distributions GIVEN THE TARGET'S f (decoder [start, f*, <T>] -> log P(t1 | f*),
[start, f*, <I>] -> log P(i1 | f*); full view, inference only). Ranks (0 = top) among (a) all codes of that level and (b) only
the codes that occur under f* in the catalog. Combined with C's text / image lists by tools/r22_e2_report.py.
Out: torch file {tf, t1, i1, rank_t1_all, rank_t1_inf, rank_i1_all, rank_i1_inf, n_t1_under_f, n_i1_under_f}.
"""
import argparse
import os

import torch
from transformers import T5Tokenizer

from mfd_data import Catalog, view_maps
from mfd_eval import load_mfd_model, load_split
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import encode, view_batches
from utils import parse_dataset_args, parse_global_args, parse_test_args


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = load_mfd_model(args.ckpt_path).cuda().eval()
    model.config.use_cache = False
    cat = Catalog(args, tok, "cuda")
    T, I = tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
    lt, li = cat.levels["t"][0], cat.levels["i"][0]
    nF = cat.nF
    present = {}
    for name, lv in (("t", lt), ("i", li)):                  # (nF, n codes) bool: code occurs under f in the catalog
        m = torch.zeros(nF, len(cat.tok[lv]), dtype=torch.bool, device="cuda")
        m[cat.code[:, 0], cat.code[:, lv]] = True
        present[name] = m
    ds, coll = load_split(args, "test", tok)
    vmap = view_maps(ds.indices)
    out = {k: [] for k in ("tf", "t1", "i1", "rank_t1_all", "rank_t1_inf", "rank_i1_all", "rank_i1_inf", "n_t1_under_f", "n_i1_under_f")}
    start = model.config.decoder_start_token_id
    with torch.no_grad():
        for vb, tgt in view_batches(ds, coll, vmap, args.test_batch_size):
            tgt = tgt.cuda()
            B = len(tgt)
            enc, m = encode(model, vb["full"]["input_ids"].cuda(), vb["full"]["attention_mask"].cuda())
            tf = cat.code[tgt, 0]
            ftok = cat.tok[0][tf]
            rows = torch.arange(B, device="cuda")
            z = torch.full_like(ftok, start)
            for name, lv, br in (("t1", lt, T), ("i1", li, I)):
                dec = torch.stack([z, ftok, torch.full_like(ftok, br)], 1)
                lp = model.last_logprobs(enc, m, rows, dec, cols=[cat.tok[lv]])[0]          # (B, n codes)
                y = cat.code[tgt, lv]
                ty = lp.gather(1, y[:, None])
                pres = present[name[0]][tf]
                out[f"rank_{name}_all"].append((lp > ty).sum(1).cpu())
                out[f"rank_{name}_inf"].append(((lp > ty) & pres).sum(1).cpu())
                out[f"n_{name}_under_f"].append(pres.sum(1).cpu())
                out[name].append(y.cpu())
            out["tf"].append(tf.cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    # codebook organisation: for each t1 / i1 code, the number of distinct f it appears under in the catalog
    out["t1_codes_n_f"] = present["t"].sum(0).cpu()
    out["i1_codes_n_f"] = present["i"].sum(0).cpu()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    torch.save(out, args.out_file)
    print({k: float(v.float().median()) for k, v in out.items() if k.startswith("rank")})


if __name__ == "__main__":
    main()
