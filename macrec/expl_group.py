"""mmgen addition (docs/EXPLORATION.md Part 3, extra check): does the f distribution of a trained model put its mass in
the right coarse REGION of f codes when its top-1 f is wrong? Inference only.

Groups of f codes = k-means (seed 0, sklearn) of the f codebook vectors of the SID's own tokenizer (G in 16/32/64), or a
random partition of the same sizes (control). Group score = logsumexp of log P(f) over the group's f codes (first decoder
step, restricted to catalog f; same inputs as mfd_eval). Reports group top-1 / top-5 accuracy, the same for random groups,
and f-level coverage when the kept K f codes are chosen inside the top groups (keep f by log P(f) + log P(group)).
Writes --out_file (json).
"""
import argparse
import json

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import T5Tokenizer

from mfd_data import Catalog, MFDCollator, MFDDataset
from mfd_model import MFDT5
from utils import parse_dataset_args, parse_global_args, parse_test_args


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    tok = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = MFDT5.from_pretrained(args.ckpt_path).cuda().eval()
    cat = Catalog(args, tok, "cuda")
    coll = MFDCollator(args, tok)
    sd = torch.load(args.tokenizer_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    book = sd["shared_rq.vq_layers.0.embedding.weight"].float().numpy()
    fcode = [int(t.split("_")[1][:-1]) for t in tok.convert_ids_to_tokens(cat.tok[0].tolist())]   # local f -> code
    vec = book[fcode]
    nF = len(fcode)
    rng = np.random.default_rng(0)
    groups = {}
    for G in (16, 32, 64):
        groups[f"kmeans{G}"] = KMeans(G, n_init=10, random_state=0).fit_predict(vec)
        groups[f"random{G}"] = rng.permutation(np.arange(nF) % G)
    res = {"ckpt": args.ckpt_path, "n_f": nF, "splits": {}}
    for split in args.splits.split(","):
        ds = MFDDataset(args, split)
        LP, TF = [], []
        for s in range(0, len(ds), args.test_batch_size):
            batch = [ds[j] for j in range(s, min(s + args.test_batch_size, len(ds)))]
            inp = coll(batch)
            ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
            lp = model.last_logprobs(model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask,
                                     torch.arange(len(batch), device="cuda"),
                                     torch.full((len(batch), 1), model.config.decoder_start_token_id, device="cuda"),
                                     cols=[cat.tok[0]])[0]
            LP.append(lp.cpu()); TF.append(cat.code[torch.tensor([d["labels"] for d in batch], device="cuda"), 0].cpu())
        lp, tf = torch.cat(LP), torch.cat(TF)
        fr = (lp > lp.gather(1, tf[:, None])).sum(1)
        out = {"users": len(tf), "f_top1": (fr == 0).float().mean().item(), "f_top20": (fr < 20).float().mean().item()}
        for name, g in groups.items():
            gt = torch.tensor(g)
            G = int(gt.max()) + 1
            m = lp.max(1, keepdim=True).values
            gl = torch.zeros(len(tf), G).index_add_(1, gt, (lp - m).exp()).log() + m       # log P(group)
            tg = gt[tf]
            gr = (gl > gl.gather(1, tg[:, None])).sum(1)
            d = {"groups": G, "group_top1": (gr == 0).float().mean().item(), "group_top5": (gr < 5).float().mean().item(),
                 "group_top1_when_f_top1_wrong": (gr[fr > 0] == 0).float().mean().item(),
                 "mean_f_per_group": nF / G}
            for K in (20, 100):
                sc = lp + gl[:, gt]                                                   # keep f inside likely groups
                kth = sc.topk(K, 1).values[:, -1:]
                d[f"coverage_K{K}_group_keep"] = (sc.gather(1, tf[:, None]) >= kth).float().mean().item()
                d[f"coverage_K{K}_base"] = (fr < K).float().mean().item()
            out[name] = d
        res["splits"][split] = out
        print(split, json.dumps(out), flush=True)
    json.dump(res, open(args.out_file, "w"), indent=1)


if __name__ == "__main__":
    main()
