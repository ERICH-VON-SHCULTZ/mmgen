"""MFD stage 1 vs the MD chain decoder, per user (docs/MFD_REPORT.md §3, §5). Reads the saved MFD ranked lists
(ranked_{split}.pt) and the chain's beams (save_seqrec_20.json), recomputes pass 1 (log P(f)) of both models to
get each user's target-f rank, and reports:
  - paired bootstrap (same users, 2,000 resamples) of MFD mode - chain for HR@10 / NDCG@10;
  - ranking given coverage on the SAME users (target f in MFD's top K_f AND among the chain's 20 beams);
  - HR@10 by the target f's rank under each model's pass 1.
Run inside the env (sbatch/mfd/paired_mfd.sbatch). Writes paired_{split}.json into the eval dir.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, os.environ.get("MACREC", "/scratch/yw8866/mmgen/macrec"))
from mfd_data import Catalog, MFDCollator, MFDDataset, item_token_table  # noqa: E402
from mfd_model import MFDT5  # noqa: E402
from utils import parse_dataset_args, parse_global_args, parse_test_args  # noqa: E402


def rank_of(ids, tg, n=10):
    """0-based rank of the target in each user's list (n if absent from the top n)."""
    m = ids[:, :n] == tg[:, None]
    return torch.where(m.any(1), m.int().argmax(1), torch.full_like(tg, n))


def ndcg(r, n=10):
    return torch.where(r < n, 1.0 / torch.log2(r.float() + 2), torch.zeros(r.shape))


@torch.no_grad()
def f_ranks(model, coll, ds, cat, bs=256):
    start = model.config.decoder_start_token_id
    out = []
    for s in range(0, len(ds), bs):
        batch = [ds[j] for j in range(s, min(s + bs, len(ds)))]
        inp = coll(batch)
        ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
        o = model(input_ids=ids, attention_mask=mask,
                  decoder_input_ids=torch.full((ids.shape[0], 1), start, device="cuda")) \
            if not isinstance(model, MFDT5) else None
        if o is None:
            enc = model.encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            lp = model.last_logprobs(enc, mask, torch.arange(ids.shape[0], device="cuda"),
                                     torch.full((ids.shape[0], 1), start, device="cuda"), cols=[cat.tok[0]])[0]
        else:
            lp = o.logits[:, -1].float().log_softmax(-1)[:, cat.tok[0]]
        tf = cat.code[torch.tensor([d["labels"] for d in batch], device="cuda"), 0]
        out.append((lp > lp.gather(1, tf[:, None])).sum(1).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    p.add_argument("--mfd_run", required=True)
    p.add_argument("--chain_run", required=True)
    p.add_argument("--k_f", type=int, default=20)
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    ev = f"{args.mfd_run}/eval_kf{args.k_f}"
    for split in args.splits.split(","):
        R = torch.load(f"{ev}/ranked_{split}.pt")
        tg = R["targets"]
        U = len(tg)
        # MFD f ranks
        tok = T5Tokenizer.from_pretrained(args.mfd_run)
        mfd = MFDT5.from_pretrained(args.mfd_run).cuda().eval()
        cat = Catalog(args, tok, "cuda")
        ds = MFDDataset(args, split)
        assert torch.equal(torch.tensor([ds[j]["labels"] for j in range(len(ds))]), tg), "user order differs"
        fr_mfd = f_ranks(mfd, MFDCollator(args, tok), ds, cat)
        del mfd
        # chain f ranks + beams
        ctok = T5Tokenizer.from_pretrained(args.chain_run)
        chain = T5ForConditionalGeneration.from_pretrained(args.chain_run).cuda().eval()
        ccat = Catalog(args, ctok, "cuda")
        cc = MFDCollator.__new__(MFDCollator); cc.tokenizer = ctok
        cc.tab = item_token_table(args, ctok); cc.T = cc.I = ctok.pad_token_id
        fr_chain = f_ranks(chain, cc, ds, ccat)
        del chain
        sv = json.load(open(f"{args.chain_run}/eval/{split}/save_seqrec_20.json"))
        sid2item = ds.sid2item
        # the chain's eval saves users in sampler order; all_users[k] = dataset index of saved user k
        assert sorted(sv["all_users"]) == list(range(U))
        assert all(tg[u] == sid2item[t.strip().replace(" ", "")] for u, t in zip(sv["all_users"], sv["all_targets"]))
        f_of = lambda x: x.strip().replace(" ", "").split(">")[0] + ">"
        cids, cin_beams = torch.full((U, 20), -1), torch.zeros(U, dtype=torch.bool)
        for k, u in enumerate(sv["all_users"]):
            beams = sv["all_outputs"][k * 20:(k + 1) * 20]
            sc = sv["all_scores"][k * 20:(k + 1) * 20]
            order = sorted(range(20), key=lambda j: -sc[j])
            for r, j in enumerate(order):
                cids[u, r] = sid2item.get(beams[j].strip().replace(" ", ""), -1)
            cin_beams[u] = f_of(sv["all_targets"][k]) in {f_of(b) for b in beams}
        r_chain = rank_of(cids, tg)
        cov_mfd = fr_mfd < args.k_f
        res = {"users": U, "mfd_coverage": cov_mfd.float().mean().item(),
               "chain_target_f_among_beams": cin_beams.float().mean().item(),
               "chain_hr10": (r_chain < 10).float().mean().item(), "chain_ndcg10": ndcg(r_chain).mean().item(),
               "modes": {}}
        both = cov_mfd & cin_beams
        rng = np.random.default_rng(0)
        boot = rng.integers(0, U, size=(args.n_boot, U))
        buckets = [(0, 1), (1, 5), (5, 20), (20, 10 ** 9)]
        res["chain_hr10_by_f_rank"] = {f"{a + 1}-{b}" if b < 10 ** 9 else f">{a}":
                                       dict(users=int(((fr_chain >= a) & (fr_chain < b)).sum()),
                                            hr10=(r_chain[(fr_chain >= a) & (fr_chain < b)] < 10).float().mean().item())
                                       for a, b in buckets}
        for m in ("text", "image", "joint"):
            r = rank_of(R[m][0], tg)
            d_h = ((r < 10).float() - (r_chain < 10).float()).numpy()
            d_n = (ndcg(r) - ndcg(r_chain)).numpy()
            bh, bn = d_h[boot].mean(1), d_n[boot].mean(1)
            res["modes"][m] = dict(
                hr10=(r < 10).float().mean().item(), ndcg10=ndcg(r).mean().item(),
                diff_hr10=float(d_h.mean()), ci95_hr10=[float(np.percentile(bh, 2.5)), float(np.percentile(bh, 97.5))],
                diff_ndcg10=float(d_n.mean()), ci95_ndcg10=[float(np.percentile(bn, 2.5)), float(np.percentile(bn, 97.5))],
                wins=int(((r < 10) & (r_chain >= 10)).sum()), losses=int(((r >= 10) & (r_chain < 10)).sum()),
                hr10_given_both_covered=(r[both] < 10).float().mean().item(),
                hr10_given_mfd_covered=(r[cov_mfd] < 10).float().mean().item(),
                hr10_by_f_rank={f"{a + 1}-{b}" if b < 10 ** 9 else f">{a}":
                                dict(users=int(((fr_mfd >= a) & (fr_mfd < b)).sum()),
                                     hr10=(r[(fr_mfd >= a) & (fr_mfd < b)] < 10).float().mean().item())
                                for a, b in buckets})
        res["both_covered_users"] = int(both.sum())
        res["chain_hr10_given_both_covered"] = (r_chain[both] < 10).float().mean().item()
        res["chain_hr10_given_mfd_covered"] = (r_chain[cov_mfd] < 10).float().mean().item()
        res["chain_hr10_given_mfd_not_covered"] = (r_chain[~cov_mfd] < 10).float().mean().item()
        res["f_rank_agreement"] = dict(
            mfd_top1=(fr_mfd == 0).float().mean().item(), chain_top1=(fr_chain == 0).float().mean().item(),
            mfd_topK=cov_mfd.float().mean().item(), chain_topK=(fr_chain < args.k_f).float().mean().item(),
            both_topK=((fr_chain < args.k_f) & cov_mfd).float().mean().item(),
            spearman_like_same_rank=(fr_mfd == fr_chain).float().mean().item())
        json.dump(res, open(f"{ev}/paired_{split}.json", "w"), indent=1)
        print(f"[{split}]", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
