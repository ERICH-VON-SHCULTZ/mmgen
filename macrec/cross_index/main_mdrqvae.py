"""mmgen addition: train MD-RQ-VAE (models/mdrqvae.py) on one MACRec dataset.

Inputs are the baselines' own feature files (<D>.emb-llama-td.npy, <D>.emb-ViT-L-14.npy), rows in
item-id order; the tokenizer trains on the whole catalog, as the baseline tokenizers do.
Writes into --out_dir: run_config.json (before training), train_log.jsonl (every --eval_every
epochs: losses, codebook utilisation per level, full-SID collision rate, reconstruction cosine),
best.pth (lowest full-SID collision rate, later epoch wins ties, as main_single.py / A,B select)
and final.pth.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.mdrqvae import MDRQVAE


def prep(x, mode):
    """Input handling. The reconstruction target is the same prepared vector."""
    x = torch.from_numpy(x.astype(np.float32))
    if mode == "raw":
        return x
    x = F.normalize(x, dim=-1)
    if mode == "l2":
        return x
    if mode == "l2_sqrtd":                  # unit per-dimension RMS
        return x * x.shape[-1] ** 0.5
    raise ValueError(mode)


@torch.no_grad()
def evaluate(model, xt, xi, bs=4096):
    model.eval()
    codes, ct, ci = [], [], []
    for s in range(0, xt.shape[0], bs):
        e = model(xt[s:s + bs], xi[s:s + bs])
        codes.append(e["codes"])
        ct.append(F.cosine_similarity(e["xh_t"], xt[s:s + bs], dim=-1))
        ci.append(F.cosine_similarity(e["xh_i"], xi[s:s + bs], dim=-1))
    codes = torch.cat(codes).cpu().numpy()
    model.train()
    n = codes.shape[0]
    uniq = len({tuple(r) for r in codes.tolist()})
    return codes, {"util": [int(len(np.unique(codes[:, j]))) for j in range(codes.shape[1])],
                   "full_sid_collision": (n - uniq) / n, "unique_sids": uniq,
                   "rec_cos_text": float(torch.cat(ct).mean()), "rec_cos_image": float(torch.cat(ci).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Instruments")
    p.add_argument("--data_root", default="/scratch/yw8866/mmgen_runs/data")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_shared", type=int, default=1)
    p.add_argument("--n_modal", type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    p.add_argument("--input_mode", default="l2_sqrtd", choices=["raw", "l2", "l2_sqrtd"])
    p.add_argument("--fusion", default="mlp", choices=["mlp", "xattn"])
    p.add_argument("--info_nce_weight", type=float, default=1.0)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--dead_code_reset", type=int, default=0, help="reset unused codes every N epochs (0 = off)")
    p.add_argument("--dead_code_until", type=int, default=0, help="stop resets after this epoch (0 = never)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--note", default="")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    d = f"{a.data_root}/{a.dataset}/{a.dataset}"
    xt = prep(np.load(f"{d}.emb-llama-td.npy"), a.input_mode).cuda()
    xi = prep(np.load(f"{d}.emb-ViT-L-14.npy"), a.input_mode).cuda()
    assert xt.shape[0] == xi.shape[0]
    model = MDRQVAE(xt.shape[1], xi.shape[1], n_shared=a.n_shared, n_modal=a.n_modal,
                    codebook_size=a.codebook_size, e_dim=a.e_dim, layers=a.layers, fusion=a.fusion,
                    info_nce_weight=a.info_nce_weight, quant_loss_weight=a.quant_loss_weight).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    cfg = dict(vars(a), items=int(xt.shape[0]), text_dim=int(xt.shape[1]), image_dim=int(xi.shape[1]),
               n_params=sum(q.numel() for q in model.parameters()), gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, job_id=os.environ.get("SLURM_JOB_ID", ""))
    json.dump(cfg, open(f"{a.out_dir}/run_config.json", "w"), indent=1)

    log = open(f"{a.out_dir}/train_log.jsonl", "w")
    best = (np.inf, -1)
    t0 = time.time()
    N = xt.shape[0]
    for ep in range(a.epochs):
        perm = torch.randperm(N, device=xt.device)
        agg = {}
        for s in range(0, N, a.batch_size):
            b = perm[s:s + a.batch_size]
            e = model(xt[b], xi[b])
            loss, parts = model.compute_loss(e, xt[b], xi[b])
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in dict(parts, loss=loss.item()).items():
                agg[k] = agg.get(k, 0) + v * len(b) / N
        resets = 0
        if a.dead_code_reset and (ep + 1) % a.dead_code_reset == 0 and (not a.dead_code_until or ep < a.dead_code_until):
            model.eval()
            with torch.no_grad():
                resets = model.reset_dead_codes(model.encode(xt, xi))
            model.train()
        if (ep + 1) % a.eval_every == 0 or ep == 0:
            _, st = evaluate(model, xt, xi)
            rec = dict(epoch=ep + 1, seconds=round(time.time() - t0, 1), resets=resets, **agg, **st)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            if st["full_sid_collision"] <= best[0]:
                best = (st["full_sid_collision"], ep + 1)
                torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": ep + 1}, f"{a.out_dir}/best.pth")
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": a.epochs}, f"{a.out_dir}/final.pth")
    json.dump({"best_epoch": best[1], "best_full_sid_collision": best[0], "train_seconds": time.time() - t0},
              open(f"{a.out_dir}/train_summary.json", "w"), indent=1)
    print("best", best)


if __name__ == "__main__":
    main()
