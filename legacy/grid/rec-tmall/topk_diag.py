"""What occupies the top-10 when MFD misses? (MFD-only repeats stuck at 0.51; covered new items at 0.001)
Trains the standard MFD reranker (R2_rw1 capped, seed 0) with weights kept, then per TEST event classifies
candidates ahead of the GT: SAME = GT's own item under another behavior (behavior hedge); HIST = another
history item (competing repeat); NEW = non-history candidate."""
import argparse, copy, json, os, sys
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))
import numpy as np, torch
from coverage_expansion import Pool, log
from sweep2_rerank import Ranker, make_scal
import nip_rerank as NIP

def main():
    ap = argparse.ArgumentParser()
    for k in ("spec", "other_result", "model_path", "out"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--arm", default="MFD")
    for k, v in (("cap", 36864), ("heads", 4), ("epochs", 15), ("patience", 3), ("batch", 8), ("eval_batch", 16), ("seed", 0)):
        ap.add_argument("--" + k, type=int, default=v)
    for k, v in (("wb", 1.0), ("dropout", 0.3), ("lr", 1e-3), ("wd", 1e-3), ("min_delta", 0.0)):
        ap.add_argument("--" + k, type=float, default=v)
    args = ap.parse_args(); args.specialist = False; args._model_path = args.model_path
    fh = open(args.out, "w"); dev = "cuda"
    sp = json.load(open(args.spec))[args.arm]
    pool = Pool(args.arm, sp["tr"], sp["va"], sp["te"], sp["sid"], args.wb, dev, fh, kpool=sp.get("kpool"))
    if not os.path.exists(args.model_path):
        NIP.run_cell(pool, 1.0, args.cap, dev, copy.copy(args), fh)
    g = torch.Generator().manual_seed(args.seed + 4242)
    U = [(torch.randn(w, 128, generator=g) * 0.02).to(dev) for w in pool.widths]
    model = Ranker(U, 128, pool.tr.mem.shape[2], 2, args.heads, args.dropout, pool.nb, 0).to(dev)
    model.load_state_dict(torch.load(args.model_path, map_location=dev)["main"]); model.eval()
    d = pool.te; N = d.N; uid = pool.uid["te"].numpy()
    ch = torch.load(args.other_result, map_location="cpu", weights_only=False)
    cmap = dict(zip(ch["uid"].tolist(), ch["has_gt"].astype(bool).tolist()))
    ccov = np.array([cmap[u] for u in uid])
    mcov = d.has_gt.numpy(); rep = NIP.gt_is_repeat(d); pv = d.gt_b.numpy() == 0
    rank = np.full(N, -1); comp = np.zeros((N, 3)); above = np.zeros((N, 3))
    nhist = np.zeros(N); brank = np.full(N, -1); hist_above_base = np.zeros(N)
    with torch.no_grad():
        for s in range(0, N, args.eval_batch):
            idx = torch.arange(s, min(s + args.eval_batch, N))
            f, t, i, scal = make_scal(pool, "te", model, d, idx, dev, ())
            valid = d.valid[idx].to(dev)
            S = model(f, t, i, scal, d.base[idx].float().to(dev), valid, d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev))
            hm = NIP.hist_match(d, idx, f, t, i, dev)
            top = S.topk(10, -1).indices
            gt = d.gt_idx[idx].to(dev); has = gt >= 0; gc = gt.clamp(min=0)
            same = ((f == f.gather(1, gc[:, None])) & (t == t.gather(1, gc[:, None])) & (i == i.gather(1, gc[:, None]))
                    & (torch.arange(f.shape[1], device=dev)[None] != gc[:, None]))
            kind = torch.where(same, 0, torch.where(hm, 1, 2))
            tk = kind.gather(1, top)
            c3 = torch.stack([(tk == k).sum(1) for k in range(3)], 1).float()
            gsc = S.gather(1, gc[:, None]); r = (S > gsc).sum(1)
            ab = torch.stack([((S > gsc) & (kind == k) & valid).sum(1).clamp(max=10) for k in range(3)], 1).float()
            e = idx.numpy()
            bs = d.base[idx].float().to(dev).masked_fill(~valid, float("-inf")); gb_ = bs.gather(1, gc[:, None])
            nhist[e] = (hm & valid).sum(1).cpu().numpy()
            brank[e] = torch.where(has, (bs > gb_).sum(1), torch.full_like(r, -1)).cpu().numpy()
            hist_above_base[e] = ((bs > gb_) & hm & valid).sum(1).clamp(max=10).cpu().numpy()
            rank[e] = torch.where(has, r, torch.full_like(r, -1)).cpu().numpy()
            comp[e] = c3.cpu().numpy(); above[e] = ab.cpu().numpy()
    hit = (rank >= 0) & (rank < 10)
    pops = {"OWN-ONLY repeat, MISS": mcov & ~ccov & rep & pv & ~hit, "OWN-ONLY repeat, hit": mcov & ~ccov & rep & pv & hit,
            "BOTH repeat, MISS": mcov & ccov & rep & pv & ~hit, "BOTH repeat, hit": mcov & ccov & rep & pv & hit,
            "BOTH pv NEW, hit": mcov & ccov & ~rep & pv & hit, "BOTH pv NEW, MISS": mcov & ccov & ~rep & pv & ~hit,
            "OWN-ONLY pv NEW, MISS": mcov & ~ccov & ~rep & pv & ~hit}
    log("TOP-10 COMPOSITION (MFD reranker, seed 0, TEST). SAME=GT item other behavior; HIST=other history item; NEW=non-history", fh)
    log(f"  arm={args.arm}   pool: mean history candidates per event {nhist.mean():.1f} (p50 {np.median(nhist):.0f})", fh)
    log(f"  {'population':<24}{'n':>7}{'GT rank p50':>12}{'base rank p50':>14}{'hist above GT (base)':>21}  {'top10 SAME':>10}{'HIST':>6}{'NEW':>6}   {'above-GT SAME':>13}{'HIST':>6}{'NEW':>6}{'#hist in pool':>14}", fh)
    for nm, q in pops.items():
        if not q.sum(): continue
        c = comp[q].mean(0); a = above[q].mean(0)
        log(f"  {nm:<24}{int(q.sum()):>7,}{int(np.median(rank[q])):>12,}{int(np.median(brank[q])):>14,}{hist_above_base[q].mean():>21.2f}  {c[0]:>10.2f}{c[1]:>6.2f}{c[2]:>6.2f}   {a[0]:>13.2f}{a[1]:>6.2f}{a[2]:>6.2f}{nhist[q].mean():>14.1f}", fh)
    q = pops["OWN-ONLY repeat, MISS"]
    if q.sum():
        log(f"\n  MFD-ONLY repeat misses: GT rank 10-19 in {((rank[q] >= 10) & (rank[q] < 20)).mean():.3f}; "
            f"a SAME-item slot above the GT in {(above[q][:, 0] > 0).mean():.3f}", fh)
    fh.close()

if __name__ == "__main__":
    main()
