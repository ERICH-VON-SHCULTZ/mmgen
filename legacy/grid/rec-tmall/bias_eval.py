"""Per-behavior score bias, tuned on VAL, applied identically in form to both arms (saved models, no retraining).
The top-10 hedges ~2 slots per item across behaviors because the reranker's within-item behavior ordering is
wrong ~1/3 of the time; a global per-behavior bias shifts that ordering for every item at once. Grid over
(pv, fav, cart, buy) biases, chosen per arm on VAL, reported on TEST with the group breakdown."""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))
import numpy as np, torch
from coverage_expansion import Pool, log
from sweep2_rerank import Ranker, make_scal
import nip_rerank as NIP
GRID = list(itertools.product((0.0, 0.5, 1.0, 1.5, 2.0, 3.0), (-2.0, -1.0, 0.0, 1.0), (-2.0, -1.0, 0.0, 1.0), (-2.0, -1.0, 0.0, 1.0)))

def scores(pool, m, split, d, dev):
    S = torch.empty(d.N, d.C, dtype=torch.float16)
    with torch.no_grad():
        for s in range(0, d.N, 16):
            idx = torch.arange(s, min(s + 16, d.N))
            f, t, i, scal = make_scal(pool, split, m, d, idx, dev, ())
            S[idx] = m(f, t, i, scal, d.base[idx].float().to(dev), d.valid[idx].to(dev), d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev)).half().cpu()
    return S

def hits(S, d, bias, dev):
    br = d.branch.to(dev); b = torch.tensor(bias, device=dev)[br]; out = np.zeros(d.N, bool)
    gt = d.gt_idx.to(dev)
    for s in range(0, d.N, 4096):
        x = S[s:s + 4096].to(dev).float() + b; g = gt[s:s + 4096]; has = g >= 0
        gs = x.gather(1, g.clamp(min=0)[:, None]); r = (x > gs).sum(1)
        out[s:s + 4096] = (has & (r < 10)).cpu().numpy()
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"; fh = open(a.out, "w")
sp = json.load(open(a.spec))[a.arm]
pool = Pool(a.arm, sp["tr"], sp["va"], sp["te"], sp["sid"], 1.0, dev, fh, kpool=sp.get("kpool"))
res = {}
for mp in a.models.split(","):
    g = torch.Generator().manual_seed(4242)
    U = [(torch.randn(w, 128, generator=g) * 0.02).to(dev) for w in pool.widths]
    m = Ranker(U, 128, pool.tr.mem.shape[2], 2, 4, 0.3, pool.nb, 0).to(dev)
    m.load_state_dict(torch.load(mp, map_location=dev)["main"]); m.eval()
    Sv, St = scores(pool, m, "va", pool.va, dev), scores(pool, m, "te", pool.te, dev)
    val = {bb: hits(Sv, pool.va, bb, dev).mean() for bb in GRID}
    best = max(GRID, key=lambda bb: val[bb])
    h0, hb = hits(St, pool.te, (0.0,) * 4, dev), hits(St, pool.te, best, dev)
    log(f"  [{a.arm}] {os.path.basename(mp)}: VAL none {val[(0.0,)*4]:.4f} -> best {val[best]:.4f} at bias(pv,fav,cart,buy)={best}   TEST {h0.mean():.4f} -> {hb.mean():.4f}", fh)
    o = np.argsort(pool.uid["te"].numpy()); res[mp] = {"uid": pool.uid["te"].numpy()[o], "h0": h0[o], "hb": hb[o], "best": best}
torch.save({"res": res, "rep": NIP.gt_is_repeat(pool.te)[np.argsort(pool.uid["te"].numpy())],
            "has": pool.te.has_gt.numpy()[np.argsort(pool.uid["te"].numpy())],
            "gb": pool.te.gt_b.numpy()[np.argsort(pool.uid["te"].numpy())]}, a.out.replace(".txt", ".pt"))
fh.close()
