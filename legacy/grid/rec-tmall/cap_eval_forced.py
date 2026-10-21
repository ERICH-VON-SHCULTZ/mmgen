"""Behavior-hedge cap: keep at most K behaviors per item in the top-10 (K=1 is the failed dedup D).
Saved seed-0 rerankers for both arms; K chosen on VAL, reported on TEST, identical policy family both arms."""
import argparse, json, os, sys
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))
import numpy as np, torch
from coverage_expansion import Pool, log
from sweep2_rerank import Ranker, make_scal
import nip_rerank as NIP
KS = (1, 2, 3, 99)                      # 99 = no cap (= P0)

def capk_hits(S, key, valid, gt, has, K, M=96):
    top = S.topk(min(M, S.shape[1]), -1); k = key.gather(1, top.indices); Mm = k.shape[1]
    earlier = torch.tril(torch.ones(Mm, Mm, dtype=torch.bool, device=S.device), -1)
    nprev = ((k.unsqueeze(2) == k.unsqueeze(1)) & earlier.unsqueeze(0)).sum(-1)      # earlier copies of this item
    keep = (nprev < K) & (top.values > float("-inf"))
    sel = keep & (keep.long().cumsum(1) <= 10)
    ch = torch.zeros_like(valid); ch.scatter_(1, top.indices, sel); ch &= valid
    hit = has & ch.gather(1, gt.clamp(min=0)[:, None]).squeeze(1)
    if K == 1:                                   # ITEM-level hit under cap1 = the ceiling with correct behavior
        gkey = key.gather(1, gt.clamp(min=0)[:, None])
        item_hit = has & ((key == gkey) & ch).any(1)
        return hit, item_hit
    return hit

def run(arm, sp, model_path, other, dev, fh):
    pool = Pool(arm, sp["tr"], sp["va"], sp["te"], sp["sid"], 1.0, dev, fh, kpool=sp.get("kpool"))
    g = torch.Generator().manual_seed(4242)
    U = [(torch.randn(w, 128, generator=g) * 0.02).to(dev) for w in pool.widths]
    m = Ranker(U, 128, pool.tr.mem.shape[2], 2, 4, 0.3, pool.nb, 0).to(dev)
    m.load_state_dict(torch.load(model_path, map_location=dev)["main"]); m.eval()
    out = {}
    for split, d in (("va", pool.va), ("te", pool.te)):
        H = {K: np.zeros(d.N, bool) for K in KS}
        with torch.no_grad():
            for s in range(0, d.N, 16):
                idx = torch.arange(s, min(s + 16, d.N))
                f, t, i, scal = make_scal(pool, split, m, d, idx, dev, ())
                valid = d.valid[idx].to(dev)
                S = m(f, t, i, scal, d.base[idx].float().to(dev), valid, d.mem[idx].float().to(dev), d.mem_pad[idx].to(dev))
                key = (f * pool.k1 + t) * pool.k2 + i; gt = d.gt_idx[idx].to(dev)
                for K in KS:
                    r = capk_hits(S, key, valid, gt, gt >= 0, K)
                    if K == 1:
                        r, ih = r; H.setdefault("item", np.zeros(d.N, bool))[idx.numpy()] = ih.cpu().numpy()
                    H[K][idx.numpy()] = r.cpu().numpy()
        out[split] = H
    o = np.argsort(pool.uid["te"].numpy())
    return out, pool.uid["te"].numpy()[o], {K: out["te"][K][o] for K in list(KS) + ["item"]}, NIP.gt_is_repeat(pool.te)[o], pool.te.has_gt.numpy()[o], pool.te.gt_b.numpy()[o]

ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); sp = json.load(open(a.spec)); fh = open(a.out, "w"); dev = "cuda"
R = {}
for arm, mp in (("MFD", "/scratch/yw8866/rec-tmall/force_exp/cache/MFD__wnew1__seed0.model.pt"), ("CH", "/scratch/yw8866/rec-tmall/nip_mix/topk_ch_seed0.model.pt")):
    R[arm] = run(arm, sp[arm], mp, None, dev, fh)
    va = {K: R[arm][0]["va"][K].mean() for K in KS}; te = {K: R[arm][0]["te"][K].mean() for K in KS}
    log(f"  [{arm}] VAL " + " ".join(f"cap{K}={va[K]:.4f}" for K in KS) + "   TEST " + " ".join(f"cap{K}={te[K]:.4f}" for K in KS), fh)
assert np.array_equal(R["MFD"][1], R["CH"][1])
Kstar = {arm: max(KS, key=lambda K: R[arm][0]["va"][K].mean()) for arm in R}
log(f"\n  VAL-chosen cap: MFD {Kstar['MFD']}, chain {Kstar['CH']}", fh)
for lab, km, kc in (("no cap (today)", 99, 99), ("VAL-chosen", Kstar["MFD"], Kstar["CH"]), ("cap2 both", 2, 2),
                    ("CEILING: 10 distinct items, correct behavior", "item", "item")):
    hm, hc = R["MFD"][2][km], R["CH"][2][kc]
    log(f"  {lab:<16} MFD {hm.mean():.4f}  chain {hc.mean():.4f}  lead {hm.mean()-hc.mean():+.5f}", fh)
rep = R["MFD"][3]; pv = R["MFD"][5] == 0; mc = R["MFD"][4].astype(bool); cc = R["CH"][4].astype(bool)
for nm, q in (("MFD-ONLY repeat", mc & ~cc & rep & pv), ("BOTH repeat", mc & cc & rep & pv), ("ALL pv repeat", rep & pv), ("conversions", R["MFD"][5] > 0)):
    log(f"  {nm:<16} n={int(q.sum()):>6,}  MFD " + " ".join(f"cap{K}={R['MFD'][2][K][q].mean():.4f}" for K in KS)
        + f" item={R['MFD'][2]['item'][q].mean():.4f}   chain " + " ".join(f"cap{K}={R['CH'][2][K][q].mean():.4f}" for K in KS)
        + f" item={R['CH'][2]['item'][q].mean():.4f}", fh)
fh.close()
