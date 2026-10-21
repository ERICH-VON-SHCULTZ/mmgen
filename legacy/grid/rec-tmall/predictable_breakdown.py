"""Coverage / P(rank<10 | covered) / recall@10 by event group, best MFD vs chain, same setting.

Setting (both arms): 808 decoder states; reranker R2_rw1, capped 36,864 training events; TEST 83,005 events;
5 seeds (seedvar/cache). MFD = P3 wide pool, K_pool=2000; chain = K_pool=500.

Groups (t3 = behavior AND item must match):
  pv repeat   page-view whose GT item tuple already appears in the user's history
  pv new      page-view whose GT item is not in the history
  fav / cart / buy   conversions (also split repeat / new)
  PREDICTABLE = pv repeat + all conversions;  NEW = pv new
"Repeat" is defined in each arm's own SID space (its dump's history). The primary table partitions events
with MFD's definition so both arms are scored on the SAME event sets; a check table uses chain's own.
"""

import numpy as np
import torch

C = "/scratch/yw8866/rec-tmall/seedvar/cache"
DUMPS = {"MFD": "/scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt",
         "CH": "/scratch/yw8866/rec-tmall/matched_rerank/chain808_test_dump.pt"}
SEEDS = range(5)


def repeat_flag(path):
    c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["cols"]
    uid = c["user_id"].long().numpy(); o = np.argsort(uid)
    gf, gt, gi = (c[k].long().numpy()[o] for k in ("gt_f", "gt_t", "gt_i"))
    h = c["hist_ids"].long().numpy()[o]; H = h.shape[1] // 4; h = h[:, :H * 4].reshape(-1, H, 4)
    v = h[:, :, 0] >= 0
    rep = (v & (h[:, :, 1] == gf[:, None]) & (h[:, :, 2] == gt[:, None]) & (h[:, :, 3] == gi[:, None])).any(1)
    return uid[o], rep, c["gt_b"].long().numpy()[o]


def load(arm, sd):
    r = torch.load(f"{C}/{arm}__seed{sd}.pt", map_location="cpu", weights_only=False)
    o = np.argsort(r["uid"])
    return {k: r[k][o] for k in ("uid", "has_gt", "hit_after", "hit_before")}


uidM, repM, gb = repeat_flag(DUMPS["MFD"])
uidC, repC, gbC = repeat_flag(DUMPS["CH"])
assert np.array_equal(uidM, uidC) and np.array_equal(gb, gbC)
R = {a: [load(a, s) for s in SEEDS] for a in ("MFD", "CH")}
for a in R:
    for r in R[a]:
        assert np.array_equal(r["uid"], uidM)
N = len(uidM)
assert N == 83005


def groups(rep):
    pv = gb == 0
    g = {"pv repeat": pv & rep, "pv new": pv & ~rep}
    for b, nm in ((1, "fav"), (2, "cart"), (3, "buy")):
        g[nm] = gb == b
    g["conversions (fav+cart+buy)"] = gb > 0
    g["  conv repeat"] = (gb > 0) & rep
    g["  conv new"] = (gb > 0) & ~rep
    g["PREDICTABLE (pv repeat + conv)"] = (pv & rep) | (gb > 0)
    g["NEW (pv new)"] = pv & ~rep
    g["ALL"] = np.ones(N, bool)
    return g


def stats(arm, m):
    cov = R[arm][0]["has_gt"].astype(bool)            # coverage is seed-invariant (pool is fixed)
    covered = cov & m
    hits = np.array([(r["hit_after"] > 0)[m].sum() for r in R[arm]], float)
    base = np.array([(r["hit_before"] > 0)[m].sum() for r in R[arm]], float)
    n = m.sum(); nc = covered.sum()
    return dict(n=n, cov=nc / n, pc=hits / max(nc, 1), rec=hits / n, base=base / n, hits=hits)


def table(rep, title):
    out = [f"\n=== {title} ===",
           f"{'group':<32}{'n':>8} | {'MFD cov':>8}{'P(r<10|c)':>18}{'recall@10':>18}{'base rec':>9} | "
           f"{'chain cov':>9}{'P(r<10|c)':>18}{'recall@10':>18}{'base rec':>9} | {'lead contrib':>13}"]
    for nm, m in groups(rep).items():
        a, b = stats("MFD", m), stats("CH", m)
        f = lambda x: f"{x.mean():.4f}±{x.std(ddof=1):.4f}"
        contrib = (a["hits"] - b["hits"]) / N
        out.append(f"{nm:<32}{a['n']:>8,} | {a['cov']:>8.4f}{f(a['pc']):>18}{f(a['rec']):>18}{a['base'].mean():>9.4f} | "
                   f"{b['cov']:>9.4f}{f(b['pc']):>18}{f(b['rec']):>18}{b['base'].mean():>9.4f} | "
                   f"{contrib.mean():>+8.5f}±{contrib.std(ddof=1):.5f}")
    return out


lines = ["Setting: 808 decoder states; R2_rw1 reranker, capped; TEST 83,005 events; mean ± std over 5 seeds.",
         "MFD = P3 wide pool K_pool=2000; chain = K_pool=500. cov = GT in pool; P(r<10|c) = recall among covered;",
         "base rec = recall@10 of the generator's own score before reranking; lead contrib = (MFD hits - chain hits)/83,005."]
lines += table(repM, "PRIMARY: events partitioned by MFD's repeat definition (same event sets for both arms)")
lines += table(repC, "CHECK: events partitioned by chain's own repeat definition")
lines.append(f"\nrepeat-flag agreement between the two SID spaces: {(repM == repC).mean():.4f} "
             f"(MFD marks {repM.mean():.4f} of events as repeats, chain {repC.mean():.4f})")
txt = "\n".join(lines)
print(txt)
open("/scratch/yw8866/rec-tmall/predictable_breakdown.txt", "w").write(txt + "\n")
