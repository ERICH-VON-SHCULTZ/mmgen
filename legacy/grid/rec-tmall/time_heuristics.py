"""Go/no-go for timestamps: does TIME since an item's last view rank the returning item better than list
position did (~0.59 item-level top-10 hit on the 1,020 forced-only repeats)? CPU, TEST dump + ts sidecar."""
import numpy as np, torch
H = 40
P = "/scratch/yw8866/rec-tmall/force_exp/pools/fh_test.pt"
c = torch.load(P, map_location="cpu", weights_only=False, mmap=True)["cols"]
z = np.load(P + ".ts.npz")
uid = c["user_id"].long().numpy(); assert np.array_equal(uid, z["user_id"])
o = np.argsort(uid)
h = c["hist_ids"].long().numpy()[o][:, :H * 4].reshape(-1, H, 4); v = h[:, :, 0] >= 0
key = np.where(v, (h[:, :, 1] * 4096 + h[:, :, 2]) * 4096 + h[:, :, 3], -1)
gkey = (c["gt_f"].long().numpy()[o] * 4096 + c["gt_t"].long().numpy()[o]) * 4096 + c["gt_i"].long().numpy()[o]
gb = c["gt_b"].long().numpy()[o]; hb = h[:, :, 0]
tsh, tst, ok = z["ts_hist"][o], z["ts_tgt"][o], z["ok"][o]
print(f"timestamp alignment on TEST: {ok.mean():.4f}")
rep = (key == gkey[:, None]).any(1)
M = torch.load("/scratch/yw8866/rec-tmall/force_exp/cache/MFD__wnew1__seed0.pt", map_location="cpu", weights_only=False); mc = M["has_gt"][np.argsort(M["uid"])].astype(bool)
Cc = torch.load("/scratch/yw8866/rec-tmall/nip_exp/cache/CH__wnew1__seed0.pt", map_location="cpu", weights_only=False); cc = Cc["has_gt"][np.argsort(Cc["uid"])].astype(bool)
idx = np.nonzero(rep & (gb == 0) & ok)[0]
K, B, V, T, TT = key[idx], hb[idx], v[idx], tsh[idx].astype(float), tst[idx].astype(float)
n = V.sum(1, keepdims=True).clip(min=1); pos = np.arange(H)[None]
rec = np.where(V, (pos + 1) / n, 0)
same = (K[:, :, None] == K[:, None, :]) & V[:, :, None] & V[:, None, :]
cnt = same.sum(2); later = np.triu(np.ones((H, H), bool), 1)[None]
is_last = V & ~(same & later).any(2)
gap = np.where(V, TT[:, None] - T, np.inf)                      # seconds between the item's view and the request
lgap = np.log1p(np.clip(gap, 0, None))
last_gap = TT - np.where(V, T, -np.inf).max(1)                  # time since the user's last action (event-level)
in_sess = (gap < 1800)                                          # viewed within 30 min of the request
cnt_sess = (same & in_sess[:, None, :]).sum(2)
cnt_day = (same & (gap < 86400)[:, None, :]).sum(2)
conv = np.stack([(same & (B[:, None, :] == k)).any(2) for k in (1, 2, 3)], -1).any(-1)
last_act = np.where(V, T, -np.inf).max(1)                       # the user's last history action (no target time)
lgap2 = np.log1p(np.clip(np.where(V, last_act[:, None] - T, np.inf), 0, None))
cnt_sess2 = (same & (np.where(V, last_act[:, None] - T, np.inf) < 1800)[:, None, :]).sum(2)
tgt = np.array([np.nonzero(K[e] == gkey[idx][e])[0][-1] for e in range(len(idx))])
def hit(score, k=10):
    s = np.where(is_last, score, -np.inf); g = s[np.arange(len(idx)), tgt]
    return (s > g[:, None]).sum(1) < k
pops = {"BOTH": mc[idx] & cc[idx], "forced-only": mc[idx] & ~cc[idx], "all pv repeats": np.ones(len(idx), bool)}
rules = {"POSITION: freq + recency + conv (best old)": np.log1p(cnt) + 2 * rec + 0.5 * conv,
         "TIME: most recent in seconds": -lgap,
         "TIME: views in session, then recency": cnt_sess * 10 - lgap,
         "TIME: views today + recency": np.log1p(cnt_day) * 3 - lgap,
         "TIME: -log gap + log count + conv": -lgap + np.log1p(cnt) + 0.5 * conv,
         "TIME+POS: -log gap + 2 rec + log count + conv": -lgap + 2 * rec + np.log1p(cnt) + 0.5 * conv,
         "NO-TARGET-TIME: -log gap to last action + ...": -lgap2 + 2 * rec + np.log1p(cnt) + 0.5 * conv,
         "NO-TARGET-TIME: views in last session, then ...": cnt_sess2 * 10 - lgap2 + np.log1p(cnt)}
print(f"{'item-level top-10 hit (pv repeats)':<46}" + "".join(f"{p:>16}" for p in pops))
for nm, sc in rules.items():
    hh = hit(sc); print(f"{nm:<46}" + "".join(f"{hh[q].mean():>16.4f}" for q in pops.values()))
print(f"{'reranker (forced pool, seed 0)':<46}{0.983:>16.4f}{0.587:>16.4f}{0.954:>16.4f}")
print("n: " + ", ".join(f"{p}={int(q.sum()):,}" for p, q in pops.items()))
fo = pops["forced-only"]
print(f"forced-only: median seconds from last user action to request {np.median(last_gap[fo]):,.0f}; "
      f"GT item viewed within 30 min of request in {in_sess[np.arange(len(idx)), tgt][fo].mean():.3f}")
