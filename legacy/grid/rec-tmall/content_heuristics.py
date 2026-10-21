"""Go/no-go for RAW CONTENT (SigLIP2 text + image, 1152-d each) on the hard repeats: does similarity between a
past item and what the user viewed most recently identify which past item returns? Item-level top-10 hit on
TEST pv repeats; slot identity = MFD SID tuple (as before), content per slot via the recovered item id."""
import pickle, numpy as np, torch
H = 40
P = "/scratch/yw8866/rec-tmall/force_exp/pools/fh_test.pt"
def id_map(p):
    m = pickle.load(open(p, "rb"))
    return {int(k): int(v) for k, v in m.items()} if isinstance(m, dict) else {int(i): r for r, i in enumerate(m)}
def mm(path, n):
    tot = __import__("os").path.getsize(path) // 4; d = tot // n
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, d))
E = {}
for nm, base in (("text", "/scratch/yw8866/rec-tmall/text_embedding/siglip2-so400m-patch14-384"),
                 ("image", "/scratch/yw8866/rec-tmall/image_embedding/siglip2-so400m-patch14-384")):
    ids = id_map(base + "_ids.pkl"); E[nm] = (ids, mm(base + ".npy", len(ids)))
c = torch.load(P, map_location="cpu", weights_only=False, mmap=True)["cols"]
z = np.load(P + ".ts.npz"); uid = c["user_id"].long().numpy(); assert np.array_equal(uid, z["user_id"])
o = np.argsort(uid)
h = c["hist_ids"].long().numpy()[o][:, :H * 4].reshape(-1, H, 4); v = h[:, :, 0] >= 0
key = np.where(v, (h[:, :, 1] * 4096 + h[:, :, 2]) * 4096 + h[:, :, 3], -1)
gkey = (c["gt_f"].long().numpy()[o] * 4096 + c["gt_t"].long().numpy()[o]) * 4096 + c["gt_i"].long().numpy()[o]
gb = c["gt_b"].long().numpy()[o]; hb = h[:, :, 0]; ith = z["item_hist"][o]; ok = z["ok"][o]
rep = (key == gkey[:, None]).any(1)
M = torch.load("/scratch/yw8866/rec-tmall/force_exp/cache/MFD__wnew1__seed0.pt", map_location="cpu", weights_only=False); mc = M["has_gt"][np.argsort(M["uid"])].astype(bool)
Cc = torch.load("/scratch/yw8866/rec-tmall/nip_exp/cache/CH__wnew1__seed0.pt", map_location="cpu", weights_only=False); cc = Cc["has_gt"][np.argsort(Cc["uid"])].astype(bool)
idx = np.nonzero(rep & (gb == 0) & ok)[0]
K, B, V, IT = key[idx], hb[idx], v[idx], ith[idx]
def emb(nm):
    ids, X = E[nm]; out = np.zeros(IT.shape + (X.shape[1],), np.float32); have = np.zeros(IT.shape, bool)
    rows = np.array([[ids.get(int(x), -1) for x in r] for r in IT])
    m = (rows >= 0) & V; out[m] = X[rows[m]]; have = m
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-9)
    return out, have
Tt, ht_ = emb("text"); Ti, hi_ = emb("image")
n = V.sum(1).clip(min=1); pos = np.arange(H)[None]
rec = np.where(V, (pos + 1) / n[:, None], 0)
same = (K[:, :, None] == K[:, None, :]) & V[:, :, None] & V[:, None, :]
cnt = same.sum(2); later = np.triu(np.ones((H, H), bool), 1)[None]; is_last = V & ~(same & later).any(2)
conv = np.stack([(same & (B[:, None, :] == k)).any(2) for k in (1, 2, 3)], -1).any(-1)
def ctx(T, have, k):
    """mean embedding of the user's last k history items"""
    w = V & have & (pos >= (n[:, None] - k)); c_ = (T * w[..., None]).sum(1); return c_ / np.linalg.norm(c_, axis=-1, keepdims=True).clip(min=1e-9)
sims = {}
for k in (1, 3, 5):
    sims[f"text k={k}"] = np.einsum("ehd,ed->eh", Tt, ctx(Tt, ht_, k))
    sims[f"image k={k}"] = np.einsum("ehd,ed->eh", Ti, ctx(Ti, hi_, k))
tgt = np.array([np.nonzero(K[e] == gkey[idx][e])[0][-1] for e in range(len(idx))])
def hit(score, kk=10):
    s = np.where(is_last, score, -np.inf); g = s[np.arange(len(idx)), tgt]
    return (s > g[:, None]).sum(1) < kk
pos_rule = np.log1p(cnt) + 2 * rec + 0.5 * conv
pops = {"BOTH": mc[idx] & cc[idx], "forced-only": mc[idx] & ~cc[idx], "all pv repeats": np.ones(len(idx), bool)}
rules = {"POSITION (best old)": pos_rule}
for nm, s in sims.items():
    rules[f"CONTENT {nm}"] = s
    rules[f"POSITION + 2*CONTENT {nm}"] = pos_rule + 2 * s
rules["POSITION + text k=3 + image k=3"] = pos_rule + 2 * sims["text k=3"] + 2 * sims["image k=3"]
print(f"content coverage: text {ht_[V].mean():.4f}, image {hi_[V].mean():.4f} of history slots")
print(f"{'item-level top-10 hit (pv repeats)':<44}" + "".join(f"{p:>16}" for p in pops))
for nm, sc in rules.items():
    hh = hit(sc); print(f"{nm:<44}" + "".join(f"{hh[q].mean():>16.4f}" for q in pops.values()))
print(f"{'reranker (forced pool, seed 0)':<44}{0.983:>16.4f}{0.587:>16.4f}{0.954:>16.4f}")
