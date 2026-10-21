"""How predictable is 'the next page-view is a REPEAT' from the history alone? (test split, MFD SID space)
Simple per-event features; reports each feature's AUC and a logistic-regression AUC (fit on half, scored on
the other half). A mixture can only allocate slots per event as well as this gate separates the two cases."""
import numpy as np, torch
c = torch.load('/scratch/yw8866/rec-tmall/coverage_exp/pools/p3k_test.pt', map_location='cpu', weights_only=False, mmap=True)['cols']
gb = c['gt_b'].long().numpy(); gf, gt, gi = (c[k].long().numpy() for k in ('gt_f', 'gt_t', 'gt_i'))
h = c['hist_ids'].long().numpy(); H = h.shape[1] // 4; h = h[:, :H * 4].reshape(-1, H, 4); v = h[:, :, 0] >= 0
rep = (v & (h[:, :, 1] == gf[:, None]) & (h[:, :, 2] == gt[:, None]) & (h[:, :, 3] == gi[:, None])).any(1)
key = (h[:, :, 1] * 4096 + h[:, :, 2]) * 4096 + h[:, :, 3]; key = np.where(v, key, -1 - np.arange(H)[None, :])
n = v.sum(1)
same = (key[:, :, None] == key[:, None, :]) & v[:, :, None] & v[:, None, :]
cnt = same.sum(2)                                            # occurrences of each history item within the history
distinct = np.array([len(set(k[m].tolist())) for k, m in zip(key, v)])
# 'self-repeat rate': how often an item in this history re-occurs LATER in the same history (the user's habit)
later = np.triu(np.ones((H, H), bool), 1)[None]
reoccur = ((same & later).any(2) & v).sum(1) / np.maximum(n - 1, 1)
last_rep = np.array([ (cnt[e, np.nonzero(v[e])[0].max()] > 1) if v[e].any() else False for e in range(len(v))])
nbeh = np.stack([((h[:, :, 0] == b) & v).sum(1) / np.maximum(n, 1) for b in range(4)], 1)
F = {'history length': n, 'distinct items / length': distinct / np.maximum(n, 1), 'self-repeat rate': reoccur,
     'max item count': cnt.max(1), 'last item repeated': last_rep.astype(float),
     'share pv': nbeh[:, 0], 'share cart': nbeh[:, 2], 'share buy': nbeh[:, 3]}
pv = gb == 0; y = rep[pv]
def auc(x, y):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    # average ranks for ties
    xs = x[o]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = y.sum(); q = len(y) - p
    return (r[y].sum() - p * (p + 1) / 2) / (p * q)
print(f'pv events {pv.sum():,}; next item is a repeat in {y.mean():.4f}')
X = np.stack([f[pv].astype(float) for f in F.values()], 1)
for (nm, f) in F.items():
    a = auc(f[pv].astype(float), y); print(f'  {nm:<26} AUC {max(a, 1-a):.4f} ({"+" if a >= .5 else "-"})')
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); idx = rng.permutation(len(y)); a, b = idx[:len(y)//2], idx[len(y)//2:]
mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
lr = LogisticRegression(max_iter=2000).fit((X[a]-mu)/sd, y[a]); p = lr.predict_proba((X[b]-mu)/sd)[:, 1]
print(f'  logistic regression (all features) held-out AUC {auc(p, y[b]):.4f}')
for q in (0.1, 0.3, 0.5):
    thr = np.quantile(p, 1 - q); sel = p >= thr
    print(f'    top-{int(q*100)}% most-likely-repeat events: repeat rate {y[b][sel].mean():.3f};  bottom {int((1-q)*100)}%: {y[b][~sel].mean():.3f}')
