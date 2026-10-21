"""Revisit model: WHICH past item comes back next? Trained on the full training split (millions of repeat
events, vs ~37k for the rerankers). Structural slot features only (no SID/item embeddings), so it applies to
both arms identically, with each arm's own history-tuple identity at test time.

Per history slot (oldest->newest, up to H=40): recency position, behavior, #occurrences of the item in the
window, first-occurrence recency, is-last-occurrence, the item's behavior set (4 flags), and event-level
(#distinct, length). A small transformer scores slots; softmax over LAST-occurrence slots; target = the slot
of the target item's last occurrence. Go/no-go: item-level top-10 hit on the TEST dump's repeat events
versus the reranker's (BOTH repeats 0.983, forced-only repeats 0.587, forced pool, seed 0)."""
import argparse, glob, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
H = 40; NF = 11

def read_split(folder, max_users):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    files = sorted(glob.glob(os.path.join(folder, "*.tfrecord.gz")))
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    spec = {"sequence_data": tf.io.VarLenFeature(tf.int64), "behavior_data": tf.io.VarLenFeature(tf.int64)}
    out = []
    for rec in ds:
        e = tf.io.parse_single_example(rec, spec)
        out.append((tf.sparse.to_dense(e["sequence_data"]).numpy(), tf.sparse.to_dense(e["behavior_data"]).numpy()))
        if len(out) >= max_users:
            break
    return out

def windows(seqs, max_events, rng, all_positions=True):
    """-> hist ids (E,H) with -1 pad (oldest->newest, left-aligned), hist beh (E,H), label slot (E,)"""
    HI, HB, LB = [], [], []
    for seq, beh in seqs:
        L = len(seq)
        ts = range(1, L) if all_positions else [L - 1]
        for t in ts:
            lo = max(0, t - H); hid = seq[lo:t]; tgt = seq[t]
            occ = np.nonzero(hid == tgt)[0]
            if occ.size == 0:
                continue
            n = len(hid); a = np.full(H, -1, np.int64); b = np.full(H, -1, np.int64)
            a[:n] = hid; b[:n] = beh[lo:t]
            HI.append(a); HB.append(b); LB.append(occ[-1])
        if len(LB) >= max_events:
            break
    return np.stack(HI), np.stack(HB), np.array(LB)

def slot_feats(hid, hb):
    """hid (E,H) identity codes (-1 pad), hb (E,H) behaviors -> (E,H,NF) float, last-occ mask (E,H)"""
    E = hid.shape[0]; v = hid >= 0
    n = v.sum(1, keepdims=True).clip(min=1)
    pos = np.arange(H)[None].repeat(E, 0)
    rec = np.where(v, (pos + 1) / n, 0.0)
    same = (hid[:, :, None] == hid[:, None, :]) & v[:, :, None] & v[:, None, :]
    cnt = same.sum(2)
    later = np.triu(np.ones((H, H), bool), 1)[None]
    is_last = v & ~(same & later).any(2)
    first = np.where(same, rec[:, None, :], 2.0).min(2)
    bset = np.stack([(same & (hb[:, None, :] == k)).any(2) for k in range(4)], -1)
    ndist = (is_last.sum(1, keepdims=True) / H).repeat(H, 1)
    f = np.concatenate([rec[..., None], np.log1p(cnt)[..., None], np.where(v, first, 0)[..., None],
                        is_last[..., None], bset, ndist[..., None], (n / H).repeat(H, 1)[..., None],
                        (v * (1 - rec))[..., None]], -1)
    return f.astype(np.float32), is_last

class Revisit(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.beh = nn.Embedding(5, d); self.pos = nn.Embedding(H, d); self.num = nn.Linear(NF, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers); self.head = nn.Linear(d, 1)
    def forward(self, feats, hb, v):
        x = self.num(feats) + self.beh(hb.clamp(min=-1) + 1) + self.pos(torch.arange(H, device=feats.device))[None]
        x = self.tf(x, src_key_padding_mask=~v)
        return self.head(x).squeeze(-1)

def batches(hid, hb, lab, bs, dev, shuffle, rng):
    idx = rng.permutation(len(lab)) if shuffle else np.arange(len(lab))
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]; f, last = slot_feats(hid[j], hb[j])
        yield (torch.from_numpy(f).to(dev), torch.from_numpy(hb[j]).to(dev), torch.from_numpy(hid[j] >= 0).to(dev),
               torch.from_numpy(last).to(dev), None if lab is None else torch.from_numpy(lab[j]).to(dev))

def topk_hit(model, hid, hb, lab, dev, k=10, bs=4096):
    model.eval(); hits = []
    with torch.no_grad():
        for f, b, v, last, y in batches(hid, hb, lab, bs, dev, False, None):
            sc = model(f, b, v).masked_fill(~last, float("-inf"))
            hits.append(((sc > sc.gather(1, y[:, None])).sum(1) < k).cpu())
    return float(torch.cat(hits).float().mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/scratch/yw8866/rec-tmall/tfrecords_mb_userdrop")
    ap.add_argument("--max_users", type=int, default=10**9); ap.add_argument("--max_train", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--out", required=True)
    ap.add_argument("--test_dump", required=True); ap.add_argument("--mfd_res", required=True); ap.add_argument("--ch_res", required=True)
    a = ap.parse_args(); dev = "cuda"; rng = np.random.default_rng(0); t0 = time.time()
    lines = []
    def log(m): print(m, flush=True); lines.append(m); open(a.out, "w").write("\n".join(lines) + "\n")
    tr = read_split(os.path.join(a.data, "training"), a.max_users)
    hid, hb, lab = windows(tr, a.max_train, rng); del tr
    va = read_split(os.path.join(a.data, "evaluation"), 30000)
    vh, vb, vl = windows(va, 200000, rng, all_positions=False); del va
    log(f"train repeat events {len(lab):,}  val repeat events (last position) {len(vl):,}  ({time.time()-t0:.0f}s)")
    model = Revisit().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -1, None
    for ep in range(1, a.epochs + 1):
        model.train()
        for f, b, v, last, y in batches(hid, hb, lab, 1024, dev, True, rng):
            sc = model(f, b, v).masked_fill(~last, float("-inf"))
            loss = F.cross_entropy(sc, y); opt.zero_grad(); loss.backward(); opt.step()
        h = topk_hit(model, vh, vb, vl, dev)
        # simple baselines on the same VAL events: most-recent-first, and most-frequent-then-recent
        log(f"ep{ep} VAL item-level top-10 hit {h:.4f}  ({time.time()-t0:.0f}s)")
        if h > best: best, state = h, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    fr, _ = slot_feats(vh, vb)
    rec_score = np.where(slot_feats(vh, vb)[1], fr[..., 0], -1.0)
    rank_rec = (rec_score > np.take_along_axis(rec_score, vl[:, None], 1)).sum(1)
    log(f"VAL baseline 'most recent distinct item first' top-10 hit {(rank_rec < 10).mean():.4f}   revisit model {best:.4f}")
    torch.save(state, a.out.replace(".txt", ".pt"))
    # ---- TEST dump: the reranker's events, identity = MFD SID tuple ----
    c = torch.load(a.test_dump, map_location="cpu", weights_only=False, mmap=True)["cols"]
    uid = c["user_id"].long().numpy(); o = np.argsort(uid)
    h = c["hist_ids"].long().numpy()[o]; h = h[:, :H * 4].reshape(-1, H, 4)
    key = np.where(h[:, :, 0] >= 0, (h[:, :, 1] * 4096 + h[:, :, 2]) * 4096 + h[:, :, 3], -1)
    gkey = (c["gt_f"].long().numpy()[o] * 4096 + c["gt_t"].long().numpy()[o]) * 4096 + c["gt_i"].long().numpy()[o]
    gb = c["gt_b"].long().numpy()[o]; hbt = h[:, :, 0]
    rep = (key == gkey[:, None]).any(1)
    idx = np.nonzero(rep)[0]
    lastslot = np.array([np.nonzero(key[e] == gkey[e])[0][-1] for e in idx])
    M = torch.load(a.mfd_res, map_location="cpu", weights_only=False); mc = M["has_gt"][np.argsort(M["uid"])].astype(bool)
    Cc = torch.load(a.ch_res, map_location="cpu", weights_only=False); cc = Cc["has_gt"][np.argsort(Cc["uid"])].astype(bool)
    rr = {}
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            j = idx[s:s + 4096]; f, last = slot_feats(key[j], hbt[j])
            sc = model(torch.from_numpy(f).to(dev), torch.from_numpy(hbt[j]).to(dev), torch.from_numpy(key[j] >= 0).to(dev))
            sc = sc.masked_fill(~torch.from_numpy(last).to(dev), float("-inf"))
            y = torch.from_numpy(lastslot[s:s + 4096]).to(dev)
            for e, r in zip(j, (sc > sc.gather(1, y[:, None])).sum(1).cpu().numpy()):
                rr[e] = r
    r = np.array([rr[e] for e in idx]); pv = gb[idx] == 0
    for nm, q in (("BOTH pv repeats", mc[idx] & cc[idx] & pv), ("forced-only pv repeats (MFD pool, not chain)", mc[idx] & ~cc[idx] & pv),
                  ("all pv repeats", pv), ("conversions (repeat)", ~pv)):
        log(f"TEST {nm:<46} n={int(q.sum()):>6,}  revisit item-level top-10 hit {(r[q] < 10).mean():.4f}   top-5 {(r[q] < 5).mean():.4f}")
    log("reference (reranker, forced pool seed 0, 10 distinct items): BOTH 0.983, forced-only 0.587, all pv repeats 0.954")

if __name__ == "__main__":
    main()
