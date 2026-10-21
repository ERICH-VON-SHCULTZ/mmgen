"""Recover per-event timestamps for the reranker dumps from the filtered raw logs (the TFRecords drop them).
The builder sorts every user's rows by (user_id, timestamp) (stable in file order) and writes test/eval users
unaltered, and each dump event is (last <=40 items -> next item) for a user. So an event's timestamps are the
user's last 41 sorted timestamps. Alignment is VERIFIED with the behavior sequence (hist_b + gt_b), per event.
Output per dump: <dump>.ts.npz with, in dump row order: user_id, ts_hist (N,40) seconds (-1 pad, left-aligned
like hist_ids), ts_tgt (N,), ok (N,) alignment flag."""
import argparse, datetime as dt, os, time
import numpy as np, torch
SEP = "\x01"
BMAP = {"click": 0, "collect": 1, "cart": 2, "alipay": 3}
LOGS = ["tianchi_2014002_rec_tmall_log_parta.txt", "tianchi_2014002_rec_tmall_log_partb.txt", "tianchi_2014002_rec_tmall_log_partc.txt"]
EPOCH = dt.datetime(2013, 1, 1)

def secs(s):
    return int((dt.datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19])) - EPOCH).total_seconds())

ap = argparse.ArgumentParser(); ap.add_argument("--dumps", required=True); ap.add_argument("--logdir", default="/scratch/yw8866/rec-tmall/valid_data")
a = ap.parse_args(); t0 = time.time()
dumps = a.dumps.split(","); cols = {}
need = set()
for p in dumps:
    c = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["cols"]
    cols[p] = {"uid": c["user_id"].long().numpy(), "hb": c["hist_ids"].long().numpy()[:, 0::4][:, :40], "gb": c["gt_b"].long().numpy()}
    need |= set(cols[p]["uid"].tolist())
print(f"users needed: {len(need):,}  ({time.time()-t0:.0f}s)", flush=True)
rows = {}; order = 0
for fn in LOGS:
    with open(os.path.join(a.logdir, fn), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(SEP, 3)
            if len(parts) < 4:
                continue
            u = parts[1].strip()
            try:
                uid = int(u[1:] if u.startswith("u") else u)
            except ValueError:
                continue
            if uid not in need:
                continue
            b = BMAP.get(parts[2].strip(), -1)
            if b < 0:
                continue
            try:
                ts = secs(parts[3].strip())
            except (ValueError, IndexError):
                continue
            try:
                iid = int(parts[0].strip())
            except ValueError:
                continue
            rows.setdefault(uid, []).append((ts, order, b, iid)); order += 1
    print(f"  {fn}: users with rows {len(rows):,}  ({time.time()-t0:.0f}s)", flush=True)
for p in dumps:
    C = cols[p]; N = len(C["uid"]); tsh = np.full((N, 40), -1, np.int64); tst = np.full(N, -1, np.int64); ok = np.zeros(N, bool)
    ith = np.full((N, 40), -1, np.int64); itt = np.full(N, -1, np.int64)
    for e in range(N):
        r = rows.get(int(C["uid"][e]))
        if not r:
            continue
        r = sorted(r)                                   # (ts, file order): the builder's stable lexsort
        beh = np.array([x[2] for x in r]); ts = np.array([x[0] for x in r]); it = np.array([x[3] for x in r])
        hb = C["hb"][e]; n = int((hb >= 0).sum())
        if len(beh) < n + 1:
            continue
        # target = last item; history = the n items before it
        if beh[-1] == C["gb"][e] and np.array_equal(beh[-1 - n:-1], hb[:n]):
            ok[e] = True; tsh[e, :n] = ts[-1 - n:-1]; tst[e] = ts[-1]; ith[e, :n] = it[-1 - n:-1]; itt[e] = it[-1]
    np.savez(p + ".ts.npz", user_id=C["uid"], ts_hist=tsh, ts_tgt=tst, ok=ok, item_hist=ith, item_tgt=itt)
    print(f"{os.path.basename(p)}: aligned {ok.mean():.4f} of {N:,} events  ({time.time()-t0:.0f}s)", flush=True)
