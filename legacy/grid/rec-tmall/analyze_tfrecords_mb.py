"""
Analyze behavior distribution and sequence statistics in the tfrecords_mb dataset.

Reads all GZIP TFRecords from training/, evaluation/, testing/ and reports:
  - Per-split user counts and interaction counts
  - Global behavior frequency (pv=0, fav=1, cart=2, buy=3)
  - Per-split behavior frequency
  - Sequence length distribution (in items, i.e. tokens / 1 since behavior is separate)
  - Fraction of users who have at least one buy interaction
"""

import collections
import os
import sys

import numpy as np
import tensorflow as tf

DATA_DIR = "/scratch/yw8866/rec-tmall/tfrecords_mb"
SPLITS   = ("training", "evaluation", "testing")

# Must match BEHAVIOR_MAP in build_tmall_tfrecords_mb.py
BEHAVIOR_NAMES = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}

def files_in(split):
    d = os.path.join(DATA_DIR, split)
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tfrecord.gz")
    )

feature_spec = {
    "user_id":       tf.io.VarLenFeature(tf.int64),
    "sequence_data": tf.io.VarLenFeature(tf.int64),
    "behavior_data": tf.io.VarLenFeature(tf.int64),
}

def parse(raw):
    return tf.io.parse_single_example(raw, feature_spec)

def read_split(split):
    """Return (list_of_seq_lengths, flat behavior array) for one split."""
    paths = files_in(split)
    ds = tf.data.TFRecordDataset(paths, compression_type="GZIP", num_parallel_reads=8)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

    seq_lens   = []
    bhv_counts = collections.Counter()
    n_users    = 0
    has_buy    = 0

    for ex in ds:
        bhv = tf.sparse.to_dense(ex["behavior_data"]).numpy()
        n_users += 1
        seq_lens.append(len(bhv))
        for b in bhv:
            bhv_counts[int(b)] += 1
        if 3 in bhv:
            has_buy += 1

    return n_users, seq_lens, bhv_counts, has_buy

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TFRecords-MB dataset analysis")
print(f"  Data dir: {DATA_DIR}")
print("=" * 65)

global_bhv   = collections.Counter()
global_lens  = []
global_users = 0
global_buy_users = 0

for split in SPLITS:
    print(f"\n[{split}]  reading ...", flush=True)
    n_users, seq_lens, bhv_counts, has_buy = read_split(split)
    total_interactions = sum(bhv_counts.values())
    global_bhv   += bhv_counts
    global_lens  += seq_lens
    global_users += n_users
    global_buy_users += has_buy

    print(f"  Users               : {n_users:,}")
    print(f"  Total interactions  : {total_interactions:,}")
    print(f"  Users with ≥1 buy   : {has_buy:,}  ({has_buy/n_users*100:.1f}%)")
    print(f"  Seq len — min={min(seq_lens)}  max={max(seq_lens)}  "
          f"mean={np.mean(seq_lens):.1f}  median={np.median(seq_lens):.1f}")
    print(f"\n  Behavior distribution:")
    for bid in sorted(BEHAVIOR_NAMES):
        cnt = bhv_counts[bid]
        pct = cnt / total_interactions * 100 if total_interactions else 0
        print(f"    {BEHAVIOR_NAMES[bid]:<6} (id={bid}): {cnt:>10,}  ({pct:6.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("GLOBAL SUMMARY (all splits combined)")
print("=" * 65)
total = sum(global_bhv.values())
print(f"  Total users         : {global_users:,}")
print(f"  Total interactions  : {total:,}")
print(f"  Users with ≥1 buy   : {global_buy_users:,}  ({global_buy_users/global_users*100:.1f}%)")
print(f"  Seq len — min={min(global_lens)}  max={max(global_lens)}  "
      f"mean={np.mean(global_lens):.1f}  median={int(np.median(global_lens))}")

print(f"\n  Behavior distribution:")
for bid in sorted(BEHAVIOR_NAMES):
    cnt = global_bhv[bid]
    pct = cnt / total * 100 if total else 0
    print(f"    {BEHAVIOR_NAMES[bid]:<6} (id={bid}): {cnt:>10,}  ({pct:6.2f}%)")

print("\n  ── Compare to MMMB paper (ijcai dataset) ──")
mmmb = {"pv": 83.6, "buy": 8.1, "fav": 8.1, "cart": 0.2}
for bname, paper_pct in sorted(mmmb.items()):
    bid = next(k for k, v in BEHAVIOR_NAMES.items() if v == bname)
    our_pct = global_bhv[bid] / total * 100
    print(f"    {bname:<6}: ours={our_pct:6.2f}%   paper={paper_pct:.1f}%   "
          f"delta={our_pct - paper_pct:+.1f}pp")

print("=" * 65)