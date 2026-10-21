import os
import sys
from collections import defaultdict, Counter

DATA_DIR = "/scratch/yw8866/rec-tmall"
SEP = "\x01"

LOG_FILES = [
    "tianchi_2014002_rec_tmall_log_parta.txt",
    "tianchi_2014002_rec_tmall_log_partb.txt",
    "tianchi_2014002_rec_tmall_log_partc.txt",
]
PRODUCT_FILE = "tianchi_2014001_rec_tmall_product.txt"
REVIEW_FILE  = "tianchi_2014003_rec_tmall_review.txt"

# ── Log analysis ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Analyzing log files (user-item interactions)...")
print("=" * 60)

users          = set()
items_log      = set()
behavior_count = Counter()                   # global behavior type counts
user_behavior  = defaultdict(Counter)        # user -> {behavior: count}
total_log_rows = 0
malformed_log  = 0

for fname in LOG_FILES:
    path = os.path.join(DATA_DIR, fname)
    print(f"  Reading {fname} ...", flush=True)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(SEP)
            if len(parts) < 4:
                malformed_log += 1
                continue
            item_id, user_id, btype = parts[0], parts[1], parts[2]
            users.add(user_id)
            items_log.add(item_id)
            behavior_count[btype] += 1
            user_behavior[user_id][btype] += 1
            total_log_rows += 1

print(f"\n[Log] Total interaction rows : {total_log_rows:,}")
print(f"[Log] Unique users            : {len(users):,}")
print(f"[Log] Unique items (in log)   : {len(items_log):,}")
print(f"[Log] Malformed rows skipped  : {malformed_log:,}")

print("\n[Log] Behavior type distribution:")
for btype, cnt in behavior_count.most_common():
    pct = cnt / total_log_rows * 100
    print(f"  {btype:<20s}: {cnt:>15,}  ({pct:.2f}%)")

# ── Per-user behavior statistics ──────────────────────────────────────────────
print("\n[Log] Per-user total interaction count statistics:")
user_totals = [sum(v.values()) for v in user_behavior.values()]
user_totals.sort()
n = len(user_totals)
import statistics
print(f"  min    : {user_totals[0]:,}")
print(f"  max    : {user_totals[-1]:,}")
print(f"  mean   : {statistics.mean(user_totals):.2f}")
print(f"  median : {statistics.median(user_totals):.2f}")
print(f"  stdev  : {statistics.stdev(user_totals):.2f}")

# percentile helper
def pct(lst, p):
    idx = int(len(lst) * p / 100)
    return lst[min(idx, len(lst) - 1)]

print(f"  p25    : {pct(user_totals, 25):,}")
print(f"  p75    : {pct(user_totals, 75):,}")
print(f"  p90    : {pct(user_totals, 90):,}")
print(f"  p99    : {pct(user_totals, 99):,}")

# Bin distribution
bins = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, float("inf")]
labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-200",
          "201-500", "501-1000", "1001-5000", "5001+"]
bin_counts = Counter()
for t in user_totals:
    for i, b in enumerate(bins):
        if t <= b:
            bin_counts[labels[i]] += 1
            break

print("\n[Log] Distribution of total interactions per user:")
print(f"  {'Bin':<12} {'#Users':>10} {'%Users':>8}")
for label in labels:
    c = bin_counts[label]
    print(f"  {label:<12} {c:>10,} {c/n*100:>7.2f}%")

# Per-user, per-behavior-type stats
print("\n[Log] Per-user behavior-type counts (mean across users who have that behavior):")
btype_user_counts = defaultdict(list)
for uid, bctr in user_behavior.items():
    for btype, cnt in bctr.items():
        btype_user_counts[btype].append(cnt)

for btype in sorted(btype_user_counts):
    vals = sorted(btype_user_counts[btype])
    n_b = len(vals)
    print(f"  {btype}")
    print(f"    users with this behavior: {n_b:,}  ({n_b/len(users)*100:.2f}%)")
    print(f"    mean={statistics.mean(vals):.2f}  median={statistics.median(vals):.1f}  "
          f"max={vals[-1]:,}  p90={pct(vals,90):,}")

# ── Product analysis ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Analyzing product file...")
print("=" * 60)

items_product  = set()
items_with_img = set()
brands         = set()
categories     = set()
sellers        = set()
total_prod     = 0
malformed_prod = 0

with open(os.path.join(DATA_DIR, PRODUCT_FILE), "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split(SEP)
        total_prod += 1
        if len(parts) < 3:
            malformed_prod += 1
            continue
        item_id = parts[0]
        items_product.add(item_id)
        img_url = parts[2] if len(parts) > 2 else ""
        if img_url.strip():
            items_with_img.add(item_id)
        if len(parts) > 3 and parts[3].strip():
            categories.add(parts[3].strip())
        if len(parts) > 4 and parts[4].strip():
            brands.add(parts[4].strip())
        if len(parts) > 5 and parts[5].strip():
            sellers.add(parts[5].strip())

print(f"[Product] Total product rows         : {total_prod:,}")
print(f"[Product] Unique items               : {len(items_product):,}")
print(f"[Product] Items with image URL       : {len(items_with_img):,}")
print(f"[Product] Unique categories          : {len(categories):,}")
print(f"[Product] Unique brands              : {len(brands):,}")
print(f"[Product] Unique sellers             : {len(sellers):,}")
print(f"[Product] Malformed rows             : {malformed_prod:,}")

# Coverage
items_in_both = items_log & items_product
print(f"\n[Coverage] Items in log & product    : {len(items_in_both):,}")
print(f"[Coverage] Items in log only         : {len(items_log - items_product):,}")
print(f"[Coverage] Items in product only     : {len(items_product - items_log):,}")
img_covered = items_log & items_with_img
print(f"[Coverage] Log items with image      : {len(img_covered):,}  ({len(img_covered)/len(items_log)*100:.2f}%)")

# ── Review analysis ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Analyzing review file...")
print("=" * 60)

review_users  = set()
review_items  = set()
total_reviews = 0
malformed_rev = 0

with open(os.path.join(DATA_DIR, REVIEW_FILE), "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split(SEP)
        total_reviews += 1
        if len(parts) < 2:
            malformed_rev += 1
            continue
        review_items.add(parts[0])
        review_users.add(parts[1])

print(f"[Review] Total reviews               : {total_reviews:,}")
print(f"[Review] Unique items reviewed       : {len(review_items):,}")
print(f"[Review] Unique users who reviewed   : {len(review_users):,}")
print(f"[Review] Malformed rows              : {malformed_rev:,}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Users (log)            : {len(users):,}")
print(f"  Items (log)            : {len(items_log):,}")
print(f"  Items (product file)   : {len(items_product):,}")
print(f"  Items with picture URL : {len(items_with_img):,}")
print(f"  Total interactions     : {total_log_rows:,}")
print(f"  Behavior types         : {len(behavior_count)}  -> {list(behavior_count.keys())}")
print(f"  Reviews                : {total_reviews:,}")
print("=" * 60)
