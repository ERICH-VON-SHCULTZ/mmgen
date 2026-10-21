"""
Bounded debug: sample a capped number of lines from each raw Tmall log file
and report the distinct behavior-type strings (field index 2) with counts.

Capped at MAX_LINES_PER_FILE so it never scans the full multi-GB files.
0.95% of even 200k lines is ~1.9k rows, so a modest cap still surfaces the
rarest behavior (alipay).
"""

import collections
import os

DATA_DIR = "/scratch/yw8866/rec-tmall"
SEP = "\x01"
MAX_LINES_PER_FILE = 2_000_000  # bounded sample per file

LOG_FILES = [
    "tianchi_2014002_rec_tmall_log_parta.txt",
    "tianchi_2014002_rec_tmall_log_partb.txt",
    "tianchi_2014002_rec_tmall_log_partc.txt",
]

counts = collections.Counter()
total = 0
malformed = 0

for fname in LOG_FILES:
    path = os.path.join(DATA_DIR, fname)
    print(f"Sampling up to {MAX_LINES_PER_FILE:,} lines from {fname} ...", flush=True)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES_PER_FILE:
                break
            parts = line.split(SEP, 3)
            if len(parts) < 4:
                malformed += 1
                continue
            btype = parts[2].strip()
            counts[btype] += 1
            total += 1

print("\n" + "=" * 55)
print(f"Distinct behavior strings (sampled {total:,} rows, malformed {malformed:,}):")
print("=" * 55)
for btype, cnt in counts.most_common():
    print(f"  {repr(btype):<15} : {cnt:>12,}  ({cnt/total*100:6.2f}%)")

# Show how the CURRENT build map would classify each string
print("\nCurrent BEHAVIOR_MAP in build_tmall_tfrecords_mb.py:")
current_map = {"pv": 0, "fav": 1, "cart": 2, "buy": 3}
UNKNOWN = 0
names = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
print(f"  {current_map}   (UNKNOWN_BEHAVIOR={UNKNOWN} -> {names[UNKNOWN]})")
print("\nHow each raw string is currently mapped:")
for btype, cnt in counts.most_common():
    mapped_id = current_map.get(btype, UNKNOWN)
    matched = "MATCH" if btype in current_map else "FELL THROUGH to UNKNOWN"
    print(f"  {repr(btype):<15} -> id={mapped_id} ({names[mapped_id]:<4})  [{matched}]")