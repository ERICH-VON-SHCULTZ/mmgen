"""
Download all product images from tianchi_2014001_rec_tmall_product.txt.
Images saved as IMAGE_DIR/{subfolder}/{item_id}.jpg where each subfolder
holds up to SUBFOLDER_SIZE images.
- Skips items already archived in squashFS (archived_ids.txt).
- Skips items that permanently failed in a prior run (failed_downloads.txt).
- Skips files already on disk — safe to resume after a job kill.
- Uses a thread pool for concurrent I/O.

Usage:
  python download_image.py            # full download
  python download_image.py --test     # probe 1000 random URLs, no files saved
  python download_image.py --test 500 # probe N random URLs
"""

import os
import sys
import random
import time
import logging
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PRODUCT_FILE      = "/scratch/yw8866/rec-tmall/tianchi_2014001_rec_tmall_product.txt"
IMAGE_DIR         = "/scratch/yw8866/rec-tmall/image_data"
FAILED_LOG        = "/scratch/yw8866/rec-tmall/failed_downloads.txt"
ARCHIVED_IDS_FILE = "/scratch/yw8866/rec-tmall/archived_ids.txt"  # IDs already in squashFS
SEP               = "\x01"

NUM_WORKERS    = 32      # concurrent threads
TIMEOUT        = 15      # seconds per request
MAX_RETRIES    = 3       # retries only for transient errors (420, network)
RATE_BACKOFF   = 10      # base seconds to back off on HTTP 420
SUBFOLDER_SIZE   = 10_000  # max images per subdirectory
SUBFOLDER_OFFSET = 485     # old squashFS used 0000–0484; new batch continues from 0485
SUBFOLDER_FMT    = "04d"   # zero-padded 4-digit folder names to match old squashFS
LOG_EVERY      = 10_000  # print progress every N completions

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# Errors that will never succeed on retry — skip these on subsequent runs
PERMANENT_FAIL_CODES = {"HTTP 400", "HTTP 404"}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_permanent_failures(log_path: str) -> set[str]:
    """Return item IDs previously failed with non-retryable errors."""
    skip: set[str] = set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2 and parts[1].strip() in PERMANENT_FAIL_CODES:
                    skip.add(parts[0])
    except FileNotFoundError:
        pass
    return skip


def load_archived_ids(path: str) -> set[str]:
    """Return item IDs already packed into a squashFS archive (one ID per line)."""
    ids: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                iid = line.strip()
                if iid:
                    ids.add(iid)
    except FileNotFoundError:
        pass
    return ids


def parse_product_file(path: str) -> list[tuple[str, str]]:
    """Return list of (item_id, img_url) for rows that have a non-empty URL."""
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(SEP)
            if len(parts) < 3:
                continue
            item_id = parts[0].strip()
            img_url = parts[2].strip()
            if item_id and img_url:
                items.append((item_id, img_url))
    return items


def probe_one(item_id: str, url: str) -> tuple[str, str, str, str]:
    """Mirrors download_one logic exactly but never writes to disk.
    Returns (item_id, 'ok'|'fail', error_msg, url).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True, headers=HEADERS)
            resp.close()
            if resp.status_code == 200:
                return item_id, "ok", "", url
            elif resp.status_code == 420:
                time.sleep(RATE_BACKOFF * (2 ** (attempt - 1)))
                continue
            else:
                return item_id, "fail", f"HTTP {resp.status_code}", url
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                return item_id, "fail", str(e), url

    return item_id, "fail", "HTTP 420 after retries", url


def test_mode(items: list[tuple[str, str]], n: int = 1000) -> None:
    sample = random.sample(items, min(n, len(items)))
    log.info(f"[TEST] Probing {len(sample)} randomly sampled URLs (same logic as full download) ...")

    results: list[tuple[str, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(probe_one, iid, url): iid for iid, url in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(sample):
                log.info(f"  {i}/{len(sample)} probed ...")

    success     = sum(1 for _, s, _, _ in results if s == "ok")
    failures    = [(iid, err, url) for iid, s, err, url in results if s != "ok"]
    fail_counts = Counter(err for _, err, _ in failures)

    log.info("=" * 60)
    log.info(f"[TEST] Results for {len(sample)} sampled URLs:")
    log.info(f"  Success (200)   : {success:,}  ({success/len(sample)*100:.1f}%)")
    log.info(f"  Failed          : {len(failures):,}  ({len(failures)/len(sample)*100:.1f}%)")
    if fail_counts:
        log.info("  Failure breakdown:")
        for reason, cnt in fail_counts.most_common():
            log.info(f"    {reason:<45s}: {cnt:>5,}  ({cnt/len(sample)*100:.1f}%)")
        log.info("  Sample failing URLs (up to 5 per status):")
        shown: Counter = Counter()
        for iid, err, url in failures:
            if shown[err] < 5:
                log.info(f"    [{err}] item={iid}  url={url}")
                shown[err] += 1

    log.info("=" * 60)
    log.info("[TEST] No files were saved. Re-run without --test to start the full download.")


def download_one(item_id: str, url: str, out: Path) -> tuple[str, str, str]:
    """Download a single image to out. Returns (item_id, 'ok'|'fail', error_msg)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True, headers=HEADERS)
            if resp.status_code == 200:
                tmp = out.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                tmp.rename(out)
                return item_id, "ok", ""
            elif resp.status_code == 420:
                time.sleep(RATE_BACKOFF * (2 ** (attempt - 1)))
                continue
            else:
                return item_id, "fail", f"HTTP {resp.status_code}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                return item_id, "fail", str(e)

    return item_id, "fail", "HTTP 420 after retries"


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    log.info("Parsing product file...")
    items = parse_product_file(PRODUCT_FILE)
    log.info(f"  Total items with image URL: {len(items):,}")

    # Build skip sets
    archived    = load_archived_ids(ARCHIVED_IDS_FILE)
    perm_failed = load_permanent_failures(FAILED_LOG)
    skip        = archived | perm_failed
    log.info(f"  Archived in squashFS       : {len(archived):,}")
    log.info(f"  Permanent failures to skip : {len(perm_failed):,}")

    # Assign each remaining item a stable dest path: subfolder = position // SUBFOLDER_SIZE.
    # Using position in the filtered list (not the global list) so subfolders fill
    # evenly from 0.  Deterministic across restarts so resume-after-kill works correctly.
    remaining = [(iid, url) for iid, url in items if iid not in skip]
    todo_all = [
        (iid, url, Path(IMAGE_DIR) / format(i // SUBFOLDER_SIZE + SUBFOLDER_OFFSET, SUBFOLDER_FMT) / f"{iid}.jpg")
        for i, (iid, url) in enumerate(remaining)
    ]

    num_subfolders = (len(todo_all) + SUBFOLDER_SIZE - 1) // SUBFOLDER_SIZE
    first_folder = format(SUBFOLDER_OFFSET, SUBFOLDER_FMT)
    last_folder  = format(SUBFOLDER_OFFSET + num_subfolders - 1, SUBFOLDER_FMT)
    for idx in range(num_subfolders):
        (Path(IMAGE_DIR) / format(idx + SUBFOLDER_OFFSET, SUBFOLDER_FMT)).mkdir(parents=True, exist_ok=True)
    log.info(f"  Subfolders created         : {num_subfolders} ({first_folder}–{last_folder}, ≤{SUBFOLDER_SIZE:,} images each)")

    # Filter already-on-disk (safe resume after job kill)
    on_disk = sum(1 for _, _, out in todo_all if out.exists())
    todo = [(iid, url, out) for iid, url, out in todo_all if not out.exists()]
    log.info(f"  Already on disk            : {on_disk:,}")
    log.info(f"  To download                : {len(todo):,}")

    if not todo:
        log.info("Nothing to do.")
        return

    success     = 0
    done        = 0
    fail_counts: Counter = Counter()

    failed_file = open(FAILED_LOG, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(download_one, iid, url, out): iid
                   for iid, url, out in todo}
        for fut in as_completed(futures):
            item_id, status, err = fut.result()
            done += 1
            if status == "ok":
                success += 1
            else:
                fail_counts[err] += 1
                failed_file.write(f"{item_id}\t{err}\n")
                failed_file.flush()

            if done % LOG_EVERY == 0 or done == len(todo):
                pct = done / len(todo) * 100
                total_failed = sum(fail_counts.values())
                log.info(
                    f"  Progress: {done:,}/{len(todo):,} ({pct:.1f}%)  "
                    f"ok={success:,}  failed={total_failed:,}"
                )

    failed_file.close()
    total_failed = sum(fail_counts.values())
    log.info("=" * 60)
    log.info(f"Done.  Downloaded: {success:,}  |  Failed: {total_failed:,}")
    if fail_counts:
        log.info("  Failure breakdown:")
        for reason, cnt in fail_counts.most_common():
            log.info(f"    {reason:<45s}: {cnt:>8,}  ({cnt/done*100:.1f}%)")
    log.info(f"Failed item IDs written to: {FAILED_LOG}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--test":
        n = int(args[1]) if len(args) > 1 else 1000
        log.info("Parsing product file...")
        items = parse_product_file(PRODUCT_FILE)
        log.info(f"  Total items with image URL: {len(items):,}")
        test_mode(items, n)
    else:
        main()