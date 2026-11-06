#!/usr/bin/env python3
"""Download exactly one (MAIN) image per item in the user_k/item_k k-core set.

Streams meta/reviews (via cats_prepare), computes the k-core item set, then
downloads `{asin}_MAIN.jpg` with high-concurrency aiohttp.  Resume-safe
(existing files are skipped) and retries on transient errors / 429.

Usage:
  python data/dl_10core_images.py <category> <user_k> <item_k> [concurrency]
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cats_prepare as C  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


async def fetch_one(session, sem, asin, urls, img_dir, stats):
    path = os.path.join(img_dir, f"{asin}_MAIN.jpg")
    if os.path.exists(path):
        stats["exists"] += 1
        return
    async with sem:
        for attempt in range(3):
            for url in urls:
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30, connect=10)
                    ) as r:
                        if r.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(1.5 * (attempt + 1) + random.random())
                            continue
                        r.raise_for_status()
                        data = await r.read()
                        if not data:
                            continue
                        tmp = path + ".part"
                        with open(tmp, "wb") as f:
                            f.write(data)
                        os.replace(tmp, path)
                        stats["ok"] += 1
                        return
                except Exception:
                    await asyncio.sleep(0.4 * (attempt + 1) + random.random() * 0.4)
        stats["fail"] += 1
        stats["failed_items"].append(asin)


async def run(cat, img_dir, items, asin2url, conc):
    missing = [a for a in items if not os.path.exists(os.path.join(img_dir, f"{a}_MAIN.jpg"))]
    log(f"{cat}: items={len(items)} already={len(items) - len(missing)} missing={len(missing)} conc={conc}")
    stats = {"ok": 0, "exists": len(items) - len(missing), "fail": 0, "failed_items": []}
    if not missing:
        log(f"{cat}: nothing to download")
        return stats
    sem = asyncio.Semaphore(conc)
    conn = aiohttp.TCPConnector(limit=conc, limit_per_host=conc, ttl_dns_cache=300)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; image-fetch/1.0)"}
    t0 = time.time()
    async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
        tasks = [asyncio.create_task(fetch_one(session, sem, a, asin2url[a], img_dir, stats)) for a in missing]
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                log(f"{cat}: {done}/{len(missing)} ok={stats['ok']} fail={stats['fail']} ({done/el:.1f}/s)")
    log(f"{cat}: DONE ok={stats['ok']} exists={stats['exists']} fail={stats['fail']} in {time.time()-t0:.0f}s")
    return stats


def main():
    cat = sys.argv[1]
    uk = int(sys.argv[2])
    ik = int(sys.argv[3])
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 192
    reviews = os.path.join(HERE, f"{cat}.jsonl.gz")
    meta = os.path.join(HERE, f"meta_{cat}.jsonl.gz")
    img_dir = os.path.join(HERE, f"{cat}_images")
    os.makedirs(img_dir, exist_ok=True)

    log(f"== {cat} 10-core image download: user_k={uk} item_k={ik} conc={conc} ==")
    id_title, asin2url = C.scan_meta(meta)
    user_items, item_users, _ = C.scan_reviews(reviews, id_title)
    keep_u, keep_i = C.kcore_peel(user_items, item_users, uk, ik)

    items = sorted(a for a in keep_i if a in asin2url)
    with open(os.path.join(HERE, f"{cat}_10core_items.txt"), "w") as f:
        for a in items:
            f.write(a + "\n")
    log(f"{cat}: k-core items={len(keep_i)} with_image_url={len(items)}")

    stats = asyncio.run(run(cat, img_dir, items, asin2url, conc))
    with open(os.path.join(HERE, f"{cat}_10core_images_failed.jsonl"), "w") as f:
        for a in stats["failed_items"]:
            f.write(a + "\n")
    log(f"== {cat} ALL DONE ok={stats['ok']} fail={stats['fail']} ==")


if __name__ == "__main__":
    main()
