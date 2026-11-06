"""Generate a tiny synthetic Amazon-style dataset for smoke tests.

Creates gzip-compressed reviews/metadata plus placeholder MAIN images so that
``data/possess_data.py`` can run end-to-end without downloading anything.
"""

import argparse
import gzip
import json
import os


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--category", default="SmokeTest")
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--items", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, f"{args.category}_images")
    os.makedirs(img_dir, exist_ok=True)

    asins = [f"SMOKE{i:04d}" for i in range(args.items)]
    metadata = []
    for i, asin in enumerate(asins):
        with open(os.path.join(img_dir, f"{asin}_MAIN.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xd9")
        metadata.append(
            {
                "parent_asin": asin,
                "title": f"Synthetic Smoke Item {i}",
                "description": f"A tiny synthetic product used for smoke testing item {i}.",
                "store": "SmokeBrand",
                "categories": [["Smoke", "Test"]],
                "images": [{"variant": "MAIN", "hi_res": f"file://{asin}_MAIN.jpg"}],
            }
        )

    reviews = []
    base_ts = 1_700_000_000_000
    for u in range(args.users):
        user = f"user{u:03d}"
        for k in range(8):
            item_idx = (u * 3 + k) % args.items
            asin = asins[item_idx]
            reviews.append(
                {
                    "parent_asin": asin,
                    "user_id": user,
                    "timestamp": base_ts + u * 10_000 + k * 1_000,
                    "rating": 5.0,
                    "text": f"Smoke review {k} from {user} for {asin}.",
                    "title": f"Review {k}",
                }
            )

    write_jsonl_gz(os.path.join(args.out_dir, f"{args.category}.jsonl.gz"), reviews)
    write_jsonl_gz(os.path.join(args.out_dir, f"meta_{args.category}.jsonl.gz"), metadata)
    print(
        f"[smoke-data] wrote {len(reviews)} reviews, {len(metadata)} items, "
        f"{len(os.listdir(img_dir))} images to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
