import sys, os, gzip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cats_prepare as C

cat = sys.argv[1]; uk = int(sys.argv[2]); ik = int(sys.argv[3])
reviews = os.path.join(C.HERE, f"{cat}.jsonl.gz")
meta = os.path.join(C.HERE, f"meta_{cat}.jsonl.gz")
id_title, _ = C.scan_meta(meta)
user_items, item_users, _ = C.scan_reviews(reviews, id_title)
keep_u, keep_i = C.kcore_peel(user_items, item_users, uk, ik)
out = os.path.join(C.HERE, f"{cat}_10core.jsonl.gz")
n = 0
with gzip.open(out, "wt", encoding="utf-8", compresslevel=1) as fo:
    for line in C.open_gz_lines(reviews):
        try:
            r = C.parse(line)
        except Exception:
            continue
        u = r.get("user_id"); a = r.get("parent_asin") or r.get("asin")
        if u in keep_u and a in keep_i:
            fo.write(line if line.endswith("\n") else line + "\n"); n += 1
print(f"WRITTEN {cat} {uk}-core: users={len(keep_u)} items={len(keep_i)} reviews={n} -> {out}", flush=True)
