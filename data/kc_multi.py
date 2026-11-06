import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cats_prepare as C
cat = sys.argv[1]
reviews = os.path.join(C.HERE, f"{cat}.jsonl.gz")
meta = os.path.join(C.HERE, f"meta_{cat}.jsonl.gz")
id_title, _ = C.scan_meta(meta)
user_items, item_users, _ = C.scan_reviews(reviews, id_title)
for k in (10, 15, 20):
    ku, ki = C.kcore_peel(user_items, item_users, k, k)
    n = sum(1 for u in ku for a in user_items[u] if a in ki)
    print(f"RESULT {cat} {k}-core: users={len(ku)} items={len(ki)} interactions={n}", flush=True)
