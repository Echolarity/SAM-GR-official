import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cats_prepare as C
cat=sys.argv[1]; uk=int(sys.argv[2]); ik=int(sys.argv[3])
reviews=f"data/{cat}.jsonl.gz"; meta=f"data/meta_{cat}.jsonl.gz"
id_title,_=C.scan_meta(meta)
user_items,item_users,_=C.scan_reviews(reviews, id_title)
keep_u,keep_i=C.kcore_peel(user_items,item_users,uk,ik)
n=sum(1 for u in keep_u for a in user_items[u] if a in keep_i)
print(f"RESULT {cat} {uk}-core: users={len(keep_u)} items={len(keep_i)} interactions={n}", flush=True)
