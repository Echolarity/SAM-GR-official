import sys, os, gzip, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cats_prepare as C

cat = sys.argv[1]; k = int(sys.argv[2])
reviews = os.path.join(C.HERE, f"{cat}.jsonl.gz")
meta = os.path.join(C.HERE, f"meta_{cat}.jsonl.gz")
id_title, _ = C.scan_meta(meta)

# unique-item / unique-user graph (strict)
user_items = collections.defaultdict(set); item_users = collections.defaultdict(set)
for line in C.open_gz_lines(reviews):
    try: r = C.parse(line)
    except Exception: continue
    u = r.get("user_id"); a = r.get("parent_asin") or r.get("asin")
    if u and a and a in id_title:
        user_items[u].add(a); item_users[a].add(u)

# standard queue-based k-core peel on unique degrees
deg_u = {u: len(v) for u, v in user_items.items()}
deg_i = {a: len(v) for a, v in item_users.items()}
rem_u = {u for u, d in deg_u.items() if d < k}
rem_i = {a for a, d in deg_i.items() if d < k}
qu, qi = list(rem_u), list(rem_i)
while qu or qi:
    while qu:
        u = qu.pop()
        for a in user_items[u]:
            if a in rem_i: continue
            deg_i[a] -= 1
            if deg_i[a] < k: rem_i.add(a); qi.append(a)
    while qi:
        a = qi.pop()
        for u in item_users[a]:
            if u in rem_u: continue
            deg_u[u] -= 1
            if deg_u[u] < k: rem_u.add(u); qu.append(u)
keep_u = set(user_items) - rem_u; keep_i = set(item_users) - rem_i
n_inter = sum(1 for u in keep_u for a in user_items[u] if a in keep_i)
print(f"STRICT {cat} {k}-core: users={len(keep_u)} items={len(keep_i)} interactions={n_inter}", flush=True)

out = os.path.join(C.HERE, f"{cat}_strict{k}core.jsonl.gz")
n = 0
with gzip.open(out, "wt", encoding="utf-8", compresslevel=1) as fo:
    for line in C.open_gz_lines(reviews):
        try: r = C.parse(line)
        except Exception: continue
        u = r.get("user_id"); a = r.get("parent_asin") or r.get("asin")
        if u in keep_u and a in keep_i:
            fo.write(line if line.endswith("\n") else line + "\n"); n += 1
print(f"WRITTEN {cat}: {n} reviews -> {out}", flush=True)
