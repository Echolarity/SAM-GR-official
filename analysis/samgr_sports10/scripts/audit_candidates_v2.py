"""Offline audit v2 (fast): beam collapse, prefix expansion, candidate coverage, merge sweep."""
from __future__ import annotations

import collections
import json
import math
import os
import re

import numpy as np
import pandas as pd

BASE = "/data02/sht/SAM-GR"
OUT = os.path.join(BASE, "analysis", "samgr_sports10")
RESULTS = os.path.join(OUT, "results")
DATA = os.path.join(BASE, "data", "Sports_and_Outdoors")
AMZ = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
RES2 = os.path.join(BASE, "results", "qwen2.5-0.5b", "full",
                    "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict", "final_result.json")
RES4 = os.path.join(BASE, "results", "qwen2.5-0.5b", "full",
                    "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_ep4", "final_result.json")
ID2SID = os.path.join(DATA, "id2sid.json")
TRAIN_INTER = os.path.join(AMZ, "Sports_and_Outdoors.train.inter")
COLLAB = os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy")
SID_RE = re.compile(r"<a_\d+><b_\d+><c_\d+><d_\d+>")
TOK_RE = re.compile(r"<[abcd]_\d+>")
KS = [1, 3, 5, 10, 20, 50]


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def topk_subset(x, subset, k=50, chunk=512):
    xs = l2n(x[subset])
    xa = l2n(x)
    out = np.empty((len(subset), k), dtype=np.int32)
    for st in range(0, len(subset), chunk):
        en = min(st + chunk, len(subset))
        sim = xs[st:en] @ xa.T
        sim[np.arange(en - st), subset[st:en]] = -2.0
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        out[st:en] = np.take_along_axis(part, order, axis=1)
    return out


def prefixes(sid, level):
    return "".join(TOK_RE.findall(sid)[:level])


def eval_ranks(ranks, n):
    out = {}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r < k)
        dcg = sum(1.0 / math.log2(r + 2) for r in ranks if r is not None and r < k)
        out[f"HR@{k}"] = hits / n
        out[f"NDCG@{k}"] = dcg / n
    return out


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def save(obj, name):
    path = os.path.join(RESULTS, name)
    json.dump(obj, open(path, "w"), indent=2)
    print("saved", path, flush=True)


def main():
    out = {}
    id2sid = json.load(open(ID2SID, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in id2sid.items()}
    n_items = len(item2sid)
    sid2items = collections.defaultdict(list)
    for i, s in item2sid.items():
        sid2items[s].append(i)
    sids = sorted(sid2items)
    print(f"items={n_items} unique_sids={len(sids)}", flush=True)

    d2 = json.load(open(RES2, encoding="utf-8"))
    d4 = json.load(open(RES4, encoding="utf-8"))
    n = len(d2)

    pop = np.zeros(n_items, dtype=np.float64)
    with open(TRAIN_INTER, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                pop[int(p[2])] += 1
    sid_pop = {s: float(sum(pop[i] for i in items)) for s, items in sid2items.items()}
    pop_sids = [s for s, _ in sorted(sid_pop.items(), key=lambda kv: -kv[1])[:200]]

    preds2, preds4, targets, last_sids = [], [], [], []
    for x in d2:
        preds2.append([str(v).strip().strip('"') for v in x["predict"]])
        targets.append(str(x["output"]).strip().strip('"'))
        hist = SID_RE.findall(str(x["input"]))
        last_sids.append(hist[-1] if hist else pop_sids[0])
    for x in d4:
        preds4.append([str(v).strip().strip('"') for v in x["predict"]])

    ranks2 = [p.index(t) if t in p else None for p, t in zip(preds2, targets)]
    base = eval_ranks(ranks2, n)
    out["base"] = {"n": n, "metrics": base}
    print("base", {k: round(base[k], 4) for k in ["HR@1", "HR@50", "NDCG@50"]}, flush=True)

    top1 = collections.Counter(p[0] for p in preds2)
    all_beam = collections.Counter(s for p in preds2 for s in p)
    uniq = [len(set(p)) for p in preds2]
    out["collapse"] = {
        "unique_top1": len(top1),
        "top10_top1_share": sum(v for _, v in top1.most_common(10)) / n,
        "unique_beam_sids": len(all_beam),
        "mean_unique_per_sample": float(np.mean(uniq)),
        "min_unique_per_sample": int(np.min(uniq)),
        "samples_lt50_unique": int(np.sum(np.array(uniq) < 50)),
        "hit_rank_1": sum(1 for r in ranks2 if r == 0) / n,
        "hit_rank_le_10": sum(1 for r in ranks2 if r is not None and r < 10) / n,
        "hit_rank_le_50": sum(1 for r in ranks2 if r is not None and r < 50) / n,
    }
    print("collapse", out["collapse"], flush=True)

    # prefix coverage and expansion
    pref_sids = {1: collections.defaultdict(list), 2: collections.defaultdict(list),
                 3: collections.defaultdict(list)}
    for s in sids:
        for lv in [1, 2, 3]:
            pref_sids[lv][prefixes(s, lv)].append(s)
    pref_stats = {}
    for lv in [1, 2, 3]:
        cov = 0
        sizes = []
        for p, t in zip(preds2, targets):
            ps = {prefixes(x, lv) for x in p}
            cov += int(prefixes(t, lv) in ps)
            sizes.append(sum(len(pref_sids[lv][q]) for q in ps))
        pref_stats[f"level_{lv}"] = {
            "coverage": cov / n,
            "n_prefixes_used": len(pref_sids[lv]),
            "mean_sids_per_prefix": float(np.mean([len(v) for v in pref_sids[lv].values()])),
            "mean_candidate_sids": float(np.mean(sizes)),
            "p90_candidate_sids": float(np.percentile(sizes, 90)),
        }
    out["prefix"] = pref_stats
    print("prefix", {k: (round(v["coverage"], 4), round(v["mean_candidate_sids"], 1))
                     for k, v in pref_stats.items()}, flush=True)
    save(out, "audit_candidates.json")

    # last-item subset for neighbor search
    last_items = sorted({i for ls in last_sids for i in sid2items[ls]})
    lindex = {i: k for k, i in enumerate(last_items)}
    print(f"unique last items={len(last_items)}", flush=True)

    collab = np.load(COLLAB).astype(np.float32)
    cf_top = topk_subset(collab, last_items, k=50)
    print("cf done", flush=True)
    sem_top = {}
    for name in ["text_only", "multimodal"]:
        x = np.load(os.path.join(AMZ, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy"),
                    mmap_mode="r").astype(np.float32)
        sem_top[name] = topk_subset(x, last_items, k=50)
        print(name, "done", flush=True)

    # coverage at K=20/50
    coverage = {}
    for K in [20, 50]:
        hit = collections.Counter()
        csize = collections.Counter()
        for p, t, ls in zip(preds2, targets, last_sids):
            b = set(p)
            c, tx, mm = set(), set(), set()
            for i in sid2items[ls]:
                row = lindex[i]
                c.update(item2sid[int(j)] for j in cf_top[row, :K])
                tx.update(item2sid[int(j)] for j in sem_top["text_only"][row, :K])
                mm.update(item2sid[int(j)] for j in sem_top["multimodal"][row, :K])
            pp = set(pop_sids[:K])
            u1 = b | c
            u2 = u1 | tx
            u3 = u2 | mm
            u4 = u3 | pp
            hit["beam"] += int(t in b)
            hit["cf_only"] += int(t in c)
            hit["text_only"] += int(t in tx)
            hit["mm_only"] += int(t in mm)
            hit["pop_only"] += int(t in pp)
            hit["beam+cf"] += int(t in u1)
            hit["beam+cf+text"] += int(t in u2)
            hit["beam+cf+text+mm"] += int(t in u3)
            hit["beam+cf+text+mm+pop"] += int(t in u4)
            csize["cf"] += len(c)
            csize["text"] += len(tx)
            csize["mm"] += len(mm)
            csize["union"] += len(u4)
        coverage[f"K{K}"] = {k: v / n for k, v in hit.items()}
        coverage[f"K{K}"]["mean_cands"] = {k: v / n for k, v in csize.items()}
        print("K", K, {k: round(v, 4) for k, v in hit.items()}, flush=True)
    out["coverage"] = coverage
    save(out, "audit_candidates.json")

    # prefix expansion + optional union
    exp = {}
    for lv in [1, 2]:
        cov = 0
        sizes = []
        cov_u = 0
        sizes_u = []
        for p, t, ls in zip(preds2, targets, last_sids):
            ps = {prefixes(x, lv) for x in p}
            cand = set()
            for q in ps:
                cand.update(pref_sids[lv][q])
            cov += int(t in cand)
            sizes.append(len(cand))
            # + CF/text/mm/pop union at K=20
            c, tx, mm = set(), set(), set()
            for i in sid2items[ls]:
                row = lindex[i]
                c.update(item2sid[int(j)] for j in cf_top[row, :20])
                tx.update(item2sid[int(j)] for j in sem_top["text_only"][row, :20])
                mm.update(item2sid[int(j)] for j in sem_top["multimodal"][row, :20])
            u = cand | set(p) | c | tx | mm | set(pop_sids[:20])
            cov_u += int(t in u)
            sizes_u.append(len(u))
        exp[f"level_{lv}"] = {
            "coverage": cov / n,
            "mean_cands": float(np.mean(sizes)),
            "p90_cands": float(np.percentile(sizes, 90)),
            "coverage_plus_cf_text_mm_pop20": cov_u / n,
            "mean_cands_plus": float(np.mean(sizes_u)),
        }
        print("expand", lv, exp[f"level_{lv}"], flush=True)
    out["prefix_expansion"] = exp

    # 2ep + ep4 union
    uh = 0
    us = []
    for p2, p4, t in zip(preds2, preds4, targets):
        u = set(p2) | set(p4)
        uh += int(t in u)
        us.append(len(u))
    out["beam_union_2ep_4ep"] = {"coverage": uh / n, "mean_cands": float(np.mean(us)),
                                  "p90_cands": float(np.percentile(us, 90)),
                                  "unique_4_beam": len({s for p in preds4 for s in p})}
    print("beam union", out["beam_union_2ep_4ep"], flush=True)
    save(out, "audit_candidates.json")

    # post-hoc merge sweep
    sid_rep = {s: max(items, key=lambda i: pop[i]) for s, items in sid2items.items()}
    all_items = np.arange(n_items)
    cf_all_top1 = topk_subset(collab, all_items, k=1)[:, 0]
    rep_top1_sid = {s: item2sid[int(cf_all_top1[sid_rep[s]])] for s in sids}
    current_merged = n_items - len(sids)
    merge_rows = []
    for ratio in [0.20, 0.25, 0.30]:
        target_merged = int(ratio * n_items)
        additional = target_merged - current_merged
        mapping = {s: s for s in sids}
        merged = set()
        for s in sorted(sids, key=lambda x: sid_pop[x]):
            if additional <= 0:
                break
            if s in merged:
                continue
            j = rep_top1_sid[s]
            if j == s or sid_pop.get(j, 0) < sid_pop[s]:
                continue
            mapping[s] = j
            merged.add(s)
            additional -= len(sid2items[s])
        for s in sids:
            while mapping[s] != mapping[mapping[s]]:
                mapping[s] = mapping[mapping[s]]
        group_items = collections.Counter()
        for s in sids:
            group_items[mapping[s]] += len(sid2items[s])
        ranks = []
        for p, t in zip(preds2, targets):
            mp = list(dict.fromkeys(mapping[s] for s in p))
            mt = mapping[t]
            ranks.append(mp.index(mt) if mt in mp else None)
        m = eval_ranks(ranks, n)
        avg_group = float(np.mean([group_items[mapping[t]] for t in targets]))
        row = {"ratio": ratio, "unique_sids": len(group_items),
               "collision": 1 - len(group_items) / n_items,
               "max_group": int(max(group_items.values())),
               "avg_target_group": avg_group,
               **{k: m[k] for k in ["HR@1", "HR@10", "HR@50", "NDCG@50"]}}
        merge_rows.append(row)
        print("merge", row, flush=True)
    out["merge_sweep"] = merge_rows
    pd.DataFrame(merge_rows).to_csv(os.path.join(RESULTS, "audit_merge_sweep.csv"), index=False)
    save(out, "audit_candidates.json")
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
