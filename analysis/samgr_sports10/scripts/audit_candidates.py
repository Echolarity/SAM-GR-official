"""Offline audit of saved Sports strict-10-core beams (no model / GPU).

Reads the existing 50-beam predictions and computes:
  - beam duplication / collapse statistics
  - target rank distribution (sanity check vs metrics.json)
  - prefix coverage (a, a+b, a+b+c) vs full-SID coverage
  - candidate coverage of CF / text / multimodal / popularity sources
  - oracle HR@1 and HR@50 for each candidate union
  - 2ep + ep4 beam union
  - post-hoc SID merge sweep (15/20/25/30/35%) at SID level

Output: results/audit_candidates.json, results/audit_candidates.csv
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys

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


def topk_all(x, k=100, chunk=512):
    x = l2n(x)
    n = len(x)
    out = np.empty((n, k), dtype=np.int32)
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sim = x[st:en] @ x.T
        sim[np.arange(en - st), np.arange(st, en)] = -2.0
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        out[st:en] = np.take_along_axis(part, order, axis=1)
    return out


def prefixes(sid, level):
    toks = TOK_RE.findall(sid)
    return "".join(toks[:level])


def eval_ranks(ranks, n):
    out = {}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r < k)
        dcg = sum(1.0 / math.log2(r + 2) for r in ranks if r is not None and r < k)
        out[f"HR@{k}"] = hits / n
        out[f"NDCG@{k}"] = dcg / n
    return out


def main():
    print("[1] load id2sid", flush=True)
    id2sid = json.load(open(ID2SID, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in id2sid.items()}
    n_items = len(item2sid)
    sid2items = collections.defaultdict(list)
    for i, s in item2sid.items():
        sid2items[s].append(i)
    sids = sorted(sid2items)
    print(f"    items={n_items} unique_sids={len(sids)} max_group={max(len(v) for v in sid2items.values())}")

    print("[2] load beams", flush=True)
    d2 = json.load(open(RES2, encoding="utf-8"))
    d4 = json.load(open(RES4, encoding="utf-8"))
    n = len(d2)
    print(f"    n={n}")

    # popularity from train.inter (target column)
    pop = np.zeros(n_items, dtype=np.float64)
    with open(TRAIN_INTER, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                pop[int(p[2])] += 1
    sid_pop = collections.Counter()
    for s, items in sid2items.items():
        sid_pop[s] = float(sum(pop[i] for i in items))
    pop_sids = [s for s, _ in sid_pop.most_common(200)]

    print("[3] parse beams / targets / histories", flush=True)
    preds2, preds4, targets, last_sids = [], [], [], []
    for k, x in enumerate(d2):
        p = [str(v).strip().strip('"') for v in x["predict"]]
        preds2.append(p)
        t = str(x["output"]).strip().strip('"')
        targets.append(t)
        hist = SID_RE.findall(str(x["input"]))
        last_sids.append(hist[-1] if hist else pop_sids[0])
        if k < 5:
            assert t in sid2items, (t, x["output"])
    for x in d4:
        preds4.append([str(v).strip().strip('"') for v in x["predict"]])

    # sanity: metrics from ranks
    ranks2 = []
    for p, t in zip(preds2, targets):
        ranks2.append(p.index(t) if t in p else None)
    base_metrics = eval_ranks(ranks2, n)
    print("    reproduced:", {k: round(base_metrics[k], 4) for k in ["HR@1", "HR@50", "NDCG@50"]})

    # collapse stats
    top1 = collections.Counter(p[0] for p in preds2)
    all_beam = collections.Counter(s for p in preds2 for s in p)
    uniq_per_sample = [len(set(p)) for p in preds2]
    top1_share = sum(v for _, v in top1.most_common(10)) / n
    collapse = {
        "unique_top1": len(top1),
        "top10_top1_share": top1_share,
        "unique_beam_sids": len(all_beam),
        "mean_unique_per_sample": float(np.mean(uniq_per_sample)),
        "min_unique_per_sample": int(np.min(uniq_per_sample)),
        "samples_with_lt50_unique": int(np.sum(np.array(uniq_per_sample) < 50)),
    }
    print("    collapse:", collapse)

    # prefix coverage / prefix group sizes
    prefix_stats = {}
    for level in [1, 2, 3]:
        hit = 0
        pref_sizes = collections.Counter()
        for s in sids:
            pref_sizes[prefixes(s, level)] += len(sid2items[s])
        for p, t in zip(preds2, targets):
            if prefixes(t, level) in {prefixes(x, level) for x in p}:
                hit += 1
        vals = np.array(list(pref_sizes.values()), dtype=np.float64)
        prefix_stats[f"level_{level}"] = {
            "coverage": hit / n,
            "n_prefixes": len(pref_sizes),
            "mean_items_per_prefix": float(vals.mean()),
            "p90_items_per_prefix": float(np.percentile(vals, 90)),
            "max_items_per_prefix": int(vals.max()),
        }
    print("    prefix:", {k: round(v["coverage"], 4) for k, v in prefix_stats.items()})

    print("[4] CF top-100 for all items", flush=True)
    collab = np.load(COLLAB).astype(np.float32)
    cf_top = topk_all(collab, k=100)

    print("[5] semantic top-100 for all items", flush=True)
    sem_top = {}
    for name in ["text_only", "multimodal"]:
        x = np.load(os.path.join(AMZ, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy"),
                    mmap_mode="r").astype(np.float32)
        sem_top[name] = topk_all(x, k=100)
        print(f"    {name} done", flush=True)

    print("[6] candidate coverage", flush=True)
    # incremental union at K=50
    def cand_sets(k):
        cf, tx, mm, pp = set(), set(), set(), set()
        for p, t, ls in zip(preds2, targets, last_sids):
            items = sid2items[ls]
            cf.update(item2sid[int(j)] for i in items for j in cf_top[i, :k])
            tx.update(item2sid[int(j)] for i in items for j in sem_top["text_only"][i, :k])
            mm.update(item2sid[int(j)] for i in items for j in sem_top["multimodal"][i, :k])
        pp = set(pop_sids[:k])
        return cf, tx, mm, pp

    coverage = {}
    for k in [10, 20, 50]:
        cf, tx, mm, pp = cand_sets(k)
        hit = {"beam": 0, "beam+cf": 0, "beam+cf+text": 0, "beam+cf+text+mm": 0,
               "beam+cf+text+mm+pop": 0, "cf_only": 0, "text_only": 0, "mm_only": 0, "pop_only": 0}
        sizes = {"cf": 0, "text": 0, "mm": 0, "union": 0}
        for p, t, ls in zip(preds2, targets, last_sids):
            items = sid2items[ls]
            b = set(p)
            c = {item2sid[int(j)] for i in items for j in cf_top[i, :k]}
            tx = {item2sid[int(j)] for i in items for j in sem_top["text_only"][i, :k]}
            mm = {item2sid[int(j)] for i in items for j in sem_top["multimodal"][i, :k]}
            pp = set(pop_sids[:k])
            hit["beam"] += int(t in b)
            hit["cf_only"] += int(t in c)
            hit["text_only"] += int(t in tx)
            hit["mm_only"] += int(t in mm)
            hit["pop_only"] += int(t in pp)
            u1 = b | c
            u2 = u1 | tx
            u3 = u2 | mm
            u4 = u3 | pp
            hit["beam+cf"] += int(t in u1)
            hit["beam+cf+text"] += int(t in u2)
            hit["beam+cf+text+mm"] += int(t in u3)
            hit["beam+cf+text+mm+pop"] += int(t in u4)
            sizes["cf"] += len(c)
            sizes["text"] += len(tx)
            sizes["mm"] += len(mm)
            sizes["union"] += len(u4)
        coverage[f"K{k}"] = {kk: v / n for kk, v in hit.items()}
        coverage[f"K{k}"]["mean_cands"] = {kk: v / n for kk, v in sizes.items()}
        print(f"    K={k}", {kk: round(v, 4) for kk, v in hit.items()}, flush=True)

    # beam union 2ep+4ep
    union_hit = 0
    union_sizes = 0
    for p2, p4, t in zip(preds2, preds4, targets):
        u = set(p2) | set(p4)
        union_hit += int(t in u)
        union_sizes += len(u)
    beam_union = {"coverage": union_hit / n, "mean_cands": union_sizes / n,
                  "unique_4_beam": len({s for p in preds4 for s in p})}

    # post-hoc merge sweep
    print("[7] post-hoc merge sweep", flush=True)
    sid_rep = {s: max(items, key=lambda i: pop[i]) for s, items in sid2items.items()}
    rep_top1 = {s: int(cf_top[sid_rep[s], 0]) for s in sids}
    merge_rows = []
    current_merged = n_items - len(sids)
    base_pred_mapped = None
    for ratio in [0.15, 0.20, 0.25, 0.30, 0.35]:
        target_merged = int(ratio * n_items)
        additional = target_merged - current_merged
        mapping = {s: s for s in sids}
        merged = set()
        if additional > 0:
            for s in sorted(sids, key=lambda x: sid_pop[x]):
                if additional <= 0:
                    break
                if s in merged:
                    continue
                j = item2sid[rep_top1[s]]
                if j == s or sid_pop[j] < sid_pop[s]:
                    continue
                mapping[s] = j
                merged.add(s)
                additional -= len(sid2items[s])
        # resolve chains
        for s in sids:
            while mapping[s] != mapping[mapping[s]]:
                mapping[s] = mapping[mapping[s]]
        # group stats
        groups = collections.Counter(mapping[s] for s in sids)
        # metrics
        ranks = []
        for p, t in zip(preds2, targets):
            mp = [mapping[s] for s in p]
            mp = list(dict.fromkeys(mp))
            mt = mapping[t]
            ranks.append(mp.index(mt) if mt in mp else None)
        m = eval_ranks(ranks, n)
        # average target group size (items)
        group_items = collections.Counter()
        for s in sids:
            group_items[mapping[s]] += len(sid2items[s])
        avg_group = float(np.mean([group_items[mapping[t]] for t in targets]))
        merge_rows.append({
            "ratio": ratio,
            "merged_items": int(n_items - sum(len(sid2items[s]) for s in groups)),
            "unique_sids": len(groups),
            "collision": 1 - len(groups) / n_items,
            "max_group": int(max(group_items.values())),
            "avg_target_group": avg_group,
            **{k: m[k] for k in ["HR@1", "HR@10", "HR@50", "NDCG@50"]},
        })
        print("    ratio", ratio, merge_rows[-1], flush=True)

    # save
    out = {
        "n": n,
        "base_metrics": base_metrics,
        "collapse": collapse,
        "prefix": prefix_stats,
        "coverage": coverage,
        "beam_union_2ep_4ep": beam_union,
        "merge_sweep": merge_rows,
        "rank_hist": {
            "hit_rank_1": sum(1 for r in ranks2 if r == 0) / n,
            "hit_rank_le_10": sum(1 for r in ranks2 if r is not None and r < 10) / n,
            "hit_rank_le_50": sum(1 for r in ranks2 if r is not None and r < 50) / n,
        },
    }
    json.dump(out, open(os.path.join(RESULTS, "audit_candidates.json"), "w"), indent=2)
    pd.DataFrame(merge_rows).to_csv(os.path.join(RESULTS, "audit_merge_sweep.csv"), index=False)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
