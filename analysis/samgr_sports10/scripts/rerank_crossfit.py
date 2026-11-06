"""Phase 1a: 5-fold cross-fit diagnostic for candidate reranking on Sports test beams.

No model inference; only saved beams + embeddings.  Each sample is predicted by a
reranker trained on the other 4 folds, so the estimate is out-of-fold (diagnostic,
not the final valid-trained result).

Recipes:
  safe       : beam(2ep+4ep) + CF20 + text20 + multimodal20 + popularity20
  aggressive : safe + a+b prefix expansion

Outputs: results/rerank_crossfit.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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


def prefixes(s, lv):
    return "".join(TOK_RE.findall(s)[:lv])


def sim_matrix(x, rows, chunk=512):
    xa = l2n(x)
    out = np.empty((len(rows), len(x)), dtype=np.float32)
    for st in range(0, len(rows), chunk):
        en = min(st + chunk, len(rows))
        out[st:en] = l2n(x[rows[st:en]]) @ xa.T
    return out


def topk_from_sim(s, k):
    # s: (n_rows, n_items); self is irrelevant for the candidate scores
    part = np.argpartition(-s, k, axis=1)[:, :k]
    vals = np.take_along_axis(s, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1)


def eval_ranks(ranks, n):
    out = {}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r < k)
        dcg = sum(1.0 / math.log2(r + 2) for r in ranks if r is not None and r < k)
        out[f"HR@{k}"] = hits / n
        out[f"NDCG@{k}"] = dcg / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["safe", "aggressive", "both"], default="both")
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args()
    print("load data", flush=True)
    id2sid = json.load(open(ID2SID, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in id2sid.items()}
    n_items = len(item2sid)
    sid2items = collections.defaultdict(list)
    for i, s in item2sid.items():
        sid2items[s].append(i)
    sids = sorted(sid2items)
    n = len(json.load(open(RES2, encoding="utf-8")))
    d2 = json.load(open(RES2, encoding="utf-8"))
    d4 = json.load(open(RES4, encoding="utf-8"))

    pop = np.zeros(n_items, dtype=np.float64)
    with open(TRAIN_INTER, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                pop[int(p[2])] += 1
    sid_pop = {s: float(sum(pop[i] for i in items)) for s, items in sid2items.items()}
    pop_sids = [s for s, _ in sorted(sid_pop.items(), key=lambda kv: -kv[1])[:50]]

    beam2, beam4, targets, last_sids = [], [], [], []
    for x in d2:
        beam2.append([str(v).strip().strip('"') for v in x["predict"]])
        targets.append(str(x["output"]).strip().strip('"'))
        hist = SID_RE.findall(str(x["input"]))
        last_sids.append(hist[-1] if hist else pop_sids[0])
    for x in d4:
        beam4.append([str(v).strip().strip('"') for v in x["predict"]])

    last_items = sorted({max(sid2items[ls], key=lambda i: pop[i]) for ls in last_sids})
    lindex = {i: k for k, i in enumerate(last_items)}
    print(f"n={n} last_items={len(last_items)}", flush=True)

    print("similarity matrices", flush=True)
    collab = np.load(COLLAB).astype(np.float32)
    S_cf = sim_matrix(collab, last_items)
    cf_top = topk_from_sim(S_cf, 20)
    S_tx = sim_matrix(np.load(os.path.join(AMZ, "Sports_and_Outdoors.emb-Qwen3VL-text_only.npy"),
                              mmap_mode="r").astype(np.float32), last_items)
    tx_top = topk_from_sim(S_tx, 20)
    S_mm = sim_matrix(np.load(os.path.join(AMZ, "Sports_and_Outdoors.emb-Qwen3VL-multimodal.npy"),
                              mmap_mode="r").astype(np.float32), last_items)
    mm_top = topk_from_sim(S_mm, 20)
    print("similarity done", flush=True)

    pref2 = collections.defaultdict(list)
    for s in sids:
        pref2[prefixes(s, 2)].append(s)
    sid_group = {s: len(v) for s, v in sid2items.items()}

    feature_names = [
        "rank2", "rank4", "in2", "in4", "p1_count", "p2_count", "p3_count",
        "p2_best_rank", "p3_best_rank", "cf_score", "tx_score", "mm_score",
        "pop_log", "group_size", "src_beam", "src_prefix", "src_cf", "src_tx",
        "src_mm", "src_pop", "n_sources",
    ]

    def build(use_prefix):
        X, y, ranges = [], [], [None] * n
        cands_per = [None] * n
        coverage = 0
        for idx, (p2, p4, t, ls) in enumerate(zip(beam2, beam4, targets, last_sids)):
            b2set, b4set = set(p2), set(p4)
            beamset = b2set | b4set
            r2 = {s: i for i, s in enumerate(p2)}
            r4 = {s: i for i, s in enumerate(p4)}
            # prefix stats from 2ep beam
            p1c = collections.Counter(prefixes(s, 1) for s in p2)
            p2c = collections.Counter(prefixes(s, 2) for s in p2)
            p3c = collections.Counter(prefixes(s, 3) for s in p2)
            p2best, p3best = {}, {}
            for i, s in enumerate(p2):
                p2best[prefixes(s, 2)] = min(p2best.get(prefixes(s, 2), 999), i)
                p3best[prefixes(s, 3)] = min(p3best.get(prefixes(s, 3), 999), i)
            pref2_beam = set(p2c)
            prefix_cands = set()
            if use_prefix:
                for q in pref2_beam:
                    prefix_cands.update(pref2[q])
            row = lindex[max(sid2items[ls], key=lambda i: pop[i])]
            cf = {item2sid[int(j)] for j in cf_top[row]}
            tx = {item2sid[int(j)] for j in tx_top[row]}
            mm = {item2sid[int(j)] for j in mm_top[row]}
            pp = set(pop_sids[:20])
            cand = beamset | cf | tx | mm | pp | prefix_cands
            if t not in cand:
                continue
            coverage += 1
            cands = sorted(cand)
            cands_per[idx] = cands
            start = len(X)
            for s in cands:
                q1, q2, q3 = prefixes(s, 1), prefixes(s, 2), prefixes(s, 3)
                items = sid2items[s]
                cf_s = float(max(S_cf[row, i] for i in items))
                tx_s = float(max(S_tx[row, i] for i in items))
                mm_s = float(max(S_mm[row, i] for i in items))
                src_beam = int(s in beamset)
                src_pref = int(s in prefix_cands)
                src_cf = int(s in cf)
                src_tx = int(s in tx)
                src_mm = int(s in mm)
                src_pop = int(s in pp)
                X.append([
                    r2.get(s, 999), r4.get(s, 999), int(s in b2set), int(s in b4set),
                    p1c.get(q1, 0), p2c.get(q2, 0), p3c.get(q3, 0),
                    p2best.get(q2, 999), p3best.get(q3, 999), cf_s, tx_s, mm_s,
                    math.log1p(sid_pop[s]), sid_group[s],
                    src_beam, src_pref, src_cf, src_tx, src_mm, src_pop,
                    src_beam + src_pref + src_cf + src_tx + src_mm + src_pop,
                ])
                y.append(1 if s == t else 0)
            ranges[idx] = (start, len(X))
        return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
                ranges, cands_per, coverage)

    out_path = os.path.join(RESULTS, "rerank_crossfit.json")
    all_res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [("safe", False), ("aggressive", True)]
    if args.recipe != "both":
        todo = [(nm, up) for nm, up in todo if nm == args.recipe]
    for name, use_prefix in todo:
        print("build", name, flush=True)
        X, y, ranges, cands_per, coverage = build(use_prefix)
        print(f"  rows={len(X)} samples_with_target={coverage} coverage={coverage / n:.4f}", flush=True)
        sample_ids = np.array([i for i, r in enumerate(ranges) if r is not None], dtype=np.int32)
        row_sample = np.empty(len(X), dtype=np.int32)
        for i, r in enumerate(ranges):
            if r is not None:
                row_sample[r[0]:r[1]] = i
        fold_of_sample = {sid: k % 5 for k, sid in enumerate(sample_ids)}
        row_fold = np.array([fold_of_sample[s] for s in row_sample], dtype=np.int8)

        ranks = [None] * n
        ranks_llm = [None] * n
        ranks_rrf = [None] * n
        for f in range(5):
            tr = row_fold != f
            te = row_fold == f
            clf = HistGradientBoostingClassifier(
                max_iter=args.max_iter, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0,
                random_state=0, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            te_cum = np.cumsum(te)
            for sid in [s for s in sample_ids if fold_of_sample[s] == f]:
                a, b = ranges[sid]
                ps = int(te_cum[a - 1]) if a > 0 else 0
                pe = int(te_cum[b - 1])
                pr = prob[ps:pe]
                cand = cands_per[sid]
                order = np.argsort(-pr)
                t = targets[sid]
                for k, oi in enumerate(order):
                    if cand[oi] == t:
                        ranks[sid] = k
                        break
                r2 = {c: k for k, c in enumerate(beam2[sid])}
                r4 = {c: k for k, c in enumerate(beam4[sid])}
                sc = np.array([
                    (1.0 / (60 + r2[c]) if c in r2 else 0.0) +
                    (1.0 / (60 + r4[c]) if c in r4 else 0.0)
                    for c in cand
                ])
                rrf = np.argsort(-sc)
                for k, oi in enumerate(rrf):
                    if cand[oi] == t:
                        ranks_rrf[sid] = k
                        break
                ranks_llm[sid] = r2.get(t, None)
        m = eval_ranks(ranks, n)
        m_rrf = eval_ranks(ranks_rrf, n)
        m_llm = eval_ranks(ranks_llm, n)
        all_res[name] = {
            "coverage": coverage / n,
            "mean_cands": float(np.mean([len(c) for c in cands_per if c])),
            "rows": int(len(X)),
            "reranker": m,
            "rrf_beam_union": m_rrf,
            "llm_beam2": m_llm,
        }
        print(name, "coverage", round(coverage / n, 4),
              "rerank HR@1/50", round(m["HR@1"], 4), round(m["HR@50"], 4),
              "rrf HR@50", round(m_rrf["HR@50"], 4), flush=True)
        json.dump(all_res, open(out_path, "w"), indent=2)
    print("CROSSFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
