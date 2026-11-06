"""Train candidate reranker on valid beams, evaluate once on test beams.

Clean protocol:
  - reranker fit on valid labels only
  - popularity / CF / text / multimodal neighbours from train split only
  - test labels used once for final metrics

Recipe: beam(2ep+4ep) + CF20 + text20 + multimodal20 + popularity20
(the "safe" recipe; a+b prefix expansion gave no cross-fit gain)
"""
from __future__ import annotations

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
R = os.path.join(BASE, "results", "qwen2.5-0.5b", "full")
TEST2 = os.path.join(R, "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict", "final_result.json")
TEST4 = os.path.join(R, "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_ep4", "final_result.json")
VALID2 = os.path.join(R, "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_valid", "final_result.json")
VALID4 = os.path.join(R, "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_ep4_valid", "final_result.json")
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


def load_beams(path):
    d = json.load(open(path, encoding="utf-8"))
    beams, targets, last_sids = [], [], []
    for x in d:
        beams.append([str(v).strip().strip('"') for v in x["predict"]])
        targets.append(str(x["output"]).strip().strip('"'))
        hist = SID_RE.findall(str(x["input"]))
        last_sids.append(hist[-1] if hist else "")
    return beams, targets, last_sids


def main():
    print("load", flush=True)
    id2sid = json.load(open(ID2SID, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in id2sid.items()}
    n_items = len(item2sid)
    sid2items = collections.defaultdict(list)
    for i, s in item2sid.items():
        sid2items[s].append(i)
    sids = sorted(sid2items)
    sid_pop = {}
    pop = np.zeros(n_items, dtype=np.float64)
    with open(TRAIN_INTER, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                pop[int(p[2])] += 1
    for s, items in sid2items.items():
        sid_pop[s] = float(sum(pop[i] for i in items))
    pop_sids = [s for s, _ in sorted(sid_pop.items(), key=lambda kv: -kv[1])[:50]]

    v2, vt, vls = load_beams(VALID2)
    v4, _, _ = load_beams(VALID4)
    t2, tt, tls = load_beams(TEST2)
    t4, _, _ = load_beams(TEST4)
    n_valid, n_test = len(vt), len(tt)
    assert len(v2) == len(v4) == n_valid and len(t2) == len(t4) == n_test

    all_last = sorted({max(sid2items[ls], key=lambda i: pop[i])
                       for ls in (vls + tls) if ls})
    lindex = {i: k for k, i in enumerate(all_last)}
    print(f"valid={n_valid} test={n_test} last_items={len(all_last)}", flush=True)

    print("similarity matrices", flush=True)
    collab = np.load(COLLAB).astype(np.float32)
    S_cf = sim_matrix(collab, all_last)
    cf_top = topk_from_sim(S_cf, 20)
    S_tx = sim_matrix(np.load(os.path.join(AMZ, "Sports_and_Outdoors.emb-Qwen3VL-text_only.npy"),
                              mmap_mode="r").astype(np.float32), all_last)
    tx_top = topk_from_sim(S_tx, 20)
    S_mm = sim_matrix(np.load(os.path.join(AMZ, "Sports_and_Outdoors.emb-Qwen3VL-multimodal.npy"),
                              mmap_mode="r").astype(np.float32), all_last)
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
        "cf_pct", "tx_pct", "mm_pct", "pop_pct", "best_pct",
    ]

    def build(b2s, b4s, ts, lss):
        X, y, ranges = [], [], [None] * len(ts)
        cands_per = [None] * len(ts)
        coverage = 0
        for idx, (p2, p4, t, ls) in enumerate(zip(b2s, b4s, ts, lss)):
            b2set, b4set = set(p2), set(p4)
            beamset = b2set | b4set
            r2 = {s: i for i, s in enumerate(p2)}
            r4 = {s: i for i, s in enumerate(p4)}
            p1c = collections.Counter(prefixes(s, 1) for s in p2)
            p2c = collections.Counter(prefixes(s, 2) for s in p2)
            p3c = collections.Counter(prefixes(s, 3) for s in p2)
            p2best, p3best = {}, {}
            for i, s in enumerate(p2):
                p2best[prefixes(s, 2)] = min(p2best.get(prefixes(s, 2), 999), i)
                p3best[prefixes(s, 3)] = min(p3best.get(prefixes(s, 3), 999), i)
            row = lindex[max(sid2items[ls], key=lambda i: pop[i])]
            cf = {item2sid[int(j)] for j in cf_top[row]}
            tx = {item2sid[int(j)] for j in tx_top[row]}
            mm = {item2sid[int(j)] for j in mm_top[row]}
            pp = set(pop_sids[:20])
            cand = beamset | cf | tx | mm | pp
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
                src_pref = 0
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
            block = X[start:]
            def pct(v):
                v = np.asarray(v, dtype=np.float64)
                order = np.argsort(np.argsort(-v))
                return order / (len(v) - 1 + 1e-9)
            cfp = pct([r[9] for r in block])
            txp = pct([r[10] for r in block])
            mmp = pct([r[11] for r in block])
            popp = pct([r[12] for r in block])
            for r, a, b, c, d in zip(block, cfp, txp, mmp, popp):
                r.extend([float(a), float(b), float(c), float(d), float(max(a, b, c))])
            ranges[idx] = (start, len(X))
        return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
                ranges, cands_per, coverage)

    print("build valid", flush=True)
    Xv, yv, rv, cv, covv = build(v2, v4, vt, vls)
    print(f"valid rows={len(Xv)} coverage={covv / n_valid:.4f} mean_cands={np.mean([len(c) for c in cv if c]):.1f}", flush=True)
    print("build test", flush=True)
    Xt, yt, rt, ct, covt = build(t2, t4, tt, tls)
    print(f"test rows={len(Xt)} coverage={covt / n_test:.4f} mean_cands={np.mean([len(c) for c in ct if c]):.1f}", flush=True)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        random_state=0, class_weight="balanced")
    print("fit", flush=True)
    clf.fit(Xv, yv)
    prob = clf.predict_proba(Xt)[:, 1]
    print("predict", flush=True)

    ranks, ranks_rrf, ranks_2, ranks_4 = [None] * n_test, [None] * n_test, [None] * n_test, [None] * n_test
    for sid, rng in enumerate(rt):
        if rng is None:
            continue
        a, b = rng
        pr = prob[a:b]
        cand = ct[sid]
        t = tt[sid]
        order = np.argsort(-pr)
        for k, oi in enumerate(order):
            if cand[oi] == t:
                ranks[sid] = k
                break
        r2 = {c: k for k, c in enumerate(t2[sid])}
        r4 = {c: k for k, c in enumerate(t4[sid])}
        ranks_2[sid] = r2.get(t)
        ranks_4[sid] = r4.get(t)
        sc = np.array([(1.0 / (60 + r2[c]) if c in r2 else 0.0) +
                       (1.0 / (60 + r4[c]) if c in r4 else 0.0) for c in cand])
        for k, oi in enumerate(np.argsort(-sc)):
            if cand[oi] == t:
                ranks_rrf[sid] = k
                break

    res = {
        "protocol": "train on valid, evaluate once on test; candidate sources from train split only",
        "recipe": "beam(2ep+4ep)+CF20+text20+multimodal20+popularity20",
        "valid": {
            "coverage": covv / n_valid,
            "mean_cands": float(np.mean([len(c) for c in cv if c])),
            "rows": int(len(Xv)),
        },
        "test": {
            "coverage": covt / n_test,
            "mean_cands": float(np.mean([len(c) for c in ct if c])),
            "rows": int(len(Xt)),
            "reranker": eval_ranks(ranks, n_test),
            "rrf_beam_union": eval_ranks(ranks_rrf, n_test),
            "llm_2ep": eval_ranks(ranks_2, n_test),
            "llm_4ep": eval_ranks(ranks_4, n_test),
        },
        "features": feature_names,
    }
    # valid LLM baselines (not used for fit)
    v_ranks_2 = [None] * n_valid
    v_ranks_4 = [None] * n_valid
    for sid, t in enumerate(vt):
        r2 = {c: k for k, c in enumerate(v2[sid])}
        r4 = {c: k for k, c in enumerate(v4[sid])}
        v_ranks_2[sid] = r2.get(t)
        v_ranks_4[sid] = r4.get(t)
    res["valid"]["llm_2ep"] = eval_ranks(v_ranks_2, n_valid)
    res["valid"]["llm_4ep"] = eval_ranks(v_ranks_4, n_valid)
    json.dump(res, open(os.path.join(RESULTS, "rerank_valid_to_test_pct.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("RERANK_VALID_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
