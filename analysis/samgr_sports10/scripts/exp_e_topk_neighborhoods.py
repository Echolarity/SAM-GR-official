"""Top-K neighbourhood analysis across collaborative / text / image / joint spaces.

Outputs
-------
results/neighbors_spaces.npz                  Top-50 neighbours and cosines
results/exp_e_pair_overlap.csv                pairwise Jaccard + recall overlap
results/exp_e_neighborhood_metrics.csv        quality metrics per space x K
results/exp_e_neighborhood_metrics_by_group.csv  same, split by head/mid/tail
results/exp_e_per_item.npz                    per-item metric arrays
results/exp_e_data_stats.json                 dataset / graph statistics
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, l2n, load_collab, load_emb, load_items,  # noqa: E402
                    topk_neighbors)

KS = [10, 20, 50]
SPACES = ["collab", "text", "image", "joint"]
PAIRS = [
    ("collab", "text"),
    ("collab", "image"),
    ("collab", "joint"),
    ("text", "image"),
    ("text", "joint"),
    ("image", "joint"),
]
SEED = 42
DATA = "/data02/sht/SAM-GR/data/Amazon/Sports_and_Outdoors"


def load_train_sequences():
    df = pd.read_csv(os.path.join(DATA, "Sports_and_Outdoors.train.inter"), sep="\t")
    df.columns = ["user", "seq", "target"]
    seqs = []
    user_items = defaultdict(set)
    n_inter = np.zeros(0, dtype=np.int64)
    for u, s, t in df.itertuples(index=False):
        items = [int(x) for x in str(s).split()] + [int(t)]
        seqs.append(items)
        user_items[int(u)].update(items)
    return seqs, user_items


def build_graphs(seqs, user_items, n_items):
    users = sorted(user_items)
    uidx = {u: i for i, u in enumerate(users)}
    rows, cols = [], []
    for u in users:
        for it in user_items[u]:
            rows.append(uidx[u])
            cols.append(it)
    data = np.ones(len(rows), dtype=np.float32)
    U = csr_matrix((data, (rows, cols)), shape=(len(users), n_items), dtype=np.float32)
    n_user_i = np.asarray(U.sum(axis=0)).ravel().astype(np.int32)
    C = np.asarray((U.T @ U).toarray(), dtype=np.float32)
    np.rint(C, out=C)
    C = C.astype(np.int32)
    np.fill_diagonal(C, 0)

    T = np.zeros((n_items, n_items), dtype=np.int32)
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b:
                T[a, b] += 1
    T_out = T.sum(axis=1).astype(np.int64)
    return C, T, T_out, n_user_i, len(users)


def category_arrays(items, n_items):
    leaf_names, l2_names = [], []
    for i in range(n_items):
        cats = [c.strip() for c in (items[str(i)].get("categories") or "").split(",") if c.strip()]
        leaf_names.append(cats[-1] if cats else "Unknown")
        l2_names.append(cats[2] if len(cats) >= 3 else (cats[-1] if cats else "Unknown"))
    leaf_uniq = {c: i for i, c in enumerate(sorted(set(leaf_names)))}
    l2_uniq = {c: i for i, c in enumerate(sorted(set(l2_names)))}
    leaf = np.array([leaf_uniq[c] for c in leaf_names], dtype=np.int32)
    l2 = np.array([l2_uniq[c] for c in l2_names], dtype=np.int32)
    return leaf, l2, leaf_names, l2_names


def popularity_groups(n_user_i):
    n = len(n_user_i)
    order = np.argsort(-n_user_i, kind="stable")
    sizes = [n // 3, n // 3, n - 2 * (n // 3)]
    groups = np.empty(n, dtype=np.int8)
    specs = []
    start = 0
    for name, sz in zip(["head", "mid", "tail"], sizes):
        idx = order[start:start + sz]
        groups[idx] = {"head": 0, "mid": 1, "tail": 2}[name]
        specs.append({
            "group": name, "n_items": int(sz),
            "pop_min": int(n_user_i[idx].min()), "pop_max": int(n_user_i[idx].max()),
            "pop_mean": float(n_user_i[idx].mean()),
            "pop_median": float(np.median(n_user_i[idx])),
            "user_interactions": int(n_user_i[idx].sum()),
        })
        start += sz
    total = max(int(n_user_i.sum()), 1)
    for s in specs:
        s["interaction_share"] = s["user_interactions"] / total
    return groups, specs



def random_neighbour_sets(n, k, n_user_i, rng, n_draws=3):
    """Uniform and popularity-matched random neighbour controls (n_draws averaged)."""
    sets = {"uniform": [], "pop": []}
    p = n_user_i.astype(np.float64)
    p = p / p.sum()
    cdf = np.cumsum(p)
    for _ in range(n_draws):
        u = rng.integers(0, n, size=(n, k))
        r = np.searchsorted(cdf, rng.random((n, k)), side="right")
        for mat in (u, r):
            bad = mat == np.arange(n)[:, None]
            if bad.any():
                mat[bad] = rng.integers(0, n, size=int(bad.sum()))
        sets["uniform"].append(u)
        sets["pop"].append(r)
    return sets


def average_metrics(metric_list):
    out = {}
    for name in metric_list[0]:
        out[name] = np.nanmean(np.stack([m[name] for m in metric_list], axis=0), axis=0).astype(np.float32)
    return out

def pair_metrics_for_items(nb, C, T, T_out, n_user_i, n_users, leaf, l2,
                           text, image):
    """Per-item means for one (space, K) neighbourhood matrix."""
    n, k = nb.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nb.reshape(-1).astype(np.int64)
    c = C[rows, cols].astype(np.float64)
    ni = n_user_i[rows].astype(np.float64)
    nj = n_user_i[cols].astype(np.float64)
    p_cond = c / np.maximum(ni, 1.0)

    t = T[rows, cols].astype(np.float64)
    tout = T_out[rows].astype(np.float64)
    p_trans = np.where(tout[:, None].reshape(-1) > 0, t / np.maximum(tout, 1.0), np.nan)

    p_ij = c / max(n_users, 1)
    pi = ni / max(n_users, 1)
    pj = nj / max(n_users, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.log(p_ij / np.maximum(pi * pj, 1e-30)) / (-np.log(np.maximum(p_ij, 1e-30)))
    npmi[c <= 0] = np.nan

    p_ij_s = (c + 0.5) / (n_users + 1.0)
    pi_s = (ni + 0.5) / (n_users + 1.0)
    pj_s = (nj + 0.5) / (n_users + 1.0)
    npmi_s = np.log(p_ij_s / np.maximum(pi_s * pj_s, 1e-30)) / (-np.log(np.maximum(p_ij_s, 1e-30)))

    def reshape(x):
        return x.reshape(n, k)

    p_cond = reshape(p_cond)
    p_trans = reshape(p_trans)
    npmi = reshape(npmi)
    npmi_s = reshape(npmi_s)
    c = reshape(c)
    t = reshape(t)

    # category agreement with the anchor
    same_leaf = (leaf[nb] == leaf[:, None]).mean(axis=1)
    same_l2 = (l2[nb] == l2[:, None]).mean(axis=1)
    # dominant category share inside the neighbour set
    dom_leaf = np.empty(n, dtype=np.float32)
    for i in range(n):
        dom_leaf[i] = np.bincount(leaf[nb[i]], minlength=leaf.max() + 1).max() / k

    # text / image cosine between anchor and each neighbour
    def mean_cos(x):
        out = np.empty(n, dtype=np.float32)
        for st in range(0, n, 64):
            en = min(st + 64, n)
            b = np.arange(st, en)
            out[b] = np.einsum("bkd,bd->bk", x[nb[b]], x[b], optimize=True).mean(axis=1)
        return out

    text_cos = mean_cos(text)
    image_cos = mean_cos(image)

    return {
        "cat_purity_leaf": same_leaf.astype(np.float32),
        "cat_purity_l2": same_l2.astype(np.float32),
        "cat_majority_leaf": dom_leaf,
        "cooc_prob": p_cond.mean(axis=1).astype(np.float32),
        "cooc_cov": (c > 0).mean(axis=1).astype(np.float32),
        "trans_prob": np.nanmean(p_trans, axis=1).astype(np.float32),
        "trans_cov": (t > 0).mean(axis=1).astype(np.float32),
        "npmi": np.nanmean(npmi, axis=1).astype(np.float32),
        "npmi_cov": (~np.isnan(npmi)).mean(axis=1).astype(np.float32),
        "npmi_smooth": npmi_s.mean(axis=1).astype(np.float32),
        "text_cos": text_cos,
        "image_cos": image_cos,
    }


def summarise(metric_arrays, groups, n_user_i):
    rows = []
    for group in ["all", "head", "mid", "tail"]:
        m = np.ones(len(n_user_i), dtype=bool) if group == "all" else groups == {"head": 0, "mid": 1, "tail": 2}[group]
        row = {"group": group, "n_items": int(m.sum())}
        for name, arr in metric_arrays.items():
            v = arr[m]
            row[name] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[name + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        rows.append(row)
    return rows


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    items = load_items()
    n = len(items)
    print(f"items={n}")

    reps = {
        "collab": l2n(load_collab()),
        "text": l2n(load_emb("text_only")),
        "image": l2n(load_emb("image_only")),
        "joint": l2n(load_emb("multimodal")),
    }
    for k, v in reps.items():
        assert v.shape == (n, v.shape[1]), (k, v.shape)
        print(k, v.shape)

    # Top-50 neighbours, self excluded.
    nb, cos = {}, {}
    for s in SPACES:
        nb[s], cos[s] = topk_neighbors(reps[s], max(KS))
        print("neighbours", s, nb[s].shape, f"{time.time()-t0:.1f}s")
    np.savez_compressed(
        os.path.join(RESULTS, "neighbors_spaces.npz"),
        **{f"{s}_idx": nb[s] for s in SPACES},
        **{f"{s}_cos": cos[s] for s in SPACES},
    )

    # ---------- pairwise overlap ----------
    overlap_rows = []
    for a, b in PAIRS:
        for k in KS:
            inter = np.array([len(np.intersect1d(nb[a][i, :k], nb[b][i, :k])) for i in range(n)], dtype=np.float32)
            union = 2 * k - inter
            jac = inter / np.maximum(union, 1)
            rec = inter / k
            overlap_rows.append({
                "space_a": a, "space_b": b, "pair": f"{a}-{b}", "k": k,
                "jaccard_mean": float(jac.mean()), "jaccard_std": float(jac.std()),
                "jaccard_median": float(np.median(jac)),
                "recall_overlap_mean": float(rec.mean()), "recall_overlap_std": float(rec.std()),
                "recall_overlap_median": float(np.median(rec)),
                "zero_frac": float((inter == 0).mean()),
            })
    ov = pd.DataFrame(overlap_rows)
    ov.to_csv(os.path.join(RESULTS, "exp_e_pair_overlap.csv"), index=False)
    print(ov.to_string(index=False))

    # ---------- graphs, categories, popularity ----------
    print("loading train sequences ...")
    seqs, user_items = load_train_sequences()
    C, T, T_out, n_user_i, n_users = build_graphs(seqs, user_items, n)
    leaf, l2, leaf_names, l2_names = category_arrays(items, n)
    groups, group_specs = popularity_groups(n_user_i)
    pd.DataFrame(group_specs).to_csv(os.path.join(RESULTS, "exp_e_popularity_groups.csv"), index=False)
    print("users", n_users, "cooc nnz", int((C > 0).sum()), "transitions", int(T.sum()))
    print(pd.DataFrame(group_specs).to_string(index=False))

    # ---------- random-neighbour controls ----------
    rand_metrics = {}
    for k in KS:
        rand_sets = random_neighbour_sets(n, k, n_user_i, rng, n_draws=3)
        for ctrl, mats in rand_sets.items():
            ms = [pair_metrics_for_items(mat, C, T, T_out, n_user_i, n_users,
                                         leaf, l2, reps["text"], reps["image"])
                  for mat in mats]
            rand_metrics[(k, ctrl)] = average_metrics(ms)
        print("random controls", k, f"{time.time()-t0:.1f}s")

    # ---------- neighbourhood quality ----------
    metric_rows = []
    per_item = {}
    for s in SPACES:
        for k in KS:
            m = pair_metrics_for_items(nb[s][:, :k], C, T, T_out, n_user_i, n_users,
                                       leaf, l2, reps["text"], reps["image"])
            key = f"{s}_k{k}"
            for name, arr in m.items():
                per_item[f"{key}_{name}"] = arr
            for ctrl in ["uniform", "pop"]:
                rm = rand_metrics[(k, ctrl)]
                for name, arr in rm.items():
                    per_item[f"{key}_rand_{ctrl}_{name}"] = arr
                    m[f"rand_{ctrl}_{name}"] = arr
                # additive deltas for bounded / similarity metrics
                for name in ["cat_purity_leaf", "cat_purity_l2", "cat_majority_leaf",
                             "npmi", "npmi_smooth", "text_cos", "image_cos"]:
                    m[f"delta_{ctrl}_{name}"] = m[name] - rm[name]
            for row in summarise(m, groups, n_user_i):
                for ctrl in ["uniform", "pop"]:
                    # aggregate ratio of means is the stable lift; per-item ratios
                    # are dominated by zero random denominators.
                    for name in ["cooc_prob", "trans_prob"]:
                        row[f"lift_{ctrl}_{name}_agg"] = row[name] / max(row[f"rand_{ctrl}_{name}"], 1e-12)
                row.update({"space": s, "k": k})
                metric_rows.append(row)
            print("metrics", s, k, f"{time.time()-t0:.1f}s")

    df = pd.DataFrame(metric_rows)
    cols = ["space", "k", "group", "n_items",
            "cat_purity_leaf", "cat_purity_leaf_std", "cat_purity_l2", "cat_majority_leaf",
            "cooc_prob", "cooc_prob_std", "cooc_cov",
            "trans_prob", "trans_prob_std", "trans_cov",
            "npmi", "npmi_std", "npmi_cov", "npmi_smooth",
            "text_cos", "text_cos_std", "image_cos", "image_cos_std"]
    extra = []
    for ctrl in ["uniform", "pop"]:
        extra += [f"rand_{ctrl}_cat_purity_leaf", f"rand_{ctrl}_cooc_prob",
                  f"rand_{ctrl}_trans_prob", f"rand_{ctrl}_npmi",
                  f"rand_{ctrl}_text_cos", f"rand_{ctrl}_image_cos",
                  f"delta_{ctrl}_cat_purity_leaf", f"delta_{ctrl}_text_cos",
                  f"delta_{ctrl}_image_cos", f"lift_{ctrl}_cooc_prob_agg",
                  f"lift_{ctrl}_trans_prob_agg"]
    cols += extra
    df = df[cols]
    df.to_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics.csv"), index=False)
    df[df.group != "all"].to_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics_by_group.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, "exp_e_per_item.npz"), **per_item)

    stats = {
        "n_items": n, "n_users": n_users, "n_train_rows": len(seqs),
        "cooc_nonzero_pairs": int((C > 0).sum()), "transition_count": int(T.sum()),
        "leaf_categories": len(leaf_names and set(leaf_names)),
        "l2_categories": len(set(l2_names)),
        "ks": KS, "spaces": SPACES,
        "pair_overlap": overlap_rows,
        "popularity_groups": group_specs,
    }
    with open(os.path.join(RESULTS, "exp_e_data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(df[df.group == "all"].to_string(index=False))
    print("EXP_E_DONE", f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
