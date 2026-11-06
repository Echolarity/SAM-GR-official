"""Experiment A: compare collaborative / textual / visual neighbourhood structures."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FIGURES, RESULTS, category_labels, jaccard_at_k, l2n,  # noqa: E402
                    load_collab, load_emb, load_items, overlap_at_k, topk_neighbors)

KS = [5, 10, 20, 50]
PAIRS = [("collab", "text"), ("collab", "visual"), ("text", "visual")]


def main():
    items = load_items()
    reps = {
        "collab": l2n(load_collab()),
        "text": l2n(load_emb("text_only")),
        "visual": l2n(load_emb("image_only")),
    }
    for name, x in reps.items():
        print(f"{name}: {x.shape}")
    n = len(reps["collab"])
    for name, x in reps.items():
        assert len(x) == n, f"{name} rows {len(x)} != {n} items"
        assert np.isfinite(x).all(), f"{name} has non-finite values"

    nb = {name: topk_neighbors(x, max(KS))[0] for name, x in reps.items()}

    rows = []
    for a, b in PAIRS:
        for k in KS:
            ov = overlap_at_k(nb[a], nb[b], k)
            jc = jaccard_at_k(nb[a], nb[b], k)
            rows.append({"rep_a": a, "rep_b": b, "k": k,
                         "overlap_mean": float(ov.mean()), "overlap_std": float(ov.std()),
                         "jaccard_mean": float(jc.mean()), "jaccard_std": float(jc.std())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_a_neighborhood_overlap.csv"), index=False)

    # rank correlation on random item pairs
    rng = np.random.default_rng(42)
    m = 30000
    ia = rng.integers(0, n, m)
    ib = rng.integers(0, n, m)
    valid = ia != ib
    ia, ib = ia[valid], ib[valid]
    corr_rows = []
    for a, b in PAIRS:
        sa = np.sum(reps[a][ia] * reps[a][ib], axis=1)
        sb = np.sum(reps[b][ia] * reps[b][ib], axis=1)
        rho, _ = spearmanr(sa, sb)
        corr_rows.append({"rep_a": a, "rep_b": b, "spearman": float(rho), "pairs": len(ia)})
    pd.DataFrame(corr_rows).to_csv(os.path.join(RESULTS, "exp_a_rank_correlation.csv"), index=False)

    # ---------- t-SNE (per representation + joint) ----------
    labels = category_labels(items)
    top = pd.Series(labels).value_counts().head(8).index.tolist()

    def colorize(ax, lab, title):
        cmap = plt.get_cmap("tab10")
        for j, c in enumerate(top):
            idx = [i for i, l in enumerate(lab) if l == c]
            if idx:
                ax.scatter(tsne_pool[idx, 0], tsne_pool[idx, 1], s=6, alpha=0.75,
                           color=cmap(j % 10), label=c[:22])
        other = [i for i, l in enumerate(lab) if l not in top]
        if other:
            ax.scatter(tsne_pool[other, 0], tsne_pool[other, 1], s=6, alpha=0.35,
                       color="lightgray", label="Other")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    sample = np.sort(rng.choice(n, size=min(1500, n), replace=False))
    lab_s = [labels[i] for i in sample]
    for name, x in reps.items():
        xs = PCA(n_components=50, random_state=42).fit_transform(x[sample])
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
                   max_iter=1000).fit_transform(xs)
        tsne_pool = emb
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        colorize(ax, lab_s, f"t-SNE: {name} (n={len(sample)})")
        ax.legend(fontsize=7, loc="best", frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, f"fig_a_tsne_{name}.png"), dpi=200)
        plt.close(fig)

    # joint: z-score each space, PCA 50 each, concat
    parts = []
    for name, x in reps.items():
        z = (x[sample] - x[sample].mean(0, keepdims=True)) / (x[sample].std(0, keepdims=True) + 1e-9)
        parts.append(PCA(n_components=50, random_state=42).fit_transform(z))
    joint = np.concatenate(parts, axis=1)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
               max_iter=1000).fit_transform(joint)
    tsne_pool = emb
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colorize(ax, lab_s, "t-SNE: joint (z-scored, PCA-50 each, concat)")
    ax.legend(fontsize=7, loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_a_tsne_joint.png"), dpi=200)
    plt.close(fig)

    # ---------- overlap bar chart ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width = 0.26
    xs = np.arange(len(KS))
    for j, (a, b) in enumerate(PAIRS):
        vals = [df[(df.rep_a == a) & (df.rep_b == b) & (df.k == k)].overlap_mean.iloc[0] for k in KS]
        ax.bar(xs + (j - 1) * width, vals, width, label=f"{a} vs {b}")
    ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_ylabel("mean neighbour overlap")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Top-K neighbourhood overlap across representation spaces", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_a_overlap_bar.png"), dpi=200)
    plt.close(fig)

    # ---------- overlap distributions at K=10 ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for a, b in PAIRS:
        ov = overlap_at_k(nb[a], nb[b], 10)
        ax.hist(ov, bins=25, alpha=0.55, label=f"{a} vs {b}")
    ax.set_xlabel("per-item overlap (K=10)"); ax.set_ylabel("#items")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Distribution of neighbour overlap (K=10)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_a_overlap_dist.png"), dpi=200)
    plt.close(fig)

    print(df.to_string(index=False))
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print("EXP_A_DONE")


if __name__ == "__main__":
    main()
