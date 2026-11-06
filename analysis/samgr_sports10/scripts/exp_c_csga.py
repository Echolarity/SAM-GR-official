"""Experiment C: CSGA neighbourhood design at the representation level.

We cannot retrain the LLM for each CSGA variant in the available budget; instead we
compare the *representations* produced by (a) no aggregation, (b) collaborative
neighbour aggregation (CSGA), (c) semantic neighbour aggregation, (d) random
neighbour aggregation, using the same multimodal semantic feature and the same
aggregation form h = m + 0.5 * mean(neighbours).  Downstream retraining is
explicitly reported as not run.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FIGURES, RESULTS, l2n, load_collab, load_emb,  # noqa: E402
                    overlap_at_k, topk_neighbors)

K = 10
ALPHA = 0.5


def agg(m, nb, alpha=ALPHA):
    a = m[nb].mean(axis=1)
    return l2n(m + alpha * a)


def main():
    m = l2n(load_emb("multimodal"))
    collab = l2n(load_collab())
    n = len(m)
    rng = np.random.default_rng(42)

    collab_nb, _ = topk_neighbors(collab, K)
    sem_nb, _ = topk_neighbors(m, K)
    rand_nb = rng.integers(0, n, size=(n, K))
    # avoid self-loops in random neighbours
    for i in range(n):
        rand_nb[i][rand_nb[i] == i] = (i + 1) % n

    variants = {
        "no aggregation": m,
        "collaborative (CSGA)": agg(m, collab_nb),
        "semantic neighbours": agg(m, sem_nb),
        "random neighbours": agg(m, rand_nb),
    }

    # clustering proxy on a fixed sample
    sample = np.sort(rng.choice(n, size=min(3000, n), replace=False))
    rows = []
    for name, x in variants.items():
        cos_m = float(np.sum(x * m, axis=1).mean())
        ov_c = float(overlap_at_k(topk_neighbors(x, K)[0], collab_nb, K).mean())
        ov_s = float(overlap_at_k(topk_neighbors(x, K)[0], sem_nb, K).mean())
        km = MiniBatchKMeans(n_clusters=256, random_state=42, n_init=3, batch_size=1024)
        km.fit(x[sample])
        # normalized inertia (per-sample, per-dim) for comparability
        inertia = float(km.inertia_ / len(sample))
        rows.append({"variant": name, "cos_to_semantic": cos_m,
                     "overlap_with_collab_nb": ov_c, "overlap_with_semantic_nb": ov_s,
                     "kmeans_inertia": inertia})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_c_csga_variants.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    labels = df["variant"].tolist()
    for ax, col, title in zip(
        axes,
        ["cos_to_semantic", "overlap_with_collab_nb", "kmeans_inertia"],
        ["cosine to original semantic feature", "neighbour overlap with collaborative graph",
         "k-means inertia (lower = more compact)"]):
        ax.bar(range(len(labels)), df[col], color=["#4C72B0", "#C44E52", "#55A868", "#8172B2"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_c_csga_variants.png"), dpi=200)
    plt.close(fig)
    print("EXP_C_DONE")


if __name__ == "__main__":
    main()
