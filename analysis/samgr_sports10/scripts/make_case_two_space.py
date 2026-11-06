"""Two-space case study: for one anchor, show its text top-10 and collab top-10
neighbours inside the TEXT space and inside the COLLABORATIVE space.

This directly visualises the cross-space misalignment: in text space the text
neighbours are close to the anchor while the collaborative neighbours are not;
in collaborative space the opposite holds.  Each panel annotates the mean cosine
from the anchor to each neighbour group in that space.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb, topk_neighbors  # noqa: E402

K = 10
COL = {"text": "#4C72B0", "collab": "#C44E52"}


def main():
    rng = np.random.default_rng(0)
    raw_text = load_emb("text_only")
    raw_coll = load_collab()
    # neighbour structure from the raw (L2-normalised) spaces, as used by CSGA / Exp A
    t_nb, t_sim = topk_neighbors(l2n(raw_text), K)
    c_nb, c_sim = topk_neighbors(l2n(raw_coll), K)
    # mean-centre the spaces only for visualisation / cross-space distance reporting
    text = l2n(raw_text - raw_text.mean(0, keepdims=True))
    coll = l2n(raw_coll - raw_coll.mean(0, keepdims=True))
    n = len(text)
    cand = pd.read_csv(os.path.join(RESULTS, "case_candidates.csv")).head(3)

    def embed(x, idx):
        z = (x[idx] - x[idx].mean(0, keepdims=True)) / (x[idx].std(0, keepdims=True) + 1e-9)
        if x.shape[1] > 50:
            z = PCA(n_components=50, random_state=42).fit_transform(z)
        return TSNE(n_components=2, perplexity=20, init="pca", random_state=42,
                    max_iter=800).fit_transform(z)

    def panel(ax, anchor, space_name, reps, show_group, hide_group, title):
        T = list(t_nb[anchor]); C = list(c_nb[anchor])
        fixed = list(dict.fromkeys([anchor] + T + C))
        others = [j for j in rng.choice(n, 400, replace=False) if j not in fixed and j != anchor][:180]
        idx = fixed + others
        emb = embed(reps, idx)
        pos = {g: k for k, g in enumerate(idx)}
        ax.scatter(emb[[pos[o] for o in others], 0], emb[[pos[o] for o in others], 1],
                   s=7, color="lightgray", alpha=0.55)
        for grp, col, mk, lab in [(T, COL["text"], "o", "text top-10"),
                                  (C, COL["collab"], "^", "collab top-10")]:
            xy = [pos[j] for j in grp]
            ax.scatter(emb[xy, 0], emb[xy, 1], s=50, color=col, marker=mk,
                       edgecolor="white", linewidth=0.5, label=lab)
        ax.scatter(emb[pos[anchor], 0], emb[pos[anchor], 1], s=160, marker="*",
                   color="black", zorder=5, label="anchor")
        # mean cosine from anchor to each group, in THIS space
        ct = float((reps[anchor] @ reps[T].T).mean())
        cc = float((reps[anchor] @ reps[C].T).mean())
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\ncos(anchor, text-nb)={ct:.2f} | cos(anchor, collab-nb)={cc:.2f}", fontsize=8)

    fig, axes = plt.subplots(3, 2, figsize=(9.4, 12.0))
    for r, (_, row) in enumerate(cand.iterrows()):
        a = int(row.anchor)
        panel(axes[r, 0], a, "text", text, "text", "collab",
              f"#{a} in TEXT space (title: {str(row.title)[:46]})")
        panel(axes[r, 1], a, "collab", coll, "collab", "text",
              f"#{a} in COLLAB space")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Cross-space misalignment (representations mean-centred for visualisation):\n"
                 "the two neighbour sets are close in their own space and far in the other", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(FIGURES, "fig_case_two_space.png"), dpi=210)
    plt.close(fig)

    # best single candidate, larger
    a = int(cand.anchor.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    panel(axes[0], a, "text", text, "text", "collab", "TEXT space")
    panel(axes[1], a, "collab", coll, "collab", "text", "COLLABORATIVE space")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"Case study #{a}: collaborative and semantic neighbourhoods are misaligned", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "fig_case_best_two_space.png"), dpi=220)
    plt.close(fig)
    print("CASE_TWO_SPACE_DONE")


if __name__ == "__main__":
    main()
