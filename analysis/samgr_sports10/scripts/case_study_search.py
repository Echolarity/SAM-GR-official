"""Search for compelling case studies: anchors whose text and collaborative
neighbourhoods are (i) disjoint and (ii) far apart in each other's space.

Selection score per anchor i (text top-10 T, collab top-10 C):
  sep = (1 - mean cos_text(i, C)) + (1 - mean cos_collab(i, T))
i.e. collaborative neighbours should be far in TEXT space, and text neighbours
should be far in COLLAB space.  We also require both neighbourhoods to be tight
in their own space.

Outputs:
  results/case_candidates.csv          top candidates + titles
  figures/fig_case_candidates.png      3x3 grid of joint projections
  figures/fig_case_best.png            best candidate, larger
"""
from __future__ import annotations

import json
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
from common import (FIGURES, RESULTS, l2n, load_collab, load_emb, load_items,  # noqa: E402
                    topk_neighbors)

K = 10
COLORS = {"text": "#4C72B0", "collab": "#C44E52"}


def main():
    rng = np.random.default_rng(0)
    text = l2n(load_emb("text_only"))
    coll = l2n(load_collab())
    items = load_items()
    n = len(text)

    t_nb, t_sim = topk_neighbors(text, K)
    c_nb, c_sim = topk_neighbors(coll, K)

    rows = []
    for i in range(n):
        T = t_nb[i]; C = c_nb[i]
        if len(np.intersect1d(T, C)) > 0:
            continue
        tight_t = float(t_sim[i].mean())          # text nb closeness in text space
        tight_c = float(c_sim[i].mean())          # collab nb closeness in collab space
        tdc = float(1.0 - (text[i] @ text[C].T).mean())       # collab nb far in text space
        cdt = float(1.0 - (coll[i] @ coll[T].T).mean())       # text nb far in collab space
        sep = tdc + cdt
        rows.append({"anchor": i, "sep": sep, "tdc": tdc, "cdt": cdt,
                     "tight_text": tight_t, "tight_collab": tight_c,
                     "title": (items.get(str(i), {}).get("title") or "")[:90]})
    df = pd.DataFrame(rows)
    # require both neighbourhoods to be reasonably tight in their own space
    qt = df.tight_text.quantile(0.5); qc = df.tight_collab.quantile(0.5)
    cand = df[(df.tight_text >= qt) & (df.tight_collab >= qc)].sort_values("sep", ascending=False)
    top = cand.head(9).reset_index(drop=True)
    top.to_csv(os.path.join(RESULTS, "case_candidates.csv"), index=False)
    print(top[["anchor", "sep", "tdc", "cdt", "tight_text", "tight_collab", "title"]].to_string(index=False))

    # joint space for visualisation
    def joint_proj(idx):
        out = []
        for x in (text, coll):
            z = (x[idx] - x[idx].mean(0, keepdims=True)) / (x[idx].std(0, keepdims=True) + 1e-9)
            out.append(PCA(n_components=50, random_state=42).fit_transform(z))
        return np.concatenate(out, axis=1)

    def draw(ax, anchor, title_fs=8, legend=False):
        T = list(t_nb[anchor]); C = list(c_nb[anchor])
        fixed = list(dict.fromkeys([anchor] + T + C))
        others = [j for j in rng.choice(n, 400, replace=False) if j not in fixed and j != anchor][:180]
        idx = fixed + others
        J = joint_proj(idx)
        emb = TSNE(n_components=2, perplexity=20, init="pca", random_state=42,
                   max_iter=800).fit_transform(J)
        pos = {g: k for k, g in enumerate(idx)}
        ax.scatter(emb[[pos[o] for o in others], 0], emb[[pos[o] for o in others], 1],
                   s=7, color="lightgray", alpha=0.55)
        tx = [pos[j] for j in T]; cx = [pos[j] for j in C]
        ax.scatter(emb[tx, 0], emb[tx, 1], s=48, color=COLORS["text"], edgecolor="white",
                   linewidth=0.5, label="text top-10")
        ax.scatter(emb[cx, 0], emb[cx, 1], s=48, color=COLORS["collab"], marker="^",
                   edgecolor="white", linewidth=0.5, label="collab top-10")
        ax.scatter(emb[pos[anchor], 0], emb[pos[anchor], 1], s=150, marker="*",
                   color="black", zorder=5, label="anchor")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title_fs and f"#{anchor}  sep={float(df.loc[df.anchor==anchor,'sep'].iloc[0]):.3f}", fontsize=title_fs)
        if legend:
            ax.legend(fontsize=6.5, frameon=False, loc="best")

    # 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5))
    for ax, a in zip(axes.ravel(), top.anchor.tolist()):
        draw(ax, int(a), title_fs=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Candidate case studies: collaborative vs semantic neighbourhoods", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGURES, "fig_case_candidates.png"), dpi=200)
    plt.close(fig)

    # best
    best = int(top.anchor.iloc[0])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    draw(ax, best, title_fs=10, legend=True)
    ax.set_title(f"Case study #{best}: collaborative vs semantic neighbours\n"
                 f"(sep={top.sep.iloc[0]:.3f}, overlap=0, title: {top.title.iloc[0][:70]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_case_best.png"), dpi=220)
    plt.close(fig)
    print("best anchor:", best, "title:", top.title.iloc[0])
    print("CASE_SEARCH_DONE")


if __name__ == "__main__":
    main()
