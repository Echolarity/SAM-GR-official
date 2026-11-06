"""Combined 3-panel figure for the WWW paper (full text width).

Layout (2 x 2, ~7.0 x 3.9 in):
  (a) top-10 neighbourhood overlap across representation spaces  [linear 0-40%]
  (b) fused-representation CKA vs each source space              [0-100%]
  (c) two-space case study: text space | collab space t-SNE      [needs server data]

Panel (c) requires the project data on the server (common.py / RESULTS).
Panels (a) and (b) are self-contained.

Run on the server:  python fig_challenge_combined.py
Outputs: fig_challenge_combined.{png,pdf,svg} at 600 dpi (vector fonts).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 7,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = os.environ.get(
    "FIG_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "www_paper_figures"),
)
os.makedirs(OUT, exist_ok=True)

C_TEXT, C_COLLAB = "#4C72B0", "#C44E52"
C_GRAY = "#AEB4BC"

def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(axis="y", color="#DEE2E8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

def tag(ax, label):
    ax.text(-0.09, 1.04, label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="bottom", ha="left")

# ------------------------------------------------------------ figure ------
fig = plt.figure(figsize=(7.0, 4.0), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.10],
                      hspace=0.33, wspace=0.30)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c1 = fig.add_subplot(gs[1, 0])
ax_c2 = fig.add_subplot(gs[1, 1])

# ------------------------------------------------- (a) neighbourhood overlap
pairs = ["C–T", "C–V", "C–M", "T–V", "T–M"]
overlap = [1.22, 1.13, 1.26, 15.75, 36.37]
colors = [C_COLLAB, C_COLLAB, C_COLLAB, C_TEXT, C_TEXT]

x = np.arange(len(pairs))
bars = ax_a.bar(x, overlap, width=0.58, color=colors,
                edgecolor="white", linewidth=0.8, zorder=3)
for bar, v in zip(bars, overlap):
    ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{v:.1f}",
              ha="center", va="bottom", fontsize=6.6, color="#222222")
ax_a.axhline(0.09, color="#888888", linestyle=(0, (3, 2)), linewidth=0.9, zorder=2)
ax_a.text(0.015, 0.965, "permutation chance = 0.09%", transform=ax_a.transAxes,
          ha="left", va="top", fontsize=6.2, color="#888888")
ax_a.set_ylim(0, 40)
ax_a.set_yticks([0, 10, 20, 30, 40])
ax_a.set_xticks(x, pairs)
ax_a.tick_params(axis="x", labelsize=7.5)
ax_a.set_ylabel("Top-10 overlap (%)")
strip(ax_a)
tag(ax_a, "(a)")

# ------------------------------------------------------------ (b) CKA ----
labels = ["Text", "Visual", "Multimodal"]
cka = [70.3, 7.1, 94.2]
cka_colors = ["#4C78A8", "#59A14F", "#8B79B8"]

x = np.arange(len(labels))
bars = ax_b.bar(x, cka, width=0.52, color=cka_colors,
                edgecolor="white", linewidth=0.8, zorder=3)
for bar, v in zip(bars, cka):
    ax_b.text(bar.get_x() + bar.get_width() / 2, v + 2.0, f"{v:.1f}",
              ha="center", va="bottom", fontsize=6.8, color="#222222")
ax_b.set_ylim(0, 100)
ax_b.set_yticks([0, 25, 50, 75, 100])
ax_b.set_xticks(x, labels)
ax_b.set_ylabel("CKA similarity (%)")
strip(ax_b)
tag(ax_b, "(b)")

# ------------------------------------------- (c) two-space case study ----
# Requires the project data (common.py / RESULTS) on the server.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb, topk_neighbors  # noqa: E402

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

K = 10

def case_panel(ax, anchor, reps, nb_text, nb_coll, title):
    rng = np.random.default_rng(0)
    n = len(reps)
    fixed = list(dict.fromkeys([anchor] + list(nb_text) + list(nb_coll)))
    others = [j for j in rng.choice(n, 400, replace=False)
              if j not in fixed and j != anchor][:180]
    idx = fixed + others

    z = (reps[idx] - reps[idx].mean(0, keepdims=True)) / (reps[idx].std(0, keepdims=True) + 1e-9)
    if z.shape[1] > 50:
        z = PCA(n_components=50, random_state=42).fit_transform(z)
    emb = TSNE(n_components=2, perplexity=20, init="pca",
               random_state=42, max_iter=800).fit_transform(z)
    pos = {g: k for k, g in enumerate(idx)}

    ax.scatter(emb[[pos[o] for o in others], 0], emb[[pos[o] for o in others], 1],
               s=6, color=C_GRAY, alpha=0.55, linewidths=0, zorder=1)
    for grp, col, mk in [(list(nb_text), C_TEXT, "o"),
                         (list(nb_coll), C_COLLAB, "^")]:
        xy = [pos[j] for j in grp]
        ax.scatter(emb[xy, 0], emb[xy, 1], s=52, color=col, marker=mk,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(emb[pos[anchor], 0], emb[pos[anchor], 1], s=150, marker="*",
               color="black", zorder=5)

    ct = float((reps[anchor] @ reps[list(nb_text)].T).mean())
    cc = float((reps[anchor] @ reps[list(nb_coll)].T).mean())
    ax.text(0.03, 0.97, f"cos  text {ct:.2f} | collab {cc:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.6, color="#333333",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC", lw=0.5))

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7); s.set_color("#BBBBBB")
    ax.set_title(title, fontsize=8, pad=3)

def main():
    raw_text = load_emb("text_only")
    raw_coll = load_collab()
    t_nb, _ = topk_neighbors(l2n(raw_text), K)
    c_nb, _ = topk_neighbors(l2n(raw_coll), K)
    # mean-centre only for visualisation, as in the original script
    text = l2n(raw_text - raw_text.mean(0, keepdims=True))
    coll = l2n(raw_coll - raw_coll.mean(0, keepdims=True))

    import pandas as pd
    anchor = int(pd.read_csv(os.path.join(RESULTS, "case_candidates.csv"))
                 .head(1).anchor.iloc[0])

    case_panel(ax_c1, anchor, text, t_nb[anchor], c_nb[anchor], "Text space")
    case_panel(ax_c2, anchor, coll, t_nb[anchor], c_nb[anchor], "Collaborative space")
    tag(ax_c1, "(c)")

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_TEXT,
               markeredgecolor="white", markersize=6, label="text top-10"),
        Line2D([], [], marker="^", color="none", markerfacecolor=C_COLLAB,
               markeredgecolor="white", markersize=6, label="collab top-10"),
        Line2D([], [], marker="*", color="none", markerfacecolor="black",
               markeredgecolor="none", markersize=9, label="anchor"),
    ]
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=3, frameon=False,
                  columnspacing=1.4, handletextpad=0.35)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_challenge_combined.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved to:", os.path.abspath(OUT))
    for name in sorted(os.listdir(OUT)):
        if name.startswith("fig_challenge_combined"):
            print(" ", os.path.join(os.path.abspath(OUT), name))

if __name__ == "__main__":
    main()
