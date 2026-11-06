"""Paper-quality figures for the Top-K neighbourhood / CSGA weight analysis."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS  # noqa: E402

SPACES = ["collab", "text", "image", "joint"]
COLORS = {"collab": "#C44E52", "text": "#4C72B0", "image": "#55A868", "joint": "#8172B2"}
LABELS = {"collab": "Collaborative", "text": "Text", "image": "Image", "joint": "Joint multimodal"}
PAIRS = [("collab", "text"), ("collab", "image"), ("collab", "joint"),
         ("text", "image"), ("text", "joint"), ("image", "joint")]
KS = [10, 20, 50]
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_overlap():
    df = pd.read_csv(os.path.join(RESULTS, "exp_e_pair_overlap_validcollab.csv"))
    pair_names = [f"{a}-{b}" for a, b in PAIRS]
    jac = np.array([[df[(df.pair == p) & (df.k == k)].jaccard_mean.iloc[0] for k in KS] for p in pair_names])
    rec = np.array([[df[(df.pair == p) & (df.k == k)].recall_overlap_mean.iloc[0] for k in KS] for p in pair_names])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), gridspec_kw={"hspace": 0.45})
    for ax, mat, title, vmax in zip(
        axes, [jac, rec],
        [r"Mean Jaccard overlap  $|A\cap B| / |A\cup B|$",
         r"Mean neighbour recall overlap  $|A\cap B| / K$"],
        [0.30, 0.42],
    ):
        cmap = LinearSegmentedColormap.from_list("paper", ["#f7f7f7", "#fdae6b", "#a63603"])
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"K={k}" for k in KS])
        ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([f"{a} vs {b}" for a, b in PAIRS])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.55 * vmax else "#222222", fontsize=8)
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Top-K neighbourhood overlap between representation spaces (Sports 10-core)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_e_overlap_heatmap.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "fig_e_overlap_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_quality():
    df = pd.read_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics_validcollab.csv"))
    df = df[df.group == "all"]
    panels = [
        ("cat_purity_leaf", "rand_pop_cat_purity_leaf", "Category purity (leaf)", None),
        ("lift_pop_cooc_prob_agg", None, "User co-occurrence lift", 1.0),
        ("lift_pop_trans_prob_agg", None, "Sequential transition lift", 1.0),
        ("npmi", "rand_pop_npmi", "Normalized PMI", None),
        ("text_cos", "rand_pop_text_cos", "Text cosine to target", None),
        ("image_cos", "rand_pop_image_cos", "Visual cosine to target", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    for ax, (col, rand_col, title, hline) in zip(axes.ravel(), panels):
        for s in SPACES:
            d = df[df.space == s].sort_values("k")
            ax.plot(d.k, d[col], marker="o", ms=4, lw=1.6, color=COLORS[s], label=LABELS[s])
            if rand_col is not None:
                ax.plot(d.k, d[rand_col], ls="--", lw=1.1, color=COLORS[s], alpha=0.45)
        if rand_col is not None:
            ax.plot([], [], ls="--", color="#777777", lw=1.1, label="pop-matched random")
        if hline is not None:
            ax.axhline(hline, color="#777777", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("K"); ax.set_title(title)
        ax.grid(alpha=0.18, lw=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Neighbourhood quality across representation spaces (Sports 10-core)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(os.path.join(FIGURES, "fig_e_neighborhood_quality.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "fig_e_neighborhood_quality.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_groups():
    df = pd.read_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics_validcollab.csv"))
    df = df[(df.k == 10) & (df.group.isin(["head", "mid", "tail"]))]
    panels = [
        ("cat_purity_leaf", "Category purity (leaf)"),
        ("lift_pop_cooc_prob_agg", "User co-occurrence lift"),
        ("lift_pop_trans_prob_agg", "Sequential transition lift"),
        ("npmi", "Normalized PMI"),
    ]
    groups = ["head", "mid", "tail"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
    width = 0.19
    for ax, (col, title) in zip(axes, panels):
        xs = np.arange(len(groups))
        for j, s in enumerate(SPACES):
            vals = [df[(df.space == s) & (df.group == g)][col].iloc[0] for g in groups]
            ax.bar(xs + (j - 1.5) * width, vals, width, color=COLORS[s], label=LABELS[s])
        if col.startswith("lift"):
            ax.axhline(1.0, color="#777777", ls=":", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(["Head", "Mid", "Tail"])
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.18, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Neighbourhood quality by item popularity group (K=10)", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "fig_e_head_mid_tail.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "fig_e_head_mid_tail.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_csga():
    with open(os.path.join(RESULTS, "exp_f_csga_metrics_validcollab.json")) as f:
        M = json.load(f)
    E = np.load(os.path.join(RESULTS, "exp_f_csga_edges.npz"))
    per = pd.read_csv(os.path.join(RESULTS, "exp_f_csga_per_item_validcollab.csv"))
    valid_anchor = np.isin(E["anchor"], per.item.values)
    anchor = E["anchor"][valid_anchor]; neighbor = E["neighbor"][valid_anchor]
    w = E["weight"][valid_anchor].astype(np.float64)
    c = E["collab_cos"][valid_anchor].astype(np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

    ax = axes[0, 0]
    hb = ax.hexbin(c, w, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    ax.set_xlabel("Collaborative cosine $s_{ij}$")
    ax.set_ylabel("Learned CSGA weight $a_{ij}$")
    ax.set_title("Weight vs. collaborative similarity", fontsize=9.5)
    ax.text(0.03, 0.95,
            f"global Spearman = {M['weight_vs_collab_cosine']['spearman_global']:.3f}\n"
            f"per-item mean = {M['weight_vs_collab_cosine']['spearman_per_item_mean']:.3f}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    ax = axes[0, 1]
    # reconstruct mutual-pair differences from the edge table
    n = int(per.item.max()) + 1
    wd = {int(i) * n + int(j): float(x) for i, j, x in zip(anchor, neighbor, w)}
    diffs = [abs(x - wd[int(j) * n + int(i)]) for i, j, x in zip(anchor, neighbor, w)
             if int(j) * n + int(i) in wd]
    ax.hist(diffs, bins=50, color="#C44E52", alpha=0.85)
    ax.axvline(np.mean(diffs), color="black", lw=1.4, label=f"mean={np.mean(diffs):.3f}")
    ax.axvline(np.median(diffs), color="#333333", lw=1.0, ls="--", label=f"median={np.median(diffs):.3f}")
    ax.set_xlabel("$|a_{ij}-a_{ji}|$ on mutual neighbour pairs")
    ax.set_ylabel("#pairs")
    ax.set_title("Directional asymmetry of learned weights", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    ax.hist(per.attention_entropy_norm, bins=50, color="#4C72B0", alpha=0.85)
    ax.axvline(1.0, color="#777777", ls=":", lw=1.2, label="uniform = 1")
    ax.axvline(M["entropy"]["entropy_norm_mean"], color="black", lw=1.4,
               label=f"mean={M['entropy']['entropy_norm_mean']:.3f}")
    ax.set_xlabel("Normalized attention entropy $H/\\log K$")
    ax.set_ylabel("#items")
    ax.set_title("Attention entropy (K=10)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    T = M["transition_lift_top5_vs_unweighted"]
    vals = [T["mean_bottom5_trans"], T["mean_unweighted_trans"], T["mean_top5_trans"]]
    names = ["Bottom-5\nweights", "All-10\n(unweighted)", "Top-5\nweights"]
    bars = ax.bar(names, vals, color=["#bdbdbd", "#777777", "#55A868"], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.00025, f"{v*100:.2f}%",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Mean sequential transition $P(j|i)$")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Top-weight transition lift", fontsize=9.5)
    ax.text(0.03, 0.95,
            f"lift = {T['mean_lift']:.2f} (mean), {T['median_lift']:.2f} (median)\n"
            f"items > 1 = {T['frac_items_lift_gt_1']*100:.0f}%, n={T['n_items']}",
            transform=ax.transAxes, va="top", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))
    fig.suptitle("CSGA learned neighbour weights: weak cosine alignment, near-uniform attention,\n"
                 "directional asymmetry, and modest transition lift", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "fig_f_csga_weights.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "fig_f_csga_weights.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_overlap()
    fig_quality()
    fig_groups()
    fig_csga()
    print("FIGURES_DONE")
