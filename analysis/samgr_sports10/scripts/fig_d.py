"""Standalone version of panel d: neighbourhood overlap (linear percent y-axis).

Extracted from "Combined, simplified.txt" (panel_overlap). Changes vs. original:
  - y-axis changed from log scale to linear, in percent, range 0-40
Everything else (colors, layout, fonts, annotations) kept the same.

Outputs:
  figures/fig_d_neighbourhood_overlap.png
  figures/fig_d_neighbourhood_overlap.pdf
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS  # noqa: E402
# # Paths originally provided by common.py; adjust if your data lives elsewhere.
# BASE = os.path.dirname(os.path.abspath(__file__))
# RESULTS = os.path.join(BASE, "results")
# FIGURES = os.path.join(BASE, "figures")

COLORS = {
    "collab": "#C44E52",
    "text": "#4C72B0",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _clean(ax):
    ax.tick_params(length=2.5, width=0.7, color="#555555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_overlap(ax, stats):
    pairs = [("collab", "text"), ("collab", "image"), ("collab", "multimodal"),
             ("text", "image"), ("text", "multimodal")]
    obs = [stats.loc[stats.pair == f"{a}-{b}", "overlap_obs"].iloc[0] * 100.0
           for a, b in pairs]
    x = np.array([0, 1, 2, 3.55, 4.55])
    cols = [COLORS["collab"]] * 3 + [COLORS["text"]] * 2
    ax.bar(x, obs, width=0.62, color=cols, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, v in zip(x, obs):
        ax.text(xi, v + 1.0, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=7.8, fontweight="bold")
    # chance = stats["overlap_null_mean"].mean() * 100.0
    # ax.axhline(chance, color="#8A8A8A", lw=0.8, ls=":", zorder=2)
    # ax.text(2.82, chance + 1.6, "chance", ha="center", va="bottom",
    #         fontsize=7.2, color="#666666")
    ax.set_ylim(0, 40)
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_xlim(-0.6, 5.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["collab\ntext", "collab\nimage", "collab\nmulti",
                        "text\nimage", "text\nmulti"], fontsize=7.6)
    ax.set_ylabel("top-10 neighbour overlap  (%)")
    ax.axvspan(-0.55, 2.55, color="#FBEAEA", zorder=0)
    ax.axvspan(3.1, 5.0, color="#EAF1FA", zorder=0)
    ax.text(1.0, 41.0, "collab vs semantic", ha="center", fontsize=7.8,
            color=COLORS["collab"], fontweight="bold")
    ax.text(4.05, 41.0, "semantic vs semantic", ha="center", fontsize=7.8,
            color=COLORS["text"], fontweight="bold")
    # ax.set_title("d   Neighbourhoods barely overlap", loc="left", pad=7)
    _clean(ax)


def main():
    stats = pd.read_csv(os.path.join(RESULTS, "exp_a_statistical_tests.csv"))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    panel_overlap(ax, stats)

    png = os.path.join(FIGURES, "fig_d_neighbourhood_overlap.png")
    pdf = os.path.join(FIGURES, "fig_d_neighbourhood_overlap.pdf")
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print("saved", png, pdf, sep="\n")


if __name__ == "__main__":
    main()
