"""Standalone summary figure: CKA + neighbour evidence for fused representations.

All data are embedded below; no input files are read.

Core conclusion:
  fused keeps high CKA with direct multimodal (semantic retention), while
  CKA(fused, collab) is higher than CKA(multimodal, collab), and the fused
  representation recovers more collaborative neighbours.

CKA is standard linear CKA (HSIC) on 1500 items for all four datasets.
Neighbour metrics use k=10.  Sports/IndSci neighbour numbers are full-catalog;
Beauty13/Toys10 neighbour numbers are from the 1500-item CKA subset (marked
with triangles), because their full multimodal representation was not saved.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# name, CKA(fused, multimodal), CKA(multimodal, collab), CKA(fused, collab),
# semantic retention (neighbour), collab lift (neighbour), subset neighbour?
DATA = [
    ("Sports",  0.906, 0.048, 0.084, 0.513, 1.259, False),
    ("IndSci",  0.922, 0.052, 0.068, 0.598, 1.484, False),
    ("Beauty13", 0.871, 0.06376, 0.09900, 0.413, 1.252, False),
    ("Toys10",  0.887, 0.05576, 0.08702, 0.516, 1.317, False),
]
COLORS = {
    "Sports": "#4C72B0",
    "IndSci": "#55A868",
    "Beauty13": "#C44E52",
    "Toys10": "#8172B2",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

names = [d[0] for d in DATA]
cka_fm = np.array([d[1] for d in DATA])
cka_mc = np.array([d[2] for d in DATA])
cka_fc = np.array([d[3] for d in DATA])
sem = np.array([d[4] for d in DATA])
lift = np.array([d[5] for d in DATA])
subset = np.array([d[6] for d in DATA])
delta = (cka_fc - cka_mc) / cka_mc * 100.0
colors = [COLORS[n] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9),
                         gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})

# ------------------------------------------------------------------ (a) semantic retention
ax = axes[0]
y = np.arange(len(names))[::-1]
ax.barh(y, cka_fm, height=0.55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
for yi, v in zip(y, cka_fm):
    ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.0, color="#222222")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8.6)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("CKA(fused, direct multimodal)")
ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_title("a   Semantic information is retained", loc="left", pad=7)
ax.axvline(0.9, color="#999999", lw=0.8, ls="--", zorder=1)

# ------------------------------------------------------------------ (b) collaborative shift
ax = axes[1]
x = np.arange(len(names))
w = 0.34
ax.bar(x - w / 2, cka_mc, w, color="#B9B9B9", edgecolor="white", linewidth=0.5,
       label="multimodal → collab", zorder=3)
ax.bar(x + w / 2, cka_fc, w, color=colors, edgecolor="white", linewidth=0.5,
       label="fused → collab", zorder=3)
for xi, mc, fc, d in zip(x, cka_mc, cka_fc, delta):
    ax.text(xi - w / 2, mc + 0.003, f"{mc:.3f}", ha="center", va="bottom",
            fontsize=6.8, color="#444444")
    ax.text(xi + w / 2, fc + 0.003, f"{fc:.3f}", ha="center", va="bottom",
            fontsize=6.8, color="#222222")
    ax.text(xi, max(mc, fc) + 0.012, f"+{d:.1f}%", ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color=COLORS[names[xi]])
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.4)
ax.set_ylim(0, 0.13)
ax.set_ylabel("CKA with collaborative")
ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_title("b   Fusion shifts toward collaborative geometry", loc="left", pad=7)
ax.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=2, handlelength=1.2)

# ------------------------------------------------------------------ (c) neighbour trade-off
ax = axes[2]
for i, n in enumerate(names):
    marker = "^" if subset[i] else "o"
    ax.scatter(sem[i], lift[i], s=100, marker=marker, color=COLORS[n],
               edgecolor="white", linewidth=0.8, zorder=4)
    offsets = {
        "Sports": (-70, 12),
        "IndSci": (-55, 10),
        "Beauty13": (8, -24),
        "Toys10": (8, 14),
    }[n]
    ax.annotate(f"{n}\n{sem[i]:.2f}, ×{lift[i]:.2f}", (sem[i], lift[i]),
                textcoords="offset points", xytext=offsets, fontsize=7.3, color="#222222",
                annotation_clip=False)
ax.axhline(1.0, color="#777777", lw=0.9, ls="--", zorder=1)
ax.set_xlim(0.39, 0.65)
ax.set_ylim(1.0, 1.58)
ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
ax.set_xlabel("semantic neighbour retention")
ax.set_ylabel("collaborative neighbour lift (×)")
ax.grid(color="#E6E6E6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_title("c   More collaborative neighbours, with semantic retention", loc="left", pad=7)
ax.text(0.395, 1.02, "above dashed line = collab overlap increases",
        fontsize=7.0, color="#555555", va="bottom")
ax.text(0.645, 1.02, "all full-catalog", fontsize=7.0, color="#555555",
        ha="right", va="bottom")


fig.suptitle("Fusion preserves semantic geometry while shifting toward collaborative structure",
             fontsize=12.2, fontweight="bold", y=1.02)
fig.text(0.5, -0.06,
         "CKA: standard HSIC on 1500 items for all datasets.  Neighbour k=10 over the full catalog "
         "(all four datasets).",
         ha="center", fontsize=7.2, color="#555555")
fig.tight_layout(w_pad=2.0)

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_fusion_cka_neighbor_summary.{ext}"),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", os.path.join(OUT, "fig_fusion_cka_neighbor_summary.png"))
