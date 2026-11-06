"""Standalone panel (b): CKA of the fused representation with each modality.

Values are the Sports strict-10-core numbers saved in
results/preliminary_metrics.csv (linear CKA over 1500 sampled items):
    fused-text = 0.703, fused-image = 0.071, fused-multimodal = 0.942
No input files are read; this only redraws the bar chart.

Output: fig_panel_fused_cka.{png,pdf,svg}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["text", "image", "multimodal"]
CKA = [0.703, 0.071, 0.942]
COLORS = ["#4C72B0", "#55A868", "#8172B2"]

OUT = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.4,
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

fig, ax = plt.subplots(figsize=(3.4, 2.6))
y = [2, 1, 0]
ax.barh(y, CKA, height=0.54, color=COLORS, edgecolor="white", linewidth=0.6, zorder=3)
for yi, v in zip(y, CKA):
    ax.text(v + 0.025, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.2, color="#222222")
ax.set_yticks(y)
ax.set_yticklabels(LABELS)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("CKA with fused representation")
ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_title("b   Fusion is text-dominated", loc="left", pad=7)
ax.tick_params(length=2.5, width=0.7, color="#555555")

fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_panel_fused_cka.{ext}"),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", os.path.join(OUT, "fig_panel_fused_cka.png"))
