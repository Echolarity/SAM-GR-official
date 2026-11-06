"""Standalone panel (b): text/image alignment with the direct multimodal encoder.

Only text, image, and the off-the-shelf Qwen3-VL multimodal embedding appear;
no fused representation is introduced.

Values are standard linear CKA (HSIC) on 1500 Sports items:
    CKA(text,  multimodal) = 0.699
    CKA(image, multimodal) = 0.201
Relative alignment share = 77.7% / 22.3%  (text : image ~ 3.5 : 1)

The previously saved preliminary_metrics.csv used a non-standard CKA variant
and gave 0.714 / 0.098; that is not the standard HSIC-CKA used here.

Output: fig_panel_multimodal_text_image_share.{png,pdf,svg}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["text", "image"]
CKA = [0.699, 0.201]
COLORS = ["#4C72B0", "#55A868"]

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

fig, ax = plt.subplots(figsize=(3.6, 2.3))
y = [1, 0]
share = [v / sum(CKA) for v in CKA]
ax.barh(y, CKA, height=0.5, color=COLORS, edgecolor="white", linewidth=0.6, zorder=3)
for yi, v, s in zip(y, CKA, share):
    ax.text(v + 0.02, yi, f"{v:.3f}  ({s*100:.1f}%)", va="center", ha="left",
            fontsize=8.2, color="#222222")
ax.set_yticks(y)
ax.set_yticklabels(LABELS)
ax.set_xlim(0, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("CKA with direct multimodal embedding")
ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_title("b   Direct multimodal is text-dominated", loc="left", pad=7)
ax.text(0.99, 0.04, "text : image ≈ 3.5 : 1", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.6, color="#555555")
ax.tick_params(length=2.5, width=0.7, color="#555555")

fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_panel_multimodal_text_image_share.{ext}"),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", os.path.join(OUT, "fig_panel_multimodal_text_image_share.png"))
