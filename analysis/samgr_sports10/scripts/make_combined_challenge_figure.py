"""Combined, simplified two-challenge figure (only the two requested figures).

Challenge 1 (modality analysis): cross-space CKA, fusion composition, SID agreement.
Challenge 2 (collaborative analysis): neighbourhood overlap, intrinsic dimension,
global linear alignment.

Reads only saved CSVs/embeddings; recomputes the PCA cumulative-variance curves.
Outputs:
  figures/fig_two_challenges_combined.png
  figures/fig_two_challenges_combined.pdf
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb  # noqa: E402

N_SAMPLE = 3000
SPACES = ["collab", "text", "image", "multimodal", "fused"]
COLORS = {
    "collab": "#C44E52",
    "text": "#4C72B0",
    "image": "#55A868",
    "multimodal": "#8172B2",
    "fused": "#B8860B",
}
PAIR_LABEL = {
    "collab-text": "collab–text",
    "collab-image": "collab–image",
    "collab-multimodal": "collab–multimodal",
    "text-image": "text–image",
    "text-multimodal": "text–multimodal",
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


def effective_dim_curves():
    rng = np.random.default_rng(0)
    reps = {
        "collab": l2n(load_collab()),
        "text": l2n(load_emb("text_only")),
        "image": l2n(load_emb("image_only")),
        "multimodal": l2n(load_emb("multimodal")),
    }
    sample = np.sort(rng.choice(len(reps["text"]), N_SAMPLE, replace=False))
    out = {}
    for k, v in reps.items():
        x = v[sample].astype(np.float64)
        x -= x.mean(0, keepdims=True)
        p = PCA(n_components=min(500, x.shape[1], len(x) - 1),
                svd_solver="randomized", random_state=42).fit(x)
        out[k] = np.cumsum(p.explained_variance_ratio_)
    return out


def load_numbers():
    stats = pd.read_csv(os.path.join(RESULTS, "exp_a_statistical_tests.csv"))
    align = pd.read_csv(os.path.join(RESULTS, "exp_a_geometry_alignment.csv"))
    dims = pd.read_csv(os.path.join(RESULTS, "exp_a_geometry_dims.csv"))
    ari = pd.read_csv(os.path.join(RESULTS, "exp_b_partition_agreement.csv"))
    cka = pd.read_csv(os.path.join(RESULTS, "preliminary_metrics.csv"))
    C = pd.DataFrame(np.eye(len(SPACES)), index=SPACES, columns=SPACES)
    for _, r in cka[cka.type == "cka"].iterrows():
        C.loc[r.a, r.b] = C.loc[r.b, r.a] = float(r.value)
    return stats, align, dims, ari, C


def _clean(ax):
    ax.tick_params(length=2.5, width=0.7, color="#555555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_cka(ax, C):
    vals = C.values
    im = ax.imshow(vals, vmin=0, vmax=1, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(SPACES)))
    ax.set_yticks(range(len(SPACES)))
    ax.set_xticklabels(["collab", "text", "image", "multi-\nmodal", "fused"], fontsize=8.0)
    ax.set_yticklabels(["collab", "text", "image", "multi-\nmodal", "fused"], fontsize=8.0)
    ax.tick_params(length=0, pad=2)
    for i in range(len(SPACES)):
        for j in range(len(SPACES)):
            v = vals[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if v < 0.55 else "#1A1A1A")
    # highlight the complementary text-image pair
    for (i, j) in [(1, 2), (2, 1)]:
        ax.add_patch(Rectangle((j - 0.43, i - 0.43), 0.86, 0.86, fill=False,
                               edgecolor="#39FF88", linewidth=1.3, zorder=5))
    ax.set_title("a   Cross-space CKA", loc="left", pad=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.035, shrink=0.82)
    cb.ax.tick_params(labelsize=7.2, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label("CKA", fontsize=7.5, labelpad=2)


def panel_fused(ax, C):
    names = ["text", "image", "multimodal"]
    vals = [C.loc["fused", n] for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.54, color=[COLORS[n] for n in names],
            edgecolor="white", linewidth=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.025, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=8.2, color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels(["text", "image", "multimodal"])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("CKA with fused representation")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_title("b   Fusion is text-dominated", loc="left", pad=7)
    _clean(ax)


def panel_sid(ax, ari):
    x = np.arange(len(ari))
    w = 0.34
    ax.bar(x - w / 2, ari.ARI, w, color="#4C72B0", label="ARI", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, ari.NMI, w, color="#DD8452", label="NMI", zorder=3,
           edgecolor="white", linewidth=0.5)
    for xi, v in zip(x - w / 2, ari.ARI):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2)
    for xi, v in zip(x + w / 2, ari.NMI):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax.axvspan(3 - 0.5, 3 + 0.5, color="#EAF5EA", zorder=0)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(["a", "a+b", "a+b+c", "full SID"])
    ax.set_ylabel("agreement with text-only")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2,
              columnspacing=1.0)
    ax.set_title("c   Final SID partition unchanged", loc="left", pad=7)
    _clean(ax)


def panel_overlap(ax, stats):
    pairs = [("collab", "text"), ("collab", "image"), ("collab", "multimodal"),
             ("text", "image"), ("text", "multimodal")]
    obs = [stats.loc[stats.pair == f"{a}-{b}", "overlap_obs"].iloc[0] for a, b in pairs]
    x = np.array([0, 1, 2, 3.55, 4.55])
    cols = [COLORS["collab"]] * 3 + [COLORS["text"]] * 2
    ax.bar(x, obs, width=0.62, color=cols, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, v in zip(x, obs):
        ax.text(xi, v * 1.22, f"{v * 100:.1f}%", ha="center", va="bottom",
                fontsize=7.8, fontweight="bold")
    chance = stats["overlap_null_mean"].mean()
    ax.axhline(chance, color="#8A8A8A", lw=0.8, ls=":", zorder=2)
    ax.text(2.82, chance * 1.38, "chance", ha="center", va="bottom",
            fontsize=7.2, color="#666666")
    ax.set_yscale("log")
    ax.set_ylim(4.5e-4, 0.75)
    ax.set_xlim(-0.6, 5.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["collab\ntext", "collab\nimage", "collab\nmulti",
                        "text\nimage", "text\nmulti"], fontsize=7.6)
    ax.set_ylabel("top-10 neighbour overlap  (log)")
    ax.axvspan(-0.55, 2.55, color="#FBEAEA", zorder=0)
    ax.axvspan(3.1, 5.0, color="#EAF1FA", zorder=0)
    ax.text(1.0, 0.68, "collab vs semantic", ha="center", fontsize=7.8,
            color=COLORS["collab"], fontweight="bold")
    ax.text(4.05, 0.68, "semantic vs semantic", ha="center", fontsize=7.8,
            color=COLORS["text"], fontweight="bold")
    ax.set_title("d   Neighbourhoods barely overlap", loc="left", pad=7)
    _clean(ax)


def panel_dims(ax, cum, dims):
    keys = ["collab", "text", "image", "multimodal"]
    for k in keys:
        row = dims[dims.space == k].iloc[0]
        c = cum[k]
        ax.plot(np.arange(1, len(c) + 1), c, color=COLORS[k], lw=1.8,
                label=f"{k}  ({int(row.d90)})", zorder=3)
    ax.axhline(0.9, color="#8A8A8A", lw=0.8, ls=":")
    ax.text(1.04, 0.915, "90%", fontsize=7.2, color="#666666", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1, 500)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative variance")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2,
              handlelength=1.3, title="d$_{90}$", title_fontsize=7.2)
    ax.grid(color="#ECECEC", linewidth=0.5, zorder=0)
    ax.set_title("e   Different intrinsic dimensionality", loc="left", pad=7)
    _clean(ax)


def panel_align(ax, align):
    rows = ["collab-text", "collab-image", "text-image", "text-multimodal"]
    y = np.arange(len(rows))[::-1]
    vals = [align[align.pair == p].iloc[0].cca_mean for p in rows]
    r2 = [align[align.pair == p].iloc[0].procrustes_r2 for p in rows]
    cols = [COLORS["collab"], COLORS["collab"], COLORS["text"], COLORS["text"]]
    ax.barh(y, vals, height=0.5, color=cols, alpha=0.9,
            edgecolor="white", linewidth=0.6, zorder=3)
    for yi, v, rr in zip(y, vals, r2):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.0)
        ax.text(0.025, yi, f"R²={rr:.2f}", va="center", ha="left",
                fontsize=7.2, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([PAIR_LABEL[p] for p in rows])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("mean CCA")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_title("f   No global linear alignment", loc="left", pad=7)
    _clean(ax)


def main():
    stats, align, dims, ari, C = load_numbers()
    cum = effective_dim_curves()

    fig = plt.figure(figsize=(14.0, 8.2))
    gs = fig.add_gridspec(
        2, 3, left=0.065, right=0.985, bottom=0.085, top=0.875,
        wspace=0.30, hspace=0.52,
        width_ratios=[1.12, 0.94, 1.06],
        height_ratios=[1.0, 1.0],
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

    panel_cka(axes[0, 0], C)
    panel_fused(axes[0, 1], C)
    panel_sid(axes[0, 2], ari)
    panel_overlap(axes[1, 0], stats)
    panel_dims(axes[1, 1], cum, dims)
    panel_align(axes[1, 2], align)

    # row headers
    fig.text(0.065, 0.955, "CHALLENGE 1", fontsize=13, fontweight="bold",
             color=COLORS["text"], ha="left", va="center")
    fig.text(0.175, 0.955, "Modality dominance & redundancy", fontsize=12.2,
             fontweight="bold", color="#1A1A1A", ha="left", va="center")
    fig.text(0.065, 0.925, "Text dominates; the visual signal is under-used.",
             fontsize=9.0, color="#555555", ha="left", va="center")

    fig.text(0.065, 0.495, "CHALLENGE 2", fontsize=13, fontweight="bold",
             color=COLORS["collab"], ha="left", va="center")
    fig.text(0.175, 0.495, "Semantic–collaborative fusion", fontsize=12.2,
             fontweight="bold", color="#1A1A1A", ha="left", va="center")
    fig.text(0.065, 0.465, "Different geometry; barely overlapping neighbourhoods.",
             fontsize=9.0, color="#555555", ha="left", va="center")

    # subtle separators
    fig.add_artist(Line2D([0.065, 0.985], [0.912, 0.912], color="#DDDDDD", lw=0.9))
    fig.add_artist(Line2D([0.065, 0.985], [0.452, 0.452], color="#DDDDDD", lw=0.9))

    png = os.path.join(FIGURES, "fig_two_challenges_combined.png")
    pdf = os.path.join(FIGURES, "fig_two_challenges_combined.pdf")
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print("saved", png, pdf, sep="\n")


if __name__ == "__main__":
    main()
