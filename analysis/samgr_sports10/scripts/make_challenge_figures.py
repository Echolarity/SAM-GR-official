"""Publication figures for the two paper challenges (Sports strict 10-core).

Challenge 1 (semantic-collaborative misalignment):
  (a) top-10 neighbour overlap vs permutation null + Mantel r
  (b) effective dimensionality (PCA cumulative variance)
  (c) global linear alignability (Procrustes R^2, mean CCA)

Challenge 2 (modality dominance / under-utilization):
  (d) linear CKA between spaces
  (e) CKA of the fused representation to text / image / multimodal
  (f) SID partition agreement (ARI/NMI per prefix level)

Outputs:
  figures/fig_challenge1_misalignment.png
  figures/fig_challenge2_modality.png
  figures/fig_two_challenges.png
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
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb  # noqa: E402

N_SAMPLE = 3000
SPACES = ["collab", "text", "image", "multimodal", "fused"]
COLORS = {"collab": "#C44E52", "text": "#4C72B0", "image": "#55A868",
          "multimodal": "#8172B2", "fused": "#B8860B"}
PAIRS = [("collab", "text"), ("collab", "image"), ("collab", "multimodal"),
         ("text", "image"), ("text", "multimodal")]
PAIR_LABEL = {"collab-text": "collab–text", "collab-image": "collab–image",
              "collab-multimodal": "collab–multimodal", "text-image": "text–image",
              "text-multimodal": "text–multimodal"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def effective_dim_curves():
    rng = np.random.default_rng(0)
    reps = {"collab": l2n(load_collab()), "text": l2n(load_emb("text_only")),
            "image": l2n(load_emb("image_only")),
            "multimodal": l2n(load_emb("multimodal"))}
    sample = np.sort(rng.choice(len(reps["text"]), N_SAMPLE, replace=False))
    out = {}
    for k, v in reps.items():
        x = v[sample].astype(np.float64)
        x = x - x.mean(0, keepdims=True)
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
    sub = cka[cka.type == "cka"]
    for _, r in sub.iterrows():
        C.loc[r.a, r.b] = C.loc[r.b, r.a] = float(r.value)
    return stats, align, dims, ari, C


def panel_overlap(ax, stats):
    obs = [stats.loc[stats.pair == f"{a}-{b}", "overlap_obs"].iloc[0] for a, b in PAIRS]
    nul = [stats.loc[stats.pair == f"{a}-{b}", "overlap_null_mean"].iloc[0] for a, b in PAIRS]
    mr = [stats.loc[stats.pair == f"{a}-{b}", "mantel_r"].iloc[0] for a, b in PAIRS]
    cols = [COLORS[a] for a, _ in PAIRS]
    x = np.arange(len(PAIRS))
    w = 0.36
    ax.bar(x - w / 2, obs, w, color=cols, edgecolor="white", linewidth=0.4,
           label="observed", zorder=3)
    ax.bar(x + w / 2, nul, w, color="#C8C8C8", edgecolor="white", linewidth=0.4,
           label="permutation null", zorder=3)
    for xi, v, m in zip(x - w / 2, obs, mr):
        ax.text(xi, v * 1.16, f"{v * 100:.1f}%\nr={m:.2f}", ha="center", va="bottom",
                fontsize=6.9, linespacing=1.15)
    ax.set_yscale("log")
    ax.set_ylim(3.5e-4, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["collab–text", "collab–image", "collab–multimodal",
                        "text–image", "text–multimodal"], rotation=20, ha="right", fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("mean top-10 neighbour overlap (log scale)")
    ax.set_title("(a) Collab–semantic neighborhoods barely overlap", loc="left")
    ax.axhline(nul[0], color="#8A8A8A", lw=0.7, ls=":", zorder=1)
    ax.text(4.47, nul[0] * 1.25, "chance", ha="right", fontsize=6.8, color="#666666")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2)


def panel_dims(ax, cum, dims):
    for k in ["collab", "text", "image", "multimodal"]:
        row = dims[dims.space == k].iloc[0]
        c = cum[k]
        ax.plot(np.arange(1, len(c) + 1), c, color=COLORS[k], lw=1.5,
                label=f"{k}  PR={row.PR:.0f}, d90={int(row.d90)}")
    ax.axhline(0.9, color="#8A8A8A", lw=0.7, ls=":")
    ax.text(1.05, 0.905, "90% variance", fontsize=6.8, color="#666666", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1, 500)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("# principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("(b) Different effective dimensionality", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=6.6)


def panel_align(ax, align):
    rows = [f"{a}-{b}" for a, b in
            [("collab", "text"), ("collab", "image"), ("text", "image"),
             ("text", "multimodal")]]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-6.2, 0, color="#F0F0F0", zorder=0)
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)
    for yi, p in zip(y, rows):
        r = align[align.pair == p].iloc[0]
        ax.plot([r.procrustes_r2, r.cca_mean], [yi, yi], color="#9A9A9A", lw=1.0, zorder=2)
        ax.scatter(r.procrustes_r2, yi, s=46, color="#4C72B0", zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.scatter(r.cca_mean, yi, s=46, color="#DD8452", zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.text(r.procrustes_r2 - 0.12, yi + 0.16, f"{r.procrustes_r2:.2f}",
                ha="right", va="center", fontsize=7, color="#2F4F6F")
        ax.text(r.cca_mean + 0.05, yi + 0.16, f"{r.cca_mean:.2f}",
                ha="left", va="center", fontsize=7, color="#8A4B1E")
    ax.set_yticks(y)
    ax.set_yticklabels([PAIR_LABEL[p] for p in rows])
    ax.set_xlim(-6.2, 1.12)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("score (higher = more linearly alignable)")
    ax.set_title("(c) Collab–semantic: no global linear map", loc="left")
    ax.text(-6.05, 3.38, "R² < 0: worse than predicting the mean", fontsize=6.6, color="#555555")
    handles = [Line2D([], [], marker="o", ls="", color="#4C72B0", ms=6, label="Procrustes R²"),
               Line2D([], [], marker="o", ls="", color="#DD8452", ms=6, label="mean CCA")]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01),
              fontsize=6.8)


def panel_cka(ax, C):
    im = ax.imshow(C.values, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(SPACES)))
    ax.set_xticklabels(SPACES, rotation=35, ha="right")
    ax.set_yticks(range(len(SPACES)))
    ax.set_yticklabels(SPACES)
    for i in range(len(SPACES)):
        for j in range(len(SPACES)):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.55 else "black")
    ax.set_title("(d) Cross-space linear CKA", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    ax.add_patch(plt.Rectangle((1.5, 0.5), 1, 1, fill=False, edgecolor="#2ECC71", lw=1.6))
    ax.text(0.5, -0.30, "text–image CKA = 0.02: complementary, not redundant",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color="#1E7A3C")


def panel_fused(ax, C):
    names = ["text", "image", "multimodal"]
    vals = [C.loc["fused", n] for n in names]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, height=0.52,
                   color=[COLORS[n] for n in names], edgecolor="white")
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("CKA with fused representation")
    ax.set_title("(e) Text dominates; image is ignored", loc="left")
    ax.text(0.98, 0.30, "image CKA ≈ 10× smaller\nthan text (text-dominated fusion)",
            transform=ax.transAxes, ha="right", fontsize=6.6, color="#444444", va="center")
    for s in ["left"]:
        ax.spines[s].set_visible(True)


def panel_ari(ax, ari):
    x = np.arange(len(ari))
    ax.bar(x - 0.19, ari.ARI, 0.36, color="#4C72B0", label="ARI", zorder=3)
    ax.bar(x + 0.19, ari.NMI, 0.36, color="#DD8452", label="NMI", zorder=3)
    for xi, v in zip(x - 0.19, ari.ARI):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    ax.set_ylim(0, 1.13)
    ax.set_xticks(x)
    ax.set_xticklabels(["prefix a", "a+b", "a+b+c", "final SID"])
    ax.set_ylabel("agreement (full vs text-only)")
    ax.set_title("(f) Final SID partition identical (ARI 0.999)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7)


def build_separate(cum, stats, align, dims, ari, C):
    # ---------------- Challenge 1 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.75),
                             gridspec_kw={"width_ratios": [1.32, 1.0, 1.12]})
    panel_overlap(axes[0], stats)
    panel_dims(axes[1], cum, dims)
    panel_align(axes[2], align)
    fig.suptitle("Challenge 1 — Semantic–collaborative misalignment "
                 "(Sports, strict 10-core)", fontsize=11.5, y=1.02)
    fig.tight_layout(w_pad=2.0)
    p1 = os.path.join(FIGURES, "fig_challenge1_misalignment.png")
    fig.savefig(p1)
    plt.close(fig)

    # ---------------- Challenge 2 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8),
                             gridspec_kw={"width_ratios": [1.06, 0.9, 1.04]})
    panel_cka(axes[0], C)
    panel_fused(axes[1], C)
    panel_ari(axes[2], ari)
    fig.suptitle("Challenge 2 — Modality dominance and visual under-utilization "
                 "(Sports, strict 10-core)", fontsize=11.5, y=1.02)
    fig.tight_layout(w_pad=2.2)
    p2 = os.path.join(FIGURES, "fig_challenge2_modality.png")
    fig.savefig(p2)
    plt.close(fig)

    # ---------------- Combined one-stop version ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.7),
                             gridspec_kw={"width_ratios": [1.32, 1.0, 1.12],
                                          "hspace": 0.62, "wspace": 0.30})
    panel_overlap(axes[0, 0], stats)
    panel_dims(axes[0, 1], cum, dims)
    panel_align(axes[0, 2], align)
    panel_cka(axes[1, 0], C)
    panel_fused(axes[1, 1], C)
    panel_ari(axes[1, 2], ari)
    fig.text(0.5, 0.975, "Challenge 1 — semantic–collaborative misalignment: different geometry, "
             "almost disjoint neighborhoods, no global linear alignment",
             ha="center", fontsize=11)
    fig.text(0.5, 0.485, "Challenge 2 — modality dominance: text drives the fused space, "
             "visual information is under-utilized",
             ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p3 = os.path.join(FIGURES, "fig_two_challenges.png")
    fig.savefig(p3)
    plt.close(fig)
    print("saved", p1, p2, p3, sep="\n")


def main():
    stats, align, dims, ari, C = load_numbers()
    cum = effective_dim_curves()
    print(C.round(3))
    build_separate(cum, stats, align, dims, ari, C)
    print("CHALLENGE_FIGURES_DONE")


if __name__ == "__main__":
    main()
