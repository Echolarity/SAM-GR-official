"""Build the SAM-GR preliminary / motivation figure from the Sports strict-10-core artifacts.

Outputs (analysis/samgr_sports10/figures):
  fig_preliminary_v1.png       2x3 layout (paper candidate, detailed)
  fig_preliminary_v2.png       1x3 layout (paper candidate, compact)
  fig_preliminary_cka.png      cross-space linear CKA heatmap
  fig_preliminary_case.png     case study: text vs collaborative neighbours
  preliminary_metrics.csv      underlying numbers
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
from common import (FIGURES, RESULTS, l2n, load_collab, load_emb,  # noqa: E402
                    topk_neighbors)

SPACES = ["collab", "text", "image", "multimodal", "fused"]
COLORS = {"collab": "#C44E52", "text": "#4C72B0", "image": "#55A868",
          "multimodal": "#8172B2", "fused": "#CCB974"}


def cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx, Ky = X @ X.T, Y @ Y.T
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    def h(K):
        A = H @ K @ H
        return float(np.sum(A * A))
    return h(Kx @ Ky) / np.sqrt(h(Kx @ Kx) * h(Ky @ Ky))


def main():
    rng = np.random.default_rng(0)
    reps = {
        "collab": l2n(load_collab()),
        "text": l2n(load_emb("text_only")),
        "image": l2n(load_emb("image_only")),
        "multimodal": l2n(load_emb("multimodal")),
        "fused": l2n(load_emb("csga_moe_collab_v2")),
    }
    n = len(reps["text"])
    sample = rng.choice(n, size=1500, replace=False)

    # ---- CKA matrix ----
    C = np.eye(len(SPACES))
    for a in range(len(SPACES)):
        for b in range(a + 1, len(SPACES)):
            C[a, b] = C[b, a] = cka(reps[SPACES[a]][sample], reps[SPACES[b]][sample])
    print("CKA matrix:\n", pd.DataFrame(C, index=SPACES, columns=SPACES).round(4))

    # ---- per-item dependence of fused on each modality ----
    dep = {}
    for name in ["text", "image", "multimodal"]:
        c = np.sum(reps["fused"] * reps[name], axis=1)
        dep[name] = c
    dep_stats = {k: {"mean": float(v.mean()), "median": float(np.median(v)),
                     "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
                 for k, v in dep.items()}

    # ---- experiment artefacts ----
    ov = pd.read_csv(os.path.join(RESULTS, "exp_a_neighborhood_overlap.csv"))
    ari = pd.read_csv(os.path.join(RESULTS, "exp_b_partition_agreement.csv"))
    omoe = json.load(open(os.path.join(RESULTS, "exp_d_omoe_stats.json")))

    # ---- case-study anchor (disjoint text/collab neighbourhoods) ----
    text_nb, text_cos = topk_neighbors(reps["text"], 10)
    coll_nb, coll_cos = topk_neighbors(reps["collab"], 10)
    best = None
    for i in range(n):
        inter = len(np.intersect1d(text_nb[i], coll_nb[i]))
        if inter == 0:
            score = text_cos[i].mean() + coll_cos[i].mean()
            if best is None or score > best[0]:
                best = (score, i)
    _, anchor = best
    # joint projection: z-score + PCA each space, concat, t-SNE
    sub = list(dict.fromkeys([anchor] + list(text_nb[anchor]) + list(coll_nb[anchor])))
    others = [j for j in rng.choice(n, 300, replace=False) if j not in sub and j != anchor][:200]
    subset = sub + others
    def proj(x):
        z = (x[subset] - x[subset].mean(0, keepdims=True)) / (x[subset].std(0, keepdims=True) + 1e-9)
        return PCA(n_components=50, random_state=42).fit_transform(z)
    joint = np.concatenate([proj(reps["text"]), proj(reps["collab"])], axis=1)
    emb = TSNE(n_components=2, perplexity=20, init="pca", random_state=42,
               max_iter=1000).fit_transform(joint)
    pos = {g: k for k, g in enumerate(subset)}
    text_set = set(int(j) for j in text_nb[anchor])
    coll_set = set(int(j) for j in coll_nb[anchor])

    # ---------- figures ----------
    def panel_case(ax):
        ax.scatter(emb[[pos[o] for o in others], 0], emb[[pos[o] for o in others], 1],
                   s=8, color="lightgray", alpha=0.6, label="other items")
        tx = [pos[int(j)] for j in text_nb[anchor]]
        cx = [pos[int(j)] for j in coll_nb[anchor]]
        ax.scatter(emb[tx, 0], emb[tx, 1], s=55, color=COLORS["text"],
                   edgecolor="white", linewidth=0.5, label="text top-10 neighbours")
        ax.scatter(emb[cx, 0], emb[cx, 1], s=55, color=COLORS["collab"], marker="^",
                   edgecolor="white", linewidth=0.5, label="collab top-10 neighbours")
        ax.scatter(emb[pos[anchor], 0], emb[pos[anchor], 1], s=160, marker="*",
                   color="black", label="anchor item", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Case study: collaborative vs semantic neighbours", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="best")

    def panel_overlap(ax):
        ks = [5, 10, 20, 50]
        for pair, col, mk in [(("collab", "text"), COLORS["text"], "o"),
                              (("collab", "visual"), COLORS["collab"], "^"),
                              (("text", "visual"), COLORS["image"], "s")]:
            a, b = pair
            sub = ov[(ov.rep_a == a) & (ov.rep_b == b)]
            ax.plot(ks, [sub[sub.k == k].overlap_mean.iloc[0] for k in ks],
                    marker=mk, color=col, label=f"{a} vs {b}")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylabel("mean neighbour overlap")
        ax.set_title("Neighbourhood overlap across spaces", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)

    def panel_cka(ax):
        im = ax.imshow(C, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(SPACES))); ax.set_xticklabels(SPACES, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(SPACES))); ax.set_yticklabels(SPACES, fontsize=7)
        for p in range(len(SPACES)):
            for q in range(len(SPACES)):
                ax.text(q, p, f"{C[p,q]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if C[p, q] < 0.5 else "black")
        ax.set_title("Linear CKA between representation spaces", fontsize=9)
        return im

    def panel_dep(ax):
        data = [dep[k] for k in ["text", "image", "multimodal"]]
        bp = ax.boxplot(data, labels=["fused vs text", "fused vs image", "fused vs multi"],
                        widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], [COLORS["text"], COLORS["image"], COLORS["multimodal"]]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_ylabel("per-item cosine")
        ax.set_title("Fused representation is text-dominated", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)

    def panel_ari(ax):
        x = np.arange(len(ari))
        ax.bar(x, ari["ARI"], color="#4C72B0", label="ARI")
        ax.plot(x, ari["NMI"], color="#C44E52", marker="o", label="NMI")
        ax.set_xticks(x); ax.set_xticklabels(ari["prefix_level"].map(lambda l: ["a", "a+b", "a+b+c", "SID"][l - 1]))
        ax.set_ylim(0, 1.05)
        ax.set_title("Full vs text-only SID partition agreement", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)

    def panel_omoe(ax):
        names = list(omoe.keys())
        sims = [np.mean(np.abs(np.array(omoe[k]["expert_sim"])[~np.eye(5, dtype=bool)])) for k in names]
        ents = [omoe[k]["mean_gate_entropy"] for k in names]
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w / 2, sims, w, color="#C44E52", label="expert mean |cos|")
        ax.bar(x + w / 2, np.array(ents) / np.log(5), w, color="#55A868", label="gate entropy / ln5")
        ax.set_xticks(x); ax.set_xticklabels(["full", "no ortho", "no load"], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("OMoE: orthogonality & load balancing", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)

    # v1: 2x3 detailed
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    panel_overlap(axes[0, 0]); panel_case(axes[0, 1]); panel_cka(axes[0, 2])
    panel_dep(axes[1, 0]); panel_ari(axes[1, 1]); panel_omoe(axes[1, 2])
    fig.suptitle("Preliminary evidence: semantic-collaborative misalignment, modality dominance, and expert correlation",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_v1.png"), dpi=220)
    plt.close(fig)

    # v2: compact 1x3
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    panel_overlap(axes[0]); panel_dep(axes[1]); panel_omoe(axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_v2.png"), dpi=220)
    plt.close(fig)

    # individual candidates
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = panel_cka(ax); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_preliminary_cka.png"), dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    panel_case(ax); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_case.png"), dpi=220); plt.close(fig)

    # metrics table
    rows = []
    for a in range(len(SPACES)):
        for b in range(a + 1, len(SPACES)):
            rows.append({"type": "cka", "a": SPACES[a], "b": SPACES[b], "value": float(C[a, b])})
    for k, v in dep_stats.items():
        rows.append({"type": "fused_dependence", "a": k, "b": "fused", "value": v["mean"]})
    for _, r in ari.iterrows():
        rows.append({"type": "sid_partition_ari", "a": "full", "b": f"text_only_level_{r.prefix_level}", "value": r.ARI})
    for name in omoe:
        off = np.array(omoe[name]["expert_sim"])[~np.eye(5, dtype=bool)]
        rows.append({"type": "omoe_expert_mean_abs_cos", "a": name, "b": "", "value": float(np.mean(np.abs(off)))})
        rows.append({"type": "omoe_gate_entropy", "a": name, "b": "", "value": omoe[name]["mean_gate_entropy"]})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "preliminary_metrics.csv"), index=False)
    print("anchor item:", anchor, "text/collab neighbour overlap:", len(text_set & coll_set))
    print("PRELM_FIG_DONE")


if __name__ == "__main__":
    main()
