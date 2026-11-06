"""Statistical evidence for semantic-collaborative misalignment.

Two complementary tests, both with permutation nulls that destroy the item
correspondence while preserving each space's own geometry:

1) Mantel test: Pearson correlation between the upper-triangular similarity
   entries of two spaces; null by permuting item labels of the second space.
2) Neighbourhood-overlap permutation test: mean Top-K overlap; null by pairing
   each anchor with another (random) item's neighbour list in the second space.

A positive control (text vs multimodal, known to be strongly related) shows what
a genuinely aligned pair looks like.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb  # noqa: E402

N_SAMPLE = 1200
K = 10
N_PERM = 500
PAIRS = [("collab", "text"), ("collab", "image"), ("collab", "multimodal"),
         ("text", "image"), ("text", "multimodal")]


def mantel(Sa, Sb, rng, n_perm=N_PERM):
    n = len(Sa)
    u = np.triu_indices(n, 1)
    a = Sa[u]
    b = Sb[u]
    def pear(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))
    obs = pear(a, b)
    null = np.empty(n_perm)
    for t in range(n_perm):
        p = rng.permutation(n)
        null[t] = pear(a, Sb[np.ix_(p, p)][u])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    pval = (1.0 + float(np.sum(null >= obs))) / (1.0 + n_perm)
    return obs, float(null.mean()), float(null.std()), z, pval, null


def topk_for_sample(Xs, Xall, self_idx=None, k=K):
    """Neighbour lists for the sampled anchors against all items (self excluded)."""
    sim = Xs @ Xall.T
    if self_idx is not None:
        sim[np.arange(len(Xs)), self_idx] = -2.0
    return np.argpartition(-sim, k, axis=1)[:, :k]


def main():
    rng = np.random.default_rng(0)
    reps = {
        "collab": l2n(load_collab()),
        "text": l2n(load_emb("text_only")),
        "image": l2n(load_emb("image_only")),
        "multimodal": l2n(load_emb("multimodal")),
    }
    n_all = len(reps["text"])
    sample = np.sort(rng.choice(n_all, size=N_SAMPLE, replace=False))
    S = {k: reps[k][sample] @ reps[k][sample].T for k in reps}
    nb = {k: topk_for_sample(reps[k][sample], reps[k], self_idx=sample) for k in reps}

    rows = []
    mantel_nulls = {}
    for a, b in PAIRS:
        r, nm, ns, z, pv, null = mantel(S[a], S[b], rng)
        mantel_nulls[(a, b)] = null
        # neighbourhood overlap: observed vs permutation null
        obs = np.mean([len(np.intersect1d(nb[a][i], nb[b][i])) / K for i in range(N_SAMPLE)])
        null_ov = np.empty(N_PERM)
        for t in range(N_PERM):
            perm = rng.permutation(N_SAMPLE)
            null_ov[t] = np.mean([len(np.intersect1d(nb[a][i], nb[b][perm[i]])) / K for i in range(N_SAMPLE)])
        z_ov = (obs - null_ov.mean()) / (null_ov.std() + 1e-12)
        p_ov = (1.0 + float(np.sum(null_ov >= obs))) / (1.0 + N_PERM)
        rows.append({"pair": f"{a}-{b}", "mantel_r": r, "mantel_null_mean": nm, "mantel_null_std": ns,
                     "mantel_z": z, "mantel_p": pv, "overlap_obs": obs,
                     "overlap_null_mean": float(null_ov.mean()), "overlap_null_std": float(null_ov.std()),
                     "overlap_z": z_ov, "overlap_p": p_ov})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_a_statistical_tests.csv"), index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))

    # (a) pairwise similarity scatter: weak pair vs positive control
    ax = axes[0]
    pairs_show = [("collab", "text"), ("text", "multimodal")]
    colors = ["#C44E52", "#4C72B0"]
    u = np.triu_indices(N_SAMPLE, 1)
    idx = rng.choice(len(u[0]), size=min(40000, len(u[0])), replace=False)
    for (a, b), c in zip(pairs_show, colors):
        r = float(df[(df.pair == f"{a}-{b}")].mantel_r.iloc[0])
        ax.scatter(S[a][u][idx], S[b][u][idx], s=2, alpha=0.18, color=c,
                   label=f"{a} vs {b} (Mantel r={r:.3f})")
    ax.set_xlabel("cosine similarity in space A"); ax.set_ylabel("cosine similarity in space B")
    ax.set_title("Pairwise similarity agreement", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, markerscale=4)

    # (b) overlap observed vs permutation null
    ax = axes[1]
    xs = np.arange(len(PAIRS))
    obs = [df[df.pair == f"{a}-{b}"].overlap_obs.iloc[0] for a, b in PAIRS]
    nul = [df[df.pair == f"{a}-{b}"].overlap_null_mean.iloc[0] for a, b in PAIRS]
    ax.bar(xs - 0.2, obs, 0.4, label="observed", color="#4C72B0")
    ax.bar(xs + 0.2, nul, 0.4, label="permutation null", color="lightgray")
    ax.set_xticks(xs); ax.set_xticklabels([f"{a}-{b}" for a, b in PAIRS], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(f"Top-{K} overlap"); ax.set_title("Neighbour overlap vs null", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for i, v in enumerate(obs):
        ax.text(i - 0.2, v + 0.002, f"{v:.3f}", ha="center", fontsize=6)

    # (c) Mantel null distributions
    ax = axes[2]
    for (a, b), c in zip(pairs_show, colors):
        null = mantel_nulls[(a, b)]
        ax.hist(null, bins=30, alpha=0.5, color=c, label=f"{a}-{b} null")
        r = float(df[(df.pair == f"{a}-{b}")].mantel_r.iloc[0])
        ax.axvline(r, color=c, lw=2)
    ax.set_xlabel("Mantel r"); ax.set_ylabel("#permutations")
    ax.set_title("Permutation null of Mantel r", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle("Statistical evidence: collaborative and semantic similarity structures are weakly aligned\n"
                 "(text vs multimodal is the positive control)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_statistical.png"), dpi=220)
    plt.close(fig)

    # single-column compact candidate
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for (a, b), c in zip(pairs_show, colors):
        r = float(df[(df.pair == f"{a}-{b}")].mantel_r.iloc[0])
        ax.scatter(S[a][u][idx], S[b][u][idx], s=2, alpha=0.18, color=c,
                   label=f"{a} vs {b} (r={r:.3f})")
    ax.set_xlabel("cosine similarity A"); ax.set_ylabel("cosine similarity B")
    ax.set_title("Similarity agreement (Mantel)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, markerscale=4)
    ax = axes[1]
    ax.bar(xs - 0.2, obs, 0.4, label="observed", color="#4C72B0")
    ax.bar(xs + 0.2, nul, 0.4, label="null", color="lightgray")
    ax.set_xticks(xs); ax.set_xticklabels([f"{a}-{b}" for a, b in PAIRS], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(f"Top-{K} overlap"); ax.set_title("Neighbour overlap vs null", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_statistical_v2.png"), dpi=220)
    plt.close(fig)
    print("STATS_DONE")


if __name__ == "__main__":
    main()
