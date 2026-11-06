"""Geometry / distribution / alignment analysis (Challenge 1, precise)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import skew
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, l2n, load_collab, load_emb

N, K, KNN = 3000, 50, 10

def eff_dim(X):
    Xc = X - X.mean(0, keepdims=True)
    p = PCA(n_components=min(500, Xc.shape[1], len(Xc) - 1), svd_solver="randomized", random_state=42).fit(Xc)
    lam = p.explained_variance_ratio_
    return float(1 / np.sum(lam ** 2)), int(np.searchsorted(np.cumsum(lam), 0.9) + 1), np.cumsum(lam)

def mle_id(X, k=KNN):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    r = np.sort(d[:, 1:], axis=1) + 1e-12
    return float(np.mean(1 / (np.mean(np.log(r[:, -1:] / r[:, :-1]), axis=1) + 1e-12)))

def knn_geo(X, k=KNN):
    d, i = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    occ = np.bincount(i[:, 1:].ravel(), minlength=len(X)).astype(float)
    return float(d[:, 1].mean()), float(skew(occ))

def align(A, B, k=K):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    A5 = PCA(n_components=k, svd_solver="randomized", random_state=42).fit_transform(A)
    B5 = PCA(n_components=k, svd_solver="randomized", random_state=42).fit_transform(B)
    R, _ = orthogonal_procrustes(A5, B5)
    r2 = float(1 - np.sum((B5 - A5 @ R) ** 2) / np.sum(B5 ** 2))
    cca = CCA(n_components=10, max_iter=2000).fit(A5, B5)
    xs, ys = cca.transform(A5, B5)
    cc = [float(np.corrcoef(xs[:, i], ys[:, i])[0, 1]) for i in range(xs.shape[1])]
    return r2, cc

def main():
    rng = np.random.default_rng(0)
    reps = {"collab": l2n(load_collab()), "text": l2n(load_emb("text_only")),
            "image": l2n(load_emb("image_only")), "multimodal": l2n(load_emb("multimodal"))}
    sample = np.sort(rng.choice(len(reps["text"]), N, replace=False))
    X = {k: v[sample] for k, v in reps.items()}
    dims = {}
    for k, v in X.items():
        pr, d90, cum = eff_dim(v); d1, hs = knn_geo(v)
        dims[k] = {"PR": pr, "d90": d90, "d1": d1, "hub_skew": hs, "cum": cum}
        print(k, {kk: round(dims[k][kk], 3) for kk in ["PR", "d90", "d1", "hub_skew"]}, flush=True)
    pairs = [("collab", "text"), ("collab", "image"), ("text", "image"), ("text", "multimodal")]
    al = {}
    for a, b in pairs:
        r2, cc = align(X[a], X[b])
        al[f"{a}-{b}"] = {"procrustes_r2": r2, "cca_mean": float(np.mean(cc)), "cca_top": cc[0], "cc_curve": cc}
        print(f"{a}-{b}: procrustes_r2={r2:.3f} cca_mean={np.mean(cc):.3f} cca_top={cc[0]:.3f}", flush=True)
    pd.DataFrame([{"space": k, **{kk: dims[k][kk] for kk in ["PR", "d90", "d1", "hub_skew"]}} for k in dims]).to_csv(
        os.path.join(RESULTS, "exp_a_geometry_dims.csv"), index=False)
    pd.DataFrame([{"pair": p, **{kk: v[kk] for kk in ["procrustes_r2", "cca_mean", "cca_top"]}} for p, v in al.items()]).to_csv(
        os.path.join(RESULTS, "exp_a_geometry_alignment.csv"), index=False)
    json.dump({"dims": {k: {kk: v for kk, v in d.items() if kk != "cum"} for k, d in dims.items()},
               "alignment": {p: {kk: v for kk, v in a.items() if kk != "cc_curve"} for p, a in al.items()}},
              open(os.path.join(RESULTS, "exp_a_geometry.json"), "w"), indent=2)

    spaces = ["collab", "text", "image", "multimodal"]
    colors = {"collab": "#C44E52", "text": "#4C72B0", "image": "#55A868", "multimodal": "#8172B2"}
    names = list(al.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8))
    ax = axes[0]
    for k in spaces:
        c = dims[k]["cum"]
        ax.plot(np.arange(1, len(c) + 1), c, color=colors[k],
                label=f"{k} (PR={dims[k]['PR']:.0f}, d90={dims[k]['d90']})")
    ax.axhline(0.9, ls=":", color="gray"); ax.set_xscale("log")
    ax.set_xlabel("# PCs"); ax.set_ylabel("cumulative explained variance")
    ax.set_title("Effective dimensionality", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    for p, c in zip(names, ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]):
        ax.plot(range(1, 11), al[p]["cc_curve"], marker="o", ms=3, color=c, label=p)
    ax.set_xlabel("canonical component"); ax.set_ylabel("canonical correlation")
    ax.set_title("CCA: linear alignment strength", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[2]
    r2 = [al[p]["procrustes_r2"] for p in names]; cca = [al[p]["cca_mean"] for p in names]
    xs = np.arange(len(names)); w = 0.38
    ax.bar(xs - w / 2, r2, w, label="Procrustes R²", color="#4C72B0")
    ax.bar(xs + w / 2, cca, w, label="mean CCA", color="#DD8452")
    for xi, v in zip(xs - w / 2, r2):
        ax.text(xi, v - 0.12 if v < 0 else v + 0.02, f"{v:.2f}", ha="center",
                va="top" if v < 0 else "bottom", fontsize=6.5)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(min(r2) - 0.6, 1.02); ax.set_title("Linear alignability: signed Procrustes R² / mean CCA", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Distribution / geometry: different intrinsic dimensionality and hard-to-align subspaces", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "fig_preliminary_geometry.png"), dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ax = axes[0]
    for k in spaces:
        c = dims[k]["cum"]
        ax.plot(np.arange(1, len(c) + 1), c, color=colors[k], label=f"{k} (d90={dims[k]['d90']})")
    ax.axhline(0.9, ls=":", color="gray"); ax.set_xscale("log")
    ax.set_xlabel("# PCs"); ax.set_ylabel("cum. explained var."); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("Effective dimensionality", fontsize=9)
    ax = axes[1]
    ax.bar(xs - w / 2, r2, w, label="Procrustes R²", color="#4C72B0")
    ax.bar(xs + w / 2, cca, w, label="mean CCA", color="#DD8452")
    for xi, v in zip(xs - w / 2, r2):
        ax.text(xi, v - 0.12 if v < 0 else v + 0.02, f"{v:.2f}", ha="center",
                va="top" if v < 0 else "bottom", fontsize=6.5)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(min(r2) - 0.6, 1.02); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Linear alignability (signed)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "fig_preliminary_geometry_v2.png"), dpi=220); plt.close(fig)
    print("GEOMETRY_DONE")

if __name__ == "__main__":
    main()
