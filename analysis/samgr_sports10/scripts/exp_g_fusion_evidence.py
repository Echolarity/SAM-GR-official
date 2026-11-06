"""Experiment G: can CKA / neighbour overlap validate the fused representation?

Two analyses, on the *saved item representations* (no retraining):

1. Standard linear CKA (HSIC) between each representation and
   (a) the direct multimodal embedding  -> semantic retention
   (b) the SASRec collaborative embedding -> collaborative alignment.
2. Top-k neighbour overlap: for each representation, the fraction of its
   semantic neighbours (top-k of direct multimodal) and collaborative neighbours
   (top-k of SASRec) that it recovers, vs a random-neighbour baseline.

Both are global/local *representation* diagnostics, not causal proofs.  The
downstream HR@20 panel is the utility check.

Representations and provenance are documented in
`results/exp_g_provenance.md`.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
sys.path.insert(0, HERE)
from common import l2n, overlap_at_k, topk_neighbors  # noqa: E402

B = "/data02/sht/SAM-GR"
SPORTS = {
    "name": "Sports_and_Outdoors",
    "root": f"{B}/data/Amazon/Sports_and_Outdoors/Sports_and_Outdoors.emb-Qwen3VL-%s.npy",
    "collab": f"{B}/sasrec_models/Sports_and_Outdoors/sasrec_item_emb.npy",
    "files": {
        "text": "text_only",
        "image": "image_only",
        "multimodal": "multimodal",
        "fused": "csga_moe_collab_v2",
    },
    "n": 11246,
}
INDSCI = {
    "name": "Industrial_and_Scientific",
    "root": f"{B}/data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.emb-Qwen3VL-%s.npy",
    "collab": f"{B}/sasrec_models/Industrial_and_Scientific/sasrec_item_emb.npy",
    "files": {
        "text": "text_only",
        "image": "image_only",
        "multimodal": "multimodal",
        "csga_collab": "csga_collab",
        "moe_collab": "moe_collab",
        "csga_moe_collab": "csga_moe_collab",
        "fused": "csga_moe_collab_v2",
    },
    "n": 27222,
}

# Controlled downstream runs on IndSci (task1, 2 epochs, same SFT recipe).
INDSCI_HR = {
    "text-only": "Industrial_and_Scientific-sft-dsz2-t1",
    "image-only": "Industrial_and_Scientific-sft-dsz2_image_only",
    "multimodal": "Industrial_and_Scientific-sft-dsz2_mm",
    "csga_collab": "Industrial_and_Scientific-sft-dsz2_csga_alone",
    "csga_moe_collab": "Industrial_and_Scientific-sft-dsz2_csga_omoe",
    "fused": "Industrial_and_Scientific-sft-dsz2_csga_omoe_v2",
}
RESULT_BASE = f"{B}/results/qwen2.5-0.5b/full"
K = 10
CKA_N = 1500
CKA_BOOT_N = 1000
CKA_BOOT = 60
OV_BOOT = 3000
RNG = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 9.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})
COLORS = {
    "multimodal": "#4C72B0",
    "csga_collab": "#55A868",
    "csga_moe_collab": "#8172B2",
    "fused": "#B8860B",
    "moe_collab": "#8C8C8C",
    "text": "#4C72B0",
    "image": "#55A868",
}


def load_dataset(cfg):
    R = {}
    for name, suffix in cfg["files"].items():
        R[name] = l2n(np.load(cfg["root"] % suffix).astype(np.float32))
    R["collab"] = l2n(np.load(cfg["collab"]).astype(np.float32))
    return R


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ Kx @ H, H @ Ky @ H
    return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))


def cka_matrix(R, names):
    idx = RNG.choice(len(R["multimodal"]), CKA_N, replace=False)
    M = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            M[i, j] = M[j, i] = cka(R[names[i]][idx], R[names[j]][idx])
    return pd.DataFrame(M, index=names, columns=names)


def cka_boot_diff(R, a, b, c, d, n_boot=CKA_BOOT, n=CKA_BOOT_N):
    out = []
    for _ in range(n_boot):
        idx = RNG.choice(len(R["multimodal"]), n, replace=False)
        out.append(cka(R[a][idx], R[b][idx]) - cka(R[c][idx], R[d][idx]))
    out = np.asarray(out)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _topk_fast(x, k, chunk=4096):
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
            n = xt.shape[0]
            kk = min(k, n - 1)
            idx = np.empty((n, kk), dtype=np.int64)
            cos = np.empty((n, kk), dtype=np.float32)
            for st in range(0, n, chunk):
                en = min(st + chunk, n)
                sim = xt[st:en] @ xt.T
                rows = torch.arange(en - st, device=device)
                sim[rows, torch.arange(st, en, device=device)] = -2.0
                vals, ix = torch.topk(sim, kk, dim=1)
                idx[st:en] = ix.cpu().numpy()
                cos[st:en] = vals.cpu().numpy()
            del xt
            torch.cuda.empty_cache()
            return idx, cos
    except Exception as exc:
        print("[warn] GPU topk failed, using CPU:", exc, flush=True)
    return topk_neighbors(x, k)


def neighbor_stats(R, k=K, spaces=None):
    if spaces is None:
        spaces = list(R)
    sem_nb, _ = _topk_fast(R["multimodal"], k)
    col_nb, _ = _topk_fast(R["collab"], k)
    stats = {}
    for name in spaces:
        nb, _ = _topk_fast(R[name], k)
        stats[name] = {
            "sem": overlap_at_k(nb, sem_nb, k).astype(np.float32),
            "collab": overlap_at_k(nb, col_nb, k).astype(np.float32),
        }
    # random baseline
    rnd_sem, rnd_col = [], []
    for seed in range(3):
        rr = np.random.default_rng(seed)
        rnb = np.stack([
            rr.choice(np.delete(np.arange(len(R["multimodal"])), i), k, replace=False)
            for i in range(len(R["multimodal"]))
        ])
        rnd_sem.append(float(overlap_at_k(rnb, sem_nb, k).mean()))
        rnd_col.append(float(overlap_at_k(rnb, col_nb, k).mean()))
    return stats, float(np.mean(rnd_sem)), float(np.mean(rnd_col))


def boot_overlap_diff(x, y, n_boot=OV_BOOT):
    d = x.astype(np.float64) - y.astype(np.float64)
    idx = RNG.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_dataset(cfg, variants, neighbor_spaces=None):
    R = load_dataset(cfg)
    names = list(cfg["files"]) + ["collab"]
    M = cka_matrix(R, names)
    M.to_csv(os.path.join(RESULTS, f"exp_g_cka_{cfg['name']}.csv"))
    if neighbor_spaces is None:
        neighbor_spaces = list(R)
    stats, rnd_sem, rnd_col = neighbor_stats(R, K, neighbor_spaces)

    rows = []
    base_c = float(stats["multimodal"]["collab"].mean())
    base_s = float(stats["multimodal"]["sem"].mean())
    for name in variants:
        ov_s = float(stats[name]["sem"].mean())
        ov_c = float(stats[name]["collab"].mean())
        dc, dclo, dchi = boot_overlap_diff(stats[name]["collab"], stats["multimodal"]["collab"])
        ds, dslo, dshi = boot_overlap_diff(stats[name]["sem"], np.ones_like(stats[name]["sem"]))
        rows.append({
            "space": name,
            "cka_multimodal": float(M.loc[name, "multimodal"]) if name != "multimodal" else 1.0,
            "cka_collab": float(M.loc[name, "collab"]),
            "ov_sem": ov_s,
            "ov_collab": ov_c,
            "sem_retention": ov_s / base_s,
            "collab_lift": ov_c / base_c,
            "d_collab": dc, "d_collab_lo": dclo, "d_collab_hi": dchi,
            "d_sem": ds, "d_sem_lo": dslo, "d_sem_hi": dshi,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"exp_g_overlap_{cfg['name']}.csv"), index=False)

    boot = []
    for name in variants:
        if name == "multimodal":
            continue
        dm, dlo, dhi = cka_boot_diff(R, name, "collab", "multimodal", "collab")
        boot.append({"space": name, "cka_collab_diff": dm,
                     "cka_collab_diff_lo": dlo, "cka_collab_diff_hi": dhi})
    pd.DataFrame(boot).to_csv(
        os.path.join(RESULTS, f"exp_g_cka_bootstrap_{cfg['name']}.csv"), index=False)
    print(f"\n=== {cfg['name']} ===")
    print("CKA:\n", M.round(3).to_string())
    print("random overlap: sem=%.6f collab=%.6f" % (rnd_sem, rnd_col))
    print(df.round(4).to_string(index=False))
    print("CKA collab diff:\n", pd.DataFrame(boot).round(4).to_string(index=False))
    return M, df, (rnd_sem, rnd_col)


def downstream_hr():
    rows = []
    for label, run in INDSCI_HR.items():
        p = os.path.join(RESULT_BASE, run, "metrics.json")
        m = json.load(open(p))
        rows.append({"system": label, "HR@20": float(m["HR@20"]),
                     "NDCG@20": float(m["NDCG@20"]), "run": run})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_g_downstream_hr.csv"), index=False)
    print("\n=== downstream ===\n", df.round(4).to_string(index=False))
    return df


def plot(Ms, Ds, hr, rnd):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.1, 1.15]})
    # (a) CKA trade-off on IndSci
    ax = axes[0]
    M, D = Ms["Industrial_and_Scientific"], Ds["Industrial_and_Scientific"]
    pts = ["multimodal", "csga_collab", "csga_moe_collab", "fused", "moe_collab"]
    for p in pts:
        x = 1.0 if p == "multimodal" else float(M.loc[p, "multimodal"])
        y = float(M.loc[p, "collab"])
        ax.scatter(x, y, s=58, color=COLORS.get(p, "#333333"),
                   edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(p.replace("_", "\n"), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.0, color="#333333")
    ax.axhline(float(M.loc["multimodal", "collab"]), color="#999999",
               lw=0.8, ls="--", zorder=1)
    ax.axvline(1.0, color="#999999", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0.15, 1.06)
    ax.set_ylim(0.045, 0.085)
    ax.set_xlabel("CKA with direct multimodal")
    ax.set_ylabel("CKA with collaborative")
    ax.set_title("a   CKA: retention vs collab", loc="left", pad=6)

    # (b) neighbour overlap relative to multimodal
    ax = axes[1]
    groups = ["multimodal", "csga_collab", "csga_moe_collab", "fused"]
    x = np.arange(len(groups))
    sem = [float(D.loc[D.space == g, "sem_retention"].iloc[0]) for g in groups]
    col = [float(D.loc[D.space == g, "collab_lift"].iloc[0]) for g in groups]
    w = 0.36
    ax.bar(x - w / 2, sem, w, color="#4C72B0", edgecolor="white", linewidth=0.5,
           label="semantic retention", zorder=3)
    ax.bar(x + w / 2, col, w, color="#55A868", edgecolor="white", linewidth=0.5,
           label="collab lift", zorder=3)
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.axhline(rnd["Industrial_and_Scientific"][1] /
               float(D.loc[D.space == "multimodal", "ov_collab"].iloc[0]),
               color="#55A868", lw=0.8, ls=":", zorder=1)
    for xi, v in zip(x - w / 2, sem):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    for xi, v in zip(x + w / 2, col):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["multi\nmodal", "CSGA", "CSGA+\nOMoE", "CSGA+\nOMoE v2"], fontsize=7.4)
    ax.set_ylim(0, 1.95)
    ax.set_ylabel("ratio to direct multimodal")
    ax.set_title("b   Neighbour overlap (k=10), relative", loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")

    # (c) downstream HR@20
    ax = axes[2]
    order = ["text-only", "image-only", "multimodal", "csga_collab",
             "csga_moe_collab", "fused"]
    vals = [float(hr.loc[hr.system == s, "HR@20"].iloc[0]) * 100 for s in order]
    colors = ["#4C72B0", "#C44E52", "#8172B2", "#55A868", "#999999", "#B8860B"]
    bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    bars[-1].set_edgecolor("#B8860B")
    bars[-1].set_linewidth(1.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["text", "image", "multi\nmodal", "CSGA", "CSGA+\nOMoE", "CSGA+\nOMoE v2"],
                       fontsize=7.2)
    ax.set_ylabel("HR@20 (%)")
    ax.set_ylim(0, 6.0)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("c   Downstream HR@20 (IndSci)", loc="left", pad=6)

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"fig_g_fusion_evidence.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.join(FIGURES, "fig_g_fusion_evidence.png"))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    Ms, Ds, rnds = {}, {}, {}
    if only != "indsci":
        Ms["Sports_and_Outdoors"], Ds["Sports_and_Outdoors"], rnds["Sports_and_Outdoors"] = run_dataset(
            SPORTS, ["multimodal", "fused", "text", "image"],
            ["multimodal", "fused", "text", "image", "collab"])
    Ms["Industrial_and_Scientific"], Ds["Industrial_and_Scientific"], rnds["Industrial_and_Scientific"] = run_dataset(
        INDSCI, ["multimodal", "csga_collab", "csga_moe_collab", "fused"],
        ["multimodal", "csga_collab", "csga_moe_collab", "fused", "collab"])
    hr = downstream_hr()
    plot(Ms, Ds, hr, rnds)


if __name__ == "__main__":
    main()
