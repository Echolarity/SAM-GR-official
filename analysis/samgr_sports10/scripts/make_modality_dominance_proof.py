"""Modality-dominance proof from the Industrial & Scientific modality ablation.

Controlled comparison: all runs use the same task1-only 2-epoch SFT recipe and
differ only in the item embedding -> RQ-VAE -> SID pipeline:

  text-only          Industrial_and_Scientific-sft-dsz2-t1
  multimodal         Industrial_and_Scientific-sft-dsz2_mm
  multimodal+collab  Industrial_and_Scientific-sft-dsz2_mmc
  CSGA+OMoE v2       Industrial_and_Scientific-sft-dsz2_csga_omoe_v2

The text-only run is run_t1.sh, evaluated on the text-only SID data left by the
text-only pipeline (log IndSci_text_only_dsz2_task1_ep2.log). Its test inputs
are token-identical to the text-only pipeline's test set, and its history
lengths match the multimodal run sample by sample.

Outputs
  figures/fig_modality_dominance_proof.png / .pdf
  results/modality_dominance_indSci.csv
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.dirname(HERE)
FIGURES = os.path.join(OUT_ROOT, "figures")
RESULTS = os.path.join(OUT_ROOT, "results")
BASE = "/data02/sht/SAM-GR/results/qwen2.5-0.5b/full"
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

RUNS = {
    "text-only": "Industrial_and_Scientific-sft-dsz2-t1",
    "image-only": "Industrial_and_Scientific-sft-dsz2_image_only",
    "multimodal": "Industrial_and_Scientific-sft-dsz2_mm",
    "multimodal\n+collab": "Industrial_and_Scientific-sft-dsz2_mmc",
    "CSGA+OMoE\nv2": "Industrial_and_Scientific-sft-dsz2_csga_omoe_v2",
}
COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#B8860B"]
KS = [1, 5, 10, 20, 50]
N_BOOT = 3000
RNG = np.random.default_rng(0)

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


def load_metrics(run_dir: str):
    data = json.load(open(os.path.join(BASE, run_dir, "final_result.json")))
    n = len(data)
    hr = {k: np.zeros(n, dtype=np.int16) for k in KS}
    ndcg = {k: np.zeros(n, dtype=np.float64) for k in KS}
    norm = math.log(2)
    for i, sample in enumerate(data):
        preds = [str(p).strip('"\n').strip() for p in sample["predict"]]
        target = str(sample["output"]).strip(' \n"')
        rank = len(preds)
        for j, pred in enumerate(preds):
            if pred == target:
                rank = j
                break
        for k in KS:
            if rank < k:
                hr[k][i] = 1
                ndcg[k][i] = (1.0 / math.log(rank + 2)) * norm
    return hr, ndcg


def ci_mean(x, n_boot=1500):
    n = len(x)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(x, y, n_boot=N_BOOT):
    d = x.astype(np.int16) - y.astype(np.int16)
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_p(x, y):
    a = int(((y == 1) & (x == 0)).sum())
    b = int(((y == 0) & (x == 1)).sum())
    if a + b == 0:
        return 1.0, a, b
    z = (abs(a - b) - 1.0) / math.sqrt(a + b)
    return math.erfc(z / math.sqrt(2)), a, b


def main():
    H, N = {}, {}
    for name, run in RUNS.items():
        print("loading", run, flush=True)
        H[name], N[name] = load_metrics(run)

    rows = []
    for name in RUNS:
        hr_lo, hr_hi = ci_mean(H[name][20])
        nd_lo, nd_hi = ci_mean(N[name][20])
        rows.append({
            "system": name.replace("\n", " "),
            "HR@20": float(H[name][20].mean()),
            "HR@20_lo": hr_lo,
            "HR@20_hi": hr_hi,
            "NDCG@20": float(N[name][20].mean()),
            "NDCG@20_lo": nd_lo,
            "NDCG@20_hi": nd_hi,
        })

    ref = "text-only"
    deltas = []
    for name in ["image-only", "multimodal", "multimodal\n+collab", "CSGA+OMoE\nv2"]:
        diff, lo, hi = paired_delta(H[name][20], H[ref][20])
        p, a, b = mcnemar_p(H[name][20], H[ref][20])
        ndiff, nlo, nhi = paired_delta(N[name][20], N[ref][20])
        deltas.append({
            "system": name.replace("\n", " "),
            "dHR20": diff, "dHR20_lo": lo, "dHR20_hi": hi,
            "p": p, "win": a, "loss": b,
            "dNDCG20": ndiff, "dNDCG20_lo": nlo, "dNDCG20_hi": nhi,
        })
        print(f"{name!r}: dHR@20={diff:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3g} "
              f"discordant {a}/{b}", flush=True)

    out_csv = os.path.join(RESULTS, "modality_dominance_indSci.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "HR@20", "HR@20_lo", "HR@20_hi",
                    "NDCG@20", "NDCG@20_lo", "NDCG@20_hi"])
        for r in rows:
            w.writerow([r["system"], f"{r['HR@20']:.6f}", f"{r['HR@20_lo']:.6f}",
                        f"{r['HR@20_hi']:.6f}", f"{r['NDCG@20']:.6f}",
                        f"{r['NDCG@20_lo']:.6f}", f"{r['NDCG@20_hi']:.6f}"])
        w.writerow([])
        w.writerow(["comparison", "dHR@20", "dHR@20_lo", "dHR@20_hi", "p",
                    "text_wins", "system_wins", "dNDCG@20", "dNDCG@20_lo", "dNDCG@20_hi"])
        for d in deltas:
            w.writerow([d["system"], f"{d['dHR20']:.6f}", f"{d['dHR20_lo']:.6f}",
                        f"{d['dHR20_hi']:.6f}", f"{d['p']:.6g}", d["win"], d["loss"],
                        f"{d['dNDCG20']:.6f}", f"{d['dNDCG20_lo']:.6f}",
                        f"{d['dNDCG20_hi']:.6f}"])
    print("saved", out_csv, flush=True)

    names = list(RUNS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    hr = np.array([r["HR@20"] for r in rows]) * 100
    nd = np.array([r["NDCG@20"] for r in rows]) * 100
    hr_err = np.array([[r["HR@20"] - r["HR@20_lo"], r["HR@20_hi"] - r["HR@20"]] for r in rows]).T * 100
    nd_err = np.array([[r["NDCG@20"] - r["NDCG@20_lo"], r["NDCG@20_hi"] - r["NDCG@20"]] for r in rows]).T * 100
    w = 0.36
    ax.bar(x - w / 2, hr, w, color=[c for c in COLORS], edgecolor="white",
           linewidth=0.5, zorder=3, label="HR@20")
    ax.bar(x + w / 2, nd, w, color=[c for c in COLORS], edgecolor="white",
           linewidth=0.5, alpha=0.45, hatch="//", zorder=3, label="NDCG@20")
    ax.errorbar(x - w / 2, hr, yerr=hr_err, fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2, zorder=4)
    ax.errorbar(x + w / 2, nd, yerr=nd_err, fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2, zorder=4)
    for xi, v in zip(x - w / 2, hr):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    for xi, v in zip(x + w / 2, nd):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.0)
    ax.set_ylabel("metric (%)")
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("a   Single modalities are comparable; CSGA+OMoE adds the gain", loc="left", pad=7)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2)

    ax = axes[1]
    y = np.arange(len(deltas))[::-1]
    for yi, d, c in zip(y, deltas, ["#C44E52", "#55A868", "#8172B2", "#B8860B"]):
        lo, hi = d["dHR20_lo"] * 100, d["dHR20_hi"] * 100
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter([d["dHR20"] * 100], [yi], s=52, color=c, edgecolor="white",
                   linewidth=0.6, zorder=4)
        stars = "***" if d["p"] < 0.001 else "**" if d["p"] < 0.01 else "*" if d["p"] < 0.05 else "n.s."
        ax.text(hi + 0.08, yi, f"{d['dHR20']*100:+.2f}pp  {stars}", va="center",
                fontsize=7.6, color="#222222")
    ax.axvline(0, color="#555555", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([d["system"] for d in deltas], fontsize=8.2)
    ax.set_xlabel("paired ΔHR@20 vs text-only (pp)")
    d_lo = min(d["dHR20_lo"] for d in deltas) * 100
    d_hi = max(d["dHR20_hi"] for d in deltas) * 100
    ax.set_xlim(d_lo - 0.3, d_hi + 0.7)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("b   Paired user-level comparison", loc="left", pad=7)

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"fig_modality_dominance_proof.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.join(FIGURES, "fig_modality_dominance_proof.png"), flush=True)


if __name__ == "__main__":
    main()
