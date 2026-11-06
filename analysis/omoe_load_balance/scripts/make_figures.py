"""Figures 1-6 for the OMoE load-balancing study (PDF + 300 dpi PNG)."""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(RES, "tables")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)
from lb_core import EXPERT_NAMES, EXPERT_TYPES, FULL_LAMBDA, LAMBDAS  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
GROUP_COLORS = {"text": "#4C72B0", "vision": "#DD8452", "aligned": "#55A868", "shared": "#C44E52"}
MODEL_COLORS = {"NoLoad": "#7F7F7F", "FullLoad": "#B2182B"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_master():
    return pd.read_csv(os.path.join(TAB, "master.csv"))


def agg(model_df, col):
    g = model_df.groupby("lambda_load")[col]
    return g.mean(), g.std()


def fig1(df):
    sub = df[df.lambda_load.isin([0.0, FULL_LAMBDA])]
    # mean_pi is a list in JSON; re-read the per-model analysis JSON.
    means = {}
    for lam, label in [(0.0, "NoLoad"), (FULL_LAMBDA, "FullLoad")]:
        vals = []
        for f in glob.glob(os.path.join(RES, "analysis", f"lam*_seed*.json")):
            d = json.load(open(f))
            if abs(d["lambda_load"] - lam) < 1e-15:
                vals.append(d["routing"]["mean_pi"])
        means[label] = np.array(vals)
    x = np.arange(len(EXPERT_NAMES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, (label, arr) in enumerate(means.items()):
        if len(arr) == 0:
            continue
        mu, sd = arr.mean(0), arr.std(0)
        colors = [GROUP_COLORS[t.replace("_specific", "") if t != "shared" else "shared"]
                  for t in EXPERT_TYPES]
        bars = ax.bar(x + (k - 0.5) * w, mu, w, yerr=sd, capsize=3,
                      color=colors, alpha=0.45 if label == "NoLoad" else 0.95,
                      edgecolor="black", linewidth=0.5,
                      hatch="///" if label == "NoLoad" else None)
        for xi, v in zip(x, mu):
            ax.text(xi + (k - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        # individual seeds, so the seed-dependent collapse direction is visible
        rng = np.random.default_rng(0)
        for j in range(len(EXPERT_NAMES)):
            jitter = rng.uniform(-0.09, 0.09, size=len(arr))
            ax.scatter(x[j] + (k - 0.5) * w + jitter, arr[:, j], s=9, color="black",
                       alpha=0.55, zorder=4, linewidths=0)
    ax.axhline(1 / len(EXPERT_NAMES), color="gray", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({t.replace('_specific','')})" for n, t in zip(EXPERT_NAMES, EXPERT_TYPES)],
                       fontsize=7)
    ax.set_ylabel("mean gate probability")
    ax.set_title("Expert mean gate weight: NoLoad vs FullLoad")
    handles = [Patch(facecolor=GROUP_COLORS[g], label=f"{g}-specific") for g in ("text", "vision", "aligned")]
    handles.append(Patch(facecolor=GROUP_COLORS["shared"], label="shared"))
    handles += [Patch(facecolor="white", edgecolor="black", hatch="///", label="NoLoad"),
                Patch(facecolor="white", edgecolor="black", label="FullLoad")]
    ax.legend(handles=handles, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_ylim(0, max(0.28, ax.get_ylim()[1]))
    save(fig, "fig1_expert_mean_gate")


def fig2(df):
    metrics = [
        ("load_cv", "Load CV", False),
        ("effective_experts", "Effective experts", False),
        ("dead_rate_0.01M", "Dead expert rate (0.01/M)", False),
        ("routing_js_mean", "Routing JS divergence", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for ax, (col, title, _) in zip(axes, metrics):
        sub = df[df.lambda_load.isin([0.0, FULL_LAMBDA])]
        data = [sub[sub.lambda_load == lam][col].dropna().values for lam in (0.0, FULL_LAMBDA)]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"))
        for patch, lam in zip(bp["boxes"], (0.0, FULL_LAMBDA)):
            patch.set_facecolor(MODEL_COLORS["NoLoad" if lam == 0 else "FullLoad"])
            patch.set_alpha(0.7)
        for k, (vals, lam) in enumerate(zip(data, (0.0, FULL_LAMBDA))):
            ax.scatter(np.full(len(vals), k + 1), vals, color="black", s=12, zorder=3)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["NoLoad", "FullLoad"])
        ax.set_title(title)
    fig.suptitle("Expert balance statistics (3 seeds; points = individual seeds)", y=1.03)
    fig.tight_layout()
    save(fig, "fig2_balance_stats")


def fig3(df):
    def mean_std(col):
        sub = df[df.lambda_load.isin([0.0, FULL_LAMBDA])]
        return {lam: (sub[sub.lambda_load == lam][col].mean(), sub[sub.lambda_load == lam][col].std())
                for lam in (0.0, FULL_LAMBDA)}

    panels = [
        ("A. Expert-group routing mass", ["mass_text", "mass_vision", "mass_aligned", "mass_shared"],
         ["text", "vision", "aligned", "shared"]),
        ("B. Norm-aware group contribution",
         ["norm_share_text", "norm_share_vision", "norm_share_aligned", "norm_share_shared"],
         ["text", "vision", "aligned", "shared"]),
        ("C. Counterfactual input contribution", ["q_text", "q_vision", "q_aligned"],
         ["text", "vision", "aligned"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (title, cols, groups) in zip(axes, panels):
        x = np.arange(2)
        bottom = np.zeros(2)
        for col, g in zip(cols, groups):
            ms = mean_std(col)
            vals = np.array([ms[0.0][0], ms[FULL_LAMBDA][0]])
            errs = np.array([ms[0.0][1], ms[FULL_LAMBDA][1]])
            ax.bar(x, vals, 0.6, bottom=bottom, color=GROUP_COLORS[g], label=g,
                   yerr=errs, capsize=2, edgecolor="white", linewidth=0.6)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.06:
                    ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(["NoLoad", "FullLoad"])
        ax.set_ylim(0, 1.02); ax.set_ylabel("share")
        ax.set_title(title, fontsize=9)
    handles = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in ("text", "vision", "aligned", "shared")]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Modality contribution under three decompositions", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_modality_contribution")


def valid_ndcg(df):
    # Prefer LLM-generator validation results, fall back to the retrieval proxy.
    out = {}
    for lam in LAMBDAS:
        vals = []
        for f in glob.glob(os.path.join(RES, "downstream", f"val_lam*_seed*.json")):
            d = json.load(open(f))
            if abs(d.get("lambda_load", -1) - lam) < 1e-15 and d.get("NDCG@10") is not None:
                vals.append(d["NDCG@10"])
        out[lam] = (np.mean(vals), np.std(vals), "generator") if vals else (None, None, None)
    if any(v[0] is not None for v in out.values()):
        return out
    for lam in LAMBDAS:
        vals = df[df.lambda_load == lam]["proxy_valid_ndcg10"].dropna().values
        out[lam] = (float(vals.mean()), float(vals.std()), "proxy") if len(vals) else (None, None, None)
    return out


def fig4(df):
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))
    vn = valid_ndcg(df)
    xs = np.array(LAMBDAS)
    series = [
        ("Validation NDCG@10", None, vn),
        ("Load CV", "load_cv", None),
        ("Counterfactual MDI", "cf_mdi", None),
        ("Routing JS divergence", "routing_js_mean", None),
    ]
    for ax, (title, col, special) in zip(axes, series):
        if special is not None:
            y = np.array([special[l][0] if special[l][0] is not None else np.nan for l in LAMBDAS])
            e = np.array([special[l][1] if special[l][1] is not None else np.nan for l in LAMBDAS])
            src = next((special[l][2] for l in LAMBDAS if special[l][2]), "")
            ax.errorbar(xs, y, yerr=e, marker="o", capsize=3, color="#2166AC", lw=1.5)
            ax.set_title(f"{title}\n({src})", fontsize=9)
        else:
            mu, sd = agg(df, col)
            ax.errorbar(mu.index.values, mu.values, yerr=sd.values, marker="o", capsize=3,
                        color="#2166AC", lw=1.5)
            ax.set_title(title)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xticks(LAMBDAS)
        ax.set_xticklabels(["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
        ax.axvline(FULL_LAMBDA, color=MODEL_COLORS["FullLoad"], ls="--", lw=0.9, alpha=0.8)
        ax.set_xlabel("$\\lambda_{load}$")
    fig.suptitle("Balance–specialization trade-off (routing: mean ± std over 3 seeds; "
                 "validation NDCG@10: seed 42, 200-step generator)", y=1.03)
    fig.tight_layout()
    save(fig, "fig4_lambda_tradeoff")


def fig5(df):
    g = pd.read_csv(os.path.join(TAB, "groups_long.csv"))
    dims = [("popularity", ["tail", "mid", "head"], "Popularity"),
            ("consistency", ["low", "mid", "high"], "Text-image consistency")]
    mods = [("q_text", "text"), ("q_vision", "vision"), ("q_aligned", "aligned")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), sharey=True)
    for r, (dim, order, dlabel) in enumerate(dims):
        for c, (col, mname) in enumerate(mods):
            ax = axes[r, c]
            width = 0.36
            for k, lam in enumerate((0.0, FULL_LAMBDA)):
                sub = g[(g.lambda_load == lam) & (g.dimension == dim)]
                mu = [sub[sub.group == o][col].mean() for o in order]
                sd = [sub[sub.group == o][col].std() for o in order]
                ax.bar(np.arange(len(order)) + (k - 0.5) * width, mu, width,
                       yerr=sd, capsize=2,
                       color=MODEL_COLORS["NoLoad" if lam == 0 else "FullLoad"],
                       alpha=0.85, label="NoLoad" if lam == 0 else "FullLoad")
            ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order)
            ax.set_title(f"{dlabel}: q_{mname}", fontsize=9)
            if c == 0:
                ax.set_ylabel("mean share")
            if r == 0 and c == 0:
                ax.legend(frameon=False)
    fig.suptitle("Group-level counterfactual modality contribution", y=1.01)
    fig.tight_layout()
    save(fig, "fig5_group_modality")


def fig6(df):
    names = {0.0: "NoLoad", FULL_LAMBDA: "FullLoad"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(11246, size=800, replace=False))
    for ax, lam in zip(axes, (0.0, FULL_LAMBDA)):
        f = os.path.join(RES, "analysis", f"lam{('0' if lam == 0 else '1e-1')}_seed42.npz")
        if not os.path.exists(f):
            ax.axis("off")
            continue
        with np.load(f) as z:
            pi_all = z["pi"]
        idx2 = idx[idx < len(pi_all)]
        pi = pi_all[idx2]
        ent = -(pi * np.log(pi + 1e-12)).sum(1)
        dom = pi.argmax(1)
        order = np.lexsort((-ent, dom))
        im = ax.imshow(pi[order], aspect="auto", interpolation="nearest", cmap="magma",
                       vmin=0, vmax=max(0.6, float(pi.max())))
        ax.set_xticks(range(pi.shape[1]))
        ax.set_xticklabels(EXPERT_NAMES, rotation=25, ha="right", fontsize=7)
        bounds = np.cumsum(np.bincount(dom[order], minlength=pi.shape[1]))[:-1]
        for b in bounds:
            ax.axhline(b - 0.5, color="cyan", lw=0.8, alpha=0.9)
        ax.set_title(f"{names[lam]} (items sorted by Top-1 expert, then entropy)")
        ax.set_ylabel("sampled items")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="gate probability")
    save(fig, "fig6_routing_heatmap")


def main():
    df = load_master()
    fig1(df); fig2(df); fig3(df); fig4(df); fig5(df); fig6(df)
    print("FIGURES_DONE", flush=True)


if __name__ == "__main__":
    main()
