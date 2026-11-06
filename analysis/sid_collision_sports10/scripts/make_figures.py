import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/data02/sht/SAM-GR"
OUT = os.path.join(BASE, "analysis", "sid_collision_sports10")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

STATS = json.load(open(os.path.join(RES, "model_stats.json")))
BEAMS = json.load(open(os.path.join(RES, "beam_stats.json")))
COLL = json.load(open(os.path.join(RES, "collided_analysis.json")))
PREFIX = pd.read_csv(os.path.join(RES, "prefix_stats.csv"))

PROD = ["Full SAM-GR", "TIGER", "MMGRec (shared benchmark SID)", "CARD"]
SHORT = ["Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
COLORS = {
    "Full SAM-GR": "#4C72B0",
    "TIGER": "#DD8452",
    "MMGRec (shared benchmark SID)": "#55A868",
    "CARD": "#C44E52",
    "Base joint-VLM (300ep)": "#8172B3",
    "CSGA-only (300ep)": "#937860",
    "OMoE-only (300ep)": "#DA8BC3",
    "Full SAM-GR (300ep)": "#8C8C8C",
}


def get(model, stage="merged", key=None):
    s = STATS[f"{model}|{stage}"]
    return s[key] if key else s


def fig_overview():
    models = ["Full SAM-GR", "TIGER", "CARD", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    labels = ["Full\n(5k ep)", "TIGER\n(5k ep)", "CARD", "Base\n(300 ep)", "CSGA\n(300 ep)", "OMoE\n(300 ep)", "Full\n(300 ep)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    x = np.arange(len(models))
    ax = axes[0, 0]
    vals = [get(m, key="unique_full_sids") for m in models]
    ax.bar(x, vals, color=[COLORS[m] for m in models])
    ax.set_ylabel("unique full SIDs")
    ax.set_title("(a) Unique full SIDs (higher is better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 80, str(v), ha="center", fontsize=7)

    ax = axes[0, 1]
    vals = [100 * get(m, key="excess_collision_rate") for m in models]
    ax.bar(x, vals, color=[COLORS[m] for m in models])
    ax.set_ylabel("(N-U)/N (%)")
    ax.set_title("(b) Excess collision rate (lower is better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)

    ax = axes[1, 0]
    vals = [100 * get(m, key="collided_item_ratio") for m in models]
    ax.bar(x, vals, color=[COLORS[m] for m in models])
    ax.set_ylabel("items in buckets >1 (%)")
    ax.set_title("(c) Collided-item ratio (lower is better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)

    ax = axes[1, 1]
    vals = [get(m, key="max_bucket") for m in models]
    ax.bar(x, vals, color=[COLORS[m] for m in models])
    ax.set_yscale("log")
    ax.set_ylabel("max bucket size (log)")
    ax.set_title("(d) Maximum collision bucket")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, str(v), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_collision_overview.png"), dpi=220)
    plt.close(fig)


def hist_bins(hist):
    edges = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
    labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", ">50"]
    out = []
    for lo, hi in edges:
        out.append(sum(v for k, v in hist.items() if lo <= int(k) <= hi))
    return out, labels


def fig_hist():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    prod = ["Full SAM-GR", "TIGER", "CARD"]
    short = ["Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    for ax, group, title in zip(axes, [prod, short], ["(a) Recorded / production SIDs", "(b) 300-epoch component ablation"]):
        xs = np.arange(9)
        width = 0.8 / len(group)
        for gi, m in enumerate(group):
            vals, labels = hist_bins(get(m, key="bucket_histogram"))
            ax.bar(xs + (gi - (len(group) - 1) / 2) * width, vals, width, label=m, color=COLORS[m])
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("bucket size")
        ax.set_ylabel("number of buckets (log)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_bucket_histogram.png"), dpi=220)
    plt.close(fig)


def fig_prefix():
    models = ["Full SAM-GR", "TIGER", "CARD", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in models:
        d = PREFIX[(PREFIX.model == m) & (PREFIX.stage == "merged")]
        axes[0].plot(d.depth, d.unique_prefix_ratio, marker="o", label=m, color=COLORS[m])
        axes[1].plot(d.depth, d.collision_rate, marker="o", label=m, color=COLORS[m])
    axes[0].set_xticks([1, 2, 3, 4]); axes[0].set_xlabel("prefix depth")
    axes[0].set_ylabel("unique-prefix ratio")
    axes[0].set_title("(a) Unique-prefix ratio by depth")
    axes[0].grid(alpha=0.25)
    axes[1].set_xticks([1, 2, 3, 4]); axes[1].set_xlabel("prefix depth")
    axes[1].set_ylabel("prefix collision rate")
    axes[1].set_title("(b) Prefix collision rate by depth")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_prefix_collision.png"), dpi=220)
    plt.close(fig)


def fig_sinkhorn():
    models = ["Full SAM-GR", "TIGER", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    labels = ["Full\n(5k)", "TIGER\n(5k)", "Base\n(300)", "CSGA\n(300)", "OMoE\n(300)", "Full\n(300)"]
    stages = ["raw", "sinkhorn", "merged"]
    names = ["raw (argmin)", "after Sinkhorn", "after 15% merge"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(models)); w = 0.26
    for si, (stage, name) in enumerate(zip(stages, names)):
        vals = [100 * get(m, stage=stage, key="excess_collision_rate") for m in models]
        axes[0].bar(x + (si - 1) * w, vals, w, label=name)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("(N-U)/N (%)")
    axes[0].set_title("(a) Excess collision before/after Sinkhorn and merge")
    axes[0].legend(frameon=False, fontsize=7)
    for si, (stage, name) in enumerate(zip(stages, names)):
        vals = [get(m, stage=stage, key="max_bucket") for m in models]
        axes[1].bar(x + (si - 1) * w, vals, w, label=name)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("max bucket (log)")
    axes[1].set_title("(b) Maximum bucket size by stage")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sinkhorn_stages.png"), dpi=220)
    plt.close(fig)


def fig_collided():
    models = ["Full SAM-GR", "TIGER", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    labels = ["Full\n(5k)", "TIGER\n(5k)", "Base\n(300)", "CSGA\n(300)", "OMoE\n(300)", "Full\n(300)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    x = np.arange(len(models)); w = 0.2
    for key, name, ax, title in [
        ("text_cos_mean", "text cosine", axes[0], "(a) Text cosine of collided pairs"),
        ("image_cos_mean", "image cosine", axes[1], "(b) Image cosine of collided pairs"),
        ("cooccur_rate", "co-interaction rate", axes[2], "(c) Same-user co-interaction"),
    ]:
        col = [COLL[m]["all_pairs"].get(key, np.nan) for m in models]
        rnd = [COLL[m]["random_baseline"].get(key, np.nan) for m in models]
        dec = [COLL[m]["same_popularity_decile_baseline"].get(key, np.nan) for m in models]
        ax.bar(x - w, col, w, label="collided pairs", color="#4C72B0")
        ax.bar(x, rnd, w, label="random pairs", color="#C44E52")
        ax.bar(x + w, dec, w, label="same-popularity decile", color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        if key != "cooccur_rate":
            ax.set_ylim(0.65, 0.95)
        axes[0].set_ylabel("cosine")
        axes[2].set_ylabel("rate")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_collided_item_analysis.png"), dpi=220)
    plt.close(fig)


def fig_item_level():
    models = ["Full SAM-GR", "TIGER", "MMGRec", "CARD"]
    labels = ["Full SAM-GR", "TIGER", "MMGRec", "CARD"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(models)); w = 0.26
    sid = [100 * BEAMS[m]["sid_level_hr"]["HR@50"] for m in models]
    exp = [100 * BEAMS[m]["expected_item_level_hr_uniform"]["HR@50"] for m in models]
    single = [100 * BEAMS[m]["singleton_only_item_level_hr"]["HR@50"] for m in models]
    ax.bar(x - w, sid, w, label="SID-level (current eval)", color="#4C72B0")
    ax.bar(x, exp, w, label="expected item-level (uniform resolver)", color="#55A868")
    ax.bar(x + w, single, w, label="singleton-only item-level (lower bound)", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("HR@50 (%)")
    ax.set_title("SID-level vs item-level HR@50")
    for i, (a, b, c) in enumerate(zip(sid, exp, single)):
        ax.text(i - w, a + 0.1, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i, b + 0.1, f"{b:.2f}", ha="center", fontsize=7)
        ax.text(i + w, c + 0.1, f"{c:.2f}", ha="center", fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_item_level_gap.png"), dpi=220)
    plt.close(fig)


def fig_collision_source():
    prod = json.load(open(os.path.join(RES, "collided_prod_stage_analysis.json")))
    short = json.load(open(os.path.join(RES, "collided_stage_analysis.json")))
    groups = [
        ("Full raw", prod["Full SAM-GR|raw"]),
        ("Full merged", prod["Full SAM-GR|merged"]),
        ("TIGER raw", prod["TIGER|raw"]),
        ("TIGER merged", prod["TIGER|merged"]),
        ("CSGA pre-merge", short["CSGA-only (300ep)|sinkhorn"]),
        ("CSGA merged", short["CSGA-only (300ep)|merged"]),
        ("Full300 pre-merge", short["Full SAM-GR (300ep)|sinkhorn"]),
        ("Full300 merged", short["Full SAM-GR (300ep)|merged"]),
    ]
    labels = [g[0] for g in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title, ylab in [
        (axes[0], "text_cos_mean", "(a) Text cosine", "cosine"),
        (axes[1], "image_cos_mean", "(b) Image cosine", "cosine"),
        (axes[2], "cooccur_rate", "(c) Same-user co-interaction", "rate"),
    ]:
        vals = [g[1][key] for g in groups]
        colors = ["#4C72B0" if i % 2 == 0 else "#C44E52" for i in range(len(groups))]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.005 if key != "cooccur_rate" else 0.001), f"{v:.3f}", ha="center", fontsize=6)
    axes[0].set_ylim(0.85, 1.0)
    axes[1].set_ylim(0.72, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_collision_source.png"), dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    fig_overview()
    fig_hist()
    fig_prefix()
    fig_sinkhorn()
    fig_collided()
    fig_item_level()
    fig_collision_source()
    print("FIGURES_DONE")
