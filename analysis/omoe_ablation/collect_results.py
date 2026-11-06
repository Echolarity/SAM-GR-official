"""Aggregate the OMoE ablation into CSV / Markdown / figures."""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REP = os.path.join(RESULTS, "representation")
WORK = os.path.join(HERE, "workspaces")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load_all():
    rows = {}
    for p in sorted(glob.glob(os.path.join(REP, "*.json"))):
        d = json.load(open(p))
        name = d.get("variant")
        rows.setdefault(name, {}).update({
            "variant": name,
            "group": d.get("group", ""),
            "notes": d.get("notes", ""),
            "params_M": (d.get("params") or 0) / 1e6 if d.get("params") else None,
            "omoe_train_s": d.get("train_time_sec"),
            "omoe_rec_loss": (d.get("last_losses") or {}).get("rec"),
            "omoe_total_loss": (d.get("last_losses") or {}).get("total"),
            "fused_std": d.get("fused_std"),
            "fused_mean_cos": d.get("fused_mean_cos"),
            "fused_nn_dup": d.get("fused_nn_dup"),
            "expert_mean_abs_cos": d.get("mean_abs_expert_cos"),
            "expert_max_abs_cos": d.get("max_abs_expert_cos"),
            "gate_cv": d.get("gate_load_cv"),
            "batch_load_cv": d.get("batch_load_cv"),
            "gate_entropy": d.get("gate_entropy"),
            "item_entropy": d.get("mean_item_entropy"),
            "item_entropy_std": d.get("std_item_entropy"),
            "routing_item_var": d.get("routing_item_var"),
            "routing_total_var": d.get("routing_total_var"),
            "direct_share": d.get("direct_share"),
            "moe_share": d.get("moe_share"),
            "cos_direct_moe": d.get("cos_direct_moe"),
            "rel_diff_no_moe": d.get("rel_diff_no_moe"),
            "rel_diff_no_direct": d.get("rel_diff_no_direct"),
            "aligned_norm_share": d.get("aligned_norm_share"),
            "branch_norm_share": d.get("branch_norm_share"),
            "cos_aligned_branch": d.get("cos_aligned_branch"),
        })
    for p in glob.glob(os.path.join(WORK, "*", "rq_metrics.json")):
        v = os.path.basename(os.path.dirname(p))
        d = json.load(open(p))
        rows.setdefault(v, {"variant": v}).update({
            "recon_mse": d.get("recon_mse"),
            "recon_rmse": d.get("recon_rmse"),
            "codebook_ppl_mean": d.get("mean_codebook_perplexity"),
            "codebook_ppl_min": d.get("min_codebook_perplexity"),
            "codebook_ppl_max": d.get("max_codebook_perplexity"),
            "raw_collision": d.get("raw_collision_rate"),
            "final_collision": d.get("final_collision_rate"),
            "final_unique_sids": d.get("final_unique_sids"),
            "rq_params_M": (d.get("rq_params") or 0) / 1e6 if d.get("rq_params") else None,
            "rq_epochs": d.get("rq_epochs"),
            **{f"ppl_layer{i}": (d.get("per_layer") or [{}]*4)[i].get("perplexity") for i in range(4)},
            **{f"used_codes_layer{i}": (d.get("per_layer") or [{}]*4)[i].get("used_codes") for i in range(4)},
        })
    for p in glob.glob(os.path.join(RESULTS, "downstream_*.json")):
        d = json.load(open(p))
        v = d.get("variant")
        rows.setdefault(v, {"variant": v}).update(d)
    return rows


def md_table(df: pd.DataFrame, cols: list[tuple[str, str]], path: str):
    lines = ["| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for _, key in cols:
            val = r.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("—")
            elif isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def barplot(df, col, path, title, ylabel, lower_better=False, highlight=("01_full_omoe",)):
    if col not in df.columns:
        return
    d = df[df[col].notna()].sort_values(col, ascending=lower_better)
    if d.empty:
        return
    colors = ["#C44E52" if v in highlight else "#4C72B0" for v in d["variant"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(d)), d[col], color=colors)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["variant"], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_all()
    df = pd.DataFrame(list(rows.values())).sort_values("variant")
    df.to_csv(os.path.join(RESULTS, "ablation_master.csv"), index=False)

    paper_cols = [
        ("Variant", "variant"),
        ("HR@5", "HR@5"), ("HR@10", "HR@10"), ("HR@20", "HR@20"),
        ("NDCG@5", "NDCG@5"), ("NDCG@10", "NDCG@10"), ("NDCG@20", "NDCG@20"),
        ("Recon MSE", "recon_mse"),
        ("Codebook PPL", "codebook_ppl_mean"),
        ("Collision", "final_collision"),
        ("Expert abs cos", "expert_mean_abs_cos"),
        ("Load CV", "gate_cv"),
        ("Item H", "item_entropy"),
        ("Routing var", "routing_item_var"),
        ("Params (M)", "params_M"),
        ("OMoE time (s)", "omoe_train_s"),
    ]
    md_table(df, paper_cols, os.path.join(RESULTS, "ablation_table.md"))

    # representation-only table with all variants
    rep_cols = [
        ("Variant", "variant"),
        ("Fused cos", "fused_mean_cos"),
        ("Fused dup", "fused_nn_dup"),
        ("Expert abs cos", "expert_mean_abs_cos"),
        ("Expert max", "expert_max_abs_cos"),
        ("Load CV", "gate_cv"),
        ("Batch load CV", "batch_load_cv"),
        ("Gate H", "gate_entropy"),
        ("Item H", "item_entropy"),
        ("Routing var", "routing_item_var"),
        ("Direct share", "direct_share"),
        ("MoE share", "moe_share"),
        ("Aligned share", "aligned_norm_share"),
        ("Branch share", "branch_norm_share"),
        ("Params M", "params_M"),
        ("OMoE s", "omoe_train_s"),
    ]
    md_table(df, rep_cols, os.path.join(RESULTS, "representation_table.md"))

    barplot(df, "expert_mean_abs_cos", os.path.join(FIG, "fig_expert_cos.png"),
            "Expert-output mean |cosine| (lower = more specialised)", "mean |cos|", lower_better=True)
    barplot(df, "gate_cv", os.path.join(FIG, "fig_load_cv.png"),
            "Expert-load coefficient of variation (lower = more balanced)", "CV of mean gate", lower_better=True)
    barplot(df, "fused_mean_cos", os.path.join(FIG, "fig_fused_cos.png"),
            "Fused-feature mean inter-item cosine (lower = less collapse)", "mean cosine", lower_better=True)
    barplot(df, "item_entropy", os.path.join(FIG, "fig_item_entropy.png"),
            "Per-item routing entropy", "entropy (nats)")
    barplot(df, "recon_mse", os.path.join(FIG, "fig_recon_mse.png"),
            "RQ-VAE reconstruction MSE (1000 epochs, same budget)", "MSE", lower_better=True)
    barplot(df, "final_collision", os.path.join(FIG, "fig_collision.png"),
            "Full-SID collision rate (merge 15%)", "collision", lower_better=True)
    barplot(df, "HR@20", os.path.join(FIG, "fig_hr20.png"),
            "Downstream HR@20 (short-budget retraining)", "HR@20")

    show = ["variant", "recon_mse", "final_collision", "expert_mean_abs_cos",
            "gate_cv", "item_entropy", "direct_share", "moe_share"]
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))
    print("\nWrote results/ablation_master.csv, results/ablation_table.md, results/representation_table.md, figures/*.png")


if __name__ == "__main__":
    main()
