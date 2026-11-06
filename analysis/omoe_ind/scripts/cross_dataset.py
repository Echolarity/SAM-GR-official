"""Cross-dataset / cross-modality comparison and Markdown report."""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/data02/sht/SAM-GR"
OUT = os.path.join(BASE, "analysis", "omoe_ind")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

DATASETS = [
    ("Sports_and_Outdoors", "Sports strict-10core", "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict"),
    ("Industrial_and_Scientific", "IndSci", "Industrial_and_Scientific-sft-dsz2_csga_omoe_v2"),
    ("Beauty_and_Personal_Care", "Beauty strict", "Beauty_and_Personal_Care-sft-dsz2_csga_omoe_v2_strict"),
    ("Books", "Books top1500", "Books-sft-dsz2_csga_omoe_v2"),
    ("Toys_and_Games", "Toys 5core-top30k", "Toys_and_Games-sft-dsz2_csga_omoe_v2"),
]

NAMES = ["text", "image", "csga", "shared-1", "shared-2"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def df_to_md(df, floatfmt=".4f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:{floatfmt}}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def count_inter(p):
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


def metadata_row(cat, label):
    root = os.path.join(BASE, "data", "Amazon", cat)
    items = load_json(os.path.join(root, f"{cat}.item.json"))
    n = len(items)
    titles, descs, cats, has_img = [], [], [], []
    for v in items.values():
        t = str(v.get("title", "")); d = str(v.get("description", ""))
        titles.append(t); descs.append(d)
        c = str(v.get("categories", ""))
        parts = [x.strip() for x in c.split(",") if x.strip()]
        cats.append(parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown"))
        has_img.append(bool(v.get("image_paths")))
    text_len = np.array([len(t) + len(d) for t, d in zip(titles, descs)])
    pop = np.zeros(n, dtype=np.int64)
    train_p = os.path.join(root, f"{cat}.train.inter")
    if os.path.exists(train_p):
        with open(train_p) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                for j in parts[1].split():
                    pop[int(j)] += 1
                pop[int(parts[2])] += 1
    q50, q80 = np.quantile(pop, [0.50, 0.80])
    head = pop[pop >= q80].mean() if (pop >= q80).any() else 0
    mid = pop[(pop >= q50) & (pop < q80)].mean() if ((pop >= q50) & (pop < q80)).any() else 0
    tail = pop[pop < q50].mean() if (pop < q50).any() else 0
    top_cat = Counter(cats).most_common(2)
    emb = {}
    for k in ["text_only", "image_only", "multimodal", "csga_moe_collab_v2"]:
        emb[k] = os.path.exists(os.path.join(root, f"{cat}.emb-Qwen3VL-{k}.npy"))
    aligned = os.path.join(BASE, "analysis", "omoe_ind", "results", "csga_aligned.npy") \
        if cat == "Industrial_and_Scientific" else \
        os.path.join(BASE, "analysis", "samgr_sports10", "results", "csga_aligned.npy") \
        if cat == "Sports_and_Outdoors" else ""
    return {
        "Dataset": label, "Items": n,
        "Train": count_inter(train_p),
        "Test": count_inter(os.path.join(root, f"{cat}.test.inter")),
        "Missing image": int(sum(1 for x in has_img if not x)),
        "Text len mean": float(text_len.mean()),
        "Text len median": float(np.median(text_len)),
        "Top category": top_cat[0][0] if top_cat else "-",
        "Head/mid/tail pop": f"{head:.0f}/{mid:.0f}/{tail:.0f}",
        "text/image/mm emb": f"{int(emb['text_only'])}/{int(emb['image_only'])}/{int(emb['multimodal'])}",
        "CSGA aligned": bool(aligned and os.path.exists(aligned)),
        "Final fused emb": bool(emb["csga_moe_collab_v2"]),
    }


def main():
    # ---------- metadata table ----------
    meta_rows = [metadata_row(cat, label) for cat, label, _ in DATASETS]
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(os.path.join(RES, "cross_dataset_metadata.csv"), index=False)

    # ---------- downstream results ----------
    main_rows = []
    for cat, label, run in DATASETS:
        p = os.path.join(BASE, "results", "qwen2.5-0.5b", "full", run, "metrics.json")
        if os.path.exists(p):
            m = load_json(p)
            main_rows.append({"Dataset": label, "Run": run,
                              "HR@10": m.get("HR@10"), "HR@50": m.get("HR@50"),
                              "NDCG@10": m.get("NDCG@10"), "NDCG@50": m.get("NDCG@50")})
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(os.path.join(RES, "cross_dataset_main_results.csv"), index=False)

    # IndSci modality ablation (same task1/ep2 setting where possible)
    ind_runs = [
        ("Text-only (task1/ep2)", "Industrial_and_Scientific-sft-dsz2-t1"),
        ("Image-only", "Industrial_and_Scientific-sft-dsz2_image_only"),
        ("Multimodal", "Industrial_and_Scientific-sft-dsz2_mm"),
        ("Multimodal+Collab", "Industrial_and_Scientific-sft-dsz2_mmc"),
        ("CSGA alone", "Industrial_and_Scientific-sft-dsz2_csga_alone"),
        ("CSGA + OMoE v1", "Industrial_and_Scientific-sft-dsz2_csga_omoe"),
        ("CSGA + OMoE v2", "Industrial_and_Scientific-sft-dsz2_csga_omoe_v2"),
    ]
    ind_rows = []
    for label, run in ind_runs:
        p = os.path.join(BASE, "results", "qwen2.5-0.5b", "full", run, "metrics.json")
        if os.path.exists(p):
            m = load_json(p)
            ind_rows.append({"Variant": label, "HR@10": m.get("HR@10"), "HR@50": m.get("HR@50"),
                             "NDCG@10": m.get("NDCG@10"), "NDCG@50": m.get("NDCG@50")})
    ind = pd.DataFrame(ind_rows)
    ind.to_csv(os.path.join(RES, "ind_downstream_modalities.csv"), index=False)

    # ---------- gating comparison ----------
    sp1 = load_json(os.path.join(BASE, "analysis", "omoe_sports10", "results", "part1_global_stats.json"))
    sp2 = load_json(os.path.join(BASE, "analysis", "omoe_sports10", "results", "part2_instance_stats.json"))
    in1 = load_json(os.path.join(RES, "part1_global_stats.json"))
    in2 = load_json(os.path.join(RES, "part2_instance_stats.json"))
    gate_rows = []
    for label, p1, p2 in [("Sports strict", sp1, sp2), ("IndSci", in1, in2)]:
        gate_rows.append({
            "Dataset": label,
            **{f"mean_{NAMES[j]}": p1["mean_gate"][j] for j in range(5)},
            **{f"top1_{NAMES[j]}": p1["top1_freq"][j] for j in range(5)},
            **{f"contrib_{NAMES[j]}": p1["contribution_share"][j] for j in range(5)},
            "load_cv": p1["load_cv_across_experts"],
            "entropy": p1["mean_item_entropy"],
            "JS_mean": p2["js_mean"],
            "PC1_var": p2["pca_explained_variance_ratio"][0],
            "PC1_textlen_rho": p2["pc1_textlen_spearman"],
        })
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(os.path.join(RES, "cross_dataset_gating.csv"), index=False)

    # IndSci test cases used by the retrieval proxy (exclude target-in-history)
    test_p = os.path.join(BASE, "data", "Amazon", "Industrial_and_Scientific",
                          "Industrial_and_Scientific.test.inter")
    n_test = count_inter(test_p)
    excluded = 0
    with open(test_p) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            if int(parts[2]) in [int(x) for x in parts[1].split()]:
                excluded += 1
    n_cases = n_test - excluded

    # ---------- figures ----------
    # IndSci downstream modality comparison
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(ind)); w = 0.36
    ax.bar(x - w / 2, ind["HR@50"], w, color="#4C72B0", label="HR@50")
    ax.bar(x + w / 2, ind["NDCG@50"], w, color="#DD8452", label="NDCG@50")
    ax.set_xticks(x); ax.set_xticklabels(ind["Variant"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_title("IndSci: text / image / fusion downstream results (existing records)")
    ax.legend(frameon=False)
    for i, r in ind.iterrows():
        ax.text(i - w / 2, r["HR@50"] + 0.001, f"{r['HR@50']:.4f}", ha="center", fontsize=7)
        ax.text(i + w / 2, r["NDCG@50"] + 0.001, f"{r['NDCG@50']:.4f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare_ind_downstream.png"), bbox_inches="tight")
    plt.close(fig)

    # Top-1 and contribution share comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(5); w = 0.36
    for k, (label, color) in enumerate([("Sports strict", "#4C72B0"), ("IndSci", "#DD8452")]):
        top1 = [sp1["top1_freq"][j] if label == "Sports strict" else in1["top1_freq"][j] for j in range(5)]
        axes[0].bar(x + (k - 0.5) * w, top1, w, color=color, label=label)
        contrib = [sp1["contribution_share"][j] if label == "Sports strict" else in1["contribution_share"][j] for j in range(5)]
        axes[1].bar(x + (k - 0.5) * w, contrib, w, color=color, label=label)
    axes[0].set_xticks(x); axes[0].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[0].set_ylabel("top-1 frequency"); axes[0].set_title("(a) Which expert wins the route?")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(x); axes[1].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[1].set_ylabel("contribution share"); axes[1].set_title("(b) Expert-mixture contribution")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare_gating.png"), bbox_inches="tight")
    plt.close(fig)

    # IndSci missing-image group
    g = pd.read_csv(os.path.join(RES, "part3_group_stats.csv"))
    sub = g[g.dimension == "missing_image"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(5); w = 0.36
    for k, (grp, color) in enumerate([("has_image", "#4C72B0"), ("missing_image", "#C44E52")]):
        row = sub[sub.group == grp].iloc[0]
        vals = [row[f"gate_{n}"] for n in NAMES]
        ax.bar(x + (k - 0.5) * w, vals, w, color=color, label=f"{grp} (n={int(row['n'])})")
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=20, ha="right")
    ax.set_ylabel("mean gate weight")
    ax.set_title("IndSci: missing-image items route almost entirely to CSGA")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ind_missing_image.png"), bbox_inches="tight")
    plt.close(fig)

    # PC1 loadings comparison
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(5); w = 0.36
    ax.bar(x - w / 2, sp2["pca_components"][0], w, color="#4C72B0", label="Sports strict")
    ax.bar(x + w / 2, in2["pca_components"][0], w, color="#DD8452", label="IndSci")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=20, ha="right")
    ax.set_ylabel("PC1 loading"); ax.set_title("Routing PC1 axis differs by dataset")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare_pc1.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------- report ----------
    def pct(x): return f"{x*100:.2f}%"
    def g3(dim, group, expert):
        row = g[(g.dimension == dim) & (g.group == group)]
        return float(row[f"gate_{expert}"].iloc[0])

    p4 = pd.read_csv(os.path.join(RES, "part4_intervention_stats.csv"))
    p4 = p4.set_index("condition")
    base = pd.read_csv(os.path.join(RES, "part4_retrieval_proxy.csv"))
    b = base[base.condition == "baseline"].iloc[0]
    def rel(cond, metric):
        return (p4.loc[cond, metric] - b[metric]) / max(b[metric], 1e-12) * 100

    meta_md = df_to_md(meta, ".2f")
    main_md = df_to_md(main_df, ".4f")
    ind_md = df_to_md(ind, ".4f")
    gate_compact = gate_df[["Dataset", "load_cv", "entropy", "JS_mean", "PC1_var",
                            "PC1_textlen_rho", "top1_text", "top1_csga", "top1_shared-2",
                            "contrib_text", "contrib_image", "contrib_csga",
                            "contrib_shared-1", "contrib_shared-2"]].copy()
    gate_compact.columns = ["Dataset", "Load CV", "Entropy", "JS mean", "PC1 var",
                            "PC1–textlen ρ", "Top1 text", "Top1 CSGA", "Top1 shared-2",
                            "Contrib text", "Contrib image", "Contrib CSGA",
                            "Contrib shared-1", "Contrib shared-2"]
    gate_md = df_to_md(gate_compact, ".4f")

    md = f"""# 其他设定与文本/图像模态：扩展分析

本报告回答“其他地方的数据、以及文本/图像相关数据”的问题。结论先行：

- **只有 Sports 和 Industrial_and_Scientific（IndSci）保存了完整的 text_only / image_only / multimodal / CSGA / OMoE 中间特征**，因此只有这两个设定能做逐 expert 的 gating 分析。
- **IndSci 的 OMoE 路由轴与 Sports 完全不同**：Sports 是 **text ↔ shared-2**，IndSci 是 **CSGA ↔ shared-1/shared-2**。IndSci 的 text expert 在专家混合项中只占 **{in1['contribution_share'][0]*100:.1f}%**，而 image 占 **{in1['contribution_share'][1]*100:.1f}%**；Sports 恰好相反（text {sp1['contribution_share'][0]*100:.1f}%，image {sp1['contribution_share'][1]*100:.1f}%）。
- **IndSci 的缺图组（1,318 items）是最强的路由信号**：CSGA gate **{g3('missing_image','missing_image','csga'):.4f}**，image gate **{g3('missing_image','missing_image','image'):.6f}**，熵仅 **{float(g[(g.dimension=='missing_image')&(g.group=='missing_image')]['mean_entropy'].iloc[0]):.3f}**；η²(image)=0.872。Sports 只有 2 个缺图样本，无法做这个分析。
- **IndSci 的既有下游结果**显示 text-only（HR@50 {float(ind[ind.Variant=='Text-only (task1/ep2)']['HR@50'].iloc[0]):.4f}）与 image-only（{float(ind[ind.Variant=='Image-only']['HR@50'].iloc[0]):.4f}）接近、text 略高；CSGA alone 已经很强（{float(ind[ind.Variant=='CSGA alone']['HR@50'].iloc[0]):.4f}），OMoE v2 最好（{float(ind[ind.Variant=='CSGA + OMoE v2']['HR@50'].iloc[0]):.4f}），但相对 CSGA alone 的增益只有 **+{float(ind[ind.Variant=='CSGA + OMoE v2']['HR@50'].iloc[0])-float(ind[ind.Variant=='CSGA alone']['HR@50'].iloc[0]):.4f} HR@50**。主要增益来自 CSGA，而不是图像。
- **Beauty / Books / Toys 只保存了最终融合特征** `csga_moe_collab_v2`，没有 text/image/multimodal/CSGA 中间特征；因此无法从现有记录恢复逐 expert gating。下表给出可用的数据统计与主结果，并明确标注不可比之处。

> **可复现性边界（IndSci）**：Sports 的 OMoE 复现与 `exp_d` 记录逐位一致；IndSci 没有对应的 `exp_d` 记录，且 `image_only` 特征在 2026-10-24 重新生成（晚于 2026-10-19 的 `csga_omoe_v2` 运行）。因此 IndSci 的 gating 统计是对**当前特征缓存**的重建，不能声称与当时生产模型逐位一致；IndSci 的下游指标来自既有运行记录。缺图组的强信号还依赖 `image_only` 对缺图 item 的回退方式（当前缓存中缺图 item 的 `image_only` 不是正常视觉特征）；若生产运行的 `image_only` 处理不同，缺图组的 gate 数值可能变化，但“缺图会触发 CSGA 接管”的方向性结论仍值得进一步验证。

---

## 1. 数据与产物可用性

{meta_md}

**说明**
- `text/image/mm emb` 表示是否存在 `text_only` / `image_only` / `multimodal` 三个 4096 维 embedding；`CSGA aligned` 表示是否有可用的 CSGA 对齐特征；`Final fused emb` 是最终 `csga_moe_collab_v2` 特征。
- **Sports strict-10core**：11,246 items，完整产物；本系列前一份报告已做 Full OMoE 分析。
- **IndSci**：27,222 items，完整产物；本报告新做 CSGA + OMoE 复现与 Full OMoE 分析。
- **Beauty**：27,510 items，只有最终融合特征；缺图 12 个（见上表）。
- **Books top1500**：1,500 items，只有最终融合特征；item 数很少，HR@50 与其它数据集不可直接比较。
- **Toys 5core-top30k**：31,131 items，只有最终融合特征；主结果与 strict 版本差异很大，比较时需注明设定。

---

## 2. IndSci：文本 / 图像 / 融合的端到端结果（既有记录）

{ind_md}

**解读**
- **text-only 与 image-only 接近，text 略高**：HR@50 {float(ind[ind.Variant=='Text-only (task1/ep2)']['HR@50'].iloc[0]):.4f} vs {float(ind[ind.Variant=='Image-only']['HR@50'].iloc[0]):.4f}，NDCG@50 {float(ind[ind.Variant=='Text-only (task1/ep2)']['NDCG@50'].iloc[0]):.4f} vs {float(ind[ind.Variant=='Image-only']['NDCG@50'].iloc[0]):.4f}。所以不能简单说“IndSci 图像比文本强”；两者是接近的，真正拉开差距的是 CSGA。
- **多模态融合本身没有超过 text-only 的 HR@50**：Multimodal {float(ind[ind.Variant=='Multimodal']['HR@50'].iloc[0]):.4f}、Multimodal+Collab {float(ind[ind.Variant=='Multimodal+Collab']['HR@50'].iloc[0]):.4f}，都低于 text-only {float(ind[ind.Variant=='Text-only (task1/ep2)']['HR@50'].iloc[0]):.4f}；但 Multimodal+Collab 的 NDCG@50（{float(ind[ind.Variant=='Multimodal+Collab']['NDCG@50'].iloc[0]):.4f}）高于 text-only（{float(ind[ind.Variant=='Text-only (task1/ep2)']['NDCG@50'].iloc[0]):.4f}）。
- **CSGA alone 很强**：HR@50 {float(ind[ind.Variant=='CSGA alone']['HR@50'].iloc[0]):.4f}，已经超过 text/image/multimodal。说明在 IndSci 上，协同图聚合的贡献很大。
- **OMoE v2 最好但增益很小**：HR@50 {float(ind[ind.Variant=='CSGA + OMoE v2']['HR@50'].iloc[0]):.4f}，相对 CSGA alone +{float(ind[ind.Variant=='CSGA + OMoE v2']['HR@50'].iloc[0])-float(ind[ind.Variant=='CSGA alone']['HR@50'].iloc[0]):.4f}；NDCG@50 {float(ind[ind.Variant=='CSGA + OMoE v2']['NDCG@50'].iloc[0]):.4f} vs {float(ind[ind.Variant=='CSGA alone']['NDCG@50'].iloc[0]):.4f}。这是一个**诚实的天花板提示**：在 IndSci 上，融合的边际收益不大，主要收益来自 CSGA 本身。

![IndSci downstream](figures/fig_compare_ind_downstream.png)

---

## 3. IndSci：Full OMoE gating 与 expert utilization

**全局统计（27,222 items）**
- Mean gate：text {in1['mean_gate'][0]:.4f}、image {in1['mean_gate'][1]:.4f}、CSGA {in1['mean_gate'][2]:.4f}、shared-1 {in1['mean_gate'][3]:.4f}、shared-2 {in1['mean_gate'][4]:.4f}；load CV = **{in1['load_cv_across_experts']:.4f}**。
- 平均 per-item entropy = **{in1['mean_item_entropy']:.4f}**；effective experts = **{in1['effective_experts_per_item']:.3f}**（低于 Sports 的 4.122）。
- Top-1：CSGA **{in1['top1_freq'][2]*100:.1f}%**、shared-2 **{in1['top1_freq'][4]*100:.1f}%**、shared-1 **{in1['top1_freq'][3]*100:.1f}%**、text {in1['top1_freq'][0]*100:.2f}%、image {in1['top1_freq'][1]*100:.2f}%。
- 专家混合项贡献份额：text **{in1['contribution_share'][0]*100:.1f}%**、image **{in1['contribution_share'][1]*100:.1f}%**、CSGA {in1['contribution_share'][2]*100:.1f}%、shared-1 {in1['contribution_share'][3]*100:.1f}%、shared-2 {in1['contribution_share'][4]*100:.1f}%。
- 专家输出平均 |cos| = {in1['mean_abs_offdiag_cos']:.4f}。

**路由轴**
- PC1 解释 **{in2['pca_explained_variance_ratio'][0]*100:.1f}%** 方差；载荷：text {in2['pca_components'][0][0]:+.3f}、image {in2['pca_components'][0][1]:+.3f}、CSGA **{in2['pca_components'][0][2]:+.3f}**、shared-1 **{in2['pca_components'][0][3]:+.3f}**、shared-2 **{in2['pca_components'][0][4]:+.3f}**。
- PC1 与文本长度 Spearman ρ = **{in2['pc1_textlen_spearman']:+.3f}**；与 text–image consistency ρ = **{in2['pc1_consistency_spearman']:+.3f}**；与 popularity ρ = **{in2['pc1_popularity_spearman']:+.3f}**。
- **解释**：IndSci 的 PC1 是 **CSGA vs shared-1/shared-2**，而不是 Sports 的 text vs shared-2。长文本/高一致性 → CSGA；短文本/低一致性 → shared-1/shared-2。text expert 的载荷接近 0，说明它几乎不参与主路由轴。

![IndSci routing heatmap](figures/fig2_routing_heatmap.png)

![IndSci routing PCA](figures/fig2_routing_pca.png)

**分组分析（关键组）**
- **缺图组（n=1,318）**：CSGA **{g3('missing_image','missing_image','csga'):.4f}**、text {g3('missing_image','missing_image','text'):.4f}、image **{g3('missing_image','missing_image','image'):.6f}**、shared-1 {g3('missing_image','missing_image','shared-1'):.6f}、shared-2 {g3('missing_image','missing_image','shared-2'):.6f}；熵 {float(g[(g.dimension=='missing_image')&(g.group=='missing_image')]['mean_entropy'].iloc[0]):.3f}。这是全数据最强的路由信号（η²(image)=0.872，η²(CSGA)=0.443）。
- **文本长度**：short → CSGA {g3('text_length','short','csga'):.4f}、shared-1 {g3('text_length','short','shared-1'):.4f}、shared-2 {g3('text_length','short','shared-2'):.4f}；long → CSGA **{g3('text_length','long','csga'):.4f}**、shared-1 {g3('text_length','long','shared-1'):.4f}、shared-2 {g3('text_length','long','shared-2'):.4f}。η²(CSGA)=0.347，η²(shared-1)=0.490，η²(shared-2)=0.520。
- **Text–image consistency**：high → CSGA **{g3('consistency','high','csga'):.4f}**、shared-1 {g3('consistency','high','shared-1'):.4f}、shared-2 {g3('consistency','high','shared-2'):.4f}；low → CSGA {g3('consistency','low','csga'):.4f}、shared-1 {g3('consistency','low','shared-1'):.4f}、shared-2 {g3('consistency','low','shared-2'):.4f}。
- **Category**：text gate 的 η² 最大（0.136）；CSGA 的 η² 仅 0.009。类别主要移动 text expert，但不改变 CSGA/shared 的主轴。
- **Popularity**：效应较弱（η² ≤0.011），head 组 CSGA 略高、tail 组 shared-1/2 略高。

![IndSci group weights](figures/fig3_group_expert_weights.png)

![IndSci missing image](figures/fig_ind_missing_image.png)

**干预（IndSci，n={n_cases} 条 test 交互，排除 target 出现在 history 的 {excluded} 条；无 title-only 缓存；且特征缩放混合在 LayerNorm 后是 no-op，故 truncate-description 未测）**
- **mask-text**：gate 几乎不变（text {p4.loc['mask-text','gate_text']-p4.loc['baseline','gate_text']:+.4f}、CSGA {p4.loc['mask-text','gate_csga']-p4.loc['baseline','gate_csga']:+.4f}），表示漂移 **{p4.loc['mask-text','repr_cos_drift_mean']:.4f}**，HR@50 相对 **{rel('mask-text','HR@50'):+.2f}%**。
- **mask-image**：CSGA **{p4.loc['mask-image','gate_csga']-p4.loc['baseline','gate_csga']:+.4f}**，shared-1 {p4.loc['mask-image','gate_shared-1']-p4.loc['baseline','gate_shared-1']:+.4f}、shared-2 {p4.loc['mask-image','gate_shared-2']-p4.loc['baseline','gate_shared-2']:+.4f}；漂移 {p4.loc['mask-image','repr_cos_drift_mean']:.4f}，HR@50 {rel('mask-image','HR@50'):+.2f}%。
- **shuffle-image**：CSGA {p4.loc['shuffle-image','gate_csga']-p4.loc['baseline','gate_csga']:+.4f}，text {p4.loc['shuffle-image','gate_text']-p4.loc['baseline','gate_text']:+.4f}；漂移 **{p4.loc['shuffle-image','repr_cos_drift_mean']:.4f}**，HR@50 **{rel('shuffle-image','HR@50'):+.2f}%**。
- **noise-image**：漂移 {p4.loc['noise-image','repr_cos_drift_mean']:.4f}，HR@50 {rel('noise-image','HR@50'):+.2f}%。
- **解读**：IndSci 的 gate 对文本 mask 几乎没有反应（因为 text expert 在主路由轴上权重接近 0），但文本 mask 仍通过表示影响推荐（-5.9%）；图像 shuffle 同时改变表示和推荐（-7.5%）；mask-image 主要把 CSGA 的权重分给 shared，但推荐只掉 0.8%，说明 CSGA/text 仍提供冗余信息。

![IndSci interventions](figures/fig4_intervention_gate_shift.png)

---

## 4. Sports vs IndSci：路由结构对比

{gate_md}

![Top-1 and contribution](figures/fig_compare_gating.png)

![PC1 loadings](figures/fig_compare_pc1.png)

**对比结论**
- **Sports**：text ↔ shared-2 轴（PC1 98.2%，text/shared-2 载荷 -0.650/+0.746）；文本长度是最强分组因素；image 在混合项中仅 3.6%。
- **IndSci**：CSGA ↔ shared-1/shared-2 轴（PC1 93.5%，CSGA +0.828，shared-1 -0.401，shared-2 -0.388）；文本长度移动 CSGA vs shared；text 在混合项中仅 2.5%，image 28.4%。
- **共同点**：五个专家的平均 gate 都很接近 0.2；都没有 load-dead expert；路由都不是“均匀使用”，而是有清晰的条件结构。
- **不同点**：哪个专家“活跃”高度依赖数据集。Sports 的 text expert 承担主路由；IndSci 的 text expert 几乎不主导，CSGA 才是主专家。**不能把 Sports 的 text-length 结论直接外推到其它品类。**

---

## 5. Beauty / Books / Toys：只有元数据与最终融合特征

（数据统计见第 1 节表。）

**可用性**
- 这三个数据集只有 `csga_moe_collab_v2` 最终融合 embedding，没有 text_only / image_only / multimodal / CSGA 对齐特征。OMoE 的 gate 是在这些中间特征上计算的；只给最终融合向量**无法反推出 5 个 expert 的 gate**（融合是 `aligned + scale*(out_proj(concat)+Σg_k e_k)`，信息已经混合）。
- 因此本报告不对这三个数据集报告逐 expert gating；只报告 item 数、交互数、缺图数、文本长度、类别、流行度和主结果。
- 若需要 Full OMoE 分析，必须重新生成 text/image/multimodal 嵌入（Qwen3-VL，约 1–2 小时/数据集，取决于卡数），再跑 CSGA（约 15 秒–1 分钟）和 OMoE-v2（约 7 分钟）。这是可行的，但不再是“用现有记录”。

**主结果（既有记录，注意设定差异）**

{main_md}

- **Beauty strict**：HR@50 {float(main_df[main_df.Dataset=='Beauty strict']['HR@50'].iloc[0]):.4f}，NDCG@50 {float(main_df[main_df.Dataset=='Beauty strict']['NDCG@50'].iloc[0]):.4f}。
- **Books top1500**：HR@50 {float(main_df[main_df.Dataset=='Books top1500']['HR@50'].iloc[0]):.4f}，NDCG@50 {float(main_df[main_df.Dataset=='Books top1500']['NDCG@50'].iloc[0]):.4f}；item 只有 1,500 个，候选集远小于其它数据集，**不能与 Sports/IndSci 直接比较**。
- **Toys 5core-top30k**：HR@50 {float(main_df[main_df.Dataset=='Toys 5core-top30k']['HR@50'].iloc[0]):.4f}，NDCG@50 {float(main_df[main_df.Dataset=='Toys 5core-top30k']['NDCG@50'].iloc[0]):.4f}；strict 版本结果不同（HR@50 0.0088），说明设定选择会显著影响数值。

---

## 6. 结论与建议

1. **文本/图像模态的相对价值是数据集相关的**：Sports 文本强、图像弱（image 混合项 3.6%）；IndSci 的文本混合项只有 2.5%、图像混合项 28.4%，但下游 text-only 与 image-only 的 HR@50 接近（0.0566 vs 0.0523）。也就是说，IndSci 的 image expert 在融合中贡献大，并不等于 image-only 单独就优于 text-only；路由贡献与单模态预测能力是两件事。
2. **IndSci 的缺图信号非常强**：1,318 个缺图 item 几乎全部路由到 CSGA（gate 0.900），image gate 接近 0。这是“路由器确实感知模态缺失”的最干净证据；Sports 因缺图样本太少无法验证。
3. **OMoE 的增益依赖底座**：IndSci 上 CSGA alone 已经达到 HR@50 {float(ind[ind.Variant=='CSGA alone']['HR@50'].iloc[0]):.4f}，OMoE v2 为 {float(ind[ind.Variant=='CSGA + OMoE v2']['HR@50'].iloc[0]):.4f}，边际增益很小。Sports 上 OMoE v2 的增益需要与其它底座对比才能判断；不能只看 OMoE 的绝对值。
4. **不要跨数据集外推路由结论**：Sports 的“文本长度 → text expert”不适用于 IndSci；IndSci 的“长文本/高一致性 → CSGA”也不适用于 Sports。论文中应按数据集陈述，或把路由轴作为数据集特性。
5. **Beauty/Books/Toys 的 gating 分析需要重新生成中间特征**；现有记录只能支持元数据统计和最终融合结果。若需要，应按同一 pipeline 为 Beauty 或 Toys 补跑中间特征（优先选择缺图/文本长度分布更有代表性的数据集）。

---

## 7. 文件

- 本报告：`analysis/omoe_ind/report.md`
- IndSci 图：`analysis/omoe_ind/figures/fig1_expert_load.png`、`fig2_routing_heatmap.png`、`fig2_routing_pca.png`、`fig3_group_expert_weights.png`、`fig4_intervention_gate_shift.png`、`fig5_expert_cases.png`
- 跨数据集图：`fig_compare_ind_downstream.png`、`fig_compare_gating.png`、`fig_compare_pc1.png`、`fig_ind_missing_image.png`
- 表：`analysis/omoe_ind/results/cross_dataset_metadata.csv`、`cross_dataset_main_results.csv`、`ind_downstream_modalities.csv`、`cross_dataset_gating.csv`、`part1–part5*.csv/json`
- Sports 报告：`analysis/omoe_sports10/report.md`
"""
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote", os.path.join(OUT, "report.md"))


if __name__ == "__main__":
    main()
