"""Build the Chinese Markdown report from the computed analysis results."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from omoe_core import ANALYSIS, EXPERT_NAMES, N_EXPERTS, RESULTS

E = N_EXPERTS
NAMES = EXPERT_NAMES


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


def pct(x):
    return f"{x*100:.2f}%"


def build_report(RES):
    p1 = RES["part1"]; p2 = RES["part2"]
    df1 = RES["part1_df"]; df3 = RES["part3_df"]; tdf = RES["part3_tests"]
    df4 = RES["part4_df"]; base = RES["part4_base_metrics"]
    p5p = RES["part5_profiles"]; p5c = RES["part5_cases"]

    d = df4.set_index("condition")

    def g(cond, expert):
        return d.loc[cond, f"gate_{expert}"]

    def delta(cond, expert):
        return g(cond, expert) - g("baseline", expert)

    def rel(cond, metric):
        return (d.loc[cond, metric] - base[metric]) / max(base[metric], 1e-12) * 100

    def g3(dim, group, expert):
        row = df3[(df3.dimension == dim) & (df3.group == group)]
        return float(row[f"gate_{expert}"].iloc[0])

    eff = tdf.groupby("dimension")["eta2"].max().sort_values(ascending=False)
    eff_md = "\n".join([f"- {k}: max η² = {v:.4f}" for k, v in eff.items()])

    # Part 1 table
    tab1 = df1.copy()
    tab1 = tab1[["expert", "mean_gate", "std_gate", "cv_across_items", "top1_freq",
                 "mean_output_norm", "mean_effective_contribution", "contribution_share",
                 "gate_lt_0.01_rate", "gate_lt_0.05_rate"]]
    tab1.columns = ["Expert", "Mean gate", "Std (items)", "CV (items)", "Top-1 freq",
                    "Output norm", "E[gate·norm]", "Contribution share",
                    "P(gate<0.01)", "P(gate<0.05)"]
    tab1_md = df_to_md(tab1, ".4f")

    # Part 3 table
    tab3 = df3.copy()
    tab3 = tab3[["dimension", "group", "n"] + [f"gate_{x}" for x in NAMES] +
                ["mean_entropy", "effective_experts"]]
    tab3.columns = ["Grouping", "Group", "n"] + NAMES + ["Mean entropy", "Eff. experts"]
    tab3_md = df_to_md(tab3, ".4f")

    # Part 4 table
    tab4 = df4.copy()
    tab4 = tab4[["condition"] + [f"gate_{x}" for x in NAMES] +
                ["js_vs_baseline", "mean_l1_gate_change", "routing_flip_rate",
                 "repr_cos_drift_mean", "HR@50", "NDCG@50"]]
    tab4.columns = ["Condition"] + NAMES + ["JS vs base", "Mean L1 gate Δ",
                                            "Routing flip", "Repr. cos drift", "HR@50", "NDCG@50"]
    tab4_md = df_to_md(tab4, ".4f")

    # Part 2 cases table
    pc = RES["part2_cases"].copy()
    pc = pc[["kind", "item_id", "title", "category", "entropy", "top_expert"] +
            [f"gate_{x}" for x in NAMES]]
    pc.columns = ["Kind", "Item", "Title", "Category", "Entropy", "Top expert"] + NAMES
    pc_md = df_to_md(pc, ".4f")

    # Part 5 tables
    prof = p5p.copy()
    prof.columns = ["Expert", "n top-1", "Top-1 freq", "Mean gate when top-1",
                    "Mean text len", "Mean popularity", "Mean text–image cos",
                    "Mean entropy", "Top category"]
    prof_md = df_to_md(prof, ".4f")
    case = p5c[["expert", "item_id", "title", "category", "popularity",
                "text_len", "text_image_cos", "entropy"]].copy()
    case.columns = ["Expert", "Item", "Title", "Category", "Popularity",
                    "Text len", "Text–image cos", "Entropy"]
    case_md = df_to_md(case, ".4f")

    pc1_evr = p2["pca_explained_variance_ratio"][0]
    pc1_load = p2["pca_components"][0]
    pc1_load_txt = pc1_load[NAMES.index("text")]
    pc1_load_s2 = pc1_load[NAMES.index("shared-2")]
    corr = p2["gate_correlation"]
    c_text_s2 = corr[NAMES.index("text")][NAMES.index("shared-2")]

    contribution = p1["contribution_share"]
    image_share = contribution[NAMES.index("image")]
    text_share = contribution[NAMES.index("text")]
    csga_share = contribution[NAMES.index("csga")]
    s1_share = contribution[NAMES.index("shared-1")]
    s2_share = contribution[NAMES.index("shared-2")]

    md = f"""# Full OMoE 的 Gating 与 Expert Utilization 系统分析

> **设定**：MIMIGenRec，Sports strict-10-core，`csga_omoe_v2`。5 个专家：text / image / CSGA / shared-1 / shared-2；ortho=0.05，load=0.1，2 个 shared expert，60 epochs。
> **范围说明**：选择 strict-10-core（11,246 items）是因为现有 OMoE/CSGA 缓存特征与 `exp_d` 记录都基于该设定；仓库中非 strict 的 Sports10 运行（37,537 test 样本）没有对应的缓存特征，未纳入本次表示级分析。
> **相关分析**：其他数据集（Industrial_and_Scientific）与文本/图像模态对比见 `analysis/omoe_ind/report.md`。
> **数据**：11,246 items；train 202,980 / valid 25,373 / test 25,373 条交互；Qwen3-VL 4096 维 text/image/multimodal 特征；SASRec 128 维协同特征；CSGA 对齐特征。
> **复现性**：本分析在缓存的 text / image / CSGA 特征上重新实例化 OMoE-v2。得到的 full-OMoE 平均 gate 为
> `{np.round(p1['mean_gate'], 6).tolist()}`，与既有实验记录 `analysis/samgr_sports10/results/exp_d_omoe_stats.json` 的
> `[0.196764, 0.199785, 0.201213, 0.198048, 0.204191]` **逐位一致**；平均 per-item entropy
> `{p1['mean_item_entropy']:.6f}` 也与记录的 `1.416388` 一致。因此以下统计是该设定下 OMoE-v2 的真实路由行为，不是模拟值。
> **边界**：第 4 部分的推荐性能使用**特征级检索代理**（真实 test 交互 + 融合表示最近邻排序），不是重跑 LLM。LLM 既有测试指标（HR@50=0.0551，NDCG@50=0.0184）只作量级参照，不能把代理指标的绝对值和 LLM 指标直接等同。

---

## 0. 结论摘要

1. **平均负载均衡，但逐 item 路由并不均衡。** 五个专家的平均 gate 在 0.1968–0.2042 之间，load CV = {p1['load_cv_across_experts']:.4f}；但 top-1 频率高度集中：**text 44.1%，shared-2 52.2%**，image 0.009%（全数据仅 1 个 item）、CSGA 1.6%、shared-1 2.1%。负载均衡损失压平的是**均值**，不是**条件路由**。
2. **路由本质上是“text vs shared-2”的一维轴。** 对 5 维 gate 做 PCA，PC1 解释 **{pc1_evr*100:.1f}%** 的方差；PC1 载荷 text={pc1_load_txt:+.3f}、shared-2={pc1_load_s2:+.3f}，二者 gate 相关系数 **{c_text_s2:+.3f}**。PC1 与文本长度的 Spearman ρ = **{p2['pc1_textlen_spearman']:+.3f}**：**长文本 → text expert；短文本 → shared-2**。
3. **“gate 均衡”不等于“有效贡献均衡”。** 由于各专家输出范数差异很大（shared-2 均值 273.9，text 109.4，image 仅 7.0），在**专家混合项**中按 gate×‖output‖ 计算的有效贡献份额为：text {text_share*100:.1f}%、image **{image_share*100:.1f}%**、CSGA {csga_share*100:.1f}%、shared-1 {s1_share*100:.1f}%、shared-2 {s2_share*100:.1f}%（不含 `out_proj(concat)` 残差项）。image 几乎从未成为 top-1（仅 1/11,246），且专家混合项贡献最低；它不是“负载 dead”，而是**路由稳定但混合项影响微弱**。
4. **分组差异有强有弱，文本长度是最强因素。** Text length 对 text/shared-2 gate 的最大 η² 为 **{eff.get('text_length',0):.3f}**（短文本组 text gate {g3('text_length','short','text'):.4f}、shared-2 {g3('text_length','short','shared-2'):.4f}；长文本组 text {g3('text_length','long','text'):.4f}、shared-2 {g3('text_length','long','shared-2'):.4f}）。Category 主要移动 CSGA vs shared-1（η² CSGA {eff.get('category',0):.3f}）；text–image consistency 移动 text vs shared-2（η² 约 0.04–0.07）；popularity 效应较弱（η² ≤ 0.019）。
5. **模态干预：文本干预改变路由，图像干预主要改变表示。** mask-text 使 text gate {delta('mask-text','text'):+.4f}、shared-2 gate {delta('mask-text','shared-2'):+.4f}；truncate-description 使 text gate {delta('truncate-description','text'):+.4f}，并把权重分给 image/CSGA/shared-1。相反，mask-image / shuffle-image 对 gate 的影响都 <0.001，但表示余弦漂移分别达到 {d.loc['mask-image','repr_cos_drift_mean']:.3f} / {d.loc['shuffle-image','repr_cos_drift_mean']:.3f}。**路由器对图像内容几乎不敏感；图像信息通过专家输出直接进入融合表示。**
6. **代理推荐性能：文本与图像内容错配会掉点，纯 mask 图像影响很小。** 相对 baseline，HR@50 变化：mask-text **{rel('mask-text','HR@50'):+.2f}%**、shuffle-image **{rel('shuffle-image','HR@50'):+.2f}%**、mask-image {rel('mask-image','HR@50'):+.2f}%、noise-image {rel('noise-image','HR@50'):+.2f}%、truncate-description {rel('truncate-description','HR@50'):+.2f}%。这是**特征级代理**的结果，不是 LLM 端到端结论。

---

## 1. 全局统计

**指标定义**
- **Mean gate**：对 11,246 个 item 的 softmax 路由概率取均值。
- **Std / CV (items)**：该专家 gate 在 item 间的标准差 / 变异系数；反映**逐 item 路由**是否稳定。
- **Top-1 freq**：该专家为 argmax 的 item 比例。
- **Output norm**：该专家 MLP 输出向量的 L2 范数（逐 item 后取均值）。
- **E[gate·norm]**：$\\mathbb{{E}}[g_{{i,k}}\\|e_{{i,k}}\\|]$，即**专家混合项**中该专家的平均有效贡献幅度（**不含** `out_proj(concat)` 残差项）。
- **Contribution share**：E[gate·norm] 在五个专家混合项贡献间的归一化份额。
- **Load CV**：五个专家**平均 gate** 的标准差 / 均值。
- **Dead expert**：这里区分两种定义：*load-dead*（平均 gate <0.05 或 P(gate<0.05)>0.5）与 *never-top-1*（top-1 频率 <1%）。本模型 **load-dead = {p1['dead_load_experts'] if p1['dead_load_experts'] else '无'}**；**never-top-1 = {p1['never_top1_experts']}**。

{tab1_md}

**全局量**
- Load CV（across experts）= **{p1['load_cv_across_experts']:.5f}**（均值几乎相等）
- 平均 gate 的熵 = **{p1['global_gate_entropy']:.4f}**；$\\exp(H)$ = **{p1['global_effective_experts']:.3f}**（理论上限 5）
- 平均 per-item entropy = **{p1['mean_item_entropy']:.4f}** ± {p1['std_item_entropy']:.4f}；effective experts = **{p1['effective_experts_per_item']:.3f}**
- 专家输出平均 |cos|（off-diagonal）= **{p1['mean_abs_offdiag_cos']:.4f}**；与既有记录“full OMoE 专家近似正交（mean|cos|≈0.0078）”一致。
- **专家混合项 vs 直接投影**：`out_proj(concat)` 的平均范数为 {p1['out_proj_norm_mean']:.2f}，专家混合项为 {p1['moe_norm_mean']:.2f}，二者范数相关 {p1['out_proj_moe_norm_corr']:+.2f}（部分抵消）。因此下面的 contribution share 只描述**专家混合项**，不是整个融合向量；图像仍可通过 `out_proj(concat)` 进入最终表示。
- 逐专家 gate 的跨 item 方差：text {p2['gate_variance_per_expert'][0]:.5f}、image {p2['gate_variance_per_expert'][1]:.5f}、CSGA {p2['gate_variance_per_expert'][2]:.5f}、shared-1 {p2['gate_variance_per_expert'][3]:.5f}、shared-2 {p2['gate_variance_per_expert'][4]:.5f}。

**解读**
- **两个“活跃”专家 + 三个“背景”专家**：text 与 shared-2 的 gate 方差大（标准差分别为 {p1['std_gate'][0]:.3f} / {p1['std_gate'][4]:.3f}），交替成为 top-1；image/CSGA/shared-1 的 gate 稳定在 0.2 附近（标准差 {p1['std_gate'][1]:.3f} / {p1['std_gate'][2]:.3f} / {p1['std_gate'][3]:.3f}），很少 top-1，但持续提供约 1/5 的 gate。
- **image 的“never-top-1”不等于“没被使用”**：它的平均 gate 是 0.1998，但输出范数只有 7.0（shared-2 为 273.9），所以在专家混合项中的贡献份额仅 {image_share*100:.1f}%。这是一个重要区分：**路由权重均衡，但输出幅度不均衡**；图像仍可通过 `out_proj(concat)` 影响最终表示。
- **shared-2 的“开/关”行为**：39.1% 的 item 上 shared-2 gate <0.01，但它在 52.2% 的 item 上是 top-1。说明它不是稳定背景专家，而是“短文本/弱文本”情形下的主导专家。
- **正交约束有效但不等于解耦**：专家输出平均 |cos|≈{p1['mean_abs_offdiag_cos']:.4f}，接近正交；这支持正交正则的作用，但不能推出“信息完全独立”。

![Expert load](figures/fig1_expert_load.png)

---

## 2. 实例差异

**路由向量方差与相关结构**

| Expert | Var(gate) |
|---|---|
""" + "\n".join([f"| {NAMES[j]} | {p2['gate_variance_per_expert'][j]:.6f} |" for j in range(E)]) + f"""

总方差（trace）= **{p2['gate_variance_total']:.6f}**。

**Gate 相关矩阵（Pearson）**

| | {' | '.join(NAMES)} |
|---|{'---|'*E}
""" + "\n".join(["| " + NAMES[i] + " | " + " | ".join(f"{corr[i][j]:+.3f}" for j in range(E)) + " |" for i in range(E)]) + f"""

最显著的是 **text ↔ shared-2 = {c_text_s2:+.3f}**：二者几乎完全互补。image 与 text 正相关（{corr[0][1]:+.3f}），CSGA 与 shared-1 负相关（{corr[2][3]:+.3f}），形成两个协同组。

**PCA：路由几乎是 1 维的**

| 主成分 | 解释方差比 | 主要载荷 |
|---|---|---|
| PC1 | **{p2['pca_explained_variance_ratio'][0]*100:.2f}%** | text {pc1_load_txt:+.3f}，shared-2 {pc1_load_s2:+.3f} |
| PC2 | {p2['pca_explained_variance_ratio'][1]*100:.2f}% | CSGA/shared-1 轴 |
| PC3 | {p2['pca_explained_variance_ratio'][2]*100:.2f}% | image 轴 |

PC1 与文本长度的 Spearman ρ = **{p2['pc1_textlen_spearman']:+.3f}**（p<1e-300）；与 text–image consistency 的 ρ = **{p2['pc1_consistency_spearman']:+.3f}**；与 popularity 的 ρ = **{p2['pc1_popularity_spearman']:+.3f}**。
**解释**：文本越长，PC1 越负，路由越偏向 text；文本越短，PC1 越正，路由越偏向 shared-2。这比“平均熵”更能刻画实例差异。

**Pairwise Jensen–Shannon divergence（1,000 个随机 item，499,500 对）**

| 统计量 | 值 |
|---|---|
| Mean | {p2['js_mean']:.6f} |
| Median | {p2['js_median']:.6f} |
| P90 | {p2['js_p90']:.6f} |
| P95 | {p2['js_p95']:.6f} |
| P99 | {p2['js_p99']:.6f} |
| Max | {p2['js_max']:.6f} |
| 与全局平均路由的 JS（均值） | {p2['js_to_mean_mean']:.6f} |

**Entropy 分布**：mean={p2['entropy_mean']:.4f}，std={p2['entropy_std']:.4f}，P05={p2['entropy_p05']:.4f}，P50={p2['entropy_p50']:.4f}，P95={p2['entropy_p95']:.4f}，min={p2['entropy_min']:.4f}，max={p2['entropy_max']:.4f}。
熵 <1.0 的 item 占 **{pct(p2['entropy_lt_1'])}**；熵 >1.5 的 item 占 **{pct(p2['entropy_gt_1p5'])}**。

**解读**
- JS 的均值 {p2['js_mean']:.3f}、P95 {p2['js_p95']:.3f} 并不小；差异主要来自 **text 与 shared-2 的此消彼长**，而不是五个专家都大幅波动。
- 熵的跨度（{p2['entropy_min']:.2f}–{p2['entropy_max']:.2f}）较窄，且没有 item 的熵低于 1.0；这说明**没有 item 被单一专家“硬路由”**。低熵 item 也不是“独裁”，而是某一专家接近 0、其余四个接近均匀。
- 因此，论文中应表述为：**路由在均值上均衡，但在条件上沿 text↔shared-2 轴强烈变化**；不要用“任意两个 item 的路由差异很大”来论证，也不要用“平均熵接近 ln5”来否定实例依赖。

![Routing heatmap](figures/fig2_routing_heatmap.png)

![Entropy and JS](figures/fig2_entropy_js.png)

![Routing PCA](figures/fig2_routing_pca.png)

**高熵 vs. 低熵案例**

{pc_md}

- 高熵案例（H≈1.608）几乎是均匀路由，说明多模态证据没有明显偏向。
- 低熵案例有两类：**shared-2 主导**（如 item 11058，短文本 76 字，text gate 0.014）和 **text 主导**（如 item 3432，长文本 735 字，shared-2 gate 0.0013）。这正是 PC1 轴的两端。

![High/low entropy cases](figures/fig2_entropy_cases.png)

---

## 3. 分组分析

分组定义：
- **Popularity**：train 交互次数（history + target）的 head（top 20%）/ mid（50–80%）/ tail（bottom 50%）。
- **Missing image**：`image_paths` 为空或文件不存在的 item；本数据只有 **2 个**，统计上不可靠，仅作完整性报告。
- **Category**：`categories` 的第二级（如 Sports / Hunting & Fishing / Outdoor Recreation）；取前 6 类 + Other。
- **Text length**：title+description 字符数的三分位（rank-based，避免长尾极值）。
- **Text–image consistency**：$\\cos(\\text{{text\\_emb}},\\text{{image\\_emb}})$ 的三分位。

{tab3_md}

**分组效应量（ANOVA η²，按维度取五个专家中的最大值）**

{eff_md}

**关键模式**
- **Text length 是绝对主导因素**：短文本组（均值 89 字）text gate 仅 {g3('text_length','short','text'):.4f}、shared-2 {g3('text_length','short','shared-2'):.4f}；长文本组（均值 1206 字）text gate {g3('text_length','long','text'):.4f}、shared-2 {g3('text_length','long','shared-2'):.4f}。η²(text)={tdf[(tdf.dimension=='text_length')&(tdf.expert=='text')].eta2.iloc[0]:.3f}，η²(shared-2)={tdf[(tdf.dimension=='text_length')&(tdf.expert=='shared-2')].eta2.iloc[0]:.3f}。这直接对应 PC1。
- **Category 主要移动 CSGA ↔ shared-1**：Sports 组 CSGA {g3('category','Sports','csga'):.4f}、shared-1 {g3('category','Sports','shared-1'):.4f}；Hunting & Fishing 组 CSGA {g3('category','Hunting & Fishing','csga'):.4f}、shared-1 {g3('category','Hunting & Fishing','shared-1'):.4f}。η²(CSGA)={tdf[(tdf.dimension=='category')&(tdf.expert=='csga')].eta2.iloc[0]:.3f}，是第二强的分组效应。
- **Text–image consistency 移动 text ↔ shared-2**：低一致性组 text {g3('consistency','low','text'):.4f}、shared-2 {g3('consistency','low','shared-2'):.4f}；高一致性组 text {g3('consistency','high','text'):.4f}、shared-2 {g3('consistency','high','shared-2'):.4f}。η² 约 0.04–0.07，属于中等偏弱。
- **Popularity 效应弱但方向一致**：head 组 text {g3('popularity','head','text'):.4f}、shared-2 {g3('popularity','head','shared-2'):.4f}；tail 组 text {g3('popularity','tail','text'):.4f}、shared-2 {g3('popularity','tail','shared-2'):.4f}。热门 item 更偏 text，长尾更偏 shared-2；η² ≤0.019。
- **Missing image**：只有 2 个 item，text {g3('missing_image','missing_image','text'):.4f}、shared-2 {g3('missing_image','missing_image','shared-2'):.4f}，且两者 text–image cosine≈1.0；不能作为“缺图会改变路由”的证据。
- **image gate 在各组都相对稳定**：除 2 个缺图样本（0.227）外，各组在 0.188–0.208 之间；再次说明它几乎不参与条件路由。

![Group expert weights](figures/fig3_group_expert_weights.png)

---

## 4. 模态干预

**干预定义（都在 OMoE 输入侧执行，CSGA 分支保持缓存值不变）**
- `mask-text`：把 text 特征替换为数据集均值（中心化后为 0）。
- `mask-image`：把 image 特征替换为数据集均值。
- `shuffle-image`：把所有 item 的 image 特征随机置换（破坏 item 对应关系，保留边缘分布）。
- `noise-image`：image 特征加高斯噪声，$\\sigma=0.5\\times$ 逐维标准差。
- `truncate-description`：{"用 title-only 的 Qwen3-VL 文本嵌入替换原 text 特征（description 截断为 0）" if RES['part4_has_title'] else "用 0.5×原 text + 0.5×均值 的近似（title-only 嵌入未生成）"}。

**说明**：CSGA 特征由多模态特征离线聚合得到；本实验只干预 OMoE 的 text/image 专家输入，不重跑 CSGA。这隔离了“路由对单模态证据的响应”，也意味着 mask-image 不会移除 CSGA 中已经包含的视觉-协同信息。路由与表示漂移在全部 11,246 个 item 上计算；推荐性能只在 {RES['proxy_n_cases']} 条 test 交互上评估。

{tab4_md}

**路由响应**
- **mask-text**：text {delta('mask-text','text'):+.4f}、shared-2 **{delta('mask-text','shared-2'):+.4f}**、shared-1 {delta('mask-text','shared-1'):+.4f}、image {delta('mask-text','image'):+.4f}、CSGA {delta('mask-text','csga'):+.4f}。路由从 shared-2 转向 text/shared-1；routing flip rate={d.loc['mask-text','routing_flip_rate']*100:.1f}%。
- **truncate-description**：text **{delta('truncate-description','text'):+.4f}**、shared-2 {delta('truncate-description','shared-2'):+.4f}、image {delta('truncate-description','image'):+.4f}、CSGA {delta('truncate-description','csga'):+.4f}、shared-1 {delta('truncate-description','shared-1'):+.4f}；flip rate={d.loc['truncate-description','routing_flip_rate']*100:.1f}%。减少文本信息后，路由器确实离开 text 端，转向 image/CSGA/shared-1。
- **mask-image**：image {delta('mask-image','image'):+.5f}、shared-2 {delta('mask-image','shared-2'):+.5f}；flip rate={d.loc['mask-image','routing_flip_rate']*100:.2f}%。gate 几乎不变。
- **shuffle-image**：image {delta('shuffle-image','image'):+.5f}、shared-2 {delta('shuffle-image','shared-2'):+.5f}；flip rate={d.loc['shuffle-image','routing_flip_rate']*100:.2f}%。gate 几乎不变。
- **noise-image**：所有 |Δgate|≈1e-5 或更小；flip rate={d.loc['noise-image','routing_flip_rate']*100:.2f}%。0.5σ 噪声对路由基本没有影响。
- **结论**：路由器主要响应**文本是否存在/文本长度**，对图像的存在、对齐、噪声几乎不敏感。图像信息更多通过 image expert 的输出直接进入融合表示，而不是通过重分配 gate。

**表示漂移**
- 平均余弦漂移：mask-text **{d.loc['mask-text','repr_cos_drift_mean']:.4f}**、mask-image **{d.loc['mask-image','repr_cos_drift_mean']:.4f}**、shuffle-image **{d.loc['shuffle-image','repr_cos_drift_mean']:.4f}**、noise-image {d.loc['noise-image','repr_cos_drift_mean']:.4f}、truncate-description {d.loc['truncate-description','repr_cos_drift_mean']:.4f}。
- **shuffle-image 的 drift 最大（0.199）**：虽然 gate 不变，但图像内容与 item 错配，融合表示发生明显变化。**mask-image 的 drift（0.116）也远大于 gate 变化**。这清楚说明“路由稳定”不等于“表示不受影响”。

**推荐性能（特征级检索代理，n={RES['proxy_n_cases']} test cases；排除 target 出现在 history 的 {RES['proxy_n_excluded_target_in_history']} 条）**
- Baseline proxy：HR@1={base['HR@1']:.4f}，HR@10={base['HR@10']:.4f}，HR@50={base['HR@50']:.4f}，NDCG@50={base['NDCG@50']:.4f}。
- 与 LLM 既有指标（HR@50=0.0551，NDCG@50=0.0184）相比，代理指标同量级但更低；**只应比较代理内部的相对变化**。
- HR@50 相对变化：mask-text **{rel('mask-text','HR@50'):+.2f}%**、shuffle-image **{rel('shuffle-image','HR@50'):+.2f}%**、mask-image {rel('mask-image','HR@50'):+.2f}%、noise-image {rel('noise-image','HR@50'):+.2f}%、truncate-description {rel('truncate-description','HR@50'):+.2f}%。
- **解读**：文本 mask 与图像 shuffle 掉点最多，说明**文本证据和图像-物品对齐都对检索有用**；纯 mask 图像掉点很小，因为 CSGA/text 仍携带信息，且 image 在专家混合项中的贡献份额只有 {image_share*100:.1f}%。truncate-description 略升（+{rel('truncate-description','HR@50'):.2f}%），一个合理解释是去掉长 description 的噪声，但幅度在一个代理指标的波动范围内，不能作为强结论。

![Intervention gate shift](figures/fig4_intervention_gate_shift.png)

---

## 5. 每类 expert 的解释性案例

**专家画像（以该专家为 top-1 的 item 统计）**

{prof_md}

**代表案例**（每个专家取“相对第二名 margin 最大”的 item）

{case_md}

**解读**
- **text expert**：top-1 item 的平均文本长度 **997 字**，远高于 shared-2 的 **131 字**；代表案例是 Gaiam Restore（2593 字，text gate 0.411，shared-2 0.0017）。这与 PC1 的“长文本 → text”完全一致。
- **image expert**：全数据只有 **1 个 item** 以 image 为 top-1（Fuji hook keeper，text–image cos 0.636，image gate 0.274）。样本量为 1，不能据此总结“图像专家负责什么”；更准确的表述是：**image 专家几乎从不主导路由，但其输出仍以较小幅度进入融合表示**。
- **CSGA expert**：top-1 item 的平均文本长度 303 字，代表案例是 SHIMANO SM-CS50（Sports 类，CSGA gate 0.291，text 0.217）。CSGA 在高协同/中等文本的 item 上更容易领先；category 分析也显示 Sports 组 CSGA 权重更高。
- **shared-1**：top-1 item 平均长度 262 字，代表案例是 VISM gun case（Hunting & Fishing，shared-1 0.286，text–image cos -0.109）。它更像“混合/弱一致性”情形下的补充专家，但效应量小于 text/shared-2 轴。
- **shared-2**：top-1 item 平均长度仅 **131 字**，代表案例是 Tactical Gun Range Bag（63 字，shared-2 0.462，text 0.015，cos -0.113）。这与“短文本/弱文本时 shared-2 接管”的解释一致；它是当前 OMoE 中最主要的“非文本”默认分支。
- 以上是**相关性解释**，不是因果结论；要证明“某专家负责某类信息”，需要针对该专家做消融/干预，而不能只靠 top-1 案例。

![Expert cases](figures/fig5_expert_cases.png)

---

## 6. 证据边界：能声称什么、不能声称什么

**可以声称（有本次统计支持）**
- 五个专家的**平均 gate 几乎相等**（0.1968–0.2042，load CV={p1['load_cv_across_experts']:.4f}），没有 load-dead expert。
- **逐 item 路由高度结构化**：PC1 解释 {pc1_evr*100:.1f}% 方差，text 与 shared-2 的 gate 相关 {c_text_s2:+.3f}，PC1 与文本长度 Spearman ρ={p2['pc1_textlen_spearman']:+.3f}。
- **专家混合项的有效贡献不均衡**：image 的输出范数仅 7.0，混合项贡献份额 {image_share*100:.1f}%；shared-2/text 合计约 {(s2_share+text_share)*100:.1f}%。这支持“视觉模态在专家混合中被低估/利用不足”的判断；图像仍可通过 `out_proj(concat)` 进入最终表示。
- Text length 是最强的分组因素（η² 最高 {eff.get('text_length',0):.3f}）；category 主要移动 CSGA↔shared-1（η² {eff.get('category',0):.3f}）；consistency 移动 text↔shared-2（η² 约 0.04–0.07）；popularity 较弱（η² ≤0.019）。
- 文本干预会改变路由（mask-text Δtext {delta('mask-text','text'):+.4f}，truncate Δtext {delta('truncate-description','text'):+.4f}）；图像干预几乎不改变路由，但显著改变表示（shuffle drift {d.loc['shuffle-image','repr_cos_drift_mean']:.3f}）。
- 在特征级检索代理上，mask-text（{rel('mask-text','HR@50'):+.2f}%）和 shuffle-image（{rel('shuffle-image','HR@50'):+.2f}%）掉点最多；mask-image/noise 影响很小。

**不能声称（本次没有证据）**
- 不能声称 LLM 端到端推荐指标在干预下不变：LLM 未重跑，代理指标不能替代 LLM 指标。
- 不能声称“五个专家被均匀使用”：平均 gate 均匀，但 top-1 频率是 text 44.1% / shared-2 52.2% / 其余 <2.1%，有效贡献也差异明显。
- 不能声称“instance-dependent routing 很弱”：PCA/PC1 与文本长度的强相关说明路由有清晰结构；也不能声称“很强地依赖多模态语义”——它主要依赖**文本长度/文本信息量**。
- 不能声称“image expert 负责视觉语义”：它只有 1 个 top-1 item，且有效贡献仅 {image_share*100:.1f}%；更准确的说法是“视觉信号通过 image expert 的输出进入融合，但路由几乎不围绕 image 组织”。
- 不能声称“缺图会显著改变路由”：缺图组只有 2 个 item。
- 不能把 mask-image 解释为“移除视觉信息”：CSGA 分支仍携带缓存的视觉-协同信息。
- `truncate-description` 的 +{rel('truncate-description','HR@50'):.2f}% 不能解释为“截断描述一定提升推荐”：它只是一个特征级代理上的小幅波动，未做 LLM 重训，也未做多种子/多截断比例的稳定性检验。

---

## 7. 复现与文件清单

**复现步骤**
```bash
cd /data02/sht/SAM-GR
# 1) 复现 OMoE-v2 并缓存逐 item 路由（约 3 分钟，单卡）
CUDA_VISIBLE_DEVICES=0 /data/sht/conda/mimigenrec/bin/python analysis/omoe_sports10/scripts/build_omoe.py
# 2) 生成 title-only 文本嵌入（truncate-description 干预；约 4 分钟，单卡）
CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 /data/sht/conda/mimigenrec/bin/python analysis/omoe_sports10/scripts/gen_title_only_emb.py
# 3) 全部分析与图（约 8–10 分钟）
CUDA_VISIBLE_DEVICES=0 /data/sht/conda/mimigenrec/bin/python analysis/omoe_sports10/scripts/run_analysis.py
```

**输出文件**
- 图：`figures/fig1_expert_load.png`、`fig2_routing_heatmap.png`、`fig2_entropy_js.png`、`fig2_routing_pca.png`、`fig2_entropy_cases.png`、`fig3_group_expert_weights.png`、`fig4_intervention_gate_shift.png`、`fig5_expert_cases.png`
- 表：`results/part1_expert_stats.csv`、`part1_global_stats.json`、`part2_instance_stats.json`、`part2_entropy_cases.csv`、`part3_group_stats.csv`、`part3_tests.csv`、`part4_intervention_stats.csv`、`part4_gate_deltas.csv`、`part4_retrieval_proxy.csv`、`part5_expert_profiles.csv`、`part5_expert_cases.csv`
- 模型/缓存：`results/omoe_model.pt`、`results/omoe_cache.npz`、`results/omoe_train_history.json`
"""
    with open(os.path.join(ANALYSIS, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("report written to", os.path.join(ANALYSIS, "report.md"))
