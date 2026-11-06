# 其他设定与文本/图像模态：扩展分析

本报告回答“其他地方的数据、以及文本/图像相关数据”的问题。结论先行：

- **只有 Sports 和 Industrial_and_Scientific（IndSci）保存了完整的 text_only / image_only / multimodal / CSGA / OMoE 中间特征**，因此只有这两个设定能做逐 expert 的 gating 分析。
- **IndSci 的 OMoE 路由轴与 Sports 完全不同**：Sports 是 **text ↔ shared-2**，IndSci 是 **CSGA ↔ shared-1/shared-2**。IndSci 的 text expert 在专家混合项中只占 **2.5%**，而 image 占 **28.4%**；Sports 恰好相反（text 25.3%，image 3.6%）。
- **IndSci 的缺图组（1,318 items）是最强的路由信号**：CSGA gate **0.9003**，image gate **0.000389**，熵仅 **0.331**；η²(image)=0.872。Sports 只有 2 个缺图样本，无法做这个分析。
- **IndSci 的既有下游结果**显示 text-only（HR@50 0.0566）与 image-only（0.0523）接近、text 略高；CSGA alone 已经很强（0.0643），OMoE v2 最好（0.0650），但相对 CSGA alone 的增益只有 **+0.0007 HR@50**。主要增益来自 CSGA，而不是图像。
- **Beauty / Books / Toys 只保存了最终融合特征** `csga_moe_collab_v2`，没有 text/image/multimodal/CSGA 中间特征；因此无法从现有记录恢复逐 expert gating。下表给出可用的数据统计与主结果，并明确标注不可比之处。

> **可复现性边界（IndSci）**：Sports 的 OMoE 复现与 `exp_d` 记录逐位一致；IndSci 没有对应的 `exp_d` 记录，且 `image_only` 特征在 2026-10-24 重新生成（晚于 2026-10-19 的 `csga_omoe_v2` 运行）。因此 IndSci 的 gating 统计是对**当前特征缓存**的重建，不能声称与当时生产模型逐位一致；IndSci 的下游指标来自既有运行记录。缺图组的强信号还依赖 `image_only` 对缺图 item 的回退方式（当前缓存中缺图 item 的 `image_only` 不是正常视觉特征）；若生产运行的 `image_only` 处理不同，缺图组的 gate 数值可能变化，但“缺图会触发 CSGA 接管”的方向性结论仍值得进一步验证。

---

## 1. 数据与产物可用性

| Dataset | Items | Train | Test | Missing image | Text len mean | Text len median | Top category | Head/mid/tail pop | text/image/mm emb | CSGA aligned | Final fused emb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sports strict-10core | 11246 | 202980 | 25373 | 2 | 518.83 | 220.00 | Sports | 468/165/61 | 1/1/1 | True | True |
| IndSci | 27222 | 313089 | 39137 | 1318 | 506.54 | 216.00 | Additive Manufacturing Products | 243/59/17 | 1/1/1 | True | True |
| Beauty strict | 27510 | 622885 | 77861 | 12 | 406.69 | 192.00 | Hair Care | 1157/458/136 | 0/0/0 | False | True |
| Books top1500 | 1500 | 563480 | 70436 | 171 | 9311.86 | 3910.00 | Mystery | 5085/1854/621 | 0/0/0 | False | True |
| Toys 5core-top30k | 31131 | 557242 | 69656 | 7 | 650.67 | 482.00 | Preschool | 672/240/51 | 0/0/0 | False | True |

**说明**
- `text/image/mm emb` 表示是否存在 `text_only` / `image_only` / `multimodal` 三个 4096 维 embedding；`CSGA aligned` 表示是否有可用的 CSGA 对齐特征；`Final fused emb` 是最终 `csga_moe_collab_v2` 特征。
- **Sports strict-10core**：11,246 items，完整产物；本系列前一份报告已做 Full OMoE 分析。
- **IndSci**：27,222 items，完整产物；本报告新做 CSGA + OMoE 复现与 Full OMoE 分析。
- **Beauty**：27,510 items，只有最终融合特征；缺图 12 个（见上表）。
- **Books top1500**：1,500 items，只有最终融合特征；item 数很少，HR@50 与其它数据集不可直接比较。
- **Toys 5core-top30k**：31,131 items，只有最终融合特征；主结果与 strict 版本差异很大，比较时需注明设定。

---

## 2. IndSci：文本 / 图像 / 融合的端到端结果（既有记录）

| Variant | HR@10 | HR@50 | NDCG@10 | NDCG@50 |
|---|---|---|---|---|
| Text-only (task1/ep2) | 0.0362 | 0.0566 | 0.0286 | 0.0329 |
| Image-only | 0.0349 | 0.0523 | 0.0288 | 0.0325 |
| Multimodal | 0.0351 | 0.0547 | 0.0273 | 0.0315 |
| Multimodal+Collab | 0.0360 | 0.0544 | 0.0301 | 0.0341 |
| CSGA alone | 0.0419 | 0.0643 | 0.0342 | 0.0390 |
| CSGA + OMoE v1 | 0.0414 | 0.0635 | 0.0343 | 0.0391 |
| CSGA + OMoE v2 | 0.0433 | 0.0650 | 0.0353 | 0.0401 |

**解读**
- **text-only 与 image-only 接近，text 略高**：HR@50 0.0566 vs 0.0523，NDCG@50 0.0329 vs 0.0325。所以不能简单说“IndSci 图像比文本强”；两者是接近的，真正拉开差距的是 CSGA。
- **多模态融合本身没有超过 text-only 的 HR@50**：Multimodal 0.0547、Multimodal+Collab 0.0544，都低于 text-only 0.0566；但 Multimodal+Collab 的 NDCG@50（0.0341）高于 text-only（0.0329）。
- **CSGA alone 很强**：HR@50 0.0643，已经超过 text/image/multimodal。说明在 IndSci 上，协同图聚合的贡献很大。
- **OMoE v2 最好但增益很小**：HR@50 0.0650，相对 CSGA alone +0.0007；NDCG@50 0.0401 vs 0.0390。这是一个**诚实的天花板提示**：在 IndSci 上，融合的边际收益不大，主要收益来自 CSGA 本身。

![IndSci downstream](figures/fig_compare_ind_downstream.png)

---

## 3. IndSci：Full OMoE gating 与 expert utilization

**全局统计（27,222 items）**
- Mean gate：text 0.2012、image 0.1998、CSGA 0.1975、shared-1 0.2004、shared-2 0.2011；load CV = **0.0067**。
- 平均 per-item entropy = **1.3633**；effective experts = **3.909**（低于 Sports 的 4.122）。
- Top-1：CSGA **43.4%**、shared-2 **47.6%**、shared-1 **7.7%**、text 1.04%、image 0.29%。
- 专家混合项贡献份额：text **2.5%**、image **28.4%**、CSGA 23.8%、shared-1 22.8%、shared-2 22.5%。
- 专家输出平均 |cos| = 0.0052。

**路由轴**
- PC1 解释 **93.5%** 方差；载荷：text +0.019、image -0.057、CSGA **+0.828**、shared-1 **-0.401**、shared-2 **-0.388**。
- PC1 与文本长度 Spearman ρ = **+0.689**；与 text–image consistency ρ = **+0.311**；与 popularity ρ = **+0.145**。
- **解释**：IndSci 的 PC1 是 **CSGA vs shared-1/shared-2**，而不是 Sports 的 text vs shared-2。长文本/高一致性 → CSGA；短文本/低一致性 → shared-1/shared-2。text expert 的载荷接近 0，说明它几乎不参与主路由轴。

![IndSci routing heatmap](figures/fig2_routing_heatmap.png)

![IndSci routing PCA](figures/fig2_routing_pca.png)

**分组分析（关键组）**
- **缺图组（n=1,318）**：CSGA **0.9003**、text 0.0986、image **0.000389**、shared-1 0.000360、shared-2 0.000331；熵 0.331。这是全数据最强的路由信号（η²(image)=0.872，η²(CSGA)=0.443）。
- **文本长度**：short → CSGA 0.0585、shared-1 0.2841、shared-2 0.2839；long → CSGA **0.3896**、shared-1 0.0862、shared-2 0.0889。η²(CSGA)=0.347，η²(shared-1)=0.490，η²(shared-2)=0.520。
- **Text–image consistency**：high → CSGA **0.3111**、shared-1 0.1551、shared-2 0.1557；low → CSGA 0.1279、shared-1 0.2291、shared-2 0.2312。
- **Category**：text gate 的 η² 最大（0.136）；CSGA 的 η² 仅 0.009。类别主要移动 text expert，但不改变 CSGA/shared 的主轴。
- **Popularity**：效应较弱（η² ≤0.011），head 组 CSGA 略高、tail 组 shared-1/2 略高。

![IndSci group weights](figures/fig3_group_expert_weights.png)

![IndSci missing image](figures/fig_ind_missing_image.png)

**干预（IndSci，n=37166 条 test 交互，排除 target 出现在 history 的 1971 条；无 title-only 缓存；且特征缩放混合在 LayerNorm 后是 no-op，故 truncate-description 未测）**
- **mask-text**：gate 几乎不变（text +0.0002、CSGA +0.0007），表示漂移 **0.0385**，HR@50 相对 **-5.87%**。
- **mask-image**：CSGA **-0.0285**，shared-1 +0.0058、shared-2 +0.0090；漂移 0.0705，HR@50 -0.79%。
- **shuffle-image**：CSGA -0.0052，text +0.0169；漂移 **0.0825**，HR@50 **-7.54%**。
- **noise-image**：漂移 0.0030，HR@50 +0.79%。
- **解读**：IndSci 的 gate 对文本 mask 几乎没有反应（因为 text expert 在主路由轴上权重接近 0），但文本 mask 仍通过表示影响推荐（-5.9%）；图像 shuffle 同时改变表示和推荐（-7.5%）；mask-image 主要把 CSGA 的权重分给 shared，但推荐只掉 0.8%，说明 CSGA/text 仍提供冗余信息。

![IndSci interventions](figures/fig4_intervention_gate_shift.png)

---

## 4. Sports vs IndSci：路由结构对比

| Dataset | Load CV | Entropy | JS mean | PC1 var | PC1–textlen ρ | Top1 text | Top1 CSGA | Top1 shared-2 | Contrib text | Contrib image | Contrib CSGA | Contrib shared-1 | Contrib shared-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sports strict | 0.0129 | 1.4164 | 0.0980 | 0.9824 | -0.9428 | 0.4415 | 0.0164 | 0.5215 | 0.2529 | 0.0360 | 0.2204 | 0.2300 | 0.2607 |
| IndSci | 0.0067 | 1.3633 | 0.1242 | 0.9346 | 0.6886 | 0.0104 | 0.4343 | 0.4755 | 0.0255 | 0.2843 | 0.2380 | 0.2277 | 0.2246 |

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

| Dataset | Run | HR@10 | HR@50 | NDCG@10 | NDCG@50 |
|---|---|---|---|---|---|
| Sports strict-10core | Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict | 0.0218 | 0.0551 | 0.0112 | 0.0184 |
| IndSci | Industrial_and_Scientific-sft-dsz2_csga_omoe_v2 | 0.0433 | 0.0650 | 0.0353 | 0.0401 |
| Beauty strict | Beauty_and_Personal_Care-sft-dsz2_csga_omoe_v2_strict | 0.0086 | 0.0243 | 0.0044 | 0.0078 |
| Books top1500 | Books-sft-dsz2_csga_omoe_v2 | 0.0813 | 0.1788 | 0.0482 | 0.0691 |
| Toys 5core-top30k | Toys_and_Games-sft-dsz2_csga_omoe_v2 | 0.0407 | 0.0570 | 0.0324 | 0.0359 |

- **Beauty strict**：HR@50 0.0243，NDCG@50 0.0078。
- **Books top1500**：HR@50 0.1788，NDCG@50 0.0691；item 只有 1,500 个，候选集远小于其它数据集，**不能与 Sports/IndSci 直接比较**。
- **Toys 5core-top30k**：HR@50 0.0570，NDCG@50 0.0359；strict 版本结果不同（HR@50 0.0088），说明设定选择会显著影响数值。

---

## 6. 结论与建议

1. **文本/图像模态的相对价值是数据集相关的**：Sports 文本强、图像弱（image 混合项 3.6%）；IndSci 的文本混合项只有 2.5%、图像混合项 28.4%，但下游 text-only 与 image-only 的 HR@50 接近（0.0566 vs 0.0523）。也就是说，IndSci 的 image expert 在融合中贡献大，并不等于 image-only 单独就优于 text-only；路由贡献与单模态预测能力是两件事。
2. **IndSci 的缺图信号非常强**：1,318 个缺图 item 几乎全部路由到 CSGA（gate 0.900），image gate 接近 0。这是“路由器确实感知模态缺失”的最干净证据；Sports 因缺图样本太少无法验证。
3. **OMoE 的增益依赖底座**：IndSci 上 CSGA alone 已经达到 HR@50 0.0643，OMoE v2 为 0.0650，边际增益很小。Sports 上 OMoE v2 的增益需要与其它底座对比才能判断；不能只看 OMoE 的绝对值。
4. **不要跨数据集外推路由结论**：Sports 的“文本长度 → text expert”不适用于 IndSci；IndSci 的“长文本/高一致性 → CSGA”也不适用于 Sports。论文中应按数据集陈述，或把路由轴作为数据集特性。
5. **Beauty/Books/Toys 的 gating 分析需要重新生成中间特征**；现有记录只能支持元数据统计和最终融合结果。若需要，应按同一 pipeline 为 Beauty 或 Toys 补跑中间特征（优先选择缺图/文本长度分布更有代表性的数据集）。

---

## 7. 文件

- 本报告：`analysis/omoe_ind/report.md`
- IndSci 图：`analysis/omoe_ind/figures/fig1_expert_load.png`、`fig2_routing_heatmap.png`、`fig2_routing_pca.png`、`fig3_group_expert_weights.png`、`fig4_intervention_gate_shift.png`、`fig5_expert_cases.png`
- 跨数据集图：`fig_compare_ind_downstream.png`、`fig_compare_gating.png`、`fig_compare_pc1.png`、`fig_ind_missing_image.png`
- 表：`analysis/omoe_ind/results/cross_dataset_metadata.csv`、`cross_dataset_main_results.csv`、`ind_downstream_modalities.csv`、`cross_dataset_gating.csv`、`part1–part5*.csv/json`
- Sports 报告：`analysis/omoe_sports10/report.md`
