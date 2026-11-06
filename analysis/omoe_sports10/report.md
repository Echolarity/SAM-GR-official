# Full OMoE 的 Gating 与 Expert Utilization 系统分析

> **设定**：MIMIGenRec，Sports strict-10-core，`csga_omoe_v2`。5 个专家：text / image / CSGA / shared-1 / shared-2；ortho=0.05，load=0.1，2 个 shared expert，60 epochs。
> **范围说明**：选择 strict-10-core（11,246 items）是因为现有 OMoE/CSGA 缓存特征与 `exp_d` 记录都基于该设定；仓库中非 strict 的 Sports10 运行（37,537 test 样本）没有对应的缓存特征，未纳入本次表示级分析。
> **相关分析**：其他数据集（Industrial_and_Scientific）与文本/图像模态对比见 `analysis/omoe_ind/report.md`。
> **数据**：11,246 items；train 202,980 / valid 25,373 / test 25,373 条交互；Qwen3-VL 4096 维 text/image/multimodal 特征；SASRec 128 维协同特征；CSGA 对齐特征。
> **复现性**：本分析在缓存的 text / image / CSGA 特征上重新实例化 OMoE-v2。得到的 full-OMoE 平均 gate 为
> `[0.196764, 0.199785, 0.201213, 0.198048, 0.204191]`，与既有实验记录 `analysis/samgr_sports10/results/exp_d_omoe_stats.json` 的
> `[0.196764, 0.199785, 0.201213, 0.198048, 0.204191]` **逐位一致**；平均 per-item entropy
> `1.416388` 也与记录的 `1.416388` 一致。因此以下统计是该设定下 OMoE-v2 的真实路由行为，不是模拟值。
> **边界**：第 4 部分的推荐性能使用**特征级检索代理**（真实 test 交互 + 融合表示最近邻排序），不是重跑 LLM。LLM 既有测试指标（HR@50=0.0551，NDCG@50=0.0184）只作量级参照，不能把代理指标的绝对值和 LLM 指标直接等同。

---

## 0. 结论摘要

1. **平均负载均衡，但逐 item 路由并不均衡。** 五个专家的平均 gate 在 0.1968–0.2042 之间，load CV = 0.0129；但 top-1 频率高度集中：**text 44.1%，shared-2 52.2%**，image 0.009%（全数据仅 1 个 item）、CSGA 1.6%、shared-1 2.1%。负载均衡损失压平的是**均值**，不是**条件路由**。
2. **路由本质上是“text vs shared-2”的一维轴。** 对 5 维 gate 做 PCA，PC1 解释 **98.2%** 的方差；PC1 载荷 text=-0.650、shared-2=+0.746，二者 gate 相关系数 **-0.990**。PC1 与文本长度的 Spearman ρ = **-0.943**：**长文本 → text expert；短文本 → shared-2**。
3. **“gate 均衡”不等于“有效贡献均衡”。** 由于各专家输出范数差异很大（shared-2 均值 273.9，text 109.4，image 仅 7.0），在**专家混合项**中按 gate×‖output‖ 计算的有效贡献份额为：text 25.3%、image **3.6%**、CSGA 22.0%、shared-1 23.0%、shared-2 26.1%（不含 `out_proj(concat)` 残差项）。image 几乎从未成为 top-1（仅 1/11,246），且专家混合项贡献最低；它不是“负载 dead”，而是**路由稳定但混合项影响微弱**。
4. **分组差异有强有弱，文本长度是最强因素。** Text length 对 text/shared-2 gate 的最大 η² 为 **0.779**（短文本组 text gate 0.0335、shared-2 0.4011；长文本组 text 0.3816、shared-2 0.0052）。Category 主要移动 CSGA vs shared-1（η² CSGA 0.243）；text–image consistency 移动 text vs shared-2（η² 约 0.04–0.07）；popularity 效应较弱（η² ≤ 0.019）。
5. **模态干预：文本干预改变路由，图像干预主要改变表示。** mask-text 使 text gate +0.0252、shared-2 gate -0.0476；truncate-description 使 text gate -0.0299，并把权重分给 image/CSGA/shared-1。相反，mask-image / shuffle-image 对 gate 的影响都 <0.001，但表示余弦漂移分别达到 0.116 / 0.199。**路由器对图像内容几乎不敏感；图像信息通过专家输出直接进入融合表示。**
6. **代理推荐性能：文本与图像内容错配会掉点，纯 mask 图像影响很小。** 相对 baseline，HR@50 变化：mask-text **-7.78%**、shuffle-image **-9.57%**、mask-image -0.90%、noise-image -0.80%、truncate-description +1.00%。这是**特征级代理**的结果，不是 LLM 端到端结论。

---

## 1. 全局统计

**指标定义**
- **Mean gate**：对 11,246 个 item 的 softmax 路由概率取均值。
- **Std / CV (items)**：该专家 gate 在 item 间的标准差 / 变异系数；反映**逐 item 路由**是否稳定。
- **Top-1 freq**：该专家为 argmax 的 item 比例。
- **Output norm**：该专家 MLP 输出向量的 L2 范数（逐 item 后取均值）。
- **E[gate·norm]**：$\mathbb{E}[g_{i,k}\|e_{i,k}\|]$，即**专家混合项**中该专家的平均有效贡献幅度（**不含** `out_proj(concat)` 残差项）。
- **Contribution share**：E[gate·norm] 在五个专家混合项贡献间的归一化份额。
- **Load CV**：五个专家**平均 gate** 的标准差 / 均值。
- **Dead expert**：这里区分两种定义：*load-dead*（平均 gate <0.05 或 P(gate<0.05)>0.5）与 *never-top-1*（top-1 频率 <1%）。本模型 **load-dead = 无**；**never-top-1 = ['image']**。

| Expert | Mean gate | Std (items) | CV (items) | Top-1 freq | Output norm | E[gate·norm] | Contribution share | P(gate<0.01) | P(gate<0.05) |
|---|---|---|---|---|---|---|---|---|---|
| text | 0.1968 | 0.1620 | 0.8231 | 0.4415 | 109.3968 | 9.7973 | 0.2529 | 0.0000 | 0.3125 |
| image | 0.1998 | 0.0128 | 0.0642 | 0.0001 | 6.9752 | 1.3950 | 0.0360 | 0.0000 | 0.0000 |
| csga | 0.2012 | 0.0250 | 0.1241 | 0.0164 | 41.6490 | 8.5351 | 0.2204 | 0.0000 | 0.0000 |
| shared-1 | 0.1980 | 0.0354 | 0.1788 | 0.0205 | 44.1914 | 8.9098 | 0.2300 | 0.0000 | 0.0000 |
| shared-2 | 0.2042 | 0.1855 | 0.9083 | 0.5215 | 273.8619 | 10.0963 | 0.2607 | 0.3905 | 0.4362 |

**全局量**
- Load CV（across experts）= **0.01292**（均值几乎相等）
- 平均 gate 的熵 = **1.6094**；$\exp(H)$ = **5.000**（理论上限 5）
- 平均 per-item entropy = **1.4164** ± 0.0753；effective experts = **4.122**
- 专家输出平均 |cos|（off-diagonal）= **0.0078**；与既有记录“full OMoE 专家近似正交（mean|cos|≈0.0078）”一致。
- **专家混合项 vs 直接投影**：`out_proj(concat)` 的平均范数为 29.15，专家混合项为 21.24，二者范数相关 -0.31（部分抵消）。因此下面的 contribution share 只描述**专家混合项**，不是整个融合向量；图像仍可通过 `out_proj(concat)` 进入最终表示。
- 逐专家 gate 的跨 item 方差：text 0.02623、image 0.00016、CSGA 0.00062、shared-1 0.00125、shared-2 0.03440。

**解读**
- **两个“活跃”专家 + 三个“背景”专家**：text 与 shared-2 的 gate 方差大（标准差分别为 0.162 / 0.185），交替成为 top-1；image/CSGA/shared-1 的 gate 稳定在 0.2 附近（标准差 0.013 / 0.025 / 0.035），很少 top-1，但持续提供约 1/5 的 gate。
- **image 的“never-top-1”不等于“没被使用”**：它的平均 gate 是 0.1998，但输出范数只有 7.0（shared-2 为 273.9），所以在专家混合项中的贡献份额仅 3.6%。这是一个重要区分：**路由权重均衡，但输出幅度不均衡**；图像仍可通过 `out_proj(concat)` 影响最终表示。
- **shared-2 的“开/关”行为**：39.1% 的 item 上 shared-2 gate <0.01，但它在 52.2% 的 item 上是 top-1。说明它不是稳定背景专家，而是“短文本/弱文本”情形下的主导专家。
- **正交约束有效但不等于解耦**：专家输出平均 |cos|≈0.0078，接近正交；这支持正交正则的作用，但不能推出“信息完全独立”。

![Expert load](figures/fig1_expert_load.png)

---

## 2. 实例差异

**路由向量方差与相关结构**

| Expert | Var(gate) |
|---|---|
| text | 0.026231 |
| image | 0.000164 |
| csga | 0.000623 |
| shared-1 | 0.001254 |
| shared-2 | 0.034401 |

总方差（trace）= **0.062674**。

**Gate 相关矩阵（Pearson）**

| | text | image | csga | shared-1 | shared-2 |
|---|---|---|---|---|---|
| text | +1.000 | +0.794 | -0.672 | +0.799 | -0.990 |
| image | +0.794 | +1.000 | -0.341 | +0.733 | -0.856 |
| csga | -0.672 | -0.341 | +1.000 | -0.759 | +0.620 |
| shared-1 | +0.799 | +0.733 | -0.759 | +1.000 | -0.837 |
| shared-2 | -0.990 | -0.856 | +0.620 | -0.837 | +1.000 |

最显著的是 **text ↔ shared-2 = -0.990**：二者几乎完全互补。image 与 text 正相关（+0.794），CSGA 与 shared-1 负相关（-0.759），形成两个协同组。

**PCA：路由几乎是 1 维的**

| 主成分 | 解释方差比 | 主要载荷 |
|---|---|---|
| PC1 | **98.24%** | text -0.650，shared-2 +0.746 |
| PC2 | 0.93% | CSGA/shared-1 轴 |
| PC3 | 0.80% | image 轴 |

PC1 与文本长度的 Spearman ρ = **-0.943**（p<1e-300）；与 text–image consistency 的 ρ = **-0.225**；与 popularity 的 ρ = **-0.135**。
**解释**：文本越长，PC1 越负，路由越偏向 text；文本越短，PC1 越正，路由越偏向 shared-2。这比“平均熵”更能刻画实例差异。

**Pairwise Jensen–Shannon divergence（1,000 个随机 item，499,500 对）**

| 统计量 | 值 |
|---|---|
| Mean | 0.097994 |
| Median | 0.055404 |
| P90 | 0.238812 |
| P95 | 0.250860 |
| P99 | 0.260191 |
| Max | 0.276043 |
| 与全局平均路由的 JS（均值） | 0.056173 |

**Entropy 分布**：mean=1.4164，std=0.0753，P05=1.3407，P50=1.3831，P95=1.5714，min=1.3252，max=1.6086。
熵 <1.0 的 item 占 **0.00%**；熵 >1.5 的 item 占 **16.58%**。

**解读**
- JS 的均值 0.098、P95 0.251 并不小；差异主要来自 **text 与 shared-2 的此消彼长**，而不是五个专家都大幅波动。
- 熵的跨度（1.33–1.61）较窄，且没有 item 的熵低于 1.0；这说明**没有 item 被单一专家“硬路由”**。低熵 item 也不是“独裁”，而是某一专家接近 0、其余四个接近均匀。
- 因此，论文中应表述为：**路由在均值上均衡，但在条件上沿 text↔shared-2 轴强烈变化**；不要用“任意两个 item 的路由差异很大”来论证，也不要用“平均熵接近 ln5”来否定实例依赖。

![Routing heatmap](figures/fig2_routing_heatmap.png)

![Entropy and JS](figures/fig2_entropy_js.png)

![Routing PCA](figures/fig2_routing_pca.png)

**高熵 vs. 低熵案例**

| Kind | Item | Title | Category | Entropy | Top expert | text | image | csga | shared-1 | shared-2 |
|---|---|---|---|---|---|---|---|---|---|---|
| high | 5368 | CastMate Systems Nylon Deck Loop, Pad Eye, Tie Down 3/8" Opening for Kayaks and Canoes. 10 Pieces | Sports | 1.6086 | shared-1 | 0.1982 | 0.2023 | 0.1880 | 0.2137 | 0.1979 |
| high | 3442 | Blue Desert SmarTube Hydration System | Outdoor Recreation | 1.6085 | shared-2 | 0.1857 | 0.2028 | 0.1951 | 0.2076 | 0.2089 |
| high | 4571 | Franklin Sports NFL Kids Football Uniform Set - NFL Youth Football Costume for Boys & Girls - Set Includes Helmet, Jerse... | Fan Shop | 1.6079 | shared-1 | 0.1841 | 0.2036 | 0.1908 | 0.2149 | 0.2066 |
| low | 11058 | Bekith 2 Pack Adjustable Powerlock Rod Holder with Combo Mount, Black Finish | Hunting & Fishing | 1.3252 | shared-2 | 0.0142 | 0.1757 | 0.2067 | 0.1414 | 0.4620 |
| low | 6598 | Continental Ultra Sport III Folding PureGrip Set of 2 | Sports | 1.3261 | shared-2 | 0.0137 | 0.1840 | 0.2163 | 0.1311 | 0.4550 |
| low | 3432 | Oakley Men's Oo9014 Gascan Rectangular Sunglasses | Sports & Outdoor Recreation Accessories | 1.3264 | text | 0.4127 | 0.2006 | 0.1633 | 0.2221 | 0.0013 |

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
- **Text–image consistency**：$\cos(\text{text\_emb},\text{image\_emb})$ 的三分位。

| Grouping | Group | n | text | image | csga | shared-1 | shared-2 | Mean entropy | Eff. experts |
|---|---|---|---|---|---|---|---|---|---|
| popularity | head | 2258 | 0.2205 | 0.2018 | 0.1986 | 0.2015 | 0.1776 | 1.4107 | 4.0990 |
| popularity | mid | 3402 | 0.2155 | 0.2014 | 0.1984 | 0.2016 | 0.1831 | 1.4124 | 4.1059 |
| popularity | tail | 5586 | 0.1758 | 0.1980 | 0.2040 | 0.1945 | 0.2278 | 1.4211 | 4.1416 |
| missing_image | has_image | 11244 | 0.1967 | 0.1998 | 0.2012 | 0.1980 | 0.2042 | 1.4164 | 4.1222 |
| missing_image | missing_image | 2 | 0.3010 | 0.2273 | 0.2207 | 0.2446 | 0.0064 | 1.3975 | 4.0451 |
| category | Sports | 4150 | 0.1784 | 0.2004 | 0.2166 | 0.1829 | 0.2217 | 1.4243 | 4.1551 |
| category | Hunting & Fishing | 2974 | 0.2089 | 0.1998 | 0.1893 | 0.2128 | 0.1893 | 1.4131 | 4.1087 |
| category | Outdoor Recreation | 2521 | 0.2084 | 0.1990 | 0.1908 | 0.2082 | 0.1936 | 1.4130 | 4.1083 |
| category | Exercise & Fitness | 761 | 0.1930 | 0.1981 | 0.2037 | 0.1869 | 0.2182 | 1.4045 | 4.0734 |
| category | Sports & Outdoor Recreation Accessories | 474 | 0.2296 | 0.2021 | 0.1961 | 0.2015 | 0.1707 | 1.4075 | 4.0857 |
| category | Fan Shop | 131 | 0.1796 | 0.1983 | 0.1963 | 0.1956 | 0.2302 | 1.4196 | 4.1356 |
| category | Other | 235 | 0.1987 | 0.1987 | 0.1967 | 0.2004 | 0.2055 | 1.4087 | 4.0907 |
| text_length | short | 3749 | 0.0335 | 0.1878 | 0.2160 | 0.1616 | 0.4011 | 1.4098 | 4.0953 |
| text_length | mid | 3748 | 0.1752 | 0.2033 | 0.2069 | 0.2082 | 0.2063 | 1.4814 | 4.3993 |
| text_length | long | 3749 | 0.3816 | 0.2082 | 0.1808 | 0.2243 | 0.0052 | 1.3579 | 3.8880 |
| consistency | low | 3749 | 0.1528 | 0.1955 | 0.2028 | 0.1896 | 0.2593 | 1.4149 | 4.1163 |
| consistency | mid | 3748 | 0.2067 | 0.2002 | 0.1997 | 0.2003 | 0.1931 | 1.4175 | 4.1269 |
| consistency | high | 3749 | 0.2307 | 0.2036 | 0.2011 | 0.2043 | 0.1602 | 1.4167 | 4.1235 |

**分组效应量（ANOVA η²，按维度取五个专家中的最大值）**

- text_length: max η² = 0.7786
- category: max η² = 0.2425
- consistency: max η² = 0.0683
- popularity: max η² = 0.0187
- missing_image: max η² = 0.0008

**关键模式**
- **Text length 是绝对主导因素**：短文本组（均值 89 字）text gate 仅 0.0335、shared-2 0.4011；长文本组（均值 1206 字）text gate 0.3816、shared-2 0.0052。η²(text)=0.779，η²(shared-2)=0.760。这直接对应 PC1。
- **Category 主要移动 CSGA ↔ shared-1**：Sports 组 CSGA 0.2166、shared-1 0.1829；Hunting & Fishing 组 CSGA 0.1893、shared-1 0.2128。η²(CSGA)=0.243，是第二强的分组效应。
- **Text–image consistency 移动 text ↔ shared-2**：低一致性组 text 0.1528、shared-2 0.2593；高一致性组 text 0.2307、shared-2 0.1602。η² 约 0.04–0.07，属于中等偏弱。
- **Popularity 效应弱但方向一致**：head 组 text 0.2205、shared-2 0.1776；tail 组 text 0.1758、shared-2 0.2278。热门 item 更偏 text，长尾更偏 shared-2；η² ≤0.019。
- **Missing image**：只有 2 个 item，text 0.3010、shared-2 0.0064，且两者 text–image cosine≈1.0；不能作为“缺图会改变路由”的证据。
- **image gate 在各组都相对稳定**：除 2 个缺图样本（0.227）外，各组在 0.188–0.208 之间；再次说明它几乎不参与条件路由。

![Group expert weights](figures/fig3_group_expert_weights.png)

---

## 4. 模态干预

**干预定义（都在 OMoE 输入侧执行，CSGA 分支保持缓存值不变）**
- `mask-text`：把 text 特征替换为数据集均值（中心化后为 0）。
- `mask-image`：把 image 特征替换为数据集均值。
- `shuffle-image`：把所有 item 的 image 特征随机置换（破坏 item 对应关系，保留边缘分布）。
- `noise-image`：image 特征加高斯噪声，$\sigma=0.5\times$ 逐维标准差。
- `truncate-description`：用 title-only 的 Qwen3-VL 文本嵌入替换原 text 特征（description 截断为 0）。

**说明**：CSGA 特征由多模态特征离线聚合得到；本实验只干预 OMoE 的 text/image 专家输入，不重跑 CSGA。这隔离了“路由对单模态证据的响应”，也意味着 mask-image 不会移除 CSGA 中已经包含的视觉-协同信息。路由与表示漂移在全部 11,246 个 item 上计算；推荐性能只在 24852 条 test 交互上评估。

| Condition | text | image | csga | shared-1 | shared-2 | JS vs base | Mean L1 gate Δ | Routing flip | Repr. cos drift | HR@50 | NDCG@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1968 | 0.1998 | 0.2012 | 0.1980 | 0.2042 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0404 | 0.0118 |
| mask-text | 0.2220 | 0.2033 | 0.2001 | 0.2180 | 0.1566 | 0.0022 | 0.1126 | 0.0256 | 0.1143 | 0.0372 | 0.0105 |
| mask-image | 0.1971 | 0.1999 | 0.2014 | 0.1979 | 0.2036 | 0.0000 | 0.0135 | 0.0039 | 0.1164 | 0.0400 | 0.0114 |
| shuffle-image | 0.1970 | 0.1998 | 0.2013 | 0.1979 | 0.2040 | 0.0000 | 0.0188 | 0.0056 | 0.1988 | 0.0365 | 0.0105 |
| noise-image | 0.1968 | 0.1998 | 0.2012 | 0.1981 | 0.2042 | 0.0000 | 0.0023 | 0.0006 | 0.0022 | 0.0400 | 0.0116 |
| truncate-description | 0.1668 | 0.2148 | 0.2183 | 0.2127 | 0.1874 | 0.0012 | 0.1589 | 0.2020 | 0.1220 | 0.0408 | 0.0115 |

**路由响应**
- **mask-text**：text +0.0252、shared-2 **-0.0476**、shared-1 +0.0199、image +0.0035、CSGA -0.0011。路由从 shared-2 转向 text/shared-1；routing flip rate=2.6%。
- **truncate-description**：text **-0.0299**、shared-2 -0.0168、image +0.0150、CSGA +0.0170、shared-1 +0.0147；flip rate=20.2%。减少文本信息后，路由器确实离开 text 端，转向 image/CSGA/shared-1。
- **mask-image**：image +0.00014、shared-2 -0.00057；flip rate=0.39%。gate 几乎不变。
- **shuffle-image**：image +0.00002、shared-2 -0.00019；flip rate=0.56%。gate 几乎不变。
- **noise-image**：所有 |Δgate|≈1e-5 或更小；flip rate=0.06%。0.5σ 噪声对路由基本没有影响。
- **结论**：路由器主要响应**文本是否存在/文本长度**，对图像的存在、对齐、噪声几乎不敏感。图像信息更多通过 image expert 的输出直接进入融合表示，而不是通过重分配 gate。

**表示漂移**
- 平均余弦漂移：mask-text **0.1143**、mask-image **0.1164**、shuffle-image **0.1988**、noise-image 0.0022、truncate-description 0.1220。
- **shuffle-image 的 drift 最大（0.199）**：虽然 gate 不变，但图像内容与 item 错配，融合表示发生明显变化。**mask-image 的 drift（0.116）也远大于 gate 变化**。这清楚说明“路由稳定”不等于“表示不受影响”。

**推荐性能（特征级检索代理，n=24852 test cases；排除 target 出现在 history 的 521 条）**
- Baseline proxy：HR@1=0.0019，HR@10=0.0111，HR@50=0.0404，NDCG@50=0.0118。
- 与 LLM 既有指标（HR@50=0.0551，NDCG@50=0.0184）相比，代理指标同量级但更低；**只应比较代理内部的相对变化**。
- HR@50 相对变化：mask-text **-7.78%**、shuffle-image **-9.57%**、mask-image -0.90%、noise-image -0.80%、truncate-description +1.00%。
- **解读**：文本 mask 与图像 shuffle 掉点最多，说明**文本证据和图像-物品对齐都对检索有用**；纯 mask 图像掉点很小，因为 CSGA/text 仍携带信息，且 image 在专家混合项中的贡献份额只有 3.6%。truncate-description 略升（+1.00%），一个合理解释是去掉长 description 的噪声，但幅度在一个代理指标的波动范围内，不能作为强结论。

![Intervention gate shift](figures/fig4_intervention_gate_shift.png)

---

## 5. 每类 expert 的解释性案例

**专家画像（以该专家为 top-1 的 item 统计）**

| Expert | n top-1 | Top-1 freq | Mean gate when top-1 | Mean text len | Mean popularity | Mean text–image cos | Mean entropy | Top category |
|---|---|---|---|---|---|---|---|---|
| text | 4965 | 0.4415 | 0.3729 | 996.7158 | 195.7865 | 0.1516 | 1.3674 | Sports |
| image | 1 | 0.0001 | 0.2741 | 93.0000 | 30.0000 | 0.6356 | 1.4444 | Hunting & Fishing |
| csga | 184 | 0.0164 | 0.2556 | 303.3098 | 186.4185 | 0.1802 | 1.5731 | Sports |
| shared-1 | 231 | 0.0205 | 0.2478 | 261.6797 | 189.7446 | 0.1621 | 1.5758 | Hunting & Fishing |
| shared-2 | 5865 | 0.5215 | 0.3758 | 131.2351 | 154.8941 | 0.0910 | 1.4467 | Sports |

**代表案例**（每个专家取“相对第二名 margin 最大”的 item）

| Expert | Item | Title | Category | Popularity | Text len | Text–image cos | Entropy |
|---|---|---|---|---|---|---|---|
| text | 287 | Gaiam Gaiam Restore | Exercise & Fitness | 145 | 2593 | 0.0382 | 1.3322 |
| image | 8296 | Fuji EHKM-BC E-Z Hook Keeper, Black Chrome | Hunting & Fishing | 30 | 93 | 0.6356 | 1.4444 |
| csga | 6023 | SHIMANO SM-CS50 Housing Stops Concave Mount | Sports | 175 | 236 | 0.0431 | 1.5301 |
| shared-1 | 10250 | VISM by NcStar CVCP2960T-28 Pistol Subgun Gun Case, Tan, 28" x 13" | Hunting & Fishing | 66 | 200 | -0.1092 | 1.5522 |
| shared-2 | 5252 | Tactical Gun Range Bag Deluxe Pistol Shooting Range Duffle Bags | Hunting & Fishing | 48 | 63 | -0.1131 | 1.3322 |

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
- 五个专家的**平均 gate 几乎相等**（0.1968–0.2042，load CV=0.0129），没有 load-dead expert。
- **逐 item 路由高度结构化**：PC1 解释 98.2% 方差，text 与 shared-2 的 gate 相关 -0.990，PC1 与文本长度 Spearman ρ=-0.943。
- **专家混合项的有效贡献不均衡**：image 的输出范数仅 7.0，混合项贡献份额 3.6%；shared-2/text 合计约 51.4%。这支持“视觉模态在专家混合中被低估/利用不足”的判断；图像仍可通过 `out_proj(concat)` 进入最终表示。
- Text length 是最强的分组因素（η² 最高 0.779）；category 主要移动 CSGA↔shared-1（η² 0.243）；consistency 移动 text↔shared-2（η² 约 0.04–0.07）；popularity 较弱（η² ≤0.019）。
- 文本干预会改变路由（mask-text Δtext +0.0252，truncate Δtext -0.0299）；图像干预几乎不改变路由，但显著改变表示（shuffle drift 0.199）。
- 在特征级检索代理上，mask-text（-7.78%）和 shuffle-image（-9.57%）掉点最多；mask-image/noise 影响很小。

**不能声称（本次没有证据）**
- 不能声称 LLM 端到端推荐指标在干预下不变：LLM 未重跑，代理指标不能替代 LLM 指标。
- 不能声称“五个专家被均匀使用”：平均 gate 均匀，但 top-1 频率是 text 44.1% / shared-2 52.2% / 其余 <2.1%，有效贡献也差异明显。
- 不能声称“instance-dependent routing 很弱”：PCA/PC1 与文本长度的强相关说明路由有清晰结构；也不能声称“很强地依赖多模态语义”——它主要依赖**文本长度/文本信息量**。
- 不能声称“image expert 负责视觉语义”：它只有 1 个 top-1 item，且有效贡献仅 3.6%；更准确的说法是“视觉信号通过 image expert 的输出进入融合，但路由几乎不围绕 image 组织”。
- 不能声称“缺图会显著改变路由”：缺图组只有 2 个 item。
- 不能把 mask-image 解释为“移除视觉信息”：CSGA 分支仍携带缓存的视觉-协同信息。
- `truncate-description` 的 +1.00% 不能解释为“截断描述一定提升推荐”：它只是一个特征级代理上的小幅波动，未做 LLM 重训，也未做多种子/多截断比例的稳定性检验。

---

## 7. 复现与文件清单

**复现步骤**
```bash
cd /data02/sht/MIMIGenRec
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
