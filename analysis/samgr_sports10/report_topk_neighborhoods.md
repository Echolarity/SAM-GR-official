# Sports 10-core：四类表示空间的 Top-K 邻域与 CSGA 权重分析

> 数据：`Sports_and_Outdoors` strict 10-core；表示：collaborative / text / image / joint-multimodal；K ∈ {10, 20, 50}。
> 所有统计只使用训练集构造共现与转移，不使用 valid/test 标签。主分析排除 599 个在训练集中从未出现、SASRec 嵌入全零的商品（5.3%）；其余 10,647 个商品构成分析子集。全量结果在 `results/exp_e_*` 中保留，结论与主分析一致。

## 0. 结论摘要

1. **collaborative 与三类语义邻域几乎不重合。** K=10 时 collab–joint 的 Jaccard 仅 0.0069、neighbor recall overlap 仅 1.29%；text–joint 则达到 0.247 / 36.0%。collab 与语义邻域的重叠虽然显著高于 hub-preserving 零假设（约 14×），但只相当于真正语义邻域的 1/25 左右。K 增大时重叠缓慢上升，K=50 时 collab–joint recall 仍只有 2.79%，text–joint 为 38.76%。
2. **joint-multimodal 明显由文本主导。** text–joint 的 recall overlap（36.0%）远高于 image–joint（29.2%）和 text–image（15.1%）；Jaccard 分别为 0.247、0.188、0.091。joint 空间继承了文本语义，同时保留一部分视觉信息。
3. **collaborative 邻域的类别一致性很弱。** K=10 时 collab 的 leaf-category purity 只有 4.9%，text/image/joint 分别为 34.4%/44.0%/39.8%；L2-category purity 也只有 45.2% vs 70.3%/67.4%/70.7%。collab 邻域并非随机（随机约 0.47% leaf purity），但它主要停留在“大类/部门”层面，细粒度商品意图很弱。
4. **collaborative 的共现/转移优势远小于预期。** 相对 popularity-matched 随机邻居，K=10 时 collab 的用户共现概率为 4.50×、序列转移概率为 11.91×；text 为 4.33×/11.07×，joint 为 3.97×/9.89×。collab 只比语义邻域高约 4%–20%，并不是数量级上的优势。绝对概率也很低：collab 共现 1.60%、转移 0.257%，且只有 18.3% 的邻居对在训练用户中共现、3.3% 的邻居对有直接转移。
5. **NPMI 反而显示 collab 邻域的“意外共现”最弱。** K=10 时 NPMI：collab 0.414 < text 0.429 ≈ image 0.424 ≈ joint 0.423；随机基线约 0.275。collab 的共现概率略高，主要是邻居更偏热门（邻居平均 popularity percentile 54.2% vs 语义空间 50%–51.5%），NPMI 去 popularity 后优势消失。
6. **collab 邻域在 tail 商品上进一步退化。** leaf purity 从 head 的 7.3% 降到 tail 的 2.9%；text/image/joint 在三个分组中稳定在 32%–46%。collab 的 co-occurrence/transition lift 在 tail 反而更高（因为 tail 的随机基线更低），但绝对概率和类别一致性都更差。
7. **CSGA 学到的权重接近均匀、与原始 cosine 只有弱相关、且强烈不对称。** 权重–collab cosine 的全局 Spearman 仅 0.111（逐 item 均值 0.182，中位数 0.200），top-5 重合率 56.5%（随机 50%）。归一化 attention entropy 均值 0.978，有效邻居数 9.52/10；top-1 权重均值 0.161，top-5 权重质量 0.621（均匀为 0.5）。互惠边上的 \(|a_{ij}-a_{ji}|\) 均值 0.0395、中位数 0.0323，30.3% 的互惠对差异 >0.05，且 \(a_{ij}\) 与 \(a_{ji}\) 相关系数为 −0.179。
8. **CSGA 权重对序列转移只有温和提升。** top-5 权重邻居 vs 全部 10 个邻居：平均转移概率 1.24% vs 1.09%，mean lift 1.12、median lift 1.59；56.1% 的商品 lift>1（Wilcoxon p≈8.5e-11），但 43.9% 的商品没有提升，bottom-5 的 lift 为 0.88。对用户共现的提升更弱：aggregate ratio 仅 1.06、median 1.00。也就是说，CSGA 权重更像“轻微偏向序列一致邻居的平滑”，而不是强 collaborative 选择。
9. **case study 直观复现了上述分歧。** 6 个分层随机案例中，collab Top-10 与 joint Top-10 完全不相交；collab 邻居跨类别（如 Fitbit → 自行车包/泳镜/吊床/手套），joint 邻居全部是同一 leaf category（如 fitness trackers）。CSGA 的 top-4 权重都在 0.10–0.20，接近均匀，说明聚合主要是在“平均”这些跨类别邻居。
10. **与现有方案性能的关系。** CSGA+OMoE v2 在 HR@50/NDCG@50 上优于 SASRec 与 TIGER（0.0551/0.0184 vs 0.0449/0.0141、0.0199/0.0074），reranker 进一步提升到 0.0813/0.0323。但表示层面的证据表明，增益更可能来自 joint multimodal SID + OMoE 融合 + reranking，而不是一个干净的 collaborative 注意力图；collab 图本身噪声较大，CSGA 权重也接近均匀。

---

## 1. 数据、表示与评价口径

### 1.1 数据与表示

| 项目 | 数值 |
|---|---:|
| 商品数 | 11,246 |
| 训练用户数 | 17,060 |
| 训练 prefix-target 行数 | 202,980 |
| 去重后的 user–item 交互 | 217,233 |
| 非零共现商品对 | 2,935,796 |
| 有向相邻转移 | 1,728,908 |
| 零 collaborative 嵌入商品（排除） | 599（5.3%） |
| 主分析商品数 | 10,647 |
| leaf / L2 类别数 | 294 / 63 |

四个表示空间：

- **collaborative**：`sasrec_models/Sports_and_Outdoors/sasrec_item_emb.npy`，128 维 SASRec item embedding。
- **text**：`Sports_and_Outdoors.emb-Qwen3VL-text_only.npy`，4096 维。
- **image**：`Sports_and_Outdoors.emb-Qwen3VL-image_only.npy`，4096 维。
- **joint-multimodal**：`Sports_and_Outdoors.emb-Qwen3VL-multimodal.npy`，4096 维，由文本+图像联合编码。

所有向量做 L2 归一化，余弦相似度取 top-K，排除自身。K 固定为 10/20/50；K=50 的邻域是独立 top-50，不是 K=10 的简单扩展。

### 1.2 指标定义

- **Jaccard overlap**：\(|A_i^K \cap B_i^K| / |A_i^K \cup B_i^K|\)，对商品取平均。
- **Neighbor recall overlap**：\(|A_i^K \cap B_i^K| / K\)，即 prior report 中的 `overlap@K`。
- **Hub-preserving null recall**：随机置换另一个空间的 anchor，保留每个空间各自的邻居列表边缘分布。期望重叠为 \(\sum_v d_A(v)d_B(v)/(N^2K)\)，其中 \(d_A(v)\) 是 v 出现在多少个 top-K 列表中的度数。该零假设比“均匀随机 K 子集”更严格，因为 top-K 列表本身高度 hubby。
- **Category purity**：邻居中与目标商品同 leaf category 的比例；另报 L2（第三级类别，如 Cycling）purity 和邻居集合的 dominant-leaf share。
- **User-level co-occurrence probability**：\(P(j|i)=c_{ij}/n_i\)，\(c_{ij}\) 为同时交互过 i、j 的训练用户数，\(n_i\) 为交互过 i 的用户数。另报 coverage \(= \Pr(c_{ij}>0)\)。
- **Sequential transition probability**：在训练序列的相邻对中，\(P(j|i)=T_{ij}/\sum_{j'}T_{ij'}\)；跳过 i=j 的相邻重复。另报 coverage。
- **NPMI**：对 \(c_{ij}>0\) 的邻居对，\(\mathrm{NPMI}=\log\frac{P(i,j)}{P(i)P(j)} / (-\log P(i,j))\)。同时计算 +0.5 平滑版本；表中主值为 observed-pair NPMI，coverage 单独说明。
- **Text / visual cosine**：目标与邻居在 text / image 空间的余弦均值。
- **Head/mid/tail**：按训练集 unique-user popularity 排序，等商品数三等分；head 3,748、mid 3,748、tail 3,750（全量）。head/mid/tail 分别占 65.9%/22.8%/11.3% 的 user–item 交互。
- **Random controls**：对每个 anchor 从经验 popularity 分布独立抽 K 个商品（3 次平均），得到 popularity-matched random；用于 co-occurrence/transition lift 以及 category/cosine 的差值。

### 1.3 CSGA 权重反推

训练脚本没有保存 scorer checkpoint，但保存了 CSGA 聚合后的表示：

\[
h_i = m_i + 0.5 \sum_{j \in N_i} a_{ij} m_j,
\]

其中 \(m\) 是均值中心化的 joint-multimodal 特征。固定 collab top-10 邻居后，对每个商品解 10 维最小二乘：

\[
\min_{a_i} \left\| 2(h_i-m_i) - \sum_{j \in N_i} a_{ij} m_j \right\|_2^2.
\]

反推结果：平均相对残差 \(1.76\times10^{-5}\)，最大 \(3.38\times10^{-5}\)；权重和均为 1.000；裁剪前没有负权重。说明在该缓存表示上可以几乎精确恢复有效聚合权重。注意：这是 CSGA 聚合前的有效权重，不是 OMoE 融合后的权重；缓存来自同一 CSGA 过程的重跑，不是生产 checkpoint 的原始 scorer，详见第 8 节限制。

---

## 2. Top-K 空间重叠

![Top-K overlap](figures/fig_e_overlap_heatmap.png)

**表 1. 四空间两两 Top-K 重叠（主分析子集 N=10,647）。Null 为 hub-preserving 随机置换的期望 recall。**

| Pair | K | Jaccard | Recall overlap | Null recall | ×Null |
|---|---:|---:|---:|---:|---:|
| collab–text | 10 | 0.0079 | 0.0146 | 0.0009 | 16.0× |
| collab–text | 20 | 0.0104 | 0.0196 | 0.0018 | 10.7× |
| collab–text | 50 | 0.0160 | 0.0304 | 0.0046 | 6.7× |
| collab–image | 10 | 0.0067 | 0.0122 | 0.0009 | 13.5× |
| collab–image | 20 | 0.0090 | 0.0167 | 0.0018 | 9.2× |
| collab–image | 50 | 0.0134 | 0.0254 | 0.0045 | 5.6× |
| collab–joint | 10 | 0.0069 | 0.0129 | 0.0009 | 14.2× |
| collab–joint | 20 | 0.0092 | 0.0175 | 0.0018 | 9.6× |
| collab–joint | 50 | 0.0146 | 0.0279 | 0.0045 | 6.1× |
| text–image | 10 | 0.0908 | 0.1507 | 0.0010 | 158.3× |
| text–image | 20 | 0.0951 | 0.1603 | 0.0019 | 83.9× |
| text–image | 50 | 0.1033 | 0.1746 | 0.0048 | 36.7× |
| text–joint | 10 | 0.2473 | 0.3600 | 0.0011 | 319.7× |
| text–joint | 20 | 0.2544 | 0.3736 | 0.0022 | 166.5× |
| text–joint | 50 | 0.2626 | 0.3876 | 0.0057 | 68.4× |
| image–joint | 10 | 0.1884 | 0.2918 | 0.0011 | 262.6× |
| image–joint | 20 | 0.1964 | 0.3077 | 0.0022 | 137.8× |
| image–joint | 50 | 0.2074 | 0.3255 | 0.0057 | 57.2× |

**解读：**

- collab 与任一语义空间的 recall overlap 只有 1.2%–3.0%（K=10–50），Jaccard 只有 0.7%–1.6%。它们显著高于 null（6×–16×），说明不是完全随机；但相对 text–joint 的 36%–39%，几乎可以视为“不同邻域”。
- text–joint 的重叠远高于 image–joint，说明 joint multimodal 主要继承文本语义；image–joint 仍保留约 29%–33% 的 recall overlap，视觉信号没有被文本完全覆盖。
- text–image 只有 15%–18% recall overlap，说明文本与视觉确实提供了互补信息；joint 不是简单的 text + image 交集，而是更偏向 text 的联合表示。
- K 从 10 增到 50 时，collab–semantic 的 recall 从 1.2%–1.3% 升到 2.5%–3.0%，但 text–joint 只从 36.0% 升到 38.8%。collab 的重叠增长主要来自“更大的候选集合里偶然碰到”，而不是二者在同一语义邻域上收敛。

---

## 3. 邻域质量

![Neighborhood quality](figures/fig_e_neighborhood_quality.png)

**表 2. 各空间/各 K 的邻域质量（N=10,647）。Cooc/Trans 的 ×rand 为相对 popularity-matched random 的 lift；NPMI 为 observed-pair 均值。**

| Space | K | Cat leaf | Cat L2 | Cooc P | ×rand | Trans P | ×rand | NPMI | Text cos | Visual cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| collab | 10 | 0.049 | 0.452 | 0.0160 | 4.50× | 0.00257 | 11.91× | 0.414 | 0.895 | 0.763 |
| collab | 20 | 0.044 | 0.438 | 0.0141 | 4.05× | 0.00210 | 9.93× | 0.412 | 0.894 | 0.761 |
| collab | 50 | 0.038 | 0.416 | 0.0119 | 3.39× | 0.00158 | 7.44× | 0.411 | 0.893 | 0.759 |
| text | 10 | 0.344 | 0.703 | 0.0154 | 4.33× | 0.00239 | 11.07× | 0.429 | 0.968 | 0.845 |
| text | 20 | 0.277 | 0.660 | 0.0123 | 3.54× | 0.00167 | 7.88× | 0.424 | 0.964 | 0.834 |
| text | 50 | 0.188 | 0.591 | 0.0091 | 2.59× | 0.00106 | 4.99× | 0.419 | 0.959 | 0.817 |
| image | 10 | 0.440 | 0.674 | 0.0139 | 3.91× | 0.00215 | 9.96× | 0.424 | 0.922 | 0.924 |
| image | 20 | 0.379 | 0.634 | 0.0111 | 3.17× | 0.00157 | 7.39× | 0.420 | 0.918 | 0.914 |
| image | 50 | 0.282 | 0.566 | 0.0081 | 2.30× | 0.00097 | 4.58× | 0.417 | 0.914 | 0.898 |
| joint | 10 | 0.398 | 0.707 | 0.0141 | 3.97× | 0.00213 | 9.89× | 0.423 | 0.954 | 0.890 |
| joint | 20 | 0.323 | 0.661 | 0.0114 | 3.26× | 0.00151 | 7.15× | 0.420 | 0.950 | 0.878 |
| joint | 50 | 0.217 | 0.586 | 0.0085 | 2.40× | 0.00098 | 4.63× | 0.416 | 0.944 | 0.859 |

K=10 时 popularity-matched random 的参考值：leaf purity 0.0047，cooc 0.0035，transition 0.00022，NPMI 0.275，text cos 0.881，visual cos 0.741。K=50 时随机值略有变化（cooc 0.0035、transition 0.00021、NPMI 0.291、text cos 0.881、visual cos 0.741）。

### 3.1 Category purity

- K=10：image 的 leaf purity 最高（44.0%），joint 39.8%，text 34.4%，collab 仅 4.9%。L2 purity 则是 joint 70.7% ≈ text 70.3% > image 67.4% > collab 45.2%。
- collab 的 L2 purity 45.2% 说明它并非完全跨域：邻居大多仍在 Sports & Outdoors 大类内；但 leaf purity 4.9% 说明细粒度商品类型几乎不对齐。对推荐而言，这相当于“知道是运动户外，但不知道是哪种运动/哪种配件”。
- 随 K 增大，所有空间 purity 都下降（邻域被长尾/噪声稀释）。K=50 时 collab leaf purity 3.8%，text/image/joint 18.8%/28.2%/21.7%。image 对 K 最稳健（44.0%→28.2%），text 下降最快（34.4%→18.8%）。
- 邻居集合的 dominant-leaf share 也一致：K=10 时 collab 18.7%，text/image/joint 46.2%/56.5%/52.2%。collab 邻域内部本身就很杂。

### 3.2 User-level co-occurrence

- 绝对概率：K=10 collab 1.60%、text 1.54%、image 1.39%、joint 1.41%。collab 最高，但只比 text 高 4%，比 joint 高 13%。
- 相对 popularity-matched random：collab 4.50×、text 4.33×、image 3.91×、joint 3.97×。collab 的 lift 最高，但优势很小；K=50 时 collab 3.39× vs text 2.59×、joint 2.40×，优势扩大，但仍不是数量级差距。
- **Coverage 很低**：K=10 时只有 18.3% 的 collab 邻居对在训练用户中共现；text/image/joint 为 16.8%/15.2%/15.7%。K=50 时分别为 15.1%/11.5%/10.2%/10.9%。因此绝对概率应理解为“在稀疏观测下的条件概率”，lift 更可靠。
- 这也解释了为什么 collab 的 P(j|i) 略高但 NPMI 更低：collab 邻居的平均 popularity percentile 为 54.2%，语义空间为 50.0%–51.5%。热门商品更容易与任何商品共现，P(j|i) 被 popularity 抬高；NPMI 去掉了这部分。

### 3.3 Sequential transition

- K=10 绝对转移概率：collab 0.257%、text 0.239%、image 0.215%、joint 0.213%。collab 最高，但只比 text 高 7.6%。
- 相对 random：collab 11.91×、text 11.07×、image 9.96×、joint 9.89×。K=50 时 collab 7.44× vs text 4.99×、joint 4.63×，collab 的序列优势随 K 增大而更明显。
- **Coverage 更低**：K=10 时只有 3.3% 的 collab 邻居对有直接转移；text/image/joint 为 3.2%/2.8%/2.8%。K=50 时降到 2.2%/1.6%/1.4%/1.5%。这说明“邻居恰好是下一个商品”的事件非常稀疏；平均转移概率的置信度有限，lift 是更稳健的指标。
- 结论：collaborative 邻域确实携带一些序列信号，但远没有达到“collaborative 图 = 高质量转移图”的程度；它与语义邻域的差距很小。

### 3.4 NPMI

- K=10：collab 0.414、text 0.429、image 0.424、joint 0.423；随机 0.275。collab 最低。
- 随 K 增大，NPMI 基本稳定（collab 0.414→0.411，text 0.429→0.419）。语义空间的 NPMI 始终略高。
- 这说明 collab 邻居对虽然更容易共现（因为偏热门），但其共现的“意外性”不如语义邻居。对“发现非热门但相关商品”的目标，collab top-K 图并不是一个强关联图。

### 3.5 Text / visual cosine

- Text cosine：collab 0.895 vs random 0.881（+0.014）；text 0.968（+0.087）；joint 0.954（+0.073）；image 0.922（+0.041）。collab 邻居只比随机邻居在文本空间近一点点。
- Visual cosine：collab 0.763 vs random 0.741（+0.022）；image 0.924（+0.182）；joint 0.890（+0.149）；text 0.845（+0.104）。collab 的视觉增益也很小。
- 这与 category purity 一致：collab top-K 邻居在文本/视觉空间都只是“稍微接近”，远不是语义近邻。image 空间在视觉维度最强，text 空间在文本维度最强，joint 在两者之间且更偏文本。

### 3.6 Head / mid / tail

![Head/mid/tail](figures/fig_e_head_mid_tail.png)

**表 3. K=10 时按 popularity 分组的邻域质量（N=10,647；tail 因排除零嵌入商品为 3,151）。**

| Space | Group | Cat leaf | Cat L2 | Cooc P | ×rand | Trans P | ×rand | NPMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| collab | head | 0.073 | 0.535 | 0.0166 | 4.35× | 0.00253 | 9.89× | 0.353 |
| text | head | 0.368 | 0.730 | 0.0157 | 4.11× | 0.00221 | 8.66× | 0.369 |
| image | head | 0.463 | 0.698 | 0.0137 | 3.58× | 0.00189 | 7.40× | 0.365 |
| joint | head | 0.422 | 0.731 | 0.0138 | 3.62× | 0.00199 | 7.80× | 0.365 |
| collab | mid | 0.042 | 0.416 | 0.0170 | 4.80× | 0.00258 | 11.72× | 0.445 |
| text | mid | 0.323 | 0.674 | 0.0161 | 4.55× | 0.00252 | 11.45× | 0.456 |
| image | mid | 0.424 | 0.649 | 0.0151 | 4.26× | 0.00230 | 10.47× | 0.452 |
| joint | mid | 0.377 | 0.677 | 0.0152 | 4.29× | 0.00220 | 10.01× | 0.451 |
| collab | tail | 0.029 | 0.396 | 0.0139 | 4.32× | 0.00261 | 15.99× | 0.495 |
| text | tail | 0.341 | 0.706 | 0.0141 | 4.36× | 0.00245 | 14.96× | 0.507 |
| image | tail | 0.432 | 0.675 | 0.0127 | 3.93× | 0.00228 | 13.92× | 0.506 |
| joint | tail | 0.394 | 0.712 | 0.0130 | 4.04× | 0.00222 | 13.57× | 0.501 |

**解读：**

- **collab 的类别一致性随 popularity 下降而崩塌**：head 7.3% → mid 4.2% → tail 2.9%（leaf）；L2 也从 53.5% 降到 39.6%。语义空间在三个分组中基本稳定（leaf 32%–46%，L2 65%–73%），说明语义邻域对 tail 更稳健。
- collab 的 cooc/transition lift 在 tail 反而更高（4.32×/15.99×），但这主要是因为 tail 的随机基线更低，且 tail 的绝对 cooc（1.39%）和 transition（0.261%）并不比 head/mid 高。不能把“lift 高”误读为“tail 邻域质量好”。
- NPMI 在 mid/tail 更高（collab 0.445/0.495 vs head 0.353），语义空间也类似（0.45–0.51）。这同样反映 tail 商品边际概率低，共现更“意外”，不代表邻域更精确。
- 对论文的启示：**collaborative 图对 head 尚有一定 coarse-category 信号，对 tail 基本失效；语义图在 tail 上明显更可靠。** 如果方法只在整体平均上报告，会掩盖 collab 在 tail 上的退化。

---

## 4. CSGA learned weights

![CSGA weights](figures/fig_f_csga_weights.png)

**表 4. CSGA 权重统计（K=10，N=10,647，106,470 条有向边）。**

| Metric | Value |
|---|---|
| Weight–collab cosine Spearman（全局） | 0.111 |
| Weight–collab cosine Spearman（逐 item 均值 / 中位数） | 0.182 / 0.200 |
| 逐 item Spearman 的 P10 / P90 | −0.261 / 0.600 |
| Top-5 weight/cosine 重合率 | 56.5%（随机 50%） |
| Weight–text / image cosine Spearman | 0.017 / 0.028 |
| 互惠边比例 | 52.7%（56,074 对） |
| Mean / median \(\|a_{ij}-a_{ji}\|\) | 0.0395 / 0.0323 |
| P90 \(\|a_{ij}-a_{ji}\|\)；差异 >0.05 的比例 | 0.0831；30.3% |
| Corr(\(a_{ij}, a_{ji}\)) | −0.179 |
| 归一化 attention entropy 均值 / 中位数 | 0.978 / 0.981 |
| Effective neighbors（exp H） | 9.52 / 10 |
| Top-1 / top-5 权重质量 | 0.161 / 0.621 |
| Top-5 vs unweighted transition P | 1.24% vs 1.09% |
| Transition lift mean / median | 1.12 / 1.59 |
| Transition lift >1 的商品比例 | 56.1%（n=2,514） |
| Bottom-5 transition P / lift | 0.93% / 0.88 |
| Top-5 vs unweighted co-occurrence P | 1.69% vs 1.60% |
| Co-occurrence lift（aggregate / median） | 1.06 / 1.00 |

### 4.1 权重–cosine 相关性弱

- 全局 Spearman 0.111，逐 item 均值 0.182、中位数 0.200；P10 为 −0.261，说明约 10% 的商品上权重与 cosine 负相关。
- top-5 重合率 56.5%，只比随机 50% 高 6.5 个百分点。也就是说，CSGA 学到的排序与原始 collaborative cosine 排序高度不一致；scorer 并没有简单放大 cosine 最高的邻居。
- 与 text/image cosine 的 Spearman 只有 0.017/0.028，接近 0。权重不是语义驱动的，这是符合设计的；但也说明它没有从语义特征中得到额外校正。
- 可能的解释：CSGA scorer 的输入是 `[s_i, s_j, cos_ij]`，在 60 epoch、K=10、softmax 温度 0.1 的训练下，模型学到的是“重建友好的平滑权重”，而不是“cosine 重排”。这与 entropy 接近 1 一致。

### 4.2 方向不对称

- 互惠边占 52.7%；在互惠边上，\(|a_{ij}-a_{ji}|\) 均值 0.0395、中位数 0.0323、P90 0.0831；30.3% 的互惠对差异 >0.05。
- \(a_{ij}\) 与 \(a_{ji}\) 的相关系数为 **−0.179**：不是简单的“你高我也高”，而是弱反向。一个可能原因是 softmax 的分母不同：i 的邻居集合和 j 的邻居集合不同；当 j 是 i 的近邻时，j 可能把权重分给了比 i 更近的其他邻居。
- 对方法含义：如果下游把 CSGA 图当作对称 collaborative 亲和度使用，这个假设不成立；如果允许有向注意力，则这种不对称是模型表达力的一部分，但需要验证它是否带来稳定收益。当前证据（transition lift 仅 1.12）不支持“强有向选择”。

### 4.3 Attention entropy 接近均匀

- 归一化 entropy 均值 0.978（1 = 完全均匀），中位数 0.981；有效邻居数 9.52/10。
- top-1 权重均值 0.161（均匀 0.1），top-5 质量 0.621（均匀 0.5）。权重确实有轻微集中，但远不是 sharp attention。
- 换句话说，CSGA 的聚合 \(h_i = m_i + 0.5\sum_j a_{ij}m_j\) 主要接近“top-10 邻居均值”，而不是“选几个最相关邻居”。这可以解释为什么 collab 图的类别噪声会直接进入聚合表示：平均了 10 个跨类别邻居。

### 4.4 Transition lift

- top-5 权重邻居的平均转移概率 1.24%，全部 10 个邻居 1.09%，bottom-5 0.93%；mean lift 1.12、median 1.59。
- 56.1% 的商品 lift>1，Wilcoxon 符号秩检验 p≈8.5e-11；但 43.9% 的商品没有提升，bottom-5 的 lift 0.88。分布是双峰的：一部分商品权重确实集中到有转移的邻居，另一部分完全相反。
- **Top-m 敏感性**：top-1 的 mean lift 1.26 但只有 16.6% 商品 >1；top-3 mean 1.19、39.4% >1；top-5 mean 1.12、56.1% >1；top-7 mean 1.07、72.3% >1；top-9 mean 1.03、90.1% >1。随着纳入更多邻居，lift 向 1 回落，说明“最靠前的少数邻居”确实更有序列一致性，但覆盖率低。
- 对用户共现的提升更弱：top-5 1.69% vs all 1.60%，aggregate ratio 1.06，median 1.00，只有 46.8% 的商品 >1（在 7,051 个有非零共现的商品上）。**CSGA 权重更像是在选“序列一致”的邻居，而不是“共同购买”的邻居。**
- 按 head/mid/tail：weight–cosine Spearman 0.177/0.187/0.180，entropy 0.978/0.978/0.978，transition lift 1.117/1.110/1.154。没有明显的 popularity 依赖；tail 的 lift 略高，但绝对转移概率仍低。

### 4.5 小结

CSGA 的 learned weights 不是“强 collaborative 注意力”：
- 与 cosine 弱相关（0.111 global / 0.182 per-item）；
- 接近均匀（entropy 0.978）；
- 方向不对称（mean |diff| 0.0395，corr −0.179）；
- 对序列转移只有温和提升（mean 1.12，median 1.59），对共现几乎无提升（1.06）。

这并不等于 CSGA 无效：即使接近均匀，它也可能通过“把 collab 邻居的 collaborative 信号注入语义表示”而起作用。但如果论文的主张是“CSGA 学到了有区分度的 collaborative 邻居权重”，当前证据不足，需要更强的正则/监督或更干净的图。

---

## 5. Case study

### 5.1 选择协议

- 候选池：有完整图像、训练集 unique-user popularity ≥5、collab Top-10 与 joint Top-10 完全不相交、collab 的 leaf purity=0、joint 的 leaf purity≥0.8、CSGA top-1 权重≥0.15。
- 在候选池中按 L2 类别分层，固定 seed=2026 随机抽 6 个类别，每个类别随机抽 1 个商品。这样既满足“随机选择”，又保证类别多样和图像可读。
- 展示：目标商品、按 CSGA 权重排序的 collaborative Top-4（下方条形为 learned weight）、joint-multimodal Top-4（下方条形为 cosine）。

**表 5. Case study 列表。**

| Case | Item | Category | Group | Pop | collab∩sem | collab purity | sem purity | top-1 w | H/Hmax |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 6684 | Fitness Technology > Activity & Fitness Trackers | mid | 13 | 0/10 | 0.0 | 0.9 | 0.173 | 0.962 |
| 2 | 2715 | Leisure Sports & Game Room > Pickleball | tail | 9 | 0/10 | 0.0 | 1.0 | 0.187 | 0.976 |
| 3 | 10142 | Electronics & Gadgets > Compasses | mid | 16 | 0/10 | 0.0 | 1.0 | 0.163 | 0.974 |
| 4 | 8364 | Accessories > Bike Racks | mid | 12 | 0/10 | 0.0 | 0.9 | 0.155 | 0.980 |
| 5 | 433 | Boating & Sailing > Battery Chargers | mid | 16 | 0/10 | 0.0 | 0.8 | 0.176 | 0.965 |
| 6 | 10663 | Shooting > Targets & Accessories | mid | 15 | 0/10 | 0.0 | 0.9 | 0.204 | 0.935 |

![Case study 1](figures/fig_case_study_part1.png)
![Case study 2](figures/fig_case_study_part2.png)

完整 6 行合并图：`figures/fig_case_study_combined.png`。

### 5.2 观察

- **Case 1（Fitbit Alta）**：collab 邻居是 bike pack accessories、goggles、hammocks、gloves；joint 邻居全部是 fitness trackers。CSGA 权重 0.11–0.17，接近均匀，等于把四类完全不同的商品平均进 Fitbit 的表示。
- **Case 2（Pickleball paddle）**：collab 邻居是 reflective gear、handlebar bags、headlamps、CO2 pumps；joint 邻居全部是 pickleball paddles。权重 0.10–0.19。
- **Case 3（Compass）**：collab 邻居是 emergency fire、tactical backpacks、game belts、first aid kits；joint 邻居全部是 compasses。这是“户外/战术”大类相关、但细粒度不对齐的典型例子。
- **Case 4（Thule roof rack）**：collab 邻居是 sports sunglasses、bike pumps、cranksets、water bottles；joint 邻居全部是 bike racks。collab 邻居都是“骑行相关”，但不是车顶架本身。
- **Case 5（Minn Kota battery charger）**：collab 邻居是 anchors、bilge pumps、interior lights、ladders；joint 邻居全部是 battery chargers。collab 是“船/户外”场景相关，但不是充电器。
- **Case 6（splatter targets）**：collab 邻居是 hunting bags、boresighters、grips、scope mounts；joint 邻居全部是 targets。collab 是“射击/狩猎”场景相关，但不是靶纸。

这些案例说明：**collaborative 邻域常常提供“场景/大类相关”但“细粒度错误”的邻居；joint 语义邻域提供“同 leaf category”的邻居。** CSGA 的 near-uniform 权重把这些跨类别邻居几乎等权平均，可能把有用的场景信号和有害的细粒度噪声一起注入。

---

## 6. 与现有 MIMIGenRec 方案性能的关系

**表 6. Sports strict 10-core 上现有方案的结果（来自已有实验记录；不同方法训练预算不同，仅作参考）。**

| Method | HR@1 | HR@10 | HR@50 | NDCG@10 | NDCG@50 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| SASRec（item-ID） | 0.0028 | 0.0148 | 0.0449 | 0.0077 | 0.0141 | best epoch 11 |
| TIGER（text-only SID） | 0.0017 | 0.0090 | 0.0199 | 0.0050 | 0.0074 | from scratch，n=24,819 |
| CSGA+OMoE v2（strict，2 ep） | 0.0039 | 0.0218 | 0.0551 | 0.0112 | 0.0184 | 主模型 |
| CSGA+OMoE v2（strict，4 ep） | 0.0035 | 0.0199 | 0.0558 | 0.0103 | 0.0180 | 4 ep |
| CSGA+OMoE v2（strict，10 ep） | 0.0036 | 0.0187 | 0.0521 | 0.0099 | 0.0170 | 10 ep |
| CSGA+OMoE v2（10 ep + anneal） | 0.0038 | 0.0189 | 0.0569 | 0.0101 | 0.0183 | anneal |
| Reranker safe（21f） | 0.0123 | 0.0373 | 0.0813 | 0.0228 | 0.0323 | coverage 12.2%，post-hoc |

> 说明：repo 中没有 Sports 10-core 的 Caser/GRU 现成结果，本次未重训；性能表覆盖已有记录的 SASRec、TIGER、CSGA+OMoE 与 reranker。SASRec 的 item embedding 同时是本文 collaborative 空间的来源。

**关系与解读：**

- CSGA+OMoE v2（2 ep）相对 SASRec：HR@50 +22.7%、NDCG@50 +30.2%；相对 TIGER：HR@50 +176.8%、NDCG@50 +149.0%。说明“多模态语义 + CSGA/OMoE”整体上比纯 item-ID 序列模型和 text-only SID 更强。
- 但表示分析显示，这个增益不太可能来自“干净的 collaborative 邻域选择”：collab 邻域类别纯度低、与语义邻域几乎不重合、CSGA 权重接近均匀且只带来 1.12× 的转移 lift。更合理的解释是：
  1. joint multimodal SID 提供了比 text-only 更丰富的语义候选；
  2. OMoE 的 residual/fusion 让模型在语义和协同信号之间做条件混合；
  3. reranker 进一步利用了多源候选（CF/text/multimodal/popularity），把 HR@50 从 0.0551 推到 0.0813。
- Reranker safe 相对主模型：HR@50 +47.6%、NDCG@50 +75.4%；相对 SASRec：HR@50 +81.2%、NDCG@50 +128.3%。这说明**候选生成之后的排序/重排是当前最大的性能杠杆**；单纯改进表示或图结构，如果不进入 reranker 的特征，可能无法完全转化为指标。
- TIGER 明显弱于 SASRec，说明在这个数据上“text-only semantic ID + from-scratch generative retrieval”并不足够；joint multimodal 和 collaborative/sequential 信号仍然重要。但 collab 图的质量问题意味着，下一步的收益更可能来自“去噪/去偏的 collab 图 + 语义门控”，而不是继续加深 CSGA 注意力。

---

## 7. 对方法设计的启示

1. **先修图，再学权重。** 当前 collab top-K 图的 leaf purity 只有 4.9%，tail 只有 2.9%；在这种图上学习 attention，权重再准也只能在噪声邻居中选。建议：
   - popularity debiasing（如 cosine 减去流行度先验、或使用 NPMI/transition 作为边权）；
   - 对称化/双向归一化（当前 \(a_{ij}\) 与 \(a_{ji}\) 负相关，说明有向 softmax 与对称图假设不匹配）；
   - category-aware pruning（至少去掉跨 L2 的边，或对跨类边降权）。
2. **CSGA 需要“锐化”或“监督”。** 当前 entropy 0.978、有效邻居 9.52/10，接近 mean pooling。如果希望 CSGA 真正选择邻居，需要：
   - 温度/entropy 正则；
   - 用 next-item 或 co-occurrence 作为辅助监督；
   - 或在 OMoE 中显式比较“sharp CSGA”与“uniform CSGA”的消融，验证 attention 是否真的有用。
3. **tail 商品需要语义回退。** collab 在 tail 上的类别纯度只有 2.9%，而 joint 仍有 39.4%。对 tail，可以提高语义邻居/语义 SID 的权重，或使用“collab 仅在高置信度时启用”的门控。
4. **评估要分 head/mid/tail。** 整体平均会掩盖 collab 在 tail 上的退化；建议在论文中报告分组邻域质量，作为方法动机或诊断。
5. **NPMI/transition 比 raw co-occurrence 更适合做图诊断。** raw P(j|i) 受 popularity 影响，collab 的 1.60% 看似最高，但 NPMI 最低；用 NPMI/transition lift 才能看出 collab 图并不“更强”。
6. **Reranker 是最直接的收益点。** 如果目标是指标，优先把多源邻域特征（CF/text/multimodal/popularity/transition）送入 reranker；如果目标是论文叙事，则需要解释为什么表示层改进没有完全被生成模型利用。

---

## 8. 限制与复现

### 8.1 限制

- **单数据集、单次划分。** 结论只在 Sports strict 10-core 上验证；Books/Toys/Industrial 等数据可能不同。
- **CSGA 权重来自缓存重跑。** 训练脚本没有保存 scorer checkpoint；我们从 `csga_aligned.npy` 反推有效权重。重构残差 1.76e-5 说明反推是精确的，但该缓存不是生产 `csga_moe_collab_v2` 的原始 checkpoint；不同随机种子可能带来数值差异。权重统计应理解为“该 CSGA 过程的一次实现”，而不是唯一真值。
- **零 collab 嵌入商品被排除。** 599 个商品从未在训练集出现，SASRec 嵌入全零；它们的 collab top-K 是退化的。主分析排除后 N=10,647；全量结果在 CSV 中，关键结论不变。
- **共现/转移非常稀疏。** K=10 时 collab 只有 18.3% 邻居对共现、3.3% 有直接转移；绝对概率的方差较大。我们因此同时报告 coverage、pop-matched lift 和 hub-preserving null。
- **NPMI 是 observed-pair 均值。** 未共现对不参与 NPMI 均值；CSV 中另有 +0.5 平滑版本 `npmi_smooth`。主表 NPMI 应结合 coverage 阅读。
- **类别粒度。** leaf category 来自 metadata 的最后一个层级，类别数 294，部分类别很细；L2 是第三级（63 类）。不同粒度会改变 purity 的绝对值，但不改变空间排序。
- **Head/mid/tail 是等商品数三分。** 不是等交互量三分；head 占 65.9% 交互，tail 占 11.3%。这是为了每个分组有足够商品数，便于统计。
- **Case study 是分层随机，不是均匀随机。** 候选池有图像完整、分歧明显等质量约束；seed=2026。这样适合论文展示，但不代表总体分布。
- **性能表来自不同训练预算的已有记录。** SASRec/TIGER/CSGA+OMoE/reranker 不是严格同预算消融；相对比较只作趋势参考。

### 8.2 复现

```bash
cd /data02/sht/MIMIGenRec
PY=/data/sht/conda/mimigenrec/bin/python

# 1. 四空间 top-K 邻域 + 指标 + random controls
$PY analysis/samgr_sports10/scripts/exp_e_topk_neighborhoods.py

# 2. 有效 collab 子集（排除 599 个零嵌入商品）
$PY analysis/samgr_sports10/scripts/exp_e_valid_subset.py

# 3. CSGA 权重反推 + 全量统计
$PY analysis/samgr_sports10/scripts/exp_f_csga_weights.py

# 4. 有效 collab 子集上的 CSGA 统计
$PY analysis/samgr_sports10/scripts/exp_f_csga_valid_subset.py

# 5. 分层随机 case study + 图
$PY analysis/samgr_sports10/scripts/select_cases.py
$PY analysis/samgr_sports10/scripts/make_case_study_figure.py

# 6. 汇总图
$PY analysis/samgr_sports10/scripts/make_neighborhood_figures.py
```

主要产物：

- 表：`results/exp_e_pair_overlap_validcollab.csv`、`results/exp_e_neighborhood_metrics_validcollab.csv`、`results/exp_f_csga_metrics_validcollab.json`、`results/exp_f_cases.csv`
- 逐商品：`results/exp_e_per_item.npz`、`results/exp_f_csga_per_item_validcollab.csv`、`results/exp_f_csga_edges.npz`
- 图：`figures/fig_e_overlap_heatmap.png`、`fig_e_neighborhood_quality.png`、`fig_e_head_mid_tail.png`、`fig_f_csga_weights.png`、`fig_case_study_part1/2.png`、`fig_case_study_combined.png`
- 全量敏感性：`results/exp_e_neighborhood_metrics.csv`、`results/exp_e_pair_overlap.csv`、`results/exp_f_csga_metrics.json`

---

## 9. 一句话总结

在 Sports 10-core 上，**collaborative 与 joint-multimodal 的 Top-K 邻域几乎是两个世界**：collab 图只有很弱的类别/共现/转移信号，且在 tail 上迅速退化；CSGA 学到的权重接近均匀、与 cosine 弱相关、方向不对称，只带来约 1.12× 的转移 lift。现有 CSGA+OMoE 的性能提升更可能来自多模态语义表示与 reranking，而不是一个强 collaborative 注意力图。下一步应优先“清洗/去偏 collab 图 + 语义门控 + 更强 reranker”，并对 head/mid/tail 分别评估。
