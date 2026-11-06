# SAM-GR 动机预实验报告（Sports strict 10‑core）

> 目标：用**现有数据与中间产物**检验 SAM‑GR 的三个动机假设，并明确哪些结论有证据、哪些没有。
> 不做方法核心改动；所有实验基于当前流水线与已保存产物。

## 0. 数据与表示（已核对）

- 数据集：Sports_and_Outdoors，**严格唯一计度 10‑core**：11,246 items；train 202,980 / valid 25,373 / test 25,373。
- 协同表示：SASRec item embedding（11,246×128）。
- 语义表示：Qwen3‑VL 文本（4096）、视觉（4096）、多模态（4096）。
- 主模型 SID：CSGA+OMoE v2 融合特征 → RQ‑VAE 4×256 → merge 15%。
- 主模型测试指标（2 epoch）：HR@1 0.0039，HR@50 0.0551，NDCG@50 0.0184。
- 所有表示按行对齐到 item id（0..11245）；相似度用 L2 归一化后的余弦。

---

## Experiment A：协同 vs 语义邻域结构

问题：协同、文本、视觉是否诱导相似的邻域？

| 对比 | overlap@5 | @10 | @20 | @50 | Spearman |
|---|---|---|---|---|---|
| 协同 vs 文本 | 0.0100 | **0.0139** | 0.0187 | 0.0290 | **0.058** |
| 协同 vs 视觉 | 0.0092 | **0.0116** | 0.0159 | 0.0243 | **0.052** |
| 文本 vs 视觉 | 0.1475 | 0.1550 | 0.1644 | 0.1783 | 0.190 |

**结论**
- **协同邻域与文本/视觉邻域几乎完全不相交**（K=10 重叠仅 1.2–1.4%，秩相关 ~0.05）。这直接支持 CSGA 的动机：协同邻域提供语义邻域之外的信息。
- 文本与视觉之间也只有 15–18% 重叠（秩相关 0.19），说明不同模态本身也不完全一致；但协同与语义的差异远大于模态间差异。
- 注意：这是**嵌入级**证据；它不能直接推出"最终 SID 分组不同"（见 Experiment B）。

图：`figures/fig_a_tsne_{collab,text,visual,joint}.png`（定性）、`fig_a_overlap_bar.png`、`fig_a_overlap_dist.png`。
表：`results/exp_a_neighborhood_overlap.csv`、`exp_a_rank_correlation.csv`。

---

## Experiment B：文本 SID vs 多模态+协同 SID

问题：不同模态构造的 SID 空间是否不同？

| 指标 | 值 |
|---|---|
| 同 item 同 SID | **0 / 11,246** |
| 逐级 token 一致率 (a/b/c/d) | 0.20% / 0.45% / 0.25% / 0.37%（随机 1/256=0.39%） |
| SID 集合 Jaccard | 0.0 |
| **最终 item 分组（SID）** | **9,655 / 9,663 组相同** |
| **ARI @ 最终 SID** | **0.9986**（NMI 1.0000） |
| ARI @ a / a+b / a+b+c | 0.126 / 0.136 / 0.606 |

**结论（重要，修正直觉）**
- **SID token 完全不重合是"码本标签置换"造成的假象**：两个 RQ‑VAE 的最终 item 分组几乎相同（ARI≈0.999），只是码字不同。
- 真正不同的是**层级前缀结构**（粗粒度 ARI 仅 0.13）；即"层次分解/生成顺序"不同，但最终物品集合相同。
- 因此**不能用"SID 重合=0"来论证不同模态组织不同物品**；该证据不成立。论文中若声称 CSGA/OMoE 改变了物品组织，目前**没有 SID 分区层面的支持**。

任务性能（参考，非严格消融）：
| 模型 | HR@1 | HR@50 | NDCG@50 |
|---|---|---|---|
| Full（CSGA+OMoE，text+image+collab，LLM） | 0.0039 | 0.0551 | 0.0184 |
| TIGER（text‑only SID，from‑scratch seq2seq） | 0.0017 | 0.0199 | 0.0074 |

**限制**：TIGER 是不同 backbone（无 T5 权重，从头训练），该对比**不能**作为"去掉某模态"的严格因果证据；重训 LLM 的模态消融未运行。

图：`fig_b_performance.png`、`fig_b_sid_overlap.png`（已改为 ARI/NMI + 随机线）。

---

## Experiment C：CSGA 邻域聚合设计（表示级）

变体（同一多模态语义特征 `m`，同样 `h = m + 0.5·mean(neighbors)`）：

| 变体 | cos to m | overlap with collab nb | overlap with semantic nb | k‑means inertia |
|---|---|---|---|---|
| 无聚合 | 1.000 | 0.0123 | 1.000 | 0.1084 |
| **协同邻居（CSGA）** | 0.984 | **0.0201** | 0.711 | **0.0636** |
| 语义邻居 | 0.996 | 0.0120 | 0.869 | 0.0707 |
| 随机邻居 | 0.982 | 0.0120 | 0.753 | 0.0624 |

**结论**
- CSGA 聚合使表示与原语义保持很高余弦（0.984），同时**对协同图的邻域对齐提升 64%**（0.0123→0.0201），并得到最紧凑的 k‑means 表示（inertia 最低）。
- **但随机邻居聚合也显著降低了 inertia（0.0624）**：平均平滑本身就会压缩表示，因此"更紧凑"不能单独作为 CSGA 有效的证据；**区分性证据是协同邻域对齐的提升**。
- 绝对提升不大（2.0% vs 1.2%），部分因为本实验用**均匀权重**而非 CSGA 的学习权重；这是下界。
- **下游重训未做**（成本），因此不能据此声称推荐指标提升。

图：`fig_c_csga_variants.png`；表：`exp_c_csga_variants.csv`。

---

## Experiment D：OMoE 专家互补性与路由

三组变体（同一 CSGA 对齐特征，训练 60 epoch；expert sim 为 5 个专家两两余弦的平均绝对值）：

| 变体 | gates (text/image/csga/shared1/shared2) | gate entropy | item entropy | expert mean\|cos\| | fused cos | nn_dup |
|---|---|---|---|---|---|---|
| Full（ortho .05 / load .1） | 0.197 / 0.200 / 0.201 / 0.198 / 0.204 | 1.609（=ln5 最大） | 1.416 | **0.0078** | 0.129 | 0.000 |
| 无正交 | 0.223 / 0.192 / 0.193 / 0.202 / 0.190 | 1.608 | 1.130 | **0.3381**（max 0.826） | 0.143 | 0.003 |
| 无负载均衡 | 0.000 / 0.000 / **1.000** / 0.000 / 0.000 | **0.000** | 0.000 | 0.0055 | 0.178 | 0.006 |

**结论**
1. **正交正则效果显著**：去掉正交后，专家输出平均 |cos| 从 **0.0078 升到 0.3381**（最大 0.826）；加回正交后专家几乎两两正交。这**支持** OMoE 的正交设计——但应表述为"专家输出近似正交"，而非"信息完全解耦"。
2. **负载均衡是必需的**：去掉后路由**坍缩到单一专家**（gate entropy 0）。此时融合表示质量取决于该专家，且**对随机种子敏感**：本次 fused cos 0.178（尚可），前一次复跑曾出现 **cos 0.997 / nn_dup 1.0** 的退化。→ 无负载均衡存在表示退化风险。
3. **路由的实例依赖性弱**：full 的平均 gate 熵 = 最大值 ln5（1.609），说明**平均路由均匀**（负载均衡的副作用）；逐 item 熵 1.416 显示一定输入依赖，但整体接近等权集成。因此 **"instance-dependent routing" 不宜作为强主张**。
4. OMoE 消融仅到表示级；下游 LLM 重训未跑。

图：`fig_d_expert_similarity.png`、`fig_d_routing.png`；表：`exp_d_omoe_variants.csv`、`exp_d_omoe_stats.json`。

---

## 论文 Preliminary Figure（候选素材，主图见下一节）

文件：`figures/fig_preliminary_v1.png`（2×3 详细版）、`fig_preliminary_v2.png`（1×3 紧凑版）、`fig_preliminary_cka.png`（CKA 热图）、`fig_preliminary_case.png`（case study）；数值见 `results/preliminary_metrics.csv`。

### Panel (a) 语义–协同错位（Challenge 1）——统计检验（非个案）
文件：`fig_preliminary_statistical.png` / `fig_preliminary_statistical_v2.png`；数值：`results/exp_a_statistical_tests.csv`。

**Mantel 检验**（两个相似度矩阵的相关；零分布=置换 item 标签，500 次）：

| 对比 | Mantel r | 零分布均值 | p | 解释 |
|---|---|---|---|---|
| **collab–text** | **0.051** | ~0 | 0.002 | 极弱 |
| **collab–image** | **0.039** | ~0 | 0.002 | 极弱 |
| **collab–multimodal** | **0.029** | ~0 | 0.002 | 极弱 |
| text–image（弱相关对照） | 0.178 | ~0 | 0.002 | 弱 |
| **text–multimodal（正对照）** | **0.577** | ~0 | 0.002 | 强 |

**Top‑10 邻域重叠**（观测 vs 置换零分布）：

| 对比 | 观测 | 零分布 | 超出 | z | p |
|---|---|---|---|---|---|
| collab–text | **1.22%** | 0.092% | +1.13pp | 41.8 | 0.002 |
| collab–image | **1.13%** | 0.090% | +1.04pp | 37.1 | 0.002 |
| collab–multimodal | **1.26%** | 0.092% | +1.17pp | 41.0 | 0.002 |
| text–image | 15.75% | 0.104% | +15.6pp | 378.9 | 0.002 |
| **text–multimodal（正对照）** | **36.37%** | 0.137% | +36.2pp | 634.4 | 0.002 |

**结论（统计层面）**
- 协同与语义结构**显著地**高于随机（p=0.002，置换检验），但**效应量极小**：重叠仅 **1.1–1.3%**，Mantel r 仅 **0.03–0.05**（可解释方差 r²≈0.001–0.003）。
- 作为对比，真正相关的模态对（text–multimodal）重叠达 **36.4%**、Mantel r=**0.577**（r²≈0.33）；协同–语义的重叠只有它的 **~1/30**。
- 因此应表述为：**协同与语义诱导的邻域结构弱相关/基本不同**，而不是"完全独立"。这比单个 case study 更有说服力（见 Panel a 的散点：collab–text 为无结构点云，text–multimodal 呈明显正相关）。
- **不要只报 z/p**（因零分布方差极小，z 会很大）；论文应报 **Mantel r、重叠率、相对正对照的比例** 这类效应量。
- **支持**：协同与语义是两种几乎独立的结构，简单拼接/全局对齐难以利用其互补性；为 CSGA"协同检索 + 语义聚合"提供依据。
- **不能说明**：两者完全无关（仍有微弱相关），也不能直接说明 CSGA 一定带来下游提升。

### Panel (b) 模态主导与利用不足（Challenge 2）
- 文本↔图像 CKA = **0.020**（几乎独立 → **互补而非冗余**）。
- 文本↔多模态 CKA = **0.714**，图像↔多模态 = **0.098** → "多模态"嵌入由文本主导。
- 融合表示↔文本/图像/多模态 CKA = **0.703 / 0.071 / 0.942** → 融合几乎等同于文本主导的多模态嵌入，**图像贡献很小**。
- Full vs text‑only 的最终 SID 分组 **ARI=0.999** → 图像/协同没有改变最终分组。
- **支持**：**模态主导 + 弱模态被忽略/利用不足**；现有融合没有把互补的视觉信息带入 SID。
- **不支持**：论文中"跨模态信息冗余普遍存在"的表述（文本–图像 CKA 仅 0.02）。建议改为 **"dominance / underutilization"**，或改用属性级/任务级冗余度量。
- **双刃**：ARI 0.999 同时也说明当前 CSGA/OMoE 没有改变 SID 分组；方法层面需加强（协同感知量化/解码重排）或重新定位贡献。

### Panel (c) OMoE 的必要性
- 去掉正交：专家 mean|cos| **0.338**（max 0.826）；full：**0.008** → 正交正则有效。
- 去掉负载均衡：gate entropy ≈ **0**（路由坍缩到单专家）；full：**1.609**（=ln5）。
- **支持**：无约束专家会高度相关；无负载均衡会导致路由坍缩。

### Panel (d) 分布几何与可对齐性（对应"分布结构差异过大、不易对齐"）
文件：`figures/fig_preliminary_geometry.png`（1×3：有效维度 / CCA 曲线 / 可对齐性柱状）、`fig_preliminary_geometry_v2.png`（1×2 紧凑版）；数值：`results/exp_a_geometry_dims.csv`、`results/exp_a_geometry_alignment.csv`、`results/exp_a_geometry.json`。子采样 3000 个 item，PCA 到 50 维后计算（CCA 在原始 4096 维上过慢；MLE 内蕴维因 collab 近重复点不稳定，已弃用）。

**点云几何（distribution / geometry）**

| 空间 | 有效维度 PR | d90 | 最近邻距离 d1 | 邻居计数偏度（hubness） |
|---|---|---|---|---|
| collab (128d) | **83.3** | 92 | **0.942** | **16.49** |
| text (4096d) | 15.7 | 257 | 0.234 | 1.83 |
| image (4096d) | 32.3 | 375 | 0.365 | 2.01 |
| multimodal (4096d) | **6.0** | 193 | 0.352 | 1.65 |

- 协同空间**有效秩高**（PR≈83/128），语义空间**低秩集中**（PR 6–32）→ 两者占据的"形状"不同。
- 协同空间**hubness 极端**（偏度 16.5 vs 1.7–2.0）：少数 item 垄断邻居、大量 item 近乎孤立；语义空间局部密度均匀得多。
- 这两点是**分布/几何层面**的差异，而不是"邻居是否重叠"的关系层面差异。

**线性可对齐性（canonical / Procrustes）**

| 对比 | Procrustes R² | 平均 CCA | 最强 CCA |
|---|---|---|---|
| **collab–text** | **−5.37** | **0.479** | 0.773 |
| **collab–image** | **−2.72** | **0.433** | 0.723 |
| text–image（弱对照） | 0.153 | 0.873 | 0.908 |
| **text–multimodal（正对照）** | **0.582** | **0.950** | 0.991 |

- Procrustes R²<0 表示**最优正交映射仍不如"预测均值"**——不存在把协同空间整体旋转/翻转到语义空间的全局线性等距映射。
- 平均 CCA 仅 0.43–0.48（正对照 0.95）：只有**少数方向**可对齐（最强 0.72–0.77），整体是**低维弱相关**。
- 结论：可写"**协同与语义表示在有效维度、局部密度（hubness）上差异显著，且缺乏全局线性对齐，仅存在少数弱相关方向**"。这比"邻域不重叠"更贴近"分布结构不同、不易对齐"的原句，同时避免"完全无法对齐"的过强表述。

**使用建议**：几何图 `fig_preliminary_geometry_v2.png` 作为 Challenge 1 的机制补充。正文主图改用下面专门为两个挑战生成的三联图（定稿版），不用单独 per‑modality t‑SNE 做定量结论。

---

## 论文主图：两个挑战的统计证据（定稿版）

脚本：`scripts/make_challenge_figures.py`（只读取已保存的 CSV/JSON，并重算 PCA 累积方差曲线；300 dpi）。
文件：

- `figures/fig_challenge1_misalignment.png` —— 语义–协同错位三联图（旧编号中的 Challenge 1）
- `figures/fig_challenge2_modality.png` —— 模态主导三联图（旧编号中的 Challenge 2）
- `figures/fig_two_challenges.png` —— 旧版 2×3 合并（内部评审用，编号与上文一致）
- `figures/fig_two_challenges_combined.png` / `.pdf` —— **精简美化合并版（只合并上述两张图；按论文最新编号：Challenge 1 = modality，Challenge 2 = collaborative）**
- `figures/fig_two_challenges_combined_caption.md` —— 合并版中英文图注与读图说明

### Challenge 1（a–c）：语义–协同错位

| Panel | 内容 | 关键数字 |
|---|---|---|
| (a) | Top‑10 邻域重叠：观测 vs 置换零分布（log 纵轴），柱上标 Mantel r | collab–text **1.2%**（r=0.05）、collab–image **1.1%**（r=0.04）、collab–multimodal **1.3%**（r=0.03）；对照 text–image 15.8%（r=0.18）、text–multimodal 36.4%（r=0.58） |
| (b) | PCA 累积解释方差（横轴 PC 数 log 尺度） | collab PR=83/d90=92，text 16/257，image 32/375，multimodal 6/193 |
| (c) | 全局线性可对齐性：Procrustes R²（蓝）与平均 CCA（橙），横轴为得分 | collab–text **R²=−5.37**, CCA 0.48；collab–image **−2.72**, 0.43；text–image 0.15/0.87；text–multimodal 0.58/0.95 |

**可直接写入论文的结论**：协同与语义表示 (i) 有效维度差 1 个数量级以上（83 vs 6–32），(ii) 局部 hubness 极端（16.5 vs 1.7–2.0），(iii) 不存在全局线性等距映射（Procrustes R²<0，平均 CCA 仅 0.43–0.48，正对照 0.95），(iv) 邻域重叠仅 1.1–1.3%，且显著高于置换零分布（p=0.002），但效应量只有正对照的 ~1/30。→ **"分布/几何结构不同、全局难对齐、仅有少数弱相关方向"**，而不是"完全无关"。

### Challenge 2（d–f）：模态主导与视觉利用不足

| Panel | 内容 | 关键数字 |
|---|---|---|
| (d) | 五个空间的线性 CKA 热图；绿框标出 text–image | text–image **0.020**（互补，不冗余）；text–multimodal 0.714；collab 与所有语义空间 0.016–0.037 |
| (e) | 融合表示对各模态的 CKA | fused–text **0.703**、fused–image **0.071**、fused–multimodal **0.942** |
| (f) | full vs text‑only 的 SID 分区一致性（逐级前缀 + 最终 SID） | 前缀 a/a+b/a+b+c ARI = 0.13/0.14/0.61；**最终 SID ARI=0.999** |

**可直接写入论文的结论**：融合空间由文本主导（fused–text 0.70、fused–multimodal 0.94），视觉贡献低一个数量级（0.07）；且加入图像+协同后**最终 item 分区几乎不变**（ARI 0.999）→ **模态主导 + 弱模态利用不足**。

**必须修正的表述**：text–image CKA 仅 0.02，**不支持"跨模态冗余普遍存在"**。建议把 Challenge 2 改写为 "modality dominance and under‑utilization of complementary modalities"（文本主导、视觉互补但未被利用），而非 "redundancy"。若要保留 redundancy 一词，需要补充属性级/任务级冗余度量（当前数据没有）。

**双刃剑**：panel (f) 的 ARI=0.999 同时说明当前 CSGA/OMoE 没有改变最终 SID 分组；这既支持"视觉未被利用"的动机，也要求方法部分给出真正改变分组/解码的机制（协同一致性量化或解码重排），否则审稿人会质疑方法贡献。

---

## 对论文的指导（基于证据）

**可以支撑的**
- Exp A：**协同邻域与语义邻域几乎不相交** → 这是 CSGA 动机最强的证据，建议放正文。
- Exp D：**正交正则把专家输出相似度从 |cos|≈0.34 降到 ≈0.008**，且**无负载均衡时路由坍缩到单一专家** → 可作为 OMoE 正交/负载均衡设计的直接证据（注意 no-load 的表示退化是 seed 敏感的，出现过一次）。
- Exp C：CSGA 在保持语义的同时提高协同邻域对齐（+64%），可作为设计合理性的弱证据。

**应弱化/避免的**
- ❌ "不同模态产生不同的 SID 组织"（Experiment B：最终分组 ARI 0.999，token 差异仅来自码本置换）。
- ❌ "专家严格正交/完全解耦"（正交影响很小）。
- ❌ "instance‑dependent routing"（gate 平均熵为最大值，路由均匀）。
- ❌ 用 TIGER 对比证明模态因果（backbone 不同）。

**建议的方法/实验补充**
- 若要在方法层体现 CSGA/OMoE 的价值：给量化/融合加入**协同一致性目标**，使 SID 分组真正改变；或把协同信号用于**解码重排**（见下）。
- 若要支撑路由的实例依赖性：降低 load_weight 或改用熵正则，并报告 gate 方差/路由分布；同时监控坍缩。

---

## 与"提分小技巧"的衔接（不改方法核心）

- Experiment A 表明协同邻域与语义邻域互补；离线审计把候选从"LLM beam + CF"升级为"beam(2ep+4ep) + CF + text + multimodal + popularity"。
- 旧粗重排原型（半集调参/半集评估）只有 HR@50 0.0475，已废弃；按审计结论重做后的 clean 协议结果见下文 **"重排提分实验"**：
  - 候选覆盖 **12.24%**（test），学习式重排 HR@1 **0.0123**、HR@50 **0.0813**、NDCG@50 **0.0323**。
- 结论：**多源候选扩充 + 学习式重排**是不改生成模型核心、可解释为"collaborative-guided decoding / candidate reranking"的提分路径；prefix-2 扩展虽把 oracle 上限抬到 17.8%，但当前特征下重排无法兑现，暂不采用。
---

## 离线候选审计（基于已保存的 50-beam 结果）

脚本：`scripts/audit_candidates_v2.py`；结果：`results/audit_candidates.json`、`results/audit_merge_sweep.csv`。
对象：Sports strict 10-core，25,373 test samples；2ep 与 ep4 两份 `final_result.json`。不跑模型。

### 1. Beam 现状

| 指标 | 值 |
|---|---|
| 每样本唯一 beam 数 | 50 / 50（无重复，去重无收益） |
| 全局唯一 top-1 SID | **225** |
| top-10 预测 SID 占全部 top-1 的比例 | **72.7%** |
| 全部 beam 覆盖的唯一 SID | 2,309 / 9,663 |
| target 在 beam 内 | **5.51%** |
| target 在 top-10 / top-1 | 2.18% / 0.39% |

→ 生成坍缩严重；HR@1 的瓶颈主要是**排序**（oracle HR@1 = 5.51%，当前 0.39%），不是 beam 重复。

### 2. Prefix 覆盖率（coarse-to-fine 的空间）

| 层级 | target prefix 在 beam 内 | 唯一 prefix 数 | 平均候选 SID 数 |
|---|---|---|---|
| a | **53.36%** | 69 | 3,401 |
| a+b | **12.13%** | 3,078 | **296** |
| a+b+c | 6.25% | 8,952 | 71 |
| full SID | 5.51% | — | 50 |

→ **a+b 是甜点**：覆盖率 12.1%（是 full SID 的 2.2 倍），候选仅 ~296 个；a 覆盖 53% 但候选 3,400 个，噪声太大；a+b+c 只比 full 多 0.7pp，不值得。

### 3. 候选源覆盖率（K=20 / K=50）

| 候选集 | K=20 覆盖率 | K=50 覆盖率 | K=50 平均候选数 |
|---|---|---|---|
| LLM beam | 5.51% | 5.51% | 50 |
| + CF（最后 item 协同邻居） | 7.06% | 8.44% | 70 |
| + text 语义 | 9.94% | 13.00% | 72 |
| + multimodal 语义 | 10.93% | 14.65% | 73 |
| + popularity | **11.37%** | **15.68%** | 264 |

- 单独看：text top-50 = 5.54%，multimodal top-50 = 5.34%，CF top-50 = 3.36%，popularity top-50 = 3.03%。
- text/multimodal 与 LLM beam 几乎同等强，且互补；这是当前最被低估的候选源。
- 2ep + ep4 beam 并集：**8.10%**，平均 83 个候选（+2.6pp，几乎零成本）。

### 4. Prefix-2 扩展 + 多源

| 候选集 | 覆盖率 | 平均候选数 |
|---|---|---|
| a+b 扩展 | 12.13% | 296 |
| a+b 扩展 + beam + CF/text/mm/pop@20 | **17.31%** | 377 |

→ 这是目前找到的最高候选上限：**17.31%**，高于 MMQ HR@50 7.03% 和 CARD 6.55%。剩下的是排序问题。

### 5. merge 扫描（post-hoc，SID-level）

| 总 merge ratio | 唯一 SID | collision | 平均 target 组大小 | HR@1 | HR@50 | NDCG@50 |
|---|---|---|---|---|---|---|
| 0.15（当前） | 9,663 | 14.1% | 1.51 | 0.00394 | 0.05510 | 0.01837 |
| 0.20 | 9,035 | 19.7% | 1.79 | 0.00394 | 0.05604 | 0.01857 |
| 0.25 | 8,505 | 24.4% | 1.93 | 0.00394 | 0.05628 | 0.01862 |

→ 15%→25% 只带来 **+2.1% 相对 HR@50**，HR@1 完全不变，但 collision 从 14.1% 涨到 24.4%、平均组大小从 1.51 涨到 1.93。**不建议为重训 merge 花 5–6 小时**；item-level 会更难看。

### 6. 审计结论

1. **不要优先加 merge**：收益 <2%，代价是歧义和重训。
2. **T1 多源候选已落地**：safe recipe 覆盖 **12.24%**（test；single beam 5.51%），已高于 MMQ/CARD 的 HR@50；prefix-2 扩展虽到 17.8%，但重排不兑现，暂不用。
3. **T2 重排已落地**：valid→test clean 协议 HR@1 **0.0123**、HR@50 **0.0813**、NDCG@50 **0.0323**；oracle 上限 12.24%，排序仍是主要空间。
4. **text/multimodal 候选必须加入**：它们单独就有 5.5%/5.3% 覆盖率，且与 beam 互补。
5. **不需要 beam 去重**：50/50 唯一。
6. **评测必须同时报 SID-level 和 item-level**：merge/组碰撞会让 SID-level 虚高。

---

## 重排提分实验（Phase 1a/1b：5-fold 诊断 + valid→test 清洁验证）

脚本：`scripts/rerank_crossfit.py`（Phase 1a）、`scripts/rerank_valid_to_test.py`（Phase 1b）。
结果：`results/rerank_crossfit.json`、`results/rerank_valid_to_test.json`、`results/rerank_valid_to_test_pct.json`。
候选 recipe（safe）：`beam(2ep+4ep) + CF20 + text20 + multimodal20 + popularity20`；候选源全部来自 train split 的协同/语义邻居，不使用 test 标签。

### 参数方案

- LLM 生成：Qwen2.5-0.5B 的 2ep / ep4 checkpoint；50 beams，`max_new_tokens=256`，`do_sample=False`，`length_penalty=0.0`，trie 约束解码；valid/test 各 25,373 条。
- 候选源：2ep+4ep beam 并集；SASRec 协同 top-20；Qwen3VL text-only top-20；Qwen3VL multimodal top-20；train popularity top-20；去重后平均 **142.2** 个候选（test）。
- 特征（21）：`rank2, rank4, in2, in4, p1_count, p2_count, p3_count, p2_best_rank, p3_best_rank, cf_score, tx_score, mm_score, pop_log, group_size, src_beam, src_prefix, src_cf, src_tx, src_mm, src_pop, n_sources`；safe recipe 中 `src_prefix≡0`（保留给 aggressive 变体）。
- 重排器：`HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0, class_weight='balanced')`。
- 训练/评估：valid 上 3,414 个"目标在候选集内"样本，487,501 行候选（3,414 正 / 484,087 负）；test 3,105 个覆盖样本、441,405 行；test 只评一次；指标分母均为 25,373。
- 完整 HR/NDCG：`results/rerank_full_metrics.md`、`results/rerank_full_metrics.csv`。

### Phase 1a：test 上 5-fold cross-fit（诊断，非最终数）

| recipe | 覆盖 | 平均候选 | HR@1 | HR@10 | HR@50 | NDCG@50 |
|---|---|---|---|---|---|---|
| safe | 12.24% | 142.2 | 0.0123 | 0.0372 | **0.0819** | 0.0326 |
| aggressive（+a+b 扩展） | 17.80% | 366.8 | 0.0123 | 0.0370 | 0.0826 | 0.0327 |
| RRF（2ep+4ep beam） | 8.10% | 83 | 0.0035 | 0.0220 | 0.0608 | 0.0196 |
| LLM 2ep beam | 5.51% | 50 | 0.0039 | 0.0218 | 0.0551 | 0.0184 |

→ a+b 扩展把 oracle 上限从 12.2% 抬到 17.8%，但重排 HR@50 只从 8.19% 到 8.26%（噪声级），**当前特征下不采用 aggressive**；safe recipe 已把 HR@50 相对 LLM 提升 **+48.6%**、HR@1 提升 **3.1×**。

### Phase 1b：valid 训练、test 只评一次（clean protocol）

| 模型（valid 上） | HR@1 | HR@50 | NDCG@50 |
|---|---|---|---|
| LLM 2ep | 0.0041 | 0.0589 | 0.0196 |
| LLM 4ep | 0.0051 | 0.0663 | 0.0221 |

| 方法（test） | 候选覆盖 | HR@1 | HR@10 | HR@50 | NDCG@50 |
|---|---|---|---|---|---|
| LLM 2ep | 5.51% | 0.00394 | 0.02179 | 0.05510 | 0.01837 |
| LLM 4ep | 5.58% | 0.00347 | 0.01986 | 0.05585 | 0.01800 |
| RRF（2ep+4ep） | 8.10% | 0.00367 | 0.02195 | 0.06077 | 0.01962 |
| **Reranker（safe，21 特征）** | **12.24%** | **0.01234** | **0.03732** | **0.08135** | **0.03227** |
| Reranker + per-sample percentile（26 特征） | 12.24% | 0.01230 | 0.03756 | 0.08064 | 0.03238 |

- 重排模型：`HistGradientBoostingClassifier(max_iter=200, lr=0.06, leaves=31, class_weight='balanced')`；只在 valid 的 3,414 个"目标在候选集内"样本上训练，test 评估一次。
- 相对 LLM 2ep：HR@1 **0.00394→0.01234（3.13×）**，HR@50 **0.05510→0.08135（+47.6%）**，NDCG@50 **0.01837→0.03227（+75.6%）**。
- 相对 MMQ w/o BAF（HR@50 0.0703）高 **+15.7%**；相对 CARD（0.0655）高 **+24.2%**；但二者是 end-to-end 方法，本重排使用额外候选源和 valid 标签，**不能直接宣称全面 SOTA**，论文中应作为可选 post-hoc 组件单独报告。
- percentile 变体与 base 差异在噪声内（18 个样本量级），保留 21 特征 base 作为主结果，26 特征作为稳健性检查。
- 图：`figures/fig_rerank_gain.png`（HR@1 / HR@50 / NDCG@50，LLM vs RRF vs reranker）。

### 实现要点与限制

- 重排只改变**候选排序**，不改变生成模型/SID/CSGA/OMoE；可作为 "collaborative-guided decoding / candidate reranking" 写，但需明确它是额外模块。
- 覆盖面仍是硬上限：test 仅 **12.24%** 的 target 进入候选集；剩余 87.76% 任何重排都无法召回。因此 HR@50 不可能超过 12.24%。
- SID-level 指标；merge 组碰撞（14.1%）会让同一 SID 组内多 item 共享命中，论文应同时说明组大小/item-level 口径。
- 单数据集、单 seed；重排超参由 Phase 1a 的 test cross-fit 选定，Phase 1b 的 test 只评一次，仍有轻微选择偏差；需要第二数据集复核。
- 目前**不建议**为提分去增大 merge（+2.1% 相对 HR@50，HR@1 不变，collision 24.4%）；也不建议直接上 prefix-2 扩展（oracle 高但重排不兑现）。

---

## 限制与复现

- 单数据集（Sports strict 10‑core）、主模型单 seed；A/C 的变体未做下游重训。
- TIGER 无 T5 权重（from scratch）。
- Exp C 用均匀聚合权重（非学习权重），是下界。
- SID token 比较必须用排列不变指标（ARI/NMI）；t‑SNE 仅定性。
- 复现：`scripts/run_all.sh`（A/B/C/D）、`scripts/fix_plots.sh`（A/D 绘图修复）、`scripts/rerank_crossfit.py`（Phase 1a）、`scripts/rerank_valid_to_test.py` / `_pct.py`（Phase 1b）、`scripts/make_rerank_figure.py`（图）；valid beam 由 `run_valid_eval_2ep.sh` / `run_valid_eval_ep4.sh` 生成；结果在 `results/`，图在 `figures/`。

---

## 新增：Top-K 邻域与 CSGA learned weights 分析

完整报告见 **[report_topk_neighborhoods.md](report_topk_neighborhoods.md)**；对应脚本 `scripts/exp_e_topk_neighborhoods.py`、`exp_e_valid_subset.py`、`exp_f_csga_weights.py`、`exp_f_csga_valid_subset.py`，图在 `figures/fig_e_*`、`fig_f_csga_weights.png`、`fig_case_study_*`。

核心结论（Sports strict 10-core，K=10/20/50）：

- collab 与 text/image/joint 的 Top-K 几乎不重合：K=10 时 collab–joint Jaccard **0.0069**、neighbor recall **1.29%**；text–joint 为 **0.247 / 36.0%**。collab 与语义空间的重叠显著高于 hub-preserving null（约 14×），但仅为真正语义邻域的约 1/25。
- joint-multimodal 明显由文本主导：text–joint recall 36.0% > image–joint 29.2% > text–image 15.1%。
- collab 邻域类别一致性弱：K=10 leaf purity **4.9%**（text/image/joint 34.4%/44.0%/39.8%），L2 purity 45.2% vs 70.3%/67.4%/70.7%；tail 上 collab leaf purity 仅 **2.9%**。
- collab 的用户共现/序列转移相对 random 为 **4.50×/11.91×**，但 text 也有 4.33×/11.07×、joint 3.97×/9.89×；collab 的优势远小于预期，且 NPMI 反而最低（0.414 vs 0.429/0.424/0.423）。
- CSGA 权重：与 collab cosine 的全局 Spearman 仅 **0.111**（逐 item 0.182）；归一化 entropy **0.978**（有效邻居 9.52/10）；互惠边 mean |a_ij−a_ji| **0.0395**、corr **−0.179**；top-5 vs unweighted transition lift mean **1.12**、median **1.59**，仅 56.1% 商品 >1。
- 6 个分层随机 case study 中 collab Top-10 与 joint Top-10 全部不相交：collab 邻居跨类别，joint 邻居同 leaf category；CSGA top-4 权重均在 0.10–0.20，接近均匀。
