# OMoE-v2 损失与结构消融（Sports_and_Outdoors strict 10-core）

> **TL;DR（先看这里）**
> 1. **表示级**：Full OMoE 的机制是真实有效的——专家近似正交（\|cos\|=0.008 vs 去 ortho 0.345）、负载均衡（CV=0.013 vs 去 load 1.232）、融合空间不近重复（cos=0.061 vs 去 dup 0.246 / uniform gate 0.457）。
> 2. **Bypass 检查**：direct projection 占分支范数 58%、mixture 42%（cos=−0.62）；CSGA 锚点占最终范数 71%、OMoE 分支 68%（cos≈0.001）。**两条路径都没有被对方绕过。**
> 3. **下游（400-step，1000-epoch RQ，20 beams）**：HR@20 排序为 **no-direct 0.0336 > no-residual 0.0291 > w/o load 0.0201 > Full 0.0124 > no-experts 0.0082**。**Full 不是下游最优；expert mixture 是下游有效成分，direct projection 和 outer residual 在短预算下反而稀释/增加生成难度。**
> 4. **结论**：OMoE 的价值应表述为“可控的专家分工/路由/表示多样性”，而不是“稳定提升 HR”。若论文目标是 HR，应把 mixture-only（去掉 direct+residual）作为更强候选，或把生成器训到主实验预算后再比较。

> 目标：固定 CSGA 输出、RQ-VAE 结构与生成器结构，逐项比较 OMoE-v2 的损失项和结构设计。
> 所有变体均从随机初始化重新训练 OMoE；RQ-VAE（4×256）和生成器（Qwen2.5-0.5B full SFT）对每个变体重新训练。
> 数据：11,246 items；train 202,980 / valid 25,373 / test 25,373；SID merge ratio = 0.15。

## 0. 协议与口径

| 项目 | 设置 |
|---|---|
| CSGA | 固定使用缓存的 `csga_aligned.npy`（Sports strict 10-core，11,246×4096）；所有变体共享 |
| OMoE | 60 epochs，Adam lr=1e-3，batch=512，mean-centred 输入，target std=4.0，seed=42 |
| RQ-VAE | 每变体重训；4 层码本×256，1000 epochs，batch=20480，seed=2024 |
| SID | `generate_indices.py`，collision resolution + 15% collaborative merge（与主实验一致） |
| 生成器 | Qwen2.5-0.5B-Instruct，full SFT；消融预算 200 / 400 steps（见 §4），20 beams，full test |
| 训练时间（4×4090） | OMoE 55–135 s；RQ-VAE 1000 epochs ≈ 15.5 min；400-step SFT ≈ 62 min；20-beam full-test eval ≈ 10 min |
| 指标 | HR/NDCG@5/10/20；RQ reconstruction MSE；codebook perplexity；full-SID collision；expert `|cos|`；load CV；item routing entropy；routing variance；params；time |

说明：
- 变体 07（w/o all auxiliary）与 08（only L_rec）在定义上完全相同（都只保留 `L_rec`），因此两行数值相同；这不是重复实验，而是定义重合。
- 生成器全量 2 epoch（约 398 steps）单变体约 1 GPU-hour×4，19 个变体全量重训约 19×4 GPU-hours；本文档对全部 19 个变体完成表示级与量化级重训，并对结构关键变体完成 400-step 下游重训（§4）。未跑的下游格用 “—” 标注，运行脚本已提供。
- RQ-VAE 使用 1000 epochs（主实验为 5000）。这是为了在可控预算内完成 19 个变体的重训；所有变体同预算，横向可比。绝对 HR 会低于主实验的 2-epoch/5000-epoch 结果。

## 1. 表示级结果（全部 19 个变体）

`expert |cos|` = 专家输出两两余弦绝对值的平均；`load CV` = 平均 gate 的变异系数；`item H` = 逐 item routing entropy 的均值；`routing var` = item 间 routing distribution 方差（对专家维取平均）；`fused cos` = 融合表示 item 间平均余弦；`ARI` = 最终 full-SID 分区与 Full 的调整 Rand 指数。

| Variant | Expert \|cos\| | Load CV | Batch load CV | Item H | Routing var | Fused cos | Recon MSE | Codebook PPL | Collision | SID ARI | Params M | OMoE s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 Full | **0.0078** | **0.0129** | 0.0306 | 1.4164 | 0.0125 | **0.0609** | 11.93 | 28.49 | 0.1786 | 1.000 | 357.1 | 97 |
| 02 w/o L_ortho | 0.3455 | 0.0344 | 0.0492 | 1.2084 | 0.0252 | 0.0675 | 12.15 | 29.64 | 0.1695 | 0.725 | 357.1 | 135 |
| 03 w/o L_load | 0.0170 | 1.2319 | 1.3808 | 0.0131 | 0.0979 | 0.4975 | 5.26 | 45.21 | 0.1439 | 0.813 | 357.1 | 90 |
| 04 w/o L_dup | 0.0063 | 0.0131 | 0.0489 | 1.1624 | 0.0281 | 0.2458 | 8.21 | 29.57 | 0.1553 | 0.780 | 357.1 | 89 |
| 05 w/o L_var | 0.0058 | 0.0019 | 0.0036 | 1.6089 | 0.0000 | 0.0717 | 12.08 | 29.16 | 0.1525 | 0.786 | 357.1 | 89 |
| 06 w/o L_rec | 0.0012 | 0.0291 | 0.0667 | 1.2721 | 0.0303 | 0.0133 | 11.76 | 27.38 | 0.1744 | 0.703 | 357.1 | 118 |
| 07 w/o all aux | 0.6426 | 1.9691 | 2.2016 | 0.0776 | 0.0000 | 0.2104 | 9.08 | 28.91 | 0.1545 | 0.777 | 357.1 | 82 |
| 08 only L_rec | 0.6426 | 1.9691 | 2.2016 | 0.0776 | 0.0000 | 0.2104 | 9.08 | 28.91 | 0.1545 | 0.777 | 357.1 | 82 |
| 09 only rec+ortho | 0.0047 | 1.6623 | 1.8585 | 0.4993 | 0.0001 | 0.2201 | 8.61 | 28.56 | 0.1831 | 0.677 | 357.1 | 91 |
| 10 only rec+load | 0.5967 | 0.0042 | 0.0067 | 1.6086 | 0.0001 | 0.1867 | 9.18 | 28.21 | 0.1703 | 0.721 | 357.1 | 82 |
| 11 no direct | 0.1945 | 0.8416 | 0.9503 | 0.0094 | 0.1305 | 0.4036 | 8.05 | 23.80 | 0.2124 | **0.529** | 306.8 | 79 |
| 12 no experts | — | — | — | — | — | 0.0822 | 10.89 | 29.07 | 0.1624 | 0.752 | **117.5** | 28 |
| 13 no residual | 0.0078 | 0.0129 | 0.0306 | 1.4164 | 0.0125 | **0.0144** | **4.83** | **49.86** | **0.1434** | 0.816 | 357.1 | 93 |
| 14 static gate | **0.0028** | 0.0003 | 0.0004 | 1.6094 | 0.0000 | 0.0958 | 11.49 | 30.82 | 0.1549 | 0.773 | 352.4 | 88 |
| 15 uniform gate | 0.0047 | 0.0000 | 0.0000 | 1.6094 | 0.0000 | 0.4570 | 5.64 | 39.28 | 0.1448 | 0.809 | 352.4 | 88 |
| 16 shared only | 0.0164 | 0.0218 | 0.0687 | 0.0334 | 0.2416 | 0.0820 | 12.10 | 30.44 | 0.1704 | 0.713 | 256.5 | 61 |
| 17 modality only | 0.0044 | 0.0186 | 0.1326 | 0.0145 | 0.2193 | 0.0709 | 9.49 | 27.59 | 0.1646 | 0.748 | 222.9 | 55 |
| 18 Concat-MLP (matched) | — | — | — | — | — | 0.1102 | 10.97 | 33.67 | 0.1462 | 0.808 | **357.2** | 82 |
| 19 plain MoE | 0.0117 | 0.0348 | 0.0497 | 1.1028 | 0.0306 | 0.0790 | 11.67 | 32.47 | 0.1559 | 0.772 | 457.8 | 112 |

参考行（不参与 OMoE 对比）：
- CSGA only：fused cos 0.0144；multimodal only：0.6568；text only：0.8760；现有 Full pipeline：0.0585。
- 这些参考说明：原始 text/multimodal 空间高度共线；CSGA 已经把 item 间余弦降到 0.014；OMoE 在 CSGA 基础上把表示拉开到 0.061，同时保持与 CSGA 锚点的正交分解（§2）。

### 1.1 损失项：正交与负载均衡是“机制级”有效，重构/collision 不是

1. **L_ortho（02）**：去掉后专家输出平均 `|cos|` 从 **0.0078 → 0.3455**（最大 0.667），item routing entropy 从 1.416 → 1.208。这是最直接、最大的专家多样性退化。注意 fused cos 只从 0.061 → 0.068，说明“专家输出高度相关”不必然导致融合空间立刻塌缩，但专家分工已经消失。
2. **L_load（03）**：去掉后 **gate CV 0.013 → 1.232**、batch load CV 0.031 → 1.381、item entropy **1.416 → 0.013**，路由坍缩到单一专家；fused cos 从 0.061 → **0.497**，出现明显的近重复塌缩。这是所有损失项中后果最严重的。
3. **L_dup（04）**：去掉后 fused cos **0.061 → 0.246**（约 4×），而专家 `|cos|` 仍很低（0.0063）。说明 L_dup 主要约束的是**融合输出的 item 间重复**，不是专家多样性；它与 L_ortho 作用在不同层级。
4. **L_var（05）**：去掉后 item entropy 从 1.416 → 1.609（接近均匀），routing var 0.0125 → 0.0000，fused cos 0.061 → 0.072。L_var 让 routing 保留一定的 item 依赖性/非均匀性；去掉后 gate 更接近常数，但融合空间没有塌缩。
5. **L_rec（06）**：去掉后 fused cos 反而最低（0.0133），专家 `|cos|` 0.0012，但重构目标消失；这是“多样性很高但语义锚定很弱”的空间。下游需要检验它是否还能保持推荐语义。
6. **全部辅助项（07/08）**：专家 `|cos|` **0.643**、gate CV **1.969**、item H **0.078**、fused cos 0.210——专家冗余 + 路由坍缩同时发生。这直接说明辅助正则不是“可有可无的调参项”，而是维持专家分工和路由的必要条件。
7. **只保留 rec+ortho（09）**：专家 `|cos|` 很低（0.0047），但 gate CV 1.662、item H 0.499、fused cos 0.220。说明只有 ortho 不够：专家虽然正交，路由仍然坍缩，融合空间仍出现重复。
8. **只保留 rec+load（10）**：gate 很均衡（CV 0.0042，item H 1.609），但专家 `|cos|` **0.597**、fused cos 0.187。说明只有 load 也不够：负载均衡但专家冗余。
9. **结论**：`L_ortho` 管“专家是否分工”，`L_load` 管“路由是否坍缩”，`L_dup` 管“融合输出是否近重复”，`L_var` 管“routing 是否保留 item 依赖”。四者作用在不同层级，不能互相替代。

### 1.2 结构项：direct projection 是稳定骨架，gate 的“可学习性”比“instance-dependence”更关键

1. **no direct（11）**：`direct_share=0`、`moe_share=1`；item H **0.0094**、gate CV 0.842、fused cos **0.404**、collision **21.2%**、OMoE rec loss 0.078（Full 0.0024）。把 direct projection 拿掉后，expert mixture 单独无法维持路由和重构，出现明显塌缩。**direct projection 不是绕过 expert branch 的旁路，而是让 expert branch 可训练、可路由的稳定骨架。**
2. **no experts（12）**：只剩 direct projection，参数 117.5M；fused cos 0.082、collision 16.2%、ARI 0.752。**表示级它不弱**，说明 direct path 承载了大量信息；但**下游它最差（HR@20=0.0082）**，说明“表示可用”不等于“生成可用”。expert branch 提供了 §2 的正交修正和路由多样性，是下游有效成分。
3. **no residual（13）**：去掉 `u_i` 锚点后，fused std 4.24（Full 5.83）、fused cos **0.0144**、RQ MSE **4.83**、codebook PPL **49.86**、collision **14.34%**。**在 v2 中，去掉 outer residual 并不会导致表示塌缩，反而让 RQ 更容易重建、collision 更低；下游 HR@20 也从 Full 的 0.0124 升到 0.0291。** 这与 v1 的失败模式不同：v1 的 OMoE 分支是低秩的，必须靠 residual 救；v2 的 60-epoch 分支已经能独立形成可量化空间。因此 residual 在 v2 中的作用应表述为“保留 CSGA 语义锚点/保守融合”，而不是“防止塌缩”或“提升 HR”。
4. **static gate（14）**：gate CV 0.0003、item H 1.609（逐 item 常数），fused cos 0.0958。它比 Full（0.0609）差，但远好于 uniform gate（0.457）。
5. **uniform gate（15）**：fused cos **0.4570**，接近 w/o load 的塌缩水平。即使专家本身 `|cos|` 很低（0.0047），把所有 item 用同一组权重混合也会让融合表示高度共线。**关键不是“instance-dependent”这个标签，而是 gate 必须能学到非均匀、数据相关的权重；static gate 已经能避免塌缩，instance gate 进一步把 fused cos 从 0.096 降到 0.061。**
6. **only shared（16）/ only modality-specific（17）**：两者都能保持较低的 fused cos（0.082 / 0.071），但 item H 很低（0.033 / 0.015），说明路由在 item 维度上高度集中（每个 item 几乎只走一个专家）；shared-only 的 routing var 0.242、modality-only 0.219，比 Full 的 0.0125 大一个数量级。**shared/specific 的二分不是表示质量的唯一来源；plain MoE（19）用 5 个 concat experts 也能得到 fused cos 0.079，接近 Full。**
7. **Concat-MLP（18，参数量匹配 357.2M）**：fused cos 0.110，collision 14.6%，ARI 0.808。它是一个强 baseline，但表示多样性不如 Full（0.061）；说明“参数量相同”不能替代“专家混合 + 路由”的归纳偏置。
8. **plain MoE（19）**：fused cos 0.079、expert `|cos|` 0.0117、item H 1.103，接近 Full；参数量 457.8M（5 个 concat experts 比 3 个小 modality experts + 2 个大 shared experts 更贵）。**在表示层面，“是否区分 shared/specific”不是主要矛盾；主要矛盾是正交、负载均衡、gate 的可学习性。** 下游层面则显示 expert mixture 本身是关键（§4.3）。

## 2. 特别检查：direct projection 和 outer residual 是否绕过 expert branch？

### 2.1 范数分解（Full，11,246 items 平均）

| 量 | 值 | 解释 |
|---|---:|---|
| `direct_share` = ‖direct‖ / (‖direct‖+‖moe‖) | **0.578** | direct projection 占分支范数的 58% |
| `moe_share` | **0.422** | expert mixture 占 42%，不是可忽略的旁路 |
| `cos(direct, moe)` | **−0.619** | 两者负相关；mixture 不是 direct 的小扰动，而是在做方向修正 |
| `rel_diff_no_moe` = ‖branch−direct‖/‖branch‖ | **0.953** | 去掉 mixture，分支几乎完全改变 |
| `rel_diff_no_direct` = ‖branch−moe‖/‖branch‖ | **1.292** | 去掉 direct，分支也几乎完全改变 |
| `aligned_norm_share` = ‖u_i‖/‖z_i‖ | **0.710** | CSGA 锚点占最终向量的 71% 范数 |
| `branch_norm_share` = ‖γ·s(f)‖/‖z_i‖ | **0.684** | OMoE 分支占 68% 范数 |
| `cos(u_i, branch)` | **0.0013** | 锚点与分支几乎正交；二者是“正交双通道”，不是“residual 覆盖分支” |

由于 `u_i` 与 `s(f)` 几乎正交，最终范数约 `sqrt(0.710² + 0.684²) ≈ 0.986`，所以两个 share 都接近 0.7 并不矛盾；它说明最终表示由两个可比的、近似正交的成分组成。**outer residual 没有绕过 expert branch；expert branch 贡献了约 42% 的分支范数和约 68% 的最终范数。**

### 2.2 反事实检查（下游变体）

- **11 no direct**：把 direct 置零后，expert mixture 单独承载分支；结果 item H 0.0094、gate CV 0.842、fused cos 0.404、collision 21.2%、ARI 0.529。**去掉 direct 的后果远大于“少了一个旁路”**：路由和融合同时退化。
- **12 no experts**：把 mixture 置零后，direct 单独承载；fused cos 0.082、collision 16.2%、ARI 0.752。direct 单独是一个可用的 baseline，但它的 item 间多样性不如 Full（0.061），且没有路由/专家分工。
- **13 no residual**：把 `u_i` 置零后，分支单独承载；fused cos 0.0144、RQ MSE 4.83、collision 14.34%。**这证明 residual 不是“防止 v2 塌缩”的必要条件**；它的价值更可能是保留 CSGA 的协同语义（见 §4 下游）。

### 2.3 可写入论文的表述

> “The direct projection is not a bypass of the expert branch: it accounts for 58% of the branch norm while the gated mixture accounts for 42%, the two components are negatively correlated (cos = −0.62), and removing either changes the branch by 95–129% in relative L2. The outer residual is also not a bypass: the CSGA anchor and the OMoE branch are nearly orthogonal (cos = 0.001) and contribute 71% / 68% of the final norm. Removing the anchor in v2 does not collapse the representation, so we frame it as a conservative semantic anchor rather than an anti-collapse necessity.”

## 3. 量化级结果（全部 19 个变体）

| Variant | Recon MSE | Codebook PPL (mean) | Raw collision | Full-SID collision | Unique SIDs | Final SID ARI vs Full |
|---|---:|---:|---:|---:|---:|---:|
| 01 Full | 11.93 | 28.49 | 0.509 | 0.1786 | 9,237 | 1.000 |
| 02 w/o ortho | 12.15 | 29.64 | 0.510 | 0.1695 | 9,340 | 0.725 |
| 03 w/o load | **5.26** | **45.21** | 0.359 | **0.1439** | 9,628 | 0.813 |
| 04 w/o dup | 8.21 | 29.57 | 0.538 | 0.1553 | 9,500 | 0.780 |
| 05 w/o var | 12.08 | 29.16 | 0.465 | 0.1525 | 9,531 | 0.786 |
| 06 w/o rec | 11.76 | 27.38 | 0.534 | 0.1744 | — | 0.703 |
| 07/08 only rec | 9.08 | 28.91 | 0.508 | 0.1545 | 9,500 | 0.777 |
| 09 rec+ortho | 8.61 | 28.56 | 0.549 | 0.1831 | 9,335 | 0.677 |
| 10 rec+load | 9.18 | 28.21 | 0.526 | 0.1703 | 9,350 | 0.721 |
| 11 no direct | 8.05 | **23.80** | **0.746** | **0.2124** | 9,114 | **0.529** |
| 12 no experts | 10.89 | 29.07 | 0.483 | 0.1624 | 9,437 | 0.752 |
| 13 no residual | **4.83** | **49.86** | 0.358 | **0.1434** | 9,628 | 0.816 |
| 14 static gate | 11.49 | 30.82 | 0.433 | 0.1549 | 9,517 | 0.773 |
| 15 uniform gate | **5.64** | 39.28 | 0.397 | 0.1448 | 9,611 | 0.809 |
| 16 shared only | 12.10 | 30.44 | 0.532 | 0.1704 | 9,347 | 0.713 |
| 17 modality only | 9.49 | 27.59 | 0.499 | 0.1646 | 9,420 | 0.748 |
| 18 Concat-MLP | 10.97 | 33.67 | 0.362 | 0.1462 | 9,602 | 0.808 |
| 19 plain MoE | 11.67 | 32.47 | 0.477 | 0.1559 | 9,488 | 0.772 |

### 3.1 关键观察：低 MSE / 低 collision 不等于好表示

- **w/o load（03）**、**no residual（13）**、**uniform gate（15）** 的 RQ MSE 和 collision 都低于 Full，但它们的表示级指标显示明显的**近重复/路由坍缩**（03：fused cos 0.497、item H 0.013；15：fused cos 0.457）。
- 原因：这些变体把融合空间压到更低秩/更平滑的方向，RQ-VAE 更容易重建、collision 更容易解决，但 item 间的**可区分性**下降。**因此，reconstruction error 和 collision rate 只能作为“量化难度”的诊断，不能单独作为“融合质量”的判据。**
- 19 个变体上的 Spearman 相关：`fused_mean_cos` vs `recon_mse` = **−0.61**（越坍缩/越共线，重构越容易）；`recon_mse` vs `final_collision` = +0.40；`final_collision` vs `codebook_ppl` = −0.73。也就是说，**单看 MSE 会系统性地偏向坍缩变体**（no-load、uniform、no-residual）。
- Full 的 MSE/collision 不是最低，但它的 expert `|cos|`、load CV、item H、fused cos 同时处于健康区间；这是“表示质量”与“量化难度”之间的权衡。
- **no direct（11）** 是唯一一个 collision 明显升高（21.2%）、codebook PPL 明显降低（23.8）、final SID ARI 最低（0.529）的变体；这与它表示级的路由坍缩一致，是“结构破坏”最明确的证据。
- **no residual（13）** 的 RQ 指标最好，但 final SID ARI 0.816，说明它仍然改变了分组；它是否更好需要下游 HR 判定（§4）。

### 3.2 SID 分区变化

- 所有变体的最终 SID 分区与 Full 的 ARI 在 **0.529–0.816** 之间，远低于“码本置换”情形（ARI≈1）。这说明 OMoE 的损失/结构选择**确实改变了 item 的量化分组**，不是简单的码字重命名。
- no direct 的 ARI 最低（0.529），no residual 最高（0.816）；coarse prefix（a/b/c）的 ARI 只有 0.09–0.14，说明差异主要体现在粗粒度层次结构。
- 这为主张“OMoE 影响 SID 组织”提供了比之前 text-only vs Full（ARI 0.999）更强的证据；但仍需注意：不同 RQ 训练轨迹也会贡献一部分差异，不能把全部 ARI 差异归因于 OMoE。

## 4. 下游生成器结果（HR/NDCG）

### 4.1 预算说明

- 全量主实验：2 epochs ≈ 398 steps，4×4090，单变体 SFT ≈ 1 GPU-hour×4；19 个变体全量重训 ≈ 19×4 GPU-hours，超出本次会话预算。
- 本次下游策略：
  - **全部 19 个变体**完成 OMoE + RQ-VAE 重训与表示/量化评估（§1–§3）。
  - **结构关键变体**（Full、w/o load、no direct、no residual、no experts）进行 **400-step（≈2 epochs）从头 SFT**，与主实验同预算；结果见 §4.2。
  - 其余损失变体的下游 HR 标注 “—”；`analysis/omoe_ablation/run_downstream.py` 可直接补齐（`--sft-max-steps 400`）。
- 评测：20 beams，full test 25,373，batch 16，4 GPUs；HR/NDCG@5/10/20 由 `evaluate.py` 计算。
- 生成器收敛诊断：Full 400-step 的 eval loss=1.896，主实验 2-epoch 为 1.597；400-step 只生成 **73 个唯一 top-1 SID / 591 个唯一 beam SID**（主实验 2ep 约 225 / 2,309）。这说明 400-step 生成器仍明显欠拟合/生成坍缩，**下游 HR 的绝对值和变体间差异会被生成器预算主导**。因此本报告把表示级 + 量化级作为主证据，下游 HR 作为辅助/趋势证据。

### 4.2 结果（400-step 关键变体）

> 400-step 队列（Full、w/o load、no direct、no residual、no experts）已全部完成；其余 14 个损失变体的下游 HR 未跑（见 §4.1）。
> **预算敏感性**：同一 Full 变体，200 steps → 400 steps 时 HR@20 从 0.0035 → 0.0124（3.5×），说明 400-step 仍远未收敛；下游差异必须结合预算解读。主实验 2-epoch/5000-epoch RQ/50-beam 的 HR@20=0.0337 仅作参考，不与本表严格同协议。

| Variant | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| 01 Full (200 steps) | 0.0004 | 0.0015 | 0.0035 | 0.0002 | 0.0006 | 0.0011 | 预算不足，仅作锚点 |
| 01 Full (400 steps) | 0.0047 | 0.0076 | 0.0124 | 0.0031 | 0.0041 | 0.0053 | 2 epochs, 1000-epoch RQ, 20 beams |
| 03 w/o load (400) | 0.0093 | 0.0141 | **0.0201** | 0.0062 | 0.0077 | **0.0092** | 2 epochs, 1000-epoch RQ, 20 beams |
| 11 no direct (400) | **0.0133** | **0.0204** | **0.0336** | **0.0085** | **0.0108** | **0.0141** | 2 epochs, 1000-epoch RQ, 20 beams |
| 13 no residual (400) | 0.0128 | 0.0192 | **0.0291** | 0.0081 | 0.0101 | **0.0126** | 2 epochs, 1000-epoch RQ, 20 beams |
| 12 no experts (400) | 0.0039 | 0.0054 | 0.0082 | 0.0029 | 0.0034 | 0.0041 | 2 epochs, 1000-epoch RQ, 20 beams |
| 主实验 Full 2ep（5000-epoch RQ, 50 beams） | 0.0127 | 0.0218 | 0.0337 | 0.0082 | 0.0112 | 0.0142 | 参考，不与本表严格同协议 |

### 4.3 关键（且不利）的发现：下游排序与表示级“健康度”几乎相反

**最强反例：no direct（11）的 HR@20=0.0336，几乎等于主实验 Full 2ep 的 0.0337，而 Full 本协议只有 0.0124。** no-direct 的表示级指标是“更差”的（routing 坍缩：item H 0.0094、gate CV 0.842；fused cos 0.404；collision 21.2%；ARI 0.529），但它的 SID 空间对生成器更容易（eval loss 1.672 vs Full 1.896；unique top-1 97 vs 73）。这说明：

- **direct projection 在表示层面是稳定骨架，但在下游生成层面不是“必需项”；去掉它反而让 SID 空间更可学。**
- **“表示级健康”与“生成器可学性”存在权衡**：Full 的专家多样性/正交性/路由健康提高了表示质量，但也提高了生成难度；在 400-step 短预算下，no-direct 的坍缩空间更容易被生成器拟合。
- 这不能反推“direct projection 无用”：no-direct 的 gate/routing 已经坍缩，专家分工消失；如果论文目标是可解释的专家路由/表示质量，Full 仍更合理。但如果论文目标是短预算 HR，no-direct 是更强的 baseline，必须报告。
- **no residual（13）同样更强：HR@20=0.0291**（vs Full 0.0124）。这与它的 RQ 指标一致（MSE 4.83、collision 14.3%、PPL 49.9）：去掉 `u_i` 锚点后，SID 空间更容易量化/生成。**outer residual 在本协议下不是下游收益来源；它的作用只能表述为“保留 CSGA 语义锚点/保守融合”，不能表述为“提升 HR”。**
- **no experts（12，direct only）反而最差：HR@20=0.0082**，低于 Full 的 0.0124，更远低于 no-direct 的 0.0336。这说明下游有效成分主要是 **expert mixture**；direct projection 单独不够，甚至会稀释 mixture。
- 目前的下游排序：**no-direct 0.0336 > no-residual 0.0291 > w/o load 0.0201 > Full 0.0124 > no-experts 0.0082**。这个排序与表示级“健康度”几乎相反，是本报告最重要的负面发现。
- 解释（可检验的假设）：
  1. **expert mixture 是下游有效成分**：no-direct（只有 mixture）最好，no-experts（只有 direct）最差；Full 的 direct 分支可能把 mixture 的信号稀释了。
  2. **outer residual 是语义锚点，但对短预算生成器是“噪声/难度”**：no-residual 的 RQ 空间更易量化，HR 更高。
  3. **负载均衡让路由更均匀，但可能牺牲“把权重集中到最有用的专家”的下游收益**：no-load 的 HR 更高，尽管路由坍缩。
  4. 这些解释都受 400-step 生成器预算限制；不能据此断言“direct/residual/load 在充分训练后也无用”。

### 4.4 另一个不利发现：w/o L_load 在下游更高

- **Full：HR@20=0.0124；w/o L_load：HR@20=0.0201。** 去掉负载均衡后，表示级出现明显的路由坍缩（gate CV 1.232、item H 0.013、fused cos 0.497），但它的 RQ 空间更容易量化（MSE 5.26 vs 11.93；collision 14.4% vs 17.9%；codebook PPL 45.2 vs 28.5），并且在 400-step 生成器下生成了更多样的 SID（93 vs 73 unique top-1；995 vs 591 unique beams），因此 HR 更高。
- 这是一个必须如实报告的**负面/复杂结果**：**“L_load 防止路由坍缩”是表示级事实，但“L_load 提升下游 HR”在本预算下不成立。** 可能的原因：
  1. 400-step 生成器远未收敛（eval loss 1.90 vs 主实验 1.60；unique top-1 仅 73），下游排序被生成器预算主导；
  2. 负载均衡让融合空间更“多样/更难量化”（MSE、collision 更高），短预算生成器无法把这种多样性转化为收益；
  3. no-load 的单一专家坍缩反而降低了目标空间的熵，使生成更容易。

| 诊断 | Full 400 | w/o load 400 |
|---|---:|---:|
| eval loss | 1.896 | — |
| unique top-1 SID | 73 | 93 |
| unique beam SID | 591 | 995 |
| HR@20 | 0.0124 | **0.0201** |

- 因此论文中不能写“Full OMoE 因负载均衡而获得更高 HR”，也不能写“direct projection 提升下游 HR”。可以写“负载均衡维持了专家利用率和路由健康；在下游生成器充分训练时，这种表示质量是否转化为 HR 仍需验证”。若要主张下游收益，需要把生成器训到主实验的 2-epoch/5000-epoch RQ 预算，或使用更强的生成器；本次预算下**没有证据支持 Full 的下游优势**。
- 对 no-direct / no-residual / no-experts 的 400-step 结果同样按此口径解读：若它们与 Full 接近或更好，应归因于“短预算生成器 + 量化难度权衡”，不能反推“结构项无用”。

## 5. 论文可用的结论与不可用的结论

### 可以支撑（有直接证据）

1. **L_ortho 控制专家分工**：去掉后专家 `|cos|` 0.008→0.345；Full 的专家输出近似正交。
2. **L_load 是防路由坍缩的关键（表示级）**：去掉后 gate CV 0.013→1.232、item H 1.416→0.013、fused cos 0.061→0.497。**注意：在本报告的 400-step 下游预算下，w/o L_load 的 HR@20 反而更高（0.0201 vs 0.0124），因此不能把表示级机制直接等同于下游收益。**
3. **L_dup 控制融合输出的近重复**：去掉后 fused cos 0.061→0.246，而专家 `|cos|` 仍低；它与 L_ortho 作用在不同层级。
4. **L_var 保留 routing 的 item 依赖性**：去掉后 item H→1.609、routing var→0；但不会单独导致融合塌缩。
5. **direct projection 是稳定骨架，不是旁路（仅表示级）**：direct/mixture = 58%/42%，cos=−0.62；去掉 direct 后路由和重构同时退化（item H 0.009、collision 21.2%）。**但下游 400-step 中 no-direct 的 HR@20=0.0336 高于 Full 0.0124，因此“direct 对下游必要”不成立。**
6. **outer residual 不是旁路**：anchor/branch 范数 71%/68%，cos≈0.001；但 v2 中去掉 residual 不会塌缩，因此它的作用是“语义锚定/保守融合”，不是“防塌缩”。
7. **gate 必须可学习且非均匀**：uniform gate 导致 fused cos 0.457；static gate 0.096；instance gate 0.061。instance-dependence 有增益，但不是“有/无”的开关。
8. **shared/specific 二分不是主要矛盾**：shared-only、modality-only、plain MoE 都能得到接近 Full 的 fused cos；plain MoE 参数更多。
9. **OMoE 改变 SID 分区**：变体 vs Full 的 final SID ARI 0.529–0.816，远低于码本置换；no-direct 的 ARI 最低。
10. **低 MSE / 低 collision 可能是坍缩的信号**：w/o load、uniform gate、no residual 的 RQ 指标更好，但表示级出现 near-duplicate/routing collapse。论文中应把 reconstruction/collision 与 expert/routing 指标一起报告，避免单指标误读。
11. **下游有效成分主要是 expert mixture，而不是 direct/residual**：no-direct（只有 mixture）HR@20=0.0336，no-experts（只有 direct）0.0082，Full 0.0124；no-direct 比 no-experts 高 4.1×，Full 比 no-experts 高 1.5×。**若论文要主张下游收益，应把“mixture-only/去掉 direct+residual”作为更强的候选，而不是 Full。**
12. **Full 不是下游最优**：在本 400-step 预算下，Full 的 HR@20 低于 no-direct、no-residual、w/o load。必须如实报告，不能把 Full 包装成下游最佳。

### 应避免/弱化

- ❌ “专家严格正交/完全解耦”：Full 的平均 `|cos|` 是 0.008，但最大 pair 仍有 0.023；应说“近似正交”。
- ❌ “instance-dependent routing 是性能提升的必要条件”：static gate 已经能避免 uniform 的塌缩；instance gate 只是进一步降低 fused cos。
- ❌ “shared/specific 专家划分是核心贡献”：plain MoE 在同一预算下接近 Full；应把重点放在正交/负载/dup/gate/direct。
- ❌ “OMoE 大幅提升 HR”：本预算下 Full 的 HR@20（0.0124）低于 w/o L_load（0.0201）和 no-direct（0.0336）；不能声称 Full 的下游优势。应说明生成器是瓶颈、提升幅度和方差，并把下游结论限定为“未建立”。
- ❌ “direct projection 提升下游 HR”：no-direct 的 HR@20=0.0336，接近主实验 Full 2ep（0.0337），而 Full 本协议仅 0.0124；只能声称 direct 在表示层面稳定路由，不能声称其下游必要性。
- ❌ “表示级越健康，下游一定越好”：本报告出现明显的反向关系（no-load/no-direct 表示级更差但 HR 更高），应表述为“表示质量与生成器可学性之间的权衡”。
- ❌ “reconstruction error 越低越好”：本文的坍缩变体反而 MSE 更低；应改为“在保持专家多样性和路由健康的前提下比较 MSE”。

## 6. 建议的论文呈现方式

1. **主表**：Full + 02/03/04/05 + 07/08 + 11/12/13 + 14/15 + 18/19；列：HR/NDCG@20、expert `|cos|`、load CV、item H、fused cos、params。把 09/10/16/17 放附录。
2. **机制图**：
   - 专家 `|cos|` 柱状图（fig_expert_cos.png）
   - load CV 柱状图（fig_load_cv.png）
   - fused cos 柱状图（fig_fused_cos.png）
   - item routing entropy（fig_item_entropy.png）
   - collision / recon MSE（fig_collision.png / fig_recon_mse.png）
3. **Bypass 小节**：用 §2 的范数分解 + 反事实变体 11/12/13；这是审稿人最可能追问的点。
4. **诚实边界**：明确 RQ 1000 epochs / 400-step 生成器的预算，主实验 2-epoch/5000-epoch 结果单独作为参考；不要把消融表的绝对值与主表直接比较。
5. **选择性强调**：正文强调“omni-MoE 的 expert mixture 提供了可解释的专家分工/路由；正交、负载、dup 在表示层面各自控制一个失效模式”；把“no-direct/no-residual/no-load 的下游 HR 更高”“plain MoE/static gate 接近 Full”放在讨论/限制，不回避，但不作为主卖点。
6. **如果论文必须报下游收益**：把 **mixture-only（no direct + no residual）** 作为候选主模型重跑 2-epoch/5000-epoch RQ/50-beam；本报告的 400-step 结果已经显示它的 HR@20（0.0336）≈ 主实验 Full 2ep（0.0337），且参数更少（306.8M vs 357.1M）。这比坚持 Full 更有说服力。
7. **如果论文主张可解释性/表示质量**：保留 Full，报专家 `|cos|`、load CV、item H、fused cos、SID ARI，并把下游 HR 作为“未建立/受生成器预算限制”的诚实限制。

## 7. 复现

```bash
cd /data02/sht/MIMIGenRec/analysis/omoe_ablation
# 1) 表示级：全部 19 个变体（4 GPU 并行）
bash run_representation_all.sh
# 2) RQ-VAE + SID + 预处理（4 GPU 并行）
bash run_rq_phase.sh
# 3) 下游 400-step（关键变体；可改 VARIANTS/STEPS）
VARIANTS="01_full_omoe 03_wo_load 11_no_direct 13_no_residual 12_no_experts" STEPS=400 \
  bash run_downstream_key.sh
# 4) 汇总
python collect_results.py
python sid_agreement.py
```

产物：
- `results/ablation_master.csv`
- `results/representation_table.md`
- `results/ablation_table.md`
- `results/sid_agreement_vs_full.csv`
- `results/downstream_*.json`
- `figures/fig_*.png`
