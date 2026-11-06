# Sports10core SID 碰撞分析（MIMIGenRec / SAM-GR `csga_omoe_v2`）

**数据**：Amazon `Sports_and_Outdoors` strict 10-core，11,246 items，17,904 users，
train 202,980 / valid 25,373 / test 25,373。
**主设定**：MIMIGenRec `csga_omoe_v2`（CSGA + OMoE-v2），RQ-VAE 4×256，15% SID merge。
**分析日期**：2026-10-24。
**产物**：`analysis/sid_collision_sports10/`（脚本、CSV/JSON、图、本报告）。

---

## 0. 结论摘要

1. **生产 Full SAM-GR 的最终 SID：9,663 个 unique full SID，excess collision 14.08%，
   26.71% 的 item 落在 size>1 的 bucket，最大 bucket 5。**
   TIGER 几乎相同：9,659 / 14.11% / 26.77% / 5。
   **MMGRec 主表使用 shared SID，所以碰撞指标与 Full 完全相同；它自带的 3-token SID 是
   5,992 unique / 46.72% excess / max 27，另有 1..27 的 pointer 做 1:1 消歧（见 2.1）。**
2. **14.08% 主要不是 RQ-VAE 本身造成的，而是 15% SID merge 造成的。**
   Full 的 raw argmin SID 只有 11,113 个 unique（1.18% excess），Sinkhorn 后为 11,246
   （0 碰撞）；随后 15% merge 把 1,686 个 item 并入热门邻居，才变成 9,663 / 14.08%。
3. **raw RQ-VAE 碰撞与 merge 碰撞的性质完全不同。**
   Full raw 碰撞的 144 个 item-pair：text cos 0.984、image cos 0.955、同三级类目 89.6%、
   同用户共交互 30.6%，是真正的近重复/强共交互 item；merge 后的 1,764 个 pair：
   text cos 0.888、image cos 0.759、同三级类目 31.9%、同用户共交互 3.63%，
   与“同 popularity decile 随机 pair”的 3.54% 几乎相同。
4. **碰撞 item 不是“热门 item 互相合并”，而是“长尾 item 被并入热门锚点”。**
   Full 碰撞 pair 的平均较小 popularity 只有 3.12（随机 pair 为 9.80），
   bucket 内常见的是“低流行 item + 高流行 canonical item”。
5. **full-SID 碰撞低，不等于 prefix 碰撞低。**
   Full 的 depth-1 只有 69 个 prefix（collision 99.39%，max 323），depth-2 collision 72.63%；
   CARD 虽然 full-SID collision=0，但 depth-2 collision 仍有 14.99%。
6. **SID-level 评测对 Full/TIGER/MMGRec 偏乐观。**
   Full HR@50 5.51% → 均匀 item resolver 期望 4.87% → 仅 singleton 下界 4.27%；
   TIGER 1.99% → 1.81% → 1.63%；MMGRec 3.39% → 3.09% → 2.81%；CARD 6.55% 不变
   （CARD 是 1:1 SID）。
7. **生成侧诊断：** Full/TIGER/MMGRec 的 invalid SID rate = 0（trie/词表约束），
   CARD = 0.101%（1,281 / 1,268,600，无约束 T5）；四者 duplicate-beam rate 均为 0；
   ambiguous SID prediction rate：Full 15.46%、TIGER 16.00%、MMGRec 12.43%、CARD 0。
8. **300-epoch 组件消融只用于“同预算趋势”，不能当作生产绝对值。**
   在该短预算下 CSGA-only 明显优于 Base/OMoE-only（full-SID excess 22.33% vs 33.97%/33.76%），
   Full 300ep 最好（17.21%）；但 300ep 的 RQ-VAE 远未收敛（raw collision 72–94%），
   绝对值不可与 5,000ep 的生产 Full 直接比较。

---

## 1. 口径与定义

### 1.1 数据与 test rows

| 模型 | test rows | 说明 |
|---|---:|---|
| Full SAM-GR | 25,373 | strict test 全量 |
| TIGER | 24,819 | `load_samples` 过滤掉 target 出现在 history 中的 554 行 |
| MMGRec | 25,373 | 与 Full 同一批 test rows |
| CARD | 25,372 | 去掉 1 条缺图 target；item universe 11,244（2 个缺图 item） |

### 1.2 指标定义

- **unique full-SID count**：`U = |{sid(item)}|`。
- **excess collision rate**：`(N - U) / N`。一个 size=`k` 的 bucket 贡献 `k-1`。
- **collided-item ratio**：`Σ_{g: |g|>1} |g| / N`。一个 size=`k` 的 bucket 贡献 `k`。
  因此它总是不小于 excess collision rate；Full 为 26.71% vs 14.08%。
- **pairwise collision probability**：
  `P = Σ_g n_g(n_g-1) / (N(N-1))`，即随机抽两个不同 item 得到同一 full SID 的概率。
  该指标对大 bucket 呈二次加权；Full 的碰撞分散在 1,276 个 size-2 bucket 中，所以
  `P=2.79e-5` 很小，但 26.71% 的 item 仍在碰撞 bucket 中。
- **max collision bucket**：`max_g n_g`。
- **prefix depth d**：只看前 d 个 token 的 unique prefix 数 `U_d`；
  `unique-prefix ratio = U_d/N`，`prefix collision rate = (N-U_d)/N`。
- **Sinkhorn 前后**：raw argmin → 对碰撞组做 Sinkhorn 局部重分配 → 15% merge。
- **expected item-level HR（均匀 resolver）**：若预测 SID bucket 大小为 `b` 且包含 target，
  假设在 bucket 内均匀随机选一个 item，则该 rank 命中 target 的期望为 `1/b`。
  这是“SID-level 指标若换成 item-level 会掉多少”的无偏期望，不代表某个具体 resolver。

### 1.3 关键代码位置

- Sinkhorn 局部碰撞重分配：`rq/generate_indices.py:166-185`；
  Sinkhorn 本体：`rq/models/layers.py:80-100`。
- 15% collaborative merge：`rq/generate_indices.py:41-94`。
- SID-level 评测（字符串相等，不做 item 消歧）：`evaluate.py:143-164`。
- trie 对重复 full SID 去重（`set(item_strs)`）：`evaluate.py:65`。

---

## 2. 模型与 SID 来源

| 模型 | SID 构造 | Items | RQ/训练预算 | 产物 | 备注 |
|---|---|---:|---|---|---|
| **Full SAM-GR** | RQ-VAE on `csga_moe_collab_v2` + Sinkhorn + 15% merge | 11,246 | 5,000 ep | `rq_output/Sports_and_Outdoors/best_collision_model.pth`；`data/Amazon/Sports_and_Outdoors/Sports_and_Outdoors.index.json` | 生产主设定 |
| **TIGER** | RQ-VAE on `text_only` + Sinkhorn + 15% merge | 11,246 | 5,000 ep | `rq_output/tiger_Sports_and_Outdoors/best_collision_model.pth`；`analysis/samgr_sports10/results/text_only_sid/Sports_and_Outdoors.index.json` | 文本 SID baseline |
| **MMGRec** | **共享 benchmark SID（即 Full 的 index）** | 11,246 | — | `MMGRec/.../sid_benchmark/index.json` | 不是独立 SID 构造；碰撞指标与 Full 完全相同 |
| **CARD** | NU-RQ-VAE (4×256) + collision resolution | 11,244 | 3,000 ep | `CARD/data/Sports_and_Outdoors/Sports_and_Outdoors_card_codes.npy` | full-SID 0 碰撞 |
| **Base joint-VLM (300ep)** | RQ-VAE on `multimodal` + 15% merge | 11,246 | 300 ep | 本次分析新训：`analysis/sid_collision_sports10/checkpoints/base/` | 短预算组件消融 |
| **CSGA-only (300ep)** | RQ-VAE on `csga_aligned` + 15% merge | 11,246 | 300 ep | `.../checkpoints/csga/` | 短预算组件消融 |
| **OMoE-only (300ep)** | OMoE-v2(text,image,multimodal) → RQ-VAE + 15% merge | 11,246 | 300 ep | `.../checkpoints/omoe/`；`embeddings/omoe_only.npy` | 短预算组件消融 |
| **Full SAM-GR (300ep)** | RQ-VAE on `csga_moe_collab_v2` + 15% merge | 11,246 | 300 ep | `.../checkpoints/full300/` | 与上面三者同预算的 Full |

**必须强调的三点：**

1. **MMGRec 的碰撞指标不是独立测量。** 为保证与 MIMIGenRec 可比，MMGRec 主结果
   固定使用 Full 的 shared SID（`summary.json` 中 `shared_sid.pinned_index`）。
   因此它的 9,663 / 14.08% / 26.71% 与 Full 完全相同。MMGRec 仓库自带的 3×128 RQ-VAE
   在该 split 上严重塌缩（`summary.json`：5,992 unique、max collision 27），
   只作为 diagnostic，不用于主比较。
2. **CARD 的 item universe 是 11,244，不是 11,246。** 少 2 个缺图 item；
   CARD 的 full-SID 是 1:1，所以它的 SID-level 指标等价于 item-level。
3. **Base/CSGA/OMoE/Full300 是本次为碰撞分析补训的 300-epoch 短预算 SID，不是已有生产记录。**
   300ep 远未收敛：raw collision 分别为 94.41% / 71.56% / 89.87% / 75.80%。
   它们只用于“同一 RQ-VAE/merge 配方下的组件趋势”，不能与 5,000ep 的 Full 绝对值直接比较。
   若只看趋势：CSGA-only 明显优于 Base/OMoE-only；Full 300ep 在 Sinkhorn 后最好。

### 2.1 MMGRec 的两种 SID 口径（不要混用）

MMGRec 仓库里有**两套 SID**，本次分析必须区分：

| 口径 | SID 结构 | Items | Unique | Excess | Collided items | Max bucket | 用途 |
|---|---|---:|---:|---:|---:|---:|---|
| **主结果（shared benchmark）** | 4 个 semantic token（MIMIGenRec Full index） | 11,246 | 9,663 | 14.08% | 26.71% | 5 | 与 Full/TIGER 可比的主表 |
| **仓库自带（native）** | 3 个 RQ-VAE token + 1 个 pointer（1..27） | 11,246 | 5,992（3-token） | 46.72% | 65.00% | 27 | 仓库原始设计；未用于主比较 |

- **Native 3-token prefix**：depth-1 仅 25 个 code（max 6,841），depth-2 948 个（max 292），
  depth-3 5,992 个（max 27）；即前两层严重塌缩，第三层仍有 46.72% full-SID 碰撞。
- **Native bucket histogram**：size 1: 3,937；2: 955；3: 460；4: 230；5: 120；6–10: 237；
  11–20: 50；21–50: 3；max 27。prefix collision：depth-1 99.78%、depth-2 91.57%、depth-3 46.72%。
- **Pointer 是 MMGRec 的“消歧 token”**：`stage1/tgt_mtx.npy` 的第 4 列是 1..27 的 pointer；
  加上 pointer 后，11,246 个 item 对应 11,246 个唯一 4-token 序列（1:1）。
  也就是说，MMGRec 原生设计用 **3 个 semantic token + 1 个 pointer** 来映射到最终 item，
  而不是要求 4 个 semantic token 完全唯一。主 benchmark 为了与 MIMIGenRec 可比，
  把 pointer 换成了第 4 个 semantic token（shared 4-token SID），因此主表的碰撞统计
  继承 Full，而不是 native 的 46.72%。
- **Native 碰撞 pair 的性质**（15,861 pairs）：text cos 0.8852、image cos 0.7427、
  同三级类目 17.25%、同用户共交互 **8.00%**、平均 popularity 26.35（随机 18.10）。
  这与 Full merge 的“长尾→热门锚点”不同：MMGRec native 的碰撞更多是**热门 item 的
  协同碰撞**（co-int 8.0% vs same-decile 3.54%，2.3×），但语义/类目一致性较弱。
  其 46.72% 的高碰撞由 pointer 兜底，所以 native 的 item-level 仍是 1:1。
- **哈希验证**：MMGRec 的 `sid_benchmark/index.json` 与 Full 的 `Sports_and_Outdoors.index.json`
  SHA256 完全相同（`444859d28900b3220588466646b9257a58729aefc6484aaa6f9f419a66c22948`），
  所以主表碰撞统计逐 item 相同，不是近似。
- **结论**：报告 MMGRec 时必须注明口径。主表用 shared 4-token（9,663 / 14.08% / max 5）；
  若讨论 MMGRec 自身 SID 质量，用 native 3-token + pointer（5,992 / 46.72% / max 27，
  pointer 1:1）。两者不能混在一起比较。

**300ep 组件的 embedding 定义（便于复现）：**

- **Base joint-VLM**：`Sports_and_Outdoors.emb-Qwen3VL-multimodal.npy`，即 Qwen3-VL 的 text+image 多模态表示。
- **CSGA-only**：`analysis/samgr_sports10/results/csga_aligned.npy`，即 `csga_align(multimodal, SASRec collab)`，不做 OMoE。
- **OMoE-only**：`analysis/sid_collision_sports10/embeddings/omoe_only.npy`，即 `omoe_fuse_v2(text_only, image_only, multimodal)`，不使用 CSGA。
- **Full SAM-GR (300ep)**：`Sports_and_Outdoors.emb-Qwen3VL-csga_moe_collab_v2.npy`，即 `OMoE-v2(text, image, csga_align(multimodal, collab))`。
- 四者使用**完全相同**的 RQ-VAE 配方（4×256、batch 20480、lr 1e-3、300 epochs、seed 2024）和
  15% merge，因此可以互相比较；但它们与 5,000ep 的 Full/TIGER 不是等预算。

---

## 3. 全量碰撞指标

![collision overview](figures/fig_collision_overview.png)

**T1. Final full-SID collision metrics**

| Model | Items | Unique full SIDs | Excess (N-U)/N | Collided-item ratio | Pairwise collision P | Max bucket | Mean bucket |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full SAM-GR | 11,246 | 9,663 | 14.08% | 26.71% | 2.79e-05 | 5 | 1.164 |
| TIGER | 11,246 | 9,659 | 14.11% | 26.77% | 2.80e-05 | 5 | 1.164 |
| MMGRec | 11,246 | 9,663 | 14.08% | 26.71% | 2.79e-05 | 5 | 1.164 |
| CARD | 11,244 | 11,244 | 0.00% | 0.00% | 0.00e+00 | 1 | 1.000 |
| Base joint-VLM (300ep) | 11,246 | 7,426 | 33.97% | 49.24% | 3.78e-04 | 85 | 1.514 |
| CSGA-only (300ep) | 11,246 | 8,735 | 22.33% | 36.80% | 9.24e-05 | 31 | 1.287 |
| OMoE-only (300ep) | 11,246 | 7,449 | 33.76% | 49.27% | 2.36e-04 | 54 | 1.510 |
| Full SAM-GR (300ep) | 11,246 | 9,311 | 17.21% | 31.07% | 4.27e-05 | 19 | 1.208 |

> MMGRec 行是 shared benchmark SID（与 Full 同一 index）；其 native 3-token + pointer
> 的 5,992 / 46.72% / max 27 见 2.1，不要与主表混用。

**读法：**

- Full/TIGER 的 `P` 只有 2.8e-5（约每 10 万对 item 2.8 对），因为碰撞主要是
  1,276 个 size-2 bucket；但 26.71% 的 item 至少在一个碰撞 bucket 里。两者不矛盾。
- Base/OMoE 300ep 的 `P` 更高（3.78e-4 / 2.36e-4），因为存在 max 85/54 的大 bucket；
  excess collision 与 Full 只差 2.4 倍，但 `P` 差 8–13 倍，说明其碰撞更“集中”。
- CARD 的 full-SID 完全 1:1；但它并非没有 prefix 碰撞（见第 5 节）。
- 作为参考，若 U 个 SID 等概率，`P=1/U`；Full 的 `1/9663=1.04e-4`，实际 `2.79e-5` 更低，
  因为大部分 item 在 singleton bucket，碰撞权重集中在少数 bucket。`P` 与 excess collision
  不是单调关系：Base 的 excess 只比 Full 高 2.4×，但 `P` 高 13.5×，因为它的碰撞集中在
  max 85 的大 bucket。

---

## 4. Sinkhorn reassignment 前后

![sinkhorn stages](figures/fig_sinkhorn_stages.png)

**T2. Raw / Sinkhorn / merge stages**

| Model | Stage | Unique SIDs | Excess | Collided items | Max bucket |
|---|---|---:|---:|---:|---:|
| Full SAM-GR | raw argmin | 11,113 | 1.18% | 2.28% | 4 |
| Full SAM-GR | after Sinkhorn | 11,246 | 0.00% | 0.00% | 1 |
| Full SAM-GR | after 15% merge | 9,663 | 14.08% | 26.71% | 5 |
| TIGER | raw argmin | 11,016 | 2.05% | 3.74% | 8 |
| TIGER | after Sinkhorn | 11,242 | 0.04% | 0.07% | 2 |
| TIGER | after 15% merge | 9,659 | 14.11% | 26.77% | 5 |
| Base joint-VLM (300ep) | raw argmin | 629 | 94.41% | 98.70% | 495 |
| Base joint-VLM (300ep) | after Sinkhorn | 8,377 | 25.51% | 33.93% | 80 |
| Base joint-VLM (300ep) | after 15% merge | 7,426 | 33.97% | 49.24% | 85 |
| CSGA-only (300ep) | raw argmin | 3,198 | 71.56% | 82.54% | 152 |
| CSGA-only (300ep) | after Sinkhorn | 10,127 | 9.95% | 14.25% | 28 |
| CSGA-only (300ep) | after 15% merge | 8,735 | 22.33% | 36.80% | 31 |
| OMoE-only (300ep) | raw argmin | 1,139 | 89.87% | 95.78% | 514 |
| OMoE-only (300ep) | after Sinkhorn | 8,311 | 26.10% | 34.71% | 48 |
| OMoE-only (300ep) | after 15% merge | 7,449 | 33.76% | 49.27% | 54 |
| Full SAM-GR (300ep) | raw argmin | 2,722 | 75.80% | 87.88% | 97 |
| Full SAM-GR (300ep) | after Sinkhorn | 10,815 | 3.83% | 6.38% | 19 |
| Full SAM-GR (300ep) | after 15% merge | 9,311 | 17.21% | 31.07% | 19 |

> **不适用 Sinkhorn 的模型：** CARD 使用 NU-RQ-VAE 的 collision resolution（best raw
> collision 0.0178% → generate codes 后 0，见 `CARD/logs/nurqvae_sports.log` 与
> `generate_codes_sports.log`），不是 Sinkhorn；MMGRec native 使用 3-token SID + 1..27
> pointer 做 1:1 消歧，也没有 Sinkhorn 阶段。MMGRec 主表因为直接用 Full 的 shared index，
> 所以继承 Full 的 raw/Sinkhorn/merge 三阶段。

**Sinkhorn 到底做了什么：**

- 它**不是**对全部 item 做全局最优重分配。`generate_indices.py:166-185` 先找出 full-SID
  相同的 collision groups，只对这些 group 内的 item 重新调用
  `get_indices(..., use_sk=True)`；最后一层 VQ 的 `sk_epsilon` 被设为 0.003，
  其他层为 0；最多重复 20 轮。Sinkhorn 在 latent×codebook 上做行列归一化的软分配，
  再 argmax。
- 因此 Sinkhorn 的效果取决于原始 codebook 是否可分。Full/TIGER 在 5,000ep 已收敛，
  raw collision 只有 1–2%，Sinkhorn 可以几乎完全消解（0 / 0.04%）。
- Base/OMoE 在 300ep 时 codebook 严重塌缩（raw 94%/90%，max bucket 495/514），
  Sinkhorn 20 轮后仍有 25–26% excess；Collision group 仍然很大。
- **merge 是独立于 Sinkhorn 的步骤**：15% 最不流行 item 被并入其 top-1 collaborative
  neighbor（且 neighbor 必须更 popular、仍是 canonical）。Full 的 14.08% 几乎全部来自
  这一步，而不是 Sinkhorn 残留。
  TIGER 同样：raw 2.05% → Sinkhorn 0.04% → merge 14.11%，merge 也主导最终碰撞。

**一个重要的 budget 观察：** 同样是 Full 的 embedding，5,000ep 与 300ep 的 raw collision
是 1.18% vs 75.80%，但 Sinkhorn 后是 0% vs 3.83%，merge 后是 14.08% vs 17.21%。
最终碰撞数的差距远小于 raw 差距，说明 Sinkhorn+merge 对短预算 codebook 有很强的
“碰撞压缩”作用。但 300ep 的 merge 仍把自身 raw 的 text 0.923 / same-L3 44.86% /
co-int 5.97% 稀释到 0.897 / 41.10% / 5.74%，且最终 max bucket 19 vs 5、excess 17.21% vs 14.08%。
所以“最终碰撞数接近”不等于“SID 质量相同”：300ep 的 bucket 更碎、Sinkhorn 残留更多，
只是它的 pre-merge pair 更多，使 merge 后的平均语义相似度看起来略高。

---

## 5. Prefix depth：unique-prefix ratio 与 collision rate

![prefix collision](figures/fig_prefix_collision.png)

**T3. Prefix-depth statistics（final merged SIDs）**

| Model | Depth | Unique prefixes | Unique-prefix ratio | Prefix collision rate | Collided-item ratio | Max prefix bucket |
|---|---:|---:|---:|---:|---:|---:|
| Full SAM-GR | 1 | 69 | 0.0061 | 99.39% | 100.00% | 323 |
| Full SAM-GR | 2 | 3,078 | 0.2737 | 72.63% | 92.09% | 24 |
| Full SAM-GR | 3 | 8,952 | 0.7960 | 20.40% | 36.48% | 7 |
| Full SAM-GR | 4 | 9,663 | 0.8592 | 14.08% | 26.71% | 5 |
| TIGER | 1 | 88 | 0.0078 | 99.22% | 99.96% | 397 |
| TIGER | 2 | 4,163 | 0.3702 | 62.98% | 83.82% | 39 |
| TIGER | 3 | 8,908 | 0.7921 | 20.79% | 36.74% | 16 |
| TIGER | 4 | 9,659 | 0.8589 | 14.11% | 26.77% | 5 |
| MMGRec | 1 | 69 | 0.0061 | 99.39% | 100.00% | 323 |
| MMGRec | 2 | 3,078 | 0.2737 | 72.63% | 92.09% | 24 |
| MMGRec | 3 | 8,952 | 0.7960 | 20.40% | 36.48% | 7 |
| MMGRec | 4 | 9,663 | 0.8592 | 14.08% | 26.71% | 5 |
| CARD | 1 | 256 | 0.0228 | 97.72% | 100.00% | 109 |
| CARD | 2 | 9,558 | 0.8501 | 14.99% | 26.78% | 11 |
| CARD | 3 | 11,180 | 0.9943 | 0.57% | 1.05% | 5 |
| CARD | 4 | 11,244 | 1.0000 | 0.00% | 0.00% | 1 |
| Base joint-VLM (300ep) | 1 | 34 | 0.0030 | 99.70% | 100.00% | 2583 |
| Base joint-VLM (300ep) | 2 | 145 | 0.0129 | 98.71% | 99.84% | 1759 |
| Base joint-VLM (300ep) | 3 | 315 | 0.0280 | 97.20% | 99.60% | 611 |
| Base joint-VLM (300ep) | 4 | 7,426 | 0.6603 | 33.97% | 49.24% | 85 |
| CSGA-only (300ep) | 1 | 105 | 0.0093 | 99.07% | 99.90% | 1057 |
| CSGA-only (300ep) | 2 | 727 | 0.0646 | 93.54% | 98.39% | 395 |
| CSGA-only (300ep) | 3 | 1,827 | 0.1625 | 83.75% | 93.15% | 229 |
| CSGA-only (300ep) | 4 | 8,735 | 0.7767 | 22.33% | 36.80% | 31 |
| OMoE-only (300ep) | 1 | 29 | 0.0026 | 99.74% | 99.97% | 2776 |
| OMoE-only (300ep) | 2 | 181 | 0.0161 | 98.39% | 99.80% | 1588 |
| OMoE-only (300ep) | 3 | 527 | 0.0469 | 95.31% | 98.76% | 1134 |
| OMoE-only (300ep) | 4 | 7,449 | 0.6624 | 33.76% | 49.27% | 54 |
| Full SAM-GR (300ep) | 1 | 108 | 0.0096 | 99.04% | 99.88% | 1232 |
| Full SAM-GR (300ep) | 2 | 642 | 0.0571 | 94.29% | 98.68% | 268 |
| Full SAM-GR (300ep) | 3 | 1,469 | 0.1306 | 86.94% | 95.46% | 125 |
| Full SAM-GR (300ep) | 4 | 9,311 | 0.8279 | 17.21% | 31.07% | 19 |

> 这里展示 final merged SID；raw 和 Sinkhorn 阶段的同口径 prefix 统计在 `results/prefix_stats.csv`。

**关键观察：**

- **Full 的 layer-0 严重塌缩**：只有 69 个 code 被使用（256 个中），
  depth-1 collision 99.39%，最大 prefix bucket 323。TIGER 稍好（88 个），
  但仍然只有 34% 的 codebook 被使用。
- **CARD 的层级最均衡**：layer-0 使用 256/256 个 code，depth-2 unique-prefix ratio
  85.01%，depth-3 99.43%，depth-4 100%。CARD 的 full-SID collision=0 不是因为
  “merge 少”，而是因为 NU-RQ-VAE 每一层都用满了 codebook 并做了 collision resolution。
- **depth-2 是最能区分模型的一层**：Full 27.37% unique vs CARD 85.01%；
  Base 1.29%、CSGA 6.46%、OMoE 1.61%、Full300 5.71%。300ep 的 Base/OMoE 在
  depth-2 几乎完全塌缩（145/181 个 prefix），CSGA/Full 把 prefix 数提高到 727/642。
- **Full 的 depth-4 collision（14.08%）低于 depth-2（72.63%）**，说明最后一层在
  消歧；但生成时前两层分支极大，trie 约束下的 beam search 仍很难命中。
  这也是 SID-level 指标不高的结构性原因之一。

---

## 6. Collision bucket size 直方图

![bucket histogram](figures/fig_bucket_histogram.png)

**T4. Bucket-size histogram（final merged SIDs）**

| Model | 1 | 2 | 3 | 4 | 5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SAM-GR | 8242 | 1276 | 130 | 13 | 2 | 0 | 0 | 0 | 0 |
| TIGER | 8235 | 1278 | 131 | 13 | 2 | 0 | 0 | 0 | 0 |
| CARD | 11244 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Base joint-VLM (300ep) | 5709 | 1176 | 292 | 86 | 38 | 70 | 28 | 24 | 3 |
| CSGA-only (300ep) | 7108 | 1272 | 198 | 58 | 36 | 48 | 13 | 2 | 0 |
| OMoE-only (300ep) | 5705 | 1156 | 260 | 102 | 62 | 108 | 48 | 7 | 1 |
| Full SAM-GR (300ep) | 7752 | 1307 | 193 | 37 | 13 | 7 | 2 | 0 | 0 |

- Full/TIGER 的碰撞几乎全是 size-2（1,276 / 1,278 个），只有 2 个 size-5；
  这是 15% merge “一对一并入热门邻居”的直接形态。
- Base/OMoE 300ep 有 6–10、11–20、21–50、>50 的长尾 bucket，最大 85/54；
  这种集中碰撞会严重破坏 item 级区分度，也是 `P` 偏高的原因。
- Full 300ep 的直方图已经明显优于 Base/OMoE：max 19，size≥6 只有 9 个 bucket。
  这支持“CSGA+OMoE 组合”在短预算下仍能改善 codebook 使用。

完整的 raw/Sinkhorn/merge 直方图见 `results/bucket_histogram.csv`。

---

## 7. 碰撞商品进一步分析

![collided item analysis](figures/fig_collided_item_analysis.png)

**T5. 最终 merge SID 的碰撞 pair 统计**

> 类目粒度说明：`same L3 cat.` = Amazon category path 前 3 层相同（如 Sports & Outdoors → Sports → Cycling）；`same leaf cat.` = 完整 category path 相同。

| Model | Pairs | Same L3 cat. | Same leaf cat. | Text cos | Image cos | Mean min-pop | Co-interaction | Random co-int. | Same-pop-decile co-int. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SAM-GR | 1,764 | 31.86% | 1.76% | 0.8883 | 0.7590 | 3.12 | 3.63% | 2.08% | 3.54% |
| TIGER | 1,769 | 31.88% | 1.87% | 0.8886 | 0.7595 | 3.15 | 3.67% | 2.08% | 3.54% |
| Base joint-VLM (300ep) | 23,882 | 18.73% | 0.70% | 0.8970 | 0.7685 | 8.64 | 2.36% | 2.08% | 3.54% |
| CSGA-only (300ep) | 5,841 | 40.44% | 5.58% | 0.9069 | 0.7786 | 8.32 | 7.48% | 2.08% | 3.54% |
| OMoE-only (300ep) | 14,916 | 17.75% | 0.94% | 0.8998 | 0.7717 | 8.62 | 2.54% | 2.08% | 3.54% |
| Full SAM-GR (300ep) | 2,701 | 41.10% | 4.63% | 0.8968 | 0.7685 | 5.52 | 5.74% | 2.08% | 3.54% |

**T6. raw / Sinkhorn / merge 的碰撞性质对比**（关键表）

| Model / stage | Pairs | Text cos | Image cos | Same L3 cat. | Co-interaction |
|---|---:|---:|---:|---:|---:|
| Full SAM-GR raw | 144 | 0.9840 | 0.9550 | 89.58% | 30.56% |
| Full SAM-GR merged | 1,764 | 0.8883 | 0.7590 | 31.86% | 3.63% |
| TIGER raw | 292 | 0.9887 | 0.9264 | 89.38% | 33.56% |
| TIGER merged | 1,769 | 0.8886 | 0.7595 | 31.88% | 3.67% |
| Base 300ep Sinkhorn | 22,520 | 0.9037 | 0.7774 | 18.91% | 2.24% |
| Base 300ep merged | 23,882 | 0.8970 | 0.7685 | 18.73% | 2.36% |
| CSGA 300ep raw | 134,057 | 0.9240 | 0.7974 | 51.32% | 7.47% |
| CSGA 300ep Sinkhorn | 3,790 | 0.9249 | 0.7983 | 51.42% | 9.89% |
| CSGA 300ep merged | 5,841 | 0.9069 | 0.7786 | 40.44% | 7.48% |
| OMoE 300ep Sinkhorn | 14,897 | 0.9059 | 0.7833 | 16.35% | 2.15% |
| OMoE 300ep merged | 14,916 | 0.8998 | 0.7717 | 17.75% | 2.54% |
| Full 300ep raw | 93,912 | 0.9232 | 0.7839 | 44.86% | 5.97% |
| Full 300ep Sinkhorn | 874 | 0.9203 | 0.8052 | 67.05% | 9.84% |
| Full 300ep merged | 2,701 | 0.8968 | 0.7685 | 41.10% | 5.74% |

![collision source decomposition](figures/fig_collision_source.png)

### 7.1 是否属于相同 category？

- Full 最终碰撞 pair：同三级类目 **31.86%**，同叶子类目 **1.76%**；
  随机 pair 分别为 12.27% / 0.48%。
  → 碰撞 pair 比随机更有类目一致性（约 2.6× / 3.7×），但**远不是同款/同叶子类目**。
- raw RQ-VAE 碰撞：同三级类目 **89.58%**，几乎都是同类近重复；
  merge 后降到 31.86%，说明 **merge 把大量跨细类目的长尾 item 拉进了同一 SID**。
- CARD 没有碰撞 pair，不适用；它的 prefix depth-2 碰撞仍有 14.99%，但 full-SID 1:1。

### 7.2 文本/图像余弦相似度

- Full 最终碰撞 pair：text **0.8883**、image **0.7590**；
  随机 pair：0.8810 / 0.7408；同 category pair：0.9213 / 0.8526。
  → 只比随机高 +0.007 / +0.018，远低于同 category；不是“语义近重复”。
- Full raw 碰撞：text **0.9840**、image **0.9550**，是真正的近重复。
- TIGER 与 Full 几乎一致（0.8886 / 0.7595；raw 0.9887 / 0.9264）。
- CSGA/Full 300ep 的 Sinkhorn 碰撞更语义一致（text 0.925/0.920，same L3 51%/67%），
  但 merge 后也被稀释到 0.907/0.897、40%/41%。

**解释**：`merge_collaborative_neighbors` 只按 SASRec collaborative top-1 选邻居，
不检查文本/图像相似度，也不要求同叶子类目；因此 merge 后的 bucket 语义相似度
自然接近随机，而不是接近同 category。raw RQ-VAE 的碰撞则是因为 embedding 几乎相同，
所以 text/image cos 很高。两者不是同一种“碰撞”。

### 7.3 Popularity 分布

- Full 碰撞 pair 的平均较小 popularity 只有 **3.12**（随机 pair 9.80）；
  平均 popularity 10.11（随机 18.10）；中位数 6（随机 12）。
- 这与 merge 规则一致：按 popularity 升序，把最不流行的 15% item 并入更流行的
  canonical neighbor。因此 bucket 的典型形态是 **“长尾 item + 热门锚点”**，
  不是“热门 item 互相碰撞”。
- 这也解释了为什么 Full 的碰撞 pair 共交互率低：长尾 item 本身交互就少，
  与热门锚点的共现自然少。

### 7.4 是否经常被同一用户共同交互？

- Full 最终碰撞 pair 的 co-interaction rate = **3.63%**；
  随机 pair = 2.08%（1.75×）；但 **同 popularity decile 随机 pair = 3.54%**，
  所以控制 popularity 后只有 **1.03×**，基本没有额外共交互信号。
- Full **raw** 碰撞 pair 的 co-interaction = **30.56%**；TIGER raw = **33.56%**。
  这说明 RQ-VAE 自身的碰撞确实是强共交互/近重复 item。
- Full raw 只有 144 pair，最终 merged 有 1,764 pair；merge 新增的约 1,620 pair 主导了
  最终平均值，所以 3.63% 会接近 popularity-matched 的 3.54%，而不是 raw 的 30.56%。
- CSGA-only 300ep 的 Sinkhorn 碰撞 co-interaction = **9.89%**（同 decile 3.54%，2.8×），
  Full 300ep Sinkhorn = **9.84%**（2.8×）；说明 CSGA 在 pre-merge 阶段确实
  形成了 collaborative bucket。merge 后 CSGA 降到 7.48%，Full 降到 5.74%。
- Base/OMoE 300ep 的 Sinkhorn 碰撞 co-interaction 只有 2.24%/2.15%，
  低于同 popularity decile 的 3.54%，说明没有 CSGA 时碰撞 bucket 不具备协同结构。

**结论：**
- “碰撞商品是否经常被同一用户共同交互？”——**raw RQ-VAE 碰撞：是（30–34%）**；
  **最终 merge 碰撞：基本不是（3.63%，与 popularity-matched 随机 pair 的 3.54% 持平）**。
- “是否同 category？”——**raw：89.6% 同三级类目；merge 后：31.9%，仅比随机高 2.6×**。
- “文本/图像相似度？”——**raw：0.984/0.955（近重复）；merge 后：0.888/0.759（接近随机）**。
- “popularity 分布？”——**merge bucket 是长尾 item 并入热门锚点，min-pop 均值 3.12**。

---

## 8. 多个 item 共享 full SID 时，生成模型如何映射到最终 item

### 8.1 当前代码的实际行为

- **Full SAM-GR / TIGER（MIMIGenRec 风格）**：
  - `evaluate.py:65` 用 `set(item_strs)` 构建 trie，重复 full SID 只保留**一条路径**。
  - 生成模型输出的是 SID token 串，不是 item ID。
  - `compute_metrics`（`evaluate.py:143-164`）直接把预测字符串与 target 字符串比较，
    **没有任何 item 消歧**。若 target item 属于一个 size=`b` 的 bucket，
    只要预测到该 SID，就记为命中；模型无法知道是 bucket 里的哪个 item。
  - 因此 Full/TIGER 的 HR/NDCG 是 **SID-level**，对共享 SID 的 target 是乐观的。
- **MMGRec**：
  - 主结果 `--eval_level sid`：beam 在 SID 词表上搜索；`key_to_sid` 把重复 SID
    映射到同一个 sid ID；同样是 SID-level，不做 item 消歧。
  - 代码另有 `--eval_level item`：trie 按 item 构建，`key_to_item` 对重复 SID
    只保留**第一个出现的 item**（`beam.py:21-23`），相当于“first-occurrence canonical”
    resolver；这不是主结果，也不是 most-popular resolver。
  - 仓库 native SID 的第 4 个 token 是 1..27 的 **pointer**，它本身就是一个
    item-level resolver（3 个 semantic token + pointer → 唯一 item）。但主 benchmark
    使用 shared 4-token SID，**没有启用这个 pointer**；因此主结果仍是 SID-level。
- **CARD**：
  - 1:1 SID（11,244 unique / 11,244 items），生成 token tuple 唯一对应 item；
    SID-level = item-level，不存在歧义。
- **Base/CSGA/OMoE/Full300**：
  - 本次只做了 SID 碰撞分析，没有训练对应的生成器；生成侧指标 N/A。

### 8.2 如果要在部署中映射到最终 item，可选 resolver

1. **Most-popular canonical**：每个 SID bucket 选 train popularity 最高的 item。
   实现简单，但会系统性偏向热门，长尾 target 基本不可能命中。
2. **Expand + rerank**：把预测 SID 展开为 bucket 内所有 item，再用第二个 scorer
   （SASRec/OMoE/文本相似度）排序。理论上最好，但需要额外模型和延迟。
3. **Uniform / random within bucket**：无信息时的期望；用于报告“公平的 item-level 期望”。
4. **Return the bucket as a set**：如果业务允许返回候选集合，则不需要强行消歧；
   但传统 HR/NDCG 需要单 item，必须选 1–3 之一。

### 8.3 SID-level 与 item-level 的差距（实测）

**T8. SID-level vs item-level HR@50**

| Model | SID-level HR@50 | Expected item-level HR@50（uniform resolver） | Singleton-only HR@50（下界） | Relative drop |
|---|---:|---:|---:|---:|
| Full SAM-GR | 5.51% | 4.87% | 4.27% | 11.6% |
| TIGER | 1.99% | 1.81% | 1.63% | 9.1% |
| MMGRec | 3.39% | 3.09% | 2.81% | 8.8% |
| CARD | 6.55% | 6.55% | 6.55% | 0.0% |

- Full 有 **42.84%** 的 test target 落在共享 SID 上；TIGER 42.82%；MMGRec 42.84%；
  CARD 0%。这解释了为什么 Full/TIGER/MMGRec 的 SID-level 指标会高于公平 item-level 期望。
- “Singleton-only”是悲观下界：只有 target SID 为 singleton 且预测命中才算。
  真实 item-level 取决于 resolver，介于 singleton-only 和 SID-level 之间；
  uniform 期望是中间估计。
- **因此，跨模型比较时，CARD 的 6.55% 是 item-level，而 Full 的 5.51% 是 SID-level；
  直接比较会高估 Full。** 用 uniform 期望 4.87% 更公平；用 singleton-only 4.27% 更保守。
  即使按最保守的 4.27%，Full 仍低于 CARD 的 6.55%，但差距比表面数字更大。

---

## 9. 生成侧诊断：invalid / duplicate / ambiguous

**T7. Beam-level generation diagnostics**

| Model | Rows | Invalid SID rate | Duplicate-beam rate | Ambiguous SID prediction rate | Top-1 ambiguous rate | Ambiguous target rate |
|---|---:|---:|---:|---:|---:|---:|
| Full SAM-GR | 25,373 | 0.000% | 0.000% | 15.46% | 5.12% | 42.84% |
| TIGER | 24,819 | 0.000% | 0.000% | 16.00% | 0.00% | 42.82% |
| MMGRec | 25,373 | 0.000% | 0.000% | 12.43% | 0.01% | 42.84% |
| CARD | 25,372 | 0.101% | 0.000% | 0.00% | 0.00% | 0.00% |

**定义：**

- **invalid generated SID rate**：50 个 beam 中，SID 不在 valid SID 集合中的比例。
  Full/TIGER/MMGRec 为 0，因为 trie/词表约束；CARD 无约束 T5，1,281/1,268,600 =
  **0.101%** 非法。
- **duplicate beam rate**：一行 50 个 beam 中存在重复 SID 的行比例。
  四者均为 **0**；beam search 产生的是不同前缀路径，未出现重复。
- **ambiguous SID prediction rate**：50 个 beam 中，SID 对应 bucket size>1 的比例。
  Full **15.46%**、TIGER **16.00%**、MMGRec **12.43%**、CARD **0**。
- **top-1 ambiguous rate**：第一个 beam 的 SID 是共享 SID 的行比例。
  Full **5.12%**、TIGER **0.00%**、MMGRec **0.01%**、CARD **0**。
  TIGER 的 top-1 在 24,819/24,819 行上都是同一个 singleton SID（严重 collapse），所以 top-1 歧义为 0，
  但它的中低 rank beam 仍有 16% 歧义；这不代表 TIGER 的 item 级歧义更小。
- **ambiguous target rate**：test target SID 是共享 SID 的行比例，反映评测的乐观程度。
  Full/TIGER/MMGRec 约 **42.8%**，CARD 0。

**解读：**

- 注意 **ambiguous prediction rate（15.46%）远低于 ambiguous target rate（42.84%）**：
  预测 beam 偏向热门 singleton SID，而 target 是 item-uniform，42.84% 的 target 落在共享 SID。
  两者分母/分布不同，不能混用；前者衡量“生成器输出的歧义”，后者衡量“评测本身的歧义”。
- Full 的 top-1 有 5.12% 直接落在共享 SID 上；这些行的 item 级正确性本质上不可判定，
  只能靠 resolver。
- TIGER 的 top-1 在全部 24,819 行上都 collapse 到同一个 singleton SID，导致 top-1 ambiguous=0，
  但它的 SID-level HR@1 只有 0.17%，说明“无歧义”不等于“正确”。
- CARD 的 0.101% invalid 与 0 ambiguous 是 1:1 SID + 无约束解码的代价/收益：
  它会生成少量非法 SID，但一旦合法就唯一对应 item。

---

## 10. 结果解读、合理性与边界

### 10.1 哪些结论是稳健的？

1. **15% merge 是 Full 14.08% 碰撞的主因。** 这一点由 raw 1.18% → Sinkhorn 0% →
   merge 14.08% 的直接链路证明，且我们复现的 merged index 与保存的 index
   **11,246/11,246 item 完全一致**（`match_rate=1.0`），不是估算。
2. **raw RQ-VAE 碰撞是近重复/强共交互 item；merge 碰撞不是。**
   raw text cos 0.984 / image 0.955 / same-L3 89.6% / co-interaction 30.6%；
   merge 后 0.888 / 0.759 / 31.9% / 3.63%。这是本次分析最重要的质量区分。
3. **Full 的 layer-0 codebook 塌缩严重（69/256），CARD 用满 256/256。**
   prefix 统计是直接证据；depth-2 collision Full 72.63% vs CARD 14.99%。
4. **SID-level 指标对 Full/TIGER/MMGRec 偏乐观。**
   42.84% 的 target 落在共享 SID；uniform item-level 期望比 SID-level 低 9–12% 相对值。
5. **在 300ep 同预算下，CSGA 是主要收益来源，OMoE-only 单独不明显。**
   Base 33.97% / OMoE 33.76% / CSGA 22.33% / Full 17.21%（excess collision）；
   pre-merge co-interaction：Base 2.24% / OMoE 2.15% / CSGA 9.89% / Full 9.84%。
6. **CARD 的 0 full-SID collision 是真实的，但 prefix 仍有碰撞。**
   depth-2 14.99%、depth-3 0.57%；它的优势是层级均衡 + 1:1 最终 SID，
   不是“所有 prefix 都唯一”。

### 10.2 哪些比较不能直接下结论？

1. **300ep 与 5,000ep 不能直接比绝对值。** 300ep 的 RQ-VAE raw collision 72–94%，
   远未收敛；5,000ep 的 Full/TIGER raw collision 只有 1–2%。短预算结果只用于
   **同预算组件趋势**，不能声称“Base/OMoE 生产上一定比 Full 差 2 倍”。
2. **MMGRec 与 Full 的 SID 碰撞完全相同，不能当作独立模型比较。**
   MMGRec 使用 pinned shared SID；它的碰撞指标继承自 Full。
3. **CARD 与 Full 的训练预算、SID 方法、item universe 都不同。**
   CARD 3,000ep NU-RQ-VAE + 50ep T5，Full 5,000ep RQ-VAE + 2ep SFT；
   两者的 HR 差异不能单独归因于 SID 碰撞或模型结构。
4. **merge 是设计选择，不是 bug。** `merge_collaborative_neighbors` 的 docstring 明确说
   它“intentionally merge ... to lower generation difficulty at a controlled cost of
   SID-level ambiguity”。如果目标是降低生成目标数，14% 碰撞是代价；如果目标是
   item-level 部署，这个代价偏高。
5. **text/image cos 的绝对值和 baseline 受 embedding 影响。** 我们用的是
   `text_only` / `image_only` 的 Qwen3-VL embedding；同 category baseline 为
   0.921/0.853，随机为 0.881/0.741。结论“merge 后接近随机、远低于同 category”
   在这个 embedding 空间内成立；换 embedding 可能改变绝对值，但 raw vs merge 的
   巨大差距不太可能消失。

### 10.3 已记录的性能上下文（仅作参考，不作等预算排名）

| Model | Training budget | HR@1 | HR@10 | HR@50 | NDCG@10 | NDCG@50 | Evaluation level |
|---|---|---:|---:|---:|---:|---:|---|
| Full SAM-GR | 2 epochs SFT / 5,000 ep RQ-VAE | 0.0039 | 0.0218 | 0.0551 | 0.0112 | 0.0184 | SID-level |
| TIGER | 10 epochs / 5,000 ep RQ-VAE | 0.0017 | 0.0090 | 0.0199 | 0.0050 | 0.0074 | SID-level |
| MMGRec | recorded best checkpoint | 0.0014 | 0.0132 | 0.0339 | 0.0063 | 0.0109 | SID-level |
| CARD | 50 epochs / 3,000 ep NU-RQ-VAE | 0.0039 | 0.0233 | 0.0655 | 0.0121 | 0.0212 | Item-level (1:1 SID) |

**这张表不能用来做“方法优劣”的等预算结论。** 四个模型的 SFT/生成器训练预算、
SID 构造、item universe 和评测层级都不同：
Full 只有 2 个 SFT epoch（且最后一个 eval 仍在下降），CARD 有 50 个 epoch；
Full/TIGER/MMGRec 是 SID-level，CARD 是 item-level。把 Full 的 5.51% 与 CARD 的
6.55% 直接比较会高估 Full；用 uniform item-level 期望 4.87% 更公平，用 singleton-only
4.27% 更保守。TIGER 的 top-1 在全部 24,819 行上 collapse 到同一个 SID，其 1.99%
HR@50 主要是 beam 多样性带来的，不应解读为 SID 碰撞更轻。

### 10.4 建议

- **如果主指标是 item-level**：把 15% merge 降下来（例如 0–5%），或改用
  “expand + rerank” resolver；同时报告 SID-level 和 item-level 两套指标。
- **如果主指标是 SID-level 生成**：保留 merge 但需要控制 bucket size；当前 max 5
  还可以，但 300ep 的 max 85 不可接受。建议加 bucket-size 正则或 collision-aware
  codebook loss。
- **prefix 层面**：Full 的 layer-0 只有 69 个 code，建议增加 codebook 使用正则/
  k-means init 重训，或采用 CARD 式的层级均衡策略；否则 depth-1/2 的 trie 分支
  过大，beam search 很难命中。
- **报告规范**：所有 SID-based 模型至少报告 unique SID、excess collision、
  collided-item ratio、max bucket、ambiguous target rate、invalid/duplicate beam rate；
  否则 SID-level HR 不可解释。

---

## 11. 复现与产物

### 11.1 关键命令

```bash
cd /data02/sht/MIMIGenRec

# 复现 Full/TIGER/300ep 的 raw -> Sinkhorn -> merge 三阶段
python analysis/sid_collision_sports10/scripts/reproduce_stages.py \
  --name full_samgr \
  --ckpt rq_output/Sports_and_Outdoors/best_collision_model.pth \
  --out_dir analysis/sid_collision_sports10/results/stages \
  --saved_index data/Amazon/Sports_and_Outdoors/Sports_and_Outdoors.index.json

# 全量统计 + 碰撞商品分析 + beam 诊断
python analysis/sid_collision_sports10/scripts/analyze_all.py

# 生成图表/表格
python analysis/sid_collision_sports10/scripts/make_figures.py
python analysis/sid_collision_sports10/scripts/make_tables.py
```

### 11.2 产物清单

| 文件 | 内容 |
|---|---|
| `results/model_summary.csv` | 所有模型 × raw/Sinkhorn/merge 的 T1/T2 指标 |
| `results/prefix_stats.csv` | 所有模型 × depth 1–4 的 T3 |
| `results/bucket_histogram.csv` | 所有模型 × stage 的完整 bucket histogram |
| `results/collided_analysis.json` | 最终 SID 的碰撞 pair 统计 + 三个 baseline |
| `results/collided_stage_analysis.json` | 300ep 的 raw/Sinkhorn/merge 碰撞性质 |
| `results/collided_prod_stage_analysis.json` | Full/TIGER 的 raw/Sinkhorn/merge 碰撞性质 |
| `results/beam_stats.json` | invalid/duplicate/ambiguous + SID/item-level HR |
| `results/model_stats.json` | 原始统计（含 prefix、histogram） |
| `results/mmgrec_native_sid.json` | MMGRec 自带 3-token SID + pointer 的碰撞统计与 pair 分析 |
| `results/tables.md` | 本报告所有表格的 markdown 源 |
| `figures/*.png` | 总览、直方图、prefix、Sinkhorn、碰撞分析、item-level gap |
| `checkpoints/{base,csga,omoe,full300}/` | 300ep RQ-VAE checkpoints |
| `embeddings/omoe_only.npy` | OMoE-only 融合 embedding |

### 11.3 复现一致性验证

- Full：复现的 merged SID 与 `data/Amazon/Sports_and_Outdoors/Sports_and_Outdoors.index.json`
  **11,246/11,246 item 完全一致**。
- TIGER：与 `analysis/samgr_sports10/results/text_only_sid/Sports_and_Outdoors.index.json`
  **11,246/11,246 item 完全一致**。
- OMoE 300ep（已有 `01_full_omoe`）：与保存的 index **11,246/11,246 item 完全一致**
  （使用 `epoch_49_collision_0.7956_model.pth`；该 run 的 `best_collision_model.pth`
  已被后续训练覆盖，但 epoch49 与 `rq_metrics.json` 的 best raw collision 0.79557 一致）。
- 因此本报告的 Full/TIGER 数字不是重新训练后的“新结果”，而是对现有 index 的
  **可复现拆解**。

---

## 附录 A：公式

- `U = |{sid_i}|`
- `excess = (N-U)/N`
- `collided_ratio = (1/N) Σ_g n_g · 1[n_g>1]`
- `P_pair = Σ_g n_g(n_g-1) / (N(N-1))`
- `prefix_unique_d = |{sid_i[:d]}| / N`
- `prefix_collision_d = 1 - prefix_unique_d`
- `expected_item_HR@k = (1/|R|) Σ_{r∈R} Σ_{j≤k, first occurrence} 1[b(sid_j)] · 1[sid_j = target_sid_r]`
  （实现上只需找 target SID 的首次 rank，若 ≤k 则加 `1/b`）
- `singleton_only_HR@k`：仅当 `b(target_sid)=1` 且首次 rank≤k 时计 1。

## 附录 B：原始 stage 统计文件

- `results/stages/*_stages.json`：每个模型的 raw/Sinkhorn/merge 统计与 Sinkhorn 迭代历史。
- `results/stages/*_stages.npz`：每个阶段的完整 SID codes，可直接复算任意指标。
- `results/beams/tiger_beams.npy`、`tiger_targets.npy`：TIGER 50-beam 输出（24,819 行）。
- Full 的 beam 输出：`MIMIGenRec/results/qwen2.5-0.5b/full/Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict/final_result.json`。
- MMGRec：`MMGRec/.../final/test_sid_top50.npy`、`sid_benchmark/sid_ids.npy`、`prepared.npz`。
- CARD：`CARD/results/Sports_and_Outdoors_CARD/test_predictions.npy`、`test_targets.npy`、
  `data/Sports_and_Outdoors/Sports_and_Outdoors_card_codes.npy`。

---

## 附录 C：一句话回答用户的问题

- **unique full-SID count**：Full 9,663；TIGER 9,659；MMGRec 主表 9,663（shared，native 3-token 5,992）；
  CARD 11,244；Base 7,426；CSGA 8,735；OMoE 7,449；Full300 9,311。
- **excess collision**：14.08% / 14.11% / 14.08% / 0 / 33.97% / 22.33% / 33.76% / 17.21%。
- **collided-item ratio**：26.71% / 26.77% / 26.71% / 0 / 49.24% / 36.80% / 49.27% / 31.07%。
- **pairwise collision P**：2.79e-5 / 2.80e-5 / 2.79e-5 / 0 / 3.78e-4 / 9.24e-5 / 2.36e-4 / 4.27e-5。
- **max bucket**：5 / 5 / 5 / 1 / 85 / 31 / 54 / 19。
- **Sinkhorn 前后**：Full 1.18% → 0% → 14.08%；TIGER 2.05% → 0.04% → 14.11%；
  Base 94.41% → 25.51% → 33.97%；CSGA 71.56% → 9.95% → 22.33%；
  OMoE 89.87% → 26.10% → 33.76%；Full300 75.80% → 3.83% → 17.21%。
- **碰撞商品**：raw 是近重复/强共交互（text 0.984、co-int 30.6%）；merge 后接近随机
  （text 0.888、co-int 3.63%，与 popularity-matched 3.54% 持平），且是长尾→热门锚点。
- **多 item 共享 SID**：trie 去重、只生成 SID；SID-level 评测不消歧；
  Full SID-level HR@50 5.51% vs uniform item-level 4.87% vs singleton-only 4.27%。
- **生成诊断**：invalid 0 / 0 / 0 / 0.101%；duplicate 全 0；
  ambiguous prediction 15.46% / 16.00% / 12.43% / 0。
