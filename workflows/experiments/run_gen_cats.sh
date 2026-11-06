#!/usr/bin/env bash
# =============================================================================
# Sports_and_Outdoors + Toys_and_Games 顺序跑生成式最强版 (CSGA+OMoE v2 + merge15):
#   阶段2 SASRec(collab) -> 阶段3 csga_moe_collab_v2 -> 阶段4 RQ-VAE -> 阶段5 merge15
#   -> 阶段6 task1 数据 -> 阶段7 SFT(2ep) + 评估
# 协议与 Books_top1500 / IndSci_csga_omoe_v2 完全一致, 数据为活跃用户抽样(Industry规模).
# 用法: bash workflows/nohup_run.sh run gen_sports_toys bash workflows/experiments/run_gen_cats.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

for CAT in Sports_and_Outdoors Toys_and_Games; do
  echo "############################################################"
  echo "########## [$CAT] 生成式最强版流水线 ##########"
  echo "############################################################"
  PATH="/data/sht/conda/mimigenrec/bin:$PATH" \
  CATEGORY="$CAT" \
  MODAL_CHOICE=csga_moe_collab_v2 \
  USER_K=1 ITEM_K=1 \
  ONLY_TASK1=1 \
  Optimization=dsz2_csga_omoe_v2 \
  SID_MERGE_RATIO=0.15 \
  CUDA_VISIBLE=2,3 \
  MONO_CUDA=2 \
  START_STAGE=2 \
  bash workflows/main.sh \
    && echo "########## [$CAT] 成功 ##########" \
    || echo "########## [$CAT] 失败(继续下一类别) ##########"
done

echo "ALL_DONE"
