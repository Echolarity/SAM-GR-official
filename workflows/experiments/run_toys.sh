#!/usr/bin/env bash
# Toys_and_Games 阶段5-7 (SASRec/emb/RQ-VAE 已完成), 在 GPU0/1 上与 Sports(GPU2/3) 并行.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

PATH="/data/sht/conda/mimigenrec/bin:$PATH" \
CATEGORY=Toys_and_Games \
MODAL_CHOICE=csga_moe_collab_v2 \
USER_K=1 ITEM_K=1 \
ONLY_TASK1=1 \
Optimization=dsz2_csga_omoe_v2 \
SID_MERGE_RATIO=0.15 \
CUDA_VISIBLE=0,1 \
MONO_CUDA=0 \
EVAL_CUDA_LIST='0 1' \
START_STAGE=5 \
bash workflows/main.sh
echo "TOYS_DONE rc=$?"
