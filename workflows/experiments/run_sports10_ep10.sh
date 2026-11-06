#!/usr/bin/env bash
# Sports strict-10-core main experiment: 10-epoch budget, 4 GPUs, early stopping.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

OUT="newsaves/qwen2.5-0.5b/full/Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_ep10"
if [ -f "$OUT/trainer_state.json" ]; then
  echo "[error] $OUT already has trainer_state.json; refusing to overwrite a previous run." >&2
  exit 2
fi

CATEGORY=Sports_and_Outdoors \
MODAL_CHOICE=csga_moe_collab_v2 \
USER_K=1 ITEM_K=1 \
ONLY_TASK1=1 \
Optimization=dsz2_csga_omoe_v2_strict_ep10 \
SID_MERGE_RATIO=0.15 \
CUDA_VISIBLE=0,1,2,3 \
MONO_CUDA=0 \
EVAL_CUDA_LIST='0 1 2 3' \
START_STAGE=7 \
bash workflows/main.sh

echo "SPORTS10_EP10_DONE rc=$?"
