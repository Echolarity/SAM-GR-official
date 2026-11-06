#!/usr/bin/env bash
# Resume the IndSci image-only run at stage 7 (RQ/SID/preprocess already done).
# GPU 0 is occupied by another user; use GPUs 1,2,3.
# 3 GPUs x 16 x 21 = 1008 effective batch (vs mm 4 x 16 x 16 = 1024).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

export CATEGORY=Industrial_and_Scientific
export MODAL_CHOICE=image_only
export Optimization=dsz2_image_only
export SFT_YAML="examples/train_full/${CATEGORY}/new_industry_rec_full_sft_0.5b_${Optimization}.yaml"
export EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-${Optimization}"

export ONLY_TASK1=1
export USER_K=5
export ITEM_K=5
export SID_MERGE_RATIO=0
export START_STAGE=7
export CUDA_VISIBLE=1,2,3
export MONO_CUDA=1
export EVAL_CUDA_LIST='1 2 3'

bash workflows/main.sh
