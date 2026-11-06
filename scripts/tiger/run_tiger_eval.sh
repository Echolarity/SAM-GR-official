#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 python -u scripts/tiger/eval_tiger.py \
  --category Sports_and_Outdoors --device cuda:0 --beams 50 \
  --ckpt experiments_tiger/tiger_Sports_and_Outdoors.pt \
  --out experiments_tiger/metrics_tiger_Sports_and_Outdoors.json
echo "TIGER_EVAL_DONE"
