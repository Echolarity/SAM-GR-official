#!/usr/bin/env bash
# TIGER (text-only SID) baseline on strict 10-core Sports, 2 GPUs.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"

CAT="Sports_and_Outdoors"
PORT=$((20000 + RANDOM % 40000))

echo "[$(date '+%F %T')] TIGER start: category=$CAT"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="$PORT" \
  scripts/tiger/train_tiger.py \
  --category "$CAT" \
  --data_dir ./data \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --beams 50 \
  --out_dir experiments_tiger
echo "[$(date '+%F %T')] TIGER_DONE"
