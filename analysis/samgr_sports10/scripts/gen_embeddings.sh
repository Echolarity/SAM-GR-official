#!/usr/bin/env bash
set -uo pipefail
cd /data02/sht/SAM-GR
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
ROOT="data/Amazon/Sports_and_Outdoors"
for mode in image_only multimodal; do
  out="$ROOT/Sports_and_Outdoors.emb-Qwen3VL-${mode}.npy"
  if [ -f "$out" ]; then echo "skip $mode (exists)"; continue; fi
  echo "[$(date '+%F %T')] generating $mode"
  PORT=$((20000 + RANDOM % 40000))
  CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port="$PORT" \
    rq/text2emb/multimodal_amazon_text2emb.py --dataset Sports_and_Outdoors --root "$ROOT" \
    --plm_checkpoint /data02/qy/models/Qwen3-VL-8B-Instruct --plm_name Qwen3VL \
    --mode "$mode" --batch_size 4 --max_sent_len 1024 --moe_epochs 30 --text_weight -1.0 || exit 1
done
echo "GEN_EMB_DONE"
