#!/usr/bin/env bash
set -euo pipefail
cd /data02/sht/SAM-GR
PY=/data/sht/conda/mimigenrec/bin/python
ROOT=analysis/sid_collision_sports10
DATA=data/Amazon/Sports_and_Outdoors
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

declare -A EMBS
EMBS[base]="$DATA/Sports_and_Outdoors.emb-Qwen3VL-multimodal.npy"
EMBS[csga]="$ROOT/../samgr_sports10/results/csga_aligned.npy"
EMBS[omoe]="$ROOT/embeddings/omoe_only.npy"
EMBS[full300]="$DATA/Sports_and_Outdoors.emb-Qwen3VL-csga_moe_collab_v2.npy"

for name in base csga omoe full300; do
  out="$ROOT/checkpoints/$name"
  if [ -f "$out/best_collision_model.pth" ]; then
    echo "[skip] $name"
    continue
  fi
  mkdir -p "$out"
  echo "===== training $name ====="
  "$PY" rq/rqvae.py \
    --data_path "${EMBS[$name]}" \
    --ckpt_dir "$out" \
    --lr 1e-3 \
    --epochs 300 \
    --batch_size 20480 \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 0.0 \
    --device cuda:0 > "$out/train.log" 2>&1
  echo "[done] $name"
done
