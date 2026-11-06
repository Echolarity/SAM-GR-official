#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
CAT=Sports_and_Outdoors
ROOT="./data/Amazon/$CAT"
OUT="./data/$CAT"
PORT=$((20000 + RANDOM % 40000))
echo "[$(date '+%F %T')] stage3 text2emb text_only"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT \
  rq/text2emb/multimodal_amazon_text2emb.py --dataset "$CAT" --root "$ROOT" \
  --plm_checkpoint /data02/qy/models/Qwen3-VL-8B-Instruct --plm_name Qwen3VL \
  --mode text_only --batch_size 4 --max_sent_len 1024 --moe_epochs 30 --text_weight -1.0 || exit 1
echo "[$(date '+%F %T')] stage4 RQ-VAE (text_only)"
CUDA_VISIBLE_DEVICES=0 python rq/rqvae.py --data_path "$ROOT/${CAT}.emb-Qwen3VL-text_only.npy" \
  --ckpt_dir "./rq_output/tiger_${CAT}" --lr 1e-3 --epochs 5000 --batch_size 20480 \
  --num_emb_list 256 256 256 256 --sk_epsilons 0.0 0.0 0.0 0.0 || exit 1
echo "[$(date '+%F %T')] stage5 generate_indices (merge15)"
rm -f "$ROOT"/*.index.json
CUDA_VISIBLE_DEVICES=0 python rq/generate_indices.py --dataset "$CAT" \
  --ckpt_path "./rq_output/tiger_${CAT}/best_collision_model.pth" --output_dir "$ROOT" --device cuda:0 \
  --collab_emb_path "./sasrec_models/$CAT/sasrec_item_emb.npy" \
  --train_inter "$ROOT/${CAT}.train.inter" --merge_ratio 0.15 || exit 1
echo "[$(date '+%F %T')] stage6 preprocess (only task1)"
python -m samgr.preprocess --data_dir ./data/Amazon --category "$CAT" --output_dir "$OUT" \
  --seq_sample -1 --seed 42 --data_source "$CAT" --sid_num 4 --only_task1 True || exit 1
echo "TIGER_SID_DONE"
