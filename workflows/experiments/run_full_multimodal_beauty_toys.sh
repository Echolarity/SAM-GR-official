#!/usr/bin/env bash
# Full direct-multimodal encoding for Beauty13 / Toys10.
# Beauty on GPU 1, Toys on GPU 3, in parallel. Qwen3-VL inference only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p nohup_logs
BEAUTY_LOG="nohup_logs/full_multimodal_beauty.log"
TOYS_LOG="nohup_logs/full_multimodal_toys.log"
: > "$BEAUTY_LOG"
: > "$TOYS_LOG"

run_one() {
  local cat="$1" gpu="$2" log="$3" port="$4"
  echo "################ FULL MULTIMODAL: ${cat} on GPU ${gpu} ################" >> "$log"
  CUDA_VISIBLE_DEVICES="${gpu}" torchrun --nproc_per_node=1 --master_port="${port}" \
    rq/text2emb/multimodal_amazon_text2emb.py \
    --dataset "${cat}" \
    --root "data/Amazon/${cat}" \
    --plm_checkpoint /data02/qy/models/Qwen3-VL-8B-Instruct \
    --plm_name Qwen3VL \
    --mode multimodal \
    --batch_size 8 \
    --max_sent_len 1024 >> "$log" 2>&1
  local rc=$?
  echo "################ ${cat} rc=${rc} ################" >> "$log"
  return $rc
}

echo "=== Beauty13 on GPU1, Toys10 on GPU3; start $(date) ==="
run_one Beauty_and_Personal_Care 1 "$BEAUTY_LOG" 29581 &
p1=$!
run_one Toys_and_Games 3 "$TOYS_LOG" 29582 &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "=== finished $(date); beauty_rc=${r1} toys_rc=${r2} ==="
if [ $r1 -ne 0 ] || [ $r2 -ne 0 ]; then
  echo "FULL_MULTIMODAL_FAILED"
  exit 1
fi
echo "FULL_MULTIMODAL_DONE"
