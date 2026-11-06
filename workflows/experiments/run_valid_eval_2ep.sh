#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

CATEGORY=Sports_and_Outdoors
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/data/sht/conda/mimigenrec/bin/python
EXP=newsaves/qwen2.5-0.5b/full/Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict
TMP=temp/eval-valid-Sports
OUT=results/qwen2.5-0.5b/full/Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict_valid
CUDA_LIST="0 1 3"

rm -rf "$TMP"
mkdir -p "$TMP" "$OUT"

$PY -m samgr.split_json --input_path data/${CATEGORY}/sft/valid.json --output_path "$TMP" --cuda_list "$CUDA_LIST"

seed=42
idx=0
for i in $CUDA_LIST; do
  if [[ -f "$TMP/${i}.json" ]]; then
    echo "Starting valid eval on GPU $i (seed=$((seed + idx)))"
    CUDA_VISIBLE_DEVICES=$i $PY -u -m samgr.evaluate \
      --model_name_or_path "$EXP" \
      --test_data_path "$TMP/${i}.json" \
      --result_json_path "$TMP/${i}_result.json" \
      --index_path data/${CATEGORY}/id2sid.json \
      --batch_size 8 \
      --max_new_tokens 256 \
      --num_beams 50 \
      --temperature 1.0 \
      --do_sample False \
      --length_penalty 0.0 \
      --seed $((seed + idx)) \
      --compute_metrics_flag False > "$TMP/gpu${i}.log" 2>&1 &
  fi
  idx=$((idx + 1))
done
wait

$PY -m samgr.merge_json --input_path "$TMP" --output_path "$OUT/final_result.json" --cuda_list "$CUDA_LIST" --compute_metrics False
$PY -u -m samgr.evaluate --result_json_path "$OUT/final_result.json" --metrics_only True
rm -rf "$TMP"
echo "VALID_EVAL_DONE"
