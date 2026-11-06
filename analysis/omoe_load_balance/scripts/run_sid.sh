#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
PY=/data/sht/conda/mimigenrec/bin/python
ROOT="$PWD"
PROD_RQ=/data02/sht/SAM-GR/rq_output/Sports_and_Outdoors/best_collision_model.pth
OUT="$ROOT/results/sid/prod_rq"
mkdir -p "$OUT" logs

worker () {
  local gpu="$1"; shift
  for f in "$@"; do
    local name
    name=$(basename "$f" _z.npy)
    if [ -f "$OUT/$name/metrics.json" ]; then continue; fi
    echo "[$(date '+%F %T')] gpu=$gpu $name START"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/make_sid.py \
      --data_path "$f" --ckpt "$PROD_RQ" --out_dir "$OUT/$name" --name "$name" \
      --device cuda:0 --merge_ratio 0.15 \
      > "logs/sid_${name}.log" 2>&1
    echo "[$(date '+%F %T')] gpu=$gpu $name RC=$?"
  done
}

mapfile -t ZS < <(ls "$ROOT"/results/analysis/*_z.npy 2>/dev/null)
echo "total ${#ZS[@]} representations"
part=$(( (${#ZS[@]} + 3) / 4 ))
worker 0 "${ZS[@]:0:$part}" &
p0=$!
worker 1 "${ZS[@]:$part:$part}" &
p1=$!
worker 2 "${ZS[@]:$((2*part)):$part}" &
p2=$!
worker 3 "${ZS[@]:$((3*part))}" &
p3=$!
wait $p0; wait $p1; wait $p2; wait $p3
echo "SID_DONE"
