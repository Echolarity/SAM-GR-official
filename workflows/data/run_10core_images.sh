#!/usr/bin/env bash
# 并行确保 Sports / Toys / Beauty 的 10-core 商品各有一张 MAIN 图 (断点续传, 高并发)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"

CONC="${CONC:-192}"
LOG_DIR="nohup_logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] START 10-core image download (concurrency/dataset=$CONC)"

python -u data/dl_10core_images.py Sports_and_Outdoors 10 10 "$CONC" \
    > "$LOG_DIR/dl10_Sports_and_Outdoors.log" 2>&1 &
p1=$!
python -u data/dl_10core_images.py Toys_and_Games 10 10 "$CONC" \
    > "$LOG_DIR/dl10_Toys_and_Games.log" 2>&1 &
p2=$!
python -u data/dl_10core_images.py Beauty_and_Personal_Care 10 10 "$CONC" \
    > "$LOG_DIR/dl10_Beauty_and_Personal_Care.log" 2>&1 &
p3=$!

echo "started: sports=$p1 toys=$p2 beauty=$p3"
wait $p1; r1=$?
wait $p2; r2=$?
wait $p3; r3=$?

echo "[$(date '+%F %T')] ALL DONE rc: sports=$r1 toys=$r2 beauty=$r3"
