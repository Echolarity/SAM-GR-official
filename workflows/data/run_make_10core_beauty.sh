#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
LOG="nohup_logs"; mkdir -p "$LOG"
python -u data/make_10core_reviews.py Beauty_and_Personal_Care 10 10 > "$LOG/mk10_Beauty.log" 2>&1
r1=$?
python -u data/possess_data.py --category Beauty_and_Personal_Care \
  --reviews_file data/Beauty_and_Personal_Care_10core.jsonl.gz \
  --meta_file data/meta_Beauty_and_Personal_Care.jsonl.gz \
  --img_dir data/Beauty_and_Personal_Care_images \
  --sample_size -1 --user_k 1 --item_k 1 --output_path ./data/Amazon > "$LOG/mk10_Beauty_possess.log" 2>&1
r2=$?
echo "ALL_DONE reviews_rc=$r1 possess_rc=$r2"
