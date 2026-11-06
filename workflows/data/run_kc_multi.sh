#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
mkdir -p nohup_logs
for c in Sports_and_Outdoors Toys_and_Games Beauty_and_Personal_Care; do
  python -u data/kc_multi.py "$c" > "nohup_logs/kc_multi_${c}.log" 2>&1 &
done
wait
echo ALL_DONE
