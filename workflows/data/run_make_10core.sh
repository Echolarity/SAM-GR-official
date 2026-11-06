#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
LOG="nohup_logs"; mkdir -p "$LOG"
for c in Sports_and_Outdoors Toys_and_Games; do
  if [ -d "data/Amazon/$c" ] && [ ! -d "data/Amazon/${c}_5core_top30k" ]; then
    mv "data/Amazon/$c" "data/Amazon/${c}_5core_top30k"
    echo "backup: data/Amazon/$c -> data/Amazon/${c}_5core_top30k"
  fi
done
python -u data/make_10core_reviews.py Sports_and_Outdoors 10 10 > "$LOG/mk10_Sports.log" 2>&1 & p1=$!
python -u data/make_10core_reviews.py Toys_and_Games 10 10 > "$LOG/mk10_Toys.log" 2>&1 & p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
echo "reviews extraction rc: sports=$r1 toys=$r2"
COMMON="--sample_size -1 --user_k 1 --item_k 1 --output_path ./data/Amazon"
python -u data/possess_data.py --category Sports_and_Outdoors --reviews_file data/Sports_and_Outdoors_10core.jsonl.gz --meta_file data/meta_Sports_and_Outdoors.jsonl.gz --img_dir data/Sports_and_Outdoors_images $COMMON > "$LOG/mk10_Sports_possess.log" 2>&1 & p3=$!
python -u data/possess_data.py --category Toys_and_Games --reviews_file data/Toys_and_Games_10core.jsonl.gz --meta_file data/meta_Toys_and_Games.jsonl.gz --img_dir data/Toys_and_Games_images $COMMON > "$LOG/mk10_Toys_possess.log" 2>&1 & p4=$!
wait $p3; r3=$?; wait $p4; r4=$?
echo "possess rc: sports=$r3 toys=$r4"
echo "ALL_DONE"
