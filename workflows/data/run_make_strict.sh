#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
LOG="nohup_logs"; mkdir -p "$LOG"

python -u data/make_strict_kcore.py Sports_and_Outdoors 10 > "$LOG/mkstrict_Sports_reviews.log" 2>&1 & p1=$!
python -u data/make_strict_kcore.py Toys_and_Games 10         > "$LOG/mkstrict_Toys_reviews.log" 2>&1 & p2=$!
python -u data/make_strict_kcore.py Beauty_and_Personal_Care 13 > "$LOG/mkstrict_Beauty_reviews.log" 2>&1 & p3=$!
wait $p1; r1=$?; wait $p2; r2=$?; wait $p3; r3=$?
echo "reviews rc: sports=$r1 toys=$r2 beauty=$r3"

COMMON="--sample_size -1 --user_k 1 --item_k 1 --output_path ./data/Amazon"
python -u data/possess_data.py --category Sports_and_Outdoors --reviews_file data/Sports_and_Outdoors_strict10core.jsonl.gz --meta_file data/meta_Sports_and_Outdoors.jsonl.gz --img_dir data/Sports_and_Outdoors_images $COMMON > "$LOG/mkstrict_Sports_possess.log" 2>&1 & p4=$!
python -u data/possess_data.py --category Toys_and_Games --reviews_file data/Toys_and_Games_strict10core.jsonl.gz --meta_file data/meta_Toys_and_Games.jsonl.gz --img_dir data/Toys_and_Games_images $COMMON > "$LOG/mkstrict_Toys_possess.log" 2>&1 & p5=$!
python -u data/possess_data.py --category Beauty_and_Personal_Care --reviews_file data/Beauty_and_Personal_Care_strict13core.jsonl.gz --meta_file data/meta_Beauty_and_Personal_Care.jsonl.gz --img_dir data/Beauty_and_Personal_Care_images $COMMON > "$LOG/mkstrict_Beauty_possess.log" 2>&1 & p6=$!
wait $p4; s1=$?; wait $p5; s2=$?; wait $p6; s3=$?
echo "possess rc: sports=$s1 toys=$s2 beauty=$s3"
echo ALL_DONE
