#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

shard () {
  local gpu="$1"; shift
  local list="$1"
  CUDA_VISIBLE_DEVICES="$gpu" python run_representation.py \
      --gpu 0 --only "$list" --epochs 60 --force \
      >> "logs/representation_gpu${gpu}.log" 2>&1
}

shard 0 "01,05,09,13,17" &
p0=$!
shard 1 "02,06,10,14,18" &
p1=$!
shard 2 "03,07,11,15,19" &
p2=$!
shard 3 "04,08,12,16" &
p3=$!
wait $p0; r0=$?
wait $p1; r1=$?
wait $p2; r2=$?
wait $p3; r3=$?
echo "REPRESENTATION_ALL_DONE rc=$r0,$r1,$r2,$r3"
