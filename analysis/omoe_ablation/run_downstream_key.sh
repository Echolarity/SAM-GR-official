#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VARIANTS="${VARIANTS:-01_full_omoe 03_wo_load 11_no_direct 13_no_residual 12_no_experts}"
STEPS="${STEPS:-400}"
for v in $VARIANTS; do
  echo "[$(date '+%F %T')] KEY $v steps=$STEPS START" >> logs/downstream_key.log
  python run_downstream.py --variant "$v" --sft-max-steps "$STEPS" \
    --sft-eval-steps "$STEPS" --sft-save-steps "$STEPS" \
    --gpus 0,1,2,3 --eval-gpus 0,1,2,3 --eval-batch 16 --eval-beams 20 \
    --stages sft,eval,collect >> logs/downstream_key.log 2>&1
  echo "[$(date '+%F %T')] KEY $v RC=$?" >> logs/downstream_key.log
done
echo "DOWNSTREAM_KEY_DONE" >> logs/downstream_key.log
