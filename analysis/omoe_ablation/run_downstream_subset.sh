#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VARIANTS="${VARIANTS:-01_full_omoe 03_wo_load 07_wo_all_aux 11_no_direct 12_no_experts 13_no_residual 15_uniform_gate 18_concat_mlp}"
STEPS="${STEPS:-200}"
for v in $VARIANTS; do
  echo "[$(date '+%F %T')] DOWNSTREAM $v steps=$STEPS START" >> logs/downstream_subset.log
  python run_downstream.py --variant "$v" --sft-max-steps "$STEPS" \
    --sft-eval-steps "$STEPS" --sft-save-steps "$STEPS" \
    --gpus 0,1,2,3 --eval-gpus 0,1,2,3 --eval-batch 16 --eval-beams 20 \
    --stages sft,eval,collect >> logs/downstream_subset.log 2>&1
  echo "[$(date '+%F %T')] DOWNSTREAM $v RC=$?" >> logs/downstream_subset.log
done
echo "DOWNSTREAM_SUBSET_DONE" >> logs/downstream_subset.log
