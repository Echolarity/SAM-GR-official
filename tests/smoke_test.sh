#!/usr/bin/env bash
# =============================================================================
# End-to-end smoke test for the SAM-GR workflow.
#
# Runs the real data-possession, SFT/RL preprocessing, and split/merge code on a
# tiny synthetic dataset. GPU/model-heavy stages (multimodal embedding, RQ-VAE,
# SID index generation, SFT) are replaced by deterministic stubs so the full
# workflows/main.sh control flow executes in seconds without a GPU.
#
# Usage:
#   bash tests/smoke_test.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_ROOT="${SMOKE_ROOT:-$REPO_ROOT/temp/smoke_test}"
CATEGORY="${SMOKE_CATEGORY:-SmokeTest}"

REAL_PY="${SMOKE_REAL_PYTHON:-}"
if [ -z "$REAL_PY" ]; then
  if [ -x /data/sht/conda/mimigenrec/bin/python ]; then
    REAL_PY=/data/sht/conda/mimigenrec/bin/python
  else
    REAL_PY="$(command -v python3)"
  fi
fi

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"/raw "$SMOKE_ROOT"/amazon "$SMOKE_ROOT"/data \
         "$SMOKE_ROOT"/rq_output "$SMOKE_ROOT"/model "$SMOKE_ROOT"/modelout

RAW="$SMOKE_ROOT/raw"
IMG="$RAW/${CATEGORY}_images"

echo "=== [1/3] Generate synthetic raw data ==="
"$REAL_PY" tests/make_smoke_data.py \
  --out-dir "$RAW" --category "$CATEGORY" --users 6 --items 12

# Minimal SFT config: only checked for existence because llamafactory-cli is stubbed.
cat > "$SMOKE_ROOT/smoke_sft.yaml" <<YAML
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
stage: sft
do_train: true
output_dir: $SMOKE_ROOT/modelout
YAML

export PATH="$REPO_ROOT/tests/stubs:$PATH"
export SMOKE_REAL_PYTHON="$REAL_PY"
export CATEGORY
export RAW_DATA_DIR="$RAW"
export IMG_DIR="$IMG"
export POSSESS_DATA_OUTPATH="$SMOKE_ROOT/amazon"
export OUTPUT_DIR="$SMOKE_ROOT/data"
export RQVAE_DIR="$SMOKE_ROOT/rq_output"
export SASREC_SAVE_DIR="$SMOKE_ROOT/sasrec_models"
export TEXT2EMB_MODEL="$SMOKE_ROOT/model"
export SFT_YAML="$SMOKE_ROOT/smoke_sft.yaml"
export EVAL_EXP_NAME="$SMOKE_ROOT/modelout"
export MODAL_CHOICE=text_only
export USER_K=1 ITEM_K=1 SAMPLE_SIZE=-1 ONLY_TASK1=1 START_STAGE=1
export CUDA_VISIBLE=0 MONO_CUDA=0
export EVAL_CUDA_LIST="0 1"
export Optimization=smoke
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1

echo "=== [2/3] Run workflows/main.sh (GPU stages stubbed) ==="
bash workflows/main.sh

echo "=== [3/3] Assert artifacts and real split/merge plumbing ==="
test -f "$POSSESS_DATA_OUTPATH/$CATEGORY/$CATEGORY.item.json"
test -f "$POSSESS_DATA_OUTPATH/$CATEGORY/$CATEGORY.train.inter"
test -f "$OUTPUT_DIR/new_tokens.json"
test -f "$OUTPUT_DIR/id2sid.json"
test -f "$OUTPUT_DIR/sft/train.json"
test -f "$OUTPUT_DIR/sft/valid.json"
test -f "$OUTPUT_DIR/sft/test.json"
test -f "$EVAL_EXP_NAME/config.json"

EVAL_REAL="$SMOKE_ROOT/eval_real"
mkdir -p "$EVAL_REAL"
"$REAL_PY" -m samgr.split_json \
  --input_path "$OUTPUT_DIR/sft/test.json" \
  --output_path "$EVAL_REAL" --cuda_list "0 1"
test -f "$EVAL_REAL/0.json"
test -f "$EVAL_REAL/1.json"
cp "$EVAL_REAL/0.json" "$EVAL_REAL/0_result.json"
cp "$EVAL_REAL/1.json" "$EVAL_REAL/1_result.json"
"$REAL_PY" -m samgr.merge_json \
  --input_path "$EVAL_REAL" --output_path "$EVAL_REAL/merged.json" \
  --cuda_list "0 1" --compute_metrics False
test -f "$EVAL_REAL/merged.json"
"$REAL_PY" - "$OUTPUT_DIR/sft/test.json" "$EVAL_REAL/merged.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    original = json.load(f)
with open(sys.argv[2], encoding="utf-8") as f:
    merged = json.load(f)
assert len(original) == len(merged), (len(original), len(merged))
print(f"[smoke] split/merge preserved {len(merged)} rows")
PY

echo "SMOKE TEST PASSED"
echo "Artifacts: $SMOKE_ROOT"
