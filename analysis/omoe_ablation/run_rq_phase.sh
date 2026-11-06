#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

worker () {
  local gpu="$1"; shift
  for v in "$@"; do
    echo "[$(date '+%F %T')] gpu=$gpu variant=$v START" >> "logs/rq_phase_gpu${gpu}.log"
    python run_downstream.py --variant "$v" --rq-epochs 1000 \
      --gpus "$gpu" --stages rq,index,preprocess --skip-existing \
      >> "logs/rq_phase_gpu${gpu}.log" 2>&1
    echo "[$(date '+%F %T')] gpu=$gpu variant=$v RC=$?" >> "logs/rq_phase_gpu${gpu}.log"
  done
}

worker 0 02_wo_ortho 06_wo_rec 10_only_rec_load 14_static_gate 18_concat_mlp &
p0=$!
worker 1 03_wo_load 07_wo_all_aux 11_no_direct 15_uniform_gate 19_plain_moe &
p1=$!
worker 2 04_wo_dup 08_only_rec 12_no_experts 16_shared_only &
p2=$!
worker 3 05_wo_var 09_only_rec_ortho 13_no_residual 17_modality_only 01_full_omoe &
p3=$!
wait $p0; r0=$?
wait $p1; r1=$?
wait $p2; r2=$?
wait $p3; r3=$?
echo "RQ_PHASE_DONE rc=$r0,$r1,$r2,$r3"
