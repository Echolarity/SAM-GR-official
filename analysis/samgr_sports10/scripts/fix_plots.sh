#!/usr/bin/env bash
set -uo pipefail
cd /data02/sht/SAM-GR
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
PY=/data/sht/conda/mimigenrec/bin/python
S=analysis/samgr_sports10/scripts
CUDA_VISIBLE_DEVICES=2 $PY $S/exp_a_neighborhood.py > analysis/samgr_sports10/results/exp_a.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY $S/exp_d_omoe.py > analysis/samgr_sports10/results/exp_d.log 2>&1
echo FIX_PLOTS_DONE
