#!/usr/bin/env bash
# Reproduce all SAM-GR motivational experiments on Sports strict 10-core.
set -uo pipefail
cd /data02/sht/SAM-GR
export PATH="/data/sht/conda/mimigenrec/bin:$PATH"
PY=/data/sht/conda/mimigenrec/bin/python
S=analysis/samgr_sports10/scripts
$PY $S/exp_a_neighborhood.py  | tee analysis/samgr_sports10/results/exp_a.log
$PY $S/exp_b_modality.py      | tee analysis/samgr_sports10/results/exp_b.log
$PY $S/exp_c_csga.py          | tee analysis/samgr_sports10/results/exp_c.log
$PY $S/exp_d_omoe.py          | tee analysis/samgr_sports10/results/exp_d.log
echo ALL_EXPERIMENTS_DONE
