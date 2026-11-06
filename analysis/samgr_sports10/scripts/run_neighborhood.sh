#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/data/sht/conda/mimigenrec/bin/python}

# exp_e writes neighbors_spaces.npz and per-item neighbourhood metrics.
$PY scripts/exp_e_topk_neighborhoods.py
# exp_f recovers CSGA weights and writes the popularity-group labels.
$PY scripts/exp_f_csga_weights.py
# Recompute both analyses on the non-zero-collab item subset.
$PY scripts/exp_e_valid_subset.py
$PY scripts/exp_f_csga_valid_subset.py
# Cases and paper figures.
$PY scripts/select_cases.py
$PY scripts/make_case_study_figure.py
$PY scripts/make_neighborhood_figures.py
