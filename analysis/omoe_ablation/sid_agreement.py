"""Pairwise SID-partition agreement between ablation variants and Full."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "workspaces")
RESULTS = os.path.join(HERE, "results")

FULL = os.path.join(WORK, "01_full_omoe", "index", "Sports_and_Outdoors.index.json")


def load_sid(path):
    d = json.load(open(path))
    ids = sorted(d, key=lambda x: int(x))
    return [d[i] for i in ids]


def main():
    full = load_sid(FULL)
    n = len(full)
    rows = []
    for p in sorted(glob.glob(os.path.join(WORK, "*", "index", "Sports_and_Outdoors.index.json"))):
        v = os.path.basename(os.path.dirname(os.path.dirname(p)))
        cur = load_sid(p)
        if len(cur) != n:
            continue
        row = {"variant": v}
        for l in range(1, 5):
            a = ["|".join(x[:l]) for x in full]
            b = ["|".join(x[:l]) for x in cur]
            row[f"ARI_prefix{l}"] = adjusted_rand_score(a, b)
            row[f"NMI_prefix{l}"] = normalized_mutual_info_score(a, b)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("variant")
    df.to_csv(os.path.join(RESULTS, "sid_agreement_vs_full.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
