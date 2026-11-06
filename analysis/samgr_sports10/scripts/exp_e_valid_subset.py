"""Recompute neighbourhood metrics on items with a non-zero SASRec embedding.

599 / 11,246 items never occur in the training split and therefore have an
all-zero collaborative embedding.  Their collaborative cosine is undefined
(0 for every other item), so the primary analysis excludes them.  Text/image/
joint neighbours are recomputed on the same item subset for comparability.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_collab, load_items  # noqa: E402

SPACES = ["collab", "text", "image", "joint"]
PAIRS = [("collab", "text"), ("collab", "image"), ("collab", "joint"),
         ("text", "image"), ("text", "joint"), ("image", "joint")]
KS = [10, 20, 50]
GROUPS = ["head", "mid", "tail"]


def main():
    items = load_items()
    n = len(items)
    collab = load_collab()
    valid = np.linalg.norm(collab, axis=1) > 1e-6
    print("valid", int(valid.sum()), "excluded", int((~valid).sum()))

    pi = np.load(os.path.join(RESULTS, "exp_e_per_item.npz"))
    old = pd.read_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics.csv"))
    group_map = old[old.group != "all"].drop_duplicates("space").set_index("space")  # placeholder
    # group labels are identical across spaces; recover from the original per-item table
    per = pd.read_csv(os.path.join(RESULTS, "exp_f_csga_per_item.csv"))
    group = per["group"].values

    rows = []
    for s in SPACES:
        for k in KS:
            key = f"{s}_k{k}"
            for g in ["all"] + GROUPS:
                m = valid if g == "all" else (valid & (group == g))
                row = {"space": s, "k": k, "group": g, "n_items": int(m.sum())}
                names = ["cat_purity_leaf", "cat_purity_l2", "cat_majority_leaf",
                         "cooc_prob", "cooc_cov", "trans_prob", "trans_cov",
                         "npmi", "npmi_cov", "npmi_smooth", "text_cos", "image_cos"]
                for name in names:
                    v = pi[f"{key}_{name}"][m]
                    row[name] = float(np.nanmean(v))
                    row[name + "_std"] = float(np.nanstd(v))
                for ctrl in ["uniform", "pop"]:
                    for name in ["cat_purity_leaf", "cooc_prob", "trans_prob", "npmi",
                                 "text_cos", "image_cos"]:
                        v = pi[f"{key}_rand_{ctrl}_{name}"][m]
                        row[f"rand_{ctrl}_{name}"] = float(np.nanmean(v))
                    row[f"delta_{ctrl}_cat_purity_leaf"] = row["cat_purity_leaf"] - row[f"rand_{ctrl}_cat_purity_leaf"]
                    row[f"delta_{ctrl}_text_cos"] = row["text_cos"] - row[f"rand_{ctrl}_text_cos"]
                    row[f"delta_{ctrl}_image_cos"] = row["image_cos"] - row[f"rand_{ctrl}_image_cos"]
                    row[f"lift_{ctrl}_cooc_prob_agg"] = row["cooc_prob"] / max(row[f"rand_{ctrl}_cooc_prob"], 1e-12)
                    row[f"lift_{ctrl}_trans_prob_agg"] = row["trans_prob"] / max(row[f"rand_{ctrl}_trans_prob"], 1e-12)
                    row[f"lift_{ctrl}_npmi_delta"] = row["npmi"] - row[f"rand_{ctrl}_npmi"]
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_e_neighborhood_metrics_validcollab.csv"), index=False)

    # ---------- pair overlap on the same subset ----------
    z = np.load(os.path.join(RESULTS, "neighbors_spaces.npz"))
    NB = {s: z[f"{s}_idx"] for s in SPACES}
    vidx = np.where(valid)[0]
    ov_rows = []
    for a, b in PAIRS:
        A, B = NB[a], NB[b]
        for k in KS:
            inter = np.fromiter((len(np.intersect1d(A[i, :k], B[i, :k], assume_unique=True))
                                 for i in vidx), dtype=np.int16, count=len(vidx))
            union = 2 * k - inter
            jac = inter / np.maximum(union, 1)
            rec = inter / k
            ov_rows.append({
                "space_a": a, "space_b": b, "pair": f"{a}-{b}", "k": k,
                "jaccard_mean": float(jac.mean()), "jaccard_std": float(jac.std()),
                "jaccard_median": float(np.median(jac)),
                "recall_overlap_mean": float(rec.mean()), "recall_overlap_std": float(rec.std()),
                "recall_overlap_median": float(np.median(rec)),
                "zero_frac": float((inter == 0).mean()), "n_items": int(len(vidx)),
            })
    pd.DataFrame(ov_rows).to_csv(os.path.join(RESULTS, "exp_e_pair_overlap_validcollab.csv"), index=False)
    with open(os.path.join(RESULTS, "exp_e_validcollab_stats.json"), "w") as f:
        json.dump({"n_items": int(valid.sum()), "n_excluded_zero_collab": int((~valid).sum())}, f, indent=2)
    print(df[df.group == "all"].to_string(index=False))
    print("EXP_E_VALID_DONE")


if __name__ == "__main__":
    main()
