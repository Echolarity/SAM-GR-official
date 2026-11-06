"""Select stratified random case studies for the neighbourhood analysis."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_items  # noqa: E402

SEED = 2026
N_CASES = 6
N_SHOW = 4


def main():
    items = load_items()
    n = len(items)
    z = np.load(os.path.join(RESULTS, "neighbors_spaces.npz"))
    W = np.load(os.path.join(RESULTS, "exp_f_csga_weights.npz"))
    per = pd.read_csv(os.path.join(RESULTS, "exp_f_csga_per_item_validcollab.csv"))
    pi = np.load(os.path.join(RESULTS, "exp_e_per_item.npz"))
    w_all = W["weight"]

    leaf, l2, has_img = [], [], []
    for i in range(n):
        it = items[str(i)]
        cats = [c.strip() for c in (it.get("categories") or "").split(",") if c.strip()]
        leaf.append(cats[-1] if cats else "Unknown")
        l2.append(cats[2] if len(cats) >= 3 else (cats[-1] if cats else "Unknown"))
        has_img.append(bool(it.get("image_paths")) and os.path.exists(it["image_paths"][0]))
    leaf = np.array(leaf); l2 = np.array(l2); has_img = np.array(has_img)

    valid = per.item.values
    pop = per.set_index("item").pop_users
    group = per.set_index("item").group
    ent = per.set_index("item").attention_entropy_norm
    t1 = per.set_index("item").weight_top1
    collab = z["collab_idx"][:, :10]
    joint = z["joint_idx"][:, :10]
    ov = {int(i): len(np.intersect1d(collab[i, :10], joint[i, :10])) for i in valid}
    cp = pi["collab_k10_cat_purity_leaf"]
    jp = pi["joint_k10_cat_purity_leaf"]

    pool = {}
    for i in valid:
        i = int(i)
        if not has_img[i] or int(pop.loc[i]) < 5:
            continue
        if not all(has_img[j] for j in collab[i, :N_SHOW]):
            continue
        if not all(has_img[j] for j in joint[i, :N_SHOW]):
            continue
        if ov[i] == 0 and cp[i] == 0 and jp[i] >= 0.8 and t1.loc[i] >= 0.15:
            pool.setdefault(l2[i], []).append(i)

    cats_ok = sorted([c for c, v in pool.items() if len(v) >= 3])
    rng = np.random.default_rng(SEED)
    chosen_cats = rng.choice(cats_ok, size=min(N_CASES, len(cats_ok)), replace=False)
    rows = []
    for c in chosen_cats:
        i = int(rng.choice(pool[c]))
        order = np.argsort(-w_all[i])[:N_SHOW]
        cidx = z["collab_idx"][i][order]
        w = w_all[i][order]
        jidx = z["joint_idx"][i, :N_SHOW]
        jcos = z["joint_cos"][i, :N_SHOW]
        rows.append({
            "case": len(rows) + 1, "item": i,
            "title": items[str(i)]["title"], "cat_l2": l2[i], "cat_leaf": leaf[i],
            "group": group.loc[i], "pop_users": int(pop.loc[i]),
            "overlap10": ov[i], "cat_purity_collab": float(cp[i]),
            "cat_purity_semantic": float(jp[i]), "top1_weight": float(t1.loc[i]),
            "entropy_norm": float(ent.loc[i]),
            "collab_ids": " ".join(str(int(x)) for x in cidx),
            "collab_weights": " ".join(f"{float(x):.6f}" for x in w),
            "semantic_ids": " ".join(str(int(x)) for x in jidx),
            "semantic_cos": " ".join(f"{float(x):.6f}" for x in jcos),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp_f_cases.csv"), index=False)
    print("SEED", SEED, "categories", len(cats_ok))
    print(df[["case", "item", "cat_l2", "cat_leaf", "group", "pop_users",
              "overlap10", "top1_weight", "entropy_norm"]].to_string(index=False))
    for _, r in df.iterrows():
        print(r.case, r.item, "|", r.title[:80])


if __name__ == "__main__":
    main()
