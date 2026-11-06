"""CSGA weight statistics on items with a non-zero SASRec embedding.

All 10 neighbours of a valid anchor are also valid (zero-embedding items never
enter a valid anchor's Top-10), so only anchor filtering is needed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_collab, load_emb, load_items  # noqa: E402
from exp_e_topk_neighborhoods import (build_graphs, category_arrays,  # noqa: E402
                                      load_train_sequences, popularity_groups)

K = 10
TOP_M = 5


def main():
    items = load_items()
    n = len(items)
    collab = load_collab()
    valid = np.linalg.norm(collab, axis=1) > 1e-6
    W = np.load(os.path.join(RESULTS, "exp_f_csga_weights.npz"))
    w_all, idx_all, cos_all = W["weight"], W["idx"], W["collab_cos"]
    E = np.load(os.path.join(RESULTS, "exp_f_csga_edges.npz"))
    anchor_all = E["anchor"]; neigh_all = E["neighbor"]
    mask = valid[anchor_all]
    anchor = anchor_all[mask]; neigh = neigh_all[mask]
    wf = E["weight"][mask].astype(np.float64); cf = E["collab_cos"][mask].astype(np.float64)
    text_cos = E["text_cos"][mask].astype(np.float64)
    image_cos = E["image_cos"][mask].astype(np.float64)
    p_trans = E["trans_prob"][mask].astype(np.float64)
    p_cooc = E["cooc_prob"][mask].astype(np.float64)
    npmi = E["npmi"][mask].astype(np.float64)

    seqs, user_items = load_train_sequences()
    C, T, T_out, n_user_i, n_users = build_graphs(seqs, user_items, n)
    leaf, l2, leaf_names, l2_names = category_arrays(items, n)
    groups, _ = popularity_groups(n_user_i)

    valid_ids = np.where(valid)[0]
    w = w_all[valid_ids]; idx = idx_all[valid_ids]; cos = cos_all[valid_ids]
    nv = len(valid_ids)

    # 1. weight vs cosine
    rho_global, p_global = spearmanr(wf, cf)
    rho_item = np.empty(nv)
    for j, i in enumerate(valid_ids):
        r, _ = spearmanr(w[j], cos[j])
        rho_item[j] = r if np.isfinite(r) else np.nan
    top5_overlap = np.mean([
        len(np.intersect1d(np.argsort(-w[j])[:TOP_M], np.argsort(-cos[j])[:TOP_M])) / TOP_M
        for j in range(nv)
    ])

    # 2. asymmetry
    edge_w = {int(i) * n + int(j): float(x) for i, j, x in zip(anchor, neigh, wf)}
    diffs = [abs(x - edge_w[int(j) * n + int(i)]) for i, j, x in zip(anchor, neigh, wf)
             if int(j) * n + int(i) in edge_w]
    diffs = np.asarray(diffs)
    ai = np.array([x for i, j, x in zip(anchor, neigh, wf) if int(j) * n + int(i) in edge_w])
    aj = np.array([edge_w[int(j) * n + int(i)] for i, j, x in zip(anchor, neigh, wf)
                   if int(j) * n + int(i) in edge_w])

    # 3. entropy
    H = -(w * np.log(np.maximum(w, 1e-12))).sum(axis=1)
    Hn = H / np.log(K)
    eff = np.exp(H)
    top1 = w.max(axis=1)
    top5_mass = np.sort(w, axis=1)[:, -TOP_M:].sum(axis=1)

    # 4. transition / co-occurrence lift
    order = np.argsort(-w, axis=1)
    top_idx = np.take_along_axis(idx, order[:, :TOP_M], axis=1)
    bot_idx = np.take_along_axis(idx, order[:, -TOP_M:], axis=1)
    trans_mat = np.where(T_out[:, None] > 0, T / np.maximum(T_out[:, None], 1), np.nan)[valid_ids]
    p_top = np.nanmean(np.take_along_axis(trans_mat, top_idx, axis=1), axis=1)
    p_all = np.nanmean(np.take_along_axis(trans_mat, idx, axis=1), axis=1)
    p_bot = np.nanmean(np.take_along_axis(trans_mat, bot_idx, axis=1), axis=1)
    ok = np.isfinite(p_top) & np.isfinite(p_all) & (p_all > 0)
    lift = p_top[ok] / p_all[ok]
    lift_bot = p_bot[ok] / p_all[ok]
    wstat, wp = wilcoxon(p_top[ok], p_all[ok])
    lift_full = np.full(nv, np.nan); lift_full[ok] = lift

    cooc_mat = (C / np.maximum(n_user_i[:, None], 1))[valid_ids]
    pco_top = np.take_along_axis(cooc_mat, top_idx, axis=1).mean(axis=1)
    pco_all = np.take_along_axis(cooc_mat, idx, axis=1).mean(axis=1)
    pco_bot = np.take_along_axis(cooc_mat, bot_idx, axis=1).mean(axis=1)
    okc = pco_all > 0
    cooc_lift = pco_top[okc] / pco_all[okc]

    # per-item table
    per = pd.DataFrame({
        "item": valid_ids, "group": np.array(["head", "mid", "tail"])[groups[valid_ids]],
        "pop_users": n_user_i[valid_ids], "weight_rho": rho_item,
        "weight_top1": top1, "weight_top5_mass": top5_mass,
        "attention_entropy": H, "attention_entropy_norm": Hn,
        "effective_neighbors": eff, "trans_lift_top5": lift_full,
        "cat_leaf": [leaf_names[i] for i in valid_ids],
        "cat_l2": [l2_names[i] for i in valid_ids],
    })
    per.to_csv(os.path.join(RESULTS, "exp_f_csga_per_item_validcollab.csv"), index=False)

    group_summary = []
    for g in ["head", "mid", "tail"]:
        m = per.group == g
        group_summary.append({
            "group": g, "n_items": int(m.sum()),
            "weight_rho": float(per.loc[m, "weight_rho"].mean()),
            "attention_entropy_norm": float(per.loc[m, "attention_entropy_norm"].mean()),
            "effective_neighbors": float(per.loc[m, "effective_neighbors"].mean()),
            "weight_top1": float(per.loc[m, "weight_top1"].mean()),
            "trans_lift_top5": float(per.loc[m, "trans_lift_top5"].mean()),
        })

    metrics = {
        "n_items": nv, "n_excluded_zero_collab": int(n - nv), "n_edges": int(len(wf)),
        "weight_vs_collab_cosine": {
            "spearman_global": float(rho_global), "p_global": float(p_global),
            "spearman_per_item_mean": float(np.nanmean(rho_item)),
            "spearman_per_item_median": float(np.nanmedian(rho_item)),
            "spearman_per_item_p10": float(np.nanpercentile(rho_item, 10)),
            "spearman_per_item_p90": float(np.nanpercentile(rho_item, 90)),
            "top5_overlap_with_cosine": float(top5_overlap),
        },
        "weight_vs_text_cos_spearman": float(spearmanr(wf, text_cos)[0]),
        "weight_vs_image_cos_spearman": float(spearmanr(wf, image_cos)[0]),
        "weight_vs_transition_spearman": float(spearmanr(wf[np.isfinite(p_trans)], p_trans[np.isfinite(p_trans)])[0]),
        "asymmetry": {
            "mutual_pair_fraction": float(len(diffs) / len(wf)),
            "n_mutual_pairs": int(len(diffs)),
            "mean_abs_diff": float(diffs.mean()), "median_abs_diff": float(np.median(diffs)),
            "p90_abs_diff": float(np.percentile(diffs, 90)),
            "frac_abs_diff_gt_0.02": float((diffs > 0.02).mean()),
            "frac_abs_diff_gt_0.05": float((diffs > 0.05).mean()),
            "corr_aij_aji": float(np.corrcoef(ai, aj)[0, 1]),
        },
        "entropy": {
            "entropy_mean": float(H.mean()), "entropy_median": float(np.median(H)),
            "entropy_norm_mean": float(Hn.mean()), "entropy_norm_median": float(np.median(Hn)),
            "entropy_norm_p10": float(np.percentile(Hn, 10)),
            "entropy_norm_p90": float(np.percentile(Hn, 90)),
            "effective_neighbors_mean": float(eff.mean()),
            "effective_neighbors_median": float(np.median(eff)),
            "top1_weight_mean": float(top1.mean()),
            "top5_mass_mean": float(top5_mass.mean()),
        },
        "transition_lift_top5_vs_unweighted": {
            "n_items": int(ok.sum()),
            "mean_top5_trans": float(np.nanmean(p_top[ok])),
            "mean_unweighted_trans": float(np.nanmean(p_all[ok])),
            "mean_lift": float(np.nanmean(lift)), "median_lift": float(np.nanmedian(lift)),
            "frac_items_lift_gt_1": float(np.mean(lift > 1)),
            "wilcoxon_stat": float(wstat), "wilcoxon_p": float(wp),
            "mean_bottom5_trans": float(np.nanmean(p_bot[ok])),
            "mean_bottom5_lift": float(np.nanmean(lift_bot)),
        },
        "cooccurrence_lift_top5_vs_unweighted": {
            "n_items": int(okc.sum()),
            "mean_top5_cooc": float(pco_top.mean()),
            "mean_unweighted_cooc": float(pco_all.mean()),
            "aggregate_ratio": float(pco_top.mean() / max(pco_all.mean(), 1e-12)),
            "mean_lift": float(np.nanmean(cooc_lift)),
            "median_lift": float(np.nanmedian(cooc_lift)),
            "frac_items_lift_gt_1": float(np.nanmean(cooc_lift > 1)),
            "mean_bottom5_cooc": float(pco_bot.mean()),
            "mean_bottom5_lift": float(np.nanmean(pco_bot[okc] / pco_all[okc])),
        },
        "groups": group_summary,
    }
    with open(os.path.join(RESULTS, "exp_f_csga_metrics_validcollab.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
