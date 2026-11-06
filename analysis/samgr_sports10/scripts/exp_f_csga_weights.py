"""Recover and analyse CSGA learned neighbour weights on Sports strict 10-core.

The training script does not checkpoint the CSGA scorer, but it saves the
CSGA-aligned feature ``h_i = m_i + 0.5 * sum_j a_ij m_j`` (``m`` is the
mean-centred multimodal feature).  For the fixed collab Top-K neighbour list we
can recover the effective ``a_ij`` by solving a small KxK linear system per item;
the reconstruction residual is reported as a sanity check.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, l2n, load_collab, load_emb, load_items  # noqa: E402
from exp_e_topk_neighborhoods import (build_graphs, category_arrays,  # noqa: E402
                                      load_train_sequences, popularity_groups)

K = 10
TOP_M = 5
DATA = "/data02/sht/SAM-GR/data/Amazon/Sports_and_Outdoors"
CSGA = os.path.join(RESULTS, "csga_aligned.npy")


def recover_weights():
    nbz = np.load(os.path.join(RESULTS, "neighbors_spaces.npz"))
    idx = nbz["collab_idx"][:, :K].astype(np.int64)
    cos = nbz["collab_cos"][:, :K].astype(np.float32)
    m = load_emb("multimodal").astype(np.float32)
    h = np.load(CSGA).astype(np.float32)
    sem = m - m.mean(axis=0, keepdims=True)
    n = len(m)
    w = np.empty((n, K), dtype=np.float32)
    resid = np.empty(n, dtype=np.float32)
    t0 = time.time()
    bs = 64
    for st in range(0, n, bs):
        en = min(st + bs, n)
        A = sem[idx[st:en]]                       # B x K x D
        d = 2.0 * (h[st:en] - sem[st:en])         # B x D
        AtA = np.einsum("bkd,bld->bkl", A, A, optimize=True).astype(np.float64)
        Atd = np.einsum("bkd,bd->bk", A, d, optimize=True).astype(np.float64)
        wb = np.linalg.solve(AtA, Atd[..., None])[..., 0]
        pred = np.einsum("bkd,bk->bd", A, wb, optimize=True)
        resid[st:en] = np.linalg.norm(pred - d, axis=1) / (np.linalg.norm(d, axis=1) + 1e-12)
        w[st:en] = wb
        if st % (bs * 40) == 0:
            print("recover", st, f"{time.time()-t0:.1f}s", flush=True)
    raw_sum = w.sum(axis=1)
    neg_frac = float((w < -1e-4).mean())
    w = np.clip(w, 0.0, None)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    print("weights", w.shape, "raw_sum", float(raw_sum.mean()), "neg_frac", neg_frac,
          "resid mean/max", float(resid.mean()), float(resid.max()))
    return w.astype(np.float32), idx, cos, sem, m, resid, raw_sum, neg_frac


def main():
    t0 = time.time()
    items = load_items()
    n = len(items)
    w, idx, cos, sem, m, resid, raw_sum, neg_frac = recover_weights()
    np.savez_compressed(
        os.path.join(RESULTS, "exp_f_csga_weights.npz"),
        idx=idx, weight=w, collab_cos=cos,
        residual=resid, raw_sum=raw_sum,
    )

    # ---------- graphs / metadata ----------
    seqs, user_items = load_train_sequences()
    C, T, T_out, n_user_i, n_users = build_graphs(seqs, user_items, n)
    leaf, l2, leaf_names, l2_names = category_arrays(items, n)
    groups, group_specs = popularity_groups(n_user_i)

    rows = np.repeat(np.arange(n, dtype=np.int64), K)
    cols = idx.reshape(-1).astype(np.int64)
    wf = w.reshape(-1).astype(np.float64)
    cf = cos.reshape(-1).astype(np.float64)
    text = l2n(load_emb("text_only"))
    image = l2n(load_emb("image_only"))
    text_cos = np.einsum("kd,kd->k", text[rows], text[cols], optimize=True)
    image_cos = np.einsum("kd,kd->k", image[rows], image[cols], optimize=True)

    c = C[rows, cols].astype(np.float64)
    p_cooc = c / np.maximum(n_user_i[rows], 1)
    p_ij = c / n_users
    pi = n_user_i[rows] / n_users
    pj = n_user_i[cols] / n_users
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.log(p_ij / np.maximum(pi * pj, 1e-30)) / (-np.log(np.maximum(p_ij, 1e-30)))
    npmi[c <= 0] = np.nan
    t = T[rows, cols].astype(np.float64)
    tout = T_out[rows].astype(np.float64)
    p_trans = np.where(tout > 0, t / np.maximum(tout, 1e-12), np.nan)

    np.savez_compressed(
        os.path.join(RESULTS, "exp_f_csga_edges.npz"),
        anchor=rows.astype(np.int32), neighbor=cols.astype(np.int32),
        weight=wf.astype(np.float32), collab_cos=cf.astype(np.float32),
        text_cos=text_cos.astype(np.float32), image_cos=image_cos.astype(np.float32),
        cooc_prob=p_cooc.astype(np.float32), trans_prob=p_trans.astype(np.float32),
        npmi=npmi.astype(np.float32), shared_users=c.astype(np.int32),
        collab_rank=np.tile(np.arange(K, dtype=np.int32), n),
    )

    # ---------- 1. weight vs collaborative cosine ----------
    rho_global, p_global = spearmanr(wf, cf)
    per_item_rho = np.empty(n, dtype=np.float32)
    per_item_p = np.empty(n, dtype=np.float32)
    for i in range(n):
        r, p = spearmanr(w[i], cos[i])
        per_item_rho[i] = r if np.isfinite(r) else np.nan
        per_item_p[i] = p
    top5_cos_overlap = np.mean([
        len(np.intersect1d(np.argsort(-w[i])[:TOP_M], np.argsort(-cos[i])[:TOP_M])) / TOP_M
        for i in range(n)
    ])
    rho_text, _ = spearmanr(wf, text_cos)
    rho_image, _ = spearmanr(wf, image_cos)
    mask_t = np.isfinite(p_trans)
    rho_trans, _ = spearmanr(wf[mask_t], p_trans[mask_t])

    # ---------- 2. asymmetry a_ij vs a_ji ----------
    edge_w = {int(i) * n + int(j): float(x) for i, j, x in zip(rows, cols, wf)}
    diffs, ai, aj = [], [], []
    for i, j, x in zip(rows, cols, wf):
        y = edge_w.get(int(j) * n + int(i))
        if y is not None:
            diffs.append(abs(x - y)); ai.append(x); aj.append(y)
    diffs = np.asarray(diffs, dtype=np.float64)
    ai = np.asarray(ai); aj = np.asarray(aj)
    corr_asym = float(np.corrcoef(ai, aj)[0, 1])
    mutual_frac = len(diffs) / len(wf)

    # ---------- 3. attention entropy ----------
    H = -(w * np.log(np.maximum(w, 1e-12))).sum(axis=1)
    Hn = H / np.log(K)
    eff = np.exp(H)
    top1 = w.max(axis=1)

    # ---------- 4. top-weight vs unweighted transition lift ----------
    order = np.argsort(-w, axis=1)
    top_idx = np.take_along_axis(idx, order[:, :TOP_M], axis=1)
    bot_idx = np.take_along_axis(idx, order[:, -TOP_M:], axis=1)
    trans_mat = np.where(T_out[:, None] > 0, T / np.maximum(T_out[:, None], 1), np.nan)
    p_top = np.nanmean(np.take_along_axis(trans_mat, top_idx, axis=1), axis=1)
    p_bot = np.nanmean(np.take_along_axis(trans_mat, bot_idx, axis=1), axis=1)
    p_all = np.nanmean(np.take_along_axis(trans_mat, idx, axis=1), axis=1)
    valid = np.isfinite(p_top) & np.isfinite(p_all) & (p_all > 0)
    lift = p_top[valid] / p_all[valid]
    lift_bot = p_bot[valid] / p_all[valid]
    wstat, wp = wilcoxon(p_top[valid], p_all[valid])
    trans_lift_full = np.full(n, np.nan, dtype=np.float64)
    trans_lift_full[valid] = lift

    # same for co-occurrence / NPMI, at item level
    cooc_mat = C / np.maximum(n_user_i[:, None], 1)
    pco_top = np.take_along_axis(cooc_mat, top_idx, axis=1).mean(axis=1)
    pco_all = np.take_along_axis(cooc_mat, idx, axis=1).mean(axis=1)
    pco_bot = np.take_along_axis(cooc_mat, bot_idx, axis=1).mean(axis=1)
    valid_co = pco_all > 0
    cooc_lift = np.full(n, np.nan, dtype=np.float64)
    cooc_lift[valid_co] = pco_top[valid_co] / pco_all[valid_co]
    npmi_mat = np.full((n, n), np.nan, dtype=np.float32)

    # ---------- per-item table ----------
    per_item = pd.DataFrame({
        "item": np.arange(n),
        "weight_rho": per_item_rho,
        "weight_rho_p": per_item_p,
        "weight_top1": top1,
        "weight_top5_mass": np.sort(w, axis=1)[:, -TOP_M:].sum(axis=1),
        "attention_entropy": H,
        "attention_entropy_norm": Hn,
        "effective_neighbors": eff,
        "residual": resid,
        "pop_users": n_user_i,
        "group": np.array(["head", "mid", "tail"])[groups],
        "cat_leaf": [leaf_names[i] for i in range(n)],
        "cat_l2": [l2_names[i] for i in range(n)],
        "trans_lift_top5": trans_lift_full,
    })
    # add rank so group can be reassigned in case of ties
    per_item.to_csv(os.path.join(RESULTS, "exp_f_csga_per_item.csv"), index=False)

    group_summary = []
    for g in ["head", "mid", "tail"]:
        msk = groups == {"head": 0, "mid": 1, "tail": 2}[g]
        group_summary.append({
            "group": g, "n_items": int(msk.sum()),
            "weight_rho": float(np.nanmean(per_item_rho[msk])),
            "attention_entropy_norm": float(Hn[msk].mean()),
            "effective_neighbors": float(eff[msk].mean()),
            "weight_top1": float(top1[msk].mean()),
            "residual": float(resid[msk].mean()),
            "trans_lift_top5": float(np.nanmean(trans_lift_full[msk])),
        })
    pd.DataFrame(group_summary).to_csv(os.path.join(RESULTS, "exp_f_csga_by_group.csv"), index=False)

    metrics = {
        "k": K,
        "n_items": n,
        "n_edges": int(len(wf)),
        "recovery": {
            "mean_relative_residual": float(resid.mean()),
            "max_relative_residual": float(resid.max()),
            "mean_raw_weight_sum": float(raw_sum.mean()),
            "negative_weight_fraction_before_clip": neg_frac,
        },
        "weight_vs_collab_cosine": {
            "spearman_global": float(rho_global), "p_global": float(p_global),
            "spearman_per_item_mean": float(np.nanmean(per_item_rho)),
            "spearman_per_item_median": float(np.nanmedian(per_item_rho)),
            "spearman_per_item_p10": float(np.nanpercentile(per_item_rho, 10)),
            "spearman_per_item_p90": float(np.nanpercentile(per_item_rho, 90)),
            "top5_overlap_with_cosine": float(top5_cos_overlap),
        },
        "weight_vs_text_cos_spearman": float(rho_text),
        "weight_vs_image_cos_spearman": float(rho_image),
        "weight_vs_transition_spearman": float(rho_trans),
        "asymmetry": {
            "mutual_pair_fraction": float(mutual_frac),
            "n_mutual_pairs": int(len(diffs)),
            "mean_abs_diff": float(diffs.mean()),
            "median_abs_diff": float(np.median(diffs)),
            "p90_abs_diff": float(np.percentile(diffs, 90)),
            "frac_abs_diff_gt_0.02": float((diffs > 0.02).mean()),
            "frac_abs_diff_gt_0.05": float((diffs > 0.05).mean()),
            "corr_aij_aji": corr_asym,
        },
        "entropy": {
            "entropy_mean": float(H.mean()), "entropy_median": float(np.median(H)),
            "entropy_norm_mean": float(Hn.mean()), "entropy_norm_median": float(np.median(Hn)),
            "entropy_norm_p10": float(np.percentile(Hn, 10)),
            "entropy_norm_p90": float(np.percentile(Hn, 90)),
            "effective_neighbors_mean": float(eff.mean()),
            "effective_neighbors_median": float(np.median(eff)),
            "top1_weight_mean": float(top1.mean()),
            "top5_mass_mean": float(per_item.weight_top5_mass.mean()),
        },
        "transition_lift_top5_vs_unweighted": {
            "n_items": int(valid.sum()),
            "mean_top5_trans": float(np.nanmean(p_top[valid])),
            "mean_unweighted_trans": float(np.nanmean(p_all[valid])),
            "mean_lift": float(np.nanmean(lift)),
            "median_lift": float(np.nanmedian(lift)),
            "frac_items_lift_gt_1": float(np.mean(lift > 1)),
            "wilcoxon_stat": float(wstat), "wilcoxon_p": float(wp),
            "mean_bottom5_trans": float(np.nanmean(p_bot[valid])),
            "mean_bottom5_lift": float(np.nanmean(lift_bot)),
        },
        "cooccurrence_lift_top5_vs_unweighted": {
            "n_items": int(valid_co.sum()),
            "mean_top5_cooc": float(pco_top.mean()),
            "mean_unweighted_cooc": float(pco_all.mean()),
            "aggregate_ratio": float(pco_top.mean() / max(pco_all.mean(), 1e-12)),
            "mean_lift": float(np.nanmean(cooc_lift)),
            "median_lift": float(np.nanmedian(cooc_lift)),
            "frac_items_lift_gt_1": float(np.nanmean(cooc_lift > 1)),
            "mean_bottom5_cooc": float(pco_bot.mean()),
            "mean_bottom5_lift": float(np.nanmean(pco_bot[valid_co] / np.maximum(pco_all[valid_co], 1e-12))),
        },
        "groups": group_summary,
    }
    with open(os.path.join(RESULTS, "exp_f_csga_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("EXP_F_DONE", f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
