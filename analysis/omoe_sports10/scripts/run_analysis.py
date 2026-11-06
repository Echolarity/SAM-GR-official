"""Systematic analysis of Full OMoE-v2 gating and expert utilisation.

Setting: MIMIGenRec, Sports strict-10-core (11,246 items), csga_omoe_v2.
All routing statistics are computed from a faithful re-instantiation of the
pipeline's OMoE-v2 (exact match with the recorded exp_d full-OMoE gates).
Recommendation changes under interventions use a feature-level retrieval proxy
on the real test interactions (the LLM is not re-trained).
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omoe_core import (ANALYSIS, EXPERT_COLORS, EXPERT_NAMES, FIGURES, N_EXPERTS,
                       RESULTS, build_model, center, gate_entropy, infer, js_divergence,
                       l2n, load_csga, load_emb, load_inter, load_items,
                       production_fused, train_model)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
np.random.seed(0)

E = N_EXPERTS
NAMES = EXPERT_NAMES
COLORS = EXPERT_COLORS
RES = {}
DATASET_LABEL = "Sports strict-10-core"


def save_json(obj, name):
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def df_to_md(df, floatfmt=".4f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:{floatfmt}}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def clean(s):
    return str(s or "").replace("\n", " ").strip()


def short(s, n=95):
    s = clean(s)
    return s[:n] + ("..." if len(s) > n else "")


# --------------------------------------------------------------------------- #
# Data / metadata
# --------------------------------------------------------------------------- #
def load_all():
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    x_c = load_csga()
    cache = np.load(os.path.join(RESULTS, "omoe_cache.npz"))
    items = load_items()
    n = len(x_t)
    ids = np.arange(n)

    titles, descs, cats, has_img, text_len = [], [], [], [], []
    for i in range(n):
        it = items[str(i)]
        t = clean(it.get("title", ""))
        d = clean(it.get("description", ""))
        c = clean(it.get("categories", ""))
        titles.append(t); descs.append(d); cats.append(c)
        has_img.append(bool(it.get("image_paths")))
        text_len.append(len(t) + len(d))

    # second-level category (top-level is almost always "Sports & Outdoors")
    cat2 = []
    for c in cats:
        parts = [p.strip() for p in c.split(",") if p.strip()]
        cat2.append(parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown"))

    pop = np.zeros(n, dtype=np.int64)
    for _, hist, tgt in load_inter("train"):
        for j in hist:
            pop[j] += 1
        pop[tgt] += 1

    xt_n, xi_n = l2n(x_t), l2n(x_i)
    consistency = (xt_n * xi_n).sum(1)

    meta = {
        "titles": np.asarray(titles, dtype=object), "descs": np.asarray(descs, dtype=object),
        "cats": np.asarray(cats, dtype=object), "cat2": np.asarray(cat2, dtype=object),
        "has_img": np.asarray(has_img), "text_len": np.asarray(text_len, dtype=np.int64),
        "pop": pop, "consistency": consistency,
    }
    return x_t, x_i, x_c, cache, meta


def grouped_stats(meta, labels, order, gates, ent, dim_name):
    rows = []
    for g in order:
        m = labels == g
        if m.sum() == 0:
            continue
        gm = gates[m].mean(0)
        rows.append({
            "dimension": dim_name, "group": g, "n": int(m.sum()),
            **{f"gate_{NAMES[j]}": float(gm[j]) for j in range(E)},
            "mean_entropy": float(ent[m].mean()),
            "effective_experts": float(np.exp(ent[m].mean())),
            "mean_popularity": float(meta["pop"][m].mean()),
            "mean_text_len": float(meta["text_len"][m].mean()),
            "mean_text_image_cos": float(meta["consistency"][m].mean()),
        })
    return rows


# --------------------------------------------------------------------------- #
# Part 1: global statistics
# --------------------------------------------------------------------------- #
def part1(gates, expert_norms, ent, model, t, i, c):
    n = len(gates)
    mean_gate = gates.mean(0)
    std_gate = gates.std(0)
    top1 = gates.argmax(1)
    top1_freq = np.bincount(top1, minlength=E) / n
    gate_lt_001 = (gates < 0.01).mean(0)
    gate_lt_005 = (gates < 0.05).mean(0)
    norm_mean = expert_norms.mean(0)
    norm_std = expert_norms.std(0)
    contrib = (gates * expert_norms).mean(0)
    contrib_share = contrib / contrib.sum()

    load_cv = float(mean_gate.std() / mean_gate.mean())
    global_ent = float(-(mean_gate * np.log(mean_gate + 1e-12)).sum())
    eff_global = float(np.exp(global_ent))
    mean_ent = float(ent.mean())
    eff_item = float(np.exp(mean_ent))
    dead_load_experts = [NAMES[j] for j in range(E) if mean_gate[j] < 0.05 or gate_lt_005[j] > 0.5]
    never_top1_experts = [NAMES[j] for j in range(E) if top1_freq[j] < 0.01]

    # expert-output similarity (mean pairwise cosine)
    sim = np.zeros((E, E)); cnt = 0
    with torch.no_grad():
        for st in range(0, n, 512):
            en = min(st + 512, n)
            model(t[st:en], i[st:en], c[st:en])
            ex = model._monitor_experts
            B = en - st
            for p in range(E):
                for q in range(p, E):
                    v = F.cosine_similarity(ex[p], ex[q], dim=-1).mean().item()
                    sim[p, q] += v * B
                    if p != q:
                        sim[q, p] += v * B
            cnt += B
    sim /= cnt
    sim = np.clip(sim, -1, 1)
    np.fill_diagonal(sim, 1.0)
    off = sim[~np.eye(E, dtype=bool)]
    mean_abs_cos = float(np.abs(off).mean())

    # out_proj(concat) vs expert-mixture magnitude (context for contribution share)
    out_norms, moe_norms = [], []
    with torch.no_grad():
        for st in range(0, n, 512):
            en = min(st + 512, n)
            _ = model(t[st:en], i[st:en], c[st:en])
            feats = [model.align_text(t[st:en]), model.align_image(i[st:en]),
                     model.align_collab(c[st:en])]
            concat = torch.cat(feats, dim=-1)
            out = model.out_proj(concat)
            moe = torch.zeros_like(model._monitor_experts[0])
            for k in range(E):
                moe = moe + model._monitor_gates[:, k].unsqueeze(-1) * model._monitor_experts[k]
            out_norms.append(out.norm(dim=-1).cpu().numpy())
            moe_norms.append(moe.norm(dim=-1).cpu().numpy())
    out_norms = np.concatenate(out_norms); moe_norms = np.concatenate(moe_norms)
    out_moe_corr = float(np.corrcoef(out_norms, moe_norms)[0, 1])

    df = pd.DataFrame({
        "expert": NAMES,
        "mean_gate": mean_gate,
        "std_gate": std_gate,
        "cv_across_items": std_gate / mean_gate,
        "top1_freq": top1_freq,
        "mean_output_norm": norm_mean,
        "std_output_norm": norm_std,
        "mean_effective_contribution": contrib,
        "contribution_share": contrib_share,
        "gate_lt_0.01_rate": gate_lt_001,
        "gate_lt_0.05_rate": gate_lt_005,
    })
    df.to_csv(os.path.join(RESULTS, "part1_expert_stats.csv"), index=False)

    summary = {
        "n_items": n,
        "mean_gate": mean_gate.tolist(),
        "std_gate": std_gate.tolist(),
        "top1_freq": top1_freq.tolist(),
        "mean_output_norm": norm_mean.tolist(),
        "std_output_norm": norm_std.tolist(),
        "gate_lt_0.01_rate": gate_lt_001.tolist(),
        "gate_lt_0.05_rate": gate_lt_005.tolist(),
        "mean_effective_contribution": contrib.tolist(),
        "contribution_share": contrib_share.tolist(),
        "dead_load_experts": dead_load_experts,
        "never_top1_experts": never_top1_experts,
        "load_cv_across_experts": load_cv,
        "global_gate_entropy": global_ent,
        "global_effective_experts": eff_global,
        "max_entropy": float(np.log(E)),
        "mean_item_entropy": mean_ent,
        "std_item_entropy": float(ent.std()),
        "min_item_entropy": float(ent.min()),
        "max_item_entropy": float(ent.max()),
        "effective_experts_per_item": eff_item,
        "expert_similarity": sim.tolist(),
        "mean_abs_offdiag_cos": mean_abs_cos,
        "out_proj_norm_mean": float(out_norms.mean()),
        "moe_norm_mean": float(moe_norms.mean()),
        "out_proj_moe_norm_corr": out_moe_corr,
    }
    save_json(summary, "part1_global_stats.json")
    RES["part1"] = summary
    RES["part1_df"] = df

    # Figure 1
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    x = np.arange(E)
    axes[0].bar(x, mean_gate, yerr=std_gate, color=COLORS, capsize=4)
    axes[0].axhline(1 / E, color="gray", ls="--", lw=1)
    axes[0].set_ylim(max(0, mean_gate.min() - 0.04), mean_gate.max() + 0.04)
    axes[0].set_xticks(x); axes[0].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[0].set_ylabel("mean gate weight"); axes[0].set_title("(a) Mean gate ± std (across items)")

    axes[1].bar(x, top1_freq, color=COLORS)
    axes[1].axhline(1 / E, color="gray", ls="--", lw=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[1].set_ylabel("fraction of items"); axes[1].set_title("(b) Top-1 routing frequency")

    axes[2].bar(x, norm_mean, yerr=norm_std, color=COLORS, capsize=4)
    axes[2].set_xticks(x); axes[2].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[2].set_ylabel("L2 norm"); axes[2].set_title("(c) Expert output norm")

    axes[3].bar(x, contrib_share, color=COLORS)
    axes[3].axhline(1 / E, color="gray", ls="--", lw=1)
    axes[3].set_xticks(x); axes[3].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[3].set_ylabel("share of Σ gate·‖output‖"); axes[3].set_title("(d) Effective contribution share")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_expert_load.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 2: instance-level routing differences
# --------------------------------------------------------------------------- #
def part2(gates, ent, meta):
    n = len(gates)
    var = gates.var(0)
    cov = np.cov(gates, rowvar=False)

    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=1000, replace=False)
    G = gates[sample]
    P = G[:, None, :]
    Q = G[None, :, :]
    M = 0.5 * (P + Q)
    kl_pm = (P * np.log(np.clip(P, 1e-12, None) / np.clip(M, 1e-12, None))).sum(-1)
    kl_qm = (Q * np.log(np.clip(Q, 1e-12, None) / np.clip(M, 1e-12, None))).sum(-1)
    JS = 0.5 * (kl_pm + kl_qm)
    iu = np.triu_indices(len(sample), 1)
    js_vals = JS[iu]

    mean_gate = gates.mean(0)
    js_to_mean = np.array([js_divergence(g, mean_gate) for g in G])

    # PCA of the 5-D routing vectors
    pca = PCA(n_components=3).fit(gates)
    Z = pca.transform(gates)
    pc1_textlen = stats.spearmanr(Z[:, 0], meta["text_len"])
    pc1_cons = stats.spearmanr(Z[:, 0], meta["consistency"])
    pc1_pop = stats.spearmanr(Z[:, 0], meta["pop"])
    gate_corr = np.corrcoef(gates, rowvar=False)

    stats_out = {
        "gate_variance_per_expert": var.tolist(),
        "gate_variance_total": float(var.sum()),
        "gate_covariance_trace": float(np.trace(cov)),
        "gate_correlation": gate_corr.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pc1_textlen_spearman": float(pc1_textlen.statistic),
        "pc1_textlen_p": float(pc1_textlen.pvalue),
        "pc1_consistency_spearman": float(pc1_cons.statistic),
        "pc1_consistency_p": float(pc1_cons.pvalue),
        "pc1_popularity_spearman": float(pc1_pop.statistic),
        "pc1_popularity_p": float(pc1_pop.pvalue),
        "js_sample_size": 1000,
        "js_mean": float(js_vals.mean()),
        "js_median": float(np.median(js_vals)),
        "js_p90": float(np.quantile(js_vals, 0.90)),
        "js_p95": float(np.quantile(js_vals, 0.95)),
        "js_p99": float(np.quantile(js_vals, 0.99)),
        "js_max": float(js_vals.max()),
        "js_to_mean_mean": float(js_to_mean.mean()),
        "js_to_mean_p95": float(np.quantile(js_to_mean, 0.95)),
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "entropy_p01": float(np.quantile(ent, 0.01)),
        "entropy_p05": float(np.quantile(ent, 0.05)),
        "entropy_p50": float(np.quantile(ent, 0.50)),
        "entropy_p95": float(np.quantile(ent, 0.95)),
        "entropy_p99": float(np.quantile(ent, 0.99)),
        "entropy_min": float(ent.min()),
        "entropy_max": float(ent.max()),
        "entropy_lt_1": float((ent < 1.0).mean()),
        "entropy_gt_1p5": float((ent > 1.5).mean()),
    }
    save_json(stats_out, "part2_instance_stats.json")
    RES["part2"] = stats_out

    # Heatmap: 1000 random items, sorted within dominant-expert blocks by entropy
    dominant = G.argmax(1)
    sort_key = np.lexsort((-ent[sample], dominant))
    Gs = G[sort_key]
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    im = ax.imshow(Gs, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xticks(np.arange(E)); ax.set_xticklabels(NAMES, rotation=25, ha="right")
    ax.set_ylabel("1,000 randomly sampled items\n(sorted by dominant expert, then entropy)")
    ax.set_title("Per-item routing distribution (gate weights)")
    # block boundaries
    b = np.cumsum(np.bincount(dominant[sort_key], minlength=E))[:-1]
    for v in b:
        ax.axhline(v - 0.5, color="white", lw=0.6, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="gate weight")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_routing_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # Entropy / JS distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(ent, bins=60, color="#4C72B0", alpha=0.85)
    axes[0].axvline(np.log(E), color="black", ls="--", lw=1, label=f"ln{E}={np.log(E):.3f}")
    axes[0].axvline(ent.mean(), color="#C44E52", ls="-", lw=1.5, label=f"mean={ent.mean():.3f}")
    axes[0].set_xlabel("per-item routing entropy"); axes[0].set_ylabel("#items")
    axes[0].set_title("(a) Per-item entropy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(js_vals, bins=60, color="#55A868", alpha=0.85)
    axes[1].axvline(js_vals.mean(), color="#C44E52", lw=1.5, label=f"mean={js_vals.mean():.4f}")
    axes[1].set_xlabel("pairwise JS divergence"); axes[1].set_ylabel("#pairs")
    axes[1].set_title("(b) Pairwise JS (1,000 sampled items)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_entropy_js.png"), bbox_inches="tight")
    plt.close(fig)

    # Routing PCA: PC1 is the text-vs-shared-2 axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rng2 = np.random.default_rng(1)
    sub = rng2.choice(n, size=min(4000, n), replace=False)
    dom = gates[sub].argmax(1)
    for j in range(E):
        m = dom == j
        if m.sum() == 0:
            continue
        axes[0].scatter(np.log10(meta["text_len"][sub][m] + 1), Z[sub][m, 0],
                        s=7, alpha=0.55, color=COLORS[j], label=f"{NAMES[j]} (n={m.sum()})")
    axes[0].set_xlabel("log10(text length + 1)")
    axes[0].set_ylabel("routing PC1")
    axes[0].set_title(f"(a) PC1 vs. text length (Spearman ρ={pc1_textlen.statistic:.3f})")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)

    load1 = pca.components_[0]
    axes[1].bar(np.arange(E), load1, color=COLORS)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(np.arange(E)); axes[1].set_xticklabels(NAMES, rotation=20, ha="right")
    axes[1].set_ylabel("PC1 loading")
    axes[1].set_title(f"(b) PC1 loadings (explains {pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_routing_pca.png"), bbox_inches="tight")
    plt.close(fig)

    # High / low entropy cases
    order_desc = np.argsort(-ent)
    order_asc = np.argsort(ent)
    rows = []
    for rank, idx in enumerate(list(order_desc[:3]) + list(order_asc[:3])):
        kind = "high" if rank < 3 else "low"
        rows.append({
            "kind": kind, "item_id": int(idx),
            "title": short(meta["titles"][idx], 120),
            "category": meta["cat2"][idx],
            "popularity": int(meta["pop"][idx]),
            "text_len": int(meta["text_len"][idx]),
            "text_image_cos": float(meta["consistency"][idx]),
            **{f"gate_{NAMES[j]}": float(gates[idx, j]) for j in range(E)},
            "entropy": float(ent[idx]),
            "effective_experts": float(np.exp(ent[idx])),
            "top_expert": NAMES[int(gates[idx].argmax())],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "part2_entropy_cases.csv"), index=False)
    RES["part2_cases"] = df

    # A compact case figure
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(rows)); w = 0.15
    for j in range(E):
        ax.bar(x + (j - (E - 1) / 2) * w, [r[f"gate_{NAMES[j]}"] for r in rows], w,
               color=COLORS[j], label=NAMES[j])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['kind']}-{i+1}\nH={r['entropy']:.2f}" for i, r in enumerate(rows)], fontsize=8)
    ax.set_ylabel("gate weight")
    ax.set_title("High- vs. low-entropy routing cases")
    ax.legend(frameon=False, ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_entropy_cases.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 3: group analysis
# --------------------------------------------------------------------------- #
def part3(gates, ent, meta):
    n = len(gates)
    rows = []
    dims = {}

    # popularity: head (top 20%), mid (50-80%), tail (bottom 50%)
    pop = meta["pop"]
    q50, q80 = np.quantile(pop, [0.50, 0.80])
    pop_lab = np.where(pop >= q80, "head", np.where(pop >= q50, "mid", "tail"))
    dims["popularity"] = (pop_lab, ["tail", "mid", "head"])
    rows += grouped_stats(meta, pop_lab, ["head", "mid", "tail"], gates, ent, "popularity")

    # missing image
    img_lab = np.where(meta["has_img"], "has_image", "missing_image")
    dims["missing_image"] = (img_lab, ["has_image", "missing_image"])
    rows += grouped_stats(meta, img_lab, ["has_image", "missing_image"], gates, ent, "missing_image")

    # category (top 6 + Other)
    cnt = Counter(meta["cat2"])
    top_cats = [c for c, _ in cnt.most_common(6)]
    cat_lab = np.array([c if c in top_cats else "Other" for c in meta["cat2"]], dtype=object)
    order = top_cats + ["Other"]
    dims["category"] = (cat_lab, order)
    rows += grouped_stats(meta, cat_lab, order, gates, ent, "category")

    # text length tertiles (rank-based to handle outliers)
    tl = meta["text_len"]
    rank = np.empty(n, dtype=np.int64); rank[np.argsort(tl, kind="mergesort")] = np.arange(n)
    pct = rank / max(n - 1, 1)
    tl_lab = np.where(pct < 1/3, "short", np.where(pct < 2/3, "mid", "long"))
    dims["text_length"] = (tl_lab, ["short", "mid", "long"])
    rows += grouped_stats(meta, tl_lab, ["short", "mid", "long"], gates, ent, "text_length")

    # text-image consistency tertiles
    co = meta["consistency"]
    rank = np.empty(n, dtype=np.int64); rank[np.argsort(co, kind="mergesort")] = np.arange(n)
    pct = rank / max(n - 1, 1)
    co_lab = np.where(pct < 1/3, "low", np.where(pct < 2/3, "mid", "high"))
    dims["consistency"] = (co_lab, ["low", "mid", "high"])
    rows += grouped_stats(meta, co_lab, ["low", "mid", "high"], gates, ent, "consistency")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "part3_group_stats.csv"), index=False)
    RES["part3_df"] = df

    # Effect sizes: one-way ANOVA eta^2 and Kruskal-Wallis for each dimension/expert
    tests = []
    for dim, (lab, order) in dims.items():
        for j in range(E):
            groups = [gates[lab == g, j] for g in order if (lab == g).sum() > 0]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            H, pk = stats.kruskal(*groups)
            grand = gates[:, j].mean()
            ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ss_total = ((gates[:, j] - grand) ** 2).sum()
            eta2 = float(ss_between / max(ss_total, 1e-12))
            tests.append({"dimension": dim, "expert": NAMES[j], "eta2": eta2,
                          "anova_F": float(F), "anova_p": float(p),
                          "kruskal_H": float(H), "kruskal_p": float(pk),
                          "max_group_mean": float(max(g.mean() for g in groups)),
                          "min_group_mean": float(min(g.mean() for g in groups))})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RESULTS, "part3_tests.csv"), index=False)
    RES["part3_tests"] = tdf

    # Figure 3
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)

    def bars(ax, dim, title):
        sub = df[df.dimension == dim]
        order = dims[dim][1]
        sub = sub.set_index("group").loc[order]
        means = sub[[f"gate_{x}" for x in NAMES]].values
        x = np.arange(len(order)); w = 0.15
        for j in range(E):
            ax.bar(x + (j - (E - 1) / 2) * w, means[:, j], w, color=COLORS[j], label=NAMES[j])
        ax.axhline(1 / E, color="gray", ls="--", lw=0.9)
        lo = max(0, means.min() - 0.02); hi = means.max() + 0.02
        ax.set_ylim(lo, hi)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, ha="right")
        ax.set_ylabel("mean gate weight"); ax.set_title(title)
        return means

    bars(fig.add_subplot(gs[0, 0]), "popularity", "(a) Popularity: head / mid / tail")
    n_miss = int(df[(df.dimension == "missing_image") & (df.group == "missing_image")]["n"].iloc[0])
    bars(fig.add_subplot(gs[0, 1]), "missing_image", f"(b) Missing image (n={n_miss} for missing)")
    bars(fig.add_subplot(gs[1, 0]), "text_length", "(d) Text length tertiles")
    bars(fig.add_subplot(gs[1, 1]), "consistency", "(e) Text–image cosine consistency")

    # category heatmap
    ax = fig.add_subplot(gs[0, 2])
    sub = df[df.dimension == "category"]
    order = dims["category"][1]
    sub = sub.set_index("group").loc[order]
    mat = sub[[f"gate_{x}" for x in NAMES]].values
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=1 / E))
    ax.set_xticks(np.arange(E)); ax.set_xticklabels(NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([short(o, 22) for o in order], fontsize=7)
    ax.set_title("(c) Category (second-level)")
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            ax.text(b, a, f"{mat[a,b]:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # eta^2 heatmap
    ax = fig.add_subplot(gs[1, 2])
    pv = tdf.pivot(index="dimension", columns="expert", values="eta2")
    pv = pv[[x for x in NAMES if x in pv.columns]]
    im = ax.imshow(pv.values, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=8)
    ax.set_title("(f) Grouping effect size (ANOVA η²)")
    for a in range(pv.shape[0]):
        for b in range(pv.shape[1]):
            ax.text(b, a, f"{pv.values[a,b]:.4f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[j]) for j in range(E)]
    fig.legend(handles, NAMES, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Group-wise mean expert routing weights ({DATASET_LABEL})", fontsize=13)
    fig.savefig(os.path.join(FIGURES, "fig3_group_expert_weights.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 4: modality interventions
# --------------------------------------------------------------------------- #
def retrieval_eval(item_emb, cases, device="cuda:0", chunk=1024):
    """Feature-level next-item retrieval on the real test interactions."""
    ks = (1, 5, 10, 50)
    hits = {k: 0.0 for k in ks}; ndcg = {k: 0.0 for k in ks}
    it = torch.tensor(item_emb, dtype=torch.float32, device=device)
    n_items = it.shape[0]
    L = max(len(h) for _, h, _ in cases)
    H = np.zeros((len(cases), L), dtype=np.int64)
    M = np.zeros((len(cases), L), dtype=np.float32)
    T = np.array([t for _, _, t in cases], dtype=np.int64)
    for r, (_, h, _) in enumerate(cases):
        h = h[-L:]
        H[r, :len(h)] = h; M[r, :len(h)] = 1
    lens = torch.tensor(M.sum(1), dtype=torch.float32, device=device)
    Ht = torch.tensor(H, device=device); Mt = torch.tensor(M, device=device)
    Tt = torch.tensor(T, device=device)
    for st in range(0, len(cases), chunk):
        en = min(st + chunk, len(cases))
        hb, mb, tb = Ht[st:en], Mt[st:en], Tt[st:en]
        u = (it[hb] * mb.unsqueeze(-1)).sum(1) / lens[st:en, None]
        u = F.normalize(u, dim=-1)
        s = u @ it.T
        ts = s.gather(1, tb[:, None])
        mask = torch.zeros_like(s).scatter_(1, hb, mb)
        s = s.masked_fill(mask.bool(), -1e9)
        s.scatter_(1, tb[:, None], ts)
        rank = (s > ts).sum(1)
        for k in ks:
            hit = (rank < k).float()
            hits[k] += hit.sum().item()
            ndcg[k] += (hit / torch.log2(rank.float() + 2)).sum().item()
    n = len(cases)
    return {f"HR@{k}": hits[k] / n for k in ks} | {f"NDCG@{k}": ndcg[k] / n for k in ks}


def part4(x_t, x_i, x_c, cache, meta, model):
    n = len(x_t)
    gates0 = cache["gates"]; fused0 = cache["fused"]; ent0 = cache["entropy"]
    mean_gate0 = gates0.mean(0)
    top1_0 = gates0.argmax(1)
    cases = [(u, h, t) for (u, h, t) in load_inter("test") if t not in h]
    RES["proxy_n_cases"] = len(cases)
    RES["proxy_n_excluded_target_in_history"] = len(load_inter("test")) - len(cases)

    mean_t = x_t.mean(0, keepdims=True)
    mean_i = x_i.mean(0, keepdims=True)
    sigma_i = x_i.std(0, keepdims=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    noise = 0.5 * sigma_i * rng.standard_normal(size=x_i.shape).astype(np.float32)

    title_path = "/data02/sht/SAM-GR/data/Amazon/Sports_and_Outdoors/Sports_and_Outdoors.emb-Qwen3VL-text_title.npy"
    has_title = os.path.exists(title_path)
    if has_title:
        x_title = np.load(title_path).astype(np.float32)
    else:
        x_title = None

    conditions = [
        ("baseline", x_t, x_i),
        ("mask-text", np.repeat(mean_t, n, axis=0), x_i),
        ("mask-image", x_t, np.repeat(mean_i, n, axis=0)),
        ("shuffle-image", x_t, x_i[perm]),
        ("noise-image", x_t, (x_i + noise).astype(np.float32)),
    ]
    if x_title is not None:
        conditions.append(("truncate-description", x_title, x_i))
    else:
        conditions.append(("truncate-description", (0.5 * x_t + 0.5 * mean_t).astype(np.float32), x_i))

    rows = []; gate_deltas = []; drift_rows = []; perf_rows = []
    base_metrics = None
    for name, xt, xi in conditions:
        t = torch.from_numpy(center(xt)).float().cuda()
        i = torch.from_numpy(center(xi)).float().cuda()
        c = torch.from_numpy(center(x_c)).float().cuda()
        st = infer(model, t, i, c)
        g = st["gates"]; ent = gate_entropy(g)
        fused = production_fused(st["fused_raw"], x_c)
        delta = g.mean(0) - mean_gate0
        js_mean = js_divergence(g.mean(0), mean_gate0)
        l1 = np.abs(g - gates0).sum(1).mean()
        flip = float((g.argmax(1) != top1_0).mean())
        cos = (l2n(fused) * l2n(fused0)).sum(1)
        drift = 1.0 - cos
        metrics = retrieval_eval(l2n(fused), cases)
        if name == "baseline":
            base_metrics = metrics
        rows.append({
            "condition": name,
            **{f"gate_{NAMES[j]}": float(g[:, j].mean()) for j in range(E)},
            "mean_entropy": float(ent.mean()),
            "effective_experts": float(np.exp(ent.mean())),
            "js_vs_baseline": float(js_mean),
            "mean_l1_gate_change": float(l1),
            "routing_flip_rate": flip,
            "repr_cos_drift_mean": float(drift.mean()),
            "repr_cos_drift_median": float(np.median(drift)),
            "repr_cos_drift_p95": float(np.quantile(drift, 0.95)),
            **metrics,
        })
        gate_deltas.append({"condition": name, **{NAMES[j]: float(delta[j]) for j in range(E)}})
        drift_rows.append({"condition": name, **{f"drift_p{p}": float(np.quantile(drift, p / 100))
                                                  for p in [1, 5, 25, 50, 75, 95, 99]}})
        perf_rows.append({"condition": name, **metrics})
        del t, i, c, st, g, fused
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "part4_intervention_stats.csv"), index=False)
    gd = pd.DataFrame(gate_deltas)
    gd.to_csv(os.path.join(RESULTS, "part4_gate_deltas.csv"), index=False)
    pd.DataFrame(drift_rows).to_csv(os.path.join(RESULTS, "part4_drift_quantiles.csv"), index=False)
    pd.DataFrame(perf_rows).to_csv(os.path.join(RESULTS, "part4_retrieval_proxy.csv"), index=False)
    RES["part4_df"] = df
    RES["part4_base_metrics"] = base_metrics
    RES["part4_has_title"] = bool(x_title is not None)

    # Figure 4
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    # (a) absolute mean gates, baseline vs interventions
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(E)
    style = {"baseline": dict(color="black", ls="--", lw=2, marker="o")}
    for _, r in df.iterrows():
        y = [r[f"gate_{NAMES[j]}"] for j in range(E)]
        if r.condition == "baseline":
            ax.plot(x, y, label="baseline", **style["baseline"])
        else:
            ax.plot(x, y, marker="o", lw=1.4, label=r.condition)
    ax.axhline(1 / E, color="gray", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=20, ha="right")
    ax.set_ylabel("mean gate weight"); ax.set_title("(a) Before/after gate weights")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # (b) delta-gate heatmap
    ax = fig.add_subplot(gs[0, 1])
    sub = gd[gd.condition != "baseline"].set_index("condition")[NAMES]
    im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0))
    ax.set_xticks(np.arange(E)); ax.set_xticklabels(NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_title("(b) Δ gate weight (intervention − baseline)")
    for a in range(sub.shape[0]):
        for b in range(sub.shape[1]):
            ax.text(b, a, f"{sub.values[a,b]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # (c) representation drift
    ax = fig.add_subplot(gs[1, 0])
    sub = df[df.condition != "baseline"]
    xs = np.arange(len(sub))
    ax.bar(xs, sub["repr_cos_drift_mean"], color="#8172B3", alpha=0.9)
    ax.errorbar(xs, sub["repr_cos_drift_mean"],
                yerr=[sub["repr_cos_drift_mean"] - sub["repr_cos_drift_median"],
                      sub["repr_cos_drift_p95"] - sub["repr_cos_drift_mean"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(sub.condition, rotation=20, ha="right")
    ax.set_ylabel("1 − cosine(fused, baseline)")
    ax.set_title("(c) Representation cosine drift (mean, median, p95)")

    # (d) proxy performance delta
    ax = fig.add_subplot(gs[1, 1])
    sub = df[df.condition != "baseline"]
    xs = np.arange(len(sub)); w = 0.35
    for k, (metric, color) in enumerate([("HR@50", "#4C72B0"), ("NDCG@50", "#DD8452")]):
        base = base_metrics[metric]
        vals = (sub[metric].values - base) / max(base, 1e-9) * 100
        ax.bar(xs + (k - 0.5) * w, vals, w, color=color, label=f"{metric} Δ%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(sub.condition, rotation=20, ha="right")
    ax.set_ylabel("relative change vs. baseline (%)")
    ax.set_title("(d) Recommendation proxy change (feature-level retrieval)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Modality interventions on the OMoE-v2 routing", fontsize=13)
    fig.savefig(os.path.join(FIGURES, "fig4_intervention_gate_shift.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 5: explanatory cases per expert
# --------------------------------------------------------------------------- #
def part5(gates, ent, meta):
    top1 = gates.argmax(1)
    counts = np.bincount(top1, minlength=E)
    profiles = []
    cases = []
    for j in range(E):
        m = top1 == j
        if m.sum() > 0:
            cats = Counter(meta["cat2"][m]).most_common(1)[0][0]
            profiles.append({
                "expert": NAMES[j], "n_top1": int(m.sum()), "top1_freq": float(m.mean()),
                "mean_gate_when_top1": float(gates[m, j].mean()),
                "mean_text_len": float(meta["text_len"][m].mean()),
                "mean_popularity": float(meta["pop"][m].mean()),
                "mean_text_image_cos": float(meta["consistency"][m].mean()),
                "mean_entropy": float(ent[m].mean()),
                "top_category": cats,
            })
            # largest margin over the runner-up expert
            sg = np.sort(gates[m], axis=1)
            margin = sg[:, -1] - sg[:, -2]
            idxs = np.where(m)[0]
            idx = idxs[int(np.argmax(margin))]
        else:
            profiles.append({
                "expert": NAMES[j], "n_top1": 0, "top1_freq": 0.0,
                "mean_gate_when_top1": float("nan"), "mean_text_len": float("nan"),
                "mean_popularity": float("nan"), "mean_text_image_cos": float("nan"),
                "mean_entropy": float("nan"), "top_category": "-",
            })
            idx = int(np.argmax(gates[:, j]))
        cases.append({
            "expert": NAMES[j], "item_id": int(idx),
            "title": short(meta["titles"][idx], 120),
            "category": meta["cat2"][idx],
            "popularity": int(meta["pop"][idx]),
            "text_len": int(meta["text_len"][idx]),
            "text_image_cos": float(meta["consistency"][idx]),
            **{f"gate_{NAMES[k]}": float(gates[idx, k]) for k in range(E)},
            "entropy": float(ent[idx]),
            "top_expert": NAMES[int(gates[idx].argmax())],
        })
    pdf = pd.DataFrame(profiles)
    cdf = pd.DataFrame(cases)
    pdf.to_csv(os.path.join(RESULTS, "part5_expert_profiles.csv"), index=False)
    cdf.to_csv(os.path.join(RESULTS, "part5_expert_cases.csv"), index=False)
    RES["part5_profiles"] = pdf
    RES["part5_cases"] = cdf

    # Figure 5
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(E, 3, width_ratios=[1.1, 1.0, 2.2], hspace=0.55, wspace=0.12)
    for j in range(E):
        p = profiles[j]
        ax0 = fig.add_subplot(gs[j, 0]); ax0.axis("off")
        ax0.text(0, 0.95, f"{NAMES[j]}", fontsize=12, weight="bold", color=COLORS[j], va="top")
        ax0.text(0, 0.72,
                 f"top-1: {p['n_top1']} items ({p['top1_freq']*100:.1f}%)\n"
                 f"mean text len: {p['mean_text_len']:.0f}\n"
                 f"mean popularity: {p['mean_popularity']:.0f}\n"
                 f"mean text–image cos: {p['mean_text_image_cos']:.3f}\n"
                 f"top category: {p['top_category']}",
                 fontsize=8.5, va="top")

        ax1 = fig.add_subplot(gs[j, 1])
        c = cases[j]
        vals = [c[f"gate_{NAMES[k]}"] for k in range(E)]
        ax1.barh(np.arange(E), vals, color=[COLORS[k] if k == j else "#BDBDBD" for k in range(E)])
        ax1.set_yticks(np.arange(E)); ax1.set_yticklabels(NAMES, fontsize=7)
        ax1.invert_yaxis(); ax1.set_xlim(0, max(vals) * 1.25 + 1e-6)
        ax1.set_xlabel("gate", fontsize=8); ax1.tick_params(axis="x", labelsize=7)
        ax1.set_title(f"case item #{c['item_id']}  (H={c['entropy']:.2f})", fontsize=8)

        ax2 = fig.add_subplot(gs[j, 2]); ax2.axis("off")
        ax2.text(0, 0.95, textwrap.fill(c["title"], 72), fontsize=9.5, va="top")
        ax2.text(0, 0.45,
                 f"category: {c['category']}\n"
                 f"popularity: {c['popularity']}   text length: {c['text_len']}\n"
                 f"text–image cosine: {c['text_image_cos']:.3f}\n"
                 f"dominant expert: {c['top_expert']}",
                 fontsize=8.5, va="top")
    fig.suptitle("Explanatory cases: one representative item per expert", fontsize=14)
    fig.savefig(os.path.join(FIGURES, "fig5_expert_cases.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("loading data...", flush=True)
    x_t, x_i, x_c, cache, meta = load_all()
    gates = cache["gates"]; ent = cache["entropy"]; expert_norms = cache["expert_norms"]
    n = len(gates)

    # load the reproduced model
    model = build_model(x_t.shape[1], dev)
    ckpt = torch.load(os.path.join(RESULTS, "omoe_model.pt"), map_location=dev)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    t = torch.from_numpy(center(x_t)).float().to(dev)
    i = torch.from_numpy(center(x_i)).float().to(dev)
    c = torch.from_numpy(center(x_c)).float().to(dev)

    print("part 1...", flush=True); part1(gates, expert_norms, ent, model, t, i, c)
    print("part 2...", flush=True); part2(gates, ent, meta)
    print("part 3...", flush=True); part3(gates, ent, meta)
    print("part 4...", flush=True); part4(x_t, x_i, x_c, cache, meta, model)
    print("part 5...", flush=True); part5(gates, ent, meta)
    print("report...", flush=True)
    from report_builder import build_report
    build_report(RES)
    print("done", flush=True)


if __name__ == "__main__":
    main()
