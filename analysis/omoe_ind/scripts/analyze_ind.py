"""Run the OMoE gating analysis on Industrial_and_Scientific.

Reuses the generic Part 1/2/3/5 implementations from the Sports analysis and
adds an IndSci-specific Part 4 (interventions + retrieval proxy).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data02/sht/SAM-GR/analysis/omoe_sports10/scripts")
import omoe_core_ind as core
import run_analysis as ra

ra.RESULTS = core.RESULTS
ra.FIGURES = core.FIGURES
ra.NAMES = core.EXPERT_NAMES
ra.E = core.N_EXPERTS
ra.COLORS = core.EXPERT_COLORS
ra.RES = {}
ra.DATASET_LABEL = "IndSci"
NAMES = core.EXPERT_NAMES
E = core.N_EXPERTS
COLORS = core.EXPERT_COLORS
RESULTS = core.RESULTS
FIGURES = core.FIGURES


def clean(s):
    return str(s or "").replace("\n", " ").strip()


def short(s, n=95):
    s = clean(s)
    return s[:n] + ("..." if len(s) > n else "")


def load_all_ind():
    x_t = core.load_emb("text_only")
    x_i = core.load_emb("image_only")
    x_c = core.load_csga()
    cache = np.load(os.path.join(RESULTS, "omoe_cache.npz"))
    items = core.load_items()
    n = len(x_t)
    titles, descs, cats, has_img, text_len = [], [], [], [], []
    for i in range(n):
        it = items[str(i)]
        t = clean(it.get("title", "")); d = clean(it.get("description", ""))
        c = clean(it.get("categories", ""))
        titles.append(t); descs.append(d); cats.append(c)
        has_img.append(bool(it.get("image_paths")))
        text_len.append(len(t) + len(d))
    cat2 = []
    for c in cats:
        parts = [p.strip() for p in c.split(",") if p.strip()]
        cat2.append(parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown"))
    pop = np.zeros(n, dtype=np.int64)
    for _, hist, tgt in core.load_inter("train"):
        for j in hist:
            pop[j] += 1
        pop[tgt] += 1
    consistency = (core.l2n(x_t) * core.l2n(x_i)).sum(1)
    meta = {
        "titles": np.asarray(titles, dtype=object), "descs": np.asarray(descs, dtype=object),
        "cats": np.asarray(cats, dtype=object), "cat2": np.asarray(cat2, dtype=object),
        "has_img": np.asarray(has_img), "text_len": np.asarray(text_len, dtype=np.int64),
        "pop": pop, "consistency": consistency,
    }
    return x_t, x_i, x_c, cache, meta


def part4_ind(x_t, x_i, x_c, cache, meta, model):
    n = len(x_t)
    gates0 = cache["gates"]; fused0 = cache["fused"]; ent0 = cache["entropy"]
    mean_gate0 = gates0.mean(0); top1_0 = gates0.argmax(1)
    cases = [(u, h[-10:], t) for (u, h, t) in core.load_inter("test") if t not in h]
    ra.RES["proxy_n_cases"] = len(cases)
    ra.RES["proxy_n_excluded_target_in_history"] = len(core.load_inter("test")) - len(cases)

    mean_t = x_t.mean(0, keepdims=True)
    mean_i = x_i.mean(0, keepdims=True)
    sigma_i = x_i.std(0, keepdims=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    noise = 0.5 * sigma_i * rng.standard_normal(size=x_i.shape).astype(np.float32)

    conditions = [
        ("baseline", x_t, x_i),
        ("mask-text", np.repeat(mean_t, n, axis=0), x_i),
        ("mask-image", x_t, np.repeat(mean_i, n, axis=0)),
        ("shuffle-image", x_t, x_i[perm]),
        ("noise-image", x_t, (x_i + noise).astype(np.float32)),
        # truncate-description is not reported: no title-only embedding cache for
        # IndSci, and a feature-scale blend is a no-op after LayerNorm.
    ]
    rows = []; gate_deltas = []; drift_rows = []; perf_rows = []
    base_metrics = None
    for name, xt, xi in conditions:
        t = torch.from_numpy(core.center(xt)).float().cuda()
        i = torch.from_numpy(core.center(xi)).float().cuda()
        c = torch.from_numpy(core.center(x_c)).float().cuda()
        st = core.infer(model, t, i, c)
        g = st["gates"]; ent = core.gate_entropy(g)
        fused = core.production_fused(st["fused_raw"], x_c)
        delta = g.mean(0) - mean_gate0
        js_mean = core.js_divergence(g.mean(0), mean_gate0)
        l1 = np.abs(g - gates0).sum(1).mean()
        flip = float((g.argmax(1) != top1_0).mean())
        cos = (core.l2n(fused) * core.l2n(fused0)).sum(1)
        drift = 1.0 - cos
        metrics = ra.retrieval_eval(core.l2n(fused), cases)
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
    pd.DataFrame(gate_deltas).to_csv(os.path.join(RESULTS, "part4_gate_deltas.csv"), index=False)
    pd.DataFrame(drift_rows).to_csv(os.path.join(RESULTS, "part4_drift_quantiles.csv"), index=False)
    pd.DataFrame(perf_rows).to_csv(os.path.join(RESULTS, "part4_retrieval_proxy.csv"), index=False)
    ra.RES["part4_df"] = df
    ra.RES["part4_base_metrics"] = base_metrics
    ra.RES["part4_has_title"] = False

    # compact figure
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(E)
    for _, r in df.iterrows():
        y = [r[f"gate_{NAMES[j]}"] for j in range(E)]
        if r.condition == "baseline":
            ax.plot(x, y, color="black", ls="--", lw=2, marker="o", label="baseline")
        else:
            ax.plot(x, y, marker="o", lw=1.4, label=r.condition)
    ax.axhline(1 / E, color="gray", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=20, ha="right")
    ax.set_ylabel("mean gate weight"); ax.set_title("(a) Before/after gate weights (IndSci)")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    sub = pd.DataFrame(gate_deltas)
    sub = sub[sub.condition != "baseline"].set_index("condition")[NAMES]
    from matplotlib.colors import TwoSlopeNorm
    im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0))
    ax.set_xticks(np.arange(E)); ax.set_xticklabels(NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(sub))); ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_title("(b) Δ gate weight (intervention − baseline)")
    for a in range(sub.shape[0]):
        for b in range(sub.shape[1]):
            ax.text(b, a, f"{sub.values[a,b]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax = fig.add_subplot(gs[1, 0]); sub = df[df.condition != "baseline"]
    xs = np.arange(len(sub))
    ax.bar(xs, sub["repr_cos_drift_mean"], color="#8172B3", alpha=0.9)
    ax.errorbar(xs, sub["repr_cos_drift_mean"],
                yerr=[sub["repr_cos_drift_mean"] - sub["repr_cos_drift_median"],
                      sub["repr_cos_drift_p95"] - sub["repr_cos_drift_mean"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(sub.condition, rotation=20, ha="right")
    ax.set_ylabel("1 − cosine(fused, baseline)")
    ax.set_title("(c) Representation cosine drift (mean, median, p95)")

    ax = fig.add_subplot(gs[1, 1]); xs = np.arange(len(sub)); w = 0.35
    for k, (metric, color) in enumerate([("HR@50", "#4C72B0"), ("NDCG@50", "#DD8452")]):
        base = base_metrics[metric]
        vals = (sub[metric].values - base) / max(base, 1e-9) * 100
        ax.bar(xs + (k - 0.5) * w, vals, w, color=color, label=f"{metric} Δ%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(sub.condition, rotation=20, ha="right")
    ax.set_ylabel("relative change vs. baseline (%)")
    ax.set_title("(d) Recommendation proxy change (IndSci)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Modality interventions on IndSci OMoE-v2 routing", fontsize=13)
    fig.savefig(os.path.join(FIGURES, "fig4_intervention_gate_shift.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("loading IndSci data...", flush=True)
    x_t, x_i, x_c, cache, meta = load_all_ind()
    gates = cache["gates"]; ent = cache["entropy"]; expert_norms = cache["expert_norms"]
    model = core.build_model(x_t.shape[1], dev)
    ckpt = torch.load(os.path.join(RESULTS, "omoe_model.pt"), map_location=dev)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    t = torch.from_numpy(core.center(x_t)).float().to(dev)
    i = torch.from_numpy(core.center(x_i)).float().to(dev)
    c = torch.from_numpy(core.center(x_c)).float().to(dev)

    print("part 1...", flush=True); ra.part1(gates, expert_norms, ent, model, t, i, c)
    print("part 2...", flush=True); ra.part2(gates, ent, meta)
    print("part 3...", flush=True); ra.part3(gates, ent, meta)
    print("part 4...", flush=True); part4_ind(x_t, x_i, x_c, cache, meta, model)
    print("part 5...", flush=True); ra.part5(gates, ent, meta)
    print("done", flush=True)


if __name__ == "__main__":
    main()
