"""Experiment D: OMoE expert complementarity, routing, and ortho/load ablations.

The OMoE fusion is cheap to retrain (~minutes) unlike the LLM SFT, so we run:
  * full OMoE (ortho=0.05, load=0.1)
  * no orthogonality (ortho=0)
  * no load balancing (load=0)
on the same cached text / image / CSGA-aligned features and report expert-output
similarity, routing utilisation/entropy and fused-feature sanity.
Downstream LLM retraining for these variants is NOT run (reported as such).
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "rq", "text2emb"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES, RESULTS, load_collab, load_emb  # noqa: E402
from csga import csga_align  # noqa: E402
from omoe_v2 import OMoE2  # noqa: E402

EXPERT_NAMES = ["text", "image", "csga", "shared-1", "shared-2"]


def center(x):
    return x - x.mean(0, keepdims=True)


def train_omoe(x_t, x_i, x_c, ortho, load, epochs=60, bs=512, lr=1e-3, dev="cuda:0"):
    n, dim = x_t.shape
    t = torch.from_numpy(center(x_t)).float().to(dev)
    i = torch.from_numpy(center(x_i)).float().to(dev)
    c = torch.from_numpy(center(x_c)).float().to(dev)
    model = OMoE2(dim, ortho_weight=ortho, load_weight=load).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for st in range(0, n, bs):
            b = perm[st:st + bs]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    # inference with instrumentation
    model.eval()
    E = model.num_experts
    sim_sum = np.zeros((E, E)); sim_cnt = 0
    gate_sum = np.zeros(E); gate_cnt = 0
    ent_sum = 0.0; ent_cnt = 0
    f_sum = np.zeros(dim); f_sumsq = np.zeros(dim); f_cnt = 0
    sample = []
    with torch.no_grad():
        for st in range(0, n, bs):
            en = min(st + bs, n)
            fused, _ = model(t[st:en], i[st:en], c[st:en])
            experts = model._monitor_experts
            gates = model._monitor_gates
            B = fused.size(0)
            for p in range(E):
                for q in range(E):
                    sim_sum[p, q] += F.cosine_similarity(experts[p], experts[q], dim=-1).sum().item()
            sim_cnt += B
            gate_sum += gates.sum(0).cpu().numpy()
            gate_cnt += B
            ent = -(gates * (gates + 1e-9).log()).sum(-1)
            ent_sum += ent.sum().item(); ent_cnt += B
            f = fused.cpu().numpy()
            f_sum += f.sum(0); f_sumsq += (f ** 2).sum(0); f_cnt += B
            if len(sample) < 2000:
                sample.append(f[: min(500, B)])
    sim = sim_sum / sim_cnt
    mean_gate = gate_sum / gate_cnt
    mean_ent = ent_sum / ent_cnt
    mean_gate_ent = float(-(mean_gate * np.log(mean_gate + 1e-9)).sum())
    fs = np.concatenate(sample, 0)[:1000]
    fn = fs / (np.linalg.norm(fs, axis=1, keepdims=True) + 1e-9)
    aff = fn @ fn.T; np.fill_diagonal(aff, -2)
    nn = aff.max(1)
    return {
        "expert_sim": sim,
        "mean_gate": mean_gate,
        "mean_gate_entropy": mean_gate_ent,
        "mean_item_entropy": mean_ent,
        "fused_std": float(np.sqrt(np.maximum(f_sumsq / f_cnt - (f_sum / f_cnt) ** 2, 0).mean())),
        "fused_mean_cos": float(aff[aff > -1.5].mean()),
        "fused_nn_dup": float(np.mean(nn > 0.99)),
    }


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    x_m = load_emb("multimodal")
    collab = load_collab()

    cache = os.path.join(RESULTS, "csga_aligned.npy")
    if os.path.exists(cache):
        aligned = np.load(cache)
        print("loaded cached CSGA-aligned features", aligned.shape)
    else:
        args = SimpleNamespace(csga_topk=10, csga_epochs=60, csga_lr=1e-3)
        aligned = csga_align(x_m, collab, args)
        np.save(cache, aligned)
        print("computed CSGA-aligned features", aligned.shape)

    variants = {
        "full OMoE": (0.05, 0.1),
        "no orthogonality": (0.0, 0.1),
        "no load balancing": (0.05, 0.0),
    }
    stats = {}
    for name, (ow, lw) in variants.items():
        t0 = time.time()
        s = train_omoe(x_t, x_i, aligned, ow, lw, dev=dev)
        s["ortho_weight"] = ow; s["load_weight"] = lw
        stats[name] = s
        print(f"[{name}] {time.time()-t0:.0f}s  gate={np.round(s['mean_gate'],3)} "
              f"gateH={s['mean_gate_entropy']:.3f} itemH={s['mean_item_entropy']:.3f} "
              f"cos={s['fused_mean_cos']:.3f} nn_dup={s['fused_nn_dup']:.3f}")

    rows = []
    for name, s in stats.items():
        rows.append({"variant": name, **{f"gate_{e}": float(s["mean_gate"][j]) for j, e in enumerate(EXPERT_NAMES)},
                     "gate_entropy": s["mean_gate_entropy"], "item_entropy": s["mean_item_entropy"],
                     "fused_std": s["fused_std"], "fused_mean_cos": s["fused_mean_cos"],
                     "fused_nn_dup": s["fused_nn_dup"], "ortho_weight": s["ortho_weight"],
                     "load_weight": s["load_weight"]})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "exp_d_omoe_variants.csv"), index=False)
    json.dump({k: {"expert_sim": s["expert_sim"].tolist(),
                   "mean_gate": s["mean_gate"].tolist(),
                   "mean_gate_entropy": s["mean_gate_entropy"],
                   "mean_item_entropy": s["mean_item_entropy"],
                   "fused_std": s["fused_std"], "fused_mean_cos": s["fused_mean_cos"],
                   "fused_nn_dup": s["fused_nn_dup"]}
               for k, s in stats.items()},
              open(os.path.join(RESULTS, "exp_d_omoe_stats.json"), "w"), indent=2)

    # expert similarity heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for ax, (name, s) in zip(axes, stats.items()):
        im = ax.imshow(s["expert_sim"], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(EXPERT_NAMES))); ax.set_xticklabels(EXPERT_NAMES, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(EXPERT_NAMES))); ax.set_yticklabels(EXPERT_NAMES, fontsize=7)
        ax.set_title(name, fontsize=9)
        for p in range(len(EXPERT_NAMES)):
            for q in range(len(EXPERT_NAMES)):
                ax.text(q, p, f"{s['expert_sim'][p,q]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Mean pairwise cosine between expert outputs", fontsize=10)
    fig.savefig(os.path.join(FIGURES, "fig_d_expert_similarity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # routing + entropy
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    xs = np.arange(len(EXPERT_NAMES)); w = 0.26
    for j, (name, s) in enumerate(stats.items()):
        axes[0].bar(xs + (j - 1) * w, s["mean_gate"], w, label=name)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(EXPERT_NAMES, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("mean routing probability"); axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Expert utilisation", fontsize=9)
    axes[1].bar(range(len(stats)), [s["mean_gate_entropy"] for s in stats.values()],
                color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_xticks(range(len(stats))); axes[1].set_xticklabels(list(stats.keys()), rotation=18, ha="right", fontsize=7)
    axes[1].set_ylabel("entropy of mean gate")
    axes[1].set_title("Routing concentration", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_d_routing.png"), dpi=200)
    plt.close(fig)
    print("EXP_D_DONE")


if __name__ == "__main__":
    main()
