"""Experiment B: modality contribution / text-only SID vs full multimodal+collab SID.

IMPORTANT: exact SID-token equality is NOT a valid cross-space measure, because two
independently trained RQ-VAEs assign arbitrary (permuted) codebook labels.  We
therefore also report permutation-invariant partition agreement (ARI/NMI) at each
SID prefix level, plus token agreement against the chance level 1/256.

Available strict-10-core Sports runs:
  * full model: CSGA+OMoE v2 (text+image+collab) + 2 epochs -> results/..._strict
  * TIGER text-only SID (from-scratch T5-style seq2seq)      -> experiments_tiger
A retrained LLM "drop one modality" variant is NOT available; this is documented.
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, DATA, FIGURES, RESULTS  # noqa: E402

MAIN_IDX = os.path.join(DATA, "Sports_and_Outdoors.index.json")
TEXT_IDX = os.path.join(BASE, "analysis", "samgr_sports10", "results",
                        "text_only_sid", "Sports_and_Outdoors.index.json")
MAIN_METRICS = os.path.join(BASE, "results", "qwen2.5-0.5b", "full",
                            "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict", "metrics.json")
TIGER_METRICS = os.path.join(BASE, "experiments_tiger", "metrics_tiger_Sports_and_Outdoors.json")


def sid_stats(idx):
    sids = ["".join(v) for v in idx.values()]
    u = len(set(sids)); c = Counter(sids)
    return {"items": len(sids), "unique_sids": u, "collision": 1 - u / len(sids),
            "max_items_per_sid": max(c.values())}


def prefix_labels(idx, items, level):
    mp = {}; out = []
    for it in items:
        k = tuple(idx[it][:level])
        if k not in mp:
            mp[k] = len(mp)
        out.append(mp[k])
    return out


def main():
    main_idx = json.load(open(MAIN_IDX))
    text_idx = json.load(open(TEXT_IDX))
    items = sorted(main_idx.keys(), key=lambda x: int(x))
    n = len(items)
    assert len(text_idx) == n

    same = 0; tok_same = [0, 0, 0, 0]
    for it in items:
        a, b = main_idx[it], text_idx[it]
        if "".join(a) == "".join(b):
            same += 1
        for l in range(4):
            if a[l] == b[l]:
                tok_same[l] += 1
    set_a = set("".join(v) for v in main_idx.values())
    set_b = set("".join(v) for v in text_idx.values())

    ari, nmi = [], []
    for L in [1, 2, 3, 4]:
        a = prefix_labels(main_idx, items, L)
        b = prefix_labels(text_idx, items, L)
        ari.append(float(adjusted_rand_score(a, b)))
        nmi.append(float(normalized_mutual_info_score(a, b)))

    # item-group sets (final SID)
    def group_sets(idx):
        g = {}
        for it, s in idx.items():
            g.setdefault(tuple(s), set()).add(int(it))
        return set(frozenset(v) for v in g.values())
    G1, G2 = group_sets(main_idx), group_sets(text_idx)

    stats = {
        "main_sid": sid_stats(main_idx),
        "textonly_sid": sid_stats(text_idx),
        "same_item_same_sid_rate": same / n,
        "level_token_agreement": [t / n for t in tok_same],
        "chance_token_agreement": 1 / 256,
        "sid_set_jaccard": len(set_a & set_b) / len(set_a | set_b),
        "ari_per_level": ari,
        "nmi_per_level": nmi,
        "common_item_groups": len(G1 & G2),
        "n_item_groups_main": len(G1),
        "n_item_groups_text": len(G2),
        "n": n,
    }
    json.dump(stats, open(os.path.join(RESULTS, "exp_b_sid_overlap.json"), "w"), indent=2)
    pd.DataFrame({"prefix_level": [1, 2, 3, 4], "ARI": ari, "NMI": nmi}).to_csv(
        os.path.join(RESULTS, "exp_b_partition_agreement.csv"), index=False)
    print(json.dumps(stats, indent=2))

    # performance
    main_m = json.load(open(MAIN_METRICS))
    tiger_m = json.load(open(TIGER_METRICS))
    rows = []
    for k in [1, 3, 5, 10, 20, 50]:
        for metric in ["HR", "NDCG"]:
            rows.append({"model": "full (CSGA+OMoE, text+image+collab)",
                         "metric": f"{metric}@{k}", "value": main_m[f"{metric}@{k}"]})
            rows.append({"model": "TIGER (text-only SID)",
                         "metric": f"{metric}@{k}", "value": tiger_m[f"{metric}@{k}"]})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "exp_b_performance.csv"), index=False)

    ks = [1, 3, 5, 10, 20, 50]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, metric in zip(axes, ["HR", "NDCG"]):
        w = 0.38; xs = np.arange(len(ks))
        ax.bar(xs - w / 2, [main_m[f"{metric}@{k}"] for k in ks], w, label="full (text+image+collab)")
        ax.bar(xs + w / 2, [tiger_m[f"{metric}@{k}"] for k in ks], w, label="TIGER (text-only)")
        ax.set_xticks(xs); ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_ylabel(metric); ax.set_title(f"{metric}@K (Sports strict 10-core)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_b_performance.png"), dpi=200)
    plt.close(fig)

    # SID-space analysis (corrected)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(1, 5); w = 0.38
    axes[0].bar(x - w / 2, ari, w, label="ARI")
    axes[0].bar(x + w / 2, nmi, w, label="NMI")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["a", "a+b", "a+b+c", "SID (4)"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Item-partition agreement (permutation-invariant)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].annotate(
        f"exact same SID = {same}/{n}\n(codebook labels permuted:\ncommon item groups "
        f"{len(G1 & G2)}/{len(G1)})",
        xy=(0.03, 0.97), xycoords="axes fraction", fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
    axes[1].bar(["a", "b", "c", "d"], stats["level_token_agreement"], color="#55A868",
                label="token agreement")
    axes[1].axhline(1 / 256, ls="--", color="gray", label="chance (1/256)")
    axes[1].set_ylim(0, max(0.008, max(stats["level_token_agreement"]) * 1.6))
    axes[1].set_title("SID token-string agreement ≈ chance", fontsize=9)
    axes[1].set_ylabel("fraction of items")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_b_sid_overlap.png"), dpi=200)
    plt.close(fig)
    print("EXP_B_DONE")


if __name__ == "__main__":
    main()
