import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("results/rerank_valid_to_test.json"))
t = R["test"]
methods = ["LLM 2ep", "LLM 4ep", "RRF union", "Reranker"]
keys = ["llm_2ep", "llm_4ep", "rrf_beam_union", "reranker"]
colors = ["#9aa5b1", "#6b7a8f", "#4c78a8", "#e45756"]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax = axes[0]
vals = [t[k]["HR@1"] for k in keys]
bars = ax.bar(methods, vals, color=colors, width=0.62)
ax.set_ylabel("HR@1")
ax.set_title("(a) Top-1 hit rate (test)", fontsize=10)
ax.set_ylim(0, max(vals) * 1.35)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.04, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.25)
ax.tick_params(axis="x", rotation=15)

ax = axes[1]
x = np.arange(len(methods)); w = 0.36
v50 = [t[k]["HR@50"] for k in keys]
nd = [t[k]["NDCG@50"] for k in keys]
b1 = ax.bar(x - w/2, v50, w, label="HR@50", color="#4c78a8")
b2 = ax.bar(x + w/2, nd, w, label="NDCG@50", color="#f58518")
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15)
ax.set_ylabel("score")
ax.set_title("(b) Top-50 metrics (test)", fontsize=10)
ax.set_ylim(0, max(v50) * 1.28)
for b, v in list(zip(b1, v50)) + list(zip(b2, nd)):
    ax.text(b.get_x() + b.get_width()/2, v + max(v50)*0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Candidate reranking: valid-trained, test evaluated once", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_rerank_gain.png", dpi=300)
print("saved figures/fig_rerank_gain.png")
