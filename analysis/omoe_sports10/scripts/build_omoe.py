"""Train/reproduce the OMoE-v2 model and cache per-item routing statistics."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omoe_core import (RESULTS, EXPERT_NAMES, gate_entropy, infer, load_csga,
                       load_emb, production_fused, train_model)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device", dev)
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    x_c = load_csga()
    print("features", x_t.shape, x_i.shape, x_c.shape)

    t0 = time.time()
    model, (t, i, c), history = train_model(x_t, x_i, x_c, device=dev)
    print(f"training took {time.time()-t0:.1f}s")

    stats = infer(model, t, i, c)
    g = stats["gates"]
    h = gate_entropy(g)
    mean_gate = g.mean(0)
    print("mean gates", np.round(mean_gate, 6))
    print("mean item entropy", float(h.mean()), "effective experts", float(np.exp(h.mean())))
    # Compare with the existing exp_d record
    rec = json.load(open("/data02/sht/SAM-GR/analysis/samgr_sports10/results/exp_d_omoe_stats.json"))
    print("recorded full OMoE gates", np.round(rec["full OMoE"]["mean_gate"], 6))
    print("recorded item entropy", rec["full OMoE"]["mean_item_entropy"])

    fused = production_fused(stats["fused_raw"], x_c)
    np.savez_compressed(
        os.path.join(RESULTS, "omoe_cache.npz"),
        gates=g.astype(np.float32),
        expert_norms=stats["expert_norms"].astype(np.float32),
        fused_raw=stats["fused_raw"].astype(np.float32),
        fused=fused.astype(np.float32),
        entropy=h.astype(np.float32),
    )
    torch.save({"state_dict": model.state_dict(),
                "config": {"dim": x_t.shape[1], "shared": 2, "ortho_weight": 0.05,
                           "load_weight": 0.1, "seed": 42, "epochs": 60}},
               os.path.join(RESULTS, "omoe_model.pt"))
    json.dump({"mean_gate": mean_gate.tolist(), "mean_item_entropy": float(h.mean()),
               "effective_experts": float(np.exp(h.mean())),
               "history": history}, open(os.path.join(RESULTS, "omoe_train_history.json"), "w"), indent=2)
    print("saved", os.path.join(RESULTS, "omoe_cache.npz"))


if __name__ == "__main__":
    main()
