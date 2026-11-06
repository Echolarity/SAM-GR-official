"""Build CSGA-aligned features, train OMoE-v2, and cache IndSci routing."""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omoe_core_ind import (ANALYSIS, DATA, RESULTS, gate_entropy, infer,
                           load_csga, load_emb, production_fused, train_model)

sys.path.insert(0, "/data02/sht/SAM-GR/rq/text2emb")
from csga import csga_align  # noqa: E402


def ensure_csga():
    p = os.path.join(RESULTS, "csga_aligned.npy")
    if os.path.exists(p):
        print("csga cache exists:", p, flush=True)
        return
    x_m = load_emb("multimodal")
    x_c = np.load("/data02/sht/SAM-GR/sasrec_models/Industrial_and_Scientific/"
                  "sasrec_item_emb.npy").astype(np.float32)
    print("csga inputs", x_m.shape, x_c.shape, flush=True)
    args = SimpleNamespace(csga_topk=10, csga_epochs=60, csga_lr=1e-3, csga_batch_size=512)
    t0 = time.time()
    aligned = csga_align(x_m, x_c, args)
    print(f"csga done in {time.time()-t0:.1f}s", flush=True)
    np.save(p, aligned.astype(np.float32))
    print("saved", p, flush=True)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    ensure_csga()
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    x_c = load_csga()
    print("features", x_t.shape, x_i.shape, x_c.shape, flush=True)

    t0 = time.time()
    model, (t, i, c), history = train_model(x_t, x_i, x_c, device=dev)
    print(f"training took {time.time()-t0:.1f}s", flush=True)

    st = infer(model, t, i, c)
    g = st["gates"]; h = gate_entropy(g); mean_gate = g.mean(0)
    print("mean gates", np.round(mean_gate, 6), flush=True)
    print("mean item entropy", float(h.mean()), "effective experts", float(np.exp(h.mean())), flush=True)

    fused = production_fused(st["fused_raw"], x_c)
    np.savez_compressed(
        os.path.join(RESULTS, "omoe_cache.npz"),
        gates=g.astype(np.float32),
        expert_norms=st["expert_norms"].astype(np.float32),
        fused_raw=st["fused_raw"].astype(np.float32),
        fused=fused.astype(np.float32),
        entropy=h.astype(np.float32),
    )
    torch.save({"state_dict": model.state_dict(),
                "config": {"dim": x_t.shape[1], "shared": 2, "ortho_weight": 0.05,
                           "load_weight": 0.1, "seed": 42, "epochs": 60}},
               os.path.join(RESULTS, "omoe_model.pt"))
    json.dump({"mean_gate": mean_gate.tolist(), "mean_item_entropy": float(h.mean()),
               "effective_experts": float(np.exp(h.mean())), "history": history},
              open(os.path.join(RESULTS, "omoe_train_history.json"), "w"), indent=2)
    print("saved", os.path.join(RESULTS, "omoe_cache.npz"), flush=True)


if __name__ == "__main__":
    main()
