"""Counterfactual modality decomposition for the no-MoE (direct-only) variant.

Trains the existing structure ablation `12_no_experts` (no expert mixture; the
fused branch is the direct projection) and applies the same dataset-mean
replacement counterfactual as the main study.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "analysis", "omoe_ablation"))
sys.path.insert(0, os.path.join(BASE, "analysis", "omoe_load_balance", "scripts"))

from ablation_omoe import AblOMoE, VariantConfig, _center  # noqa: E402
from lb_core import load_all, cosine_rows  # noqa: E402

ROOT = os.path.join(BASE, "analysis", "omoe_load_balance")
OUT = os.path.join(ROOT, "results", "analysis")


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_t, x_i, x_c, meta = load_all()
    cfg = VariantConfig(name="12_no_experts", group="structure",
                        num_modality_experts=0, num_shared_experts=0,
                        seed=42, epochs=60)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    t = torch.from_numpy(_center(x_t)).to(dev)
    i = torch.from_numpy(_center(x_i)).to(dev)
    c = torch.from_numpy(_center(x_c)).to(dev)
    model = AblOMoE(x_t.shape[1], cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    n = len(x_t)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for st in range(0, n, cfg.batch_size):
            b = perm[st:st + cfg.batch_size]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 20 == 0:
            print("epoch", ep, flush=True)

    model.eval()
    with torch.no_grad():
        branch = []
        for st in range(0, n, cfg.batch_size):
            b, _ = model(t[st:st + cfg.batch_size], i[st:st + cfg.batch_size],
                         c[st:st + cfg.batch_size])
            branch.append(b.cpu().numpy())
        branch = np.concatenate(branch, 0)
        scale = cfg.target_std / float(branch.std())
        z = x_c + cfg.residual * scale * branch

        def run(et, ei, ec):
            out = []
            for st in range(0, n, cfg.batch_size):
                en = min(st + cfg.batch_size, n)
                b, _ = model(et[st:en], ei[st:en], ec[st:en])
                out.append((x_c[st:en] + cfg.residual * scale * b.cpu().numpy()))
            return np.concatenate(out, 0)

        z_nt = run(torch.zeros_like(t), i, c)
        z_nv = run(t, torch.zeros_like(i), c)
        z_na = run(t, i, torch.zeros_like(c))
    deltas = np.stack([1.0 - cosine_rows(z, z_nt),
                       1.0 - cosine_rows(z, z_nv),
                       1.0 - cosine_rows(z, z_na)], 1)
    q = deltas / (deltas.sum(1, keepdims=True) + 1e-9)
    ent = -(q * np.log(q + 1e-9)).sum(1) / np.log(3)
    mdi = 1.0 - ent
    res = {
        "variant": "12_no_experts (direct only, no MoE)",
        "lambda_load": None, "seed": 42, "n_items": n, "scale": scale,
        "q_mean": q.mean(0).tolist(), "q_std": q.std(0).tolist(),
        "q_p10": np.quantile(q, 0.10, axis=0).tolist(),
        "q_p50": np.quantile(q, 0.50, axis=0).tolist(),
        "q_p90": np.quantile(q, 0.90, axis=0).tolist(),
        "cf_entropy_mean": float(ent.mean()), "cf_mdi_mean": float(mdi.mean()),
        "cf_mdi_std": float(mdi.std()), "cf_max_share_mean": float(q.max(1).mean()),
        "deltas_mean": deltas.mean(0).tolist(),
    }
    json.dump(res, open(os.path.join(OUT, "no_moe_seed42.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "no_moe_seed42_items.npz"),
             q=q.astype(np.float32), cf_mdi=mdi.astype(np.float32))
    print(json.dumps({k: res[k] for k in ("q_mean", "cf_mdi_mean", "cf_max_share_mean")}, indent=2))


if __name__ == "__main__":
    main()
