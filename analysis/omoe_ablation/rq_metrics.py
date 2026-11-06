"""Compute RQ-VAE reconstruction / codebook / collision metrics for one variant."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "rq"))

from datasets import EmbDataset  # noqa: E402
from models.rqvae import RQVAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ckpt["args"]
    data = EmbDataset(args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ca.num_emb_list,
        e_dim=ca.e_dim,
        layers=ca.layers,
        dropout_prob=ca.dropout_prob,
        bn=ca.bn,
        loss_type=ca.loss_type,
        quant_loss_weight=ca.quant_loss_weight,
        beta=ca.beta,
        kmeans_init=ca.kmeans_init,
        kmeans_iters=ca.kmeans_iters,
        sk_epsilons=ca.sk_epsilons,
        sk_iters=ca.sk_iters,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    x = torch.from_numpy(data.embeddings.astype(np.float32)).to(dev)
    bs = 512
    losses = []
    indices = []
    with torch.no_grad():
        for st in range(0, len(x), bs):
            xb = x[st:st + bs]
            out, rq_loss, idx = model(xb)
            losses.append(float(torch.nn.functional.mse_loss(out, xb).item()))
            indices.append(idx.cpu().numpy())
    mse = float(np.mean(losses))
    idx = np.concatenate(indices, axis=0)
    n, L = idx.shape

    per_layer = []
    for l in range(L):
        c = Counter(idx[:, l].tolist())
        probs = np.array(list(c.values()), dtype=np.float64) / n
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        per_layer.append({
            "layer": l,
            "used_codes": len(c),
            "num_codes": int(ca.num_emb_list[l]),
            "entropy": ent,
            "perplexity": float(math.exp(ent)),
            "max_share": float(probs.max()),
        })
    codes = [tuple(int(v) for v in row) for row in idx]
    raw_collision = 1.0 - len(set(codes)) / n

    rq_params = int(sum(v.numel() for v in model.state_dict().values() if torch.is_tensor(v)))
    res = {
        "n_items": int(n),
        "rq_params": rq_params,
        "rq_epochs": int(ca.epochs),
        "recon_mse": mse,
        "recon_rmse": float(math.sqrt(mse)),
        "raw_collision_rate": float(raw_collision),
        "raw_unique_sids": int(len(set(codes))),
        "per_layer": per_layer,
        "mean_codebook_perplexity": float(np.mean([p["perplexity"] for p in per_layer])),
        "min_codebook_perplexity": float(np.min([p["perplexity"] for p in per_layer])),
        "max_codebook_perplexity": float(np.max([p["perplexity"] for p in per_layer])),
    }

    if args.index_path and os.path.exists(args.index_path):
        with open(args.index_path) as f:
            ix = json.load(f)
        full = Counter(tuple(v[:4]) for v in ix.values())
        res["final_collision_rate"] = float(1.0 - len(full) / len(ix))
        res["final_unique_sids"] = int(len(full))
        res["final_max_group_size"] = int(max(full.values()))
        res["final_mean_group_size"] = float(np.mean(list(full.values())))
        for l in range(min(4, L)):
            c = Counter(v[l] for v in ix.values())
            res[f"final_layer{l}_unique"] = int(len(c))
            res[f"final_layer{l}_max"] = int(max(c.values()))

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
