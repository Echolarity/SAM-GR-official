"""Encode an OMoE representation with an RQ-VAE checkpoint into SIDs.

Implements the same collision-resolution + optional collaborative merge as
``rq/generate_indices.py`` but takes an explicit ``--data_path`` so one RQ-VAE
can be reused for original and counterfactually intervened representations.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import torch

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "rq"))
from models.rqvae import RQVAE  # noqa: E402


def collision_groups(strs):
    groups = collections.defaultdict(list)
    for i, s in enumerate(strs):
        groups[s].append(i)
    return [v for v in groups.values() if len(v) > 1]


def merge_neighbors(indices, collab_path, train_inter, ratio):
    if ratio <= 0 or not collab_path or not os.path.exists(collab_path):
        return indices
    n = len(indices)
    emb = np.load(collab_path).astype(np.float32)
    if emb.shape[0] != n:
        print(f"[merge] skip {emb.shape[0]} != {n}")
        return indices
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    pop = np.zeros(n, dtype=np.float64)
    if train_inter and os.path.exists(train_inter):
        with open(train_inter, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3:
                    pop[int(p[2])] += 1
    top1 = np.empty(n, dtype=np.int64)
    for st in range(0, n, 2048):
        en = min(st + 2048, n)
        sim = emb[st:en] @ emb.T
        sim[np.arange(en - st), np.arange(st, en)] = -2.0
        top1[st:en] = sim.argmax(1)
    budget = int(ratio * n)
    order = np.argsort(pop)
    canonical = np.ones(n, dtype=bool)
    merged = 0
    for i in order:
        if merged >= budget:
            break
        i = int(i)
        j = int(top1[i])
        if j == i or not canonical[j] or not canonical[i] or pop[j] < pop[i]:
            continue
        indices[i] = list(indices[j])
        canonical[i] = False
        merged += 1
    print(f"[merge] {merged}/{n} ({merged/n:.4f})")
    return indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge_ratio", type=float, default=0.15)
    ap.add_argument("--collab_emb_path", default=os.path.join(
        BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy"))
    ap.add_argument("--train_inter", default=os.path.join(
        BASE, "data", "Amazon", "Sports_and_Outdoors", "Sports_and_Outdoors.train.inter"))
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    model = RQVAE(
        in_dim=int(np.load(args.data_path, mmap_mode="r").shape[1]),
        num_emb_list=ca.num_emb_list, e_dim=ca.e_dim, layers=ca.layers,
        dropout_prob=ca.dropout_prob, bn=ca.bn, loss_type=ca.loss_type,
        quant_loss_weight=ca.quant_loss_weight, beta=ca.beta,
        kmeans_init=ca.kmeans_init, kmeans_iters=ca.kmeans_iters,
        sk_epsilons=ca.sk_epsilons, sk_iters=ca.sk_iters,
    )
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    x = np.load(args.data_path).astype(np.float32)
    n = len(x)
    xt = torch.from_numpy(x).to(dev)

    codes = []
    recon = []
    with torch.no_grad():
        for st in range(0, n, args.batch_size):
            xb = xt[st:st + args.batch_size]
            idx = model.get_indices(xb, use_sk=False).view(xb.shape[0], -1)
            codes.append(idx.cpu().numpy())
            out, _, _ = model(xb)
            recon.append(float(torch.nn.functional.mse_loss(out, xb).item()))
    codes = np.concatenate(codes, 0)
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    indices = [[prefix[k].format(int(v)) for k, v in enumerate(row)] for row in codes]
    strs = [str(c) for c in indices]

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    rounds = 0
    while rounds < 20 and collision_groups(strs):
        groups = collision_groups(strs)
        print(f"[collision] round {rounds}: {len(groups)} groups")
        for g in groups:
            xb = xt[g]
            idx = model.get_indices(xb, use_sk=True).view(len(g), -1).cpu().numpy()
            for item, row in zip(g, idx):
                indices[item] = [prefix[k].format(int(v)) for k, v in enumerate(row)]
                strs[item] = str(indices[item])
        rounds += 1

    raw_unique = len(set(strs))
    indices = merge_neighbors(indices, args.collab_emb_path, args.train_inter, args.merge_ratio)
    strs = [str(c) for c in indices]
    counts = collections.Counter(strs)
    unique = len(counts)
    sizes = np.array(list(counts.values()), dtype=np.float64)
    pair_collision = float((sizes * (sizes - 1)).sum() / (n * (n - 1))) if n > 1 else 0.0

    per_layer = []
    for layer in range(codes.shape[1]):
        vals, cnts = np.unique(codes[:, layer], return_counts=True)
        p = cnts / cnts.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        per_layer.append({
            "layer": layer, "used_codes": int(len(vals)), "num_codes": int(ca.num_emb_list[layer]),
            "entropy": ent, "perplexity": float(math.exp(ent)), "max_share": float(p.max()),
        })

    sid_dict = {str(i): list(v) for i, v in enumerate(indices)}
    out_index = os.path.join(args.out_dir, f"{args.name}.index.json")
    with open(out_index, "w") as f:
        json.dump(sid_dict, f)

    meta = {
        "name": args.name, "data_path": args.data_path, "ckpt": args.ckpt,
        "n_items": n, "rq_epochs": int(getattr(ca, "epochs", -1)),
        "merge_ratio": args.merge_ratio,
        "recon_mse": float(np.mean(recon)),
        "raw_unique_sids": int(raw_unique),
        "raw_collision_rate": float(1.0 - raw_unique / n),
        "unique_sids": int(unique),
        "excess_collision_rate": float(1.0 - unique / n),
        "collided_item_ratio": float(sizes[sizes > 1].sum() / n),
        "max_bucket": int(sizes.max()), "mean_bucket": float(sizes.mean()),
        "pairwise_collision_prob": pair_collision,
        "per_layer": per_layer,
        "codebook_perplexity_mean": float(np.mean([p["perplexity"] for p in per_layer])),
        "codebook_perplexity_min": float(np.min([p["perplexity"] for p in per_layer])),
        "codebook_perplexity_max": float(np.max([p["perplexity"] for p in per_layer])),
        "index_path": out_index,
    }
    with open(os.path.join(args.out_dir, f"{args.name}.metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in (
        "name", "unique_sids", "excess_collision_rate", "collided_item_ratio",
        "max_bucket", "codebook_perplexity_mean", "recon_mse")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
