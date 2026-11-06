import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "rq"))

from datasets import EmbDataset
from generate_indices import merge_collaborative_neighbors
from models.rqvae import RQVAE

PREFIX = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]


def sid_stats(codes):
    keys = [tuple(int(x) for x in row) for row in codes]
    counts = collections.Counter(keys)
    n = len(keys)
    u = len(counts)
    hist = collections.Counter(counts.values())
    collided = sum(v for v in counts.values() if v > 1)
    pairs = sum(v * (v - 1) for v in counts.values())
    prefixes = []
    for depth in range(1, codes.shape[1] + 1):
        pk = [tuple(int(x) for x in row[:depth]) for row in codes]
        pc = collections.Counter(pk)
        pu = len(pc)
        prefixes.append({
            "depth": depth,
            "unique_prefixes": pu,
            "unique_prefix_ratio": pu / n,
            "collision_rate": (n - pu) / n,
            "collided_item_ratio": sum(v for v in pc.values() if v > 1) / n,
            "max_bucket": max(pc.values()),
            "mean_bucket": n / pu,
        })
    return {
        "n_items": n,
        "unique_full_sids": u,
        "excess_collision_rate": (n - u) / n,
        "collided_item_ratio": collided / n,
        "pairwise_collision_probability": pairs / (n * (n - 1)),
        "max_bucket": max(counts.values()),
        "bucket_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "prefix": prefixes,
    }


def build(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    data = EmbDataset(args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, data, args


def infer_codes(model, data, device):
    loader = DataLoader(data, num_workers=0, batch_size=64, shuffle=False)
    codes = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="raw codes", leave=False):
            idx = model.get_indices(batch.to(device), use_sk=False)
            codes.append(idx.view(-1, idx.shape[-1]).cpu().numpy())
    return np.concatenate(codes, axis=0)


def resolve_sinkhorn(model, data, codes, device, max_iter=20):
    all_indices = [[PREFIX[i].format(int(x)) for i, x in enumerate(row)] for row in codes]
    strs = np.array([str(x) for x in all_indices])
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003
    history = []
    for it in range(max_iter):
        counts = collections.Counter(strs.tolist())
        groups = [g for g in _groups(strs) if len(g) > 1]
        history.append({"iter": it, "collision_groups": len(groups), "unique": len(counts)})
        if not groups:
            break
        for items in groups:
            batch = data[items].to(device)
            with torch.no_grad():
                idx = model.get_indices(batch, use_sk=True)
            idx = idx.view(-1, idx.shape[-1]).cpu().numpy()
            for item, row in zip(items, idx):
                all_indices[item] = [PREFIX[i].format(int(x)) for i, x in enumerate(row)]
                strs[item] = str(all_indices[item])
    return all_indices, strs, history


def _groups(strs):
    mp = collections.defaultdict(list)
    for i, s in enumerate(strs.tolist()):
        mp[s].append(i)
    return list(mp.values())


def apply_merge(tokens, merge_ratio, collab, train_inter):
    if merge_ratio <= 0:
        return [list(x) for x in tokens]
    merged = merge_collaborative_neighbors(
        [list(x) for x in tokens], collab, train_inter, merge_ratio
    )
    return [list(x) for x in merged]


def tokens_to_codes(tokens):
    out = np.zeros((len(tokens), len(tokens[0])), dtype=np.int32)
    for i, row in enumerate(tokens):
        for j, tok in enumerate(row):
            out[i, j] = int(tok.rsplit("_", 1)[1].rstrip(">"))
    return out


def compare_saved(tokens, saved_path):
    if not saved_path or not os.path.exists(saved_path):
        return None
    saved = json.load(open(saved_path))
    same = 0
    for i, row in enumerate(tokens):
        if list(row) == list(saved[str(i)]):
            same += 1
    return {"saved_path": saved_path, "exact_item_matches": same, "n": len(tokens),
            "match_rate": same / len(tokens)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge_ratio", type=float, default=0.15)
    p.add_argument("--collab_emb", default=os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy"))
    p.add_argument("--train_inter", default=os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors", "Sports_and_Outdoors.train.inter"))
    p.add_argument("--saved_index", default="")
    p.add_argument("--max_iter", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model, data, ck_args = build(args.ckpt)
    model = model.to(device)

    raw = infer_codes(model, data, device)
    raw_tokens = [[PREFIX[i].format(int(x)) for i, x in enumerate(row)] for row in raw]
    resolved_tokens, _, history = resolve_sinkhorn(model, data, raw, device, max_iter=args.max_iter)
    merged_tokens = apply_merge(resolved_tokens, args.merge_ratio, args.collab_emb, args.train_inter)

    raw_codes = raw
    resolved_codes = tokens_to_codes(resolved_tokens)
    merged_codes = tokens_to_codes(merged_tokens)

    payload = {
        "name": args.name,
        "checkpoint": args.ckpt,
        "embedding": ck_args.data_path,
        "rq_epochs": ck_args.epochs,
        "merge_ratio": args.merge_ratio,
        "sinkhorn_history": history,
        "raw": sid_stats(raw_codes),
        "sinkhorn": sid_stats(resolved_codes),
        "merged": sid_stats(merged_codes),
        "match": compare_saved(merged_tokens, args.saved_index),
    }
    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.name}_stages.npz"),
        raw=raw_codes, sinkhorn=resolved_codes, merged=merged_codes,
    )
    with open(os.path.join(args.out_dir, f"{args.name}_stages.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: payload[k] for k in ["name", "rq_epochs", "merge_ratio", "match"]}, indent=2))
    for stage in ["raw", "sinkhorn", "merged"]:
        s = payload[stage]
        print(f"{stage}: unique={s['unique_full_sids']} excess={s['excess_collision_rate']:.6f} "
              f"collided={s['collided_item_ratio']:.6f} max={s['max_bucket']}")


if __name__ == "__main__":
    main()
