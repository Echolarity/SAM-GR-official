"""Feature-level retrieval proxy on valid/test interactions for every model.

This is a screening metric (mean history vector -> cosine over item z).  It is
NOT the LLM-generator NDCG; the report labels it explicitly as a proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lb_core import load_inter  # noqa: E402

ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, "results", "analysis")
OUT = os.path.join(ROOT, "results", "proxy")
os.makedirs(OUT, exist_ok=True)
KS = (5, 10, 20)


def eval_proxy(item_emb: np.ndarray, cases, device="cuda:0", chunk=512):
    it = torch.tensor(item_emb, dtype=torch.float32, device=device)
    it = F.normalize(it, dim=-1)
    L = max(len(h) for _, h, _ in cases)
    H = np.zeros((len(cases), L), dtype=np.int64)
    M = np.zeros((len(cases), L), dtype=np.float32)
    T = np.array([t for _, _, t in cases], dtype=np.int64)
    for r, (_, h, _) in enumerate(cases):
        h = h[-L:]
        H[r, :len(h)] = h
        M[r, :len(h)] = 1
    lens = torch.tensor(M.sum(1), dtype=torch.float32, device=device)
    Ht = torch.tensor(H, device=device)
    Mt = torch.tensor(M, device=device)
    Tt = torch.tensor(T, device=device)
    hits = {k: 0.0 for k in KS}
    ndcg = {k: 0.0 for k in KS}
    for st in range(0, len(cases), chunk):
        en = min(st + chunk, len(cases))
        hb, mb, tb = Ht[st:en], Mt[st:en], Tt[st:en]
        u = (it[hb] * mb.unsqueeze(-1)).sum(1) / lens[st:en, None]
        u = F.normalize(u, dim=-1)
        s = u @ it.T
        ts = s.gather(1, tb[:, None])
        mask = torch.zeros_like(s).scatter_(1, hb, mb)
        s = s.masked_fill(mask.bool(), -1e9)
        s.scatter_(1, tb[:, None], ts)
        rank = (s > ts).sum(1)
        for k in KS:
            hit = (rank < k).float()
            hits[k] += hit.sum().item()
            ndcg[k] += (hit / torch.log2(rank.float() + 2)).sum().item()
    n = len(cases)
    return {f"HR@{k}": hits[k] / n for k in KS} | {f"NDCG@{k}": ndcg[k] / n for k in KS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--splits", default="valid,test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"
    for split in args.splits.split(","):
        cases = [(u, h, t) for (u, h, t) in load_inter(split) if t not in h]
        print(f"[{split}] {len(cases)} cases", flush=True)
        for f in sorted(os.listdir(ANALYSIS)):
            if not f.endswith("_z.npy"):
                continue
            name = f[:-len("_z.npy")]
            out = os.path.join(OUT, f"{name}_{split}.json")
            if os.path.exists(out) and not args.force:
                continue
            z = np.load(os.path.join(ANALYSIS, f))
            m = eval_proxy(z, cases, dev)
            json.dump({"name": name, "split": split, **m}, open(out, "w"), indent=2)
            print(f"[{split}] {name} NDCG@10={m['NDCG@10']:.5f}", flush=True)
    print("PROXY_DONE", flush=True)


if __name__ == "__main__":
    main()
