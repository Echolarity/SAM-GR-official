"""Shared utilities for SAM-GR motivational experiments (Sports strict 10-core)."""
from __future__ import annotations

import json
import os

import numpy as np

BASE = "/data02/sht/SAM-GR"
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
OUT = os.path.join(BASE, "analysis", "samgr_sports10")
RESULTS = os.path.join(OUT, "results")
FIGURES = os.path.join(OUT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def load_emb(name: str) -> np.ndarray:
    """name in {text_only, image_only, multimodal, csga_moe_collab_v2}"""
    p = os.path.join(DATA, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy")
    return np.load(p).astype(np.float32)


def load_collab() -> np.ndarray:
    p = os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy")
    return np.load(p).astype(np.float32)


def load_items():
    p = os.path.join(DATA, "Sports_and_Outdoors.item.json")
    return json.load(open(p))


def load_index(path: str):
    """item_id (str) -> list of SID tokens"""
    return json.load(open(path))


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def topk_neighbors(x: np.ndarray, k: int, chunk: int = 2048):
    """x must be L2-normalised. Returns (idx [n,k], cos [n,k]), self excluded."""
    n = len(x)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    cos = np.empty((n, k), dtype=np.float32)
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sim = x[st:en] @ x.T
        rows = np.arange(en - st)
        sim[rows, np.arange(st, en)] = -2.0
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[st:en] = np.take_along_axis(part, order, axis=1)
        cos[st:en] = np.take_along_axis(vals, order, axis=1)
    return idx, cos


def overlap_at_k(idx_a: np.ndarray, idx_b: np.ndarray, k: int) -> np.ndarray:
    """Per-item fraction of shared neighbours among the top-k of each set."""
    out = np.empty(len(idx_a), dtype=np.float32)
    for i in range(len(idx_a)):
        sa = idx_a[i, :k]
        sb = idx_b[i, :k]
        out[i] = len(np.intersect1d(sa, sb)) / k
    return out


def jaccard_at_k(idx_a: np.ndarray, idx_b: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(idx_a), dtype=np.float32)
    for i in range(len(idx_a)):
        sa = idx_a[i, :k]
        sb = idx_b[i, :k]
        inter = len(np.intersect1d(sa, sb))
        out[i] = inter / max(len(np.union1d(sa, sb)), 1)
    return out


def category_labels(items):
    """Top-level fine-grained category used only for colouring."""
    labels = []
    for it in items.values():
        cats = (it.get("categories") or "").split(",")
        cats = [c.strip() for c in cats if c.strip()]
        if len(cats) >= 3:
            labels.append(cats[2])
        elif len(cats) == 2:
            labels.append(cats[1])
        elif cats:
            labels.append(cats[0])
        else:
            labels.append("Unknown")
    return labels
