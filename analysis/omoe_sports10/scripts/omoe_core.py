"""Core utilities for the OMoE-v2 gating / expert-utilisation analysis.

The analysis reproduces the *inference-time* OMoE-v2 used by the
``csga_omoe_v2`` pipeline (rq/text2emb/omoe_v2.py) on the cached
Sports strict-10-core features.  Training hyper-parameters and the input
centering match ``omoe_fuse_v2`` exactly; the only addition is instrumentation
(per-item gates, expert outputs and fused representations).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/data02/sht/SAM-GR"
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
ANALYSIS = os.path.join(BASE, "analysis", "omoe_sports10")
RESULTS = os.path.join(ANALYSIS, "results")
FIGURES = os.path.join(ANALYSIS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "rq", "text2emb"))
from omoe_v2 import OMoE2  # noqa: E402

EXPERT_NAMES = ["text", "image", "csga", "shared-1", "shared-2"]
EXPERT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
N_EXPERTS = len(EXPERT_NAMES)


def load_emb(name: str) -> np.ndarray:
    p = os.path.join(DATA, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy")
    return np.load(p).astype(np.float32)


def load_csga() -> np.ndarray:
    p = os.path.join(os.path.dirname(ANALYSIS), "samgr_sports10", "results", "csga_aligned.npy")
    return np.load(p).astype(np.float32)


def load_items() -> dict:
    return json.load(open(os.path.join(DATA, "Sports_and_Outdoors.item.json")))


def load_inter(split: str):
    """Return list of (user_id, [history item ids], target item id)."""
    p = os.path.join(DATA, f"Sports_and_Outdoors.{split}.inter")
    rows = []
    with open(p) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            uid, hist, tgt = parts
            hist_ids = [int(x) for x in hist.split()] if hist.strip() else []
            rows.append((int(uid), hist_ids, int(tgt)))
    return rows


def center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(0, keepdims=True)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dim: int = 4096, device: str = "cuda:0", shared: int = 2) -> OMoE2:
    return OMoE2(dim, num_shared_experts=shared, ortho_weight=0.05, load_weight=0.1).to(device)


def train_model(
    x_t, x_i, x_c, device="cuda:0", epochs=60, bs=512, lr=1e-3, seed=42, verbose=True
):
    """Reproduce omoe_fuse_v2 training; return (model, centered tensors, history)."""
    set_seed(seed)
    n, dim = x_t.shape
    t = torch.from_numpy(center(x_t)).float().to(device)
    i = torch.from_numpy(center(x_i)).float().to(device)
    c = torch.from_numpy(center(x_c)).float().to(device)
    model = build_model(dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for st in range(0, n, bs):
            b = perm[st : st + bs]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(b)
        total /= n
        gates = getattr(model, "_monitor_weights", np.zeros(model.num_experts))
        rec = float(getattr(model, "_monitor_loss_rec_real", 0.0))
        ortho = float(getattr(model, "_monitor_loss_ortho_real", 0.0))
        history.append({"epoch": ep, "loss": total, "rec": rec, "ortho": ortho,
                        "gates": np.asarray(gates).tolist()})
        if verbose and ep % 10 == 0:
            print(f"[OMoE] ep {ep:02d} loss={total:.4f} rec={rec:.4f} "
                  f"ortho={ortho:.4f} gates={np.round(gates, 3)}", flush=True)
    model.eval()
    return model, (t, i, c), history


@torch.no_grad()
def infer(model, t, i, c, bs=512, return_experts=True):
    """Run model on (centered) feature tensors and return per-item outputs.

    Returns dict with gates [n,E], expert_norms [n,E], fused_raw [n,dim],
    aligned_csga [n,dim] (the CSGA input), feats list, and optionally experts.
    """
    n = t.shape[0]
    E = model.num_experts
    gates = np.empty((n, E), dtype=np.float32)
    expert_norms = np.empty((n, E), dtype=np.float32)
    fused = np.empty((n, t.shape[1]), dtype=np.float32)
    expert_list = [] if return_experts else None
    for st in range(0, n, bs):
        en = min(st + bs, n)
        _ = model(t[st:en], i[st:en], c[st:en])
        g = model._monitor_gates.cpu().numpy()
        ex = [e.cpu().numpy() for e in model._monitor_experts]
        gates[st:en] = g
        for k in range(E):
            expert_norms[st:en, k] = np.linalg.norm(ex[k], axis=1)
        feats = [model.align_text(t[st:en]), model.align_image(i[st:en]), model.align_collab(c[st:en])]
        concat = torch.cat(feats, dim=-1)
        moe = torch.zeros_like(model._monitor_experts[0])
        for k in range(E):
            moe = moe + model._monitor_gates[:, k].unsqueeze(-1) * model._monitor_experts[k]
        raw = model.out_proj(concat) + moe
        fused[st:en] = raw.cpu().numpy()
        if return_experts:
            expert_list.append([e.copy() for e in ex])
    out = {"gates": gates, "expert_norms": expert_norms, "fused_raw": fused}
    if return_experts:
        # list of [n, dim] per expert
        out["experts"] = [np.concatenate([b[k] for b in expert_list], 0) for k in range(E)]
    return out


def production_fused(fused_raw: np.ndarray, aligned: np.ndarray, target_std: float = 4.0,
                     resid: float = 1.0) -> np.ndarray:
    """Reproduce omoe_fuse_v2 output scaling + CSGA residual."""
    cur = float(fused_raw.std())
    scale = target_std / cur if cur > 1e-8 else 1.0
    return aligned + resid * (fused_raw * scale)


def gate_entropy(g: np.ndarray) -> np.ndarray:
    return -(g * np.log(np.clip(g, 1e-12, None))).sum(-1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def item_ids_from_json(items: dict) -> np.ndarray:
    return np.array([int(k) for k in items.keys()], dtype=np.int64)
