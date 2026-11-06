"""Representation-level stage of the OMoE ablation.

Trains every OMoE variant on the *fixed* cached CSGA-aligned feature and saves
the fused embedding plus expert/routing/parameter diagnostics.  Downstream
retraining (RQ-VAE + generator) is handled by run_downstream.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BASE, "rq", "text2emb"))

from ablation_omoe import VariantConfig, train_and_fuse, _cos_stats  # noqa: E402
from variants import build_variants  # noqa: E402

DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
COLLAB = os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy")
CSGA_CACHE = os.path.join(BASE, "analysis", "samgr_sports10", "results", "csga_aligned.npy")
WORK = os.path.join(HERE, "workspaces")
RESULTS = os.path.join(HERE, "results", "representation")
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def load_emb(name: str) -> np.ndarray:
    return np.load(os.path.join(DATA, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy")).astype(np.float32)


def get_aligned() -> np.ndarray:
    if os.path.exists(CSGA_CACHE):
        a = np.load(CSGA_CACHE).astype(np.float32)
        print(f"[CSGA] loaded cached aligned feature {a.shape}", flush=True)
        return a
    from csga import csga_align

    args = SimpleNamespace(csga_topk=10, csga_epochs=60, csga_lr=1e-3, csga_batch_size=512)
    a = csga_align(load_emb("multimodal"), np.load(COLLAB).astype(np.float32), args).astype(np.float32)
    np.save(CSGA_CACHE, a)
    print(f"[CSGA] computed cached aligned feature {a.shape}", flush=True)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", type=str, default="", help="comma-separated variant prefixes, e.g. 01,11")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    aligned = get_aligned()
    print(f"text {x_t.shape} image {x_i.shape} aligned {aligned.shape} device={device}", flush=True)

    # reference rows (no OMoE) for context
    refs = {
        "00_csga_only": aligned,
        "00_multimodal_only": load_emb("multimodal"),
        "00_text_only": x_t,
        "00_existing_full_pipeline": load_emb("csga_moe_collab_v2"),
    }
    for name, emb in refs.items():
        std, mc, nd = _cos_stats(emb)
        json.dump(
            {"variant": name, "group": "reference", "n_items": int(len(emb)), "fused_std": std,
             "fused_mean_cos": mc, "fused_nn_dup": nd, "params": None, "train_time_sec": 0.0},
            open(os.path.join(RESULTS, f"{name}.json"), "w"), indent=2,
        )

    variants = build_variants()
    if args.only:
        keep = tuple(args.only.split(","))
        variants = [v for v in variants if v.name.startswith(keep)]
    for cfg in variants:
        if args.epochs != 60:
            cfg.epochs = args.epochs
        wdir = os.path.join(WORK, cfg.name)
        os.makedirs(wdir, exist_ok=True)
        emb_path = os.path.join(wdir, "emb.npy")
        stat_path = os.path.join(RESULTS, f"{cfg.name}.json")
        if os.path.exists(emb_path) and os.path.exists(stat_path) and not args.force:
            print(f"[skip] {cfg.name}", flush=True)
            continue
        print(f"\n===== {cfg.name} | {cfg.notes} =====", flush=True)
        emb, stats = train_and_fuse(x_t, x_i, aligned, cfg, device=device)
        np.save(emb_path, emb.astype(np.float32))
        stats["emb_path"] = emb_path
        json.dump(stats, open(stat_path, "w"), indent=2)
        print(
            f"[done] {cfg.name}: params={stats['params']/1e6:.1f}M "
            f"time={stats['train_time_sec']:.0f}s std={stats['fused_std']:.3f} "
            f"cos={stats['fused_mean_cos']:.3f} dup={stats['fused_nn_dup']:.3f} "
            f"expert|cos|={stats.get('mean_abs_expert_cos')} "
            f"gateCV={stats.get('gate_load_cv')} itemH={stats.get('mean_item_entropy')}",
            flush=True,
        )
    print("REPRESENTATION_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
