"""Recompute only Part 1 (global statistics) from the cached model/features."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omoe_core import RESULTS, build_model, center, load_csga, load_emb
from run_analysis import part1


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_t = load_emb("text_only"); x_i = load_emb("image_only"); x_c = load_csga()
    cache = np.load(os.path.join(RESULTS, "omoe_cache.npz"))
    model = build_model(x_t.shape[1], dev)
    ckpt = torch.load(os.path.join(RESULTS, "omoe_model.pt"), map_location=dev)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    t = torch.from_numpy(center(x_t)).float().to(dev)
    i = torch.from_numpy(center(x_i)).float().to(dev)
    c = torch.from_numpy(center(x_c)).float().to(dev)
    part1(cache["gates"], cache["expert_norms"], cache["entropy"], model, t, i, c)


if __name__ == "__main__":
    main()
