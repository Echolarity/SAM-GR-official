import os
import sys
from types import SimpleNamespace

import numpy as np

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "rq", "text2emb"))
from omoe_v2 import omoe_fuse_v2

DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
OUT = os.path.join(BASE, "analysis", "sid_collision_sports10", "embeddings")
os.makedirs(OUT, exist_ok=True)


def load(name):
    return np.load(os.path.join(DATA, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy")).astype(np.float32)


def main():
    x_t = load("text_only")
    x_i = load("image_only")
    x_m = load("multimodal")
    args = SimpleNamespace(
        omoe2_epochs=60,
        omoe2_lr=1e-3,
        omoe2_batch_size=512,
        omoe2_shared=2,
        omoe_target_std=4.0,
        omoe2_residual=1.0,
    )
    fused = omoe_fuse_v2(x_t, x_i, x_m, args)
    path = os.path.join(OUT, "omoe_only.npy")
    np.save(path, fused.astype(np.float32))
    print("saved", path, fused.shape)


if __name__ == "__main__":
    main()
