import json
import os
import sys

import numpy as np

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "analysis", "sid_collision_sports10", "scripts"))
from analyze_all import (DATA, STAGES, RES, bucket_pairs, load_meta, norm, pair_stats,
                         summarize_pairs)

MODELS = ["base_300ep", "csga_300ep", "omoe_300ep", "full300_300ep"]
LABELS = {
    "base_300ep": "Base joint-VLM (300ep)",
    "csga_300ep": "CSGA-only (300ep)",
    "omoe_300ep": "OMoE-only (300ep)",
    "full300_300ep": "Full SAM-GR (300ep)",
}


def main():
    _, pop, item_users, cats = load_meta()
    text = norm(np.load(os.path.join(DATA, "Sports_and_Outdoors.emb-Qwen3VL-text_only.npy")))
    image = norm(np.load(os.path.join(DATA, "Sports_and_Outdoors.emb-Qwen3VL-image_only.npy")))
    out = {}
    for name in MODELS:
        d = np.load(os.path.join(STAGES, f"{name}_stages.npz"))
        for stage in ["raw", "sinkhorn", "merged"]:
            codes = d[stage]
            _, pairs = bucket_pairs(codes)
            if len(pairs) > 200000:
                out[f"{LABELS[name]}|{stage}"] = {"n_pairs": len(pairs), "skipped": True}
                continue
            df = pair_stats(pairs, cats, pop, item_users, text, image)
            out[f"{LABELS[name]}|{stage}"] = summarize_pairs(df, f"{LABELS[name]} {stage}")
    with open(os.path.join(RES, "collided_stage_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        if v.get("skipped"):
            print(k, "SKIPPED", v["n_pairs"])
        else:
            print(k, "pairs", v["n_pairs"], "cooccur", round(v["cooccur_rate"], 4),
                  "text", round(v["text_cos_mean"], 4), "sameL3", round(v["same_level3_rate"], 4))


if __name__ == "__main__":
    main()
