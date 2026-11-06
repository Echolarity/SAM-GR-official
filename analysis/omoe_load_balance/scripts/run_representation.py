"""Train one or more (lambda_load, seed) OMoE models and run the full-item analysis."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lb_core import (EXPERT_NAMES, FULL_LAMBDA, LAMBDAS, SEEDS, TrainConfig, analyze,
                     build_metadata, conditional_z, cosine_rows, counterfactual_metrics,
                     group_mass, group_metrics, load_all, make_base_state, norm_contribution,
                     routing_metrics, save_analysis, set_seed, train_lb)

BASE = "/data02/sht/SAM-GR"
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "results", "checkpoints")
ANALYSIS = os.path.join(ROOT, "results", "analysis")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(ANALYSIS, exist_ok=True)


def tag(lam: float) -> str:
    if lam == 0:
        return "0"
    if lam == 1.0:
        return "1.0"
    return f"{lam:.0e}".replace("e-0", "e-")


def parse_list(s: str, cast):
    return [cast(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--lambdas", default="0,1e-4,1e-3,1e-2,1e-1,1.0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    seeds = parse_list(args.seeds, int)
    lambdas = parse_list(args.lambdas, float)
    cfg = TrainConfig(epochs=args.epochs)

    x_t, x_i, x_c, meta = load_all()
    n = len(x_t)
    rng = np.random.default_rng(12345)
    probe_idx = np.sort(rng.choice(n, size=min(256, n), replace=False))
    shuffle_perm = rng.permutation(n)

    jobs = [(s, l) for s in seeds for l in lambdas]
    jobs = [j for k, j in enumerate(jobs) if k % args.num_shards == args.shard]
    print(f"device={device} jobs={jobs}", flush=True)

    base_cache: dict[int, dict] = {}
    for seed, lam in jobs:
        name = f"lam{tag(lam)}_seed{seed}"
        ckpt_path = os.path.join(CKPT, f"{name}.pt")
        out_path = os.path.join(ANALYSIS, f"{name}.npz")
        if not args.force and os.path.exists(ckpt_path) and os.path.exists(out_path):
            print(f"[skip] {name}", flush=True)
            continue
        print(f"\n===== {name} (lambda={lam:g}, seed={seed}) =====", flush=True)
        if seed not in base_cache:
            base_cache[seed] = make_base_state(x_t.shape[1], cfg, seed)
        model, tr = train_lb(
            x_t, x_i, x_c, lam, seed, cfg, device,
            base_state=base_cache[seed], probe_idx=probe_idx,
        )
        torch.save({
            "state_dict": model.state_dict(), "config": cfg.__dict__,
            "lambda_load": lam, "seed": seed, "train": tr,
        }, ckpt_path)

        t0 = time.time()
        a = analyze(model, x_t, x_i, x_c, cfg)
        scale = a["scale"]
        rout = routing_metrics(a["pi"])
        gm = group_mass(a["pi"])
        nc = norm_contribution(a["pi"], a["h_norms"])

        cf_mean = conditional_z(model, x_t, x_i, x_c, a["means"], scale, cfg, "mean")
        cf_zero = conditional_z(model, x_t, x_i, x_c, a["means"], scale, cfg, "zero")
        cfm = counterfactual_metrics(cf_mean["q"], cf_mean["deltas"])
        cfz = counterfactual_metrics(cf_zero["q"], cf_zero["deltas"])

        z_mask_text = cf_mean["z_no_text"]
        z_mask_vision = cf_mean["z_no_vision"]
        z_mask_aligned = cf_mean["z_no_aligned"]
        t = torch.from_numpy(x_t - a["means"][0]).to(device)
        i = torch.from_numpy(x_i - a["means"][1]).to(device)
        c = torch.from_numpy(x_c - a["means"][2]).to(device)
        with torch.no_grad():
            p = model(t[shuffle_perm], i, c, analysis=True)
            z_shuf_text = p["aligned"].cpu().numpy() + cfg.gamma * scale * p["fused"].cpu().numpy()
            p = model(t, i[shuffle_perm], c, analysis=True)
            z_shuf_vision = p["aligned"].cpu().numpy() + cfg.gamma * scale * p["fused"].cpu().numpy()
        conds = {
            "original": a["z"], "mask_text": z_mask_text, "mask_vision": z_mask_vision,
            "mask_aligned": z_mask_aligned, "shuffle_vision": z_shuf_vision,
            "shuffle_text": z_shuf_text,
        }
        drift = {k: (1.0 - cosine_rows(a["z"], v)).astype(np.float32) for k, v in conds.items()}

        expert_cos = np.zeros((model.num_experts, model.num_experts), dtype=np.float64)
        cnt = 0
        with torch.no_grad():
            for st in range(0, n, 512):
                en = min(st + 512, n)
                p = model(t[st:en], i[st:en], c[st:en], analysis=True)
                hh = p["h"]
                b = hh.shape[0]
                for u in range(model.num_experts):
                    for v in range(u, model.num_experts):
                        val = torch.nn.functional.cosine_similarity(
                            hh[:, u], hh[:, v], dim=-1).mean().item()
                        expert_cos[u, v] += val * b
                        if u != v:
                            expert_cos[v, u] += val * b
                cnt += b
        expert_cos /= max(cnt, 1)
        np.fill_diagonal(expert_cos, 1.0)

        metrics = {
            "lambda_load": lam, "seed": seed, "n_items": n,
            "num_experts": model.num_experts, "expert_names": EXPERT_NAMES,
            "expert_types": a["expert_types"].tolist(),
            "params": int(sum(p.numel() for p in model.parameters())),
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "w_ortho": cfg.w_ortho, "w_dup": cfg.w_dup, "w_var": cfg.w_var,
            "gamma": cfg.gamma, "target_std": cfg.target_std, "scale": scale,
            "train_time_sec": tr["train_time_sec"],
            "analysis_time_sec": time.time() - t0,
            "routing": {k: v for k, v in rout.items() if k != "entropy_per_item"},
            "group_mass": {g: {k: v for k, v in d.items() if k != "values"} for g, d in gm.items()},
            "norm": {
                "share_mean": nc["norm_share_mean"], "share_std": nc["norm_share_std"],
                "mdi_mean": float(nc["norm_mdi"].mean()), "mdi_std": float(nc["norm_mdi"].std()),
                "entropy_mean": float(nc["norm_entropy"].mean()),
                "max_share_mean": float(nc["norm_max_share"].mean()),
            },
            "counterfactual_mean": {k: v for k, v in cfm.items() if not k.endswith("per_item")},
            "counterfactual_zero": {k: v for k, v in cfz.items() if not k.endswith("per_item")},
            "drift_mean": {k: float(v.mean()) for k, v in drift.items()},
            "drift_std": {k: float(v.std()) for k, v in drift.items()},
            "expert_cos": expert_cos.tolist(),
            "mean_abs_expert_cos": float(np.abs(expert_cos[~np.eye(model.num_experts, dtype=bool)]).mean()),
            "group_rows": group_metrics(a["pi"], cf_mean["q"], a["h_norms"], meta),
            "history": tr["history"],
        }
        save_analysis(out_path, a, metrics)

        np.savez(
            os.path.join(ANALYSIS, f"{name}_items.npz"),
            entropy=rout["entropy_per_item"],
            norm_mdi=nc["norm_mdi"], norm_max_share=nc["norm_max_share"],
            norm_group=nc["specific_p"],
            cf_mdi=cfm["cf_mdi_per_item"], cf_max_share=cfm["cf_max_share_per_item"],
            cf_q=cfm["q_per_item"], cf_mdi_zero=cfz["cf_mdi_per_item"],
            cf_q_zero=cfz["q_per_item"],
            mass_text=gm["text"]["values"], mass_vision=gm["vision"]["values"],
            mass_aligned=gm["aligned"]["values"], mass_shared=gm["shared"]["values"],
            drift_original=drift["original"], drift_mask_text=drift["mask_text"],
            drift_mask_vision=drift["mask_vision"], drift_mask_aligned=drift["mask_aligned"],
            drift_shuffle_vision=drift["shuffle_vision"], drift_shuffle_text=drift["shuffle_text"],
            item_ids=np.arange(n, dtype=np.int64),
        )
        np.save(os.path.join(ANALYSIS, f"{name}_z.npy"), a["z"].astype(np.float32))
        for cond, arr in conds.items():
            if cond != "original":
                np.save(os.path.join(ANALYSIS, f"{name}_z_{cond}.npy"), arr.astype(np.float32))

        print(
            f"[done] {name} cv={rout['load_cv']:.4f} eff={rout['effective_experts']:.3f} "
            f"H={rout['entropy_mean']:.3f} JS={rout['routing_js_mean']:.4f} "
            f"cfMDI={cfm['cf_mdi_mean']:.4f} dead={rout['dead_rate_0.01M']:.2f} "
            f"time={tr['train_time_sec']:.0f}s", flush=True,
        )
    print("REPRESENTATION_DONE", flush=True)


if __name__ == "__main__":
    main()
