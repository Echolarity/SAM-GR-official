"""Aggregate the load-balancing results, run paired bootstraps, write tables."""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
from lb_core import FULL_LAMBDA, load_inter  # noqa: E402

TAB = os.path.join(RESULTS, "tables")
os.makedirs(TAB, exist_ok=True)


def load_models():
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "analysis", "lam*_seed*.json"))):
        d = json.load(open(f))
        out[(d["lambda_load"], d["seed"])] = d
    return out


def load_items(name):
    p = os.path.join(RESULTS, "analysis", f"{name}_items.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def load_proxy(name):
    out = {}
    for split in ("valid", "test"):
        p = os.path.join(RESULTS, "proxy", f"{name}_{split}.json")
        if os.path.exists(p):
            out[split] = json.load(open(p))
    return out


def load_downstream(name):
    p = os.path.join(RESULTS, "downstream", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def boot_ci(diff, n_boot=5000, seed=0, chunk=500):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot, dtype=np.float64)
    for st in range(0, n_boot, chunk):
        en = min(st + chunk, n_boot)
        idx = rng.integers(0, n, size=(en - st, n))
        means[st:en] = diff[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "p_value": float(min(p, 1.0)), "n": int(n),
    }


def per_user_metrics(result_path, split="test"):
    rows = json.load(open(result_path))
    inter = load_inter(split)
    users = [u for u, _, _ in inter]
    ks = (5, 10, 20)
    out = {f"HR@{k}": [] for k in ks} | {f"NDCG@{k}": [] for k in ks}
    for r in rows:
        target = str(r.get("output", "")).strip().strip('"\n ')
        preds = r.get("predict", [])
        if isinstance(preds, str):
            preds = [preds]
        preds = [str(p).strip().strip('"\n ') for p in preds]
        rank = None
        for i, p in enumerate(preds):
            if p == target:
                rank = i
                break
        for k in ks:
            hit = 1.0 if (rank is not None and rank < k) else 0.0
            out[f"HR@{k}"].append(hit)
            out[f"NDCG@{k}"].append(hit / np.log2(rank + 2) if hit else 0.0)
    arr = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    if len(users) == len(arr["HR@10"]):
        arr["user"] = np.asarray(users)
        # one test row per user in strict leave-one-out; mean handles duplicates
        df = pd.DataFrame(arr)
        g = df.groupby("user", sort=True).mean()
        return {k: g[k].values for k in g.columns}
    return arr


def main():
    models = load_models()
    rows = []
    for (lam, seed), d in sorted(models.items()):
        r = d["routing"]
        n = d["norm"]
        c = d["counterfactual_mean"]
        cz = d["counterfactual_zero"]
        gm = d["group_mass"]
        name = f"lam{('0' if lam==0 else ('1.0' if lam==1.0 else f'{lam:.0e}'.replace('e-0','e-')))}_seed{seed}"
        proxy = load_proxy(name)
        ds = load_downstream(name)
        rows.append({
            "lambda_load": lam, "seed": seed, "name": name,
            "load_cv": r["load_cv"], "effective_experts": r["effective_experts"],
            "dead_rate_0.01M": r["dead_rate_0.01M"], "dead_rate_1e-3": r["dead_rate_1e-3"],
            "entropy_mean": r["entropy_mean"], "entropy_std": r["entropy_std"],
            "entropy_p10": r["entropy_p10"], "entropy_p50": r["entropy_p50"],
            "entropy_p90": r["entropy_p90"],
            "routing_js_mean": r["routing_js_mean"],
            "routing_var_mean": r["routing_var_mean"],
            "norm_mdi": n["mdi_mean"], "norm_max_share": n["max_share_mean"],
            "cf_mdi": c["cf_mdi_mean"], "cf_entropy": c["cf_entropy_mean"],
            "cf_max_share": c["cf_max_share_mean"],
            "cf_mdi_zero": cz["cf_mdi_mean"],
            "q_text": c["q_mean"][0], "q_vision": c["q_mean"][1], "q_aligned": c["q_mean"][2],
            "mass_text": gm["text"]["mean"], "mass_vision": gm["vision"]["mean"],
            "mass_aligned": gm["aligned"]["mean"], "mass_shared": gm["shared"]["mean"],
            "norm_share_text": n["share_mean"]["text"], "norm_share_vision": n["share_mean"]["vision"],
            "norm_share_aligned": n["share_mean"]["aligned"], "norm_share_shared": n["share_mean"]["shared"],
            "mean_abs_expert_cos": d["mean_abs_expert_cos"],
            "train_time_sec": d["train_time_sec"],
            "proxy_valid_ndcg10": proxy.get("valid", {}).get("NDCG@10"),
            "proxy_test_ndcg10": proxy.get("test", {}).get("NDCG@10"),
            "HR@10": (ds or {}).get("HR@10"), "NDCG@10": (ds or {}).get("NDCG@10"),
            "HR@20": (ds or {}).get("HR@20"), "NDCG@20": (ds or {}).get("NDCG@20"),
            "excess_collision_rate": (ds or {}).get("excess_collision_rate"),
            "codebook_perplexity_mean": (ds or {}).get("codebook_perplexity_mean"),
        })
    df = pd.DataFrame(rows).sort_values(["lambda_load", "seed"])
    df.to_csv(os.path.join(TAB, "master.csv"), index=False)

    keys = [c for c in df.columns if c not in ("lambda_load", "seed", "name")]
    agg = df.groupby("lambda_load")[keys].agg(["mean", "std"])
    agg.to_csv(os.path.join(TAB, "sweep_by_lambda.csv"))

    noload = df[df.lambda_load == 0.0]
    full = df[df.lambda_load == FULL_LAMBDA]
    col = ["lambda_load", "seed", "load_cv", "effective_experts", "dead_rate_0.01M",
           "dead_rate_1e-3", "entropy_mean", "routing_js_mean", "norm_mdi", "cf_mdi",
           "cf_max_share", "mass_text", "mass_vision", "mass_aligned", "mass_shared",
           "norm_share_text", "norm_share_vision", "norm_share_aligned", "norm_share_shared",
           "q_text", "q_vision", "q_aligned", "mean_abs_expert_cos"]
    pd.concat([noload[col], full[col]]).to_csv(
        os.path.join(TAB, "noload_vs_fullload.csv"), index=False)

    # group-level long table
    grows = []
    for (lam, seed), d in models.items():
        if lam not in (0.0, FULL_LAMBDA):
            continue
        for row in d["group_rows"]:
            grows.append({"lambda_load": lam, "seed": seed, **row})
    pd.DataFrame(grows).to_csv(os.path.join(TAB, "groups_long.csv"), index=False)

    # paired bootstrap on item-level metrics: average per-item across seeds first
    tests = []
    metrics = ["entropy", "cf_max_share", "cf_mdi", "drift_mask_text",
               "drift_mask_vision", "drift_mask_aligned"]
    by_seed = defaultdict(dict)
    for (lam, seed), d in models.items():
        if lam not in (0.0, FULL_LAMBDA):
            continue
        arr = load_items(d["name"] if "name" in d else f"lam{('0' if lam==0 else '1e-1')}_seed{seed}")
        if arr is None:
            continue
        by_seed[seed][lam] = {m: arr[m].astype(np.float64) for m in metrics if m in arr}
    for seed in sorted(by_seed):
        if 0.0 not in by_seed[seed] or FULL_LAMBDA not in by_seed[seed]:
            continue
        a = by_seed[seed][0.0]
        b = by_seed[seed][FULL_LAMBDA]
        for m in metrics:
            if m not in a or m not in b:
                continue
            res = boot_ci(a[m] - b[m], seed=seed)
            tests.append({"seed": seed, "metric": m, "contrast": "NoLoad - FullLoad", **res})
    if by_seed:
        seeds = [s for s in by_seed if 0.0 in by_seed[s] and FULL_LAMBDA in by_seed[s]]
        for m in metrics:
            if not seeds:
                continue
            a = np.mean([by_seed[s][0.0][m] for s in seeds if m in by_seed[s][0.0]], axis=0)
            b = np.mean([by_seed[s][FULL_LAMBDA][m] for s in seeds if m in by_seed[s][FULL_LAMBDA]], axis=0)
            res = boot_ci(a - b, seed=123)
            tests.append({"seed": "avg", "metric": m, "contrast": "NoLoad - FullLoad", **res})
    pd.DataFrame(tests).to_csv(os.path.join(TAB, "bootstrap_item.csv"), index=False)

    # user-level bootstrap on recommendation metrics (if generator results exist)
    utests = []
    nl_path = os.path.join(RESULTS, "downstream", "noload_original_seed42", "eval",
                           "final_result_test.json")
    fl_path = os.path.join(RESULTS, "downstream", "fullload_original_seed42", "eval",
                           "final_result_test.json")
    if os.path.exists(nl_path) and os.path.exists(fl_path):
        a = per_user_metrics(nl_path)
        b = per_user_metrics(fl_path)
        for m in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
            if m in a and m in b and len(a[m]) == len(b[m]):
                res = boot_ci(a[m] - b[m], seed=7)
                utests.append({"metric": m, "contrast": "NoLoad - FullLoad", **res})
    pd.DataFrame(utests).to_csv(os.path.join(TAB, "bootstrap_user.csv"), index=False)

    # compact markdown
    def md(df_, floatfmt=".4f"):
        cols = list(df_.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df_.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{v:{floatfmt}}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    with open(os.path.join(TAB, "tables.md"), "w") as f:
        f.write("## NoLoad vs FullLoad (seed 42)\n\n")
        f.write(md(pd.concat([noload[noload.seed == 42][col], full[full.seed == 42][col]],
                             ignore_index=True)))
        f.write("\n\n## Lambda sweep (mean over seeds)\n\n")
        sweep = agg["load_cv"]["mean"].to_frame("load_cv")
        for m in ["effective_experts", "entropy_mean", "routing_js_mean", "cf_mdi",
                  "norm_mdi", "proxy_valid_ndcg10", "proxy_test_ndcg10"]:
            if (m, "mean") in agg:
                sweep[m] = agg[(m, "mean")]
            if (m, "std") in agg:
                sweep[m + "_std"] = agg[(m, "std")]
        f.write(md(sweep.reset_index()))
        f.write("\n")
    print("AGG_DONE", flush=True)


if __name__ == "__main__":
    main()
