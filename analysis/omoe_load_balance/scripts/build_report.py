"""Build the final markdown report from the aggregated tables and figures."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
sys.path.insert(0, HERE)
from lb_core import EXPERT_NAMES, EXPERT_TYPES, FULL_LAMBDA, LAMBDAS  # noqa: E402


def md_table(df, floatfmt=".4f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:{floatfmt}}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_down(name):
    p = os.path.join(RES, "downstream", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_sid(name):
    for p in glob.glob(os.path.join(RES, "sid", "*", name, "*.metrics.json")):
        return json.load(open(p))
    return None


def load_gen_valid():
    out = {}
    for f in glob.glob(os.path.join(RES, "downstream", "val_lam*_seed42.json")):
        d = json.load(open(f))
        out[float(d["lambda_load"])] = d
    return out


def pct(a, b):
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{100*(a-b)/b:+.1f}%"


def main():
    m = pd.read_csv(os.path.join(TAB, "master.csv"))
    boot = pd.read_csv(os.path.join(TAB, "bootstrap_item.csv"))
    try:
        bootu = pd.read_csv(os.path.join(TAB, "bootstrap_user.csv"))
    except Exception:
        bootu = pd.DataFrame()
    groups = pd.read_csv(os.path.join(TAB, "groups_long.csv"))

    nol = m[m.lambda_load == 0.0]
    full = m[m.lambda_load == FULL_LAMBDA]
    seed42n = nol[nol.seed == 42].iloc[0]
    seed42f = full[full.seed == 42].iloc[0]

    sweep = m.groupby("lambda_load").agg(
        load_cv_mean=("load_cv", "mean"), load_cv_std=("load_cv", "std"),
        eff_mean=("effective_experts", "mean"), eff_std=("effective_experts", "std"),
        dead_mean=("dead_rate_0.01M", "mean"), dead_std=("dead_rate_0.01M", "std"),
        dead_1e3_mean=("dead_rate_1e-3", "mean"),
        H_mean=("entropy_mean", "mean"), H_std=("entropy_mean", "std"),
        H_p10=("entropy_p10", "mean"), H_p50=("entropy_p50", "mean"), H_p90=("entropy_p90", "mean"),
        JS_mean=("routing_js_mean", "mean"), JS_std=("routing_js_mean", "std"),
        rvar_mean=("routing_var_mean", "mean"),
        norm_mdi_mean=("norm_mdi", "mean"), cf_mdi_mean=("cf_mdi", "mean"),
        cf_mdi_std=("cf_mdi", "std"), cf_max_mean=("cf_max_share", "mean"),
        q_text_mean=("q_text", "mean"), q_vision_mean=("q_vision", "mean"),
        q_aligned_mean=("q_aligned", "mean"),
        mass_text_mean=("mass_text", "mean"), mass_vision_mean=("mass_vision", "mean"),
        mass_aligned_mean=("mass_aligned", "mean"), mass_shared_mean=("mass_shared", "mean"),
        proxy_valid_mean=("proxy_valid_ndcg10", "mean"), proxy_valid_std=("proxy_valid_ndcg10", "std"),
        proxy_test_mean=("proxy_test_ndcg10", "mean"), proxy_test_std=("proxy_test_ndcg10", "std"),
        HR10_mean=("HR@10", "mean"), NDCG10_mean=("NDCG@10", "mean"),
        HR20_mean=("HR@20", "mean"), NDCG20_mean=("NDCG@20", "mean"),
        collision_mean=("excess_collision_rate", "mean"),
        ppl_mean=("codebook_perplexity_mean", "mean"),
    ).reset_index()

    # NoLoad detailed table
    routing_rows = []
    for _, r in m.sort_values(["lambda_load", "seed"]).iterrows():
        routing_rows.append({
            "lambda": r.lambda_load, "seed": r.seed,
            "load_cv": r.load_cv, "eff_experts": r.effective_experts,
            "dead@0.01/M": r["dead_rate_0.01M"], "dead@1e-3": r["dead_rate_1e-3"],
            "H_mean": r.entropy_mean, "H_std": r.entropy_std,
            "H_p10": r.entropy_p10, "H_p50": r.entropy_p50, "H_p90": r.entropy_p90,
            "JS": r.routing_js_mean, "var_pi": r.routing_var_mean,
        })
    routing_tab = pd.DataFrame(routing_rows)

    # NoLoad vs FullLoad seed means
    cmp_rows = []
    for lam, label in [(0.0, "NoLoad"), (FULL_LAMBDA, "FullLoad")]:
        sub = m[m.lambda_load == lam]
        cmp_rows.append({
            "model": label,
            "load_cv": f"{sub.load_cv.mean():.4f} ± {sub.load_cv.std():.4f}",
            "effective_experts": f"{sub.effective_experts.mean():.3f} ± {sub.effective_experts.std():.3f}",
            "dead@0.01/M": f"{sub['dead_rate_0.01M'].mean():.3f}",
            "dead@1e-3": f"{sub['dead_rate_1e-3'].mean():.3f}",
            "normalized_H": f"{sub.entropy_mean.mean():.3f} ± {sub.entropy_mean.std():.3f}",
            "routing_JS": f"{sub.routing_js_mean.mean():.4f} ± {sub.routing_js_mean.std():.4f}",
            "norm_MDI": f"{sub.norm_mdi.mean():.3f}",
            "counterfactual_MDI": f"{sub.cf_mdi.mean():.3f} ± {sub.cf_mdi.std():.3f}",
            "cf_max_share": f"{sub.cf_max_share.mean():.3f}",
            "q_text": f"{sub.q_text.mean():.3f}", "q_vision": f"{sub.q_vision.mean():.3f}",
            "q_aligned": f"{sub.q_aligned.mean():.3f}",
            "proxy_valid_NDCG@10": f"{sub.proxy_valid_ndcg10.mean():.5f}",
            "proxy_test_NDCG@10": f"{sub.proxy_test_ndcg10.mean():.5f}",
        })
    cmp_tab = pd.DataFrame(cmp_rows)

    # modality table seed42
    mod_rows = []
    for _, r in m[(m.seed == 42) & (m.lambda_load.isin([0.0, FULL_LAMBDA]))].iterrows():
        mod_rows.append({
            "model": "NoLoad" if r.lambda_load == 0 else "FullLoad",
            "routing_text": r.mass_text, "routing_vision": r.mass_vision,
            "routing_aligned": r.mass_aligned, "routing_shared": r.mass_shared,
            "norm_text": r.norm_share_text, "norm_vision": r.norm_share_vision,
            "norm_aligned": r.norm_share_aligned, "norm_shared": r.norm_share_shared,
            "cf_text": r.q_text, "cf_vision": r.q_vision, "cf_aligned": r.q_aligned,
            "cf_MDI": r.cf_mdi, "norm_MDI": r.norm_mdi,
        })
    mod_tab = pd.DataFrame(mod_rows)

    # component ablation: no-MoE (direct only) vs OMoE NoLoad/FullLoad
    nm = json.load(open(os.path.join(RES, "analysis", "no_moe_seed42.json")))
    comp_rows = [
        {"variant": "No MoE (direct only)", "q_text": nm["q_mean"][0],
         "q_vision": nm["q_mean"][1], "q_aligned": nm["q_mean"][2],
         "cf_MDI": nm["cf_mdi_mean"], "max share": nm["cf_max_share_mean"]},
        {"variant": "OMoE NoLoad (λ=0)", "q_text": seed42n.q_text,
         "q_vision": seed42n.q_vision, "q_aligned": seed42n.q_aligned,
         "cf_MDI": seed42n.cf_mdi, "max share": seed42n.cf_max_share},
        {"variant": "OMoE FullLoad (λ=0.1)", "q_text": seed42f.q_text,
         "q_vision": seed42f.q_vision, "q_aligned": seed42f.q_aligned,
         "cf_MDI": seed42f.cf_mdi, "max share": seed42f.cf_max_share},
    ]
    comp_tab = pd.DataFrame(comp_rows)

    # group differences
    grows = []
    for dim in groups.dimension.unique():
        for g in groups[groups.dimension == dim].group.unique():
            a = groups[(groups.dimension == dim) & (groups.group == g) & (groups.lambda_load == 0.0)]
            b = groups[(groups.dimension == dim) & (groups.group == g) &
                       (groups.lambda_load == FULL_LAMBDA)]
            if len(a) == 0 or len(b) == 0:
                continue
            grows.append({
                "dimension": dim, "group": g, "n": int(a.n.mean()),
                "q_text_NoLoad": a.q_text.mean(), "q_text_FullLoad": b.q_text.mean(),
                "q_vision_NoLoad": a.q_vision.mean(), "q_vision_FullLoad": b.q_vision.mean(),
                "q_aligned_NoLoad": a.q_aligned.mean(), "q_aligned_FullLoad": b.q_aligned.mean(),
                "delta_aligned": b.q_aligned.mean() - a.q_aligned.mean(),
            })
    group_tab = pd.DataFrame(grows)

    # downstream
    dn = load_down("noload_original_seed42")
    df = load_down("fullload_original_seed42")
    gen_val = load_gen_valid()
    dc = load_down("csga_only_seed42")
    down_rows = []
    for label, d in [("NoLoad", dn), ("FullLoad", df), ("CSGA-only", dc)]:
        if d:
            down_rows.append({
                "model": label, "HR@5": d.get("HR@5"), "HR@10": d.get("HR@10"),
                "HR@20": d.get("HR@20"), "NDCG@5": d.get("NDCG@5"),
                "NDCG@10": d.get("NDCG@10"), "NDCG@20": d.get("NDCG@20"),
                "SID collision": d.get("excess_collision_rate"),
                "codebook ppl": d.get("codebook_perplexity_mean"),
            })
    down_tab = pd.DataFrame(down_rows)

    inter_rows = []
    for label, prefix in [("NoLoad", "noload"), ("FullLoad", "fullload")]:
        for cond in ("original", "mask_text", "mask_vision", "mask_aligned"):
            d = load_down(f"{prefix}_{cond}_seed42")
            if d:
                inter_rows.append({
                    "model": label, "condition": cond,
                    "HR@10": d.get("HR@10"), "NDCG@10": d.get("NDCG@10"),
                    "HR@20": d.get("HR@20"), "NDCG@20": d.get("NDCG@20"),
                    "SID collision": d.get("excess_collision_rate"),
                    "codebook ppl": d.get("codebook_perplexity_mean"),
                })
    inter_tab = pd.DataFrame(inter_rows)

    # generator validation NDCG@10 per lambda (seed 42)
    gv_rows = []
    for lam, d in sorted(gen_val.items()):
        gv_rows.append({
            "lambda": lam, "HR@5": d.get("HR@5"), "HR@10": d.get("HR@10"),
            "HR@20": d.get("HR@20"), "NDCG@5": d.get("NDCG@5"),
            "NDCG@10": d.get("NDCG@10"), "NDCG@20": d.get("NDCG@20"),
        })
    gv_tab = pd.DataFrame(gv_rows)

    # pre-existing ablation artifacts (400-step, full test)
    abl_rows = []
    abl_names = {
        "01_full_omoe": "Full OMoE (MoE+direct+residual+load)",
        "03_wo_load": "No load (MoE+direct+residual)",
        "11_no_direct": "No direct (MoE only)",
        "12_no_experts": "No MoE (direct only)",
        "13_no_residual": "No residual",
    }
    for key, label in abl_names.items():
        p = os.path.join(os.path.dirname(RES), "omoe_ablation", "results",
                         f"downstream_{key}.json")
        rp = os.path.join(os.path.dirname(RES), "omoe_ablation", "results",
                          "representation", f"{key}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            rep = json.load(open(rp)) if os.path.exists(rp) else {}
            abl_rows.append({
                "variant": label, "HR@10": d.get("HR@10"), "HR@20": d.get("HR@20"),
                "NDCG@10": d.get("NDCG@10"), "NDCG@20": d.get("NDCG@20"),
                "gate CV": rep.get("gate_load_cv"),
                "mean gate": "—" if rep.get("mean_gate") is None else
                             "/".join(f"{x:.2f}" for x in rep["mean_gate"]),
                "SID collision": d.get("final_collision_rate"),
                "codebook ppl": d.get("codebook_perplexity_mean"),
            })
    abl_tab = pd.DataFrame(abl_rows)

    # SID sweep
    sid_rows = []
    for lam in LAMBDAS:
        name = f"lam{('0' if lam==0 else ('1.0' if lam==1.0 else f'{lam:.0e}'.replace('e-0','e-')))}_seed42"
        d = load_sid(name)
        if d:
            sid_rows.append({
                "lambda": lam, "unique SIDs": d["unique_sids"],
                "excess collision": d["excess_collision_rate"],
                "collided item ratio": d["collided_item_ratio"],
                "max bucket": d["max_bucket"], "recon MSE": d["recon_mse"],
                "codebook ppl": d["codebook_perplexity_mean"],
            })
    sid_tab = pd.DataFrame(sid_rows)

    best = sweep.sort_values("cf_mdi_mean").iloc[0]
    best_proxy = sweep.sort_values("proxy_valid_mean", ascending=False).iloc[0]

    def ans_bool(x):
        return "是" if x else "否"

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# OMoE load-balancing objective: expert utilization, modality dominance, and instance-level routing
*Sports & Outdoors strict-10-core · generated {now}*

## TL;DR

- **Without load balancing the router collapses**: NoLoad (λ=0) has load CV **{nol.load_cv.mean():.2f} ± {nol.load_cv.std():.2f}**, effective experts **{nol.effective_experts.mean():.2f}**, and dead-expert rate **{nol['dead_rate_0.01M'].mean():.2f}**. Every seed routes essentially all items to one or two experts (aligned, image+aligned, and image for seeds 42/43/44 respectively).
- **The load objective fixes the average case but is not a collapse guard**: FullLoad (λ=0.1) reaches CV **{full.load_cv.mean():.4f}**, effective experts **{full.effective_experts.mean():.3f}**, dead rate **{full['dead_rate_0.01M'].mean():.2f}**, with all five experts used in **3/3 seeds**. Intermediate λ (10⁻³, 10⁻⁴) and even λ=1 can still collapse in some seeds (see §6).
- **FullLoad substantially reduces modality dominance** without erasing it: counterfactual MDI falls from **{nol.cf_mdi.mean():.3f}** to **{full.cf_mdi.mean():.3f}**, max modality share from **{nol.cf_max_share.mean():.3f}** to **{full.cf_max_share.mean():.3f}**, and aligned-expert counterfactual contribution from **{nol.q_aligned.mean():.3f}** to **{full.q_aligned.mean():.3f}**. The aligned branch remains the largest single contributor.
- **Instance-dependent routing is preserved**: FullLoad has normalized per-item routing entropy **{full.entropy_mean.mean():.3f}** (NoLoad {nol.entropy_mean.mean():.3f}) and routing JS **{full.routing_js_mean.mean():.4f}** (NoLoad {nol.routing_js_mean.mean():.4f}). λ=1 does **not** force per-item uniformity (H≈0.70 in its two successful seeds); its problem is optimization instability (one seed collapses), not over-uniformity.
- **Best representation compromise: λ=0.1 (FullLoad)**. It has the lowest counterfactual MDI ({full.cf_mdi.mean():.3f}), the highest routing entropy among non-collapsed settings ({full.entropy_mean.mean():.3f}), robust balance in 3/3 seeds, and the lowest SID collision (15.2% vs 19.5% NoLoad) on the seed-42 RQ/SID sweep.
- **Downstream HR does not follow the same ordering at the tested generator budget.** With a fixed 200-step SFT and a 5k test subset, NoLoad gets HR@20 **{dn.get('HR@20') if dn else 'n/a'}** vs FullLoad **{df.get('HR@20') if df else 'n/a'}**; the same direction appears in the existing 400-step ablation (NoLoad 0.0201 vs FullLoad 0.0124). The reason is SID difficulty, not representation quality: collapsed/merged SID spaces have fewer distinct targets, so an undertrained generator scores higher. We report this as a negative downstream result and do not claim that load balancing improves raw HR at this budget.
- **Adding the MoE branch shifts counterfactual mass toward aligned.** Direct-only (no MoE) gives q_text/q_vision/q_aligned = **{nm['q_mean'][0]:.3f}/{nm['q_mean'][1]:.3f}/{nm['q_mean'][2]:.3f}** (MDI **{nm['cf_mdi_mean']:.3f}**); FullLoad OMoE gives **{seed42f.q_text:.3f}/{seed42f.q_vision:.3f}/{seed42f.q_aligned:.3f}** (MDI **{seed42f.cf_mdi:.3f}**); NoLoad OMoE gives **{seed42n.q_text:.3f}/{seed42n.q_vision:.3f}/{seed42n.q_aligned:.3f}** (MDI **{seed42n.cf_mdi:.3f}**). Load balancing partially rebalances what MoE specialization introduces.
- **Caveat**: the load term is zero-gradient at the uniform router and only reacts after the router has moved. Once softmax saturates, the load penalty cannot recover. This explains the seed-dependent collapses at λ=10⁻³, 10⁻⁴ and 1. A router z-loss or entropy floor is recommended as a collapse guard (not part of the fixed recipe here).

## 1. Setup and acceleration strategy

All models share the **same cached CSGA-aligned feature** (`analysis/samgr_sports10/results/csga_aligned.npy`), the same text/image features, the same 5-expert architecture (text-specific, vision-specific, aligned-specific, two shared), the same direct projection, outer CSGA residual, γ=1, ortho/dup/var weights (0.05/0.05/0.1), Adam (lr 1e-3, wd 1e-5), batch 512, 60 epochs, and the same fixed seed set {{42,43,44}}. **For a given seed, all six λ values start from the exact same parameter state and use the same batch order** (the model constructor is re-seeded, and no extra RNG draws are inserted before training). Early stopping is disabled identically for every run; the final epoch checkpoint is the analysis checkpoint.

Generated models: **6 λ × 3 seeds = 18**. NoLoad and FullLoad therefore have 3 paired seeds each.

Acceleration levers actually used:
1. **Cache once, reuse everywhere**: CSGA features and item metadata are computed once; all OMoE runs reuse them.
2. **Parallel shards**: 4 GPU workers train 18 OMoE models in ~20 min wall-clock (≈90 s/model).
3. **Cheap screening**: a feature-level next-item retrieval proxy on the full validation/test interactions is computed for every model in minutes; it is reported alongside the LLM-generator validation NDCG@10; the two disagree because of SID difficulty, and every performance number is labelled with its source.
4. **Shared RQ/SID budget**: each seed-42 λ gets its own ~800–1000-epoch RQ-VAE, but the same RQ-VAE is reused for original and all counterfactual interventions of that model, so intervention SID differences are not confounded by RQ retraining.
5. **Generator only for the confirmatory conditions**: NoLoad/FullLoad original + mask text/vision/aligned. Shuffle interventions are reported as representation/SID drift only (as allowed by the protocol).

## 2. Routing and expert-utilization metrics

### NoLoad vs FullLoad (mean ± std over 3 seeds)

{md_table(cmp_tab)}

### Per-seed routing metrics (all 18 models)

{md_table(routing_tab, floatfmt='.4f')}

NoLoad's collapse direction is seed-dependent (seed 42: aligned; seed 43: image+aligned; seed 44: image). This is why the NoLoad error bars in Figure 1 are large even though every NoLoad run is collapsed.

## 3. Modality contribution (seed 42, the common seed for all downstream work)

{md_table(mod_tab, floatfmt='.4f')}

Definitions follow the protocol exactly:
- **Expert-group routing mass** = sum of gate probabilities over experts of a type. It is *not* called true modality contribution.
- **Norm-aware contribution** = Σ π·‖h‖ per group, normalized; specific groups are re-normalized to text/vision/aligned for MDI.
- **Counterfactual contribution** = 1−cos(z_full, z_without_modality), normalized; main result uses dataset-mean replacement, with zero replacement as robustness.

NoLoad is almost entirely aligned-driven: aligned routing mass **{seed42n.mass_aligned:.3f}**, aligned norm-aware share **{seed42n.norm_share_aligned:.3f}**, aligned counterfactual share **{seed42n.q_aligned:.3f}**, counterfactual MDI **{seed42n.cf_mdi:.3f}**. FullLoad cuts these to **{seed42f.mass_aligned:.3f}**, **{seed42f.norm_share_aligned:.3f}**, **{seed42f.q_aligned:.3f}**, and MDI **{seed42f.cf_mdi:.3f}**. Shared experts absorb a large fraction in FullLoad (routing mass {seed42f.mass_shared:.3f}, norm-aware {seed42f.norm_share_shared:.3f}), which is expected because shared experts see the concatenation and are not modality-specific.

### Component ablation: what the MoE branch does to modality mass

{md_table(comp_tab, floatfmt='.4f')}

The direct-only model is the most modality-balanced: its projection sees the concatenation and has no expert specialization, so q_text/q_vision/q_aligned are close to 1/3 and MDI is only {nm['cf_mdi_mean']:.3f}. Adding the MoE branch (FullLoad) moves a large share of the counterfactual mass to the aligned-specific expert (q_aligned {seed42f.q_aligned:.3f}, MDI {seed42f.cf_mdi:.3f}); removing load balancing pushes it almost entirely to aligned (q_aligned {seed42n.q_aligned:.3f}, MDI {seed42n.cf_mdi:.3f}). This is the cleanest evidence that **the load objective mitigates MoE-induced modality specialization**, even though it does not fully restore the direct-only balance.

## 4. λ sweep: balance vs specialization

{md_table(sweep[['lambda_load','load_cv_mean','load_cv_std','eff_mean','dead_mean','H_mean','H_std','JS_mean','norm_mdi_mean','cf_mdi_mean','cf_mdi_std','proxy_valid_mean','proxy_test_mean']], floatfmt='.4f')}

### Generator-based validation NDCG@10 (seed 42, fixed 200-step SFT, 5k validation subset)

{md_table(gv_tab, floatfmt='.5f') if len(gv_tab) else '_Pending._'}

The continuous-feature retrieval proxy and the LLM generator **disagree**: the proxy prefers balanced λ (10⁻²/10⁻¹), while the generator prefers collapsed NoLoad because its SID space is simpler. This is the central balance-vs-SID-difficulty trade-off of the study; section 7 quantifies it.

**Reading the table.**
- λ=0: collapsed in all seeds (CV≈1.74 on average, effective experts≈1.35, entropy≈0.011, MDI≈0.856).
- λ=10⁻⁴ and 10⁻³: highly unstable. λ=10⁻⁴ has one good seed (H≈0.91) and two collapsed seeds; λ=10⁻³ collapses in seeds 42/43 and only partially balances seed 44. Mean load CV remains 0.77 / 1.12.
- λ=10⁻²: robustly balanced in 3/3 seeds (CV≈0.022, effective experts≈4.998, dead rate=0, H≈0.647, MDI≈0.253).
- **λ=10⁻¹ (FullLoad): robustly balanced in 3/3 seeds (CV≈0.018, effective experts≈4.999, dead rate=0), highest routing entropy among non-collapsed settings (H≈0.734), lowest counterfactual MDI (≈0.211), and near-best proxy NDCG@10.**
- λ=1: mean load CV rises again (0.417) because one of three seeds collapses; successful seeds have H≈0.70 and near-uniform mean loads. The load term dominates the training loss (≈1.0 vs ≈0.10 at λ=0.1), so λ=1 is too large in practice.

### Is λ=1 over-uniform?
No. In the two successful λ=1 seeds, mean gate loads are near-uniform but **per-item** routing entropy stays ≈0.70 (max 1.0), and Top-1 frequencies remain spread over all experts. The load loss acts on the *batch-mean* gate, not on each item's gate, so it equalizes average load without forcing every item to the uniform distribution. The failure mode at large λ is **collapse to a saturated wrong solution**, not over-smoothing.

## 5. Group-level analysis

{md_table(group_tab, floatfmt='.4f')}

FullLoad reduces aligned dominance in every popularity and consistency stratum. The **between-group differences are preserved**: the spread of q_aligned across head/mid/tail is {group_tab[group_tab.dimension=='popularity'].q_aligned_FullLoad.max()-group_tab[group_tab.dimension=='popularity'].q_aligned_FullLoad.min():.4f} under FullLoad (NoLoad spread {group_tab[group_tab.dimension=='popularity'].q_aligned_NoLoad.max()-group_tab[group_tab.dimension=='popularity'].q_aligned_NoLoad.min():.4f}); the analogous consistency spread is {group_tab[group_tab.dimension=='consistency'].q_aligned_FullLoad.max()-group_tab[group_tab.dimension=='consistency'].q_aligned_FullLoad.min():.4f} (NoLoad {group_tab[group_tab.dimension=='consistency'].q_aligned_NoLoad.max()-group_tab[group_tab.dimension=='consistency'].q_aligned_NoLoad.min():.4f}). The goal was not to force all groups to 1/3; it was to reduce global collapse while retaining group structure, which is what we observe.

## 6. SID, collision and codebook perplexity (λ sweep, seed 42)

{md_table(sid_tab, floatfmt='.4f') if len(sid_tab) else '_Pending: RQ-VAE/SID jobs are running._'}

FullLoad has the **lowest excess collision (15.2%) and smallest max bucket (6)**, while NoLoad has a max bucket of 25. NoLoad's lower reconstruction MSE (7.44 vs 12.21) is a collapse artifact: a near-constant representation is easy to reconstruct but produces highly duplicated SIDs. FullLoad does **not** improve codebook perplexity (35.4 vs 43.0); both are far below the 256-code ceiling.

## 7. Downstream recommendation and interventions

### NoLoad vs FullLoad (seed 42, 200-step SFT, 5k test subset)

{md_table(down_tab, floatfmt='.4f') if len(down_tab) else '_Pending: generator SFT jobs are running._'}

**Adding OMoE vs CSGA-only** at the same 200-step/5k budget does not improve raw HR either: CSGA-only HR@20 **{dc.get('HR@20') if dc else 'n/a'}** vs FullLoad **{df.get('HR@20') if df else 'n/a'}**, with essentially the same SID collision (15.3% vs 15.2%) and a slightly higher codebook perplexity (41.1 vs 35.4). The load-balanced OMoE's advantage is representation balance and modality decomposition, not raw HR at this budget.

NoLoad wins raw HR/NDCG at this budget. This is **not** evidence that collapse is better: NoLoad's SID space has 19.5% excess collision and a near-constant aligned representation, so the generator has fewer distinct targets to learn. FullLoad has lower SID collision (15.2%) and a more diverse SID space, which is harder for a 200-step generator but is the desired representation property. The existing 400-step full-test ablation shows the same ordering (NoLoad HR@20 0.0201 vs FullLoad 0.0124) but a smaller relative gap, consistent with the generator-budget explanation.

### Pre-existing component ablation (400-step SFT, full test)

{md_table(abl_tab, floatfmt='.4f') if len(abl_tab) else '_No ablation artifacts found._'}

**Add/no-MoE answer.** Adding the expert mixture (Full OMoE) improves HR@20 over the direct-only no-MoE variant (**0.0124 vs 0.0082**, +51% relative). However, removing load balancing (0.0201) or removing the direct branch (0.0336) raises HR further. The common factor is a **simpler/collapsed SID space**, not better recommendation quality. A no-MoE model has no gate, so expert-group routing mass is undefined; only counterfactual input contribution could be defined, and it requires retraining that variant.

### Mask interventions (seed 42, same 200-step budget)

{md_table(inter_tab, floatfmt='.4f') if len(inter_tab) else '_Pending: generator SFT jobs are running. Representation/SID drift is available in `results/analysis/*.json` and the SID metrics even before generator results arrive._'}

The mask-aligned results expose the collision/difficulty confound directly: masking aligned raises excess SID collision from 15.2% to 34.8% for FullLoad and to 73.6% for NoLoad, and the apparent HR@20 jumps to 0.0136 (FullLoad) and 0.1164 (NoLoad). Those gains are **collision artifacts**, not genuine modality contributions. This is why the protocol requires collision and perplexity alongside HR.

Representation drift is available for all 18 models and all six interventions (original, mask text/vision/aligned, shuffle vision/text). It is saved per item in `results/analysis/*_items.npz` and summarized in each model's JSON (`drift_mean`). NoLoad's drift under mask text/vision is smaller than FullLoad's because NoLoad ignores those modalities — another signature of dominance.

## 8. Statistical tests

### Item-level paired bootstrap (NoLoad − FullLoad)

{md_table(boot, floatfmt='.5f')}

Negative entropy/drift differences mean NoLoad is lower; positive MDI/max-share differences mean NoLoad is more dominated. All item-level contrasts are stable across seeds (p≈0).

### User-level paired bootstrap for recommendation

{md_table(bootu, floatfmt='.5f') if len(bootu) else '_Pending: requires paired generator evaluation outputs. The script `scripts/analyze_results.py` computes this automatically once `results/downstream/noload_seed42/eval/final_result_test.json` and `.../fullload_seed42/...` exist._'}

## 9. Figures

- **Figure 1** (`figures/fig1_expert_mean_gate.pdf/png`): expert mean gate weight, NoLoad vs FullLoad, colour = expert type.
- **Figure 2** (`figures/fig2_balance_stats.pdf/png`): load CV, effective experts, dead rate, routing JS; points are individual seeds.
- **Figure 3** (`figures/fig3_modality_contribution.pdf/png`): three stacked decompositions — routing mass, norm-aware contribution, counterfactual input contribution.
- **Figure 4** (`figures/fig4_lambda_tradeoff.pdf/png`): λ sweep (symlog) — generator-based validation NDCG@10 (seed 42, 200-step), load CV, counterfactual MDI, routing JS.
- **Figure 5** (`figures/fig5_group_modality.pdf/png`): q_text/q_vision/q_aligned for popularity and text–image consistency strata.
- **Figure 6** (`figures/fig6_routing_heatmap.pdf/png`): 800-item gate heatmaps for NoLoad and FullLoad, sorted by Top-1 expert then entropy.

## 10. Direct answers to the seven report questions

1. **Does NoLoad have a few high-load experts?** Yes, unambiguously. NoLoad has load CV **{nol.load_cv.mean():.2f}**, effective experts **{nol.effective_experts.mean():.2f}**, dead-expert rate **{nol['dead_rate_0.01M'].mean():.2f}** (0.01/M threshold), and **{100*nol['dead_rate_1e-3'].mean():.0f}%** dead at 1e-3. In each seed one expert receives essentially all mass (mean gate ≥0.995 in 2/3 seeds; the remaining seed splits between two experts).
2. **Does load balancing reduce load CV and dead-expert rate?** Yes for λ=10⁻² and λ=10⁻¹: CV falls from **{nol.load_cv.mean():.2f}** to **{full.load_cv.mean():.4f}**, effective experts rise from **{nol.effective_experts.mean():.2f}** to **{full.effective_experts.mean():.3f}**, and the dead rate at 0.01/M falls from **{nol['dead_rate_0.01M'].mean():.2f}** to **{full['dead_rate_0.01M'].mean():.2f}**. However, the improvement is **not monotone in λ**: λ=10⁻³/10⁻⁴ can still collapse, and even λ=1 collapses in one seed.
3. **Does it reduce modality dominance?** Yes. Counterfactual MDI falls from **{nol.cf_mdi.mean():.3f}** to **{full.cf_mdi.mean():.3f}**; max counterfactual share falls from **{nol.cf_max_share.mean():.3f}** to **{full.cf_max_share.mean():.3f}**; aligned counterfactual contribution falls from **{nol.q_aligned.mean():.3f}** to **{full.q_aligned.mean():.3f}**. Norm-aware MDI also falls from **{nol.norm_mdi.mean():.3f}** to **{full.norm_mdi.mean():.3f}**. The aligned branch remains the largest contributor, so dominance is **mitigated, not eliminated**.
4. **Is item-level routing diversity preserved?** Yes for FullLoad. Normalized routing entropy is **{full.entropy_mean.mean():.3f} ± {full.entropy_mean.std():.3f}** (NoLoad {nol.entropy_mean.mean():.3f}); pairwise JS is **{full.routing_js_mean.mean():.4f}** (NoLoad {nol.routing_js_mean.mean():.4f}). NoLoad's pooled variance/JS is inflated by **between-seed collapse direction**: within a single-expert seed its JS is ≈0.002 and gate variance ≈0, while seed 43 splits between two experts and has JS 0.35 / variance 0.0995. FullLoad is balanced in the mean and instance-dependent **within every seed**.
5. **Does an excessively large λ over-uniformize routing?** Not in the item-level sense: λ=1's successful seeds retain H≈0.70 and spread Top-1 assignments. But λ=1 is **too large for stable optimization** (one seed collapses; training loss is dominated by the load term). The practical recommendation is not to use λ=1.
6. **Does the best λ compromise performance, balance, and modality contribution?** **No single λ does at the tested generator budget.** λ=0.1 (FullLoad) is the best **representation-level** compromise: load CV **{full.load_cv.mean():.4f}**, effective experts **{full.effective_experts.mean():.3f}**, H **{full.entropy_mean.mean():.3f}**, MDI **{full.cf_mdi.mean():.3f}**, and the lowest SID collision (15.2%). But the raw generator NDCG@10 is higher for NoLoad (validation {gen_val.get(0.0, {}).get('NDCG@10', 'n/a')} vs {gen_val.get(FULL_LAMBDA, {}).get('NDCG@10', 'n/a')}). λ=10⁻² is the second-best balance point but has lower routing diversity and higher MDI. The Pareto frontier is therefore two-dimensional: **λ=0 for raw HR at a fixed low budget, λ=0.1 for expert balance, modality balance, routing diversity and SID uniqueness.**
7. **Does FullLoad improve SID collision, perplexity, or downstream recommendation?** **SID collision: yes** (15.2% vs 19.5% excess collision; max bucket 6 vs 25). **Codebook perplexity: no clear improvement** (35.4 vs 43.0; both are far below the 256-code ceiling). **Downstream recommendation: no at the tested 200-step budget** — NoLoad HR@20 **{dn.get('HR@20') if dn else 'n/a'}** vs FullLoad **{df.get('HR@20') if df else 'n/a'}**, NDCG@10 **{dn.get('NDCG@10') if dn else 'n/a'}** vs **{df.get('NDCG@10') if df else 'n/a'}**. The existing 400-step full-test ablation reproduces the direction (NoLoad 0.0201 vs FullLoad 0.0124). This is the collision/difficulty trade-off: collapsed SID spaces are easier for an undertrained generator. We therefore report FullLoad as a representation-quality improvement, **not** as a downstream-HR improvement at this budget.

## 11. Limitations

- The load objective is the **batch-mean Switch-style** penalty used by the production OMoE. It has zero gradient at the uniform router and can be defeated by softmax saturation. This is why we report seed-level collapses rather than only seed averages.
- RQ-VAE is trained at a fixed **~800–1000-epoch** budget (production uses 5000), and codebook perplexity/collision are therefore not directly comparable to the production absolute numbers; comparisons **across λ within this study** are controlled.
- Performance is reported at two levels: a **continuous-feature retrieval proxy** for all 18 models, and a **generator-based validation NDCG@10** for seed 42 at a fixed 200-step SFT budget. The two disagree (§4), and the report labels the source of every number.
- Downstream generator results use one fixed seed (42) and a fixed SFT budget; 3-seed generator replication is out of scope for the rapid protocol.
- The `FullLoad` value 0.1 is the current validation-selected production value; this study re-validates it rather than re-selecting it from scratch.

## 12. Reproducibility

```bash
cd analysis/omoe_load_balance
# 18 OMoE models (4 shards; ~20 min on 4 GPUs)
CUDA_VISIBLE_DEVICES=0 python scripts/run_representation.py --gpu 0 --shard 0 --num-shards 4
# ... shards 1-3
python scripts/run_proxy.py --gpu 0
python scripts/analyze_results.py
python scripts/make_figures.py
# RQ/SID: scripts/make_sid.py + rq/rqvae.py
# Downstream: scripts/run_downstream_lb.py --name ... --emb ...
```

Artifacts:
- `results/checkpoints/` — 18 model checkpoints.
- `results/analysis/` — per-item `pi`, `h`, direct/mixture/fused/z, counterfactual z, per-item metrics (`.npz`), and summary JSON.
- `results/tables/` — master CSV, sweep aggregates, bootstrap results.
- `results/sid/`, `results/rq/`, `results/downstream/` — SID and generator artifacts.
- `figures/` — PDF + 300 dpi PNG.
"""
    out = os.path.join(ROOT, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
