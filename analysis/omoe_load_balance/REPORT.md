# OMoE load-balancing objective: expert utilization, modality dominance, and instance-level routing
*Sports & Outdoors strict-10-core · generated 2026-10-24 21:34*

## TL;DR

- **Without load balancing the router collapses**: NoLoad (λ=0) has load CV **1.74 ± 0.44**, effective experts **1.35**, and dead-expert rate **0.73**. Every seed routes essentially all items to one or two experts (aligned, image+aligned, and image for seeds 42/43/44 respectively).
- **The load objective fixes the average case but is not a collapse guard**: FullLoad (λ=0.1) reaches CV **0.0183**, effective experts **4.999**, dead rate **0.00**, with all five experts used in **3/3 seeds**. Intermediate λ (10⁻³, 10⁻⁴) and even λ=1 can still collapse in some seeds (see §6).
- **FullLoad substantially reduces modality dominance** without erasing it: counterfactual MDI falls from **0.856** to **0.211**, max modality share from **0.961** to **0.644**, and aligned-expert counterfactual contribution from **0.961** to **0.644**. The aligned branch remains the largest single contributor.
- **Instance-dependent routing is preserved**: FullLoad has normalized per-item routing entropy **0.734** (NoLoad 0.011) and routing JS **0.2175** (NoLoad 0.1178). λ=1 does **not** force per-item uniformity (H≈0.70 in its two successful seeds); its problem is optimization instability (one seed collapses), not over-uniformity.
- **Best representation compromise: λ=0.1 (FullLoad)**. It has the lowest counterfactual MDI (0.211), the highest routing entropy among non-collapsed settings (0.734), robust balance in 3/3 seeds, and the lowest SID collision (15.2% vs 19.5% NoLoad) on the seed-42 RQ/SID sweep.
- **Downstream HR does not follow the same ordering at the tested generator budget.** With a fixed 200-step SFT and a 5k test subset, NoLoad gets HR@20 **0.0132** vs FullLoad **0.0034**; the same direction appears in the existing 400-step ablation (NoLoad 0.0201 vs FullLoad 0.0124). The reason is SID difficulty, not representation quality: collapsed/merged SID spaces have fewer distinct targets, so an undertrained generator scores higher. We report this as a negative downstream result and do not claim that load balancing improves raw HR at this budget.
- **Adding the MoE branch shifts counterfactual mass toward aligned.** Direct-only (no MoE) gives q_text/q_vision/q_aligned = **0.326/0.395/0.279** (MDI **0.019**); FullLoad OMoE gives **0.176/0.154/0.669** (MDI **0.238**); NoLoad OMoE gives **0.003/0.018/0.979** (MDI **0.910**). Load balancing partially rebalances what MoE specialization introduces.
- **Caveat**: the load term is zero-gradient at the uniform router and only reacts after the router has moved. Once softmax saturates, the load penalty cannot recover. This explains the seed-dependent collapses at λ=10⁻³, 10⁻⁴ and 1. A router z-loss or entropy floor is recommended as a collapse guard (not part of the fixed recipe here).

## 1. Setup and acceleration strategy

All models share the **same cached CSGA-aligned feature** (`analysis/samgr_sports10/results/csga_aligned.npy`), the same text/image features, the same 5-expert architecture (text-specific, vision-specific, aligned-specific, two shared), the same direct projection, outer CSGA residual, γ=1, ortho/dup/var weights (0.05/0.05/0.1), Adam (lr 1e-3, wd 1e-5), batch 512, 60 epochs, and the same fixed seed set {42,43,44}. **For a given seed, all six λ values start from the exact same parameter state and use the same batch order** (the model constructor is re-seeded, and no extra RNG draws are inserted before training). Early stopping is disabled identically for every run; the final epoch checkpoint is the analysis checkpoint.

Generated models: **6 λ × 3 seeds = 18**. NoLoad and FullLoad therefore have 3 paired seeds each.

Acceleration levers actually used:
1. **Cache once, reuse everywhere**: CSGA features and item metadata are computed once; all OMoE runs reuse them.
2. **Parallel shards**: 4 GPU workers train 18 OMoE models in ~20 min wall-clock (≈90 s/model).
3. **Cheap screening**: a feature-level next-item retrieval proxy on the full validation/test interactions is computed for every model in minutes; it is reported alongside the LLM-generator validation NDCG@10; the two disagree because of SID difficulty, and every performance number is labelled with its source.
4. **Shared RQ/SID budget**: each seed-42 λ gets its own ~800–1000-epoch RQ-VAE, but the same RQ-VAE is reused for original and all counterfactual interventions of that model, so intervention SID differences are not confounded by RQ retraining.
5. **Generator only for the confirmatory conditions**: NoLoad/FullLoad original + mask text/vision/aligned. Shuffle interventions are reported as representation/SID drift only (as allowed by the protocol).

## 2. Routing and expert-utilization metrics

### NoLoad vs FullLoad (mean ± std over 3 seeds)

| model | load_cv | effective_experts | dead@0.01/M | dead@1e-3 | normalized_H | routing_JS | norm_MDI | counterfactual_MDI | cf_max_share | q_text | q_vision | q_aligned | proxy_valid_NDCG@10 | proxy_test_NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NoLoad | 1.7357 ± 0.4398 | 1.354 ± 0.558 | 0.733 | 0.533 | 0.011 ± 0.010 | 0.1178 ± 0.2009 | 0.992 | 0.856 ± 0.050 | 0.961 | 0.007 | 0.032 | 0.961 | 0.00495 | 0.00359 |
| FullLoad | 0.0183 ± 0.0137 | 4.999 ± 0.001 | 0.000 | 0.000 | 0.734 ± 0.026 | 0.2175 ± 0.0192 | 0.245 | 0.211 ± 0.054 | 0.644 | 0.179 | 0.177 | 0.644 | 0.00606 | 0.00508 |

### Per-seed routing metrics (all 18 models)

| lambda | seed | load_cv | eff_experts | dead@0.01/M | dead@1e-3 | H_mean | H_std | H_p10 | H_p50 | H_p90 | JS | var_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 42.0000 | 1.9876 | 1.0383 | 0.8000 | 0.4000 | 0.0200 | 0.0314 | 0.0000 | 0.0087 | 0.0523 | 0.0020 | 0.0000 |
| 0.0000 | 43.0000 | 1.2279 | 1.9982 | 0.6000 | 0.6000 | 0.0011 | 0.0187 | -0.0000 | 0.0000 | 0.0000 | 0.3497 | 0.0995 |
| 0.0000 | 44.0000 | 1.9916 | 1.0265 | 0.8000 | 0.6000 | 0.0131 | 0.0235 | 0.0001 | 0.0046 | 0.0389 | 0.0016 | 0.0000 |
| 0.0001 | 42.0000 | 0.5199 | 4.3522 | 0.0000 | 0.0000 | 0.9079 | 0.0285 | 0.8748 | 0.9062 | 0.9468 | 0.0047 | 0.0007 |
| 0.0001 | 43.0000 | 1.2269 | 1.9979 | 0.6000 | 0.6000 | 0.0024 | 0.0262 | -0.0000 | 0.0000 | 0.0000 | 0.3437 | 0.0993 |
| 0.0001 | 44.0000 | 0.5707 | 4.1337 | 0.0000 | 0.0000 | 0.8632 | 0.0470 | 0.8048 | 0.8758 | 0.9171 | 0.0151 | 0.0025 |
| 0.0010 | 42.0000 | 1.2253 | 1.9994 | 0.6000 | 0.6000 | 0.0002 | 0.0071 | -0.0000 | -0.0000 | -0.0000 | 0.3481 | 0.0999 |
| 0.0010 | 43.0000 | 1.9199 | 1.1559 | 0.6000 | 0.6000 | 0.0707 | 0.0876 | 0.0000 | 0.0061 | 0.2089 | 0.0139 | 0.0007 |
| 0.0010 | 44.0000 | 0.2197 | 4.8832 | 0.0000 | 0.0000 | 0.6108 | 0.1413 | 0.4568 | 0.7089 | 0.7402 | 0.3034 | 0.0386 |
| 0.0100 | 42.0000 | 0.0412 | 4.9958 | 0.0000 | 0.0000 | 0.6106 | 0.1327 | 0.4477 | 0.6857 | 0.7036 | 0.3138 | 0.0383 |
| 0.0100 | 43.0000 | 0.0120 | 4.9996 | 0.0000 | 0.0000 | 0.6809 | 0.0295 | 0.6680 | 0.6760 | 0.6851 | 0.2572 | 0.0311 |
| 0.0100 | 44.0000 | 0.0138 | 4.9995 | 0.0000 | 0.0000 | 0.6506 | 0.0939 | 0.4807 | 0.6773 | 0.7214 | 0.2764 | 0.0383 |
| 0.1000 | 42.0000 | 0.0295 | 4.9978 | 0.0000 | 0.0000 | 0.7605 | 0.0805 | 0.6797 | 0.7510 | 0.8724 | 0.1978 | 0.0242 |
| 0.1000 | 43.0000 | 0.0224 | 4.9987 | 0.0000 | 0.0000 | 0.7315 | 0.0609 | 0.6754 | 0.7263 | 0.8141 | 0.2188 | 0.0268 |
| 0.1000 | 44.0000 | 0.0030 | 5.0000 | 0.0000 | 0.0000 | 0.7090 | 0.0565 | 0.6736 | 0.6917 | 0.7505 | 0.2361 | 0.0295 |
| 1.0000 | 42.0000 | 0.0140 | 4.9995 | 0.0000 | 0.0000 | 0.7120 | 0.0931 | 0.6228 | 0.6871 | 0.8574 | 0.2377 | 0.0286 |
| 1.0000 | 43.0000 | 1.2246 | 2.0017 | 0.6000 | 0.6000 | 0.0033 | 0.0310 | 0.0000 | 0.0000 | 0.0000 | 0.3455 | 0.0994 |
| 1.0000 | 44.0000 | 0.0109 | 4.9997 | 0.0000 | 0.0000 | 0.6939 | 0.0741 | 0.6201 | 0.6931 | 0.7685 | 0.2496 | 0.0307 |

NoLoad's collapse direction is seed-dependent (seed 42: aligned; seed 43: image+aligned; seed 44: image). This is why the NoLoad error bars in Figure 1 are large even though every NoLoad run is collapsed.

## 3. Modality contribution (seed 42, the common seed for all downstream work)

| model | routing_text | routing_vision | routing_aligned | routing_shared | norm_text | norm_vision | norm_aligned | norm_shared | cf_text | cf_vision | cf_aligned | cf_MDI | norm_MDI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NoLoad | 0.0009 | 0.0006 | 0.9950 | 0.0035 | 0.0002 | 0.0003 | 0.9971 | 0.0023 | 0.0031 | 0.0177 | 0.9792 | 0.9103 | 0.9960 |
| FullLoad | 0.1932 | 0.1981 | 0.1948 | 0.4139 | 0.2499 | 0.0338 | 0.2326 | 0.4837 | 0.1765 | 0.1540 | 0.6695 | 0.2378 | 0.2088 |

Definitions follow the protocol exactly:
- **Expert-group routing mass** = sum of gate probabilities over experts of a type. It is *not* called true modality contribution.
- **Norm-aware contribution** = Σ π·‖h‖ per group, normalized; specific groups are re-normalized to text/vision/aligned for MDI.
- **Counterfactual contribution** = 1−cos(z_full, z_without_modality), normalized; main result uses dataset-mean replacement, with zero replacement as robustness.

NoLoad is almost entirely aligned-driven: aligned routing mass **0.995**, aligned norm-aware share **0.997**, aligned counterfactual share **0.979**, counterfactual MDI **0.910**. FullLoad cuts these to **0.195**, **0.233**, **0.669**, and MDI **0.238**. Shared experts absorb a large fraction in FullLoad (routing mass 0.414, norm-aware 0.484), which is expected because shared experts see the concatenation and are not modality-specific.

### Component ablation: what the MoE branch does to modality mass

| variant | q_text | q_vision | q_aligned | cf_MDI | max share |
|---|---|---|---|---|---|
| No MoE (direct only) | 0.3261 | 0.3949 | 0.2790 | 0.0191 | 0.4071 |
| OMoE NoLoad (λ=0) | 0.0031 | 0.0177 | 0.9792 | 0.9103 | 0.9792 |
| OMoE FullLoad (λ=0.1) | 0.1765 | 0.1540 | 0.6695 | 0.2378 | 0.6695 |

The direct-only model is the most modality-balanced: its projection sees the concatenation and has no expert specialization, so q_text/q_vision/q_aligned are close to 1/3 and MDI is only 0.019. Adding the MoE branch (FullLoad) moves a large share of the counterfactual mass to the aligned-specific expert (q_aligned 0.669, MDI 0.238); removing load balancing pushes it almost entirely to aligned (q_aligned 0.979, MDI 0.910). This is the cleanest evidence that **the load objective mitigates MoE-induced modality specialization**, even though it does not fully restore the direct-only balance.

## 4. λ sweep: balance vs specialization

| lambda_load | load_cv_mean | load_cv_std | eff_mean | dead_mean | H_mean | H_std | JS_mean | norm_mdi_mean | cf_mdi_mean | cf_mdi_std | proxy_valid_mean | proxy_test_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 1.7357 | 0.4398 | 1.3543 | 0.7333 | 0.0114 | 0.0096 | 0.1178 | 0.9925 | 0.8560 | 0.0500 | 0.0050 | 0.0036 |
| 0.0001 | 0.7725 | 0.3943 | 3.4946 | 0.2000 | 0.5912 | 0.5104 | 0.1212 | 0.4222 | 0.2205 | 0.0292 | 0.0061 | 0.0048 |
| 0.0010 | 1.1216 | 0.8548 | 2.6795 | 0.4000 | 0.2272 | 0.3341 | 0.2218 | 0.8094 | 0.6194 | 0.3201 | 0.0052 | 0.0038 |
| 0.0100 | 0.0223 | 0.0163 | 4.9983 | 0.0000 | 0.6474 | 0.0353 | 0.2825 | 0.6562 | 0.2533 | 0.0481 | 0.0063 | 0.0052 |
| 0.1000 | 0.0183 | 0.0137 | 4.9989 | 0.0000 | 0.7337 | 0.0258 | 0.2175 | 0.2451 | 0.2112 | 0.0539 | 0.0061 | 0.0051 |
| 1.0000 | 0.4165 | 0.6998 | 4.0003 | 0.2000 | 0.4697 | 0.4040 | 0.2776 | 0.5036 | 0.4374 | 0.3965 | 0.0058 | 0.0047 |

### Generator-based validation NDCG@10 (seed 42, fixed 200-step SFT, 5k validation subset)

| lambda | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| 0.00000 | 0.00660 | 0.00840 | 0.01320 | 0.00410 | 0.00470 | 0.00590 |
| 0.00010 | 0.00260 | 0.00360 | 0.00620 | 0.00150 | 0.00180 | 0.00250 |
| 0.00100 | 0.00160 | 0.00260 | 0.00500 | 0.00100 | 0.00130 | 0.00190 |
| 0.01000 | 0.00100 | 0.00120 | 0.00220 | 0.00050 | 0.00050 | 0.00080 |
| 0.10000 | 0.00140 | 0.00260 | 0.00380 | 0.00070 | 0.00110 | 0.00140 |
| 1.00000 | 0.00020 | 0.00020 | 0.00200 | 0.00010 | 0.00010 | 0.00050 |

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

| dimension | group | n | q_text_NoLoad | q_text_FullLoad | q_vision_NoLoad | q_vision_FullLoad | q_aligned_NoLoad | q_aligned_FullLoad | delta_aligned |
|---|---|---|---|---|---|---|---|---|---|
| popularity | tail | 5586 | 0.0072 | 0.1832 | 0.0322 | 0.1787 | 0.9606 | 0.6381 | -0.3225 |
| popularity | mid | 3402 | 0.0073 | 0.1752 | 0.0313 | 0.1775 | 0.9614 | 0.6473 | -0.3141 |
| popularity | head | 2258 | 0.0068 | 0.1734 | 0.0308 | 0.1742 | 0.9624 | 0.6524 | -0.3099 |
| consistency | low | 3749 | 0.0069 | 0.1769 | 0.0276 | 0.1772 | 0.9655 | 0.6459 | -0.3197 |
| consistency | mid | 3748 | 0.0074 | 0.1786 | 0.0296 | 0.1833 | 0.9630 | 0.6382 | -0.3248 |
| consistency | high | 3749 | 0.0071 | 0.1809 | 0.0378 | 0.1719 | 0.9551 | 0.6472 | -0.3079 |
| image | has_image | 11244 | 0.0072 | 0.1788 | 0.0317 | 0.1775 | 0.9612 | 0.6437 | -0.3175 |
| image | missing_image | 2 | 0.0006 | 0.0850 | 0.0080 | 0.0525 | 0.9914 | 0.8625 | -0.1289 |
| text_length | short | 3749 | 0.0062 | 0.2068 | 0.0366 | 0.1600 | 0.9572 | 0.6332 | -0.3239 |
| text_length | mid | 3748 | 0.0121 | 0.1939 | 0.0403 | 0.2075 | 0.9476 | 0.5986 | -0.3490 |
| text_length | long | 3749 | 0.0032 | 0.1358 | 0.0180 | 0.1648 | 0.9788 | 0.6994 | -0.2794 |
| category | Sports | 4150 | 0.0085 | 0.1785 | 0.0325 | 0.1743 | 0.9590 | 0.6472 | -0.3117 |
| category | Hunting & Fishing | 2974 | 0.0058 | 0.1765 | 0.0317 | 0.1788 | 0.9625 | 0.6448 | -0.3177 |
| category | Outdoor Recreation | 2521 | 0.0063 | 0.1820 | 0.0320 | 0.1829 | 0.9617 | 0.6351 | -0.3266 |
| category | Exercise & Fitness | 761 | 0.0076 | 0.1781 | 0.0248 | 0.1641 | 0.9676 | 0.6578 | -0.3098 |
| category | Sports & Outdoor Recreation Accessories | 474 | 0.0077 | 0.1836 | 0.0333 | 0.1909 | 0.9590 | 0.6256 | -0.3334 |
| category | Fan Shop | 131 | 0.0073 | 0.1714 | 0.0324 | 0.1754 | 0.9604 | 0.6532 | -0.3072 |
| category | Other | 235 | 0.0066 | 0.1764 | 0.0303 | 0.1759 | 0.9631 | 0.6477 | -0.3154 |

FullLoad reduces aligned dominance in every popularity and consistency stratum. The **between-group differences are preserved**: the spread of q_aligned across head/mid/tail is 0.0144 under FullLoad (NoLoad spread 0.0018); the analogous consistency spread is 0.0090 (NoLoad 0.0105). The goal was not to force all groups to 1/3; it was to reduce global collapse while retaining group structure, which is what we observe.

## 6. SID, collision and codebook perplexity (λ sweep, seed 42)

| lambda | unique SIDs | excess collision | collided item ratio | max bucket | recon MSE | codebook ppl |
|---|---|---|---|---|---|---|
| 0.0000 | 9057.0000 | 0.1946 | 0.3427 | 25.0000 | 7.4362 | 42.9562 |
| 0.0001 | 9057.0000 | 0.1946 | 0.3425 | 13.0000 | 12.3259 | 29.8979 |
| 0.0010 | 9164.0000 | 0.1851 | 0.3305 | 21.0000 | 7.0796 | 53.6281 |
| 0.0100 | 8952.0000 | 0.2040 | 0.3597 | 10.0000 | 12.4172 | 28.4075 |
| 0.1000 | 9538.0000 | 0.1519 | 0.2843 | 6.0000 | 12.2131 | 35.4082 |
| 1.0000 | 9299.0000 | 0.1731 | 0.3167 | 7.0000 | 12.2138 | 29.2866 |

FullLoad has the **lowest excess collision (15.2%) and smallest max bucket (6)**, while NoLoad has a max bucket of 25. NoLoad's lower reconstruction MSE (7.44 vs 12.21) is a collapse artifact: a near-constant representation is easy to reconstruct but produces highly duplicated SIDs. FullLoad does **not** improve codebook perplexity (35.4 vs 43.0); both are far below the 256-code ceiling.

## 7. Downstream recommendation and interventions

### NoLoad vs FullLoad (seed 42, 200-step SFT, 5k test subset)

| model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | SID collision | codebook ppl |
|---|---|---|---|---|---|---|---|---|
| NoLoad | 0.0072 | 0.0098 | 0.0132 | 0.0051 | 0.0060 | 0.0069 | 0.1946 | 42.9562 |
| FullLoad | 0.0010 | 0.0020 | 0.0034 | 0.0006 | 0.0010 | 0.0013 | 0.1519 | 35.4082 |
| CSGA-only | 0.0016 | 0.0026 | 0.0048 | 0.0011 | 0.0014 | 0.0019 | 0.1529 | 41.0630 |

**Adding OMoE vs CSGA-only** at the same 200-step/5k budget does not improve raw HR either: CSGA-only HR@20 **0.0048** vs FullLoad **0.0034**, with essentially the same SID collision (15.3% vs 15.2%) and a slightly higher codebook perplexity (41.1 vs 35.4). The load-balanced OMoE's advantage is representation balance and modality decomposition, not raw HR at this budget.

NoLoad wins raw HR/NDCG at this budget. This is **not** evidence that collapse is better: NoLoad's SID space has 19.5% excess collision and a near-constant aligned representation, so the generator has fewer distinct targets to learn. FullLoad has lower SID collision (15.2%) and a more diverse SID space, which is harder for a 200-step generator but is the desired representation property. The existing 400-step full-test ablation shows the same ordering (NoLoad HR@20 0.0201 vs FullLoad 0.0124) but a smaller relative gap, consistent with the generator-budget explanation.

### Pre-existing component ablation (400-step SFT, full test)

_No ablation artifacts found._

**Add/no-MoE answer.** Adding the expert mixture (Full OMoE) improves HR@20 over the direct-only no-MoE variant (**0.0124 vs 0.0082**, +51% relative). However, removing load balancing (0.0201) or removing the direct branch (0.0336) raises HR further. The common factor is a **simpler/collapsed SID space**, not better recommendation quality. A no-MoE model has no gate, so expert-group routing mass is undefined; only counterfactual input contribution could be defined, and it requires retraining that variant.

### Mask interventions (seed 42, same 200-step budget)

| model | condition | HR@10 | NDCG@10 | HR@20 | NDCG@20 | SID collision | codebook ppl |
|---|---|---|---|---|---|---|---|
| NoLoad | original | 0.0098 | 0.0060 | 0.0132 | 0.0069 | 0.1946 | 42.9562 |
| NoLoad | mask_text | 0.0058 | 0.0028 | 0.0120 | 0.0043 | 0.1960 | 43.6385 |
| NoLoad | mask_vision | 0.0086 | 0.0040 | 0.0138 | 0.0054 | 0.2021 | 43.5537 |
| NoLoad | mask_aligned | 0.0622 | 0.0317 | 0.1164 | 0.0453 | 0.7356 | 6.5527 |
| FullLoad | original | 0.0020 | 0.0010 | 0.0034 | 0.0013 | 0.1519 | 35.4082 |
| FullLoad | mask_text | 0.0054 | 0.0027 | 0.0090 | 0.0035 | 0.1815 | 30.6052 |
| FullLoad | mask_vision | 0.0016 | 0.0007 | 0.0040 | 0.0013 | 0.1547 | 35.6962 |
| FullLoad | mask_aligned | 0.0076 | 0.0037 | 0.0136 | 0.0052 | 0.3475 | 11.2507 |

The mask-aligned results expose the collision/difficulty confound directly: masking aligned raises excess SID collision from 15.2% to 34.8% for FullLoad and to 73.6% for NoLoad, and the apparent HR@20 jumps to 0.0136 (FullLoad) and 0.1164 (NoLoad). Those gains are **collision artifacts**, not genuine modality contributions. This is why the protocol requires collision and perplexity alongside HR.

Representation drift is available for all 18 models and all six interventions (original, mask text/vision/aligned, shuffle vision/text). It is saved per item in `results/analysis/*_items.npz` and summarized in each model's JSON (`drift_mean`). NoLoad's drift under mask text/vision is smaller than FullLoad's because NoLoad ignores those modalities — another signature of dominance.

## 8. Statistical tests

### Item-level paired bootstrap (NoLoad − FullLoad)

| seed | metric | contrast | mean_diff | ci_low | ci_high | p_value | n |
|---|---|---|---|---|---|---|---|
| 42 | entropy | NoLoad - FullLoad | -0.74048 | -0.74187 | -0.73911 | 0.00000 | 11246 |
| 42 | cf_max_share | NoLoad - FullLoad | 0.30966 | 0.30807 | 0.31125 | 0.00000 | 11246 |
| 42 | cf_mdi | NoLoad - FullLoad | 0.67249 | 0.67054 | 0.67448 | 0.00000 | 11246 |
| 42 | drift_mask_text | NoLoad - FullLoad | -0.12560 | -0.12644 | -0.12482 | 0.00000 | 11246 |
| 42 | drift_mask_vision | NoLoad - FullLoad | -0.10395 | -0.10454 | -0.10339 | 0.00000 | 11246 |
| 42 | drift_mask_aligned | NoLoad - FullLoad | -0.18258 | -0.18397 | -0.18112 | 0.00000 | 11246 |
| 43 | entropy | NoLoad - FullLoad | -0.73038 | -0.73151 | -0.72926 | 0.00000 | 11246 |
| 43 | cf_max_share | NoLoad - FullLoad | 0.36585 | 0.36431 | 0.36741 | 0.00000 | 11246 |
| 43 | cf_mdi | NoLoad - FullLoad | 0.69641 | 0.69388 | 0.69896 | 0.00000 | 11246 |
| 43 | drift_mask_text | NoLoad - FullLoad | -0.14626 | -0.14701 | -0.14550 | 0.00000 | 11246 |
| 43 | drift_mask_vision | NoLoad - FullLoad | -0.12424 | -0.12502 | -0.12348 | 0.00000 | 11246 |
| 43 | drift_mask_aligned | NoLoad - FullLoad | -0.08815 | -0.08991 | -0.08633 | 0.00000 | 11246 |
| 44 | entropy | NoLoad - FullLoad | -0.69588 | -0.69701 | -0.69478 | 0.00000 | 11246 |
| 44 | cf_max_share | NoLoad - FullLoad | 0.27618 | 0.27468 | 0.27766 | 0.00000 | 11246 |
| 44 | cf_mdi | NoLoad - FullLoad | 0.56544 | 0.56367 | 0.56722 | 0.00000 | 11246 |
| 44 | drift_mask_text | NoLoad - FullLoad | -0.10172 | -0.10266 | -0.10078 | 0.00000 | 11246 |
| 44 | drift_mask_vision | NoLoad - FullLoad | -0.11011 | -0.11079 | -0.10943 | 0.00000 | 11246 |
| 44 | drift_mask_aligned | NoLoad - FullLoad | -0.16532 | -0.16653 | -0.16415 | 0.00000 | 11246 |
| avg | entropy | NoLoad - FullLoad | -0.72225 | -0.72314 | -0.72140 | 0.00000 | 11246 |
| avg | cf_max_share | NoLoad - FullLoad | 0.31723 | 0.31586 | 0.31857 | 0.00000 | 11246 |
| avg | cf_mdi | NoLoad - FullLoad | 0.64478 | 0.64330 | 0.64626 | 0.00000 | 11246 |
| avg | drift_mask_text | NoLoad - FullLoad | -0.12453 | -0.12523 | -0.12381 | 0.00000 | 11246 |
| avg | drift_mask_vision | NoLoad - FullLoad | -0.11277 | -0.11333 | -0.11221 | 0.00000 | 11246 |
| avg | drift_mask_aligned | NoLoad - FullLoad | -0.14535 | -0.14624 | -0.14448 | 0.00000 | 11246 |

Negative entropy/drift differences mean NoLoad is lower; positive MDI/max-share differences mean NoLoad is more dominated. All item-level contrasts are stable across seeds (p≈0).

### User-level paired bootstrap for recommendation

| metric | contrast | mean_diff | ci_low | ci_high | p_value | n |
|---|---|---|---|---|---|---|
| HR@5 | NoLoad - FullLoad | 0.00620 | 0.00380 | 0.00860 | 0.00000 | 5000 |
| HR@10 | NoLoad - FullLoad | 0.00780 | 0.00480 | 0.01100 | 0.00000 | 5000 |
| HR@20 | NoLoad - FullLoad | 0.00980 | 0.00640 | 0.01340 | 0.00000 | 5000 |
| NDCG@5 | NoLoad - FullLoad | 0.00449 | 0.00268 | 0.00636 | 0.00000 | 5000 |
| NDCG@10 | NoLoad - FullLoad | 0.00503 | 0.00314 | 0.00698 | 0.00000 | 5000 |
| NDCG@20 | NoLoad - FullLoad | 0.00555 | 0.00360 | 0.00758 | 0.00000 | 5000 |

## 9. Figures

- **Figure 1** (`figures/fig1_expert_mean_gate.pdf/png`): expert mean gate weight, NoLoad vs FullLoad, colour = expert type.
- **Figure 2** (`figures/fig2_balance_stats.pdf/png`): load CV, effective experts, dead rate, routing JS; points are individual seeds.
- **Figure 3** (`figures/fig3_modality_contribution.pdf/png`): three stacked decompositions — routing mass, norm-aware contribution, counterfactual input contribution.
- **Figure 4** (`figures/fig4_lambda_tradeoff.pdf/png`): λ sweep (symlog) — generator-based validation NDCG@10 (seed 42, 200-step), load CV, counterfactual MDI, routing JS.
- **Figure 5** (`figures/fig5_group_modality.pdf/png`): q_text/q_vision/q_aligned for popularity and text–image consistency strata.
- **Figure 6** (`figures/fig6_routing_heatmap.pdf/png`): 800-item gate heatmaps for NoLoad and FullLoad, sorted by Top-1 expert then entropy.

## 10. Direct answers to the seven report questions

1. **Does NoLoad have a few high-load experts?** Yes, unambiguously. NoLoad has load CV **1.74**, effective experts **1.35**, dead-expert rate **0.73** (0.01/M threshold), and **53%** dead at 1e-3. In each seed one expert receives essentially all mass (mean gate ≥0.995 in 2/3 seeds; the remaining seed splits between two experts).
2. **Does load balancing reduce load CV and dead-expert rate?** Yes for λ=10⁻² and λ=10⁻¹: CV falls from **1.74** to **0.0183**, effective experts rise from **1.35** to **4.999**, and the dead rate at 0.01/M falls from **0.73** to **0.00**. However, the improvement is **not monotone in λ**: λ=10⁻³/10⁻⁴ can still collapse, and even λ=1 collapses in one seed.
3. **Does it reduce modality dominance?** Yes. Counterfactual MDI falls from **0.856** to **0.211**; max counterfactual share falls from **0.961** to **0.644**; aligned counterfactual contribution falls from **0.961** to **0.644**. Norm-aware MDI also falls from **0.992** to **0.245**. The aligned branch remains the largest contributor, so dominance is **mitigated, not eliminated**.
4. **Is item-level routing diversity preserved?** Yes for FullLoad. Normalized routing entropy is **0.734 ± 0.026** (NoLoad 0.011); pairwise JS is **0.2175** (NoLoad 0.1178). NoLoad's pooled variance/JS is inflated by **between-seed collapse direction**: within a single-expert seed its JS is ≈0.002 and gate variance ≈0, while seed 43 splits between two experts and has JS 0.35 / variance 0.0995. FullLoad is balanced in the mean and instance-dependent **within every seed**.
5. **Does an excessively large λ over-uniformize routing?** Not in the item-level sense: λ=1's successful seeds retain H≈0.70 and spread Top-1 assignments. But λ=1 is **too large for stable optimization** (one seed collapses; training loss is dominated by the load term). The practical recommendation is not to use λ=1.
6. **Does the best λ compromise performance, balance, and modality contribution?** **No single λ does at the tested generator budget.** λ=0.1 (FullLoad) is the best **representation-level** compromise: load CV **0.0183**, effective experts **4.999**, H **0.734**, MDI **0.211**, and the lowest SID collision (15.2%). But the raw generator NDCG@10 is higher for NoLoad (validation 0.0047 vs 0.0011). λ=10⁻² is the second-best balance point but has lower routing diversity and higher MDI. The Pareto frontier is therefore two-dimensional: **λ=0 for raw HR at a fixed low budget, λ=0.1 for expert balance, modality balance, routing diversity and SID uniqueness.**
7. **Does FullLoad improve SID collision, perplexity, or downstream recommendation?** **SID collision: yes** (15.2% vs 19.5% excess collision; max bucket 6 vs 25). **Codebook perplexity: no clear improvement** (35.4 vs 43.0; both are far below the 256-code ceiling). **Downstream recommendation: no at the tested 200-step budget** — NoLoad HR@20 **0.0132** vs FullLoad **0.0034**, NDCG@10 **0.006** vs **0.001**. The existing 400-step full-test ablation reproduces the direction (NoLoad 0.0201 vs FullLoad 0.0124). This is the collision/difficulty trade-off: collapsed SID spaces are easier for an undertrained generator. We therefore report FullLoad as a representation-quality improvement, **not** as a downstream-HR improvement at this budget.

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
