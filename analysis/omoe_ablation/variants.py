"""The 19 requested OMoE loss/structure variants."""
from __future__ import annotations

from ablation_omoe import VariantConfig


def build_variants() -> list[VariantConfig]:
    v: list[VariantConfig] = []

    # ---------------- loss ablation ----------------
    v.append(VariantConfig(
        name="01_full_omoe", group="loss",
        notes="Full OMoE-v2: rec + ortho + load + dup + var",
    ))
    v.append(VariantConfig(
        name="02_wo_ortho", group="loss", w_ortho=0.0,
        notes="remove expert-output orthogonality",
    ))
    v.append(VariantConfig(
        name="03_wo_load", group="loss", w_load=0.0,
        notes="remove load balancing",
    ))
    v.append(VariantConfig(
        name="04_wo_dup", group="loss", w_dup=0.0,
        notes="remove near-duplicate repulsion",
    ))
    v.append(VariantConfig(
        name="05_wo_var", group="loss", w_var=0.0,
        notes="remove variance matching",
    ))
    v.append(VariantConfig(
        name="06_wo_rec", group="loss", w_rec=0.0,
        notes="remove reconstruction loss",
    ))
    v.append(VariantConfig(
        name="07_wo_all_aux", group="loss", w_ortho=0.0, w_load=0.0, w_dup=0.0, w_var=0.0,
        notes="only L_rec (all auxiliary regularizers removed)",
    ))
    v.append(VariantConfig(
        name="08_only_rec", group="loss", w_ortho=0.0, w_load=0.0, w_dup=0.0, w_var=0.0,
        notes="only L_rec (identical to 07 by definition)",
    ))
    v.append(VariantConfig(
        name="09_only_rec_ortho", group="loss", w_load=0.0, w_dup=0.0, w_var=0.0,
        notes="rec + ortho only",
    ))
    v.append(VariantConfig(
        name="10_only_rec_load", group="loss", w_ortho=0.0, w_dup=0.0, w_var=0.0,
        notes="rec + load only",
    ))

    # ---------------- structure ablation ----------------
    v.append(VariantConfig(
        name="11_no_direct", group="structure", use_direct=False,
        notes="remove LN(W_o e) direct projection; fused branch = expert mixture",
    ))
    v.append(VariantConfig(
        name="12_no_experts", group="structure", num_modality_experts=0, num_shared_experts=0,
        notes="remove expert mixture; fused branch = direct projection",
    ))
    v.append(VariantConfig(
        name="13_no_residual", group="structure", residual=0.0,
        notes="remove u_i anchor residual; z = gamma * s(f)",
    ))
    v.append(VariantConfig(
        name="14_static_gate", group="structure", gate_type="static",
        notes="global static gate instead of instance-dependent gate",
    ))
    v.append(VariantConfig(
        name="15_uniform_gate", group="structure", gate_type="uniform",
        notes="uniform gate (1/E), no gate parameters",
    ))
    v.append(VariantConfig(
        name="16_shared_only", group="structure", num_modality_experts=0, num_shared_experts=2,
        notes="only shared experts on concat",
    ))
    v.append(VariantConfig(
        name="17_modality_only", group="structure", num_modality_experts=3, num_shared_experts=0,
        notes="only modality-specific experts",
    ))
    v.append(VariantConfig(
        name="18_concat_mlp", group="structure", concat_mlp=True,
        notes="parameter-matched Concat-MLP replaces direct+mixture+gate",
    ))
    v.append(VariantConfig(
        name="19_plain_moe", group="structure", num_modality_experts=0, num_shared_experts=5,
        notes="ordinary MoE: 5 experts on concat, no shared/specific distinction",
    ))
    return v
