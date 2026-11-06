"""Configurable OMoE-v2 for the Sports 10-core loss/structure ablation.

The reference implementation lives in ``rq/text2emb/omoe_v2.py``.  This module
keeps the same optimisation recipe (mean-centred inputs, Linear+LN alignment,
60 epochs, Adam, batch 512, smooth-L1 reconstruction, target-std scaling and an
outer residual) and exposes the switches needed by the 19 ablation variants.

Nothing here changes CSGA: callers pass the *cached* CSGA-aligned feature.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VariantConfig:
    name: str
    # losses
    w_rec: float = 1.0
    w_ortho: float = 0.05
    w_load: float = 0.1
    w_dup: float = 0.05
    w_var: float = 0.1
    # structure
    num_modality_experts: int = 3          # 3 = text/image/csga
    num_shared_experts: int = 2            # experts on the concat
    gate_type: str = "instance"            # instance | static | uniform
    use_direct: bool = True                # LN(W_o e)
    residual: float = 1.0                  # outer anchor u_i
    concat_mlp: bool = False               # parameter-matched Concat-MLP
    concat_mlp_hidden: int | None = None
    # training
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    target_std: float = 4.0
    seed: int = 42
    notes: str = ""
    group: str = ""


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


class AblOMoE(nn.Module):
    def __init__(self, dim: int, cfg: VariantConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.concat_dim = dim * 3
        # A Concat-MLP is a single branch: it must not instantiate experts/gate.
        self.num_modality_experts = 0 if cfg.concat_mlp else cfg.num_modality_experts
        self.num_shared_experts = 0 if cfg.concat_mlp else cfg.num_shared_experts
        self.num_experts = self.num_modality_experts + self.num_shared_experts

        def _align() -> nn.Sequential:
            return nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.align_text = _align()
        self.align_image = _align()
        self.align_collab = _align()

        if self.num_modality_experts > 0:
            self.expert_modality = nn.ModuleList(
                [_mlp(dim, dim) for _ in range(self.num_modality_experts)]
            )
        self.shared_experts = nn.ModuleList(
            [_mlp(self.concat_dim, dim) for _ in range(self.num_shared_experts)]
        )

        self.gate = None
        self.static_gate_logits = None
        if self.num_experts > 0 and cfg.gate_type == "instance":
            gh = max(128, self.concat_dim // 32)
            self.gate = nn.Sequential(
                nn.Linear(self.concat_dim, gh), nn.ReLU(), nn.Linear(gh, self.num_experts)
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        elif self.num_experts > 0 and cfg.gate_type == "static":
            self.static_gate_logits = nn.Parameter(torch.zeros(self.num_experts))

        self.out_proj = None
        if cfg.use_direct and not cfg.concat_mlp:
            self.out_proj = nn.Sequential(
                nn.Linear(self.concat_dim, dim), nn.LayerNorm(dim)
            )

        # A Concat-MLP replaces {direct projection + expert mixture + gate}.
        self.concat_mlp = None
        if cfg.concat_mlp:
            h = cfg.concat_mlp_hidden
            if h is None:
                h = self.matched_concat_mlp_hidden(dim, cfg)
            self.concat_mlp = nn.Sequential(
                nn.Linear(self.concat_dim, h), nn.GELU(), nn.Linear(h, dim), nn.LayerNorm(dim)
            )
            self.concat_mlp_hidden_used = int(h)

        dec_hidden = max(256, dim // 4)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dec_hidden), nn.GELU(), nn.Linear(dec_hidden, self.concat_dim)
        )

    @staticmethod
    def reference_param_count(dim: int, cfg: VariantConfig) -> int:
        """Parameter count of the corresponding non-Concat-MLP OMoE."""
        concat = dim * 3
        p = 0
        # alignment (3 x Linear+LN)
        p += 3 * (dim * dim + dim + 2 * dim)
        # modality experts
        p += cfg.num_modality_experts * (dim * dim + dim + dim * dim + dim)
        # shared experts on concat
        p += cfg.num_shared_experts * (concat * dim + dim + dim * dim + dim)
        # gate
        if cfg.num_modality_experts + cfg.num_shared_experts > 0 and cfg.gate_type == "instance":
            gh = max(128, concat // 32)
            p += concat * gh + gh + gh * (cfg.num_modality_experts + cfg.num_shared_experts) + (
                cfg.num_modality_experts + cfg.num_shared_experts
            )
        elif cfg.num_modality_experts + cfg.num_shared_experts > 0 and cfg.gate_type == "static":
            p += cfg.num_modality_experts + cfg.num_shared_experts
        # direct projection
        if cfg.use_direct:
            p += concat * dim + dim + 2 * dim
        # decoder
        dec_hidden = max(256, dim // 4)
        p += dim * dec_hidden + dec_hidden + dec_hidden * concat + concat
        return int(p)

    @classmethod
    def matched_concat_mlp_hidden(cls, dim: int, cfg: VariantConfig) -> int:
        """Choose h so that the Concat-MLP has (approximately) the same params.

        The shared alignment and decoder are present in both models, so only the
        fusion branch has to be matched.  Branch params for the reference:
        modality experts + shared experts + gate + direct projection.
        """
        concat = dim * 3
        total = cls.reference_param_count(dim, cfg)
        # subtract alignment + decoder
        align = 3 * (dim * dim + dim + 2 * dim)
        dec_hidden = max(256, dim // 4)
        decoder = dim * dec_hidden + dec_hidden + dec_hidden * concat + concat
        branch = max(total - align - decoder, 1)
        # Linear(concat,h)+bias + Linear(h,dim)+bias
        h = (branch - dim) / (concat + dim)
        return max(64, int(round(h)))

    def _aligned(self, e_t, e_i, e_c):
        return [self.align_text(e_t), self.align_image(e_i), self.align_collab(e_c)]

    def _gates(self, concat: torch.Tensor) -> torch.Tensor:
        if self.num_experts == 0:
            return None
        if self.cfg.gate_type == "uniform":
            return concat.new_full((concat.shape[0], self.num_experts), 1.0 / self.num_experts)
        if self.cfg.gate_type == "static":
            logits = self.static_gate_logits.unsqueeze(0).expand(concat.shape[0], -1)
            return F.softmax(logits, dim=-1)
        return F.softmax(self.gate(concat), dim=-1)

    def forward(self, e_t, e_i, e_c, return_diagnostics: bool = False):
        feats = self._aligned(e_t, e_i, e_c)
        concat = torch.cat(feats, dim=-1)

        experts = []
        if self.num_modality_experts > 0:
            for k in range(self.num_modality_experts):
                experts.append(self.expert_modality[k](feats[k]))
        for shared in self.shared_experts:
            experts.append(shared(concat))

        gates = self._gates(concat) if experts else None
        if experts:
            moe = torch.zeros_like(experts[0])
            for k, e in enumerate(experts):
                moe = moe + gates[:, k].unsqueeze(-1) * e
        else:
            moe = None

        direct = self.out_proj(concat) if self.out_proj is not None else None
        if self.concat_mlp is not None:
            branch = self.concat_mlp(concat)
        elif direct is not None and moe is not None:
            branch = direct + moe
        elif direct is not None:
            branch = direct
        elif moe is not None:
            branch = moe
        else:
            raise RuntimeError("variant has no fusion branch")

        if not self.training:
            return branch, None

        rec = F.smooth_l1_loss(self.decoder(branch), concat.detach()) if self.cfg.w_rec > 0 else branch.new_zeros(())
        ortho = branch.new_zeros(())
        if len(experts) >= 2 and self.cfg.w_ortho > 0:
            pairs = 0
            for a in range(len(experts)):
                for b in range(a + 1, len(experts)):
                    ortho = ortho + F.cosine_similarity(experts[a], experts[b], dim=-1).abs().mean()
                    pairs += 1
            ortho = ortho / max(1, pairs)

        load = branch.new_zeros(())
        if gates is not None and self.cfg.gate_type != "uniform" and self.cfg.w_load > 0:
            mean_gate = gates.mean(dim=0)
            load = self.num_experts * (mean_gate ** 2).sum()

        dup = branch.new_zeros(())
        if self.cfg.w_dup > 0:
            b = branch.shape[0]
            if b > 1:
                perm = torch.randperm(b, device=branch.device)
                pair_cos = F.cosine_similarity(branch, branch[perm], dim=-1)
                dup = (F.relu(pair_cos - 0.3) ** 2).mean()

        var = branch.new_zeros(())
        if self.cfg.w_var > 0:
            mean_feat = torch.stack(feats, dim=0).mean(dim=0).detach()
            if branch.shape[0] > 1:
                var = F.mse_loss(branch.std(dim=0), mean_feat.std(dim=0))

        total = (
            self.cfg.w_rec * rec
            + self.cfg.w_ortho * ortho
            + self.cfg.w_load * load
            + self.cfg.w_dup * dup
            + self.cfg.w_var * var
        )

        self._last_losses = {
            "rec": float(rec.detach().cpu().item()),
            "ortho": float(ortho.detach().cpu().item()),
            "load": float(load.detach().cpu().item()),
            "dup": float(dup.detach().cpu().item()),
            "var": float(var.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
        if return_diagnostics:
            return branch, total, {
                "experts": [e.detach() for e in experts],
                "gates": gates.detach() if gates is not None else None,
                "direct": direct.detach() if direct is not None else None,
                "moe": moe.detach() if moe is not None else None,
            }
        return branch, total


def _center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x - x.mean(axis=0, keepdims=True)


def _cos_stats(x: np.ndarray, sample_n: int = 2000) -> tuple[float, float, float]:
    s = x[: min(sample_n, len(x))]
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
    aff = s @ s.T
    np.fill_diagonal(aff, -2.0)
    nn = aff.max(axis=1)
    mean_cos = float(aff[aff > -1.5].mean())
    nn_dup = float(np.mean(nn > 0.99))
    return float(np.std(x)), mean_cos, nn_dup


@torch.no_grad()
def evaluate_variant(
    model: AblOMoE,
    t: torch.Tensor,
    i: torch.Tensor,
    c: torch.Tensor,
    aligned_np: np.ndarray,
    batch_size: int = 512,
) -> dict:
    model.eval()
    n, dim = t.shape
    E = model.num_experts
    sim_sum = np.zeros((E, E), dtype=np.float64) if E else None
    sim_cnt = 0
    gate_sum = np.zeros(E, dtype=np.float64) if E else None
    gate_sq_sum = np.zeros(E, dtype=np.float64) if E else None
    gate_cnt = 0
    ent_sum = 0.0
    batch_cv = []
    item_gate_rows = []
    direct_norm_sum = 0.0
    moe_norm_sum = 0.0
    cos_dm_sum = 0.0
    rel_no_moe_sum = 0.0
    rel_no_direct_sum = 0.0
    branch_norm_sum = 0.0
    scaled_norm_sum = 0.0
    aligned_norm_sum = 0.0
    final_norm_sum = 0.0
    cos_ab_sum = 0.0
    out = np.empty((n, dim), dtype=np.float32)
    loss_last = getattr(model, "_last_losses", {})

    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        e_t, e_i, e_c = t[st:en], i[st:en], c[st:en]
        branch, _ = model(e_t, e_i, e_c)
        # replicate inference post-processing of omoe_v2.py
        b = branch.cpu().numpy()
        cur_std = float(b.std())
        scaled = b * (model.cfg.target_std / cur_std) if cur_std > 1e-8 else b
        a = aligned_np[st:en]
        final = a + model.cfg.residual * scaled
        out[st:en] = final
        scaled_norm_sum += float(np.linalg.norm(scaled, axis=1).sum())

        if E:
            # recompute experts/gates for diagnostics (cheap)
            feats = model._aligned(e_t, e_i, e_c)
            concat = torch.cat(feats, dim=-1)
            experts = []
            if model.num_modality_experts > 0:
                for k in range(model.num_modality_experts):
                    experts.append(model.expert_modality[k](feats[k]))
            for shared in model.shared_experts:
                experts.append(shared(concat))
            gates = model._gates(concat)
            B = gates.shape[0]
            for p in range(E):
                for q in range(E):
                    sim_sum[p, q] += F.cosine_similarity(experts[p], experts[q], dim=-1).sum().item()
            sim_cnt += B
            gate_sum += gates.sum(0).cpu().numpy()
            gate_sq_sum += (gates ** 2).sum(0).cpu().numpy()
            gate_cnt += B
            ent = -(gates * (gates + 1e-9).log()).sum(-1)
            ent_sum += ent.sum().item()
            batch_cv.append(float((gates.mean(0).std() / (gates.mean(0).mean() + 1e-12)).item()))
            item_gate_rows.append(gates.cpu().numpy())

            direct = model.out_proj(concat) if model.out_proj is not None else None
            moe = torch.zeros_like(experts[0])
            for k, e in enumerate(experts):
                moe = moe + gates[:, k].unsqueeze(-1) * e
            br = direct + moe if direct is not None else moe
            bn = br.norm(dim=1)
            branch_norm_sum += bn.sum().item()
            if direct is not None:
                dn = direct.norm(dim=1)
                direct_norm_sum += dn.sum().item()
                cos_dm_sum += F.cosine_similarity(direct, moe, dim=-1).sum().item()
                rel_no_moe_sum += ((br - direct).norm(dim=1) / (bn + 1e-9)).sum().item()
            mn = moe.norm(dim=1)
            moe_norm_sum += mn.sum().item()
            if direct is not None:
                rel_no_direct_sum += ((br - moe).norm(dim=1) / (bn + 1e-9)).sum().item()
        # residual bypass diagnostics
        a_t = torch.from_numpy(a)
        s_t = torch.from_numpy(scaled)
        final_t = a_t + model.cfg.residual * s_t
        aligned_norm_sum += a_t.norm(dim=1).sum().item()
        final_norm_sum += final_t.norm(dim=1).sum().item()
        cos_ab_sum += F.cosine_similarity(a_t, s_t, dim=-1).sum().item()

    res: dict = {
        "n_items": int(n),
        "num_experts": int(E),
        "params": int(sum(p.numel() for p in model.parameters())),
        "target_std": model.cfg.target_std,
        "residual": model.cfg.residual,
        "last_losses": loss_last,
    }
    std, mean_cos, nn_dup = _cos_stats(out)
    res.update({"fused_std": std, "fused_mean_cos": mean_cos, "fused_nn_dup": nn_dup})
    res["aligned_mean_norm"] = aligned_norm_sum / n
    res["final_mean_norm"] = final_norm_sum / n
    res["cos_aligned_branch"] = cos_ab_sum / n
    res["aligned_norm_share"] = res["aligned_mean_norm"] / (res["final_mean_norm"] + 1e-9)
    res["branch_norm_share"] = (model.cfg.residual * scaled_norm_sum / n) / (
        res["final_mean_norm"] + 1e-9
    )
    if E:
        mean_gate = gate_sum / gate_cnt
        res["mean_gate"] = mean_gate.tolist()
        res["gate_entropy"] = float(-(mean_gate * np.log(mean_gate + 1e-9)).sum())
        res["gate_load_cv"] = float(mean_gate.std() / (mean_gate.mean() + 1e-12))
        res["batch_load_cv"] = float(np.mean(batch_cv))
        gates_all = np.concatenate(item_gate_rows, axis=0)
        res["mean_item_entropy"] = float(ent_sum / gate_cnt)
        item_ent = -(gates_all * np.log(gates_all + 1e-9)).sum(axis=1)
        res["std_item_entropy"] = float(item_ent.std())
        res["routing_item_var"] = float(gates_all.var(axis=0).mean())
        res["routing_total_var"] = float(gates_all.var())
        res["expert_sim"] = (sim_sum / max(1, sim_cnt)).tolist()
        off = []
        for p in range(E):
            for q in range(p + 1, E):
                off.append(abs(sim_sum[p, q] / max(1, sim_cnt)))
        res["mean_abs_expert_cos"] = float(np.mean(off)) if off else None
        res["max_abs_expert_cos"] = float(np.max(off)) if off else None
        res["mean_direct_norm"] = direct_norm_sum / n if direct_norm_sum else None
        res["mean_moe_norm"] = moe_norm_sum / n
        res["cos_direct_moe"] = cos_dm_sum / n if direct_norm_sum else None
        res["rel_diff_no_moe"] = rel_no_moe_sum / n if direct_norm_sum else None
        res["rel_diff_no_direct"] = rel_no_direct_sum / n if direct_norm_sum else None
        res["direct_share"] = (
            direct_norm_sum / (direct_norm_sum + moe_norm_sum)
            if (direct_norm_sum + moe_norm_sum) > 0 else None
        )
        res["moe_share"] = (
            moe_norm_sum / (direct_norm_sum + moe_norm_sum)
            if (direct_norm_sum + moe_norm_sum) > 0 else None
        )
    else:
        res["mean_gate"] = None
        res["gate_entropy"] = None
        res["gate_load_cv"] = None
        res["batch_load_cv"] = None
        res["mean_item_entropy"] = None
        res["std_item_entropy"] = None
        res["routing_item_var"] = None
        res["routing_total_var"] = None
        res["expert_sim"] = None
        res["mean_abs_expert_cos"] = None
        res["max_abs_expert_cos"] = None
        res["mean_direct_norm"] = None
        res["mean_moe_norm"] = None
        res["cos_direct_moe"] = None
        res["rel_diff_no_moe"] = None
        res["rel_diff_no_direct"] = None
        res["direct_share"] = None
        res["moe_share"] = None
    return res


def train_and_fuse(
    x_t: np.ndarray,
    x_i: np.ndarray,
    x_c: np.ndarray,
    cfg: VariantConfig,
    device: str = "cuda:0",
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    t = torch.from_numpy(_center(x_t)).to(device)
    i = torch.from_numpy(_center(x_i)).to(device)
    c = torch.from_numpy(_center(x_c)).to(device)
    aligned_np = np.asarray(x_c, dtype=np.float32)

    model = AblOMoE(x_t.shape[1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    n = t.shape[0]
    t0 = time.time()
    history = []
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        agg = {"rec": 0.0, "ortho": 0.0, "load": 0.0, "dup": 0.0, "var": 0.0, "total": 0.0}
        for st in range(0, n, cfg.batch_size):
            b = perm[st : st + cfg.batch_size]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in agg:
                agg[k] += getattr(model, "_last_losses", {}).get(k, 0.0) * len(b)
        for k in agg:
            agg[k] /= n
        history.append({"epoch": ep, **agg})
        total = agg["total"]
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(
                f"[{cfg.name}] ep {ep}: total={total:.4f} rec={agg['rec']:.4f} "
                f"ortho={agg['ortho']:.4f} load={agg['load']:.4f} "
                f"dup={agg['dup']:.4f} var={agg['var']:.4f}",
                flush=True,
            )
    train_time = time.time() - t0

    stats = evaluate_variant(model, t, i, c, aligned_np, batch_size=cfg.batch_size)
    stats.update(
        {
            "variant": cfg.name,
            "group": cfg.group,
            "notes": cfg.notes,
            "train_time_sec": float(train_time),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "weights": {
                "rec": cfg.w_rec,
                "ortho": cfg.w_ortho,
                "load": cfg.w_load,
                "dup": cfg.w_dup,
                "var": cfg.w_var,
            },
            "structure": {
                "num_modality_experts": cfg.num_modality_experts,
                "num_shared_experts": cfg.num_shared_experts,
                "gate_type": cfg.gate_type,
                "use_direct": cfg.use_direct,
                "residual": cfg.residual,
                "concat_mlp": cfg.concat_mlp,
                "concat_mlp_hidden": getattr(model, "concat_mlp_hidden_used", None),
            },
            "loss_history": history,
        }
    )

    # final fused feature, exactly as in omoe_fuse_v2
    model.eval()
    out = np.empty_like(aligned_np, dtype=np.float32)
    with torch.no_grad():
        for st in range(0, n, cfg.batch_size):
            en = min(st + cfg.batch_size, n)
            branch, _ = model(t[st:en], i[st:en], c[st:en])
            b = branch.cpu().numpy()
            cur = float(b.std())
            b = b * (cfg.target_std / cur) if cur > 1e-8 else b
            out[st:en] = aligned_np[st:en] + cfg.residual * b
    return out, stats
