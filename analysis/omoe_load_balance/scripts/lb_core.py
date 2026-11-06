"""Core model, training and metric utilities for the OMoE load-balancing study.

The architecture and optimisation recipe mirror ``rq/text2emb/omoe_v2.py`` and
``analysis/omoe_ablation/ablation_omoe.py`` (3 modality experts + 2 shared
experts, instance gate, direct projection, outer CSGA residual, 60 epochs,
Adam, batch 512, smooth-L1 reconstruction).  The only experimental variable is
``lambda_load``.

Analysis mode returns every intermediate required by the protocol:
    pi        [N, M]      gate probabilities
    h         [N, M, d]   expert outputs
    direct    [N, d]      out_proj(concat)
    mixture   [N, d]      sum_m pi_m h_m
    fused     [N, d]      direct + mixture
    z         [N, d]      aligned + gamma * scale * fused
    item_ids  [N]         dataset row ids
    expert_types [M]      text/vision/aligned/shared mapping
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/data02/sht/SAM-GR"
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
CSGA = os.path.join(BASE, "analysis", "samgr_sports10", "results", "csga_aligned.npy")
COLLAB = os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy")
PROD_RQ = os.path.join(BASE, "rq_output", "Sports_and_Outdoors", "best_collision_model.pth")

EXPERT_NAMES = ["text", "image", "aligned", "shared-1", "shared-2"]
EXPERT_TYPES = ["text_specific", "vision_specific", "aligned_specific", "shared", "shared"]
GROUPS = ["text", "vision", "aligned", "shared"]
NUM_EXPERTS = len(EXPERT_NAMES)
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [42, 43, 44]
FULL_LAMBDA = 0.1
EPS = 1e-12


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    w_ortho: float = 0.05
    w_dup: float = 0.05
    w_var: float = 0.1
    target_std: float = 4.0
    gamma: float = 1.0
    grad_clip: float = 1.0
    num_shared: int = 2


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


class LBOMoE(nn.Module):
    def __init__(self, dim: int, cfg: TrainConfig, load_weight: float):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.load_weight = float(load_weight)
        self.num_modalities = 3
        self.num_shared_experts = cfg.num_shared
        self.num_experts = self.num_modalities + self.num_shared_experts
        self.concat_dim = dim * self.num_modalities

        def _align() -> nn.Sequential:
            return nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.align_text = _align()
        self.align_image = _align()
        self.align_collab = _align()

        self.expert_text = _mlp(dim, dim)
        self.expert_image = _mlp(dim, dim)
        self.expert_collab = _mlp(dim, dim)
        self.shared_experts = nn.ModuleList(
            [_mlp(self.concat_dim, dim) for _ in range(self.num_shared_experts)]
        )

        gh = max(128, self.concat_dim // 32)
        self.gate = nn.Sequential(
            nn.Linear(self.concat_dim, gh), nn.ReLU(), nn.Linear(gh, self.num_experts)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.out_proj = nn.Sequential(nn.Linear(self.concat_dim, dim), nn.LayerNorm(dim))
        dec_hidden = max(256, dim // 4)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dec_hidden), nn.GELU(), nn.Linear(dec_hidden, self.concat_dim)
        )

    def parts(self, e_t: torch.Tensor, e_i: torch.Tensor, e_c: torch.Tensor) -> dict:
        feats = [self.align_text(e_t), self.align_image(e_i), self.align_collab(e_c)]
        concat = torch.cat(feats, dim=-1)
        experts = [
            self.expert_text(feats[0]),
            self.expert_image(feats[1]),
            self.expert_collab(feats[2]),
        ]
        for shared in self.shared_experts:
            experts.append(shared(concat))
        h = torch.stack(experts, dim=1)
        pi = F.softmax(self.gate(concat), dim=-1)
        mixture = (pi.unsqueeze(-1) * h).sum(dim=1)
        direct = self.out_proj(concat)
        fused = direct + mixture
        return {
            "feats": feats,
            "concat": concat,
            "h": h,
            "pi": pi,
            "direct": direct,
            "mixture": mixture,
            "fused": fused,
            "aligned": e_c,
        }

    def forward(self, e_t, e_i, e_c, analysis: bool = False):
        p = self.parts(e_t, e_i, e_c)
        if analysis:
            return p
        fused = p["fused"]
        if not self.training:
            return fused, None

        rec = F.smooth_l1_loss(self.decoder(fused), p["concat"].detach())
        experts = [p["h"][:, k] for k in range(self.num_experts)]

        ortho = fused.new_zeros(())
        pairs = 0
        for a in range(self.num_experts):
            for b in range(a + 1, self.num_experts):
                ortho = ortho + F.cosine_similarity(experts[a], experts[b], dim=-1).abs().mean()
                pairs += 1
        ortho = ortho / max(1, pairs)

        mean_pi = p["pi"].mean(dim=0)
        load = self.num_experts * (mean_pi**2).sum()

        b = fused.shape[0]
        if b > 1:
            perm = torch.randperm(b, device=fused.device)
            pair_cos = F.cosine_similarity(fused, fused[perm], dim=-1)
            dup = (F.relu(pair_cos - 0.3) ** 2).mean()
        else:
            dup = fused.new_zeros(())

        mean_feat = torch.stack(p["feats"], dim=0).mean(dim=0).detach()
        var = F.mse_loss(fused.std(dim=0), mean_feat.std(dim=0)) if b > 1 else fused.new_zeros(())

        total = (
            rec
            + self.cfg.w_ortho * ortho
            + self.load_weight * load
            + self.cfg.w_dup * dup
            + self.cfg.w_var * var
        )

        with torch.no_grad():
            pi = p["pi"].detach()
            ent = -(pi * (pi + 1e-9).log()).sum(-1) / math.log(self.num_experts)
            self._batch = {
                "pi_sum": pi.sum(0),
                "pi_sq_sum": (pi**2).sum(0),
                "ent_sum": ent.sum(),
                "ent_sq_sum": (ent**2).sum(),
                "top1": torch.bincount(pi.argmax(1), minlength=self.num_experts),
                "n": pi.shape[0],
                "rec": float(rec.detach()),
                "ortho": float(ortho.detach()),
                "load": float(load.detach()),
                "dup": float(dup.detach()),
                "var": float(var.detach()),
                "total": float(total.detach()),
            }
        return fused, total


def load_emb(name: str) -> np.ndarray:
    return np.load(os.path.join(DATA, f"Sports_and_Outdoors.emb-Qwen3VL-{name}.npy")).astype(np.float32)


def load_csga() -> np.ndarray:
    return np.load(CSGA).astype(np.float32)


def load_items() -> dict:
    return json.load(open(os.path.join(DATA, "Sports_and_Outdoors.item.json")))


def load_inter(split: str):
    rows = []
    with open(os.path.join(DATA, f"Sports_and_Outdoors.{split}.inter")) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            hist = [int(x) for x in p[1].split()] if p[1].strip() else []
            rows.append((int(p[0]), hist, int(p[2])))
    return rows


def build_metadata(n: int) -> dict:
    items = load_items()
    titles, cats, has_img, text_len = [], [], [], []
    for i in range(n):
        it = items[str(i)]
        t = str(it.get("title", ""))
        d = str(it.get("description", ""))
        c = str(it.get("categories", ""))
        titles.append(t)
        parts = [x.strip() for x in c.split(",") if x.strip()]
        cats.append(parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown"))
        ip = it.get("image_paths", "")
        has_img.append(bool(ip) and ip not in ("[]", "None"))
        text_len.append(len(t) + len(d))

    pop = np.zeros(n, dtype=np.int64)
    for _, hist, tgt in load_inter("train"):
        for j in hist:
            pop[j] += 1
        pop[tgt] += 1

    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    tn = x_t / (np.linalg.norm(x_t, axis=1, keepdims=True) + 1e-9)
    inn = x_i / (np.linalg.norm(x_i, axis=1, keepdims=True) + 1e-9)
    consistency = (tn * inn).sum(1)

    return {
        "titles": np.asarray(titles, dtype=object),
        "cat2": np.asarray(cats, dtype=object),
        "has_img": np.asarray(has_img),
        "text_len": np.asarray(text_len, dtype=np.int64),
        "pop": pop,
        "consistency": consistency,
        "item_ids": np.arange(n, dtype=np.int64),
    }


def load_all():
    x_t = load_emb("text_only")
    x_i = load_emb("image_only")
    x_c = load_csga()
    meta = build_metadata(len(x_t))
    return x_t, x_i, x_c, meta


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_base_state(dim: int, cfg: TrainConfig, seed: int) -> dict:
    set_seed(seed)
    model = LBOMoE(dim, cfg, load_weight=0.0)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _batch_dead(pi_mean: np.ndarray, thr: float) -> float:
    return float((pi_mean < thr).mean())


def train_lb(x_t, x_i, x_c, lam: float, seed: int, cfg: TrainConfig, device: str,
             base_state: dict | None = None, probe_idx: np.ndarray | None = None,
             verbose: bool = True):
    set_seed(seed)
    model = LBOMoE(x_t.shape[1], cfg, load_weight=lam).to(device)
    if base_state is not None:
        model.load_state_dict(base_state)
    # Keep the post-initialisation RNG state: all lambdas consume the same
    # number of draws at construction, hence share batch order; for seed 42 and
    # lambda=0.1 this also reproduces the production omoe_v2 trajectory.

    t = torch.from_numpy(x_t - x_t.mean(0, keepdims=True)).to(device)
    i = torch.from_numpy(x_i - x_i.mean(0, keepdims=True)).to(device)
    c = torch.from_numpy(x_c - x_c.mean(0, keepdims=True)).to(device)
    n = t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        sums = {
            "pi_sum": torch.zeros(model.num_experts, device=device),
            "pi_sq_sum": torch.zeros(model.num_experts, device=device),
            "ent_sum": 0.0, "ent_sq_sum": 0.0,
            "top1": torch.zeros(model.num_experts, dtype=torch.long, device=device),
            "n": 0,
            "rec": 0.0, "ortho": 0.0, "load": 0.0, "dup": 0.0, "var": 0.0, "total": 0.0,
        }
        batch_cv, batch_dead = [], []
        for st in range(0, n, cfg.batch_size):
            b = perm[st:st + cfg.batch_size]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            mb = model._batch
            for k in ("pi_sum", "pi_sq_sum"):
                sums[k] += mb[k]
            for k in ("ent_sum", "ent_sq_sum"):
                sums[k] += float(mb[k])
            sums["top1"] += mb["top1"]
            sums["n"] += int(mb["n"])
            for k in ("rec", "ortho", "load", "dup", "var", "total"):
                sums[k] += mb[k] * int(mb["n"])
            mpi = (mb["pi_sum"] / max(int(mb["n"]), 1)).cpu().numpy()
            if mpi.mean() > 0:
                batch_cv.append(float(mpi.std() / (mpi.mean() + EPS)))
            batch_dead.append(_batch_dead(mpi, 0.01 / model.num_experts))
        nn_ = sums["n"]
        mean_pi = (sums["pi_sum"] / nn_).cpu().numpy()
        var_pi = sums["pi_sq_sum"].cpu().numpy() / nn_ - mean_pi**2
        ent_mean = sums["ent_sum"] / nn_
        ent_var = max(sums["ent_sq_sum"] / nn_ - ent_mean**2, 0.0)
        top1 = (sums["top1"] / nn_).cpu().numpy()
        load_cv = float(mean_pi.std() / (mean_pi.mean() + EPS))
        eff = float(np.exp(-(mean_pi * np.log(mean_pi + EPS)).sum()))
        row = {
            "epoch": ep,
            "loss": sums["total"] / nn_,
            "rec": sums["rec"] / nn_,
            "ortho": sums["ortho"] / nn_,
            "load": sums["load"] / nn_,
            "dup": sums["dup"] / nn_,
            "var": sums["var"] / nn_,
            "mean_pi": mean_pi.tolist(),
            "load_cv": load_cv,
            "effective_experts": eff,
            "mean_item_entropy": float(ent_mean),
            "std_item_entropy": float(math.sqrt(ent_var)),
            "top1_freq": top1.tolist(),
            "dead_rate_0.01M": float((mean_pi < 0.01 / model.num_experts).mean()),
            "batch_load_cv_mean": float(np.mean(batch_cv)) if batch_cv else None,
            "batch_dead_rate_mean": float(np.mean(batch_dead)),
            "routing_var_mean": float(var_pi.mean()),
            "routing_var_total": float(var_pi.sum()),
        }
        if probe_idx is not None and (ep % 5 == 0 or ep == cfg.epochs - 1):
            model.eval()
            with torch.no_grad():
                p = model(t[probe_idx], i[probe_idx], c[probe_idx], analysis=True)
                row["probe_js_mean"] = float(mean_pairwise_js(p["pi"].cpu().numpy()))
            model.train()
        history.append(row)
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(
                f"[lb={lam:g} seed={seed}] ep {ep:02d} loss={row['loss']:.4f} "
                f"cv={load_cv:.4f} eff={eff:.3f} H={row['mean_item_entropy']:.3f} "
                f"pi={np.round(mean_pi, 3)}", flush=True,
            )
    model.eval()
    return model, {"history": history, "train_time_sec": time.time() - t0,
                   "lambda_load": lam, "seed": seed, "epochs": cfg.epochs}


@torch.no_grad()
def analyze(model, x_t, x_i, x_c, cfg: TrainConfig, batch_size: int = 512,
            base_scale: float | None = None):
    device = next(model.parameters()).device
    n, dim = x_t.shape
    means = [x_t.mean(0), x_i.mean(0), x_c.mean(0)]
    t = torch.from_numpy(x_t - means[0]).to(device)
    i = torch.from_numpy(x_i - means[1]).to(device)
    c = torch.from_numpy(x_c - means[2]).to(device)

    pi = np.empty((n, model.num_experts), dtype=np.float32)
    hn = np.empty((n, model.num_experts), dtype=np.float32)
    h = np.empty((n, model.num_experts, dim), dtype=np.float16)
    direct = np.empty((n, dim), dtype=np.float16)
    mixture = np.empty((n, dim), dtype=np.float16)
    fused = np.empty((n, dim), dtype=np.float16)
    z = np.empty((n, dim), dtype=np.float32)
    fused_f32 = np.empty((n, dim), dtype=np.float32)

    for st in range(0, n, batch_size):
        en = min(st + batch_size, n)
        p = model(t[st:en], i[st:en], c[st:en], analysis=True)
        pi[st:en] = p["pi"].cpu().numpy()
        hh = p["h"].cpu().numpy()
        h[st:en] = hh.astype(np.float16)
        hn[st:en] = np.linalg.norm(hh, axis=-1)
        direct[st:en] = p["direct"].cpu().numpy().astype(np.float16)
        mixture[st:en] = p["mixture"].cpu().numpy().astype(np.float16)
        fused[st:en] = p["fused"].cpu().numpy().astype(np.float16)
        fused_f32[st:en] = p["fused"].cpu().numpy()
        z[st:en] = p["aligned"].cpu().numpy()

    if base_scale is None:
        cur = float(fused_f32.std())
        base_scale = cfg.target_std / cur if cur > 1e-8 else 1.0
    z = z + cfg.gamma * base_scale * fused_f32
    return {
        "pi": pi, "h": h, "h_norms": hn, "direct": direct, "mixture": mixture,
        "fused": fused, "z": z, "aligned": x_c.astype(np.float32),
        "item_ids": np.arange(n, dtype=np.int64),
        "expert_types": np.asarray(EXPERT_TYPES),
        "scale": float(base_scale),
        "means": np.stack(means),
    }


@torch.no_grad()
def conditional_z(model, x_t, x_i, x_c, means, scale: float, cfg: TrainConfig,
                  replacement: str = "mean", batch_size: int = 512):
    device = next(model.parameters()).device
    n, dim = x_t.shape
    if replacement == "mean":
        xt, xi, xc = means[0], means[1], means[2]
    else:
        xt, xi, xc = np.zeros(dim, np.float32), np.zeros(dim, np.float32), np.zeros(dim, np.float32)

    def _run(a, b, c_):
        out = np.empty((n, dim), dtype=np.float32)
        ta = torch.from_numpy(a - means[0]).to(device)
        ib = torch.from_numpy(b - means[1]).to(device)
        cc = torch.from_numpy(c_ - means[2]).to(device)
        for st in range(0, n, batch_size):
            en = min(st + batch_size, n)
            p = model(ta[st:en], ib[st:en], cc[st:en], analysis=True)
            out[st:en] = (p["aligned"].cpu().numpy()
                          + cfg.gamma * scale * p["fused"].cpu().numpy())
        return out

    z_full = _run(x_t, x_i, x_c)
    z_nt = _run(np.repeat(xt[None], n, 0), x_i, x_c)
    z_nv = _run(x_t, np.repeat(xi[None], n, 0), x_c)
    z_na = _run(x_t, x_i, np.repeat(xc[None], n, 0))
    deltas = np.stack([
        1.0 - cosine_rows(z_full, z_nt),
        1.0 - cosine_rows(z_full, z_nv),
        1.0 - cosine_rows(z_full, z_na),
    ], axis=1)
    q = deltas / (deltas.sum(1, keepdims=True) + 1e-9)
    return {"z_full": z_full, "z_no_text": z_nt, "z_no_vision": z_nv,
            "z_no_aligned": z_na, "deltas": deltas, "q": q}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return (a * b).sum(1) / (na * nb)


def mean_pairwise_js(pi: np.ndarray, max_pairs: int = 10000, seed: int = 0) -> float:
    n = len(pi)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    p, q = pi[i], pi[j]
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(np.clip(p, EPS, None) / np.clip(m, EPS, None))).sum(1)
    kl_qm = (q * np.log(np.clip(q, EPS, None) / np.clip(m, EPS, None))).sum(1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(np.clip(a, EPS, None) / np.clip(b, EPS, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def routing_metrics(pi: np.ndarray) -> dict:
    n, m = pi.shape
    mean_pi = pi.mean(0)
    std_pi = pi.std(0)
    top1 = np.bincount(pi.argmax(1), minlength=m) / n
    load_cv = float(mean_pi.std() / (mean_pi.mean() + EPS))
    eff = float(np.exp(-(mean_pi * np.log(mean_pi + EPS)).sum()))
    ent = -(pi * np.log(pi + EPS)).sum(1) / math.log(m)
    var_pi = pi.var(0)
    return {
        "mean_pi": mean_pi.tolist(),
        "std_pi": std_pi.tolist(),
        "top1_freq": top1.tolist(),
        "load_cv": load_cv,
        "effective_experts": eff,
        "dead_rate_0.01M": float((mean_pi < 0.01 / m).mean()),
        "dead_rate_1e-3": float((mean_pi < 1e-3).mean()),
        "dead_experts_0.01M": [EXPERT_NAMES[k] for k in range(m) if mean_pi[k] < 0.01 / m],
        "dead_experts_1e-3": [EXPERT_NAMES[k] for k in range(m) if mean_pi[k] < 1e-3],
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "entropy_p10": float(np.quantile(ent, 0.10)),
        "entropy_p50": float(np.quantile(ent, 0.50)),
        "entropy_p90": float(np.quantile(ent, 0.90)),
        "routing_js_mean": mean_pairwise_js(pi),
        "routing_var_per_expert": var_pi.tolist(),
        "routing_var_mean": float(var_pi.mean()),
        "routing_var_total": float(var_pi.sum()),
        "entropy_per_item": ent.astype(np.float32),
    }


def group_mass(pi: np.ndarray) -> dict:
    idx = {
        "text": [0], "vision": [1], "aligned": [2],
        "shared": [k for k, t in enumerate(EXPERT_TYPES) if t == "shared"],
    }
    out = {}
    for g, ids in idx.items():
        w = pi[:, ids].sum(1)
        out[g] = {
            "mean": float(w.mean()), "std": float(w.std()),
            "p10": float(np.quantile(w, 0.10)), "p50": float(np.quantile(w, 0.50)),
            "p90": float(np.quantile(w, 0.90)), "values": w.astype(np.float32),
        }
    return out


def norm_contribution(pi: np.ndarray, h_norms: np.ndarray) -> dict:
    r = pi * h_norms
    idx = {
        "text": [0], "vision": [1], "aligned": [2],
        "shared": [k for k, t in enumerate(EXPERT_TYPES) if t == "shared"],
    }
    R = {g: r[:, ids].sum(1) for g, ids in idx.items()}
    total = sum(R.values()) + 1e-9
    R_norm = {g: R[g] / total for g in R}
    spec = np.stack([R["text"], R["vision"], R["aligned"]], axis=1)
    p = spec / (spec.sum(1, keepdims=True) + 1e-9)
    ent = -(p * np.log(p + 1e-9)).sum(1) / math.log(3)
    mdi = 1.0 - ent
    max_share = p.max(1)
    return {
        "R": R, "R_norm": R_norm,
        "norm_share_mean": {g: float(R_norm[g].mean()) for g in R_norm},
        "norm_share_std": {g: float(R_norm[g].std()) for g in R_norm},
        "specific_p": p.astype(np.float32),
        "norm_mdi": mdi.astype(np.float32),
        "norm_entropy": ent.astype(np.float32),
        "norm_max_share": max_share.astype(np.float32),
    }


def counterfactual_metrics(q: np.ndarray, deltas: np.ndarray) -> dict:
    p = q
    ent = -(p * np.log(p + 1e-9)).sum(1) / math.log(3)
    mdi = 1.0 - ent
    max_share = p.max(1)
    return {
        "q_mean": p.mean(0).tolist(),
        "q_std": p.std(0).tolist(),
        "q_p10": np.quantile(p, 0.10, axis=0).tolist(),
        "q_p50": np.quantile(p, 0.50, axis=0).tolist(),
        "q_p90": np.quantile(p, 0.90, axis=0).tolist(),
        "delta_mean": deltas.mean(0).tolist(),
        "cf_entropy_mean": float(ent.mean()),
        "cf_entropy_std": float(ent.std()),
        "cf_mdi_mean": float(mdi.mean()),
        "cf_mdi_std": float(mdi.std()),
        "cf_max_share_mean": float(max_share.mean()),
        "cf_max_share_std": float(max_share.std()),
        "cf_mdi_per_item": mdi.astype(np.float32),
        "cf_max_share_per_item": max_share.astype(np.float32),
        "cf_entropy_per_item": ent.astype(np.float32),
        "q_per_item": p.astype(np.float32),
    }


def save_analysis(path: str, a: dict, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        pi=a["pi"], h=a["h"], h_norms=a["h_norms"], direct=a["direct"],
        mixture=a["mixture"], fused=a["fused"], z=a["z"], aligned=a["aligned"],
        item_ids=a["item_ids"], expert_types=a["expert_types"],
        scale=np.asarray([a["scale"]], dtype=np.float32),
    )
    with open(path.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def group_labels(meta: dict) -> dict:
    n = len(meta["pop"])
    pop = meta["pop"]
    q50, q80 = np.quantile(pop, [0.50, 0.80])
    pop_lab = np.where(pop >= q80, "head", np.where(pop >= q50, "mid", "tail"))

    def _tertiles(x, low, mid, high):
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(x, kind="mergesort")] = np.arange(n)
        pct = rank / max(n - 1, 1)
        return np.where(pct < 1 / 3, low, np.where(pct < 2 / 3, mid, high))

    co = meta["consistency"]
    co_lab = _tertiles(co, "low", "mid", "high")
    tl_lab = _tertiles(meta["text_len"], "short", "mid", "long")
    img_lab = np.where(meta["has_img"], "has_image", "missing_image")
    from collections import Counter
    cnt = Counter(meta["cat2"])
    top = [c for c, _ in cnt.most_common(6)]
    cat_lab = np.array([c if c in top else "Other" for c in meta["cat2"]], dtype=object)
    return {
        "popularity": (pop_lab, ["tail", "mid", "head"]),
        "consistency": (co_lab, ["low", "mid", "high"]),
        "image": (img_lab, ["has_image", "missing_image"]),
        "text_length": (tl_lab, ["short", "mid", "long"]),
        "category": (cat_lab, top + ["Other"]),
    }


def group_metrics(pi, q, h_norms, meta) -> list[dict]:
    labels = group_labels(meta)
    rows = []
    gm = group_mass(pi)
    nc = norm_contribution(pi, h_norms)
    for dim, (lab, order) in labels.items():
        for g in order:
            m = lab == g
            if m.sum() == 0:
                continue
            rows.append({
                "dimension": dim, "group": g, "n": int(m.sum()),
                "mass_text": float(gm["text"]["values"][m].mean()),
                "mass_vision": float(gm["vision"]["values"][m].mean()),
                "mass_aligned": float(gm["aligned"]["values"][m].mean()),
                "mass_shared": float(gm["shared"]["values"][m].mean()),
                "norm_text": float(nc["R_norm"]["text"][m].mean()),
                "norm_vision": float(nc["R_norm"]["vision"][m].mean()),
                "norm_aligned": float(nc["R_norm"]["aligned"][m].mean()),
                "norm_shared": float(nc["R_norm"]["shared"][m].mean()),
                "q_text": float(q[m, 0].mean()),
                "q_vision": float(q[m, 1].mean()),
                "q_aligned": float(q[m, 2].mean()),
                "cf_mdi": float(
                    (1.0 + (q[m] * np.log(q[m] + 1e-9)).sum(1) / math.log(3)).mean()
                ),
            })
    return rows
