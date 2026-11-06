"""Full-catalog CKA + neighbour check for Beauty13 / Toys10.

Uses the full direct-multimodal npy produced by
run_full_multimodal_beauty_toys.sh, plus the existing fused and SASRec collab.
No training, no plotting.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

B = "/data02/sht/SAM-GR"
OUT = os.path.join(B, "analysis", "samgr_sports10", "results")
os.makedirs(OUT, exist_ok=True)
K = 10
CKA_N = 1500
SEED = 0

CATS = {
    "Beauty13": "Beauty_and_Personal_Care",
    "Toys10": "Toys_and_Games",
}


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ Kx @ H, H @ Ky @ H
    return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))


def topk_idx(x, k=K, chunk=2048):
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
            n = xt.shape[0]
            idx = np.empty((n, k), dtype=np.int64)
            for st in range(0, n, chunk):
                en = min(st + chunk, n)
                sim = xt[st:en] @ xt.T
                rows = torch.arange(en - st, device=dev)
                sim[rows, torch.arange(st, en, device=dev)] = -2.0
                _, ix = torch.topk(sim, k, dim=1)
                idx[st:en] = ix.cpu().numpy()
            del xt
            torch.cuda.empty_cache()
            return idx
    except Exception as e:
        print("[warn] GPU topk failed, CPU fallback:", e, flush=True)
    n = len(x)
    idx = np.empty((n, k), dtype=np.int64)
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sim = x[st:en] @ x.T
        rows = np.arange(en - st)
        sim[rows, np.arange(st, en)] = -2.0
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[st:en] = np.take_along_axis(part, order, axis=1)
    return idx


def overlap(a, b, k=K):
    return np.array([len(np.intersect1d(a[i, :k], b[i, :k])) / k for i in range(len(a))],
                    dtype=np.float64)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for label, cat in CATS.items():
        root = os.path.join(B, "data", "Amazon", cat)
        mm_path = os.path.join(root, f"{cat}.emb-Qwen3VL-multimodal.npy")
        fused_path = os.path.join(root, f"{cat}.emb-Qwen3VL-csga_moe_collab_v2.npy")
        collab_path = os.path.join(B, "sasrec_models", cat, "sasrec_item_emb.npy")
        print(f"=== {label} ===", flush=True)
        mm = l2n(np.load(mm_path).astype(np.float32))
        fused = l2n(np.load(fused_path).astype(np.float32))
        collab = l2n(np.load(collab_path).astype(np.float32))
        n = len(mm)
        assert fused.shape[0] == n == collab.shape[0]
        idx = rng.choice(n, CKA_N, replace=False)
        cka_fm = cka(fused[idx], mm[idx])
        cka_mc = cka(mm[idx], collab[idx])
        cka_fc = cka(fused[idx], collab[idx])
        print(f"  CKA fused-mm={cka_fm:.4f} mm-collab={cka_mc:.4f} fused-collab={cka_fc:.4f}", flush=True)

        nb_m = topk_idx(mm)
        nb_f = topk_idx(fused)
        nb_c = topk_idx(collab)
        ov_s = overlap(nb_f, nb_m).mean()
        ov_c = overlap(nb_f, nb_c).mean()
        base_c = overlap(nb_m, nb_c).mean()
        rnd_c = []
        for seed in range(3):
            rr = np.random.default_rng(seed)
            rnb = np.stack([rr.choice(np.delete(np.arange(n), i), K, replace=False) for i in range(n)])
            rnd_c.append(overlap(rnb, nb_c).mean())
        random_c = float(np.mean(rnd_c))
        row = {
            "dataset": label,
            "n": n,
            "protocol": "full-catalog",
            "cka_fused_multimodal": cka_fm,
            "cka_multimodal_collab": cka_mc,
            "cka_fused_collab": cka_fc,
            "delta_cka_pct": (cka_fc - cka_mc) / cka_mc * 100,
            "semantic_retention": ov_s,
            "fused_collab_overlap": ov_c,
            "multimodal_collab_overlap": base_c,
            "collab_lift": ov_c / base_c,
            "random_collab_overlap": random_c,
            "mm_path": mm_path,
            "fused_path": fused_path,
            "collab_path": collab_path,
        }
        rows.append(row)
        print("  " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, ensure_ascii=False), flush=True)

    csv_path = os.path.join(OUT, "beauty_toys_cka_neighbor_full.csv")
    json_path = os.path.join(OUT, "beauty_toys_cka_neighbor_full.json")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    md_path = os.path.join(OUT, "beauty_toys_cka_neighbor_full.md")
    with open(md_path, "w") as f:
        f.write("# Beauty13 / Toys10: full-catalog CKA + neighbour check\n\n")
        f.write("| dataset | n | CKA(fused,mm) | CKA(mm,collab) | CKA(fused,collab) | ΔCKA | sem retention | collab lift |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['n']} | {r['cka_fused_multimodal']:.4f} | "
                    f"{r['cka_multimodal_collab']:.4f} | {r['cka_fused_collab']:.4f} | "
                    f"{r['delta_cka_pct']:+.1f}% | {r['semantic_retention']:.4f} | "
                    f"{r['collab_lift']:.3f}× |\n")
        f.write("\nProtocol: full-catalog CKA sample 1500 (seed 0); neighbour k=10 over the full catalog.\n")
        f.write("Fused: `{Category}.emb-Qwen3VL-csga_moe_collab_v2.npy`; multimodal: `...-multimodal.npy`; "
                "collab: `sasrec_models/{Category}/sasrec_item_emb.npy`.\n")
    print("saved", csv_path, json_path, md_path, flush=True)


if __name__ == "__main__":
    main()
