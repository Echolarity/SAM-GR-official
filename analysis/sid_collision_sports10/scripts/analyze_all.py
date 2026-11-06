import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = "/data02/sht/SAM-GR"
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
OUT = os.path.join(BASE, "analysis", "sid_collision_sports10")
STAGES = os.path.join(OUT, "results", "stages")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TOK_RE = re.compile(r"<a_\d+><b_\d+><c_\d+><d_\d+>")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_map(path):
    idx = load_json(path)
    return {int(k): tuple(v) for k, v in idx.items()}


def stats_from_codes(codes, item_ids=None):
    keys = [tuple(int(x) for x in row) for row in codes]
    counts = collections.Counter(keys)
    n = len(keys)
    u = len(counts)
    hist = collections.Counter(counts.values())
    collided = sum(v for v in counts.values() if v > 1)
    pairs = sum(v * (v - 1) for v in counts.values())
    prefixes = []
    for depth in range(1, codes.shape[1] + 1):
        pk = [tuple(int(x) for x in row[:depth]) for row in codes]
        pc = collections.Counter(pk)
        pu = len(pc)
        prefixes.append({
            "depth": depth,
            "unique_prefixes": pu,
            "unique_prefix_ratio": pu / n,
            "collision_rate": (n - pu) / n,
            "collided_item_ratio": sum(v for v in pc.values() if v > 1) / n,
            "max_bucket": max(pc.values()),
            "mean_bucket": n / pu,
        })
    return {
        "n_items": n,
        "unique_full_sids": u,
        "excess_collision_rate": (n - u) / n,
        "collided_item_ratio": collided / n,
        "pairwise_collision_probability": pairs / (n * (n - 1)),
        "max_bucket": max(counts.values()),
        "mean_bucket": n / u,
        "bucket_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "prefix": prefixes,
    }


def load_stage_npz(name):
    p = os.path.join(STAGES, f"{name}_stages.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def build_models():
    models = {}
    for name, label in [
        ("full_samgr", "Full SAM-GR"),
        ("tiger", "TIGER"),
        ("base_300ep", "Base joint-VLM (300ep)"),
        ("csga_300ep", "CSGA-only (300ep)"),
        ("omoe_300ep", "OMoE-only (300ep)"),
        ("full300_300ep", "Full SAM-GR (300ep)"),
    ]:
        d = load_stage_npz(name)
        models[label] = {
            "stages": {s: d[s] for s in ["raw", "sinkhorn", "merged"]},
            "item_ids": np.arange(11246),
            "source": name,
        }

    main_idx = index_map(os.path.join(DATA, "Sports_and_Outdoors.index.json"))
    main_codes = np.array([[int(t.rsplit("_", 1)[1].rstrip(">")) for t in main_idx[i]] for i in range(11246)], dtype=np.int32)
    models["MMGRec (shared benchmark SID)"] = {
        "stages": {"merged": main_codes},
        "item_ids": np.arange(11246),
        "source": "shared full SID",
    }

    card_codes = np.load("/data02/sht/CARD/data/Sports_and_Outdoors/Sports_and_Outdoors_card_codes.npy")
    card_ids = np.load("/data02/sht/CARD/data/Sports_and_Outdoors/Sports_and_Outdoors_card_codes_item_ids.npy") - 1
    models["CARD"] = {
        "stages": {"merged": card_codes.astype(np.int32)},
        "item_ids": card_ids.astype(np.int64),
        "source": "CARD NU-RQ-VAE",
    }
    return models


def load_meta():
    items = load_json(os.path.join(DATA, "Sports_and_Outdoors.item.json"))
    pop = np.zeros(11246, dtype=np.int64)
    item_users = [set() for _ in range(11246)]
    with open(os.path.join(DATA, "Sports_and_Outdoors.train.inter"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            u, tgt = int(parts[0]), int(parts[2])
            pop[tgt] += 1
            item_users[tgt].add(u)
    cats = []
    for i in range(11246):
        c = items[str(i)].get("categories", "") or ""
        path = tuple(x.strip() for x in c.split(",") if x.strip())
        cats.append(path)
    return items, pop, item_users, cats


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def bucket_pairs(codes):
    groups = collections.defaultdict(list)
    for i, row in enumerate(codes):
        groups[tuple(int(x) for x in row)].append(i)
    pairs = []
    for g in groups.values():
        if len(g) < 2:
            continue
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                pairs.append((g[a], g[b]))
    return groups, pairs


def pair_stats(pairs, cats, pop, item_users, text, image):
    rows = []
    for i, j in pairs:
        ci, cj = cats[i], cats[j]
        ui, uj = item_users[i], item_users[j]
        inter = len(ui & uj)
        union = len(ui | uj)
        rows.append({
            "i": i, "j": j,
            "same_leaf": int(ci[-1:] == cj[-1:]),
            "same_level3": int(ci[:3] == cj[:3]),
            "same_level2": int(ci[:2] == cj[:2]),
            "text_cos": float(text[i] @ text[j]),
            "image_cos": float(image[i] @ image[j]),
            "pop_i": int(pop[i]), "pop_j": int(pop[j]),
            "pop_min": int(min(pop[i], pop[j])), "pop_max": int(max(pop[i], pop[j])),
            "shared_users": inter,
            "user_jaccard": inter / union if union else 0.0,
            "cooccur": int(inter > 0),
        })
    return pd.DataFrame(rows)


def random_pairs(n, size, seed=0):
    rng = np.random.default_rng(seed)
    out = set()
    while len(out) < size:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        out.add((min(i, j), max(i, j)))
    return list(out)


def matched_pairs(cats, pop, size, seed=1):
    rng = np.random.default_rng(seed)
    by_cat = collections.defaultdict(list)
    for i, c in enumerate(cats):
        by_cat[c].append(i)
    cat_groups = [v for v in by_cat.values() if len(v) > 1]
    deciles = np.digitize(pop, np.quantile(pop, np.linspace(0, 1, 11)[1:-1]))
    by_dec = collections.defaultdict(list)
    for i, d in enumerate(deciles):
        by_dec[int(d)].append(i)
    dec_groups = [v for v in by_dec.values() if len(v) > 1]
    out_cat, out_dec = set(), set()
    while len(out_cat) < size:
        g = cat_groups[int(rng.integers(0, len(cat_groups)))]
        i, j = rng.choice(g, 2, replace=False)
        out_cat.add((min(int(i), int(j)), max(int(i), int(j))))
    while len(out_dec) < size:
        g = dec_groups[int(rng.integers(0, len(dec_groups)))]
        i, j = rng.choice(g, 2, replace=False)
        out_dec.add((min(int(i), int(j)), max(int(i), int(j))))
    return list(out_cat), list(out_dec)


def summarize_pairs(df, label):
    if len(df) == 0:
        return {"label": label, "n_pairs": 0}
    return {
        "label": label,
        "n_pairs": int(len(df)),
        "same_leaf_rate": float(df["same_leaf"].mean()),
        "same_level3_rate": float(df["same_level3"].mean()),
        "same_level2_rate": float(df["same_level2"].mean()),
        "text_cos_mean": float(df["text_cos"].mean()),
        "text_cos_median": float(df["text_cos"].median()),
        "text_cos_p10": float(df["text_cos"].quantile(0.1)),
        "text_cos_p90": float(df["text_cos"].quantile(0.9)),
        "image_cos_mean": float(df["image_cos"].mean()),
        "image_cos_median": float(df["image_cos"].median()),
        "image_cos_p10": float(df["image_cos"].quantile(0.1)),
        "image_cos_p90": float(df["image_cos"].quantile(0.9)),
        "pop_mean": float(df[["pop_i", "pop_j"]].to_numpy().mean()),
        "pop_median": float(np.median(df[["pop_i", "pop_j"]].to_numpy())),
        "pop_min_mean": float(df["pop_min"].mean()),
        "cooccur_rate": float(df["cooccur"].mean()),
        "shared_users_mean": float(df["shared_users"].mean()),
        "user_jaccard_mean": float(df["user_jaccard"].mean()),
    }


def beam_metrics(beams, targets, bucket_size, ks=(1, 50)):
    n = len(beams)
    sid_hits = {k: 0 for k in ks}
    exp_hits = {k: 0.0 for k in ks}
    single_hits = {k: 0 for k in ks}
    invalid = 0
    dup_rows = 0
    ambiguous_beams = 0
    total_beams = 0
    ambig_target = 0
    top1_ambig = 0
    for b, t in zip(beams, targets):
        total_beams += len(b)
        invalid += sum(1 for x in b if x not in bucket_size)
        if len(set(b)) < len(b):
            dup_rows += 1
        ambiguous_beams += sum(1 for x in b if bucket_size.get(x, 0) > 1)
        if b and bucket_size.get(b[0], 0) > 1:
            top1_ambig += 1
        bt = bucket_size.get(t, 0)
        if bt > 1:
            ambig_target += 1
        first = None
        seen = set()
        for r, x in enumerate(b, 1):
            if x in seen:
                continue
            seen.add(x)
            if x == t:
                first = r
                break
        for k in ks:
            if first is not None and first <= k:
                sid_hits[k] += 1
                exp_hits[k] += 1.0 / bt if bt else 0.0
                if bt == 1:
                    single_hits[k] += 1
    return {
        "n_rows": n,
        "invalid_beams": invalid,
        "invalid_rate": invalid / total_beams if total_beams else 0.0,
        "duplicate_rows": dup_rows,
        "duplicate_beam_rate": dup_rows / n if n else 0.0,
        "ambiguous_beams": ambiguous_beams,
        "ambiguous_sid_prediction_rate": ambiguous_beams / total_beams if total_beams else 0.0,
        "top1_ambiguous_rate": top1_ambig / n if n else 0.0,
        "ambiguous_target_rate": ambig_target / n if n else 0.0,
        "sid_level_hr": {f"HR@{k}": sid_hits[k] / n for k in ks},
        "expected_item_level_hr_uniform": {f"HR@{k}": exp_hits[k] / n for k in ks},
        "singleton_only_item_level_hr": {f"HR@{k}": single_hits[k] / n for k in ks},
    }


def main():
    models = build_models()
    items, pop, item_users, cats = load_meta()
    text = norm(np.load(os.path.join(DATA, "Sports_and_Outdoors.emb-Qwen3VL-text_only.npy")))
    image = norm(np.load(os.path.join(DATA, "Sports_and_Outdoors.emb-Qwen3VL-image_only.npy")))

    summary_rows = []
    prefix_rows = []
    hist_rows = []
    model_stats = {}
    for label, m in models.items():
        for stage, codes in m["stages"].items():
            s = stats_from_codes(codes)
            model_stats[f"{label}|{stage}"] = s
            summary_rows.append({"model": label, "stage": stage, **{k: v for k, v in s.items()
                                                                   if k not in ("prefix", "bucket_histogram")}})
            for p in s["prefix"]:
                prefix_rows.append({"model": label, "stage": stage, **p})
            for bucket, count in s["bucket_histogram"].items():
                hist_rows.append({"model": label, "stage": stage, "bucket_size": int(bucket), "num_buckets": count})

    pd.DataFrame(summary_rows).to_csv(os.path.join(RES, "model_summary.csv"), index=False)
    pd.DataFrame(prefix_rows).to_csv(os.path.join(RES, "prefix_stats.csv"), index=False)
    pd.DataFrame(hist_rows).to_csv(os.path.join(RES, "bucket_histogram.csv"), index=False)
    with open(os.path.join(RES, "model_stats.json"), "w") as f:
        json.dump(model_stats, f, indent=2)

    rng_pairs = random_pairs(11246, 20000, seed=0)
    cat_pairs, dec_pairs = matched_pairs(cats, pop, 20000, seed=1)
    base_cat = pair_stats(rng_pairs, cats, pop, item_users, text, image)
    base_dec = pair_stats(dec_pairs, cats, pop, item_users, text, image)
    base_cat_df = pair_stats(cat_pairs, cats, pop, item_users, text, image)
    collided = {}
    collided_rows = []
    for label, m in models.items():
        codes = m["stages"]["merged"]
        groups, pairs = bucket_pairs(codes)
        df = pair_stats(pairs, cats, pop, item_users, text, image)
        collided[label] = {
            "buckets": len(groups),
            "collision_pairs": len(pairs),
            "all_pairs": summarize_pairs(df, label),
            "random_baseline": summarize_pairs(base_cat, "random"),
            "same_category_baseline": summarize_pairs(base_cat_df, "same_category"),
            "same_popularity_decile_baseline": summarize_pairs(base_dec, "same_popularity_decile"),
        }
        collided_rows.append({"model": label, **collided[label]["all_pairs"]})
    with open(os.path.join(RES, "collided_analysis.json"), "w") as f:
        json.dump(collided, f, indent=2)
    pd.DataFrame(collided_rows).to_csv(os.path.join(RES, "collided_item_summary.csv"), index=False)

    beams = {}
    full = load_json(os.path.join(BASE, "results", "qwen2.5-0.5b", "full",
                                  "Sports_and_Outdoors-sft-dsz2_csga_omoe_v2_strict", "final_result.json"))
    full_idx = index_map(os.path.join(DATA, "Sports_and_Outdoors.index.json"))
    full_bucket = collections.Counter("".join(v) for v in full_idx.values())
    beams["Full SAM-GR"] = beam_metrics(
        [[p for p in row["predict"] if TOK_RE.fullmatch(p)] for row in full],
        [row["output"] for row in full], full_bucket)

    tiger_beams = np.load(os.path.join(RES, "beams", "tiger_beams.npy"), allow_pickle=True)
    tiger_targets = np.load(os.path.join(RES, "beams", "tiger_targets.npy"), allow_pickle=True)
    tiger_idx = index_map(os.path.join(BASE, "analysis", "samgr_sports10", "results", "text_only_sid",
                                       "Sports_and_Outdoors.index.json"))
    tiger_bucket = collections.Counter("".join(v) for v in tiger_idx.values())
    beams["TIGER"] = beam_metrics([list(x) for x in tiger_beams], list(tiger_targets), tiger_bucket)

    mm_dir = "/data02/sht/MMGRec/sports10core/runs/Sports_and_Outdoors_10core"
    mm_top = np.load(os.path.join(mm_dir, "final", "test_sid_top50.npy"))
    mm_sid_ids = np.load(os.path.join(mm_dir, "sid_benchmark", "sid_ids.npy"))
    mm_prep = np.load(os.path.join(mm_dir, "prepared.npz"))
    mm_target = mm_sid_ids[mm_prep["test_target"]]
    mm_bucket = collections.Counter(int(x) for x in mm_sid_ids.tolist())
    beams["MMGRec"] = beam_metrics([list(map(int, row)) for row in mm_top],
                                   list(map(int, mm_target)), mm_bucket)

    card_codes = np.load("/data02/sht/CARD/data/Sports_and_Outdoors/Sports_and_Outdoors_card_codes.npy")
    card_valid = {tuple(int(x) for x in row) for row in card_codes}
    card_bucket = {k: 1 for k in card_valid}
    card_pred = np.load("/data02/sht/CARD/results/Sports_and_Outdoors_CARD/test_predictions.npy")
    card_tgt = np.load("/data02/sht/CARD/results/Sports_and_Outdoors_CARD/test_targets.npy")
    levels = np.arange(4)[None, None, :]
    card_pred_codes = card_pred - 1 - levels * 256
    card_tgt_codes = card_tgt - 1 - np.arange(4)[None, :] * 256
    beams["CARD"] = beam_metrics(
        [[tuple(int(x) for x in row) for row in sample] for sample in card_pred_codes],
        [tuple(int(x) for x in row) for row in card_tgt_codes], card_bucket)

    with open(os.path.join(RES, "beam_stats.json"), "w") as f:
        json.dump(beams, f, indent=2)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in
                          ("sid_level_hr", "expected_item_level_hr_uniform", "singleton_only_item_level_hr")}
                      for k, v in beams.items()}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
