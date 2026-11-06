import json
import os

import pandas as pd

BASE = "/data02/sht/SAM-GR"
RES = os.path.join(BASE, "analysis", "sid_collision_sports10", "results")
STATS = json.load(open(os.path.join(RES, "model_stats.json")))
BEAMS = json.load(open(os.path.join(RES, "beam_stats.json")))
COLL = json.load(open(os.path.join(RES, "collided_analysis.json")))
PREFIX = pd.read_csv(os.path.join(RES, "prefix_stats.csv"))

MODELS = [
    ("Full SAM-GR", "Full SAM-GR"),
    ("TIGER", "TIGER"),
    ("MMGRec (shared benchmark SID)", "MMGRec"),
    ("CARD", "CARD"),
    ("Base joint-VLM (300ep)", "Base joint-VLM (300ep)"),
    ("CSGA-only (300ep)", "CSGA-only (300ep)"),
    ("OMoE-only (300ep)", "OMoE-only (300ep)"),
    ("Full SAM-GR (300ep)", "Full SAM-GR (300ep)"),
]


def fmt_pct(x, nd=2):
    return f"{100*x:.{nd}f}"


def table_final():
    lines = ["| Model | Items | Unique full SIDs | Excess (N-U)/N | Collided-item ratio | Pairwise collision P | Max bucket | Mean bucket |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, name in MODELS:
        s = STATS[f"{key}|merged"]
        lines.append(f"| {name} | {s['n_items']:,} | {s['unique_full_sids']:,} | {fmt_pct(s['excess_collision_rate'])}% | "
                     f"{fmt_pct(s['collided_item_ratio'])}% | {s['pairwise_collision_probability']:.2e} | {s['max_bucket']} | {s['mean_bucket']:.3f} |")
    return "\n".join(lines)


def table_stages():
    models = ["Full SAM-GR", "TIGER", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    lines = ["| Model | Stage | Unique SIDs | Excess | Collided items | Max bucket |",
             "|---|---|---:|---:|---:|---:|"]
    for m in models:
        for stage, label in [("raw", "raw argmin"), ("sinkhorn", "after Sinkhorn"), ("merged", "after 15% merge")]:
            s = STATS[f"{m}|{stage}"]
            lines.append(f"| {m} | {label} | {s['unique_full_sids']:,} | {fmt_pct(s['excess_collision_rate'])}% | "
                         f"{fmt_pct(s['collided_item_ratio'])}% | {s['max_bucket']} |")
    return "\n".join(lines)


def table_prefix():
    lines = ["| Model | Depth | Unique prefixes | Unique-prefix ratio | Prefix collision rate | Collided-item ratio | Max prefix bucket |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name in MODELS:
        d = PREFIX[(PREFIX.model == key) & (PREFIX.stage == "merged")]
        for _, r in d.iterrows():
            lines.append(f"| {name} | {int(r.depth)} | {int(r.unique_prefixes):,} | {r.unique_prefix_ratio:.4f} | "
                         f"{fmt_pct(r.collision_rate)}% | {fmt_pct(r.collided_item_ratio)}% | {int(r.max_bucket)} |")
    return "\n".join(lines)


def table_hist():
    models = ["Full SAM-GR", "TIGER", "CARD", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 10**9, ">50")]
    lines = ["| Model | " + " | ".join(b[2] for b in bins) + " |",
             "|---|" + "---:|" * len(bins)]
    for m in models:
        h = {int(k): v for k, v in STATS[f"{m}|merged"]["bucket_histogram"].items()}
        vals = []
        for lo, hi, _ in bins:
            vals.append(sum(v for k, v in h.items() if lo <= k <= hi))
        lines.append(f"| {m} | " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def table_collided():
    models = ["Full SAM-GR", "TIGER", "Base joint-VLM (300ep)", "CSGA-only (300ep)", "OMoE-only (300ep)", "Full SAM-GR (300ep)"]
    lines = ["| Model | Pairs | Same L3 cat. | Same leaf cat. | Text cos | Image cos | Mean min-pop | Co-interaction | Random co-int. | Same-pop-decile co-int. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        a = COLL[m]["all_pairs"]
        r = COLL[m]["random_baseline"]
        d = COLL[m]["same_popularity_decile_baseline"]
        lines.append(f"| {m} | {a['n_pairs']:,} | {fmt_pct(a['same_level3_rate'])}% | {fmt_pct(a['same_leaf_rate'])}% | "
                     f"{a['text_cos_mean']:.4f} | {a['image_cos_mean']:.4f} | {a['pop_min_mean']:.2f} | "
                     f"{fmt_pct(a['cooccur_rate'])}% | {fmt_pct(r['cooccur_rate'])}% | {fmt_pct(d['cooccur_rate'])}% |")
    return "\n".join(lines)


def table_stage_collided():
    prod = json.load(open(os.path.join(RES, "collided_prod_stage_analysis.json")))
    short = json.load(open(os.path.join(RES, "collided_stage_analysis.json")))
    rows = []
    for k, v in list(prod.items()) + list(short.items()):
        if v.get("n_pairs", 0) == 0:
            continue
        if v.get("skipped"):
            continue
        rows.append((k, v))
    lines = ["| Model / stage | Pairs | Text cos | Image cos | Same L3 cat. | Co-interaction |",
             "|---|---:|---:|---:|---:|---:|"]
    for k, v in rows:
        lines.append(f"| {k} | {v['n_pairs']:,} | {v['text_cos_mean']:.4f} | {v['image_cos_mean']:.4f} | "
                     f"{fmt_pct(v['same_level3_rate'])}% | {fmt_pct(v['cooccur_rate'])}% |")
    return "\n".join(lines)


def table_beams():
    lines = ["| Model | Rows | Invalid SID rate | Duplicate-beam rate | Ambiguous SID prediction rate | Top-1 ambiguous rate | Ambiguous target rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in ["Full SAM-GR", "TIGER", "MMGRec", "CARD"]:
        b = BEAMS[m]
        lines.append(f"| {m} | {b['n_rows']:,} | {fmt_pct(b['invalid_rate'], 3)}% | {fmt_pct(b['duplicate_beam_rate'], 3)}% | "
                     f"{fmt_pct(b['ambiguous_sid_prediction_rate'])}% | {fmt_pct(b['top1_ambiguous_rate'])}% | "
                     f"{fmt_pct(b['ambiguous_target_rate'])}% |")
    return "\n".join(lines)


def table_item_level():
    lines = ["| Model | SID-level HR@50 | Expected item-level HR@50 (uniform resolver) | Singleton-only HR@50 (lower bound) | Relative drop vs SID-level |",
             "|---|---:|---:|---:|---:|"]
    for m in ["Full SAM-GR", "TIGER", "MMGRec", "CARD"]:
        b = BEAMS[m]
        sid = b["sid_level_hr"]["HR@50"]
        exp = b["expected_item_level_hr_uniform"]["HR@50"]
        single = b["singleton_only_item_level_hr"]["HR@50"]
        drop = (sid - exp) / sid * 100 if sid else 0
        lines.append(f"| {m} | {fmt_pct(sid)}% | {fmt_pct(exp)}% | {fmt_pct(single)}% | {drop:.1f}% |")
    return "\n".join(lines)


def table_perf():
    lines = ["| Model | Training budget | HR@1 | HR@10 | HR@50 | NDCG@10 | NDCG@50 | Evaluation level |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    lines.append("| Full SAM-GR | 2 epochs SFT / 5000 ep RQ-VAE | 0.0039 | 0.0218 | 0.0551 | 0.0112 | 0.0184 | SID-level |")
    lines.append("| TIGER | 10 epochs / 5000 ep RQ-VAE | 0.0017 | 0.0090 | 0.0199 | 0.0050 | 0.0074 | SID-level |")
    lines.append("| MMGRec | recorded best checkpoint | 0.0014 | 0.0132 | 0.0339 | 0.0063 | 0.0109 | SID-level |")
    lines.append("| CARD | 50 epochs / 3000 ep NU-RQ-VAE | 0.0039 | 0.0233 | 0.0655 | 0.0121 | 0.0212 | Item-level (1:1 SID) |")
    return "\n".join(lines)


if __name__ == "__main__":
    out = []
    for title, fn in [
        ("T1. Final full-SID collision metrics", table_final),
        ("T2. Raw / Sinkhorn / merge stages", table_stages),
        ("T3. Prefix-depth statistics", table_prefix),
        ("T4. Bucket-size histogram", table_hist),
        ("T5. Collided-item analysis (final merged SIDs)", table_collided),
        ("T6. Pre-merge vs merge-induced collisions", table_stage_collided),
        ("T7. Beam-level generation diagnostics", table_beams),
        ("T8. SID-level vs item-level HR@50", table_item_level),
        ("T9. Recorded performance context", table_perf),
    ]:
        out.append(f"## {title}\n\n{fn()}\n")
    text = "\n".join(out)
    with open(os.path.join(RES, "tables.md"), "w") as f:
        f.write(text)
    print(text)
