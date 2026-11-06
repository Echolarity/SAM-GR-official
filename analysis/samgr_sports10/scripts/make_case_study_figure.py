"""Case-study figure: target, collaborative neighbours + CSGA weights, semantic neighbours."""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIGURES, RESULTS, load_items  # noqa: E402

COL_COLLAB = "#C44E52"
COL_SEM = "#55A868"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.dpi": 300})


def load_img(path):
    im = Image.open(path).convert("RGB")
    return ImageOps.fit(im, (320, 320), method=Image.LANCZOS)


def short(s, n=34):
    return textwrap.shorten(str(s).replace("\n", " "), width=n, placeholder="...")


def add_neighbor_panel(fig, spec, ids, values, items, mode):
    sub = spec.subgridspec(2, len(ids), height_ratios=[3.6, 1.0], hspace=0.06, wspace=0.08)
    for c, (iid, val) in enumerate(zip(ids, values)):
        iid = int(iid)
        ax = fig.add_subplot(sub[0, c])
        paths = items[str(iid)].get("image_paths") or []
        if paths and os.path.exists(paths[0]):
            ax.imshow(load_img(paths[0]))
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(COL_COLLAB if mode == "collab" else COL_SEM)
            sp.set_linewidth(1.4)
        cats = [x.strip() for x in (items[str(iid)].get("categories") or "").split(",") if x.strip()]
        leaf = cats[-1] if cats else "Unknown"
        ax.set_title(short(leaf, 22), fontsize=6.6, pad=2)

        bx = fig.add_subplot(sub[1, c])
        if mode == "collab":
            bx.barh([0], [val], color=COL_COLLAB, height=0.55)
            bx.set_xlim(0, 0.36)
            bx.text(min(val + 0.008, 0.30), 0, f"w={val:.2f}", va="center", fontsize=6.0)
        else:
            bx.barh([0], [val], color=COL_SEM, height=0.55)
            bx.set_xlim(0.80, 1.0)
            bx.text(0.805, 0, f"cos={val:.2f}", va="center", fontsize=6.0)
        bx.set_yticks([]); bx.set_xticks([])
        for sp in bx.spines.values():
            sp.set_visible(False)


def make(rows, filename, suptitle):
    nrows = len(rows)
    fig_h = 3.35 * nrows + 1.05
    fig = plt.figure(figsize=(19.5, fig_h))
    outer = fig.add_gridspec(
        nrows, 3, width_ratios=[1.15, 3.5, 3.5],
        left=0.035, right=0.995, top=0.91, bottom=0.015,
        hspace=0.30, wspace=0.14,
    )
    items = load_items()
    for r, row in enumerate(rows.itertuples(index=False)):
        # ----- target -----
        ax = fig.add_subplot(outer[r, 0])
        paths = items[str(row.item)].get("image_paths") or []
        if paths and os.path.exists(paths[0]):
            ax.imshow(load_img(paths[0]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333333"); sp.set_linewidth(1.2)
        ax.set_title(f"Case {row.case}", fontsize=8.5, loc="left", fontweight="bold")
        info = (f"#{row.item}  {short(row.title, 48)}\n"
                f"{row.cat_l2} > {row.cat_leaf}\n"
                f"pop={row.pop_users} users ({row.group})\n"
                f"collab∩semantic = {row.overlap10}/10\n"
                f"CSGA top-1 w={row.top1_weight:.2f}, H/Hmax={row.entropy_norm:.2f}")
        ax.text(0.0, -0.06, info, transform=ax.transAxes, va="top", fontsize=7.2,
                linespacing=1.22, color="#222222")

        # ----- collaborative / semantic -----
        ids = [int(x) for x in str(row.collab_ids).split()]
        vals = [float(x) for x in str(row.collab_weights).split()]
        add_neighbor_panel(fig, outer[r, 1], ids, vals, items, "collab")
        ids = [int(x) for x in str(row.semantic_ids).split()]
        vals = [float(x) for x in str(row.semantic_cos).split()]
        add_neighbor_panel(fig, outer[r, 2], ids, vals, items, "sem")

    fig.text(0.285, 0.955, "Collaborative Top-4 (bar = learned CSGA weight $a_{ij}$)",
             ha="center", fontsize=10, color=COL_COLLAB, fontweight="bold")
    fig.text(0.755, 0.955, "Semantic Top-4 (bar = joint-multimodal cosine)",
             ha="center", fontsize=10, color=COL_SEM, fontweight="bold")
    fig.suptitle(suptitle, fontsize=12, y=0.992)
    fig.savefig(os.path.join(FIGURES, filename), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, filename.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(RESULTS, "exp_f_cases.csv"))
    make(df.iloc[:3], "fig_case_study_part1.png",
         "Case study (1/2): collaborative and semantic neighbours are disjoint")
    make(df.iloc[3:], "fig_case_study_part2.png",
         "Case study (2/2): collaborative and semantic neighbours are disjoint")
    make(df, "fig_case_study_combined.png",
         "Case study: target, collaborative neighbours with learned CSGA weights, and semantic neighbours\n"
         "(Sports strict 10-core; all cases have zero overlap between the two Top-10 lists)")
    print("CASE_FIGURES_DONE")


if __name__ == "__main__":
    main()
