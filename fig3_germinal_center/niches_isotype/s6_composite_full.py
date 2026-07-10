"""
S6 — Composite cell-state scores (FULL — all 8 scores)
2×4 grid of violin panels, BH-corrected pairwise significance brackets.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from itertools import combinations

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8

ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
ORGANS = ["APE", "ILE", "COL"]

BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI"
FIG_DIR = BASE + "/figures"
DATA_DIR = BASE + "/ig_validation"

per_unit = pd.read_csv(DATA_DIR + "/galt_interior_composite_per_unit.csv")
stats = pd.read_csv(DATA_DIR + "/galt_interior_composite_stats.csv")

SCORES = ["Egress", "Plasma-fate", "B-retention", "Proliferation",
          "Dark-zone", "Light-zone", "Systemic-homing", "Gut-homing"]

# Build lookup: (score, orgA, orgB) -> stars
star_lookup = {}
for _, row in stats.iterrows():
    key = (row["score"], row["organ_A"], row["organ_B"])
    star_lookup[key] = row["stars_bh"]
    star_lookup[(row["score"], row["organ_B"], row["organ_A"])] = row["stars_bh"]


def get_stars(score, oA, oB):
    return star_lookup.get((score, oA, oB), "ns")


def draw_bracket(ax, x1, x2, y, h, text, fontsize=7):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, text,
            ha="center", va="bottom", fontsize=fontsize)


fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)

for idx, score in enumerate(SCORES):
    row_i, col_i = divmod(idx, 4)
    ax = fig.add_subplot(gs[row_i, col_i])

    sub = per_unit[per_unit["score_name"] == score]

    organ_data = {o: sub[sub["organ"] == o]["value"].values for o in ORGANS}

    positions = [1, 2, 3]
    parts = ax.violinplot(
        [organ_data[o] for o in ORGANS],
        positions=positions,
        widths=0.6,
        showmedians=False,
        showextrema=False,
    )
    for pc, organ in zip(parts["bodies"], ORGANS):
        pc.set_facecolor(ORGAN_COLORS[organ])
        pc.set_alpha(0.65)
        pc.set_edgecolor("none")

    for pos, organ in zip(positions, ORGANS):
        vals = organ_data[organ]
        if len(vals) == 0:
            continue
        ax.scatter(
            np.random.default_rng(42).uniform(pos - 0.15, pos + 0.15, len(vals)),
            vals,
            s=12, color=ORGAN_COLORS[organ], alpha=0.85, zorder=3, linewidths=0
        )
        med = np.median(vals)
        ax.hlines(med, pos - 0.25, pos + 0.25, lw=1.5, colors="black", zorder=4)

    # significance brackets
    pairs = [(0, 1, ORGANS[0], ORGANS[1]),
             (1, 2, ORGANS[1], ORGANS[2]),
             (0, 2, ORGANS[0], ORGANS[2])]
    all_vals = np.concatenate([organ_data[o] for o in ORGANS if len(organ_data[o]) > 0])
    ymax = all_vals.max() if len(all_vals) > 0 else 1.0
    yrange = (all_vals.max() - all_vals.min()) if len(all_vals) > 1 else 0.1
    step = yrange * 0.12

    bracket_y = ymax + yrange * 0.08
    for pi, (i1, i2, oA, oB) in enumerate(pairs):
        stars = get_stars(score, oA, oB)
        if stars != "ns":
            draw_bracket(ax, positions[i1], positions[i2],
                         bracket_y + pi * step, step * 0.6, stars)

    ax.set_xticks(positions)
    ax.set_xticklabels(ORGANS, fontsize=8)
    ax.set_title(score, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Mean score" if col_i == 0 else "", fontsize=8)

# Panel label
fig.text(0.01, 0.97, "a", fontsize=14, fontweight="bold", va="top")

fig.suptitle("Supplementary Figure S6 — Composite cell-state scores across organs",
             fontsize=11, y=1.01, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(
        FIG_DIR + f"/fig_S6_composite_scores_full.{ext}",
        dpi=600, bbox_inches="tight", facecolor="white"
    )

print("S6 done.")
