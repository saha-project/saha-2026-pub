"""S1 — Atlas QC and methods support (4 panels: a, b, c, d)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8

ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}

DATA = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI/ig_validation"
OUT  = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI/figures"

# ── load ──────────────────────────────────────────────────────────────────────
summary  = pd.read_csv(f"{DATA}/spatial_atomic_unit_summary.csv")
cell_area = pd.read_csv(f"{DATA}/jchain_qc_cell_area.csv")
panel    = pd.read_csv(f"{DATA}/gene_panel_probe_present.csv")  # gene, APE, COL, ILE

# panel coverage per organ
panel_counts = {
    "APE": panel["APE"].sum(),
    "COL": panel["COL"].sum(),
    "ILE": panel["ILE"].sum(),
}
total_probes = len(panel)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ── panel a: per-unit cell counts, grouped by organ ───────────────────────────
organs_order = ["APE", "COL", "ILE"]
df_sorted = summary.sort_values(["organ", "unit_id"]).reset_index(drop=True)
x = np.arange(len(df_sorted))
bar_colors = [ORGAN_COLORS[o] for o in df_sorted["organ"]]
ax_a.bar(x, df_sorted["n_cells"], color=bar_colors, width=0.8, linewidth=0)

# organ dividers & labels
organ_groups = df_sorted.groupby("organ", sort=False)
tick_positions = []
tick_labels    = []
for organ in organs_order:
    idx = df_sorted[df_sorted["organ"] == organ].index
    if len(idx):
        mid = (idx[0] + idx[-1]) / 2
        tick_positions.append(mid)
        tick_labels.append(organ)
        if idx[-1] < len(df_sorted) - 1:
            ax_a.axvline(idx[-1] + 0.5, color="gray", lw=0.6, ls="--", alpha=0.5)

ax_a.set_xticks(tick_positions)
ax_a.set_xticklabels(tick_labels, fontsize=10)
ax_a.set_ylabel("Cell count per atomic unit", fontsize=9)
ax_a.set_title("Per-unit cell counts (n=37)", fontsize=10)
# legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ORGAN_COLORS[o], label=o) for o in organs_order]
ax_a.legend(handles=legend_elements, fontsize=8, frameon=False)

# ── panel b: gene panel coverage per organ ────────────────────────────────────
organs = list(panel_counts.keys())
counts = [panel_counts[o] for o in organs]
colors = [ORGAN_COLORS[o] for o in organs]
bars = ax_b.bar(organs, counts, color=colors, width=0.5, linewidth=0)
ax_b.axhline(total_probes, color="black", lw=0.8, ls="--", alpha=0.6, label=f"Total probes ({total_probes})")
ax_b.set_ylim(0, total_probes * 1.15)
for bar, cnt in zip(bars, counts):
    ax_b.text(bar.get_x() + bar.get_width()/2, cnt + total_probes*0.01,
              str(cnt), ha="center", va="bottom", fontsize=9)
ax_b.set_ylabel("Probes detected (fraction >0)", fontsize=9)
ax_b.set_title("Gene panel coverage per organ", fontsize=10)
ax_b.legend(fontsize=8, frameon=False)

# ── panel c: niche-bin cell counts per unit, stacked bars ─────────────────────
df_c = summary.sort_values(["organ", "unit_id"]).reset_index(drop=True)
x_c = np.arange(len(df_c))
niche_cols = [("n_lp_distal", "#AECFCF"), ("n_galt_edge", "#6FA8A8"), ("n_galt_interior", "#2D6B6B")]
bottoms = np.zeros(len(df_c))
for col, color in niche_cols:
    vals = df_c[col].values
    ax_c.bar(x_c, vals, bottom=bottoms, color=color, width=0.8, linewidth=0,
             label=col.replace("n_", "").replace("_", " ").title())
    bottoms += vals

ax_c.set_xticks(tick_positions)
ax_c.set_xticklabels(tick_labels, fontsize=10)
for organ in organs_order:
    idx = df_c[df_c["organ"] == organ].index
    if len(idx) and idx[-1] < len(df_c) - 1:
        ax_c.axvline(idx[-1] + 0.5, color="gray", lw=0.6, ls="--", alpha=0.5)
ax_c.set_ylabel("Cell count", fontsize=9)
ax_c.set_title("Niche-bin cell counts per atomic unit", fontsize=10)
ax_c.legend(fontsize=7, frameon=False, loc="upper right")

# ── panel d: cell-area violin per organ ───────────────────────────────────────
# cell_area has columns: organ, group, n_cells, med_area_um2, p25, p75
# we only have summary stats; draw a box-whisker-like bar using p25/med/p75
groups = ["all", "IGH+"]
group_colors = {"all": "#AAAAAA", "IGH+": "#555555"}
x_vals = np.arange(len(organs_order))
width = 0.35

for gi, grp in enumerate(groups):
    sub = cell_area[cell_area["group"] == grp].set_index("organ")
    offsets = (gi - 0.5) * width
    for xi, org in enumerate(organs_order):
        if org not in sub.index:
            continue
        row = sub.loc[org]
        med = row["med_area_um2"]
        p25 = row["p25"]
        p75 = row["p75"]
        xpos = xi + offsets
        ax_d.bar(xpos, p75 - p25, bottom=p25,
                 color=ORGAN_COLORS[org], alpha=0.6 if grp == "all" else 1.0,
                 width=width * 0.9, linewidth=0.5, edgecolor="white")
        ax_d.hlines(med, xpos - width*0.45, xpos + width*0.45,
                    color="black", lw=1.2)

ax_d.set_xticks(x_vals)
ax_d.set_xticklabels(organs_order, fontsize=10)
ax_d.set_ylabel("Cell area (µm²)", fontsize=9)
ax_d.set_title("Cell-area distribution (IQR bars, median line)", fontsize=10)
# manual legend
from matplotlib.patches import Patch
leg_elems = [Patch(facecolor="#AAAAAA", alpha=0.6, label="all cells"),
             Patch(facecolor="#555555", label="IGH+ cells")]
ax_d.legend(handles=leg_elems, fontsize=8, frameon=False)

# ── panel labels ──────────────────────────────────────────────────────────────
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

fig.patch.set_facecolor("white")
fig.savefig(f"{OUT}/fig_S1_atlas_qc.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_S1_atlas_qc.pdf", bbox_inches="tight", facecolor="white")
print("S1 saved.")
