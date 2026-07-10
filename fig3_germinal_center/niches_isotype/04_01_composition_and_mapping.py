"""
04_01_composition_and_mapping.py

Phase 04, Plan 01: Composition and niche mapping figures.
Produces:
  - figures/fig_composition_bar.pdf/png   (VIZ-04)
  - figures/fig_niche_heatmap.pdf/png     (VIZ-05)
  - figures/fig_sankey.pdf/png            (VIZ-06)

Run with:
    conda run -n saha python scripts/04_01_composition_and_mapping.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MPath
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared palettes
# ---------------------------------------------------------------------------
OKABE_ITO = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
    '#000000', '#999999', '#E69F5E', '#57C4AD', '#F7B7A3', '#8EC8F6', '#ABDDA4',
    '#FEE08B', '#D9EF8B', '#A6D96A', '#66BD63', '#1A9850', '#006837', '#543005'
]

NICHE_PALETTE = [
    '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8',
    '#FFFFBF', '#FEE090', '#FDAE61', '#D73027'
]

# ---------------------------------------------------------------------------
# Helper: set Arial / Helvetica font globally
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,   # embed fonts as TrueType in PDF
    'ps.fonttype': 42,
})

# ===========================================================================
# SECTION 1 — VIZ-04: Cell-type composition stacked bar chart
# ===========================================================================

ct = pd.read_csv(BASE_DIR / "organ_celltype_table.csv", index_col="new_clusters")
organs = ['APE', 'COL', 'ILE']

# Percentage per organ column
pct = ct.div(ct.sum(axis=0), axis=1) * 100

# Organ-exclusive types: non-zero in exactly 1 organ
exclusive_mask = (ct > 0).sum(axis=1) == 1
exclusive_types = exclusive_mask[exclusive_mask].index.tolist()

# Sort: shared (alpha) first, then exclusive (alpha)
shared_types = sorted([t for t in pct.index if t not in exclusive_types])
all_types = shared_types + sorted(exclusive_types)
pct = pct.loc[all_types, organs]
colors = {t: OKABE_ITO[i % len(OKABE_ITO)] for i, t in enumerate(all_types)}

fig, ax = plt.subplots(figsize=(3.5, 5.5))
bottom = np.zeros(3)
x = np.arange(3)

for ct_name in all_types:
    vals = pct.loc[ct_name, organs].values.astype(float)
    edgecolor = 'black' if ct_name in exclusive_types else 'none'
    lw = 0.8 if ct_name in exclusive_types else 0
    ax.bar(x, vals, bottom=bottom, color=colors[ct_name],
           edgecolor=edgecolor, linewidth=lw, width=0.6)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(organs, fontsize=9)
ax.set_ylabel('Cell type composition (%)', fontsize=9)
ax.set_ylim(0, 100)
ax.tick_params(axis='y', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend: exclusive types get asterisk
import matplotlib.patches as mpatches
legend_handles = []
for ct_name in all_types:
    label = f"{ct_name}*" if ct_name in exclusive_types else ct_name
    legend_handles.append(mpatches.Patch(
        facecolor=colors[ct_name],
        edgecolor='black' if ct_name in exclusive_types else 'none',
        linewidth=0.6,
        label=label
    ))
ax.legend(handles=legend_handles,
          bbox_to_anchor=(1.02, 1), loc='upper left',
          fontsize=6, frameon=False, ncol=1)

# Panel label
ax.text(-0.18, 1.05, 'A', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

fig.savefig(FIGURES_DIR / "fig_composition_bar.pdf", bbox_inches='tight', dpi=300)
fig.savefig(FIGURES_DIR / "fig_composition_bar.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("VIZ-04 saved")

# ===========================================================================
# SECTION 2 — VIZ-05: Niche composition heatmap
# ===========================================================================

import anndata
from scipy.stats import zscore
import seaborn as sns

adata = anndata.read_h5ad(BASE_DIR / "analysis_ready.h5ad")

# Build niche × cell_type count matrix
niche_ct = pd.crosstab(adata.obs['UTAG_names'], adata.obs['new_clusters'])

# Row-normalize to fraction within each niche
niche_frac = niche_ct.div(niche_ct.sum(axis=1), axis=0)

# Z-score across niches per cell type (column-wise)
niche_z = niche_frac.apply(lambda col: zscore(col, ddof=0), axis=0)

# Sort both axes alphabetically for consistency
niches_sorted = sorted(niche_z.index.tolist())
celltypes_sorted = sorted(niche_z.columns.tolist())
niche_z = niche_z.loc[niches_sorted, celltypes_sorted]

fig, ax = plt.subplots(figsize=(7.0, 3.5))
sns.heatmap(
    niche_z, ax=ax,
    cmap='RdBu_r', center=0, vmin=-2.5, vmax=2.5,
    linewidths=0.3, linecolor='white',
    xticklabels=True, yticklabels=True,
    cbar_kws={'label': 'z-score (fraction)', 'shrink': 0.6}
)
ax.set_xlabel('Cell type', fontsize=9)
ax.set_ylabel('UTAG niche', fontsize=9)
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.tick_params(axis='y', labelsize=8, rotation=0)
ax.text(-0.05, 1.10, 'B', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

fig.savefig(FIGURES_DIR / "fig_niche_heatmap.pdf", bbox_inches='tight', dpi=300)
fig.savefig(FIGURES_DIR / "fig_niche_heatmap.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("VIZ-05 saved")

# ===========================================================================
# SECTION 3 — VIZ-06: Sankey / ribbon diagram (pure matplotlib)
# ===========================================================================

# Build cell_type × niche flow matrix (cell counts)
flow = pd.crosstab(adata.obs['new_clusters'], adata.obs['UTAG_names'])
total = float(flow.values.sum())
flow_norm = flow / total  # fraction of all cells

cell_types = sorted(flow.index.tolist())
niches = sorted(flow.columns.tolist())

ct_colors = {t: OKABE_ITO[i % len(OKABE_ITO)] for i, t in enumerate(cell_types)}
niche_colors = {n: NICHE_PALETTE[i] for i, n in enumerate(niches)}

fig, ax = plt.subplots(figsize=(5.0, 7.0))
ax.axis('off')

left_x = 0.12
right_x = 0.88
bar_width = 0.045
gap = 0.003  # small gap between bars for readability

# Heights (proportional to total cells)
ct_heights = flow_norm.sum(axis=1)[cell_types]
niche_heights = flow_norm.sum(axis=0)[niches]

# Cumulative bottom positions for left bars (cell types)
ct_bottoms = {}
pos = 0.0
for ct_name in cell_types:
    ct_bottoms[ct_name] = pos
    pos += ct_heights[ct_name] + gap
left_total = pos - gap  # remove trailing gap

# Scale to [0, 1] on y
def scale_left(y):
    return y / left_total if left_total > 0 else y

# Cumulative bottom positions for right bars (niches)
niche_bottoms = {}
pos = 0.0
for n in niches:
    niche_bottoms[n] = pos
    pos += niche_heights[n] + gap
right_total = pos - gap

def scale_right(y):
    return y / right_total if right_total > 0 else y

y_margin = 0.02
y_range = 1.0 - 2 * y_margin

# Draw left bars (cell types)
for ct_name in cell_types:
    b = y_margin + scale_left(ct_bottoms[ct_name]) * y_range
    h = scale_left(ct_heights[ct_name]) * y_range
    rect = plt.Rectangle(
        (left_x - bar_width / 2, b), bar_width, h,
        facecolor=ct_colors[ct_name], edgecolor='none', zorder=3
    )
    ax.add_patch(rect)
    mid = b + h / 2
    ax.text(left_x - bar_width / 2 - 0.015, mid, ct_name,
            ha='right', va='center', fontsize=6.5)

# Draw right bars (niches)
for n in niches:
    b = y_margin + scale_right(niche_bottoms[n]) * y_range
    h = scale_right(niche_heights[n]) * y_range
    rect = plt.Rectangle(
        (right_x - bar_width / 2, b), bar_width, h,
        facecolor=niche_colors[n], edgecolor='none', zorder=3
    )
    ax.add_patch(rect)
    mid = b + h / 2
    ax.text(right_x + bar_width / 2 + 0.015, mid, n,
            ha='left', va='center', fontsize=6.5)

# Track current fill position within each niche bar (in raw flow units)
niche_fill = {n: niche_bottoms[n] for n in niches}

# Draw ribbons
for ct_name in cell_types:
    ct_ribbon_bottom = ct_bottoms[ct_name]
    for n in niches:
        flow_val = flow_norm.loc[ct_name, n]
        if flow_val < 5e-4:
            # Still advance counters to stay in sync
            niche_fill[n] += flow_val
            ct_ribbon_bottom += flow_val
            continue

        # Left edge (scaled)
        ly0 = y_margin + scale_left(ct_ribbon_bottom) * y_range
        ly1 = y_margin + scale_left(ct_ribbon_bottom + flow_val) * y_range

        # Right edge (scaled)
        ry0 = y_margin + scale_right(niche_fill[n]) * y_range
        ry1 = y_margin + scale_right(niche_fill[n] + flow_val) * y_range

        cx = (left_x + right_x) / 2  # bezier control x

        verts = [
            (left_x + bar_width / 2, ly0),
            (cx, ly0),
            (cx, ry0),
            (right_x - bar_width / 2, ry0),
            (right_x - bar_width / 2, ry1),
            (cx, ry1),
            (cx, ly1),
            (left_x + bar_width / 2, ly1),
            (left_x + bar_width / 2, ly0),
        ]
        codes = [
            MPath.MOVETO,
            MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO,
            MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.CLOSEPOLY,
        ]
        path = MPath(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor=ct_colors[ct_name],
            edgecolor='none',
            alpha=0.35,
            zorder=1
        )
        ax.add_patch(patch)

        ct_ribbon_bottom += flow_val
        niche_fill[n] += flow_val

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

# Column headers
ax.text(left_x, 1.02, 'Cell type', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.text(right_x, 1.02, 'UTAG niche', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.text(0.01, 1.04, 'C', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

fig.savefig(FIGURES_DIR / "fig_sankey.pdf", bbox_inches='tight', dpi=300)
fig.savefig(FIGURES_DIR / "fig_sankey.png", bbox_inches='tight', dpi=300)
plt.close(fig)
print("VIZ-06 saved")
