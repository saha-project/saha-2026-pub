"""Dot plot: DZ/LZ markers + Ig isotypes by B cell subtypes (Insitutype_Labelled)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

GENES = ['CXCR4', 'CD83', 'CXCR5', 'IGHA1', 'IGHG1', 'IGHM', 'PCNA', 'MYC']

# B cell lineage from cell_types_insitutype (consistent with all other figures)
B_CELL_TYPES = ['GCBC', 'NBC/MBC', 'FDC', 'PC']

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
adata.obs['gc_id'] = obs_assign['gc_id'].values

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

genes_present = [g for g in GENES if g in adata.var_names]
print(f"Genes: {genes_present}")

# Build groups: organ x B cell type
groups_order = []

for gc_id, organ_label in [('LN-GC4', 'LN'), ('APE-GC2', 'APE')]:
    for ct in B_CELL_TYPES:
        mask = (adata.obs['gc_id'] == gc_id) & (adata.obs['cell_types_insitutype'] == ct)
        n = mask.sum()
        if n >= 5:
            key = f'{organ_label}: {ct}'
            groups_order.append((gc_id, ct, key, n))
            print(f"  {key}: n={n}")

# Compute dot plot data
dot_data = []
for gc_id, subtype, label, n_cells in groups_order:
    mask = (adata.obs['gc_id'] == gc_id) & (adata.obs['cell_types_insitutype'] == subtype)
    X = adata[mask][:, genes_present].X
    if sp.issparse(X):
        X = X.toarray()
    for j, gene in enumerate(genes_present):
        vals = X[:, j]
        dot_data.append({
            'group': label, 'gene': gene,
            'mean_expr': float(vals.mean()),
            'pct': float((vals > 0).mean()),
            'n_cells': n_cells,
        })

dot_df = pd.DataFrame(dot_data)

# === Dot plot ===
print("\nGenerating dot plot...")
plt.rcParams.update(nature_rcparams())

n_genes = len(genes_present)
n_groups = len(groups_order)
vmax = dot_df['mean_expr'].quantile(0.95)
if vmax == 0:
    vmax = 1

fig, ax = plt.subplots(1, 1, figsize=(5.5, max(3.5, n_groups * 0.32 + 1.5)))

for i, (gc_id, subtype, label, n) in enumerate(groups_order):
    for j, gene in enumerate(genes_present):
        row = dot_df[(dot_df['group'] == label) & (dot_df['gene'] == gene)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        sc_obj = ax.scatter(j, i, s=r['pct'] * 250, c=[[r['mean_expr']]], cmap='Reds',
                           vmin=0, vmax=vmax, edgecolors='#555', linewidths=0.3, zorder=2)

ax.set_xticks(range(n_genes))
ax.set_xticklabels(genes_present, fontsize=5.5, rotation=60, ha='right')
ax.set_yticks(range(n_groups))
ylabels = [f'{label} (n={n})' for _, _, label, n in groups_order]
ax.set_yticklabels(ylabels, fontsize=4.5)

# Color y labels by organ
for i, label_obj in enumerate(ax.get_yticklabels()):
    organ = groups_order[i][0]
    label_obj.set_color('#2A6365' if 'LN' in organ else '#8B0000')

# Separator between organs
n_ln = sum(1 for g in groups_order if g[0] == 'LN-GC4')
ax.axhline(n_ln - 0.5, color='black', lw=0.8)

# Separator between DZ/LZ markers and Ig
# CXCR4, CD83, CXCR5 = indices 0-2; IGHA1, IGHG1, IGHM = 3-5; PCNA, MYC = 6-7
ax.axvline(2.5, color='#CCC', lw=0.5, ls='--')
ax.axvline(5.5, color='#CCC', lw=0.5, ls='--')

# Section labels
ax.text(1.0, -1.5, 'DZ/LZ\nmarkers', ha='center', fontsize=4.5, fontstyle='italic', color='#666')
ax.text(4.0, -1.5, 'Ig\nisotypes', ha='center', fontsize=4.5, fontstyle='italic', color='#666')
ax.text(6.5, -1.5, 'Prolif.', ha='center', fontsize=4.5, fontstyle='italic', color='#666')

# Grid
for i in range(n_groups):
    ax.axhline(i, color='#F0F0F0', lw=0.3, zorder=0)
for j in range(n_genes):
    ax.axvline(j, color='#F0F0F0', lw=0.3, zorder=0)

ax.set_xlim(-0.5, n_genes - 0.5)
ax.set_ylim(-0.5, n_groups - 0.5)
ax.invert_yaxis()
ax.set_title('B cell lineage: DZ/LZ markers, Ig isotypes, proliferation', fontsize=7)
apply_nature_style(ax)

cbar = fig.colorbar(sc_obj, ax=ax, shrink=0.35, pad=0.02, aspect=12)
cbar.set_label('Mean expr', fontsize=5)
cbar.ax.tick_params(labelsize=4)

size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
                       markersize=np.sqrt(v * 250) / 2, label=f'{int(v*100)}%',
                       markeredgecolor='#555', markeredgewidth=0.3)
                for v in [0.25, 0.50, 0.75, 1.0]]
ax.legend(handles=size_handles, title='% expr', title_fontsize=4,
          fontsize=4, loc='lower right', frameon=False, handletextpad=0.2, labelspacing=0.8)

png_path = os.path.join(FIGURES_DIR, 'dotplot_bcell_subtypes.png')
pdf_path = os.path.join(FIGURES_DIR, 'dotplot_bcell_subtypes.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("B cell subtype dot plot complete.")
