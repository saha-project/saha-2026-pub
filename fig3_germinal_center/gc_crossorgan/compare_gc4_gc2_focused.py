"""Focused dot plot + heatmap: LN-GC4 vs APE-GC2, averaged by cell type x organ."""
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

# Genes from the heatmap image (exact order: LN-up top, APE-up bottom)
LN_UP_GENES = ['IGHD', 'HLA-DQA1', 'HLA-DQB1/2', 'IGHG1']
APE_UP_GENES = ['RPS4Y1', 'IGHA1', 'CXCR4', 'FAU']
ALL_GENES = LN_UP_GENES + APE_UP_GENES

CT_ORDER = ['GCBC', 'FDC', 'NBC/MBC', 'PC', 'TC_CD4', 'TC_CD8', 'macro']
ORGAN_ORDER = ['LN', 'APE']

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
adata.obs['gc_id'] = obs_assign['gc_id'].values

# Subset to LN-GC4 and APE-GC2
focus_mask = adata.obs['gc_id'].isin(['LN-GC4', 'APE-GC2'])
adata_sub = adata[focus_mask].copy()
adata_sub.obs['organ'] = adata_sub.obs['SAHA_organcode'].values

sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# Filter to genes present in panel
genes_present = [g for g in ALL_GENES if g in adata_sub.var_names]
ln_genes_present = [g for g in LN_UP_GENES if g in genes_present]
ape_genes_present = [g for g in APE_UP_GENES if g in genes_present]
print(f"Genes present: {len(genes_present)}/{len(ALL_GENES)}")
print(f"  LN-up: {ln_genes_present}")
print(f"  APE-up: {ape_genes_present}")

# Create organ_ct grouping
adata_sub.obs['organ_ct'] = (adata_sub.obs['organ'].astype(str) + ' ' +
                              adata_sub.obs['cell_types_insitutype'].astype(str))

groups_order = []
for organ in ORGAN_ORDER:
    for ct in CT_ORDER:
        key = f'{organ} {ct}'
        if key in adata_sub.obs['organ_ct'].values:
            groups_order.append(key)

# Compute mean expression and pct per group
dot_data = []
for grp in groups_order:
    mask = adata_sub.obs['organ_ct'] == grp
    n_cells = int(mask.sum())
    if n_cells == 0:
        continue
    X = adata_sub[mask][:, genes_present].X
    if sp.issparse(X):
        X = X.toarray()
    for j, gene in enumerate(genes_present):
        vals = X[:, j]
        dot_data.append({
            'group': grp, 'gene': gene,
            'mean_expr': float(vals.mean()),
            'pct': float((vals > 0).mean()),
            'n_cells': n_cells,
        })

dot_df = pd.DataFrame(dot_data)

# Build mean expression matrix for heatmap (groups x genes)
mean_matrix = np.zeros((len(groups_order), len(genes_present)))
for i, grp in enumerate(groups_order):
    for j, gene in enumerate(genes_present):
        row = dot_df[(dot_df['group'] == grp) & (dot_df['gene'] == gene)]
        if len(row) > 0:
            mean_matrix[i, j] = row.iloc[0]['mean_expr']

# Z-score per gene (across groups) for heatmap
mean_z = np.zeros_like(mean_matrix)
for j in range(mean_matrix.shape[1]):
    col = mean_matrix[:, j]
    m, s = col.mean(), col.std()
    mean_z[:, j] = (col - m) / s if s > 0 else 0

# ============================================================
# Figure 1: Dot plot
# ============================================================
print("\nGenerating dot plot...")
plt.rcParams.update(nature_rcparams())

vmax = dot_df['mean_expr'].quantile(0.95)
if vmax == 0:
    vmax = 1

n_genes = len(genes_present)
n_groups = len(groups_order)
fig_dp, ax_dp = plt.subplots(1, 1, figsize=(5.5, 4.0))

for i, grp in enumerate(groups_order):
    for j, gene in enumerate(genes_present):
        row = dot_df[(dot_df['group'] == grp) & (dot_df['gene'] == gene)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        sc_obj = ax_dp.scatter(j, i, s=r['pct'] * 250, c=[[r['mean_expr']]], cmap='Reds',
                               vmin=0, vmax=vmax, edgecolors='#555', linewidths=0.3, zorder=2)

ax_dp.set_xticks(range(n_genes))
ax_dp.set_xticklabels(genes_present, fontsize=5, rotation=60, ha='right')
ax_dp.set_yticks(range(n_groups))
ylabels = [f'{g.split(" ")[0]}: {g.split(" ", 1)[1]}' for g in groups_order]
ax_dp.set_yticklabels(ylabels, fontsize=5)

for i, label in enumerate(ax_dp.get_yticklabels()):
    organ = groups_order[i].split(' ')[0]
    label.set_color('#2A6365' if organ == 'LN' else '#8B0000')

# Organ separator
n_ln = sum(1 for g in groups_order if g.startswith('LN'))
ax_dp.axhline(n_ln - 0.5, color='black', lw=0.8)

# Gene group separator
n_ln_genes = len(ln_genes_present)
ax_dp.axvline(n_ln_genes - 0.5, color='#999', lw=0.5, ls='--')

# Grid
for i in range(n_groups):
    ax_dp.axhline(i, color='#F0F0F0', lw=0.3, zorder=0)
for j in range(n_genes):
    ax_dp.axvline(j, color='#F0F0F0', lw=0.3, zorder=0)

ax_dp.set_xlim(-0.5, n_genes - 0.5)
ax_dp.set_ylim(-0.5, n_groups - 0.5)
ax_dp.invert_yaxis()
ax_dp.set_title('LN-GC4 vs APE-GC2: Top DE genes', fontsize=7)
apply_nature_style(ax_dp)

# Gene direction labels
ax_dp.text(n_ln_genes / 2, -1.2, 'LN-enriched', ha='center', fontsize=5, color='#2A6365', fontstyle='italic')
ax_dp.text(n_ln_genes + len(ape_genes_present) / 2, -1.2, 'APE-enriched', ha='center', fontsize=5, color='#8B0000', fontstyle='italic')

cbar = fig_dp.colorbar(sc_obj, ax=ax_dp, shrink=0.4, pad=0.02, aspect=15)
cbar.set_label('Mean expr', fontsize=5)
cbar.ax.tick_params(labelsize=4)

size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
                       markersize=np.sqrt(v * 250) / 2, label=f'{int(v*100)}%',
                       markeredgecolor='#555', markeredgewidth=0.3)
                for v in [0.25, 0.50, 0.75, 1.0]]
ax_dp.legend(handles=size_handles, title='% expr', title_fontsize=4,
             fontsize=4, loc='lower right', frameon=False, handletextpad=0.2, labelspacing=0.8)

fig_dp.savefig(os.path.join(FIGURES_DIR, '04_dotplot_focused_v4.png'), dpi=300, bbox_inches='tight')
fig_dp.savefig(os.path.join(FIGURES_DIR, '04_dotplot_focused_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 04_dotplot_focused_v4.png/pdf")

# ============================================================
# Figure 2: Heatmap (averaged by cell type x organ)
# ============================================================
print("Generating heatmap...")
plt.rcParams.update(nature_rcparams())

fig_hm, ax_hm = plt.subplots(1, 1, figsize=(5.5, 4.0))

im = ax_hm.imshow(mean_z, cmap='RdBu_r', aspect='auto', vmin=-2.5, vmax=2.5)

ax_hm.set_xticks(range(n_genes))
ax_hm.set_xticklabels(genes_present, fontsize=5, rotation=60, ha='right')
ax_hm.set_yticks(range(n_groups))
ax_hm.set_yticklabels(ylabels, fontsize=5)

for i, label in enumerate(ax_hm.get_yticklabels()):
    organ = groups_order[i].split(' ')[0]
    label.set_color('#2A6365' if organ == 'LN' else '#8B0000')

# Separators
ax_hm.axhline(n_ln - 0.5, color='black', lw=1)
ax_hm.axvline(n_ln_genes - 0.5, color='black', lw=0.8)

ax_hm.set_title('LN-GC4 vs APE-GC2: Mean expression (z-score)', fontsize=7)

# Gene direction labels
ax_hm.text(n_ln_genes / 2, -1.2, 'LN-enriched', ha='center', fontsize=5, color='#2A6365', fontstyle='italic')
ax_hm.text(n_ln_genes + len(ape_genes_present) / 2, -1.2, 'APE-enriched', ha='center', fontsize=5, color='#8B0000', fontstyle='italic')

cbar_hm = fig_hm.colorbar(im, ax=ax_hm, shrink=0.5, pad=0.02)
cbar_hm.set_label('Z-score', fontsize=5)
cbar_hm.ax.tick_params(labelsize=4)

apply_nature_style(ax_hm)

fig_hm.savefig(os.path.join(FIGURES_DIR, '04_heatmap_focused_v4.png'), dpi=300, bbox_inches='tight')
fig_hm.savefig(os.path.join(FIGURES_DIR, '04_heatmap_focused_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 04_heatmap_focused_v4.png/pdf")

print("\nFocused dot plot + heatmap complete.")
