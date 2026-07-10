"""Publication-ready dot plot: LN-GC4 vs APE-GC2 across all GC cell types."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
adata.obs['gc_id'] = obs_assign['gc_id'].values

import scanpy as sc

# Subset to LN-GC4 and APE-GC2
focus_mask = adata.obs['gc_id'].isin(['LN-GC4', 'APE-GC2'])
adata_sub = adata[focus_mask].copy()
adata_sub.obs['organ'] = adata_sub.obs['SAHA_organcode'].values

# Normalize
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# Create organ_celltype grouping
adata_sub.obs['organ_ct'] = (adata_sub.obs['organ'].astype(str) + ' ' +
                              adata_sub.obs['cell_types_insitutype'].astype(str))

# Select genes: top DE + known GC markers + key findings
de = pd.read_csv(os.path.join(OUTPUTS_DIR, 'de_results_GCBC_v4.csv'))
sig = de[(de['pval_adj'] < 0.01) & (de['log2fc'].abs() > 0.5)]
top_ape_genes = sig.nlargest(10, 'log2fc')['gene'].tolist()
top_ln_genes = sig.nsmallest(5, 'log2fc')['gene'].tolist()

GC_MARKERS = ['CXCR4', 'CXCR5', 'CD83', 'CXCL13', 'IRF4', 'MYC', 'BCL2', 'CD38']

# Combine and deduplicate, keeping order
all_genes = []
seen = set()
# Section 1: DZ/LZ markers
for g in ['CXCR4', 'CXCR5', 'CD83', 'BCL2', 'MYC', 'IRF4', 'CD38', 'CXCL13']:
    if g not in seen and g in adata_sub.var_names:
        all_genes.append(g)
        seen.add(g)
# Section 2: Top APE-up DE genes
for g in top_ape_genes:
    if g not in seen and g in adata_sub.var_names:
        all_genes.append(g)
        seen.add(g)
# Section 3: Top LN-up DE genes
for g in top_ln_genes:
    if g not in seen and g in adata_sub.var_names:
        all_genes.append(g)
        seen.add(g)
# Section 4: Additional interesting genes
for g in ['IGHA1', 'IGHG1', 'PCNA', 'CD79A', 'CD19', 'MS4A1', 'CD74', 'B2M']:
    if g not in seen and g in adata_sub.var_names:
        all_genes.append(g)
        seen.add(g)

print(f"Genes for dot plot: {len(all_genes)}")

# Cell types to show (ordered by relevance)
CT_ORDER = ['GCBC', 'FDC', 'NBC/MBC', 'PC', 'TC_CD4', 'TC_CD8', 'macro']
ORGAN_ORDER = ['LN', 'APE']

groups_order = []
for organ in ORGAN_ORDER:
    for ct in CT_ORDER:
        key = f'{organ} {ct}'
        if key in adata_sub.obs['organ_ct'].values:
            groups_order.append(key)

print(f"Groups: {groups_order}")

# Compute dot plot data: mean expression + pct expressing
dot_data = []
for grp in groups_order:
    mask = adata_sub.obs['organ_ct'] == grp
    n_cells = int(mask.sum())
    if n_cells == 0:
        continue
    X = adata_sub[mask][:, all_genes].X
    if sp.issparse(X):
        X = X.toarray()
    for j, gene in enumerate(all_genes):
        vals = X[:, j]
        dot_data.append({
            'group': grp, 'gene': gene,
            'mean_expr': float(vals.mean()),
            'pct': float((vals > 0).mean()),
            'n_cells': n_cells,
        })

dot_df = pd.DataFrame(dot_data)

# === Figure ===
print("Generating dot plot...")
plt.rcParams.update(nature_rcparams())

n_genes = len(all_genes)
n_groups = len(groups_order)

fig_width = 7.2
fig_height = max(3.5, n_groups * 0.3 + 1.5)
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

# Normalize mean_expr for color (0 to 95th percentile)
vmax = dot_df['mean_expr'].quantile(0.95)
if vmax == 0:
    vmax = 1

# Plot dots
for i, grp in enumerate(groups_order):
    for j, gene in enumerate(all_genes):
        row = dot_df[(dot_df['group'] == grp) & (dot_df['gene'] == gene)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        size = r['pct'] * 250  # scale for visibility
        color_val = min(r['mean_expr'] / vmax, 1.0)

        sc_plot = ax.scatter(j, i, s=size, c=[[r['mean_expr']]], cmap='Reds',
                            vmin=0, vmax=vmax, edgecolors='#555', linewidths=0.3,
                            zorder=2)

# Axis labels
ax.set_xticks(range(n_genes))
ax.set_xticklabels(all_genes, fontsize=4.5, rotation=60, ha='right')
ax.set_yticks(range(n_groups))

# Color-code y labels by organ
ylabels = []
for grp in groups_order:
    organ, ct = grp.split(' ', 1)
    ylabels.append(f'{organ}: {ct}')

ax.set_yticklabels(ylabels, fontsize=5)

# Color ytick labels
for i, label in enumerate(ax.get_yticklabels()):
    organ = groups_order[i].split(' ')[0]
    label.set_color('#2A6365' if organ == 'LN' else '#8B0000')

# Add horizontal line separating organs
n_ln_groups = sum(1 for g in groups_order if g.startswith('LN'))
ax.axhline(n_ln_groups - 0.5, color='black', lw=0.8, ls='-')

# Add vertical lines separating gene sections
# Section 1: GC markers (8), Section 2: APE DE, Section 3: LN DE, Section 4: additional
section_boundaries = [8]  # after GC markers
current = 8
for g in top_ape_genes:
    if g in all_genes[8:]:
        current += 1
section_boundaries.append(current)
# Find where LN genes start
for idx, g in enumerate(all_genes):
    if g in top_ln_genes and idx >= section_boundaries[-1]:
        section_boundaries.append(idx)
        break

for boundary in section_boundaries:
    if boundary < n_genes:
        ax.axvline(boundary - 0.5, color='#CCC', lw=0.5, ls='--')

# Grid lines for readability
for i in range(n_groups):
    ax.axhline(i, color='#F0F0F0', lw=0.3, zorder=0)
for j in range(n_genes):
    ax.axvline(j, color='#F0F0F0', lw=0.3, zorder=0)

ax.set_xlim(-0.5, n_genes - 0.5)
ax.set_ylim(-0.5, n_groups - 0.5)
ax.invert_yaxis()

ax.set_title('GC marker & DE gene expression: LN-GC4 vs APE-GC2', fontsize=7)
apply_nature_style(ax)

# Colorbar
cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.4, pad=0.02, aspect=15)
cbar.set_label('Mean expression\n(log-normalized)', fontsize=5)
cbar.ax.tick_params(labelsize=4)

# Size legend
size_legend_vals = [0.25, 0.50, 0.75, 1.0]
size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
                       markersize=np.sqrt(v * 250) / 2, label=f'{int(v*100)}%',
                       markeredgecolor='#555', markeredgewidth=0.3)
                for v in size_legend_vals]
leg = ax.legend(handles=size_handles, title='% expressing', title_fontsize=4,
                fontsize=4, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                frameon=False, handletextpad=0.2, labelspacing=0.8)

png_path = os.path.join(FIGURES_DIR, '04_dotplot_gc4_gc2_v4.png')
pdf_path = os.path.join(FIGURES_DIR, '04_dotplot_gc4_gc2_v4.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Dot plot complete.")
