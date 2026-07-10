"""Heatmap of top DE genes: LN-GC4 vs APE-GC2 GCBC."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
adata.obs['gc_id'] = obs_assign['gc_id'].values

de = pd.read_csv(os.path.join(OUTPUTS_DIR, 'de_results_GCBC_v4.csv'))

# Select top DE genes: top 15 APE-up + top 10 LN-up (padj < 0.01, |log2fc| > 0.5)
sig = de[(de['pval_adj'] < 0.01) & (de['log2fc'].abs() > 0.5)].copy()
top_ape = sig.nlargest(15, 'log2fc')['gene'].tolist()
top_ln = sig.nsmallest(10, 'log2fc')['gene'].tolist()

# Combine: LN-up at top, APE-up at bottom
gene_order = top_ln[::-1] + top_ape  # reverse LN so most negative is at top
gene_order = [g for g in gene_order if g in adata.var_names]  # filter to panel

print(f"Genes for heatmap: {len(gene_order)}")
print(f"  LN-up: {top_ln}")
print(f"  APE-up: {top_ape}")

# Subset to GCBC in LN-GC4 and APE-GC2
gcbc_mask = (
    adata.obs['gc_id'].isin(['LN-GC4', 'APE-GC2']) &
    (adata.obs['cell_types_insitutype'] == 'GCBC')
)
adata_sub = adata[gcbc_mask].copy()
adata_sub.obs['organ'] = adata_sub.obs['SAHA_organcode'].values

# Normalize
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# Compute mean expression per organ per gene
expr_data = {}
for organ in ['LN', 'APE']:
    mask = adata_sub.obs['organ'] == organ
    X = adata_sub[mask][:, gene_order].X
    if sp.issparse(X):
        X = X.toarray()
    expr_data[organ] = X.mean(axis=0)

mean_df = pd.DataFrame(expr_data, index=gene_order)

# Z-score per gene (across the two organs) for better visualization
mean_z = mean_df.copy()
for gene in mean_df.index:
    row_mean = mean_df.loc[gene].mean()
    row_std = mean_df.loc[gene].std()
    if row_std > 0:
        mean_z.loc[gene] = (mean_df.loc[gene] - row_mean) / row_std
    else:
        mean_z.loc[gene] = 0

# Also compute per-cell heatmap for more detail
# Sample cells for manageable heatmap
np.random.seed(42)
n_per_organ = 150
cells_ln = adata_sub.obs_names[adata_sub.obs['organ'] == 'LN']
cells_ape = adata_sub.obs_names[adata_sub.obs['organ'] == 'APE']
if len(cells_ln) > n_per_organ:
    cells_ln = np.random.choice(cells_ln, n_per_organ, replace=False)
if len(cells_ape) > n_per_organ:
    cells_ape = np.random.choice(cells_ape, n_per_organ, replace=False)

cells_ordered = list(cells_ln) + list(cells_ape)
adata_hm = adata_sub[cells_ordered][:, gene_order].copy()

X_hm = adata_hm.X
if sp.issparse(X_hm):
    X_hm = X_hm.toarray()

# Z-score per gene across all cells
X_z = np.zeros_like(X_hm)
for j in range(X_hm.shape[1]):
    col = X_hm[:, j]
    m, s = col.mean(), col.std()
    X_z[:, j] = (col - m) / s if s > 0 else 0

# Clip for display
X_z = np.clip(X_z, -3, 3)

# === Figure: 2 panels ===
print("Generating heatmap figure...")
plt.rcParams.update(nature_rcparams())

fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0),
                          gridspec_kw={'width_ratios': [1, 4]},
                          constrained_layout=True)

# Panel A: Mean expression bar (simple 2-column heatmap)
ax_mean = axes[0]
im_mean = ax_mean.imshow(mean_z.values, cmap='RdBu_r', aspect='auto',
                          vmin=-2, vmax=2)
ax_mean.set_xticks([0, 1])
ax_mean.set_xticklabels(['LN-GC4', 'APE-GC2'], fontsize=5, rotation=45, ha='right')
ax_mean.set_yticks(range(len(gene_order)))
ax_mean.set_yticklabels(gene_order, fontsize=4)
ax_mean.set_title('Mean (z-score)', fontsize=6)

# Add horizontal line separating LN-up from APE-up
n_ln_genes = len([g for g in top_ln[::-1] if g in adata.var_names])
ax_mean.axhline(n_ln_genes - 0.5, color='black', lw=1)
ax_mean.text(-0.5, n_ln_genes / 2, 'LN ↑', fontsize=5, ha='right', va='center', rotation=90, color='#2A6365', fontweight='bold')
ax_mean.text(-0.5, n_ln_genes + (len(gene_order) - n_ln_genes) / 2, 'APE ↑', fontsize=5, ha='right', va='center', rotation=90, color='#8B0000', fontweight='bold')

# Panel B: Per-cell heatmap
ax_hm = axes[1]
im_hm = ax_hm.imshow(X_z.T, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
ax_hm.set_yticks(range(len(gene_order)))
ax_hm.set_yticklabels(gene_order, fontsize=4)

# Mark organ boundary
n_ln_cells = len(cells_ln)
ax_hm.axvline(n_ln_cells - 0.5, color='black', lw=1)
ax_hm.set_xticks([n_ln_cells // 2, n_ln_cells + len(cells_ape) // 2])
ax_hm.set_xticklabels([f'LN-GC4\n(n={len(cells_ln)})', f'APE-GC2\n(n={len(cells_ape)})'], fontsize=5)

# Gene group separator
ax_hm.axhline(n_ln_genes - 0.5, color='black', lw=1)

ax_hm.set_title('Per-cell expression (z-score, 150 cells/organ)', fontsize=6)

# Colorbar
cbar = fig.colorbar(im_hm, ax=ax_hm, shrink=0.5, pad=0.02)
cbar.set_label('Z-score', fontsize=5)
cbar.ax.tick_params(labelsize=4)

fig.suptitle('Top DE genes: GCBC in LN-GC4 vs APE-GC2', fontsize=8, y=1.02)

png_path = os.path.join(FIGURES_DIR, '04_de_heatmap_v4.png')
pdf_path = os.path.join(FIGURES_DIR, '04_de_heatmap_v4.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Heatmap complete.")
