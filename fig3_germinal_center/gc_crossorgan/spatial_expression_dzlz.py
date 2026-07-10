"""Spatial expression plots: DZ/LZ markers + Ig isotypes for LN-GC4 vs APE-GC2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style, PIXEL_SIZE_UM

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

print("Loading data...")
adata = load_adata()
obs = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs.index = obs.index.astype(str)
adata.obs['gc_id'] = obs['gc_id'].values

# Normalize for expression plotting
import scanpy as sc
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

# Genes to plot: DZ markers, LZ markers, Ig isotypes, proliferation
GENES = [
    # DZ/LZ markers
    'CXCR4', 'CD83', 'CXCR5',
    # Ig isotypes
    'IGHA1', 'IGHG1', 'IGHM',
    # Proliferation
    'PCNA', 'MYC',
]
genes_present = [g for g in GENES if g in adata_norm.var_names]
print(f"Genes: {genes_present}")

FOCUS_GCS = ['LN-GC4', 'APE-GC2']

plt.rcParams.update(nature_rcparams())

n_genes = len(genes_present)
fig, axes = plt.subplots(n_genes, 2, figsize=(7.2, n_genes * 1.6), constrained_layout=True)

for col, gc_id in enumerate(FOCUS_GCS):
    gc_mask = adata.obs['gc_id'] == gc_id
    gc_pos = np.where(gc_mask.values)[0]
    coords = adata.obsm['spatial'][gc_pos] * PIXEL_SIZE_UM
    point_size = 5 if gc_id == 'LN-GC4' else 1.5

    for row, gene in enumerate(genes_present):
        ax = axes[row, col]

        gene_idx = list(adata_norm.var_names).index(gene)
        expr = adata_norm.X[gc_pos, gene_idx]
        if sp.issparse(expr):
            expr = np.asarray(expr.todense()).ravel()
        else:
            expr = np.asarray(expr).ravel()

        # Sort so high-expression cells plot on top
        order = np.argsort(expr)
        coords_sorted = coords[order]
        expr_sorted = expr[order]

        vmax = np.percentile(expr_sorted[expr_sorted > 0], 95) if (expr_sorted > 0).any() else 1.0
        if vmax == 0:
            vmax = 1.0

        sc_obj = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                           c=expr_sorted, cmap='magma', vmin=0, vmax=vmax,
                           s=point_size, linewidths=0, rasterized=True)

        pct_pos = (expr > 0).mean() * 100
        mean_val = expr.mean()

        organ_label = 'LN-GC4' if col == 0 else 'APE-GC2'
        if row == 0:
            ax.set_title(organ_label, fontsize=7)

        ax.text(0.02, 0.95, f'{gene}\n{pct_pos:.0f}% pos\nmean={mean_val:.2f}',
                transform=ax.transAxes, fontsize=4, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        apply_nature_style(ax)

        # Colorbar only on right column
        if col == 1:
            cbar = plt.colorbar(sc_obj, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=3)

# Row labels on left
for row, gene in enumerate(genes_present):
    axes[row, 0].set_ylabel(gene, fontsize=6, fontweight='bold', rotation=0,
                            labelpad=30, va='center')

fig.suptitle('Spatial gene expression: LN-GC4 vs APE-GC2', fontsize=8, y=1.01)

png_path = os.path.join(FIGURES_DIR, 'spatial_expression_dzlz.png')
pdf_path = os.path.join(FIGURES_DIR, 'spatial_expression_dzlz.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Spatial expression plots complete.")
