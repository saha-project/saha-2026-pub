"""Validate schematic: multi-panel figure testing each hypothesis from the LN vs APE GC model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.gc_utils import (
    load_adata, compute_fractional_radius, nature_rcparams,
    apply_nature_style, PIXEL_SIZE_UM, GC_CELL_TYPE_PALETTE,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

print("Loading data...")
adata = load_adata()
obs = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs.index = obs.index.astype(str)
obs['ct'] = adata.obs['cell_types_insitutype'].values
obs['organ'] = adata.obs['SAHA_organcode'].values
obs['insitu'] = adata.obs['Insitutype_Labelled'].values
fr = compute_fractional_radius(obs)
obs['fr'] = fr

# Focus GCs
FOCUS = {'LN-GC4': 'Lymph Node GC', 'APE-GC2': 'Appendix GC'}

plt.rcParams.update(nature_rcparams())

# === 6-panel figure: 3 rows x 2 cols (LN | APE) ===
fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.5), constrained_layout=True)

for col, (gc_id, title) in enumerate(FOCUS.items()):
    gc = obs[obs['gc_id'] == gc_id]
    gc_pos = np.array([adata.obs_names.tolist().index(i) for i in gc.index])

    # --- Row 1: Spatial scatter colored by cell type ---
    ax = axes[0, col]
    coords = adata.obsm['spatial'][gc_pos] * PIXEL_SIZE_UM
    colors = [GC_CELL_TYPE_PALETTE.get(ct, '#AAA') for ct in gc['ct'].values]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors,
               s=4 if gc_id == 'LN-GC4' else 1.5, linewidths=0, rasterized=True)
    ax.set_title(f'{title}\n(n={len(gc):,})', fontsize=7)
    ax.set_aspect('equal')
    ax.set_xlabel('x (um)', fontsize=5)
    ax.set_ylabel('y (um)', fontsize=5)
    ax.text(-0.12, 1.05, chr(65 + col), transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    apply_nature_style(ax)

    # --- Row 2: DZ/LZ spatial distribution ---
    ax2 = axes[1, col]
    DZLZ_COLORS = {'DZ': '#2166AC', 'LZ': '#B2182B', 'other': '#EEEEEE'}

    if gc_id.startswith('APE'):
        # Use Insitutype labels
        dz_mask = gc['insitu'] == 'DZ.GC.cell'
        lz_mask = gc['insitu'] == 'LZ.GC.cell'
        other_mask = ~dz_mask & ~lz_mask
        cell_colors = np.array(['#EEEEEE'] * len(gc))
        cell_colors[dz_mask.values] = '#2166AC'
        cell_colors[lz_mask.values] = '#B2182B'

        n_dz = dz_mask.sum()
        n_lz = lz_mask.sum()
        ratio = n_dz / n_lz if n_lz > 0 else 0

        # Plot other first, then DZ/LZ on top
        ax2.scatter(coords[other_mask.values, 0], coords[other_mask.values, 1],
                    c='#EEEEEE', s=1, linewidths=0, rasterized=True, zorder=0)
        ax2.scatter(coords[dz_mask.values, 0], coords[dz_mask.values, 1],
                    c='#2166AC', s=2, linewidths=0, rasterized=True, zorder=1, label=f'DZ (n={n_dz})')
        ax2.scatter(coords[lz_mask.values, 0], coords[lz_mask.values, 1],
                    c='#B2182B', s=2, linewidths=0, rasterized=True, zorder=1, label=f'LZ (n={n_lz})')

        ax2.set_title(f'DZ/LZ (Insitutype)\nDZ/LZ={ratio:.2f}, ΔradPos={abs(gc.loc[dz_mask, "fr"].mean() - gc.loc[lz_mask, "fr"].mean()):.3f}',
                      fontsize=6)
        ax2.legend(fontsize=4, frameon=False, loc='lower right')
    else:
        # LN: show CXCR4 expression as proxy
        cxcr4_idx = list(adata.var_names).index('CXCR4')
        cxcr4_vals = adata.layers['normalized'][gc_pos, cxcr4_idx]
        if sp.issparse(cxcr4_vals):
            cxcr4_vals = np.asarray(cxcr4_vals.todense()).ravel()
        else:
            cxcr4_vals = np.asarray(cxcr4_vals).ravel()

        sc_obj = ax2.scatter(coords[:, 0], coords[:, 1], c=cxcr4_vals, cmap='Blues',
                            s=4, linewidths=0, rasterized=True, vmin=0)
        ax2.set_title(f'CXCR4 expression (DZ proxy)\n({(cxcr4_vals > 0).sum()}/{len(cxcr4_vals)} positive, {(cxcr4_vals > 0).mean()*100:.0f}%)',
                      fontsize=6)
        plt.colorbar(sc_obj, ax=ax2, shrink=0.5, pad=0.02).ax.tick_params(labelsize=4)

    ax2.set_aspect('equal')
    ax2.set_xlabel('x (um)', fontsize=5)
    ax2.set_ylabel('y (um)', fontsize=5)
    ax2.text(-0.12, 1.05, chr(67 + col), transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top')
    apply_nature_style(ax2)

    # --- Row 3: Radial composition (key cell types) ---
    ax3 = axes[2, col]
    KEY_TYPES = ['GCBC', 'FDC', 'PC', 'TC_CD4', 'NBC/MBC', 'TC_CD8']
    bins = np.linspace(0, 1, 6)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for ct in KEY_TYPES:
        ct_cells = gc[gc['ct'] == ct]
        if len(ct_cells) == 0:
            continue
        # Fraction of this cell type per bin
        hist, _ = np.histogram(ct_cells['fr'].dropna(), bins=bins)
        hist_all, _ = np.histogram(gc['fr'].dropna(), bins=bins)
        fracs = hist / hist_all.clip(min=1)
        ax3.plot(bin_centers, fracs, color=GC_CELL_TYPE_PALETTE.get(ct, '#AAA'),
                 lw=1.2, marker='o', markersize=2, label=ct)

    ax3.set_xlabel('Fractional radius (center → edge)', fontsize=6)
    ax3.set_ylabel('Fraction', fontsize=6)
    ax3.set_title('Radial composition', fontsize=7)
    ax3.legend(fontsize=4, ncol=2, frameon=False, loc='upper right')
    ax3.set_xticks(bin_centers)
    ax3.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=4, rotation=45, ha='right')
    ax3.text(-0.12, 1.05, chr(69 + col), transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top')
    apply_nature_style(ax3)

# Shared legend for row 1
ct_legend = [mpatches.Patch(color=GC_CELL_TYPE_PALETTE.get(ct, '#AAA'), label=ct)
             for ct in ['GCBC', 'FDC', 'NBC/MBC', 'PC', 'TC_CD4', 'TC_CD8', 'macro']]
fig.legend(handles=ct_legend, loc='lower center', ncol=7, fontsize=4,
           frameon=False, bbox_to_anchor=(0.5, -0.03))

fig.suptitle('Cross-organ GC architecture: Lymph Node vs Appendix', fontsize=8, y=1.01)

png_path = os.path.join(FIGURES_DIR, 'schematic_validation.png')
pdf_path = os.path.join(FIGURES_DIR, 'schematic_validation.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# === Supplementary: Ig isotype comparison ===
print("\nGenerating Ig isotype comparison...")
fig_ig, ax_ig = plt.subplots(1, 1, figsize=(4.0, 3.0))
plt.rcParams.update(nature_rcparams())

ig_genes = ['IGHA1', 'IGHG1', 'IGHG2', 'IGHD', 'IGHM']
ig_present = [g for g in ig_genes if g in adata.var_names]

ig_data = []
for gc_id in ['LN-GC4', 'APE-GC2']:
    gc_mask = obs['gc_id'] == gc_id
    gc_pos_ig = np.array([adata.obs_names.tolist().index(i) for i in obs.index[gc_mask]])
    for gene in ig_present:
        idx = list(adata.var_names).index(gene)
        vals = adata.X[gc_pos_ig, idx]
        if sp.issparse(vals):
            vals = vals.toarray().ravel()
        ig_data.append({
            'gc_id': gc_id, 'gene': gene,
            'mean': float(vals.mean()),
            'pct_pos': float((vals > 0).mean()) * 100,
        })

ig_df = pd.DataFrame(ig_data)

x = np.arange(len(ig_present))
w = 0.35
for i, gc_id in enumerate(['LN-GC4', 'APE-GC2']):
    gc_ig = ig_df[ig_df['gc_id'] == gc_id].set_index('gene').reindex(ig_present)
    color = '#2A6365' if gc_id == 'LN-GC4' else '#8B0000'
    bars = ax_ig.bar(x + i * w, gc_ig['mean'].values, w, color=color, alpha=0.8, label=gc_id)
    # Add pct labels
    for j, (bar, pct) in enumerate(zip(bars, gc_ig['pct_pos'].values)):
        ax_ig.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{pct:.0f}%', ha='center', va='bottom', fontsize=3.5)

ax_ig.set_xticks(x + w/2)
ax_ig.set_xticklabels(ig_present, fontsize=6)
ax_ig.set_ylabel('Mean raw counts', fontsize=6)
ax_ig.set_title('Immunoglobulin isotype expression', fontsize=7)
ax_ig.legend(fontsize=5, frameon=False)
apply_nature_style(ax_ig)

ig_png = os.path.join(FIGURES_DIR, 'ig_isotype_comparison.png')
ig_pdf = os.path.join(FIGURES_DIR, 'ig_isotype_comparison.pdf')
fig_ig.savefig(ig_png, dpi=300, bbox_inches='tight')
fig_ig.savefig(ig_pdf, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {ig_png}")
print(f"Saved: {ig_pdf}")

print("\nSchematic validation complete.")
