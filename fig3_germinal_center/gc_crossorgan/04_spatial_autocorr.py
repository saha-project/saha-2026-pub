"""Phase 4 Plan 03: Moran's I spatial autocorrelation for all 889 genes per organ."""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style, PIXEL_SIZE_UM

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

GC_MARKERS = ['CXCR4', 'CXCR5', 'CD83', 'CXCL13', 'IRF4', 'MYC', 'BCL2', 'CD38']
ORGANS = [('LN_A1_3', 'LN', 'Lymph Node'), ('APE_A9_3', 'APE', 'Appendix')]

print("Loading data...")
adata = load_adata()

results = {}
for sample_id, organ_code, organ_label in ORGANS:
    print(f"\n=== {organ_label} ===")
    organ_mask = adata.obs['SAHA_Sample'] == sample_id
    adata_organ = adata[organ_mask].copy()
    print(f"  Cells: {adata_organ.n_obs}")

    # Log-normalize for spatial autocorrelation
    sc.pp.normalize_total(adata_organ, target_sum=1e4)
    sc.pp.log1p(adata_organ)

    # Build spatial neighbors graph
    # Use obsm['spatial'] (global pixel coords)
    print("  Building spatial neighbor graph...")
    sq.gr.spatial_neighbors(adata_organ, coord_type='generic', spatial_key='spatial',
                            n_neighs=10)

    # Compute Moran's I for all genes
    print("  Computing Moran's I for all 889 genes...")
    sq.gr.spatial_autocorr(adata_organ, mode='moran', genes=adata_organ.var_names.tolist(),
                           n_perms=None, n_jobs=1)

    # Extract results — squidpy stores in adata.uns
    # Check for both possible keys
    morans_key = None
    for key in ['moranI', 'spatial_autocorr']:
        if key in adata_organ.uns:
            morans_key = key
            break

    if morans_key is None:
        print(f"  WARNING: Could not find Moran's I results in adata.uns. Keys: {list(adata_organ.uns.keys())}")
        continue

    morans_df = adata_organ.uns[morans_key].copy()
    morans_df['gene'] = morans_df.index
    morans_df = morans_df.reset_index(drop=True)

    # Standardize column names
    if 'I' in morans_df.columns:
        morans_df = morans_df.rename(columns={'I': 'morans_i', 'pval_norm': 'pval', 'var_norm': 'var'})
    elif 'moranI' in morans_df.columns:
        morans_df = morans_df.rename(columns={'moranI': 'morans_i'})

    morans_df = morans_df.sort_values('morans_i', ascending=False)

    csv_path = os.path.join(OUTPUTS_DIR, f'morans_i_{organ_code}.csv')
    morans_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    results[organ_code] = morans_df

    # Print top genes
    print(f"\n  Top 15 spatially autocorrelated genes ({organ_code}):")
    for _, r in morans_df.head(15).iterrows():
        marker = " ***" if r['gene'] in GC_MARKERS else ""
        print(f"    {r['gene']:12s} I={r['morans_i']:.4f}{marker}")

    # Print GC markers specifically
    print(f"\n  GC markers Moran's I ({organ_code}):")
    for gene in GC_MARKERS:
        row = morans_df[morans_df['gene'] == gene]
        if len(row) > 0:
            r = row.iloc[0]
            rank = morans_df.index.tolist().index(r.name) + 1 if r.name in morans_df.index else '?'
            print(f"    {gene:12s} I={r['morans_i']:.4f} (rank {rank}/{len(morans_df)})")
        else:
            print(f"    {gene:12s} NOT IN PANEL")

# === Cross-organ comparison figure ===
if len(results) == 2:
    print("\nGenerating cross-organ Moran's I comparison...")
    plt.rcParams.update(nature_rcparams())

    ln_df = results['LN'].set_index('gene')
    ape_df = results['APE'].set_index('gene')

    # Merge on gene
    shared_genes = sorted(set(ln_df.index) & set(ape_df.index))
    merged = pd.DataFrame({
        'gene': shared_genes,
        'morans_i_LN': [ln_df.loc[g, 'morans_i'] for g in shared_genes],
        'morans_i_APE': [ape_df.loc[g, 'morans_i'] for g in shared_genes],
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)

    # Panel A: Scatter LN vs APE Moran's I
    ax = axes[0]
    is_gc = merged['gene'].isin(GC_MARKERS)
    ax.scatter(merged.loc[~is_gc, 'morans_i_LN'], merged.loc[~is_gc, 'morans_i_APE'],
               c='#CCCCCC', s=3, linewidths=0, alpha=0.5, rasterized=True, label='Other')
    ax.scatter(merged.loc[is_gc, 'morans_i_LN'], merged.loc[is_gc, 'morans_i_APE'],
               c='#C33334', s=15, linewidths=0.3, edgecolors='black', zorder=5, label='GC markers')

    # Label GC markers
    for _, r in merged[is_gc].iterrows():
        ax.annotate(r['gene'], (r['morans_i_LN'], r['morans_i_APE']),
                     fontsize=4, ha='left', va='bottom',
                     textcoords='offset points', xytext=(3, 2))

    # Diagonal
    lim = max(merged['morans_i_LN'].max(), merged['morans_i_APE'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=0.3, alpha=0.5)

    ax.set_xlabel("Moran's I (Lymph Node)", fontsize=6)
    ax.set_ylabel("Moran's I (Appendix)", fontsize=6)
    ax.set_title("Spatial autocorrelation: LN vs APE", fontsize=7)
    ax.legend(fontsize=5, frameon=False)
    apply_nature_style(ax)

    # Panel B: Top organ-specific genes (biggest difference in Moran's I)
    ax2 = axes[1]
    merged['diff'] = merged['morans_i_APE'] - merged['morans_i_LN']
    merged_sorted = merged.sort_values('diff')

    # Top 10 LN-enriched + top 10 APE-enriched
    top_ln = merged_sorted.head(10)
    top_ape = merged_sorted.tail(10)
    top_diff = pd.concat([top_ln, top_ape])

    colors_bar = ['#2A6365' if d < 0 else '#8B0000' for d in top_diff['diff']]
    y_pos = range(len(top_diff))
    ax2.barh(y_pos, top_diff['diff'].values, color=colors_bar, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_diff['gene'].values, fontsize=4)
    ax2.set_xlabel("ΔMoran's I (APE − LN)", fontsize=6)
    ax2.set_title("Organ-specific spatial patterning", fontsize=7)
    ax2.axvline(0, color='#333', lw=0.3)

    # Mark GC markers
    for i, gene in enumerate(top_diff['gene']):
        if gene in GC_MARKERS:
            ax2.get_yticklabels()[i].set_fontweight('bold')

    apply_nature_style(ax2)

    png_path = os.path.join(FIGURES_DIR, '04_morans_comparison.png')
    pdf_path = os.path.join(FIGURES_DIR, '04_morans_comparison.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

print("\nMoran's I spatial autocorrelation complete.")
