"""Phase 3 v2: DZ/LZ polarity — exclude small LN GCs, add pooled ratio comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.gc_utils import (
    load_adata, classify_dzlz_by_expression, nature_rcparams,
    apply_nature_style, PIXEL_SIZE_UM,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
EXCLUDE_GCS = {'LN-GC1', 'LN-GC2'}

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
gc_registry = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_registry.csv'))

adata.obs['gc_id'] = obs_assign['gc_id'].values
adata.obs['gc_member'] = obs_assign['gc_member'].values

# --- Classify DZ/LZ ---
print("Classifying DZ/LZ...")
dzlz_labels = pd.Series('unassigned', index=adata.obs_names)

# APE: Insitutype_Labelled
ape_gc_idx = adata.obs_names[
    (adata.obs['gc_member'] == True) & (adata.obs['SAHA_Sample'] == 'APE_A9_3')
]
ape_insitu = adata.obs.loc[ape_gc_idx, 'Insitutype_Labelled']
dzlz_labels.loc[ape_gc_idx[ape_insitu == 'DZ.GC.cell']] = 'DZ'
dzlz_labels.loc[ape_gc_idx[ape_insitu == 'LZ.GC.cell']] = 'LZ'
dzlz_labels.loc[ape_gc_idx[dzlz_labels.loc[ape_gc_idx] == 'unassigned']] = 'other_GC'

# LN: expression-based on GCBC
ln_gc_gcbc_mask = (
    (adata.obs['gc_member'] == True) &
    (adata.obs['SAHA_Sample'] == 'LN_A1_3') &
    (adata.obs['cell_types_insitutype'] == 'GCBC')
)
if ln_gc_gcbc_mask.sum() > 0:
    ln_dzlz = classify_dzlz_by_expression(adata, ln_gc_gcbc_mask)
    dzlz_labels.loc[ln_dzlz.index] = ln_dzlz.values

ln_gc_other = (
    (adata.obs['gc_member'] == True) &
    (adata.obs['SAHA_Sample'] == 'LN_A1_3') &
    (dzlz_labels == 'unassigned')
)
dzlz_labels.loc[ln_gc_other] = 'other_GC'
adata.obs['dzlz'] = dzlz_labels.values

# --- Compute ratios (excluding small GCs) ---
print("\nDZ/LZ ratios (v2, excluding small LN GCs):")
ratios = []
for _, row in gc_registry.iterrows():
    gc_id = row['gc_id']
    if gc_id in EXCLUDE_GCS:
        print(f"  {gc_id}: EXCLUDED (too small)")
        continue
    gc_cells = adata.obs[adata.obs['gc_id'] == gc_id]
    n_dz = (gc_cells['dzlz'] == 'DZ').sum()
    n_lz = (gc_cells['dzlz'] == 'LZ').sum()
    ratio = n_dz / n_lz if n_lz > 0 else np.nan
    ratios.append({'gc_id': gc_id, 'organ': row['organ'], 'n_dz': n_dz, 'n_lz': n_lz, 'dz_lz_ratio': ratio})
    r_str = f"{ratio:.2f}" if not np.isnan(ratio) else "NA"
    print(f"  {gc_id}: DZ={n_dz}, LZ={n_lz}, ratio={r_str}")

ratios_df = pd.DataFrame(ratios)

# Pooled ratios per organ
print("\nPooled DZ/LZ ratios:")
for organ in ['LN', 'APE']:
    org_df = ratios_df[ratios_df['organ'] == organ]
    total_dz = org_df['n_dz'].sum()
    total_lz = org_df['n_lz'].sum()
    pooled = total_dz / total_lz if total_lz > 0 else np.nan
    print(f"  {organ}: DZ={total_dz}, LZ={total_lz}, pooled ratio={pooled:.2f}")

# --- Figure: Row 1: per-GC mini panels, Row 2: ratio bar chart ---
print("\nGenerating v2 figure...")
plt.rcParams.update(nature_rcparams())
DZLZ_COLORS = {'DZ': '#2166AC', 'LZ': '#B2182B', 'intermediate': '#999999', 'other_GC': '#DDDDDD'}

kept_gc_ids = sorted(ratios_df['gc_id'].tolist())
n_gcs = len(kept_gc_ids)
ncols = min(n_gcs, 5)
nrows_panels = (n_gcs + ncols - 1) // ncols

fig = plt.figure(figsize=(7.2, nrows_panels * 2.0 + 2.5), constrained_layout=True)
gs = fig.add_gridspec(nrows_panels + 1, ncols, height_ratios=[1] * nrows_panels + [1.2])

# Mini panels
for idx, gc_id in enumerate(kept_gc_ids):
    r, c = divmod(idx, ncols)
    ax = fig.add_subplot(gs[r, c])

    gc_mask = adata.obs['gc_id'] == gc_id
    coords = adata.obsm['spatial'][gc_mask.values] * PIXEL_SIZE_UM
    colors = [DZLZ_COLORS.get(l, '#DDD') for l in adata.obs.loc[gc_mask, 'dzlz'].values]

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=1, linewidths=0, rasterized=True)

    organ = ratios_df.loc[ratios_df['gc_id'] == gc_id, 'organ'].values[0]
    r_row = ratios_df[ratios_df['gc_id'] == gc_id].iloc[0]
    r_str = f"DZ/LZ={r_row['dz_lz_ratio']:.2f}" if not np.isnan(r_row['dz_lz_ratio']) else "DZ/LZ=NA"
    method = "Insitutype" if organ == 'APE' else "CXCR4/CD83"
    ax.set_title(f"{gc_id} ({method})\n{r_str}", fontsize=6)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=4)
    apply_nature_style(ax)

# Hide unused panel slots
for idx in range(n_gcs, nrows_panels * ncols):
    r, c = divmod(idx, ncols)
    fig.add_subplot(gs[r, c]).set_visible(False)

# Bottom: ratio bar chart comparison
ax_ratio = fig.add_subplot(gs[nrows_panels, :3])
gc_ids_sorted = ratios_df.sort_values(['organ', 'gc_id'])['gc_id'].tolist()
x_pos = range(len(gc_ids_sorted))
bar_colors = ['#2166AC' if r < 1.0 else '#B2182B' for r in ratios_df.set_index('gc_id').loc[gc_ids_sorted, 'dz_lz_ratio']]

ax_ratio.bar(x_pos, ratios_df.set_index('gc_id').loc[gc_ids_sorted, 'dz_lz_ratio'].values,
             color=bar_colors, width=0.6, edgecolor='#333', linewidth=0.3)
ax_ratio.axhline(1.0, color='#333', lw=0.5, ls='--', zorder=0)
ax_ratio.set_xticks(x_pos)
ax_ratio.set_xticklabels(gc_ids_sorted, fontsize=5, rotation=45, ha='right')
ax_ratio.set_ylabel('DZ/LZ ratio', fontsize=6)
ax_ratio.set_title('DZ/LZ ratio per GC (>1 = DZ-enriched, <1 = LZ-enriched)', fontsize=7)
apply_nature_style(ax_ratio)

# Legend
legend_handles = [mpatches.Patch(color=DZLZ_COLORS[k], label=k) for k in ['DZ', 'LZ', 'intermediate', 'other_GC']]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=5,
           frameon=False, bbox_to_anchor=(0.5, -0.03))

png_path = os.path.join(FIGURES_DIR, '03_dzlz_polarity_v2.png')
pdf_path = os.path.join(FIGURES_DIR, '03_dzlz_polarity_v2.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("DZ/LZ polarity v2 complete.")
