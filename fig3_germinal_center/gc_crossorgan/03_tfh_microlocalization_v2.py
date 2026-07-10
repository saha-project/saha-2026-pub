"""Phase 3 v2: Tfh microlocalization — exclude small LN GCs, pooled comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.gc_utils import (
    load_adata, compute_fractional_radius, nature_rcparams,
    apply_nature_style, GC_CELL_TYPE_PALETTE, PIXEL_SIZE_UM,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
EXCLUDE_GCS = {'LN-GC1', 'LN-GC2'}

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
gc_registry = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_registry.csv'))

frac_radius = compute_fractional_radius(obs_assign)
obs_assign['frac_radius'] = frac_radius
obs_assign['cell_types_insitutype'] = adata.obs['cell_types_insitutype'].values
obs_assign['SAHA_Sample'] = adata.obs['SAHA_Sample'].values

# Filter: TC_CD4 in GCs, exclude small LN GCs
tc_cd4_gc = obs_assign[
    (obs_assign['gc_member'] == True) &
    (obs_assign['cell_types_insitutype'] == 'TC_CD4') &
    (~obs_assign['gc_id'].isin(EXCLUDE_GCS))
].copy()

gc_organ_map = dict(zip(gc_registry['gc_id'], gc_registry['organ']))
tc_cd4_gc['organ'] = tc_cd4_gc['gc_id'].map(gc_organ_map)

print(f"TC_CD4 in GCs (v2, excluding small LN GCs): {len(tc_cd4_gc)}")
print(f"  LN: {(tc_cd4_gc['organ'] == 'LN').sum()}")
print(f"  APE: {(tc_cd4_gc['organ'] == 'APE').sum()}")

# Assign compartments
def assign_compartment(frac_r):
    if pd.isna(frac_r):
        return 'unknown'
    if frac_r <= 1/3:
        return 'DZ-proximal'
    elif frac_r <= 2/3:
        return 'LZ-proximal'
    else:
        return 'edge'

tc_cd4_gc['compartment'] = tc_cd4_gc['frac_radius'].apply(assign_compartment)

COMPARTMENT_ORDER = ['DZ-proximal', 'LZ-proximal', 'edge']
COMP_COLORS = {'DZ-proximal': '#2166AC', 'LZ-proximal': '#B2182B', 'edge': '#999999'}
ORGANS = [('LN', 'Lymph Node'), ('APE', 'Appendix')]

print("\nCompartment distribution (v2):")
for organ_code, organ_label in ORGANS:
    organ_tc = tc_cd4_gc[tc_cd4_gc['organ'] == organ_code]
    total = len(organ_tc)
    print(f"\n{organ_label} (n={total} TC_CD4):")
    for comp in COMPARTMENT_ORDER:
        n = (organ_tc['compartment'] == comp).sum()
        print(f"  {comp}: {n} ({n/total:.1%})" if total > 0 else f"  {comp}: 0")

# --- Figure ---
print("\nGenerating v2 figure...")
plt.rcParams.update(nature_rcparams())

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)

# Row 1: Bar charts (LN | APE)
for col_idx, (organ_code, organ_label) in enumerate(ORGANS):
    ax = axes[0, col_idx]
    organ_tc = tc_cd4_gc[tc_cd4_gc['organ'] == organ_code]
    total = len(organ_tc)
    fracs = []
    counts = []
    for comp in COMPARTMENT_ORDER:
        n = (organ_tc['compartment'] == comp).sum()
        fracs.append(n / total if total > 0 else 0)
        counts.append(n)

    bars = ax.bar(range(3), fracs, color=[COMP_COLORS[c] for c in COMPARTMENT_ORDER], width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(COMPARTMENT_ORDER, fontsize=5, rotation=30, ha='right')
    ax.set_ylabel('Fraction of TC_CD4', fontsize=6)
    note = f'(n={total}, LN-GC1/2 excluded)' if organ_code == 'LN' else f'(n={total})'
    ax.set_title(f'{organ_label} {note}', fontsize=7)
    ax.set_ylim(0, max(fracs) * 1.3 if max(fracs) > 0 else 1)

    for bar, f, cnt in zip(bars, fracs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                str(cnt), ha='center', va='bottom', fontsize=5)
    apply_nature_style(ax)

# Row 2: Spatial scatter — TC_CD4 highlighted on GC background
for col_idx, (organ_code, organ_label) in enumerate(ORGANS):
    ax = axes[1, col_idx]
    sample_id = 'LN_A1_3' if organ_code == 'LN' else 'APE_A9_3'

    # Background: all GC cells (excluding small GCs) in light gray
    bg_mask = (
        (obs_assign['gc_member'] == True) &
        (obs_assign['SAHA_Sample'] == sample_id) &
        (~obs_assign['gc_id'].isin(EXCLUDE_GCS))
    )
    bg_idx = obs_assign.index[bg_mask]
    # Use integer position indexing for spatial coords
    obs_name_list = adata.obs_names.tolist()
    bg_positions = [obs_name_list.index(i) for i in bg_idx[:3000]]
    bg_coords = adata.obsm['spatial'][bg_positions] * PIXEL_SIZE_UM
    ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c='#EEEEEE', s=0.3, linewidths=0, rasterized=True, zorder=0)

    # TC_CD4 colored by compartment
    organ_tc = tc_cd4_gc[tc_cd4_gc['organ'] == organ_code]
    tc_positions = [obs_name_list.index(i) for i in organ_tc.index]
    tc_coords = adata.obsm['spatial'][tc_positions] * PIXEL_SIZE_UM
    tc_colors = [COMP_COLORS.get(c, '#999') for c in organ_tc['compartment'].values]
    ax.scatter(tc_coords[:, 0], tc_coords[:, 1], c=tc_colors, s=3, linewidths=0, rasterized=True, zorder=1)

    ax.set_title(f'{organ_label} — TC_CD4 in GCs', fontsize=7)
    ax.set_xlabel('x (um)', fontsize=5)
    ax.set_ylabel('y (um)', fontsize=5)
    ax.set_aspect('equal')
    apply_nature_style(ax)

legend_handles = [mpatches.Patch(color=COMP_COLORS[c], label=c) for c in COMPARTMENT_ORDER]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=5,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

png_path = os.path.join(FIGURES_DIR, '03_tfh_microlocalization_v2.png')
pdf_path = os.path.join(FIGURES_DIR, '03_tfh_microlocalization_v2.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Tfh microlocalization v2 complete.")
