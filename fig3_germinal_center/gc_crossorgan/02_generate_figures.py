"""Phase 2: Generate GC spatial detection figures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.gc_utils import (
    load_adata, load_metadata, nature_rcparams, apply_nature_style,
    PIXEL_SIZE_UM, GC_CELL_TYPE_PALETTE, compute_gc_boundary,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("Loading data...")
adata = load_adata()
metadata = load_metadata(OUTPUTS_DIR)
pixel_size_um = metadata['pixel_size_um']
print(f"Loaded: {adata.n_obs} cells, pixel_size={pixel_size_um}")

# Rebuild GC polygons with tuned eps
ORGANS = [('LN_A1_3', 'Lymph Node', 'LN', 300), ('APE_A9_3', 'Appendix', 'APE', 300)]
gc_polygons = {}
gc_registry = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_registry.csv'))

for sample_id, organ_label, organ_code, eps in ORGANS:
    mask = (
        (adata.obs['SAHA_Sample'] == sample_id) &
        (adata.obs['cell_types_insitutype'].isin(['GCBC', 'FDC']))
    )
    coords = adata.obsm['spatial'][mask.values]
    db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit_predict(coords)
    unique_labels = sorted(set(db) - {-1})
    gc_num = 0
    for label in unique_labels:
        cluster_coords = coords[db == label]
        if len(cluster_coords) < 50:
            continue
        gc_num += 1
        gc_id = f'{organ_code}-GC{gc_num}'
        gc_polygons[gc_id] = compute_gc_boundary(cluster_coords)
    print(f"  {organ_code}: {gc_num} GCs rebuilt")

# === eps calibration figure ===
print("Generating eps calibration figure...")
plt.rcParams.update(nature_rcparams())
fig_eps, axes_eps = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, (sample_id, organ_label, organ_code, eps) in zip(axes_eps, ORGANS):
    mask = (
        (adata.obs['SAHA_Sample'] == sample_id) &
        (adata.obs['cell_types_insitutype'].isin(['GCBC', 'FDC']))
    )
    coords = adata.obsm['spatial'][mask.values]
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    knn9 = np.sort(dists[:, 9])[::-1]
    ax.plot(knn9, lw=0.8, color='#333333')
    ax.axhline(eps, color='#C33334', lw=0.8, linestyle='--',
               label=f'eps={eps} px ({eps*pixel_size_um:.0f} um)')
    ax.set_xlabel('Points (sorted)')
    ax.set_ylabel('9-NN distance (px)')
    ax.set_title(organ_label)
    ax.legend(fontsize=6)
    apply_nature_style(ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '02_eps_calibration.png'), dpi=150)
plt.close()
print("Saved: 02_eps_calibration.png")

# === Main spatial figure ===
print("Generating main spatial figures...")
plt.rcParams.update(nature_rcparams())
actual_cell_types = sorted(adata.obs['cell_types_insitutype'].unique().tolist())
NATURE_2COL = 7.2

fig, axes = plt.subplots(1, 2, figsize=(NATURE_2COL, 3.6), constrained_layout=True)

for ax, (sample_id, organ_label, organ_code, _) in zip(axes, ORGANS):
    organ_mask = adata.obs['SAHA_Sample'] == sample_id
    coords = adata.obsm['spatial'][organ_mask.values]
    cell_types = adata.obs.loc[organ_mask, 'cell_types_insitutype'].values
    colors = [GC_CELL_TYPE_PALETTE.get(ct, '#AAAAAA') for ct in cell_types]

    ax.scatter(coords[:, 0] * pixel_size_um, coords[:, 1] * pixel_size_um,
               c=colors, s=1, linewidths=0, rasterized=True, zorder=1)

    organ_gc_ids = gc_registry.loc[gc_registry['organ'] == organ_code, 'gc_id'].tolist()
    for gc_id in organ_gc_ids:
        if gc_id not in gc_polygons:
            continue
        polygon = gc_polygons[gc_id]
        poly_um = np.array(polygon.exterior.coords) * pixel_size_um
        path = Path(poly_um)
        patch = PathPatch(path, facecolor='gray', alpha=0.15,
                          edgecolor='#555555', linestyle='--', linewidth=0.5, zorder=2)
        ax.add_patch(patch)
        cx = float(polygon.centroid.x) * pixel_size_um
        cy = float(polygon.centroid.y) * pixel_size_um
        ax.text(cx, cy, gc_id.split('-')[1], ha='center', va='center',
                fontsize=5, color='#333333', zorder=3)

    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(1, 'um', length_fraction=0.15, location='lower right',
                            frameon=False, color='black', font_properties={'size': 5})
        ax.add_artist(scalebar)
    except ImportError:
        pass

    ax.set_title(organ_label, fontsize=7)
    ax.set_xlabel('x (um)', fontsize=6)
    ax.set_ylabel('y (um)', fontsize=6)
    ax.set_aspect('equal')
    apply_nature_style(ax)

legend_handles = [mpatches.Patch(color=GC_CELL_TYPE_PALETTE.get(ct, '#AAAAAA'), label=ct)
                  for ct in actual_cell_types]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=5,
           frameon=False, bbox_to_anchor=(0.5, -0.18), title='Cell type', title_fontsize=6)

fig.savefig(os.path.join(FIGURES_DIR, '02_spatial_overview.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 02_spatial_overview.pdf")

# Individual PNGs
for sample_id, organ_label, organ_code, _ in ORGANS:
    fig_s, ax_s = plt.subplots(1, 1, figsize=(NATURE_2COL / 2, 3.6))
    plt.rcParams.update(nature_rcparams())
    organ_mask = adata.obs['SAHA_Sample'] == sample_id
    coords = adata.obsm['spatial'][organ_mask.values]
    cell_types = adata.obs.loc[organ_mask, 'cell_types_insitutype'].values
    colors = [GC_CELL_TYPE_PALETTE.get(ct, '#AAAAAA') for ct in cell_types]

    ax_s.scatter(coords[:, 0] * pixel_size_um, coords[:, 1] * pixel_size_um,
                 c=colors, s=1, linewidths=0, rasterized=True)

    organ_gc_ids = gc_registry.loc[gc_registry['organ'] == organ_code, 'gc_id'].tolist()
    for gc_id in organ_gc_ids:
        if gc_id not in gc_polygons:
            continue
        polygon = gc_polygons[gc_id]
        poly_um = np.array(polygon.exterior.coords) * pixel_size_um
        path = Path(poly_um)
        patch = PathPatch(path, facecolor='gray', alpha=0.15,
                          edgecolor='#555555', linestyle='--', linewidth=0.5)
        ax_s.add_patch(patch)

    ax_s.set_title(organ_label, fontsize=7)
    ax_s.set_xlabel('x (um)', fontsize=6)
    ax_s.set_ylabel('y (um)', fontsize=6)
    ax_s.set_aspect('equal')
    apply_nature_style(ax_s)

    png_path = os.path.join(FIGURES_DIR, f'02_spatial_overview_{organ_code}.png')
    fig_s.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig_s)
    print(f"Saved: {png_path}")

print("\nAll Phase 2 figures generated successfully.")
