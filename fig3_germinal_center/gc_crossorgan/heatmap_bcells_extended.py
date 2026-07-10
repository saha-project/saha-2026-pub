"""Extended heatmap: B cell lineage x many genes, z-scored per gene to avoid APE count bias."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

# Gene list ordered: LN-enriched first, then APE-enriched (balanced representation)
# Genes selected from DE results and known biology
GENE_SECTIONS = {
    'LN-enriched': ['IGHG1', 'IGHG2', 'IGKC', 'MALAT1', 'HLA-DQA1', 'HLA-DQB1/2',
                     'CD83', 'CD37', 'IL4R', 'BST2', 'BASP1', 'IGHD',
                     'CD74', 'HLA-DRA'],
    'Shared': ['CXCR5', 'CXCL13', 'IGHM', 'BCL2', 'MYC', 'IRF4',
               'CD38', 'MS4A1', 'CD19', 'CD79A', 'B2M'],
    'APE-enriched': ['CXCR4', 'IGHA1', 'PCNA', 'MKI67', 'TOP2A', 'STMN1',
                      'H4C3', 'H2AZ1', 'HMGN2', 'TCL1A', 'FAU', 'ENO1',
                      'PFN1', 'MIF', 'ARHGDIB'],
}

B_CELL_TYPES = ['GCBC', 'NBC/MBC', 'FDC', 'PC']
ORGANS = ['LN', 'APE']

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)
adata.obs['gc_id'] = obs_assign['gc_id'].values

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Flatten gene list, deduplicate, filter to panel
all_genes = []
section_boundaries = []
section_labels = []
seen = set()
for section, genes in GENE_SECTIONS.items():
    section_start = len(all_genes)
    for g in genes:
        if g not in seen and g in adata.var_names:
            all_genes.append(g)
            seen.add(g)
    if len(all_genes) > section_start:
        section_boundaries.append(section_start)
        section_labels.append(section)

print(f"Genes: {len(all_genes)}")

# Build groups: organ x cell type
groups = []
for organ in ORGANS:
    for ct in B_CELL_TYPES:
        gc_id = 'LN-GC4' if organ == 'LN' else 'APE-GC2'
        mask = (adata.obs['gc_id'] == gc_id) & (adata.obs['cell_types_insitutype'] == ct)
        n = mask.sum()
        if n >= 5:
            groups.append((organ, ct, gc_id, mask, n))
            print(f"  {organ}: {ct} n={n}")

# Compute mean expression matrix (groups x genes)
mean_matrix = np.zeros((len(groups), len(all_genes)))
for i, (organ, ct, gc_id, mask, n) in enumerate(groups):
    X = adata[mask][:, all_genes].X
    if sp.issparse(X):
        X = X.toarray()
    mean_matrix[i, :] = X.mean(axis=0)

# Z-score PER GENE across groups — this normalizes away the APE count bias
mean_z = np.zeros_like(mean_matrix)
for j in range(mean_matrix.shape[1]):
    col = mean_matrix[:, j]
    m, s = col.mean(), col.std()
    mean_z[:, j] = (col - m) / s if s > 0 else 0

# Clip
mean_z = np.clip(mean_z, -2.5, 2.5)

# === Figure: compressed horizontal heatmap ===
print("\nGenerating heatmap...")
plt.rcParams.update(nature_rcparams())

n_genes = len(all_genes)
n_groups = len(groups)

fig_width = 7.2
fig_height = fig_width / 8  # 1:8 ratio
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

im = ax.imshow(mean_z, cmap='RdBu_r', aspect='auto', vmin=-2.5, vmax=2.5)

# X axis: genes
ax.set_xticks(range(n_genes))
ax.set_xticklabels(all_genes, fontsize=4, rotation=60, ha='right')

# Y axis: organ: cell type (n=)
ylabels = [f'{organ}: {ct} (n={n})' for organ, ct, _, _, n in groups]
ax.set_yticks(range(n_groups))
ax.set_yticklabels(ylabels, fontsize=5.5)

# Color y labels by organ
for i, label in enumerate(ax.get_yticklabels()):
    organ = groups[i][0]
    label.set_color('#2A6365' if organ == 'LN' else '#8B0000')

# Organ separator
n_ln = sum(1 for g in groups if g[0] == 'LN')
ax.axhline(n_ln - 0.5, color='black', lw=1)

# Section separators (vertical)
for boundary in section_boundaries[1:]:
    ax.axvline(boundary - 0.5, color='#333', lw=0.5, ls='-')

# Section labels at top
for idx, (boundary, label) in enumerate(zip(section_boundaries, section_labels)):
    next_boundary = section_boundaries[idx + 1] if idx + 1 < len(section_boundaries) else n_genes
    mid = (boundary + next_boundary - 1) / 2
    ax.text(mid, -0.8, label, ha='center', va='bottom', fontsize=3.5,
            fontstyle='italic', color='#555')

ax.set_title('B cell lineage gene expression: LN-GC4 vs APE-GC2 (z-scored per gene)', fontsize=7)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=15)
cbar.set_label('Z-score', fontsize=5)
cbar.ax.tick_params(labelsize=4)

apply_nature_style(ax)

png_path = os.path.join(FIGURES_DIR, 'heatmap_bcell_extended.png')
pdf_path = os.path.join(FIGURES_DIR, 'heatmap_bcell_extended.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Extended heatmap complete.")
