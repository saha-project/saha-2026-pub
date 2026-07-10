"""
Phase 1, Plan 02: Split per-organ h5ads, filter LR databases, build organ cell type table.

Requirements: DATA-03, DATA-04, DATA-05
Input: analysis_ready.h5ad (115,293 x 1,207) — must exist from Plan 01
Outputs:
  per_organ/APE.h5ad   (24,531 cells)
  per_organ/COL.h5ad   (49,498 cells, COL_A1_5 + COL_A1_7 pooled)
  per_organ/ILE.h5ad   (41,264 cells)
  lr_databases/liana_consensus_filtered.csv  (566 pairs)
  organ_celltype_table.csv
Note: lr_databases/cellchatdb_filtered.csv is produced by filter_cellchatdb.R (called at end)
"""

import os
import subprocess
import anndata as ad
import pandas as pd
import numpy as np

WORKDIR = '/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI'
ANALYSIS_READY = os.path.join(WORKDIR, 'analysis_ready.h5ad')
PER_ORGAN_DIR = os.path.join(WORKDIR, 'per_organ')
LR_DIR = os.path.join(WORKDIR, 'lr_databases')
UTAG_CSV = os.path.join(WORKDIR, 'utag_metadata.csv')

os.makedirs(PER_ORGAN_DIR, exist_ok=True)
os.makedirs(LR_DIR, exist_ok=True)

print("=== Phase 1 Plan 02: Split and Filter ===")

# --- SECTION 1: Per-organ split (DATA-04) ---
print("\n[1/3] Splitting per-organ h5ads...")
adata = ad.read_h5ad(ANALYSIS_READY)
print(f"  Loaded analysis_ready.h5ad: {adata.shape}")

EXPECTED = {'APE': 24531, 'COL': 49498, 'ILE': 41264}
COL_NOTE = (
    "Two sections (COL_A1_5, COL_A1_7) pooled as biological replicates. "
    "Spatial CCI computed within section coordinate space; inter-section pairs "
    "exceed interaction.range=250 µm and are not scored."
)

for organ, expected_count in EXPECTED.items():
    mask = adata.obs['SAHA_organcode'] == organ
    adata_organ = adata[mask].copy()

    # Assertions
    assert adata_organ.n_obs == expected_count, (
        f"{organ}: expected {expected_count} cells, got {adata_organ.n_obs}"
    )
    assert 'spatial_um' in adata_organ.obsm, f"{organ}: obsm['spatial_um'] missing"

    # Annotate uns for Phase 2 scripts
    adata_organ.uns['organ'] = organ
    adata_organ.uns['intended_layer_cellchat'] = 'raw'
    adata_organ.uns['intended_layer_liana'] = 'normalized_all'
    adata_organ.uns['pixel_size_um'] = 0.1203
    if organ == 'COL':
        adata_organ.uns['col_note'] = COL_NOTE

    out_path = os.path.join(PER_ORGAN_DIR, f'{organ}.h5ad')
    adata_organ.write_h5ad(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {organ}: {adata_organ.n_obs} cells -> {out_path} ({size_mb:.1f} MB)")

print("  Per-organ split complete.")

# --- SECTION 2: LIANA+ consensus resource filter (DATA-03) ---
print("\n[2/3] Filtering LIANA+ consensus resource to panel genes...")
from liana.resource import select_resource

panel_genes = set(adata.var_names)
print(f"  Panel size: {len(panel_genes)} genes")

consensus = select_resource('consensus')
print(f"  Consensus resource total pairs: {len(consensus)}")  # Expected: 4624

consensus_filtered = consensus[
    consensus['ligand'].isin(panel_genes) &
    consensus['receptor'].isin(panel_genes)
].copy().reset_index(drop=True)

assert len(consensus_filtered) == 566, (
    f"Expected 566 LIANA+ pairs after filter, got {len(consensus_filtered)}"
)
liana_out = os.path.join(LR_DIR, 'liana_consensus_filtered.csv')
consensus_filtered.to_csv(liana_out, index=False)
print(f"  LIANA+ retained: {len(consensus_filtered)} pairs -> {liana_out}")

# --- SECTION 3: Organ cell type presence table (DATA-05) ---
print("\n[3/3] Building organ cell type presence table...")
utag = pd.read_csv(UTAG_CSV)
ct_counts = (
    utag.groupby(['SAHA_organcode', 'new_clusters'])
    .size()
    .unstack(fill_value=0)
)
# ct_counts: index=organ (APE/COL/ILE), columns=cell type (21 types)

MIN_CELLS = 50
shared = (ct_counts >= MIN_CELLS).all(axis=0)
# Transpose: rows=21 cell types, cols=3 organs (APE, COL, ILE)
# Shape (21, 3) — test expects shape[0] in (3,21) and shape[1] in (3,21)
ct_table = ct_counts.T.copy()  # shape: (21 cell types x 3 organs)

table_out = os.path.join(WORKDIR, 'organ_celltype_table.csv')
ct_table.to_csv(table_out)
print(f"  Organ cell type table -> {table_out}")
print(f"  Shape: {ct_table.shape}")
print(f"  Shared cell types (>=50 in all 3 organs): {shared[shared].index.tolist()}")

print("\n=== All Python steps complete ===")
print("Next: Run filter_cellchatdb.R to produce lr_databases/cellchatdb_filtered.csv")
