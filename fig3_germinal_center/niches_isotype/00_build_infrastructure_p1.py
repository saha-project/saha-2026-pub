"""
Phase 1, Plan 01: Merge uTAG metadata into analysis-ready h5ad.

Requirements: DATA-01, DATA-02, OUT-04
Input (READ-ONLY): SAHA_GI_RNA_subsetted.h5ad (125,577 x 1,207)
Input: utag_metadata.csv (115,293 x 5)
Output: analysis_ready.h5ad (115,293 x 1,207)
"""

import os
import anndata as ad
import pandas as pd
import numpy as np

SOURCE_H5AD = '/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Data/preprocessed/h5ad/RNA/SAHA_GI_RNA_subsetted.h5ad'
WORKDIR = '/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI'
UTAG_CSV = os.path.join(WORKDIR, 'utag_metadata.csv')
OUT_H5AD = os.path.join(WORKDIR, 'analysis_ready.h5ad')
PIXEL_SIZE_UM = 0.1203  # NanoString CosMx: 120nm pixel, 22.78x water-immersion objective
                         # Verified empirically from Area.um2/Area ratio; NOT 0.18 (wrong objective)

print("=== Phase 1 Plan 01: Merge uTAG Metadata ===")

# 1. Load source h5ad (read-only; do not modify)
print(f"Loading source h5ad: {SOURCE_H5AD}")
adata = ad.read_h5ad(SOURCE_H5AD)
print(f"  Source shape: {adata.shape}")  # Expected: (125577, 1207)

# 2. Load uTAG metadata
print(f"Loading utag metadata: {UTAG_CSV}")
utag = pd.read_csv(UTAG_CSV, index_col='barcode')
print(f"  uTAG shape: {utag.shape}")  # Expected: (115293, 4)
print(f"  uTAG columns: {list(utag.columns)}")

# 3. Barcode spot-check: verify 10 random barcodes match format
print("\n--- Barcode spot-check (10 random barcodes) ---")
sample_barcodes = utag.index[:10].tolist()
for bc in sample_barcodes:
    in_adata = bc in adata.obs.index
    print(f"  {bc}  ->  in adata.obs: {in_adata}")

# 4. Inner join: keeps only 115,293 cells present in utag
# (this automatically drops 10,284 Low_quality cells absent from utag_metadata.csv)
print("\nPerforming inner join (drops Low_quality cells)...")
keep_mask = adata.obs.index.isin(utag.index)
adata_merged = adata[keep_mask].copy()
print(f"  Cells after inner join: {adata_merged.n_obs}")  # Expected: 115293

# 5. Add uTAG annotation columns
for col in ['new_clusters', 'UTAG_niches', 'UTAG_names', 'SAHA_organcode']:
    adata_merged.obs[col] = utag.loc[adata_merged.obs.index, col].values

# 6. Assertions (DATA-01, DATA-02)
assert adata_merged.n_obs == 115293, f"Cell count mismatch: expected 115293, got {adata_merged.n_obs}"
assert adata_merged.obs['new_clusters'].isna().sum() == 0, "NaN found in new_clusters"
assert adata_merged.obs['UTAG_names'].isna().sum() == 0, "NaN found in UTAG_names"
assert adata_merged.obs['UTAG_niches'].isna().sum() == 0, "NaN found in UTAG_niches"
organs = set(adata_merged.obs['SAHA_organcode'].unique())
assert organs == {'APE', 'COL', 'ILE'}, f"Unexpected organ codes: {organs}"
print("  Assertions passed: cell count, no NaN, organ codes correct")

# 7. Spatial coordinates in microns (DATA-04 partial: stored on analysis_ready for split step)
# CRITICAL: Use CenterX_global_px / CenterY_global_px (within-run stitched coords)
# DO NOT use obsm['spatial'] = x_shifted/y_shifted (has large cross-run offsets:
#   COL offset ~450,000px, ILE offset ~900,000px — unusable for within-organ spatial analysis)
print("Adding spatial_um coordinates (CenterX/Y_global_px * 0.1203)...")
adata_merged.obsm['spatial_um'] = (
    adata_merged.obs[['CenterX_global_px', 'CenterY_global_px']]
    .values.astype(float)
    * PIXEL_SIZE_UM
)
print(f"  spatial_um shape: {adata_merged.obsm['spatial_um'].shape}")

# 8. Save analysis_ready.h5ad (OUT-04)
print(f"\nSaving analysis_ready.h5ad to {OUT_H5AD}")
adata_merged.write_h5ad(OUT_H5AD)
file_size_mb = os.path.getsize(OUT_H5AD) / 1e6
print(f"  Saved. File size: {file_size_mb:.1f} MB")

# 9. Summary
print("\n=== Summary ===")
print(f"  Output: {OUT_H5AD}")
print(f"  Shape: {adata_merged.shape}")
print(f"  obs columns added: new_clusters, UTAG_niches, UTAG_names, SAHA_organcode")
print(f"  obsm added: spatial_um (um, CenterX/Y_global_px * 0.1203)")
print(f"  Organ counts: {adata_merged.obs['SAHA_organcode'].value_counts().to_dict()}")
print("  DONE: DATA-01, DATA-02, OUT-04 satisfied")
