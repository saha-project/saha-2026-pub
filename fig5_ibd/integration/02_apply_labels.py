#!/usr/bin/env python3
# 02_apply_labels.py
# Apply ct_major label transfer and consolidation to adata.joint.h5ad
# Run with: conda run -n spatial_integration python scripts/02_apply_labels.py
# Working directory: project root (IBD_integration/)
#
# Label strategy (4 tiers):
#   1. SAHA_IBD_RNA.h5ad cell_id -> cell_type (C1 consolidated, 404K cells)
#   2. ct_major_transferred.csv (CosMx RDS files) -> consolidation rules (~145 extra cells)
#   3. predicted_celltype column -> consolidation rules (~14K extra cells)
#   4. Fallback "Epithelial" for any remaining unlabeled cells (~24K cells)

import scanpy as sc
import pandas as pd
import h5py
import os

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"

# ── 1. Load consolidation rules ──────────────────────────────────────────────
print("Loading consolidation rules...")
mapping_df = pd.read_csv(os.path.join(PROJ_ROOT, "outputs/label_mapping_final.csv"))
consolidation_df = mapping_df[mapping_df['status'] == 'consolidation']
consolidation_dict = dict(zip(consolidation_df['c2_label'], consolidation_df['c1_proposed']))
print(f"Consolidation rules ({len(consolidation_dict)}): {consolidation_dict}")

# ── 2. Build Tier 1 lookup: SAHA_IBD_RNA.h5ad ────────────────────────────────
print("\nBuilding Tier 1 lookup from SAHA_IBD_RNA.h5ad...")
saha_rna_path = os.path.join(PROJ_ROOT, "SAHA_IBD_RNA.h5ad")

with h5py.File(saha_rna_path, 'r') as f:
    cats = [x.decode() if isinstance(x, bytes) else str(x)
            for x in f['obs']['cell_id']['categories'][:]]
    codes = f['obs']['cell_id']['codes'][:]
    cell_ids = list(pd.Categorical.from_codes(codes, cats))

    cats_ct = [x.decode() if isinstance(x, bytes) else x
               for x in f['obs']['cell_type']['categories'][:]]
    codes_ct = f['obs']['cell_type']['codes'][:]
    cell_types = list(pd.Categorical.from_codes(codes_ct, cats_ct))

ref_df = pd.DataFrame({'cell_id': cell_ids, 'cell_type': cell_types})
print(f"SAHA_IBD_RNA rows: {len(ref_df)}, unique cell_ids: {ref_df['cell_id'].nunique()}")

def first_mode(x):
    vc = x.value_counts()
    return vc.index[0] if len(vc) > 0 else ''

tier1_lookup = ref_df.groupby('cell_id')['cell_type'].agg(first_mode).to_dict()
# Filter out empty strings
tier1_lookup = {k: v for k, v in tier1_lookup.items() if v != ''}
print(f"Tier 1 lookup entries: {len(tier1_lookup)}")

# Note: Tier 1 labels are already in C1 consolidated vocabulary
# (Lymphoid-B, Mesenchymal-Stromal, etc.) — no further consolidation needed.

# ── 3. Build Tier 2 lookup: ct_major_transferred.csv (CosMx RDS) ─────────────
print("\nBuilding Tier 2 lookup from ct_major_transferred.csv...")
ct_csv_path = os.path.join(PROJ_ROOT, "outputs/ct_major_transferred.csv")
ct_df = pd.read_csv(ct_csv_path)
print(f"CT CSV rows: {len(ct_df)}, unique barcodes: {ct_df['barcode'].nunique()}")

tier2_lookup_raw = ct_df.groupby('barcode')['ct_major'].agg(first_mode).to_dict()
tier2_lookup_raw = {k: v for k, v in tier2_lookup_raw.items() if v != ''}
# Apply consolidation rules to ct_major values
tier2_lookup = {k: consolidation_dict.get(v, v) for k, v in tier2_lookup_raw.items()}
print(f"Tier 2 lookup entries: {len(tier2_lookup)}")

# ── 4. Load adata.joint.h5ad ─────────────────────────────────────────────────
print("\nLoading adata.joint.h5ad (this may take ~60 seconds)...")
adata = sc.read_h5ad(os.path.join(PROJ_ROOT, "adata.joint.h5ad"))
print(f"Loaded: {adata.shape}")

# ── 5. Apply labels to empty cell_type cells (vectorized) ────────────────────
empty_mask = adata.obs['cell_type'] == ''
n_empty_before = int(empty_mask.sum())
print(f"\nEmpty cell_type cells before transfer: {n_empty_before}")
print(f"Sample empty obs_names: {adata.obs_names[empty_mask][:5].tolist()}")

# Convert to plain string Series (adata stores cell_type as Categorical with fixed categories)
# We need to add new categories (from RDS ct_major + predicted_celltype), so convert to object dtype
cell_type_col = adata.obs['cell_type'].astype(str).copy()

# Tier 1: SAHA_IBD_RNA (C1 consolidated labels, vectorized map)
# Use pandas Series with obs_names as index for vectorized lookup
empty_idx = adata.obs_names[empty_mask]  # pandas Index
tier1_series = pd.Series(tier1_lookup)
# Map Index to series values; result is a Series indexed by obs_names
tier1_mapped = pd.Series(empty_idx).map(tier1_series)
tier1_mapped.index = empty_idx  # restore obs_names as index
fill_t1_mask = tier1_mapped.notna() & (tier1_mapped != '')
cell_type_col.loc[fill_t1_mask[fill_t1_mask].index] = tier1_mapped[fill_t1_mask].values
tier1_filled = int(fill_t1_mask.sum())
print(f"Tier 1 (SAHA_IBD_RNA) filled: {tier1_filled}")

still_empty = cell_type_col == ''

# Tier 2: CosMx RDS ct_major (already consolidated via tier2_lookup)
tier2_series = pd.Series(tier2_lookup)
still_empty_idx = adata.obs_names[still_empty]
tier2_mapped = pd.Series(still_empty_idx).map(tier2_series)
tier2_mapped.index = still_empty_idx
fill_t2_mask = tier2_mapped.notna() & (tier2_mapped != '')
cell_type_col.loc[fill_t2_mask[fill_t2_mask].index] = tier2_mapped[fill_t2_mask].values
tier2_filled = int(fill_t2_mask.sum())
print(f"Tier 2 (CosMx RDS) additionally filled: {tier2_filled}")

still_empty = cell_type_col == ''

# Tier 3: predicted_celltype (apply consolidation rules)
pred_ct = adata.obs.loc[still_empty, 'predicted_celltype']
pred_ct_consolidated = pred_ct[pred_ct != ''].map(lambda v: consolidation_dict.get(v, v))
cell_type_col.loc[pred_ct_consolidated.index] = pred_ct_consolidated.values
tier3_filled = int((pred_ct != '').sum())
print(f"Tier 3 (predicted_celltype) additionally filled: {tier3_filled}")

still_empty = cell_type_col == ''
n_still_empty = int(still_empty.sum())

# Tier 4: Fallback for remaining cells
fallback_label = "Epithelial"
if n_still_empty > 0:
    print(f"Tier 4 (fallback='{fallback_label}') for {n_still_empty} remaining cells")
    cell_type_col.loc[still_empty] = fallback_label
else:
    print("Tier 4: no cells needed fallback")

# ── 6. Apply consolidation rules to ALL cell_type values ─────────────────────
print("\nApplying consolidation rules to all cell_type values...")
pre_consolidation_unique = sorted(cell_type_col.unique().tolist())
print(f"Cell types before consolidation: {pre_consolidation_unique}")

cell_type_col = cell_type_col.replace(consolidation_dict)

post_consolidation_unique = sorted(cell_type_col.unique().tolist())
print(f"Cell types after consolidation: {post_consolidation_unique}")

# ── 7. Validate ───────────────────────────────────────────────────────────────
n_empty_after = (cell_type_col == '').sum()
assert n_empty_after == 0, f"VALIDATION FAILED: {n_empty_after} empty cell_type values remain"
print(f"\nValidation PASSED: 0 empty cell_type values")

# ── 8. Write updated adata ───────────────────────────────────────────────────
adata.obs['cell_type'] = cell_type_col

print("\nCell type distribution (final):")
print(adata.obs['cell_type'].value_counts())

output_path = os.path.join(PROJ_ROOT, "outputs/adata_labeled.h5ad")
print(f"\nSaving to {output_path}...")
adata.write_h5ad(output_path)
print("Done.")
