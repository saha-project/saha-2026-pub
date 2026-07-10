#!/usr/bin/env python3
"""
02_relabel_ct_major.py
Re-assign all cells in adata.joint.h5ad to the corrected ct_major vocabulary.

Final 9 categories: Epithelial, Mesenchymal, B, T/NK, Plasma, Myeloid,
                    Endothelial, Neuron, Unknown

Label strategy by cell group:
  1. CosMx reference (source=reference, platform=cosmx, ~440K empty cells):
       ct_major from RDS files (outputs/ct_major_transferred.csv or re-extract)
  2. Xenium reference (source=reference, platform=xenium, ~75K cells):
       Direct consolidation: Mesenchymal-Stromal->Mesenchymal, Lymphoid-B->B, etc.
       But Xenium labels are already ct_major: Epithelial, Mesenchymal, T/NK, B,
       Plasma, Myeloid, Endothelial, Neuron — pass through with minor cleanup.
  3. SAHA (source=saha, ~100K cells):
       C1->ct_major mapping + Lymphoid-Other split via cell_type_general from SAHA_IBD_RNA.h5ad

Output: outputs/adata_labeled.h5ad (overwrite)
Run:    conda run -n spatial_integration python scripts/02_relabel_ct_major.py
Workdir: project root (IBD_integration/)
"""

import scanpy as sc
import pandas as pd
import h5py
import numpy as np
import os

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load adata.joint.h5ad (the original, unmodified file)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading adata.joint.h5ad ...")
adata = sc.read_h5ad("adata.joint.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")

# Start with a fresh ct_major column (object dtype)
adata.obs['ct_major'] = ''

# ─────────────────────────────────────────────────────────────────────────────
# 2. CosMx reference cells: load ct_major from RDS CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- CosMx reference cells ---")

# These cells have source=reference, platform=cosmx and empty cell_type
cosmx_ref_mask = (adata.obs['source'] == 'reference') & (adata.obs['platform'] == 'cosmx')
n_cosmx_ref = int(cosmx_ref_mask.sum())
print(f"  CosMx reference cells: {n_cosmx_ref}")

# Load ct_major_transferred.csv (from both CosMx_6k_4 and CosMx_1k_16)
ct_csv_path = "outputs/ct_major_transferred.csv"
print(f"  Loading {ct_csv_path} ...")
ct_df = pd.read_csv(ct_csv_path)
print(f"  CSV rows: {len(ct_df)}, unique barcodes: {ct_df['barcode'].nunique()}")
print(f"  ct_major unique in CSV: {sorted(ct_df['ct_major'].unique())}")

def first_mode(x):
    vc = x.value_counts()
    return vc.index[0] if len(vc) > 0 else ''

# Build lookup: barcode -> ct_major (most frequent if duplicates)
# ct_major values from RDS are already final: Epithelial, Mesenchymal, Endothelial,
# Plasma, Myeloid, T/NK, B, Neuron
cosmx_lookup = ct_df.groupby('barcode')['ct_major'].agg(first_mode).to_dict()
cosmx_lookup = {k: v for k, v in cosmx_lookup.items() if v != ''}
print(f"  Lookup entries: {len(cosmx_lookup)}")

# Also check ct_major_1k16.csv in case barcodes there are additional
ct_1k16_path = "outputs/ct_major_1k16.csv"
if os.path.exists(ct_1k16_path):
    ct_1k16 = pd.read_csv(ct_1k16_path)
    print(f"  ct_major_1k16.csv rows: {len(ct_1k16)}, unique barcodes: {ct_1k16['barcode'].nunique()}")
    extra_lookup = ct_1k16.groupby('barcode')['ct_major'].agg(first_mode).to_dict()
    extra_lookup = {k: v for k, v in extra_lookup.items() if v != ''}
    # Merge: ct_major_transferred.csv takes priority
    for k, v in extra_lookup.items():
        if k not in cosmx_lookup:
            cosmx_lookup[k] = v
    print(f"  Combined lookup entries: {len(cosmx_lookup)}")

# Apply to CosMx reference cells
cosmx_idx = adata.obs_names[cosmx_ref_mask]
cosmx_series = pd.Series(cosmx_idx).map(pd.Series(cosmx_lookup))
cosmx_series.index = cosmx_idx

# Cells found in lookup
found_mask = cosmx_series.notna() & (cosmx_series != '')
n_found = int(found_mask.sum())
print(f"  Cells matched in RDS lookup: {n_found} / {n_cosmx_ref}")

adata.obs.loc[found_mask[found_mask].index, 'ct_major'] = cosmx_series[found_mask].values

# For remaining CosMx cells not in RDS files (should be very few):
# Use predicted_celltype if available, else assign 'Unknown'
still_empty_cosmx = (cosmx_ref_mask) & (adata.obs['ct_major'] == '')
n_still_empty_cosmx = int(still_empty_cosmx.sum())
if n_still_empty_cosmx > 0:
    print(f"  CosMx cells not matched in RDS: {n_still_empty_cosmx} — assigning 'Unknown'")
    adata.obs.loc[still_empty_cosmx, 'ct_major'] = 'Unknown'

print(f"  CosMx ct_major distribution:")
print(adata.obs.loc[cosmx_ref_mask, 'ct_major'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Xenium reference cells: direct ct_major mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Xenium reference cells ---")

xenium_mask = (adata.obs['source'] == 'reference') & (adata.obs['platform'] == 'xenium')
n_xenium = int(xenium_mask.sum())
print(f"  Xenium reference cells: {n_xenium}")
print("  Current labels:")
print(adata.obs.loc[xenium_mask, 'cell_type'].value_counts().to_string())

# Xenium labels in adata.joint.h5ad are already in ct_major vocabulary:
# Epithelial, Mesenchymal, Plasma, T/NK, B, Myeloid, Endothelial, Neuron
# The objective mapping:
#   Epithelial -> Epithelial
#   Mesenchymal-Stromal -> Mesenchymal  (but Xenium has 'Mesenchymal' not 'Mesenchymal-Stromal')
#   Lymphoid-B -> B                     (but Xenium has 'B' directly)
#   Lymphoid-T -> T/NK                  (but Xenium has 'T/NK' directly)
#   Lymphoid-Plasma -> Plasma           (but Xenium has 'Plasma' directly)
#   Myeloid -> Myeloid
#   Endothelial -> Endothelial
#   Neuron -> Neuron
# So Xenium labels are already in final ct_major vocabulary — copy directly.
# Handle edge cases for any unexpected labels.

xenium_ct_to_ct_major = {
    'Epithelial': 'Epithelial',
    'Mesenchymal': 'Mesenchymal',
    'Mesenchymal-Stromal': 'Mesenchymal',  # just in case
    'Plasma': 'Plasma',
    'Lymphoid-Plasma': 'Plasma',            # just in case
    'T/NK': 'T/NK',
    'Lymphoid-T': 'T/NK',                  # just in case
    'B': 'B',
    'Lymphoid-B': 'B',                     # just in case
    'Myeloid': 'Myeloid',
    'Endothelial': 'Endothelial',
    'Neuron': 'Neuron',
    'Mixed': 'Unknown',
    '': 'Unknown',
}

xenium_ct = adata.obs.loc[xenium_mask, 'cell_type'].astype(str)
xenium_ct_major = xenium_ct.map(lambda x: xenium_ct_to_ct_major.get(x, 'Unknown'))
adata.obs.loc[xenium_mask, 'ct_major'] = xenium_ct_major.values

print("  Xenium ct_major distribution:")
print(adata.obs.loc[xenium_mask, 'ct_major'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAHA cells: C1->ct_major mapping + Lymphoid-Other split
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- SAHA cells ---")

saha_mask = adata.obs['source'] == 'saha'
n_saha = int(saha_mask.sum())
print(f"  SAHA cells: {n_saha}")
print("  Current labels:")
print(adata.obs.loc[saha_mask, 'cell_type'].value_counts(dropna=False).to_string())

# Build SAHA cell_id -> cell_type_general lookup from SAHA_IBD_RNA.h5ad
print("\n  Building Lymphoid-Other split lookup from SAHA_IBD_RNA.h5ad ...")
saha_rna_path = "SAHA_IBD_RNA.h5ad"

with h5py.File(saha_rna_path, 'r') as f:
    # cell_id
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f['obs']['cell_id']['categories'][:]]
    codes = f['obs']['cell_id']['codes'][:]
    cell_ids = pd.Categorical.from_codes(codes, cats)

    # cell_type
    cats_ct = [x.decode() if isinstance(x, bytes) else str(x) for x in f['obs']['cell_type']['categories'][:]]
    codes_ct = f['obs']['cell_type']['codes'][:]
    cell_types = pd.Categorical.from_codes(codes_ct, cats_ct)

    # cell_type_general
    cats_ctg = [x.decode() if isinstance(x, bytes) else str(x) for x in f['obs']['cell_type_general']['categories'][:]]
    codes_ctg = f['obs']['cell_type_general']['codes'][:]
    cell_type_general = pd.Categorical.from_codes(codes_ctg, cats_ctg)

saha_rna_df = pd.DataFrame({
    'cell_id': list(cell_ids),
    'cell_type': list(cell_types),
    'cell_type_general': list(cell_type_general),
})
print(f"  SAHA_IBD_RNA rows: {len(saha_rna_df)}, unique cell_ids: {saha_rna_df['cell_id'].nunique()}")

# For Lymphoid-Other split:
#   cell_type_general = 'APC' -> Myeloid
#   cell_type_general = 'Lymphoid' -> T/NK
lymphoid_other_rows = saha_rna_df[saha_rna_df['cell_type'] == 'Lymphoid-Other'].copy()
print(f"  Lymphoid-Other rows in SAHA_IBD_RNA: {len(lymphoid_other_rows)}")
print(f"  cell_type_general breakdown: {lymphoid_other_rows['cell_type_general'].value_counts().to_dict()}")

# Build cell_id -> ct_major for Lymphoid-Other cells
def lymphoid_other_to_ct_major(ctg):
    if ctg == 'APC':
        return 'Myeloid'
    elif ctg == 'Lymphoid':
        return 'T/NK'
    else:
        return 'Unknown'  # unexpected value

lymphoid_other_rows['ct_major'] = lymphoid_other_rows['cell_type_general'].map(lymphoid_other_to_ct_major)
lo_lookup = lymphoid_other_rows.groupby('cell_id')['ct_major'].agg(first_mode).to_dict()
print(f"  Lymphoid-Other lookup entries: {len(lo_lookup)}")

# SAHA cell_type -> ct_major base mapping (all non-Lymphoid-Other)
saha_base_mapping = {
    'Epithelial': 'Epithelial',
    'Mesenchymal-Stromal': 'Mesenchymal',
    'Lymphoid-B': 'B',
    'Lymphoid-T': 'T/NK',
    'Lymphoid-Plasma': 'Plasma',
    'Myeloid': 'Myeloid',
    'Immune-Other': 'Myeloid',    # cell_type_general = ImmuneCells
    'Stem': 'Epithelial',         # cell_type_general = OLFM4+/Paneth
    'Mixed': 'Unknown',           # cell_type_general = LowExp
    '': 'Unknown',
}

# Apply to SAHA cells
saha_obs = adata.obs[saha_mask].copy()
saha_ct = saha_obs['cell_type'].astype(str)
saha_cell_ids = saha_obs['cell_id'].astype(str)

ct_major_saha = []
n_lo_matched = 0
n_lo_unmatched = 0
for obs_name, (ct, cid) in zip(saha_obs.index, zip(saha_ct, saha_cell_ids)):
    if ct == 'Lymphoid-Other':
        if cid in lo_lookup:
            ct_major_saha.append(lo_lookup[cid])
            n_lo_matched += 1
        else:
            # Fallback: try to lookup in full SAHA_IBD_RNA and check cell_type_general
            ct_major_saha.append('Unknown')
            n_lo_unmatched += 1
    else:
        ct_major_saha.append(saha_base_mapping.get(ct, 'Unknown'))

print(f"  Lymphoid-Other matched: {n_lo_matched}, unmatched: {n_lo_unmatched}")
adata.obs.loc[saha_mask, 'ct_major'] = ct_major_saha

print("  SAHA ct_major distribution:")
print(adata.obs.loc[saha_mask, 'ct_major'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Validation ---")

n_empty = (adata.obs['ct_major'] == '').sum()
assert n_empty == 0, f"FAILED: {n_empty} cells still have empty ct_major"
print(f"  All cells have non-empty ct_major: PASSED")

valid_labels = {'Epithelial', 'Mesenchymal', 'B', 'T/NK', 'Plasma', 'Myeloid',
                'Endothelial', 'Neuron', 'Unknown'}
actual_labels = set(adata.obs['ct_major'].unique())
unexpected = actual_labels - valid_labels
assert len(unexpected) == 0, f"FAILED: unexpected labels: {unexpected}"
print(f"  All labels in valid set: PASSED")
print(f"  Label set used: {sorted(actual_labels)}")

print("\n--- ct_major distribution by platform ---")
for platform in ['cosmx', 'xenium']:
    mask = adata.obs['platform'] == platform
    print(f"\n  {platform}:")
    print(adata.obs.loc[mask, 'ct_major'].value_counts().to_string())

print("\n--- ct_major distribution by source ---")
for source in ['reference', 'saha']:
    mask = adata.obs['source'] == source
    print(f"\n  {source}:")
    print(adata.obs.loc[mask, 'ct_major'].value_counts().to_string())

print("\n--- Overall ct_major distribution ---")
print(adata.obs['ct_major'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = "outputs/adata_labeled.h5ad"
print(f"\nSaving to {out_path} ...")
adata.write_h5ad(out_path)
print(f"Done. Shape: {adata.shape}")
print("ct_major column saved with corrected labels.")
