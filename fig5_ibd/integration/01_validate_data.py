#!/usr/bin/env python
"""
DVAL-01: Validate adata.joint.h5ad schema.
Run: conda run -n spatial_integration python scripts/01_validate_data.py
Writes: outputs/adata_joint_schema_report.txt
"""
import os
import sys
import numpy as np
import scanpy as sc

os.makedirs("outputs", exist_ok=True)

print("Loading adata.joint.h5ad ...")
adata = sc.read_h5ad("adata.joint.h5ad")
print(f"  Shape: {adata.shape}  ({adata.n_obs} cells x {adata.n_vars} genes)")

report_lines = []
report_lines.append("adata.joint.h5ad Schema Report")
report_lines.append("=" * 50)
report_lines.append(f"Shape: {adata.shape}")

# --- obsm['spatial'] ---
spatial_ok = "spatial" in adata.obsm
spatial_shape = adata.obsm["spatial"].shape if spatial_ok else None
report_lines.append(f"\nobsm['spatial'] present: {spatial_ok}")
if spatial_ok:
    report_lines.append(f"obsm['spatial'] shape: {spatial_shape}")
    # Check 2D coordinates
    if spatial_shape[1] != 2:
        report_lines.append(f"  WARNING: expected 2 columns (x, y), got {spatial_shape[1]}")
else:
    report_lines.append("  ERROR: obsm['spatial'] missing — cannot build spatial graph")
    sys.exit(1)

# --- Required obs columns ---
# NOTE: 'dataset' may be stored as 'source' in this file — accept either
report_lines.append("\nobs columns check:")
required_cols = ["dataset", "platform", "cell_type"]
# Map expected column -> actual column found in obs
col_alias = {"dataset": "source"}  # fallback if 'dataset' missing
resolved_cols = {}
for col in required_cols:
    if col in adata.obs.columns:
        resolved_cols[col] = col
    elif col in col_alias and col_alias[col] in adata.obs.columns:
        resolved_cols[col] = col_alias[col]
        report_lines.append(f"  NOTE: '{col}' not found; using alias '{col_alias[col]}'")
    else:
        resolved_cols[col] = None

for col in required_cols:
    actual = resolved_cols[col]
    if actual is not None:
        n_unique = adata.obs[actual].nunique()
        n_null = adata.obs[actual].isna().sum()
        alias_note = f" (stored as '{actual}')" if actual != col else ""
        report_lines.append(f"  {col}: OK{alias_note}  ({n_unique} unique values, {n_null} nulls)")
    else:
        report_lines.append(f"  {col}: MISSING")

# --- All obs columns (informational) ---
report_lines.append(f"\nAll obs columns ({len(adata.obs.columns)}):")
for col in sorted(adata.obs.columns):
    report_lines.append(f"  {col}")

# --- Platform breakdown ---
plat_col = resolved_cols.get("platform")
if plat_col:
    report_lines.append("\nPlatform breakdown:")
    for plat, count in adata.obs[plat_col].value_counts().items():
        report_lines.append(f"  {plat}: {count:,} cells")

# --- Dataset breakdown ---
ds_col = resolved_cols.get("dataset")
if ds_col:
    report_lines.append("\nDataset breakdown:")
    for ds, count in adata.obs[ds_col].value_counts().items():
        report_lines.append(f"  {ds}: {count:,} cells")

# --- .X content check ---
report_lines.append("\n.X content check:")
report_lines.append(f"  type: {type(adata.X).__name__}")
report_lines.append(f"  dtype: {adata.X.dtype}")
sample = adata.X[:500].toarray() if hasattr(adata.X, "toarray") else adata.X[:500]
is_integer = bool(np.all(sample == sample.astype(int)))
has_negatives = bool(np.any(sample < 0))
max_val = float(sample.max())
report_lines.append(f"  integer-valued (sample 500 rows): {is_integer}")
report_lines.append(f"  has negatives: {has_negatives}")
report_lines.append(f"  max value (sample): {max_val:.2f}")

# Check layers
report_lines.append(f"\nAvailable layers: {list(adata.layers.keys())}")
has_counts_layer = "counts" in adata.layers
report_lines.append(f"layers['counts'] present: {has_counts_layer}")

# Determine raw counts location
if is_integer and not has_negatives:
    raw_counts_location = ".X (appears to be raw integer counts)"
elif has_counts_layer:
    raw_counts_location = "layers['counts']"
else:
    raw_counts_location = "UNKNOWN — .X is not integer and no layers['counts']; manual inspection required"
report_lines.append(f"\nRaw counts location: {raw_counts_location}")

# --- C1 cell type labels ---
if "cell_type" in adata.obs.columns:
    c1_labels = sorted(adata.obs["cell_type"].dropna().unique().tolist())
    report_lines.append(f"\nC1 cell types ({len(c1_labels)}):")
    for label in c1_labels:
        count = (adata.obs["cell_type"] == label).sum()
        report_lines.append(f"  {label}: {count:,} cells")
else:
    c1_labels = []
    report_lines.append("\nC1 cell types: MISSING (no cell_type column)")

# Write report
report_text = "\n".join(report_lines)
with open("outputs/adata_joint_schema_report.txt", "w") as f:
    f.write(report_text + "\n")

print(report_text)
print("\nWrote: outputs/adata_joint_schema_report.txt")

# Exit with error if critical fields are missing (accounting for aliases)
missing_cols = [c for c in required_cols if resolved_cols.get(c) is None]
if missing_cols:
    print(f"\nERROR: Missing required obs columns (no alias found either): {missing_cols}")
    sys.exit(1)
if not spatial_ok:
    print("\nERROR: obsm['spatial'] missing")
    sys.exit(1)
if raw_counts_location.startswith("UNKNOWN"):
    print("\nWARNING: Raw counts location unclear — inspect manually before Phase 2")

print("\nDVAL-01: PASSED")
