#!/usr/bin/env python3
"""
02_preprocess_c1c2.py
Preprocess the C1+C2 joint AnnData for scVI/scANVI.

Input:  outputs/adata_c1c2.h5ad  (C1 from Dulai RDS + C2 SAHA subsample)
Output: outputs/adata_preprocessed.h5ad

Steps:
  1. Load adata_c1c2.h5ad
  2. Compute gene intersection between CosMx 6K and Xenium 5K panels
     - CosMx panel: genes in any cosmx_c1 cell (6175 genes)
     - Xenium panel: genes in any xenium_c1 cell (5001 genes)
     - Intersection: shared genes (bottleneck is Xenium)
  3. Subset to intersection genes
  4. Copy raw counts to layers['counts']
  5. Log1p normalize .X (target_sum=1e4)
  6. Save

Run: conda run -n spatial_integration python scripts/02_preprocess_c1c2.py
Working directory: project root (IBD_integration/)
"""

import os
import numpy as np
import scanpy as sc

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

# ─── 1. Load ─────────────────────────────────────────────────────────────────
print("Loading outputs/adata_c1c2.h5ad ...")
adata = sc.read_h5ad("outputs/adata_c1c2.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  dataset breakdown:\n{adata.obs['dataset'].value_counts().to_string()}")

# ─── 2. Compute gene intersection ────────────────────────────────────────────
print("\nComputing gene intersection across platforms ...")

cosmx_mask  = adata.obs["platform"] == "cosmx"
xenium_mask = adata.obs["platform"] == "xenium"

# Genes present in each platform: any cell in that platform has non-zero count
# Use sum across cells — gene must appear in at least 1 cell
import scipy.sparse

def genes_expressed(adata_sub):
    """Return set of gene names with at least 1 count in the subset."""
    X = adata_sub.X
    if hasattr(X, "toarray"):
        col_sums = np.asarray(X.sum(axis=0)).flatten()
    else:
        col_sums = X.sum(axis=0).flatten()
    return set(adata_sub.var_names[col_sums > 0])

cosmx_genes  = genes_expressed(adata[cosmx_mask])
xenium_genes = genes_expressed(adata[xenium_mask])
all_genes    = set(adata.var_names)

intersect = cosmx_genes & xenium_genes
cosmx_only = cosmx_genes - xenium_genes
xenium_only = xenium_genes - cosmx_genes

print(f"  CosMx genes expressed: {len(cosmx_genes)}")
print(f"  Xenium genes expressed: {len(xenium_genes)}")
print(f"  Intersection: {len(intersect)}")
print(f"  CosMx-only: {len(cosmx_only)}")
print(f"  Xenium-only: {len(xenium_only)}")
print(f"  Total var_names: {len(all_genes)}")

# Write updated gene panel report
os.makedirs("outputs", exist_ok=True)
with open("outputs/gene_panel_report_c1c2.txt", "w") as fh:
    fh.write("Gene panel intersection report (C1+C2 rebuild)\n")
    fh.write("=" * 48 + "\n")
    fh.write(f"CosMx genes: {len(cosmx_genes)}\n")
    fh.write(f"Xenium genes: {len(xenium_genes)}\n")
    fh.write(f"Intersection: {len(intersect)}\n")
    fh.write(f"CosMx-only: {len(cosmx_only)}\n")
    fh.write(f"Xenium-only: {len(xenium_only)}\n\n")
    fh.write("Integration strategy:\n")
    fh.write("  Use ALL intersected genes — no HVG pre-filtering.\n")
    fh.write("  Let scVI/scANVI handle internal feature selection.\n\n")
    fh.write("CosMx-only gene list:\n")
    for g in sorted(cosmx_only):
        fh.write(f"  {g}\n")

print("  Saved: outputs/gene_panel_report_c1c2.txt")

# ─── 3. Subset to intersection genes ─────────────────────────────────────────
# Preserve original var order
intersect_ordered = [g for g in adata.var_names if g in intersect]
print(f"\nSubsetting to {len(intersect_ordered)} intersection genes ...")
adata = adata[:, intersect_ordered].copy()
print(f"  After subset: {adata.shape}")

# ─── 4. Copy raw counts to layers['counts'] ───────────────────────────────────
print("Copying .X to layers['counts'] ...")
# X should be raw integer counts at this point
if hasattr(adata.X, "data"):
    is_int = np.all(adata.X.data % 1 == 0)
else:
    is_int = np.all(adata.X % 1 == 0)
print(f"  .X integer-valued: {is_int}")
adata.layers["counts"] = adata.X.copy()

# ─── 5. Log1p normalize ───────────────────────────────────────────────────────
print("Normalizing .X (normalize_total + log1p) ...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
if hasattr(adata.X, "data"):
    xmax = float(adata.X.data.max())
else:
    xmax = float(adata.X.max())
print(f"  .X max after normalization: {xmax:.3f}  (expect ~9.21 = log1p(1e4))")

# ─── 6. Verify obs columns ────────────────────────────────────────────────────
print("\nVerifying obs columns ...")
for col in ["dataset", "platform", "ct_major"]:
    assert col in adata.obs.columns, f"Missing obs column: {col}"
    n_empty = (adata.obs[col].astype(str) == "").sum()
    assert n_empty == 0, f"Column {col} has {n_empty} empty values"

valid_ct = {"Epithelial", "Mesenchymal", "B", "T/NK", "Plasma",
            "Myeloid", "Endothelial", "Neuron", "Unknown"}
actual_ct = set(adata.obs["ct_major"].unique())
unexpected = actual_ct - valid_ct
assert len(unexpected) == 0, f"Unexpected ct_major values: {unexpected}"

print("  All checks PASSED")
print(f"\nct_major breakdown:")
print(adata.obs["ct_major"].value_counts().to_string())
print(f"\ndataset breakdown:")
print(adata.obs["dataset"].value_counts().to_string())

# ─── 7. Save ─────────────────────────────────────────────────────────────────
out = "outputs/adata_preprocessed.h5ad"
print(f"\nSaving to {out} ...")
adata.write_h5ad(out)
print(f"Done. Shape: {adata.shape}")
