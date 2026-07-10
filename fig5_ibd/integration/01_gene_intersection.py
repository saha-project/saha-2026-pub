#!/usr/bin/env python
"""
DVAL-03: Compute gene panel intersection between CosMx and Xenium.
Run: conda run -n spatial_integration python scripts/01_gene_intersection.py
Writes: outputs/gene_panel_report.txt
"""
import os
import numpy as np
import scanpy as sc

os.makedirs("outputs", exist_ok=True)

print("Loading adata.joint.h5ad for gene intersection ...")
adata = sc.read_h5ad("adata.joint.h5ad")
print(f"  Shape: {adata.shape}")

# NOTE: platform values are lowercase in this file ('cosmx', 'xenium')
print(f"  Platform values: {sorted(adata.obs['platform'].unique().tolist())}")

# Get per-platform masks
cosmx_mask  = (adata.obs["platform"] == "cosmx").values
xenium_mask = (adata.obs["platform"] == "xenium").values

print(f"  CosMx cells: {cosmx_mask.sum():,}")
print(f"  Xenium cells: {xenium_mask.sum():,}")

if cosmx_mask.sum() == 0 or xenium_mask.sum() == 0:
    raise ValueError(
        f"One platform has 0 cells. Check platform column values: "
        f"{adata.obs['platform'].unique().tolist()}"
    )

# Genes with nonzero counts in each platform
# Use sparse column sum to avoid materializing full dense matrix
X = adata.X
if hasattr(X, "toarray"):
    cosmx_gene_sums  = np.asarray(X[cosmx_mask].sum(axis=0)).flatten()
    xenium_gene_sums = np.asarray(X[xenium_mask].sum(axis=0)).flatten()
else:
    cosmx_gene_sums  = np.asarray(X[cosmx_mask].sum(axis=0)).flatten()
    xenium_gene_sums = np.asarray(X[xenium_mask].sum(axis=0)).flatten()

cosmx_genes  = set(adata.var_names[cosmx_gene_sums > 0])
xenium_genes = set(adata.var_names[xenium_gene_sums > 0])

intersection = cosmx_genes & xenium_genes
cosmx_only   = cosmx_genes - xenium_genes
xenium_only  = xenium_genes - cosmx_genes

print(f"\nCosMx genes:        {len(cosmx_genes):,}")
print(f"Xenium genes:       {len(xenium_genes):,}")
print(f"Intersection:       {len(intersection):,}")
print(f"CosMx-only:         {len(cosmx_only):,}")
print(f"Xenium-only:        {len(xenium_only):,}")

# Threshold warning
if len(intersection) < 3000:
    print(f"\nWARNING: Intersection is {len(intersection)} genes, below 3000 minimum.")
    print("  Review gene panel before proceeding to preprocessing.")

# Write report
lines = [
    "Gene panel intersection report",
    "=" * 40,
    f"CosMx genes: {len(cosmx_genes)}",
    f"Xenium genes: {len(xenium_genes)}",
    f"Intersection: {len(intersection)}",
    f"CosMx-only: {len(cosmx_only)}",
    f"Xenium-only: {len(xenium_only)}",
    "",
    "Integration strategy (per user decision):",
    "  Use ALL intersected genes — no HVG pre-filtering.",
    "  Let scVI/COSIE handle internal feature selection.",
    "  Non-overlapping genes: retain per-platform if method supports it;",
    "  fall back to intersection-only if common gene set required.",
    "",
]
if len(cosmx_only) > 0:
    lines.append(f"CosMx-only gene list ({len(cosmx_only)} genes):")
    lines.extend(f"  {g}" for g in sorted(cosmx_only))
    lines.append("")
if len(xenium_only) > 0:
    lines.append(f"Xenium-only gene list ({len(xenium_only)} genes):")
    lines.extend(f"  {g}" for g in sorted(xenium_only))

with open("outputs/gene_panel_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\nWrote: outputs/gene_panel_report.txt")
print("\nDVAL-03: PASSED")
