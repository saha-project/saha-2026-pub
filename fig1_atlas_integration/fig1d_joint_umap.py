"""Fig 1d / ED Fig 1 — joint UMAP of the CosMx 1,000-plex RNA reference across organs.

Runnable extract of as_run/merge_all_organs_RNA.py, sections 1-3 (load + merge, preprocessing,
Harmony, UMAP). Parameters are unchanged. Three edits make it run top to bottom:
  - the organ list is the 10-object list (as_run later overrides it with a GI-only subset);
  - as_run line 45 `adata.X = adata.layers['raw']` is applied to the merged object;
  - the Harmony output orientation is checked, since harmonypy 0.2 changed it.

Inputs: per-organ objects SAHA_<ORGAN>_RNA.h5ad from as_run/merge_all_objects_v4.py, carrying
raw counts in layers['raw'] and InSituType labels (obs['Insitutype_Broad']) from
as_run/postmerge_processing.R.

Usage: python fig1d_joint_umap.py <dir with SAHA_*_RNA.h5ad> <out.h5ad>
"""
import sys
from pathlib import Path

import harmonypy as hm
import scanpy as sc

FILES = [
    "SAHA_APE_RNA.h5ad",
    "SAHA_BM_RNA.h5ad",
    "SAHA_COL_RNA.h5ad",
    "SAHA_ILE_RNA.h5ad",
    "SAHA_LIV_RNA_A1.h5ad",
    "SAHA_LIV_RNA_A2.h5ad",
    "SAHA_LN_RNA.h5ad",
    "SAHA_PANC_RNA.h5ad",
    "SAHA_PROS_RNA.h5ad",
    "SAHA_STO_RNA.h5ad",
]

indir, out = Path(sys.argv[1]), sys.argv[2]
sc.set_figure_params(figsize=(8, 8))

adatas = [sc.read_h5ad(indir / f) for f in FILES]
adata = adatas[0].concatenate(*adatas[1:], batch_key="batch", batch_categories=FILES)
adata.X = adata.layers["raw"]

sc.pp.filter_cells(adata, min_counts=20)
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_genes(adata, min_cells=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver="arpack")
ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
# harmonypy < 0.2 returns PCs x cells (as_run transposed it); 0.2+ returns cells x PCs.
Z = ho.Z_corr
adata.obsm["X_harmony"] = Z if Z.shape[0] == adata.n_obs else Z.T

sc.pp.neighbors(adata, use_rep="X_harmony")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "Insitutype_Broad"], save="_fig1d_joint_umap.pdf")

adata.write(out)
