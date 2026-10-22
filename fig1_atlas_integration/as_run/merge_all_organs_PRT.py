import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import harmonypy as hm
import scipy.sparse as sp
import squidpy as sq
from scipy.sparse import csr_matrix
from scipy.sparse import diags

filename = "All_PRT"
organname = "All"
#celltype_id = "Insitutype_Broad"

plt.rcParams['font.family'] = 'Arial'
sc.set_figure_params(figsize=(8, 8))

# 1. List files
files = [
    "SAHA_APE_PRT.h5ad",
    "SAHA_BM_PRT.h5ad",
    "SAHA_COL_PRT.h5ad",
    "SAHA_ILE_PRT.h5ad",
    "SAHA_LIV_PRT.h5ad",
    "SAHA_LN_PRT.h5ad",
    "SAHA_PANC_PRT.h5ad",
    "SAHA_PROS_PRT.h5ad",
    "SAHA_STO_PRT.h5ad",
]
files = [
    "SAHA_APE_PRT.h5ad",
    "SAHA_COL_PRT.h5ad",
    "SAHA_ILE_PRT.h5ad",
    "SAHA_STO_PRT.h5ad",
]

color_dict = {
    "COL": "#003366",
    "APE": "#8B0000",
    "ILE": "#005F6A",
    "STO": "#C1440E",
    "LIV": "#A68B00",
    "PANC": "#550033",
    "PROS": "#004D1A",
    "LN": "#2A6365",
    "BM": "#800000",
    "CRC": "#336A99",
    "HB": "#7A6000",
    "PDAC": "#550044",
    "IBD": "#3A7C7D"
}


# 2. Load and merge
adatas = [sc.read_h5ad(f) for f in files]
adata_merged = adatas[0].concatenate(*adatas[1:], batch_key="batch", batch_categories=files)

adata_merged.X = adata_merged.layers['raw']
adata_merged.obs["SAHA_organcode"] = adata_merged.obs["SAHA_organcode"].astype("category")
adata_merged.uns["SAHA_organcode_colors"] = [color_dict[code] for code in adata_merged.obs["SAHA_organcode"].cat.categories]

# Ensure that the relevant columns are of string type
adata_merged.obs['SAHA_organcode'] = adata_merged.obs['SAHA_organcode'].astype(str)
adata_merged.obs['SAHA_batchcode'] = adata_merged.obs['SAHA_batchcode'].astype(str)
adata_merged.obs['fov'] = adata_merged.obs['fov'].astype(str)

# Create the new column 'SAHA_FOV'
adata_merged.obs['SAHA_FOV'] = (
    adata_merged.obs['SAHA_organcode'] +
    "_PRT_" +
    adata_merged.obs['SAHA_batchcode'] +
    "_" +
    adata_merged.obs['fov']
)

# Optionally, verify the result by displaying the first few rows
print(adata_merged.obs[['SAHA_organcode', 'SAHA_batchcode', 'fov', 'SAHA_FOV']].head())

# 3. Preprocessing
sc.pp.filter_cells(adata_merged, min_counts=20)
sc.pp.filter_cells(adata_merged, min_genes=1)
sc.pp.filter_genes(adata_merged, min_cells=1)
#sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
sc.pp.scale(adata_merged, max_value=10)

sc.tl.pca(adata_merged, svd_solver='arpack')
ho = hm.run_harmony(adata_merged.obsm["X_pca"], adata_merged.obs, 'SAHA_batchcode')
adata_merged.obsm["X_harmony"] = ho.Z_corr.T

sc.pp.neighbors(adata_merged, n_neighbors=40, metric = 'cosine')
sc.tl.umap(adata_merged)
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_n40Cos_scanpy.pdf')
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_n40Cos_scanpy.png')
ho = hm.run_harmony(adata_merged.obsm["X_pca"], adata_merged.obs, 'SAHA_batchcode')
adata_merged.write('./SAHA_All_PRT.h5ad')

ho = hm.run_harmony(adata_merged.obsm["X_pca"], adata_merged.obs, 'SAHA_batchcode')
sc.pp.neighbors(adata_merged, use_rep="X_harmony")
sc.tl.umap(adata_merged)
adata_merged.obsm["X_umap_after_harmony"] = adata_merged.obsm["X_umap"]
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_after_harmony_byBatchcode_scanpy.pdf')
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_after_harmony_byBatchcode_scanpy.png')

sc.pp.neighbors(adata_merged, n_neighbors=40, metric = 'cosine')
sc.tl.leiden(adata_merged,
    resolution=0.9,
    random_state=0,
    #flavor="igraph",
    n_iterations=2,
    directed=False,
)
sc.tl.louvain(adata_merged)
sc.tl.umap(adata_merged)
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_after_scanpy.pdf')
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_after_scanpy.png')

ho = hm.run_harmony(adata_merged.obsm["X_pca"], adata_merged.obs, 'batch')
adata_merged.obsm["X_harmony"] = ho.Z_corr.T
ho = hm.run_harmony(adata_merged.obsm["X_pca"], adata_merged.obs, 'SAHA_name')
adata_merged.obsm["X_harmony"] = ho.Z_corr.T



adata_merged.write('')






subset = sc.pp.subsample(adata_merged, n_obs=100000, copy=True)
sc.tl.pca(subset, svd_solver='arpack')
ho = hm.run_harmony(subset.obsm["X_pca"], subset.obs, 'batch')
sc.pp.neighbors(subset)
sc.pp.neighbors(subset, n_neighbors=40, metric = 'cosine')
sc.pp.neighbors(subset, use_rep="X_harmony", n_neighbors=40, metric = 'cosine')
sc.tl.umap(subset)
sc.tl.leiden(subset,
    resolution=0.9,
    random_state=0,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)
sc.pl.umap(subset, color=['batch'])
sc.pl.umap(subset, color=['leiden'])








sc.pp.neighbors(adata_merged)
sc.tl.umap(adata_merged)
adata_merged.obsm["X_umap_before_harmony"] = adata_merged.obsm["X_umap"]
sc.pl.umap(adata_merged, color=['batch'], title="PRT UMAP before Harmony", save='_PRT_before_harmony_scanpy.pdf')
sc.pl.umap(adata_merged, color=['batch'], title="PRT UMAP before Harmony", save='_PRT_before_harmony_scanpy.png')
sc.pl.umap(adata_merged, color=['Insitutype_Broad'], title="PRT UMAP before Harmony", save='_PRT_before_harmony_scanpy_celltype.png')

sc.pp.neighbors(adata_merged)
sc.tl.umap(adata_merged)
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_base_scanpy.pdf')
sc.pl.umap(adata_merged, color=['batch', 'SAHA_batchcode'], title="PRT UMAP after Harmony", save='_PRT_base_scanpy.png')


# 4. Visualize UMAP
sc.pl.umap(adata_merged, color='cell_type')  # or color='batch'

# 5. Find marker genes by organ
sc.tl.rank_genes_groups(adata_merged, groupby='SAHA_organcode', method='wilcoxon', reference='rest')
sc.pl.rank_genes_groups(adata_merged, n_genes=5, sharey=False)

# 6. Find marker genes by cell type
sc.tl.rank_genes_groups(adata_merged, groupby='Insitutype_Broad', method='wilcoxon', reference='rest')
sc.pl.rank_genes_groups(adata_merged, n_genes=5, sharey=False)

# 7. Extract results
df_markers = sc.get.rank_genes_groups_df(adata_merged, group=None)
df_markers.head()

N = 5
top_markers = (
    df_markers.groupby('group')            # group by organ
    .head(N)['names']                      # top N genes from each organ
    .unique()                              # unique gene names across organs
)
top_markers
sc.pl.dotplot(
    adata,
    var_names=markers,
    groupby='Insitutype_Broad',
    figsize=(10, 6),
    # optional tweaks:
    # dot_max=0.7,        # clamp the maximum dot size
    # dot_min=0.1,        # clamp the minimum dot size
    # cmap='Reds',        # color map
    # standard_scale='var' # scale gene expression across genes
)

markers = {
    "endothelial":       ["PECAM1","VWF", "ENG"],
    "epithelial":        ["EPCAM", "REG1A", "PRSS2", "KRT18"],
    "Erythroid":         ["HBB", "CD19", "CD36"],
    "B_cells":           ["CD74", "MS4A1","CD19","CD79A","CD22"],
    "dendritic_cells":   ["HLA-DPA1", "ITGAX","CD1C"],
    "plasma_cells":      ["IGHA1", "IGHG2", "MZB1", "JCHAIN"],
    "T_cells":           ["IL7R", "CD3D","CD3E","CD3G","CD2"],
    "mesenchymal_stromal":["TAGLN", "PDGFRA","PDGFRB","COL1A1","DCN"],
    "myeloid":           ["LYZ","CD14","CD68","CSF1R"],
    "neuronal_glial":    ["S100B","NGFR"],
    "stem_cells":        ["PIGR", "OLFM4", "POU5F1"]
}


### Squidpy L-R Analysis
res = sq.gr.ligrec(
    adata,
    n_perms=100,
    cluster_key="Insitutype_Broad",
    copy=True,
    use_raw=False,
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"},
    threshold=0.00001
)
sq.pl.ligrec(res, means_range=(0.5,1), alpha=0.005, remove_nonsig_interactions=False, remove_empty_interactions=True,pvalue_threshold=0.01,save="LR_GIall_ax.pdf"); plt.show()



### Get Subsets of a section 
### for example, APE lymphoid aggregates
adata_subset = adata[adata.obs['SAHA_FOV'].isin(["APE_PRT_EC01_32", "APE_PRT_EC01_33", "APE_PRT_EC01_37", "APE_PRT_EC01_38"])]
sq.gr.spatial_neighbors(adata_subset)
sq.gr.spatial_autocorr(
    adata_subset,
    mode="moran",
    n_perms=100,
    n_jobs=1,
)
adata_subset.uns["moranI"].head(10)

sq.gr.centrality_scores(adata_subset, celltype_id)
sq.pl.centrality_scores(adata_subset, celltype_id, save="SqCentrality_APE_EC01_32-38.pdf", figsize=(12,4)); plt.show()
sq.gr.co_occurrence(adata_subset, cluster_key=celltype_id)
sq.pl.co_occurrence(adata_subset, cluster_key=celltype_id, clusters="Lymphoid-B", save="SqCoOccurrence_B_APE_EC01_32-38.pdf"); plt.show()
res = sq.gr.ligrec(
    adata_subset,
    n_perms=100,
    cluster_key=celltype_id,
    copy=True,
    use_raw=False,
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"},
    threshold=0.00001
)
sq.pl.ligrec(res, means_range=(0.5,1), alpha=0.005, remove_nonsig_interactions=False, remove_empty_interactions=True, pvalue_threshold=0.01, swap_axes=False, save="LR_APE32_ax.pdf"); plt.show()
sq.pl.ligrec(res, means_range=(0.5,1), alpha=0.005, source_groups=("Lymphoid-B", "Lymphoid-T", "Lymphoid-DC", "Lymphoid-Other", "Lymphoid-Plasma", "Myeloid"), target_groups=("Lymphoid-B", "Lymphoid-T", "Lymphoid-DC", "Lymphoid-Other", "Lymphoid-Plasma", "Myeloid"), remove_nonsig_interactions=False, remove_empty_interactions=True, pvalue_threshold=0.01, swap_axes=False, save="LR_APE32_ax.pdf"); plt.show()



color_vars = [
    "leiden",
    "spatialclust_KNN_GIall",
    "Insitutype_Broad",
    "louvain",
    "spatialclust_CellType_GIall",
    "Insitutype_Labelled"
]

sq.pl.spatial_scatter(adata_subset, shape=None, color=color_vars, size=10)
plt.savefig("./figures/Sq_SpatialPlot_APE_EC01_32.pdf", format="pdf", bbox_inches="tight")
plt.close()

adata_filtered = adata_merged[adata_merged.obs['SAHA_name'] != 'SAHA_CP_COL_A1_TAP01'].copy()

