#!/usr/bin/env python3
"""
02_rebuild_from_rds.py
Build C1+C2 AnnData from:
  - C1: extracted RDS files (outputs/rds_extracted/cosmx_c1_*.mtx.gz, xenium_c1_*.mtx.gz)
  - C2: SAHA_IBD_RNA.h5ad layers['raw'] (integer counts)

The plan spec uses prefix names cosmx6k_1..4 / xenium5k_1..4.
The actual extracted files use cosmx_c1_1..4 / xenium_c1_1..4 — this script
maps between them automatically.

Output: outputs/adata_c1_c2_raw.h5ad

Run: conda run -n spatial_integration python scripts/02_rebuild_from_rds.py
Working directory: project root (IBD_integration/)
"""

import os
import gzip
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import anndata as ad
import h5py

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

EXTRACTED = "outputs/rds_extracted"
SAHA_PATH = "SAHA_IBD_RNA.h5ad"
OUT_RAW   = "outputs/adata_c1_c2_raw.h5ad"

os.makedirs("outputs", exist_ok=True)

# ─── Part A: Load C1 from extracted RDS files ─────────────────────────────────
# Files on disk use prefix cosmx_c1_1..4 / xenium_c1_1..4
# (plan spec uses cosmx6k_1..4 / xenium5k_1..4 — same data, different naming)

FILE_SPECS = [
    # (disk_prefix,    platform,  dataset)
    ("cosmx_c1_1",  "cosmx",   "cosmx_c1"),
    ("cosmx_c1_2",  "cosmx",   "cosmx_c1"),
    ("cosmx_c1_3",  "cosmx",   "cosmx_c1"),
    ("cosmx_c1_4",  "cosmx",   "cosmx_c1"),
    ("xenium_c1_1", "xenium",  "xenium_c1"),
    ("xenium_c1_2", "xenium",  "xenium_c1"),
    ("xenium_c1_3", "xenium",  "xenium_c1"),
    ("xenium_c1_4", "xenium",  "xenium_c1"),
]

c1_adatas = []

for disk_prefix, platform, dataset in FILE_SPECS:
    print(f"\n--- Loading {disk_prefix} ---")

    mtx_path      = os.path.join(EXTRACTED, f"{disk_prefix}_counts.mtx.gz")
    genes_path    = os.path.join(EXTRACTED, f"{disk_prefix}_genes.csv")
    barcodes_path = os.path.join(EXTRACTED, f"{disk_prefix}_barcodes.csv")
    meta_path     = os.path.join(EXTRACTED, f"{disk_prefix}_meta.csv")

    # Sparse matrix is genes×cells (as written by R writeMM) → transpose to cells×genes
    with gzip.open(mtx_path, "rb") as gz:
        mat = scipy.io.mmread(gz)
    mat = scipy.sparse.csr_matrix(mat.T)   # cells × genes

    genes    = pd.read_csv(genes_path)["gene"].tolist()
    barcodes = pd.read_csv(barcodes_path)["barcode"].tolist()
    meta_df  = pd.read_csv(meta_path)

    print(f"  Matrix: {mat.shape[0]} cells × {mat.shape[1]} genes")
    print(f"  Genes sample: {genes[:3]}")

    # Unique obs names: prefix_barcode
    obs_names = [f"{disk_prefix}_{b}" for b in barcodes]

    adata = ad.AnnData(
        X   = mat,
        obs = pd.DataFrame(index=obs_names),
        var = pd.DataFrame(index=genes),
    )

    meta_df.index = obs_names

    adata.obs["cell_id"]  = meta_df["cell_id"].values if "cell_id" in meta_df.columns else barcodes
    adata.obs["ct_major"] = meta_df["ct_major"].values
    adata.obs["platform"] = platform
    adata.obs["dataset"]  = dataset

    if "fov" in meta_df.columns:
        adata.obs["fov"] = meta_df["fov"].values
    if "Run_Tissue_name" in meta_df.columns:
        adata.obs["Run_Tissue_name"] = meta_df["Run_Tissue_name"].values

    # Verify integer counts
    nz_vals = mat.data[:min(1000, len(mat.data))]
    is_int = bool(np.all(nz_vals % 1 == 0))
    print(f"  Integer counts: {is_int}  |  ct_major: {sorted(adata.obs['ct_major'].unique())}")

    c1_adatas.append(adata)

print("\n=== Concatenating C1 ===")
c1 = ad.concat(c1_adatas, join="outer", fill_value=0)
c1.obs_names_make_unique()
print(f"C1 shape: {c1.shape}")
print(f"C1 ct_major:\n{c1.obs['ct_major'].value_counts().to_string()}")
print(f"C1 dataset:\n{c1.obs['dataset'].value_counts().to_string()}")

del c1_adatas

# ─── Part B: Load C2 from SAHA_IBD_RNA.h5ad layers['raw'] ────────────────────
print("\n=== Loading C2 from SAHA_IBD_RNA.h5ad ===")

# Step 1: read obs metadata (memory-efficient)
print("  Reading obs metadata ...")
with h5py.File(SAHA_PATH, "r") as f:
    obs_keys = list(f["obs"].keys())
    print(f"  obs keys: {obs_keys}")

    def read_categorical(f, key):
        cats  = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["obs"][key]["categories"][:]]
        codes = f["obs"][key]["codes"][:]
        return pd.Categorical.from_codes(codes, cats)

    def read_string_or_cat(f, key):
        node = f["obs"][key]
        if "categories" in node:
            return list(read_categorical(f, key))
        raw = node[:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

    idx_key = f["obs"].attrs.get("_index", "index")
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode()
    print(f"  obs index key: {idx_key}")

    obs_names_saha = read_string_or_cat(f, idx_key)
    cell_type      = read_categorical(f, "cell_type")
    cell_type_general = read_categorical(f, "cell_type_general") \
        if "cell_type_general" in f["obs"] else None
    cell_id_saha   = read_string_or_cat(f, "cell_id") \
        if "cell_id" in f["obs"] else obs_names_saha

    # Get var names from SAHA file
    var_idx_key = f["var"].attrs.get("_index", "index")
    if isinstance(var_idx_key, bytes):
        var_idx_key = var_idx_key.decode()
    v_node = f["var"][var_idx_key]
    if hasattr(v_node, "keys"):
        var_cats  = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in v_node["categories"][:]]
        var_codes = v_node["codes"][:]
        saha_var_names = [var_cats[c] for c in var_codes]
    else:
        saha_var_names = [x.decode() if isinstance(x, bytes) else str(x)
                          for x in v_node[:]]
    print(f"  SAHA var count: {len(saha_var_names)}, sample: {saha_var_names[:3]}")

saha_obs = pd.DataFrame({
    "cell_id":   cell_id_saha,
    "cell_type": [str(x) for x in cell_type],
}, index=obs_names_saha)
if cell_type_general is not None:
    saha_obs["cell_type_general"] = [str(x) for x in cell_type_general]

print(f"  SAHA obs: {len(saha_obs)} cells")
print(f"  cell_type unique: {sorted(saha_obs['cell_type'].astype(str).unique())}")

# Step 2: Map cell_type -> ct_major
print("\n  Mapping cell_type -> ct_major ...")

base_map = {
    "Epithelial":          "Epithelial",
    "Mesenchymal-Stromal": "Mesenchymal",
    "Lymphoid-B":          "B",
    "Lymphoid-T":          "T/NK",
    "Lymphoid-Plasma":     "Plasma",
    "Myeloid":             "Myeloid",
    "Immune-Other":        "Myeloid",
    "Stem":                "Epithelial",
    "Mixed":               "Unknown",
    "":                    "Unknown",
}

def map_ct_major(row):
    ct  = str(row["cell_type"])
    ctg = str(row.get("cell_type_general", "")) if "cell_type_general" in row.index else ""
    if ct == "Lymphoid-Other":
        if ctg == "APC":
            return "Myeloid"
        elif ctg == "Lymphoid":
            return "T/NK"
        else:
            return "Unknown"
    return base_map.get(ct, "Unknown")

saha_obs["ct_major"] = saha_obs.apply(map_ct_major, axis=1)
print(f"  SAHA ct_major:\n{saha_obs['ct_major'].value_counts().to_string()}")

# Step 3: Subsample to ~100K (stratified by ct_major)
TARGET_C2 = 100_000
n_saha = len(saha_obs)
print(f"\n  Subsampling {n_saha} -> ~{TARGET_C2} (stratified by ct_major) ...")
np.random.seed(42)
frac = min(1.0, TARGET_C2 / n_saha)
groups = []
for ct, grp in saha_obs.groupby("ct_major"):
    n_keep = max(1, round(len(grp) * frac))
    sampled = grp.sample(n=min(n_keep, len(grp)), random_state=42)
    groups.append(sampled)
saha_sub = pd.concat(groups)
if len(saha_sub) > TARGET_C2:
    saha_sub = saha_sub.sample(n=TARGET_C2, random_state=42)
print(f"  Subsampled: {len(saha_sub)} cells")
print(f"  C2 ct_major after subsample:\n{saha_sub['ct_major'].value_counts().to_string()}")

# Step 4: Load layers['raw'] for selected cells
print(f"\n  Loading layers['raw'] for {len(saha_sub)} cells ...")
all_obs_names_saha = list(obs_names_saha)
obs_name_to_idx    = {name: i for i, name in enumerate(all_obs_names_saha)}
selected_indices   = sorted([obs_name_to_idx[n] for n in saha_sub.index])
selected_arr       = np.array(selected_indices)

with h5py.File(SAHA_PATH, "r") as f:
    raw_node = f["layers"]["raw"]
    data    = raw_node["data"][:]
    indices = raw_node["indices"][:]
    indptr  = raw_node["indptr"][:]
    shape_attr = raw_node.attrs.get("shape", None)
    if shape_attr is not None:
        shape = tuple(shape_attr)
    else:
        shape = (len(indptr) - 1, int(indices.max()) + 1)

print(f"  Full raw matrix shape: {shape}")
raw_full = scipy.sparse.csr_matrix((data.astype(np.float32), indices, indptr), shape=shape)
saha_mat = raw_full[selected_arr, :]
print(f"  Sliced C2 matrix shape: {saha_mat.shape}")

# Verify integer counts
nz = saha_mat.data[:min(1000, len(saha_mat.data))]
print(f"  Integer check: {bool(np.all(nz % 1 == 0))}")
print(f"  C2 max count: {saha_mat.data.max():.0f}")

del raw_full

# Step 5: Build C2 AnnData
c2_obs = saha_sub[["cell_id", "ct_major"]].copy()
c2_obs["platform"] = "cosmx"
c2_obs["dataset"]  = "cosmx_c2"

c2 = ad.AnnData(
    X   = scipy.sparse.csr_matrix(saha_mat, dtype=np.float32),
    obs = c2_obs,
    var = pd.DataFrame(index=saha_var_names),
)
c2.obs_names = list(saha_sub.index)
print(f"C2 shape: {c2.shape}")

# ─── Part C: Concatenate C1 + C2 (inner join on shared genes) ────────────────
print("\n=== Concatenating C1 + C2 (inner join) ===")
print(f"  C1 shape: {c1.shape}")
print(f"  C2 shape: {c2.shape}")

c1_genes = set(c1.var_names)
c2_genes = set(c2.var_names)
shared   = c1_genes & c2_genes
print(f"  Shared genes (inner join): {len(shared)}")
print(f"  C1-only genes: {len(c1_genes - c2_genes)}")
print(f"  C2-only genes: {len(c2_genes - c1_genes)}")

adata_all = ad.concat([c1, c2], join="inner", fill_value=0)
adata_all.obs_names_make_unique()

print(f"\nJoint shape: {adata_all.shape}")
print(f"\nct_major × platform cross-tab:")
print(pd.crosstab(adata_all.obs["ct_major"], adata_all.obs["platform"]).to_string())
print(f"\nct_major × dataset cross-tab:")
print(pd.crosstab(adata_all.obs["ct_major"], adata_all.obs["dataset"]).to_string())

# Validate
for col in ["dataset", "platform", "ct_major"]:
    assert col in adata_all.obs.columns, f"Missing column: {col}"
    n_empty = (adata_all.obs[col].astype(str) == "").sum()
    assert n_empty == 0, f"Column {col} has {n_empty} empty values"

valid_ct = {"Epithelial", "Mesenchymal", "B", "T/NK", "Plasma",
            "Myeloid", "Endothelial", "Neuron", "Unknown"}
actual_ct = set(adata_all.obs["ct_major"].unique())
unexpected = actual_ct - valid_ct
assert len(unexpected) == 0, f"Unexpected ct_major: {unexpected}"
print("\nValidation PASSED")

# ─── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_RAW} ...")
adata_all.write_h5ad(OUT_RAW)
print(f"Done. Shape: {adata_all.shape}")
print(f"obs columns: {list(adata_all.obs.columns)}")
