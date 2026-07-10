#!/usr/bin/env python3
"""
02_build_adata.py
Build C1 AnnData from extracted RDS matrices, then add C2 (SAHA) data.

Steps:
  1. Read each of 8 extracted MTX files into AnnData
  2. Add platform ('cosmx' or 'xenium') and dataset ('cosmx_c1' or 'xenium_c1')
  3. Add ct_major from metadata
  4. Concatenate all 8 into one C1 AnnData
  5. Load SAHA_IBD_RNA.h5ad (C2), map cell_type -> ct_major
  6. Subsample C2 to ~100K cells (stratified by ct_major)
  7. Add platform='cosmx', dataset='cosmx_c2'
  8. Concatenate C1 + C2 -> outputs/adata_c1c2.h5ad

Run: conda run -n spatial_integration python scripts/02_build_adata.py
Working directory: project root (IBD_integration/)
"""

import os
import gzip
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import anndata

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

EXTRACTED = "outputs/rds_extracted"
OUT_C1C2  = "outputs/adata_c1c2.h5ad"

# ─── 1. Build C1 AnnData from extracted files ────────────────────────────────

FILE_SPECS = [
    # (prefix,          platform,  dataset)
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

for prefix, platform, dataset in FILE_SPECS:
    print(f"\n--- Loading {prefix} ---")

    mtx_path      = os.path.join(EXTRACTED, f"{prefix}_counts.mtx.gz")
    genes_path    = os.path.join(EXTRACTED, f"{prefix}_genes.csv")
    barcodes_path = os.path.join(EXTRACTED, f"{prefix}_barcodes.csv")
    meta_path     = os.path.join(EXTRACTED, f"{prefix}_meta.csv")

    # Read sparse matrix (genes x cells) -> transpose to cells x genes
    with gzip.open(mtx_path, "rb") as gz:
        mat = scipy.io.mmread(gz)
    mat = scipy.sparse.csr_matrix(mat.T)   # cells x genes

    genes    = pd.read_csv(genes_path)["gene"].tolist()
    barcodes = pd.read_csv(barcodes_path)["barcode"].tolist()
    meta_df  = pd.read_csv(meta_path)

    print(f"  Matrix: {mat.shape[0]} cells x {mat.shape[1]} genes")
    print(f"  Genes: {len(genes)}, Barcodes: {len(barcodes)}")

    # Make barcodes unique across files by prepending prefix
    obs_names = [f"{prefix}_{b}" for b in barcodes]

    # Build AnnData
    adata = anndata.AnnData(
        X   = mat,
        obs = pd.DataFrame(index=obs_names),
        var = pd.DataFrame(index=genes),
    )

    # Add metadata
    # meta_df rows are in same order as barcodes (from R write.csv)
    meta_df.index = obs_names

    adata.obs["cell_id"]  = meta_df["cell_id"].values if "cell_id" in meta_df.columns else barcodes
    adata.obs["ct_major"] = meta_df["ct_major"].values
    adata.obs["platform"] = platform
    adata.obs["dataset"]  = dataset

    # Optional QC columns
    if "fov" in meta_df.columns:
        adata.obs["fov"] = meta_df["fov"].values
    if "Run_Tissue_name" in meta_df.columns:
        adata.obs["Run_Tissue_name"] = meta_df["Run_Tissue_name"].values
    if "Area" in meta_df.columns:
        adata.obs["Area"] = meta_df["Area"].values
    if "nCount_Nanostring" in meta_df.columns:
        adata.obs["nCount"] = meta_df["nCount_Nanostring"].values
    if "nCount_Xenium" in meta_df.columns:
        adata.obs["nCount"] = meta_df["nCount_Xenium"].values

    print(f"  ct_major: {sorted(adata.obs['ct_major'].unique())}")
    print(f"  dataset={dataset}, platform={platform}")
    print(f"  AnnData shape: {adata.shape}")

    c1_adatas.append(adata)

print("\n=== Concatenating C1 AnnData ===")
c1 = anndata.concat(c1_adatas, join="outer", fill_value=0)
c1.obs_names_make_unique()
print(f"C1 shape: {c1.shape}")
print("C1 ct_major distribution:")
print(c1.obs["ct_major"].value_counts().to_string())
print("C1 dataset distribution:")
print(c1.obs["dataset"].value_counts().to_string())

# ─── 2. Add C2 (SAHA) data ───────────────────────────────────────────────────

print("\n=== Loading SAHA C2 data ===")
SAHA_PATH = "SAHA_IBD_RNA.h5ad"

# Load obs metadata only first (memory-efficient for 10GB file)
print("  Reading obs metadata from SAHA_IBD_RNA.h5ad ...")
with h5py.File(SAHA_PATH, "r") as f:
    obs_keys = list(f["obs"].keys())
    print(f"  obs keys: {obs_keys}")

    def read_categorical(f, key):
        cats = [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["obs"][key]["categories"][:]]
        codes = f["obs"][key]["codes"][:]
        return pd.Categorical.from_codes(codes, cats)

    def read_string_or_cat(f, key):
        node = f["obs"][key]
        if "categories" in node:
            return list(read_categorical(f, key))
        else:
            # plain string dataset
            raw = node[:]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]

    # obs_names
    idx_key = f["obs"].attrs.get("_index", "index")
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode()
    print(f"  obs index key: {idx_key}")

    if idx_key in f["obs"]:
        obs_names = read_string_or_cat(f, idx_key)
    else:
        obs_names = [str(i) for i in range(f["obs"]["cell_id"]["codes"].shape[0])]

    # Required obs columns
    cell_type = read_categorical(f, "cell_type")

    # cell_type_general for Lymphoid-Other split
    cell_type_general = read_categorical(f, "cell_type_general") \
        if "cell_type_general" in f["obs"] else None

    # cell_id for linkage
    cell_id = read_string_or_cat(f, "cell_id") \
        if "cell_id" in f["obs"] else obs_names

saha_obs = pd.DataFrame({
    "cell_id":   cell_id,
    "cell_type": [str(x) if (x is not None and str(x) != "nan") else "" for x in cell_type],
}, index=obs_names)
if cell_type_general is not None:
    saha_obs["cell_type_general"] = [str(x) if (x is not None and str(x) != "nan") else "" for x in cell_type_general]

print(f"  SAHA obs rows: {len(saha_obs)}")
print(f"  cell_type unique: {sorted(saha_obs['cell_type'].astype(str).unique())}")

# ─── Map cell_type -> ct_major ────────────────────────────────────────────────
print("\n  Mapping cell_type -> ct_major ...")

# Base mapping (non-Lymphoid-Other)
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
print("  SAHA ct_major distribution:")
print(saha_obs["ct_major"].value_counts().to_string())

# ─── Subsample C2 to ~100K cells (stratified by ct_major) ────────────────────
TARGET_C2 = 100_000
n_saha = len(saha_obs)
print(f"\n  Subsampling {n_saha} -> {TARGET_C2} cells (stratified by ct_major) ...")

frac = min(1.0, TARGET_C2 / n_saha)
np.random.seed(42)

# Stratified subsample
groups = []
for ct, grp in saha_obs.groupby("ct_major"):
    n_keep = max(1, round(len(grp) * frac))
    sampled = grp.sample(n=min(n_keep, len(grp)), random_state=42)
    groups.append(sampled)
saha_sub = pd.concat(groups)

# Trim to exactly TARGET_C2 if slightly over
if len(saha_sub) > TARGET_C2:
    saha_sub = saha_sub.sample(n=TARGET_C2, random_state=42)

print(f"  Subsampled: {len(saha_sub)} cells")
print("  C2 ct_major after subsample:")
print(saha_sub["ct_major"].value_counts().to_string())

# ─── Load expression matrix for subsampled C2 cells ──────────────────────────
print("\n  Loading SAHA expression matrix for subsampled cells ...")

# Get integer positions of selected cells in the full obs
all_obs_names = list(obs_names)
obs_name_to_idx = {name: i for i, name in enumerate(all_obs_names)}
selected_indices = sorted([obs_name_to_idx[n] for n in saha_sub.index])
selected_indices_arr = np.array(selected_indices)

print(f"  Reading {len(selected_indices)} rows from SAHA_IBD_RNA.h5ad ...")

with h5py.File(SAHA_PATH, "r") as f:
    # Determine X storage format
    x_node = f["X"]
    if "data" in x_node:
        # CSR or CSC sparse
        data    = x_node["data"][:]
        indices = x_node["indices"][:]
        indptr  = x_node["indptr"][:]
        shape   = tuple(x_node.attrs["shape"]) if "shape" in x_node.attrs \
                  else (len(indptr) - 1, int(indices.max()) + 1)
        enc_raw = x_node.attrs.get("encoding-type", "") if "encoding-type" in x_node.attrs else ""
        encoding = enc_raw.decode() if isinstance(enc_raw, bytes) else str(enc_raw)
        if encoding == "csr_matrix" or indptr.shape[0] == shape[0] + 1:
            full_mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            full_mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
            full_mat = full_mat.tocsr()
    else:
        # Dense array — load all then slice
        full_mat = f["X"][:]
        full_mat = scipy.sparse.csr_matrix(full_mat)

    # Get var names
    var_idx_key = f["var"].attrs.get("_index", "index")
    if isinstance(var_idx_key, bytes):
        var_idx_key = var_idx_key.decode()
    if var_idx_key in f["var"]:
        v = f["var"][var_idx_key]
        if hasattr(v, "keys"):  # categorical
            var_cats = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in v["categories"][:]]
            var_codes = v["codes"][:]
            var_names = [var_cats[c] for c in var_codes]
        else:
            var_names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in v[:]]
    else:
        var_names = [str(i) for i in range(full_mat.shape[1])]

print(f"  Full matrix shape: {full_mat.shape}")
print(f"  Slicing {len(selected_indices_arr)} rows ...")
saha_mat = full_mat[selected_indices_arr, :]
print(f"  C2 matrix shape: {saha_mat.shape}")

# Build C2 AnnData
c2_obs = saha_sub.copy()
c2_obs["platform"] = "cosmx"
c2_obs["dataset"]  = "cosmx_c2"

c2 = anndata.AnnData(
    X   = scipy.sparse.csr_matrix(saha_mat),
    obs = c2_obs,
    var = pd.DataFrame(index=var_names),
)

# Ensure obs_names match
c2.obs_names = list(saha_sub.index)
print(f"C2 shape: {c2.shape}")
print("C2 obs columns:", list(c2.obs.columns))

# ─── 3. Concatenate C1 + C2 ──────────────────────────────────────────────────
print("\n=== Concatenating C1 + C2 ===")
print(f"  C1 shape: {c1.shape}")
print(f"  C2 shape: {c2.shape}")

# Align on shared genes (outer join fills missing with 0)
joint = anndata.concat([c1, c2], join="outer", fill_value=0)
joint.obs_names_make_unique()

print(f"\nJoint shape: {joint.shape}")
print("Joint obs columns:", list(joint.obs.columns))
print("\nJoint dataset distribution:")
print(joint.obs["dataset"].value_counts().to_string())
print("\nJoint platform distribution:")
print(joint.obs["platform"].value_counts().to_string())
print("\nJoint ct_major distribution:")
print(joint.obs["ct_major"].value_counts().to_string())

# Validate required columns
for col in ["dataset", "platform", "ct_major"]:
    assert col in joint.obs.columns, f"Missing column: {col}"
    n_empty = (joint.obs[col].astype(str) == "").sum()
    assert n_empty == 0, f"Column {col} has {n_empty} empty values"

valid_ct = {"Epithelial", "Mesenchymal", "B", "T/NK", "Plasma",
            "Myeloid", "Endothelial", "Neuron", "Unknown"}
actual_ct = set(joint.obs["ct_major"].unique())
unexpected = actual_ct - valid_ct
assert len(unexpected) == 0, f"Unexpected ct_major values: {unexpected}"

print("\nValidation PASSED")
print(f"  ct_major values: {sorted(actual_ct)}")

# ─── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_C1C2} ...")
joint.write_h5ad(OUT_C1C2)
print(f"Done. Shape: {joint.shape}")
