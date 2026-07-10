#!/usr/bin/env python
"""
DVAL-02: C2->C1 cell type label fuzzy matching.
Run: conda run -n spatial_integration python scripts/01_label_mapping.py
Writes:
  outputs/label_mapping_proposed.csv
  outputs/barcode_linkage_report.txt
"""
import os
import sys
import h5py
import pandas as pd
import scanpy as sc
from thefuzz import process

os.makedirs("outputs", exist_ok=True)

FUZZY_THRESHOLD = 85  # Scores below this go to UNMATCHED for manual review

# --- Step 1: Extract C1 labels from adata.joint.h5ad ---
print("Reading C1 labels from adata.joint.h5ad ...")
adata_joint = sc.read_h5ad("adata.joint.h5ad")
# Filter out empty-string labels — they are unlabelled cells, not a valid C1 type
c1_labels_raw = sorted(adata_joint.obs["cell_type"].dropna().unique().tolist())
c1_labels = [l for l in c1_labels_raw if str(l).strip() != ""]
print(f"  C1 labels ({len(c1_labels)}): {c1_labels}")
if len(c1_labels_raw) > len(c1_labels):
    n_empty = sum(1 for l in c1_labels_raw if str(l).strip() == "")
    print(f"  NOTE: {n_empty} empty-string label(s) excluded from C1 matching targets")

# --- Step 2: Extract C2 labels from SAHA_IBD_RNA.h5ad using h5py ---
# CRITICAL: Do NOT load .X — file is 10GB. Read obs/cell_type only.
print("\nReading C2 labels from SAHA_IBD_RNA.h5ad (obs only, no .X load) ...")
c2_labels_raw = []
c2_obs_columns = []
try:
    with h5py.File("SAHA_IBD_RNA.h5ad", "r") as f:
        obs_group = f["obs"]
        c2_obs_columns = list(obs_group.keys())
        print(f"  SAHA_IBD_RNA.h5ad obs columns: {c2_obs_columns}")

        if "cell_type" in obs_group:
            ct_data = obs_group["cell_type"]
            # Handle both string datasets and categorical (encoded) datasets
            if hasattr(ct_data, "keys"):
                # Categorical: has 'categories' and 'codes' subgroups
                cats = ct_data["categories"][:]
                cats_decoded = [c.decode() if isinstance(c, bytes) else c for c in cats]
                codes = ct_data["codes"][:]
                c2_labels_raw = [cats_decoded[i] for i in codes]
            else:
                raw = ct_data[:]
                c2_labels_raw = [c.decode() if isinstance(c, bytes) else c for c in raw]
        else:
            print("  WARNING: 'cell_type' not found in obs. Trying scanpy backed mode fallback ...")
            raise KeyError("cell_type not found in h5py obs group")

except (KeyError, Exception) as e:
    print(f"  h5py read failed ({e}), trying scanpy backed mode fallback ...")
    adata_rna = sc.read_h5ad("SAHA_IBD_RNA.h5ad", backed="r")
    c2_obs_columns = list(adata_rna.obs.columns)
    print(f"  SAHA_IBD_RNA.h5ad obs columns (backed): {c2_obs_columns}")
    if "cell_type" in adata_rna.obs.columns:
        c2_labels_raw = adata_rna.obs["cell_type"].dropna().tolist()
    else:
        print(f"  ERROR: 'cell_type' not found. Available obs columns: {c2_obs_columns}")
        adata_rna.file.close()
        sys.exit(1)
    adata_rna.file.close()

c2_labels = sorted(set(str(l) for l in c2_labels_raw if str(l).strip() != ""))
print(f"  C2 unique labels ({len(c2_labels)}): {c2_labels}")

# --- Step 3: Barcode linkage check (IBD.h5ad <-> SAHA_IBD_RNA.h5ad) ---
# NOTE: obs_names differ between files:
#   IBD.h5ad:         obs_names = SAHA_cellID (e.g. "SAHA_CR_IBD_A2_EC04_1_34")
#   SAHA_IBD_RNA.h5ad: obs_names = integer strings ("0", "1", ...)
# Shared linkage key: 'cell_id' column (values like "c_1_1_1") present in both files.
print("\nChecking barcode linkage: IBD.h5ad <-> SAHA_IBD_RNA.h5ad ...")
linkage_lines = ["Barcode Linkage Report: IBD.h5ad vs SAHA_IBD_RNA.h5ad"]
linkage_lines.append("=" * 50)

def read_h5ad_col(filepath, col):
    """Read a single obs column from an h5ad file without loading .X."""
    with h5py.File(filepath, "r") as f:
        obs = f["obs"]
        if col not in obs:
            return None, list(obs.keys())
        data = obs[col]
        if hasattr(data, "keys"):
            # Categorical encoding
            cats = data["categories"][:]
            cats_dec = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
            codes = data["codes"][:]
            values = [cats_dec[i] for i in codes]
        else:
            raw = data[:]
            values = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        return values, list(obs.keys())

ibd_cell_ids, ibd_obs_columns = read_h5ad_col("IBD.h5ad", "cell_id")
rna_cell_ids, rna_obs_columns = read_h5ad_col("SAHA_IBD_RNA.h5ad", "cell_id")

# Also read obs_names (index) for reference
def read_obs_index(filepath):
    with h5py.File(filepath, "r") as f:
        obs = f["obs"]
        idx_col = obs.attrs.get("_index", "_index")
        if idx_col in obs:
            raw = obs[idx_col][:]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in raw], idx_col
        return [], idx_col

ibd_index, ibd_idx_col = read_obs_index("IBD.h5ad")
rna_index, rna_idx_col = read_obs_index("SAHA_IBD_RNA.h5ad")

linkage_lines.append(f"IBD.h5ad obs_names key: '{ibd_idx_col}' ({len(ibd_index):,} cells)")
linkage_lines.append(f"  Sample: {ibd_index[:3]}")
linkage_lines.append(f"SAHA_IBD_RNA.h5ad obs_names key: '{rna_idx_col}' ({len(rna_index):,} cells)")
linkage_lines.append(f"  Sample: {rna_index[:3]}")
linkage_lines.append("")

if ibd_cell_ids is not None and rna_cell_ids is not None:
    ibd_set = set(ibd_cell_ids)
    rna_set = set(rna_cell_ids)
    overlap = ibd_set & rna_set
    pct = 100 * len(overlap) / max(len(ibd_set), len(rna_set)) if max(len(ibd_set), len(rna_set)) > 0 else 0
    linkage_lines.append(f"Linkage key: 'cell_id' column")
    linkage_lines.append(f"IBD.h5ad unique cell_ids: {len(ibd_set):,}")
    linkage_lines.append(f"SAHA_IBD_RNA.h5ad unique cell_ids: {len(rna_set):,}")
    linkage_lines.append(f"Shared cell_ids: {len(overlap):,} ({pct:.1f}%)")
    if len(overlap) > 0:
        linkage_lines.append("Linkage method: 'cell_id' obs column (shared between both files)")
        linkage_lines.append(f"  Sample shared ids: {sorted(overlap)[:5]}")
    else:
        linkage_lines.append("WARNING: No overlap on 'cell_id'. Files may contain different cell populations.")
        linkage_lines.append(f"  Sample IBD cell_ids: {sorted(ibd_set)[:5]}")
        linkage_lines.append(f"  Sample RNA cell_ids: {sorted(rna_set)[:5]}")
else:
    linkage_lines.append("WARNING: 'cell_id' column not found in one or both files.")
    linkage_lines.append(f"  IBD.h5ad obs columns: {ibd_obs_columns[:10]}")
    linkage_lines.append(f"  SAHA_IBD_RNA.h5ad obs columns: {rna_obs_columns[:10]}")

linkage_lines.append(f"\nIBD.h5ad obs columns ({len(ibd_obs_columns)}): {ibd_obs_columns[:15]}...")

linkage_text = "\n".join(linkage_lines)
with open("outputs/barcode_linkage_report.txt", "w") as f:
    f.write(linkage_text + "\n")
print(linkage_text)

# --- Step 4: Fuzzy label matching ---
print(f"\nBuilding C2->C1 fuzzy mapping (threshold={FUZZY_THRESHOLD}) ...")

rows = []
for c2 in c2_labels:
    result = process.extractOne(c2, c1_labels)
    if result and result[1] >= FUZZY_THRESHOLD:
        rows.append({"c2": c2, "c1_proposed": result[0], "score": result[1], "best_candidate": result[0]})
    else:
        best_match = result[0] if result else ""
        best_score = result[1] if result else 0
        rows.append({
            "c2": c2,
            "c1_proposed": "UNMATCHED",
            "score": best_score,
            "best_candidate": best_match,
        })

df = pd.DataFrame(rows)
# Ensure consistent column order
for col in ["c2", "c1_proposed", "score", "best_candidate"]:
    if col not in df.columns:
        df[col] = ""
df = df[["c2", "c1_proposed", "score", "best_candidate"]]

df.to_csv("outputs/label_mapping_proposed.csv", index=False)

matched = df[df["c1_proposed"] != "UNMATCHED"]
unmatched = df[df["c1_proposed"] == "UNMATCHED"]
print(f"\nMatched: {len(matched)} / {len(df)} labels")
print(f"Unmatched (require manual review): {len(unmatched)}")
if len(unmatched) > 0:
    print("  Unmatched C2 labels:")
    for _, row in unmatched.iterrows():
        print(f"    '{row['c2']}' (best candidate: '{row['best_candidate']}', score: {row['score']})")

print(f"\nWrote: outputs/label_mapping_proposed.csv")
print("\nDVAL-02: PASSED — review outputs/label_mapping_proposed.csv before Plan 04 checkpoint")
