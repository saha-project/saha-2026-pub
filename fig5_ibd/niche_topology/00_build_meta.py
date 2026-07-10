"""00 — Build metadata-joined AnnData.

Reads ../IBD_integration/outputs/adata_c1heavy.h5ad and left-joins per-cell
metadata from dulai_percell_meta.csv (C1) + saha_c2_percell_meta.csv (C2),
then writes outputs/adata_c1heavy_meta.h5ad with the harmonized columns:

    cohort_platform, tissue_type, disease, disease_state, patient_id,
    block, fov, c2_section_id, responder_status,
    insitutype_labelled, insitutype_broad, c2_cell_type_fine

obs.index in adata_c1heavy is itself the `cell_key` used by both metadata
CSVs — confirmed empirically.

Run: python scripts/00_build_meta.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()

t0 = time.time()
print(f"[00] reading {os.path.basename(config.ADATA_C1HEAVY)} ...")
adata = ad.read_h5ad(config.ADATA_C1HEAVY)
print(f"     loaded {adata.shape} in {time.time()-t0:.1f}s")

obs = adata.obs.copy()
obs.index.name = "cell_key"

# ── C1 metadata join (Dulai: cosmx_c1_* + xenium_c1_*) ──────────────────────
print("[00] joining Dulai C1 metadata ...")
c1_meta = pd.read_csv(
    config.DULAI_META_CSV,
    usecols=["cell_key", "sample_id", "block", "tissue_type",
             "disease", "disease_state", "fov"],
).set_index("cell_key")
c1_join = obs.join(c1_meta, how="left")
print(f"     C1 cells with metadata: "
      f"{c1_join['block'].notna().sum():,} / "
      f"{(obs['dataset'] != 'cosmx_c2').sum():,} expected C1")

# ── C2 metadata join (SAHA c2) ──────────────────────────────────────────────
print("[00] joining SAHA C2 metadata ...")
c2_meta = pd.read_csv(
    config.SAHA_META_CSV,
    usecols=["cell_key", "tissue_type", "disease", "disease_state",
             "SAHA_Sample", "section_ID", "fov",
             "Insitutype_Labelled", "Insitutype_Broad",
             "cell_type", "cell_type_general"],
).set_index("cell_key")
c2_meta = c2_meta.rename(columns={
    "section_ID": "c2_section_id",
    "Insitutype_Labelled": "insitutype_labelled",
    "Insitutype_Broad": "insitutype_broad",
    "cell_type": "c2_cell_type_fine",
    "cell_type_general": "c2_cell_type_general",
})

# Coalesce — fill C1 NaNs with C2 values for shared columns
shared_cols = ["tissue_type", "disease", "disease_state", "fov"]
joined = c1_join.copy()
joined = joined.join(c2_meta[[c for c in c2_meta.columns if c not in shared_cols]],
                     how="left")
for col in shared_cols:
    if col in c2_meta.columns:
        joined[col] = joined[col].combine_first(
            c2_meta[col].reindex(joined.index)
        )

print(f"     C2 cells with section_ID: "
      f"{joined['c2_section_id'].notna().sum():,} / "
      f"{(obs['dataset'] == 'cosmx_c2').sum():,} expected C2")

# ── Derived columns ─────────────────────────────────────────────────────────
print("[00] deriving cohort_platform / patient_id / responder_status ...")

# cohort_platform from dataset
joined["cohort_platform"] = (
    joined["dataset"].astype(str).map(config.DATASET_TO_CP)
)

# patient_id: C1 → Dulai patient (1..4 from sample_id); C2 → section_ID
def _derive_patient_id(row):
    ds = str(row["dataset"])
    if ds in ("cosmx_c1", "xenium_c1"):
        sid = row.get("sample_id")
        if pd.isna(sid):
            return np.nan
        try:
            n = int(sid)
            return f"C1_P{n}"
        except (ValueError, TypeError):
            return f"C1_{sid}"
    if ds == "cosmx_c2":
        sec = row.get("c2_section_id")
        return sec if isinstance(sec, str) else np.nan
    return np.nan

joined["patient_id"] = joined.apply(_derive_patient_id, axis=1)

# responder_status:
#  - C1: "Healthy" / "IBD-no-label" by disease
#  - C2 A1_*: "IBD-no-label"
#  - C2 A2_1, A2_2: "Pending" until user supplies labels
def _derive_responder(row):
    ds = str(row["dataset"])
    if ds == "cosmx_c2":
        sec = row.get("c2_section_id")
        if isinstance(sec, str) and sec.startswith("IBD_A2"):
            return "Pending"
        return "IBD-no-label"
    # C1
    dis = row.get("disease")
    if dis == "Healthy":
        return "Healthy"
    return "IBD-no-label"

# Optional override from user-supplied labels CSV
labels_override: dict[str, str] = {}
if os.path.exists(config.RESPONDER_LABELS_CSV):
    rl = pd.read_csv(config.RESPONDER_LABELS_CSV)
    labels_override = dict(zip(rl["section_id"], rl["responder_status"]))
    print(f"     loaded responder labels: {labels_override}")

def _final_responder(row):
    sec = row.get("c2_section_id")
    if isinstance(sec, str) and sec in labels_override:
        return labels_override[sec]
    return _derive_responder(row)

joined["responder_status"] = joined.apply(_final_responder, axis=1)

# C2 disease_state was empty in source CSV — default to "IBD"
c2_mask = joined["dataset"].astype(str) == "cosmx_c2"
joined.loc[c2_mask & joined["disease_state"].isna(), "disease_state"] = "IBD"

# ── Categorical orderings ───────────────────────────────────────────────────
joined["cohort_platform"] = pd.Categorical(
    joined["cohort_platform"], categories=config.COHORT_PLATFORMS, ordered=True,
)
joined["ct_major"] = pd.Categorical(
    joined["ct_major"].astype(str), categories=config.CT_MAJOR_ORDER, ordered=True,
)

# ── Sanity checks ───────────────────────────────────────────────────────────
print("\n[00] coverage summary:")
print(joined["cohort_platform"].value_counts(dropna=False).to_string())
print("\n[00] tissue_type:")
print(joined["tissue_type"].value_counts(dropna=False).to_string())
print("\n[00] disease_state:")
print(joined["disease_state"].value_counts(dropna=False).to_string())
print("\n[00] responder_status:")
print(joined["responder_status"].value_counts(dropna=False).to_string())
print("\n[00] patient_id (top 12):")
print(joined["patient_id"].value_counts(dropna=False).head(12).to_string())

n_missing_cp = joined["cohort_platform"].isna().sum()
n_missing_tt = joined["tissue_type"].isna().sum()
assert n_missing_cp == 0, f"{n_missing_cp:,} cells missing cohort_platform"
print(f"\n[00] OK: 0 cells missing cohort_platform; "
      f"{n_missing_tt:,} cells missing tissue_type "
      f"(expected only for C2 cells without csv coverage).")

# ── Persist ─────────────────────────────────────────────────────────────────
adata.obs = joined
print(f"\n[00] writing {config.ADATA_META} ...")
adata.write_h5ad(config.ADATA_META, compression="gzip")
print(f"[00] done in {time.time()-t0:.1f}s — output: {config.ADATA_META}")
