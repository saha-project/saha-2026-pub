#!/usr/bin/env python3
"""
02_umap_balanced_rebuild.py
Generate balanced UMAP panels from the rebuilt C1+C2 scANVI AnnData.

Balanced = all Xenium cells + subsample CosMx to match Xenium count.
Computes UMAP for each of 4 embeddings:
  X_scVI_dataset, X_scVI_platform, X_scANVI_dataset, X_scANVI_platform

Colors by: platform, dataset, ct_major  → 12 PNG files total

Input:  outputs/adata_scanvi.h5ad
Output: outputs/figures/umap_balanced_{embedding}_by_{color}.png  (12 files)

Run: conda run -n spatial_integration python scripts/02_umap_balanced_rebuild.py
Working directory: project root (IBD_integration/)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)
os.makedirs("outputs/figures", exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading outputs/adata_scanvi.h5ad ...")
adata = sc.read_h5ad("outputs/adata_scanvi.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  platform:\n{adata.obs['platform'].value_counts().to_string()}")
print(f"  dataset:\n{adata.obs['dataset'].value_counts().to_string()}")
print(f"  ct_major:\n{adata.obs['ct_major'].value_counts().to_string()}")

# ─── Build balanced subset ─────────────────────────────────────────────────────
print("\n=== Building balanced subset (equal CosMx / Xenium) ===")
xenium_mask = adata.obs["platform"] == "xenium"
cosmx_mask  = adata.obs["platform"] == "cosmx"

n_xenium = int(xenium_mask.sum())
n_cosmx  = int(cosmx_mask.sum())
print(f"  Xenium cells: {n_xenium}")
print(f"  CosMx cells:  {n_cosmx}")

# Take all Xenium + subsample CosMx to match
np.random.seed(42)
xenium_idx = np.where(xenium_mask)[0]
cosmx_idx  = np.where(cosmx_mask)[0]
cosmx_sub_idx = np.random.choice(cosmx_idx, size=n_xenium, replace=False)

balanced_idx = np.concatenate([xenium_idx, cosmx_sub_idx])
balanced_idx = np.sort(balanced_idx)

adata_bal = adata[balanced_idx].copy()
print(f"  Balanced subset: {adata_bal.shape[0]} cells")
print(f"  platform:\n{adata_bal.obs['platform'].value_counts().to_string()}")

# ─── Color palettes ───────────────────────────────────────────────────────────
# ct_major palette (9 categories)
CT_PALETTE = {
    "Epithelial":   "#E69F00",
    "Mesenchymal":  "#56B4E9",
    "B":            "#009E73",
    "T/NK":         "#F0E442",
    "Plasma":       "#0072B2",
    "Myeloid":      "#D55E00",
    "Endothelial":  "#CC79A7",
    "Neuron":       "#999999",
    "Unknown":      "#DDDDDD",
}

PLATFORM_PALETTE = {
    "cosmx":   "#2166AC",
    "xenium":  "#D6604D",
}

DATASET_PALETTE = {
    "cosmx_c1": "#4393C3",
    "xenium_c1": "#D6604D",
    "cosmx_c2":  "#92C5DE",
}

def make_palette(obs_series, palette_dict):
    """Return list of colors aligned to obs_series values."""
    return [palette_dict.get(v, "#AAAAAA") for v in obs_series]

# ─── UMAP computation + plotting ─────────────────────────────────────────────
EMBEDDINGS = [
    ("X_scVI_dataset",   "scVI (batch=dataset)"),
    ("X_scVI_platform",  "scVI (batch=platform)"),
    ("X_scANVI_dataset", "scANVI (batch=dataset)"),
    ("X_scANVI_platform","scANVI (batch=platform)"),
]

COLOR_SPECS = [
    ("platform", PLATFORM_PALETTE, "Platform"),
    ("dataset",  DATASET_PALETTE,  "Dataset"),
    ("ct_major", CT_PALETTE,       "Cell type (ct_major)"),
]

for emb_key, emb_label in EMBEDDINGS:
    if emb_key not in adata_bal.obsm:
        print(f"  SKIP {emb_key} — not in obsm")
        continue

    print(f"\n=== UMAP for {emb_key} ===")
    # Use embedding directly as X_pca substitute for UMAP
    adata_bal.obsm["X_emb"] = adata_bal.obsm[emb_key]

    print("  Computing neighbors ...")
    sc.pp.neighbors(adata_bal, use_rep="X_emb", n_neighbors=30, n_pcs=None)
    print("  Computing UMAP ...")
    sc.tl.umap(adata_bal, min_dist=0.3)

    umap_coords = adata_bal.obsm["X_umap"].copy()
    emb_tag = emb_key.replace("X_", "").lower()

    for col_key, palette, col_label in COLOR_SPECS:
        out_path = f"outputs/figures/umap_balanced_{emb_tag}_by_{col_key}.png"
        print(f"  Plotting {out_path} ...")

        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot each category separately for legend
        categories = sorted(adata_bal.obs[col_key].astype(str).unique())
        for cat in categories:
            mask = adata_bal.obs[col_key].astype(str) == cat
            color = palette.get(cat, "#AAAAAA")
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=color,
                s=1.5,
                alpha=0.5,
                label=f"{cat} (n={mask.sum():,})",
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{emb_label}\nColored by {col_label}", fontsize=11)
        ax.legend(
            markerscale=5,
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
            ncol=1 if len(categories) <= 5 else 2,
        )
        ax.set_aspect("equal", "datalim")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

print("\n=== All UMAP panels generated ===")
saved = sorted(os.listdir("outputs/figures"))
print(f"Figures in outputs/figures/: {len(saved)} files")
for f in saved:
    print(f"  {f}")
