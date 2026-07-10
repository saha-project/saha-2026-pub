#!/usr/bin/env python3
"""
02_evaluate_c1c2.py
Generate balanced UMAP panels for all 4 embeddings (scVI x2, scANVI x2).

For each embedding:
  - Compute UMAP
  - Plot 3 panels: colored by platform, dataset, ct_major

Balanced sampling: equal CosMx/Xenium representation for UMAP computation.

Input:  outputs/adata_scanvi_c1c2.h5ad
Output: outputs/figures/umap_*.png

Run: conda run -n spatial_integration python scripts/02_evaluate_c1c2.py
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
import matplotlib.gridspec as gridspec

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

os.makedirs("outputs/figures", exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading outputs/adata_scanvi_c1c2.h5ad ...")
adata = sc.read_h5ad("outputs/adata_scanvi_c1c2.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  dataset:\n{adata.obs['dataset'].value_counts().to_string()}")

EMBEDDINGS = [
    ("X_scVI_dataset",   "scVI (batch=dataset)"),
    ("X_scVI_platform",  "scVI (batch=platform)"),
    ("X_scANVI_dataset", "scANVI (batch=dataset)"),
    ("X_scANVI_platform","scANVI (batch=platform)"),
]

# ─── Balanced subsample for UMAP ──────────────────────────────────────────────
# Equal CosMx / Xenium: take min(n_cosmx, n_xenium, 50000) per platform
# Then add C2 proportionally
UMAP_MAX = 80_000

np.random.seed(42)
platform_counts = adata.obs["platform"].value_counts()
print(f"\nPlatform counts: {platform_counts.to_dict()}")

n_per_platform = min(platform_counts.min(), UMAP_MAX // 2)
print(f"Using {n_per_platform} cells per platform for UMAP")

idx_list = []
for plat in ["cosmx", "xenium"]:
    mask = adata.obs["platform"] == plat
    idx = np.where(mask)[0]
    chosen = np.random.choice(idx, size=min(n_per_platform, len(idx)), replace=False)
    idx_list.append(chosen)

umap_idx = np.concatenate(idx_list)
np.random.shuffle(umap_idx)

adata_umap = adata[umap_idx].copy()
print(f"UMAP subset shape: {adata_umap.shape}")
print(f"  platform: {adata_umap.obs['platform'].value_counts().to_dict()}")
print(f"  dataset: {adata_umap.obs['dataset'].value_counts().to_dict()}")

# Color palettes
CT_COLORS = {
    "Epithelial":  "#e41a1c",
    "Mesenchymal": "#377eb8",
    "B":           "#4daf4a",
    "T/NK":        "#984ea3",
    "Plasma":      "#ff7f00",
    "Myeloid":     "#a65628",
    "Endothelial": "#f781bf",
    "Neuron":      "#999999",
    "Unknown":     "#dddddd",
}
PLATFORM_COLORS = {"cosmx": "#1f77b4", "xenium": "#ff7f0e"}
DATASET_COLORS  = {
    "cosmx_c1":  "#1f77b4",
    "xenium_c1": "#ff7f0e",
    "cosmx_c2":  "#2ca02c",
}

def make_umap_panels(adata_sub, emb_key, title_prefix, out_prefix):
    """Compute UMAP from embedding and plot 3 color panels."""
    print(f"\n  Computing UMAP for {emb_key} ...")
    adata_sub.obsm["X_emb"] = adata_sub.obsm[emb_key]

    sc.pp.neighbors(adata_sub, use_rep="X_emb", n_neighbors=15)
    sc.tl.umap(adata_sub, min_dist=0.3)

    umap_xy = adata_sub.obsm["X_umap"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title_prefix, fontsize=14)

    # Panel 1: platform
    ax = axes[0]
    for plat, color in PLATFORM_COLORS.items():
        mask = adata_sub.obs["platform"] == plat
        ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                   c=color, s=0.5, alpha=0.4, label=plat, rasterized=True)
    ax.set_title("Platform")
    ax.legend(markerscale=5, loc="upper right", fontsize=8)
    ax.axis("off")

    # Panel 2: dataset
    ax = axes[1]
    for ds, color in DATASET_COLORS.items():
        mask = adata_sub.obs["dataset"] == ds
        if mask.sum() == 0:
            continue
        ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                   c=color, s=0.5, alpha=0.4, label=ds, rasterized=True)
    ax.set_title("Dataset")
    ax.legend(markerscale=5, loc="upper right", fontsize=8)
    ax.axis("off")

    # Panel 3: ct_major
    ax = axes[2]
    for ct, color in CT_COLORS.items():
        mask = adata_sub.obs["ct_major"] == ct
        if mask.sum() == 0:
            continue
        ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                   c=color, s=0.5, alpha=0.4, label=ct, rasterized=True)
    ax.set_title("ct_major")
    ax.legend(markerscale=5, loc="upper right", fontsize=7, ncol=2)
    ax.axis("off")

    plt.tight_layout()
    out_path = f"outputs/figures/{out_prefix}_umap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path

# ─── Generate UMAPs for all 4 embeddings ──────────────────────────────────────
for emb_key, emb_label in EMBEDDINGS:
    if emb_key not in adata_umap.obsm:
        print(f"  WARNING: {emb_key} not in obsm — skipping")
        continue

    out_prefix = emb_key.lower().replace("/", "_")
    make_umap_panels(
        adata_umap.copy(),
        emb_key=emb_key,
        title_prefix=emb_label,
        out_prefix=out_prefix,
    )

print("\n=== Evaluation complete ===")
print("UMAP figures saved to outputs/figures/")
figs = sorted(os.listdir("outputs/figures"))
for f in figs:
    print(f"  {f}")
