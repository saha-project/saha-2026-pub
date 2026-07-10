#!/usr/bin/env python3
"""
02_train_scvi_c1c2.py
Train scVI on the C1+C2 preprocessed AnnData using two batch strategies.

Strategy 1: batch_key='dataset'  (cosmx_c1, xenium_c1, cosmx_c2)
Strategy 2: batch_key='platform' + categorical_covariate_keys=['dataset']

Input:  outputs/adata_preprocessed.h5ad
Output:
  outputs/scvi_dataset_model/          scVI model (strategy 1)
  outputs/scvi_platform_model/         scVI model (strategy 2)
  outputs/adata_scvi_c1c2.h5ad        AnnData with X_scVI_dataset + X_scVI_platform
  outputs/scvi_dataset_loss.png
  outputs/scvi_platform_loss.png

Run: conda run -n spatial_integration python scripts/02_train_scvi_c1c2.py
Working directory: project root (IBD_integration/)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import scvi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_integration"
os.chdir(PROJ_ROOT)

os.makedirs("outputs", exist_ok=True)

# MPS acceleration on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading outputs/adata_preprocessed.h5ad ...")
adata = sc.read_h5ad("outputs/adata_preprocessed.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  layers: {list(adata.layers.keys())}")
print(f"  dataset counts:\n{adata.obs['dataset'].value_counts().to_string()}")
print(f"  ct_major counts:\n{adata.obs['ct_major'].value_counts().to_string()}")

def plot_loss(model, out_path, title):
    """Save ELBO training curve."""
    hist = model.history
    fig, ax = plt.subplots(figsize=(8, 4))
    if "elbo_train" in hist:
        ax.plot(hist["elbo_train"], label="train ELBO")
    if "elbo_validation" in hist:
        ax.plot(hist["elbo_validation"], label="validation ELBO")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

# ─── Strategy 1: batch_key='dataset' ─────────────────────────────────────────
print("\n=== Strategy 1: batch_key='dataset' ===")

adata1 = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata1,
    layer="counts",
    batch_key="dataset",
)
print("  Setup done.")
print(f"  Batch categories: {adata1.obs['dataset'].cat.categories.tolist()}")

model1 = scvi.model.SCVI(adata1)
print(f"  Model:\n{model1}")

print("  Training ...")
model1.train(
    max_epochs=400,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 1e-3},
)
print("  Training complete.")

# Save embedding
emb1 = model1.get_latent_representation()
adata.obsm["X_scVI_dataset"] = emb1
print(f"  Embedding shape: {emb1.shape}")

model1.save("outputs/scvi_dataset_model", overwrite=True)
print("  Model saved: outputs/scvi_dataset_model")
plot_loss(model1, "outputs/scvi_dataset_loss.png", "scVI (batch_key=dataset)")

del adata1

# ─── Strategy 2: batch_key='platform' + covariate 'dataset' ──────────────────
print("\n=== Strategy 2: batch_key='platform' + categorical_covariate_keys=['dataset'] ===")

adata2 = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata2,
    layer="counts",
    batch_key="platform",
    categorical_covariate_keys=["dataset"],
)
print("  Setup done.")
print(f"  Batch categories: {adata2.obs['platform'].cat.categories.tolist()}")

model2 = scvi.model.SCVI(adata2)
print(f"  Model:\n{model2}")

print("  Training ...")
model2.train(
    max_epochs=400,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 1e-3},
)
print("  Training complete.")

emb2 = model2.get_latent_representation()
adata.obsm["X_scVI_platform"] = emb2
print(f"  Embedding shape: {emb2.shape}")

model2.save("outputs/scvi_platform_model", overwrite=True)
print("  Model saved: outputs/scvi_platform_model")
plot_loss(model2, "outputs/scvi_platform_loss.png", "scVI (batch_key=platform)")

del adata2

# ─── Save AnnData with embeddings ────────────────────────────────────────────
out = "outputs/adata_scvi_c1c2.h5ad"
print(f"\nSaving to {out} ...")
adata.write_h5ad(out)
print(f"Done. Shape: {adata.shape}")
print(f"obsm keys: {list(adata.obsm.keys())}")
