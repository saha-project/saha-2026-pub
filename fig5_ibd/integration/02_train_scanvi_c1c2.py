#!/usr/bin/env python3
"""
02_train_scanvi_c1c2.py
Train scANVI from warm-started scVI models on C1+C2 AnnData.

Uses ct_major as labels_key, unlabeled_category='Unknown'.
Loads pre-trained scVI models and fine-tunes scANVI on top.

Input:
  outputs/adata_scvi_c1c2.h5ad        AnnData with scVI embeddings
  outputs/scvi_dataset_model/          pre-trained scVI (strategy 1)
  outputs/scvi_platform_model/         pre-trained scVI (strategy 2)

Output:
  outputs/scanvi_dataset_model/
  outputs/scanvi_platform_model/
  outputs/adata_scanvi_c1c2.h5ad      AnnData + X_scANVI_dataset + X_scANVI_platform
  outputs/scanvi_dataset_loss.png
  outputs/scanvi_platform_loss.png

Run: conda run -n spatial_integration python scripts/02_train_scanvi_c1c2.py
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
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Load AnnData with scVI embeddings ───────────────────────────────────────
print("Loading outputs/adata_scvi_c1c2.h5ad ...")
adata = sc.read_h5ad("outputs/adata_scvi_c1c2.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  ct_major unique: {sorted(adata.obs['ct_major'].unique())}")
n_unknown = (adata.obs['ct_major'] == 'Unknown').sum()
print(f"  Unknown (unlabeled) cells: {n_unknown}")

def plot_loss(model, out_path, title):
    hist = model.history
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ["elbo_train", "elbo_validation", "train_classification_loss",
                "validation_classification_loss"]:
        if key in hist and len(hist[key]) > 0:
            ax.plot(hist[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

# ─── Strategy 1: batch_key='dataset' ─────────────────────────────────────────
print("\n=== scANVI Strategy 1: batch_key='dataset' ===")

adata1 = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata1,
    layer="counts",
    batch_key="dataset",
)

print("  Loading pre-trained scVI model ...")
scvi_model1 = scvi.model.SCVI.load("outputs/scvi_dataset_model", adata=adata1)
print("  scVI model loaded.")

print("  Creating scANVI model from scVI ...")
scanvi1 = scvi.model.SCANVI.from_scvi_model(
    scvi_model1,
    labels_key="ct_major",
    unlabeled_category="Unknown",
)
print(f"  scANVI model:\n{scanvi1}")
print(f"  Labeled categories: {scanvi1._label_mapping}")

print("  Training scANVI ...")
scanvi1.train(
    max_epochs=20,
    n_samples_per_label=100,
)
print("  Training complete.")

emb1 = scanvi1.get_latent_representation()
adata.obsm["X_scANVI_dataset"] = emb1
print(f"  Embedding shape: {emb1.shape}")

# Predicted labels
pred1 = scanvi1.predict()
adata.obs["scanvi_pred_dataset"] = pred1
print(f"  Predicted label distribution:\n{adata.obs['scanvi_pred_dataset'].value_counts().to_string()}")

scanvi1.save("outputs/scanvi_dataset_model", overwrite=True)
print("  Model saved: outputs/scanvi_dataset_model")
plot_loss(scanvi1, "outputs/scanvi_dataset_loss.png", "scANVI (batch_key=dataset)")

del adata1, scvi_model1, scanvi1

# ─── Strategy 2: batch_key='platform' + covariate ────────────────────────────
print("\n=== scANVI Strategy 2: batch_key='platform' + categorical_covariate_keys=['dataset'] ===")

adata2 = adata.copy()
scvi.model.SCVI.setup_anndata(
    adata2,
    layer="counts",
    batch_key="platform",
    categorical_covariate_keys=["dataset"],
)

print("  Loading pre-trained scVI model ...")
scvi_model2 = scvi.model.SCVI.load("outputs/scvi_platform_model", adata=adata2)
print("  scVI model loaded.")

print("  Creating scANVI model from scVI ...")
scanvi2 = scvi.model.SCANVI.from_scvi_model(
    scvi_model2,
    labels_key="ct_major",
    unlabeled_category="Unknown",
)
print(f"  scANVI model:\n{scanvi2}")

print("  Training scANVI ...")
scanvi2.train(
    max_epochs=20,
    n_samples_per_label=100,
)
print("  Training complete.")

emb2 = scanvi2.get_latent_representation()
adata.obsm["X_scANVI_platform"] = emb2
print(f"  Embedding shape: {emb2.shape}")

pred2 = scanvi2.predict()
adata.obs["scanvi_pred_platform"] = pred2
print(f"  Predicted label distribution:\n{adata.obs['scanvi_pred_platform'].value_counts().to_string()}")

scanvi2.save("outputs/scanvi_platform_model", overwrite=True)
print("  Model saved: outputs/scanvi_platform_model")
plot_loss(scanvi2, "outputs/scanvi_platform_loss.png", "scANVI (batch_key=platform)")

del adata2, scvi_model2, scanvi2

# ─── Save AnnData with all embeddings ────────────────────────────────────────
out = "outputs/adata_scanvi_c1c2.h5ad"
print(f"\nSaving to {out} ...")
adata.write_h5ad(out)
print(f"Done. Shape: {adata.shape}")
print(f"obsm keys: {list(adata.obsm.keys())}")
print(f"obs columns: {list(adata.obs.columns)}")
