"""02 — Biology-consistency analyses.

Reads outputs/adata_c1heavy_meta.h5ad and writes:
  outputs/biology/integration_metrics.csv     (per-cell-type kBET, iLISI, ASW(batch))
  outputs/biology/label_transfer.csv          (C1→C2 cell-type confusion)
  outputs/biology/label_transfer_summary.csv  (per ct accuracy)
  outputs/biology/umap_meta.csv               (cell-level UMAP coords + meta — small CSV for Fig A)

Spatial niche enrichment + LIANA L-R conservation are computed only if
spatial coordinates / liana-py are available; otherwise they are skipped
with a clear message and the figure script falls back to upstream cached
outputs from IBD_Exploration/outputs/phase4/{concordance,responder}/.

Run: python scripts/02_biology_consistency.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (silhouette_score, confusion_matrix,
                              classification_report, accuracy_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
config.ensure_output_dirs()
sc.settings.verbosity = 1

t0 = time.time()
print(f"[02] reading {os.path.basename(config.ADATA_META)} ...")
adata = ad.read_h5ad(config.ADATA_META)
print(f"     {adata.shape}; embedding key = {config.EMBEDDING_KEY}")
assert config.EMBEDDING_KEY in adata.obsm, \
    f"missing obsm['{config.EMBEDDING_KEY}']"
Z = np.asarray(adata.obsm[config.EMBEDDING_KEY])

# ── 0. UMAP for the figure (compute once, cache) ────────────────────────────
print("\n[02] (0) computing UMAP from scANVI embedding ...")
sc.pp.neighbors(adata, use_rep=config.EMBEDDING_KEY, n_neighbors=30,
                random_state=0)
sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=0)
umap = adata.obsm["X_umap"]

umap_df = pd.DataFrame({
    "UMAP1": umap[:, 0],
    "UMAP2": umap[:, 1],
    "ct_major": adata.obs["ct_major"].astype(str).values,
    "cohort_platform": adata.obs["cohort_platform"].astype(str).values,
    "platform": adata.obs["platform"].astype(str).values,
    "dataset": adata.obs["dataset"].astype(str).values,
    "patient_id": adata.obs["patient_id"].astype(str).values,
    "disease_state": adata.obs["disease_state"].astype(str).values,
})
umap_df.to_parquet(os.path.join(config.BIOLOGY_DIR, "umap_meta.parquet"))
print(f"     -> umap_meta.parquet ({len(umap_df):,} rows)")

# ── 1. Integration metrics per cell type ────────────────────────────────────
print("\n[02] (1) integration metrics: kBET-like + LISI + ASW(batch) ...")

batch_arr = adata.obs["dataset"].astype(str).values
ct_arr = adata.obs["ct_major"].astype(str).values

# Subsample to keep runtime manageable
RNG = np.random.default_rng(0)
N_PER_CT_MAX = 5000

metric_rows = []

# Build a global k-NN once (k=50) on the embedding
K_LISI = 50
print(f"     building global {K_LISI}-NN on {Z.shape[0]:,} × {Z.shape[1]} embedding ...")
nn = NearestNeighbors(n_neighbors=K_LISI + 1, n_jobs=-1).fit(Z)
nbr_idx = nn.kneighbors(Z, return_distance=False)[:, 1:]  # drop self
print("     k-NN ready.")

batch_codes = pd.Categorical(batch_arr).codes
ct_codes = pd.Categorical(ct_arr).codes
n_batches = batch_codes.max() + 1

for ct in config.CT_MAJOR_ORDER:
    ct_mask = ct_arr == ct
    n = int(ct_mask.sum())
    if n < 100:
        metric_rows.append(dict(ct_major=ct, n_cells=n,
                                kbet=np.nan, ilisi=np.nan,
                                clisi=np.nan, asw_batch=np.nan))
        continue
    idxs = np.where(ct_mask)[0]
    if len(idxs) > N_PER_CT_MAX:
        idxs = RNG.choice(idxs, size=N_PER_CT_MAX, replace=False)

    # iLISI = effective number of batches in each cell's neighborhood
    nbr_b = batch_codes[nbr_idx[idxs]]
    # Inverse Simpson per row
    pi_lisi = []
    for row in nbr_b:
        _, cnt = np.unique(row, return_counts=True)
        p = cnt / cnt.sum()
        pi_lisi.append(1.0 / np.sum(p ** 2))
    ilisi = float(np.mean(pi_lisi))
    # cLISI on cell types (within ct neighborhood — should be ~1 if pure)
    nbr_ct = ct_codes[nbr_idx[idxs]]
    pi_c = []
    for row in nbr_ct:
        _, cnt = np.unique(row, return_counts=True)
        p = cnt / cnt.sum()
        pi_c.append(1.0 / np.sum(p ** 2))
    clisi = float(np.mean(pi_c))

    # kBET-like: chi-square test of observed batch composition in neighborhood
    # vs. global ct-conditional batch composition. Acceptance = p > 0.05.
    global_p = (np.bincount(batch_codes[ct_mask], minlength=n_batches)
                / ct_mask.sum())
    accept = 0
    for row in nbr_b:
        obs_cnt = np.bincount(row, minlength=n_batches)
        exp_cnt = global_p * obs_cnt.sum()
        # avoid divide-by-zero
        keep = exp_cnt > 0.5
        if keep.sum() < 2:
            accept += 1
            continue
        chi2 = ((obs_cnt[keep] - exp_cnt[keep]) ** 2 / exp_cnt[keep]).sum()
        # df = (n_categories - 1)
        from scipy.stats import chi2 as _chi2
        pval = _chi2.sf(chi2, df=keep.sum() - 1)
        if pval > 0.05:
            accept += 1
    kbet = float(accept / len(idxs))

    # ASW(batch) on this ct's cells (lower = better mixing)
    if n > 5000:
        sub_idx = RNG.choice(idxs, size=5000, replace=False)
    else:
        sub_idx = idxs
    if len(np.unique(batch_codes[sub_idx])) >= 2:
        asw = float(silhouette_score(Z[sub_idx], batch_codes[sub_idx],
                                     sample_size=min(2000, len(sub_idx)),
                                     random_state=0))
    else:
        asw = np.nan

    metric_rows.append(dict(ct_major=ct, n_cells=n,
                            kbet=kbet, ilisi=ilisi, clisi=clisi,
                            asw_batch=asw))

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(os.path.join(config.BIOLOGY_DIR, "integration_metrics.csv"),
                  index=False)
print("     -> integration_metrics.csv:")
print(metrics_df.round(3).to_string(index=False))

# Nyquist check
ok = metrics_df.dropna(subset=["kbet", "ilisi"])
n_pass_kbet = (ok["kbet"] >= config.KBET_MIN).sum()
n_pass_ilisi = (ok["ilisi"] >= config.ILISI_MIN * (n_batches / 2)).sum()
print(f"     Nyquist: kbet≥{config.KBET_MIN} → {n_pass_kbet}/{len(ok)} cell types; "
      f"ilisi (effective batches >= {config.ILISI_MIN}× n_batches/2) "
      f"→ {n_pass_ilisi}/{len(ok)}")

# ── 2. Cross-cohort label-transfer accuracy (C1 → C2) ───────────────────────
print("\n[02] (2) cross-cohort label transfer (C1 → C2) ...")
c1_mask = adata.obs["dataset"].isin(["cosmx_c1", "xenium_c1"]).values
c2_mask = adata.obs["dataset"].eq("cosmx_c2").values

X_train = Z[c1_mask]
y_train = ct_arr[c1_mask]
X_test = Z[c2_mask]
y_test = ct_arr[c2_mask]

# 5-fold-style train/val split on C1 to gauge noise floor
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.2,
                                  stratify=y_train, random_state=0)
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xt, yt)
val_acc = accuracy_score(yv, clf.predict(Xv))
print(f"     C1 internal val accuracy (logistic on scANVI): {val_acc:.3f}")

# Refit on full C1 → predict on C2 (which keeps its existing labels)
clf_full = LogisticRegression(max_iter=2000, n_jobs=-1)
clf_full.fit(X_train, y_train)
y_pred = clf_full.predict(X_test)

ct_test_acc = accuracy_score(y_test, y_pred)
print(f"     C1→C2 cell-type label-transfer accuracy: {ct_test_acc:.3f}")

cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=config.CT_MAJOR_ORDER),
    index=pd.Index(config.CT_MAJOR_ORDER, name="C2_true"),
    columns=pd.Index(config.CT_MAJOR_ORDER, name="C2_predicted_from_C1"),
)
# Row-normalize → per-class recall
cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
cm_norm.to_csv(os.path.join(config.BIOLOGY_DIR, "label_transfer.csv"))
per_class = (cm_norm.values.diagonal())
pd.DataFrame({
    "ct_major": config.CT_MAJOR_ORDER,
    "n_c2_cells": cm.sum(axis=1).values,
    "recall": per_class,
}).to_csv(os.path.join(config.BIOLOGY_DIR, "label_transfer_summary.csv"),
         index=False)
print("     per-class C2 recall:")
print(pd.Series(per_class, index=config.CT_MAJOR_ORDER).round(3).to_string())
print(f"\n     overall accuracy: {ct_test_acc:.3f}; "
      f"internal C1 noise floor: {val_acc:.3f}")

print(f"\n[02] DONE in {time.time()-t0:.1f}s")
