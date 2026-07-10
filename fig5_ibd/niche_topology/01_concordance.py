"""01 — Concordance analyses.

Reads outputs/adata_c1heavy_meta.h5ad and writes per-analysis CSVs to
outputs/concordance/:

  paired_xenium_cosmx_corr.csv          (patient × ct_major × {pearson,spearman,n_genes})
  paired_xenium_cosmx_per_gene.csv      (patient × ct_major × gene × {x,y})  showcase only
  cohort_pseudobulk.csv                 (cohort_platform × ct_major × gene → log-cpm)
  cohort_pseudobulk_corrmat.csv         (27 × 27 correlation matrix)
  composition_fractions.csv             (sample × ct_major → fraction)
  composition_pair_corr.csv             (per ct_major paired CosMx↔Xenium fraction r)
  signature_scores.csv                  (cell-level scores, summarized per sample × ct × signature)
  signature_summary.csv                 (mean signature per cohort_platform × ct × signature)

Excludes CXCL8 + MUC2 per IBD_Exploration invariant.

Run: python scripts/01_concordance.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
sc.settings.verbosity = 1

t0 = time.time()
print(f"[01] reading {os.path.basename(config.ADATA_META)} ...")
adata = ad.read_h5ad(config.ADATA_META)
print(f"     {adata.shape}")

# ── Restrict to genes (drop excluded) ───────────────────────────────────────
keep_var = ~adata.var_names.isin(config.EXCLUDED_GENES)
adata = adata[:, keep_var].copy()
print(f"     after excluding {config.EXCLUDED_GENES}: {adata.shape}")

# Use raw counts where possible
if "counts" in adata.layers:
    print("     using layers['counts'] for pseudobulk")
    COUNT_LAYER = "counts"
else:
    print("     WARNING: no 'counts' layer — falling back to .X (assumed log1p)")
    COUNT_LAYER = None


def _row_pseudobulk(X: sp.csr_matrix | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum counts over cells in `mask`. Returns 1D ndarray of length n_genes."""
    if mask.sum() == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    sub = X[mask]
    if sp.issparse(sub):
        s = np.asarray(sub.sum(axis=0)).ravel()
    else:
        s = sub.sum(axis=0)
    return s.astype(np.float64)


def _logcpm(v: np.ndarray) -> np.ndarray:
    """log1p(CPM)."""
    tot = v.sum()
    if tot <= 0:
        return np.zeros_like(v)
    return np.log1p(v / tot * 1e6)


def _mean_per_cell_log(X, mask: np.ndarray, target_sum: float = 200.0) -> np.ndarray:
    """Per-cell normalize to `target_sum`, log1p, then mean across cells in mask.

    This is the cross-platform-fair pseudobulk: each cell contributes equally
    regardless of platform-specific transcript yield.
    """
    if mask.sum() == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    sub = X[mask]
    if sp.issparse(sub):
        sub = sub.tocsr().astype(np.float64)
        rs = np.asarray(sub.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        # Row-scale: divide each row by its sum, multiply by target_sum
        scale = target_sum / rs
        # Build a diagonal multiplier matrix
        D = sp.diags(scale)
        sub = D @ sub
        sub.data = np.log1p(sub.data)
        return np.asarray(sub.mean(axis=0)).ravel()
    rs = sub.sum(axis=1)
    rs[rs == 0] = 1.0
    sub = sub * (target_sum / rs[:, None])
    return np.log1p(sub).mean(axis=0)


# ── 1. Paired patient Xenium↔CosMx (C1) ─────────────────────────────────────
print("\n[01] (1) paired Xenium↔CosMx pseudobulk per (patient, ct_major) ...")
c1_mask = adata.obs["dataset"].isin(["cosmx_c1", "xenium_c1"]).values
adata_c1 = adata[c1_mask].copy()
X_c1 = adata_c1.layers[COUNT_LAYER] if COUNT_LAYER else adata_c1.X
genes = adata_c1.var_names.to_numpy()
patients = sorted(adata_c1.obs["patient_id"].dropna().unique().tolist())
ct_list = list(config.CT_MAJOR_ORDER)

paired_rows = []
showcase_patient = "C1_P4"  # I0387 inflamed UC — most balanced cell counts
showcase_per_gene = []

for p in patients:
    for ct in ct_list:
        m_cosmx = ((adata_c1.obs["patient_id"] == p)
                   & (adata_c1.obs["dataset"] == "cosmx_c1")
                   & (adata_c1.obs["ct_major"].astype(str) == ct)).values
        m_xen = ((adata_c1.obs["patient_id"] == p)
                 & (adata_c1.obs["dataset"] == "xenium_c1")
                 & (adata_c1.obs["ct_major"].astype(str) == ct)).values
        n_cosmx, n_xen = int(m_cosmx.sum()), int(m_xen.sum())
        if n_cosmx < 20 or n_xen < 20:
            paired_rows.append(dict(patient_id=p, ct_major=ct,
                                    n_cosmx=n_cosmx, n_xen=n_xen,
                                    pearson_r=np.nan, spearman_r=np.nan,
                                    n_genes=0))
            continue
        pb_cosmx = _logcpm(_row_pseudobulk(X_c1, m_cosmx))
        pb_xen = _logcpm(_row_pseudobulk(X_c1, m_xen))
        keep = (pb_cosmx > 0) | (pb_xen > 0)
        if keep.sum() < 50:
            paired_rows.append(dict(patient_id=p, ct_major=ct,
                                    n_cosmx=n_cosmx, n_xen=n_xen,
                                    pearson_r=np.nan, spearman_r=np.nan,
                                    n_genes=int(keep.sum())))
            continue
        r, _ = pearsonr(pb_cosmx[keep], pb_xen[keep])
        rho, _ = spearmanr(pb_cosmx[keep], pb_xen[keep])
        paired_rows.append(dict(patient_id=p, ct_major=ct,
                                n_cosmx=n_cosmx, n_xen=n_xen,
                                pearson_r=r, spearman_r=rho,
                                n_genes=int(keep.sum())))
        if p == showcase_patient:
            for i in np.where(keep)[0]:
                showcase_per_gene.append(dict(
                    patient_id=p, ct_major=ct, gene=genes[i],
                    cosmx_logcpm=float(pb_cosmx[i]),
                    xenium_logcpm=float(pb_xen[i]),
                ))

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(os.path.join(config.CONCORDANCE_DIR,
                              "paired_xenium_cosmx_corr.csv"), index=False)
print(f"     -> paired_xenium_cosmx_corr.csv "
      f"({paired_df['pearson_r'].notna().sum()}/{len(paired_df)} non-null pairs)")

pd.DataFrame(showcase_per_gene).to_csv(
    os.path.join(config.CONCORDANCE_DIR, "paired_xenium_cosmx_per_gene.csv"),
    index=False,
)
print(f"     -> paired_xenium_cosmx_per_gene.csv (showcase patient {showcase_patient})")

# Verification (Nyquist)
ok_per_patient = paired_df.groupby("patient_id").apply(
    lambda g: (g["pearson_r"] >= config.PAIRED_PEARSON_MIN).sum()
              / max(g["pearson_r"].notna().sum(), 1)
)
n_patients_pass = int((ok_per_patient >= config.PAIRED_FRAC_PASS_PER_PATIENT).sum())
print(f"     Nyquist check: {n_patients_pass}/{len(ok_per_patient)} patients "
      f"pass (need ≥{config.PAIRED_PATIENTS_PASS}); per-patient pass-fraction:")
print(ok_per_patient.round(2).to_string())

# ── 2. Cohort-level pseudobulk per (cohort_platform, ct_major) ─────────────
print("\n[01] (2) cohort-level pseudobulk: cohort_platform × ct_major ...")
X_full = adata.layers[COUNT_LAYER] if COUNT_LAYER else adata.X
cp_arr = adata.obs["cohort_platform"].astype(str).values
ct_arr = adata.obs["ct_major"].astype(str).values

vectors = {}
n_cells = {}
for cp in config.COHORT_PLATFORMS:
    for ct in ct_list:
        m = (cp_arr == cp) & (ct_arr == ct)
        n_cells[(cp, ct)] = int(m.sum())
        if m.sum() < 20:
            vectors[(cp, ct)] = np.full(adata.shape[1], np.nan)
            continue
        vectors[(cp, ct)] = _logcpm(_row_pseudobulk(X_full, m))

mat = pd.DataFrame(
    {f"{cp}|{ct}": v for (cp, ct), v in vectors.items()},
    index=genes,
)
mat.index.name = "gene"
mat.to_csv(os.path.join(config.CONCORDANCE_DIR, "cohort_pseudobulk.csv"))
print(f"     -> cohort_pseudobulk.csv ({mat.shape[1]} columns)")

# Correlation matrix (drop columns that are all-NaN)
valid_cols = mat.columns[~mat.isna().all(axis=0)]
corr = mat[valid_cols].corr(method="pearson")
corr.to_csv(os.path.join(config.CONCORDANCE_DIR,
                         "cohort_pseudobulk_corrmat.csv"))
print(f"     -> cohort_pseudobulk_corrmat.csv ({corr.shape})")

# ── 3. Cell-type composition fractions ──────────────────────────────────────
print("\n[01] (3) cell-type composition fractions ...")
comp = (adata.obs.groupby(["sample", "cohort_platform", "ct_major"],
                          observed=True).size()
        .rename("n").reset_index())
totals = comp.groupby(["sample", "cohort_platform"], observed=True)["n"].transform("sum")
comp["fraction"] = comp["n"] / totals
comp.to_csv(os.path.join(config.CONCORDANCE_DIR, "composition_fractions.csv"),
            index=False)
print(f"     -> composition_fractions.csv ({len(comp)} rows)")

# Paired CosMx↔Xenium composition correlation per ct_major (C1)
c1_comp = comp[comp["cohort_platform"].astype(str).str.startswith("C1-")].copy()
# patient_id from sample: cosmx_c1_<n> / xenium_c1_<n> share suffix
c1_comp["pat"] = c1_comp["sample"].str.extract(r"c1_(\d+)$")[0]
c1_comp["plat"] = np.where(c1_comp["sample"].str.contains("cosmx"),
                            "cosmx", "xenium")
piv = c1_comp.pivot_table(index=["ct_major", "pat"], columns="plat",
                          values="fraction", aggfunc="sum").reset_index()
piv = piv.dropna()
comp_corr = (piv.groupby("ct_major", observed=True)
                .apply(lambda g: pd.Series({
                    "n_pairs": len(g),
                    "pearson_r": (pearsonr(g["cosmx"], g["xenium"])[0]
                                  if len(g) >= 3 else np.nan),
                }))
                .reset_index())
comp_corr.to_csv(os.path.join(config.CONCORDANCE_DIR,
                              "composition_pair_corr.csv"), index=False)
print("     -> composition_pair_corr.csv:")
print(comp_corr.to_string(index=False))

# ── 4. IBD marker signature scoring ─────────────────────────────────────────
print("\n[01] (4) IBD marker signature scoring (sc.tl.score_genes) ...")
# Use log1p-normalized .X (already log1p in the source)
signature_summary_rows = []
for sig_name, gene_list in config.IBD_MARKER_PANEL.items():
    present = [g for g in gene_list if g in adata.var_names
               and g not in config.EXCLUDED_GENES]
    if len(present) < 2:
        print(f"     skip '{sig_name}' (only {len(present)} genes present)")
        continue
    sc.tl.score_genes(adata, gene_list=present, score_name=f"sig_{sig_name}",
                      random_state=0)
    grp = (adata.obs.groupby(["cohort_platform", "ct_major"], observed=True)
                 [f"sig_{sig_name}"].mean()
                 .reset_index()
                 .rename(columns={f"sig_{sig_name}": "mean_score"}))
    grp["signature"] = sig_name
    grp["n_genes"] = len(present)
    signature_summary_rows.append(grp)
sig_summary = pd.concat(signature_summary_rows, ignore_index=True)
sig_summary.to_csv(os.path.join(config.CONCORDANCE_DIR,
                                "signature_summary.csv"), index=False)
print(f"     -> signature_summary.csv ({len(sig_summary)} rows)")

# Cross-cohort correlation of signature means per cell type
print("\n[01] (4b) cross-cohort signature correlation per ct_major ...")
sig_pivot = sig_summary.pivot_table(
    index=["ct_major", "signature"],
    columns="cohort_platform", values="mean_score",
).reset_index()
sig_pivot.to_csv(os.path.join(config.CONCORDANCE_DIR,
                              "signature_pivot.csv"), index=False)

# Pairwise cohort_platform correlation across (ct, signature)
cp_cols = [c for c in sig_pivot.columns if c in config.COHORT_PLATFORMS]
sig_cohort_corr = sig_pivot[cp_cols].corr()
sig_cohort_corr.to_csv(os.path.join(config.CONCORDANCE_DIR,
                                    "signature_cohort_corr.csv"))
print("     cross-cohort signature corr:")
print(sig_cohort_corr.round(3).to_string())

# ── 5. HVG + platform-centered cohort pseudobulk (biology > technology) ────
print("\n[01] (5) HVG + platform-centered cohort pseudobulk ...")
# Pick HVGs using scanpy on the log1p .X
adata_hvg = adata.copy()
sc.pp.highly_variable_genes(adata_hvg, n_top_genes=400, flavor="seurat_v3",
                            layer="counts", batch_key="cohort_platform")
hvg_mask = adata_hvg.var["highly_variable"].values
hvg_genes = adata_hvg.var_names[hvg_mask].tolist()
print(f"     selected {len(hvg_genes)} HVGs")

mat_hvg = mat.loc[hvg_genes].copy()  # reuse cohort pseudobulk log-CPM

# Per-cohort_platform mean-center each gene (remove platform bias)
def _cp_of(col: str) -> str:
    return col.split("|")[0]

centered = mat_hvg.copy()
for cp in config.COHORT_PLATFORMS:
    cols = [c for c in centered.columns if _cp_of(c) == cp]
    if not cols:
        continue
    centered[cols] = centered[cols].sub(centered[cols].mean(axis=1), axis=0)

mat_hvg.to_csv(os.path.join(config.CONCORDANCE_DIR,
                            "cohort_pseudobulk_hvg.csv"))
centered.to_csv(os.path.join(config.CONCORDANCE_DIR,
                             "cohort_pseudobulk_hvg_centered.csv"))

valid_cols = centered.columns[~centered.isna().all(axis=0)]
corr_hvg = centered[valid_cols].corr(method="pearson")
corr_hvg.to_csv(os.path.join(config.CONCORDANCE_DIR,
                             "cohort_pseudobulk_hvg_centered_corrmat.csv"))

# Summarize biology > technology ratio on HVG-centered
rows = []
for i in corr_hvg.columns:
    cp_i, ct_i = i.split("|")
    for j in corr_hvg.columns:
        if i >= j: continue
        cp_j, ct_j = j.split("|")
        rows.append(dict(same_ct=(ct_i == ct_j),
                         same_cp=(cp_i == cp_j),
                         r=corr_hvg.loc[i, j]))
summary = pd.DataFrame(rows)
print("     Mean HVG-centered Pearson r:")
print(summary.groupby(["same_ct", "same_cp"], observed=True)["r"]
       .agg(["mean", "median", "count"]).round(3).to_string())
summary.to_csv(os.path.join(config.CONCORDANCE_DIR,
                            "cohort_pseudobulk_hvg_summary.csv"), index=False)

print(f"\n[01] DONE in {time.time()-t0:.1f}s")
