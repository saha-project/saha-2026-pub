"""
S7 — Per-niche cell-type composition
Panel a: Heatmap of cell-type fraction per (niche × organ)
Panel b: Marker gene expression heatmap per (niche × organ)
Niches defined by chemokine score tertiles (CCL19+CCL21+CXCL13, k=20 KNN smoothed).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8

ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
ORGANS = ["APE", "COL", "ILE"]

BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI"
FIG_DIR = BASE + "/figures"
H5AD_BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned"

CHEMOKINE_GENES = ["CCL19", "CCL21", "CXCL13"]
MARKER_GENES = ["CCL19", "CCL21", "CXCL13", "MS4A1", "CD19", "CD27", "CD38",
                "MKI67", "IRF4", "XBP1", "JCHAIN", "IGHA1", "IGHG1", "IGHG2", "IGHM", "IGHD"]
NICHE_LABELS = ["Low", "Mid", "High"]
NICHE_COLORS = {"Low": "#d1e5f0", "Mid": "#92c5de", "High": "#2166ac"}

K_SMOOTH = 20
Q_TERTILE_LOW = 0.60
Q_TERTILE_HIGH = 0.90


def compute_niches(adata_sub):
    """For one atomic unit: KNN-smooth chemokine score, assign tertile niche."""
    coords = adata_sub.obs[["x_centroid", "y_centroid"]].values
    if coords.shape[0] < K_SMOOTH + 1:
        return None

    # Get chemokine gene expression
    gene_names = list(adata_sub.var_names)
    avail = [g for g in CHEMOKINE_GENES if g in gene_names]
    if not avail:
        return None

    X = adata_sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    idx_list = [gene_names.index(g) for g in avail]
    chemo_raw = X[:, idx_list].mean(axis=1)

    # KNN smoothing
    nbrs = NearestNeighbors(n_neighbors=K_SMOOTH, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    chemo_smooth = np.array([chemo_raw[indices[i]].mean() for i in range(len(chemo_raw))])

    # Tertile binning
    lo = np.quantile(chemo_smooth, Q_TERTILE_LOW)
    hi = np.quantile(chemo_smooth, Q_TERTILE_HIGH)
    niche = np.where(chemo_smooth < lo, "Low",
                     np.where(chemo_smooth < hi, "Mid", "High"))
    return niche


# --- Collect data per organ (sample one representative unit per organ/patient combo) ---
comp_records = []   # (organ, niche, cell_type, count)
expr_records = []   # (organ, niche, gene, mean_expr)

for organ in ORGANS:
    h5ad_path = f"{H5AD_BASE}/{organ}_RNA_cleaned.h5ad"
    print(f"Loading {organ}...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    # Load into memory a sample: one unit per patient (3 patients → 3 units)
    samples = adata.obs["SAHA_Sample"].unique()
    batches = adata.obs["batch_id"].unique()

    units_done = 0
    for samp in sorted(samples)[:4]:   # up to 4 samples
        for bat in sorted(batches):
            mask = (adata.obs["SAHA_Sample"] == samp) & (adata.obs["batch_id"] == bat)
            n = mask.sum()
            if n < 200:
                continue
            sub = adata[mask].to_memory()
            niches = compute_niches(sub)
            if niches is None:
                continue

            sub.obs["niche"] = niches

            # Cell type composition
            ct_col = "Insitutype_Broad"
            for niche_label in NICHE_LABELS:
                nm = sub.obs[sub.obs["niche"] == niche_label]
                if len(nm) == 0:
                    continue
                ct_counts = nm[ct_col].value_counts()
                total = ct_counts.sum()
                for ct, cnt in ct_counts.items():
                    comp_records.append({
                        "organ": organ, "niche": niche_label,
                        "cell_type": ct, "fraction": cnt / total,
                        "unit": f"{samp}_{bat}"
                    })

                # Marker gene expression
                gene_names = list(sub.var_names)
                X = sub.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                niche_idx = sub.obs["niche"] == niche_label
                X_niche = X[niche_idx.values]
                for gene in MARKER_GENES:
                    if gene in gene_names:
                        gi = gene_names.index(gene)
                        expr_records.append({
                            "organ": organ, "niche": niche_label,
                            "gene": gene, "mean_expr": X_niche[:, gi].mean(),
                            "unit": f"{samp}_{bat}"
                        })

            units_done += 1
            if units_done >= 3:
                break
        if units_done >= 3:
            break

    adata.file.close() if hasattr(adata, "file") else None
    print(f"  {organ}: {units_done} units processed")

comp_df = pd.DataFrame(comp_records)
expr_df = pd.DataFrame(expr_records)

# Aggregate: mean fraction per (organ, niche, cell_type)
comp_agg = comp_df.groupby(["organ", "niche", "cell_type"])["fraction"].mean().reset_index()
expr_agg = expr_df.groupby(["organ", "niche", "gene"])["mean_expr"].mean().reset_index()

# --- Build pivot tables ---
comp_agg["row_label"] = comp_agg["organ"] + " | " + comp_agg["niche"]
comp_pivot = comp_agg.pivot_table(index="row_label", columns="cell_type", values="fraction", fill_value=0)
# Order rows
row_order = [f"{o} | {n}" for o in ORGANS for n in NICHE_LABELS]
row_order = [r for r in row_order if r in comp_pivot.index]
comp_pivot = comp_pivot.loc[row_order]

expr_agg["row_label"] = expr_agg["organ"] + " | " + expr_agg["niche"]
expr_pivot = expr_agg.pivot_table(index="row_label", columns="gene", values="mean_expr", fill_value=0)
expr_pivot = expr_pivot.loc[[r for r in row_order if r in expr_pivot.index]]
# Keep only requested genes in order
expr_pivot = expr_pivot.reindex(columns=[g for g in MARKER_GENES if g in expr_pivot.columns], fill_value=0)

# Z-score columns for display
from scipy.stats import zscore
comp_z = comp_pivot.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9), axis=0)
expr_z = expr_pivot.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9), axis=0)

# --- Figure ---
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45, width_ratios=[1.2, 1.6])

# Panel a: cell-type composition heatmap
ax_a = fig.add_subplot(gs[0])
im_a = ax_a.imshow(comp_z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax_a.set_xticks(range(comp_z.shape[1]))
ax_a.set_xticklabels(comp_z.columns, rotation=45, ha="right", fontsize=7)
ax_a.set_yticks(range(comp_z.shape[0]))
ax_a.set_yticklabels(comp_z.index, fontsize=7)
ax_a.set_title("Cell-type composition per niche × organ\n(Z-scored fraction)", fontsize=9, pad=8)
plt.colorbar(im_a, ax=ax_a, fraction=0.03, pad=0.04, label="Z-score")

# Color left tick labels by organ
for ytick, label in zip(ax_a.get_yticklabels(), comp_z.index):
    organ = label.split(" | ")[0]
    ytick.set_color(ORGAN_COLORS.get(organ, "black"))

# Panel b: marker gene expression heatmap
ax_b = fig.add_subplot(gs[1])
im_b = ax_b.imshow(expr_z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax_b.set_xticks(range(expr_z.shape[1]))
ax_b.set_xticklabels(expr_z.columns, rotation=45, ha="right", fontsize=7, style="italic")
ax_b.set_yticks(range(expr_z.shape[0]))
ax_b.set_yticklabels(expr_z.index, fontsize=7)
ax_b.set_title("Marker gene expression per niche × organ\n(Z-scored mean)", fontsize=9, pad=8)
plt.colorbar(im_b, ax=ax_b, fraction=0.03, pad=0.04, label="Z-score")

for ytick, label in zip(ax_b.get_yticklabels(), expr_z.index):
    organ = label.split(" | ")[0]
    ytick.set_color(ORGAN_COLORS.get(organ, "black"))

# Add niche separator lines
for ax in [ax_a, ax_b]:
    for i in range(1, len(ORGANS)):
        ax.axhline(i * len(NICHE_LABELS) - 0.5, color="white", lw=1.5)

# Panel labels
for ax, letter in zip([ax_a, ax_b], ["a", "b"]):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

fig.suptitle("Supplementary Figure S7 — Per-niche cell-type composition", fontsize=11, y=1.01, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(
        FIG_DIR + f"/fig_S7_niche_composition.{ext}",
        dpi=600, bbox_inches="tight", facecolor="white"
    )

print("S7 done.")
