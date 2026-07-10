"""
S10 — Methods supplementary
Panel a: Atomic-unit definition diagram (schematic)
Panel b: Niche-bin definition workflow for one representative unit
Panel c: Composite score formulation table
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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

BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI"
FIG_DIR = BASE + "/figures"
H5AD_BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned"

CHEMOKINE_GENES = ["CCL19", "CCL21", "CXCL13"]
K_SMOOTH = 20

# =====================================================================
# Panel b data: load one representative unit APE_A9_4 × SJ01
# =====================================================================
print("Loading APE h5ad for panel b...")
adata_ape = ad.read_h5ad(H5AD_BASE + "/APE_RNA_cleaned.h5ad", backed="r")
mask = (adata_ape.obs["SAHA_Sample"] == "APE_A9_4") & (adata_ape.obs["batch_id"] == "SJ01")
n_cells = mask.sum()
print(f"  APE_A9_4 x SJ01: {n_cells} cells")
sub = adata_ape[mask].to_memory()
adata_ape.file.close() if hasattr(adata_ape, "file") else None

coords = sub.obs[["x_centroid", "y_centroid"]].values
gene_names = list(sub.var_names)
avail = [g for g in CHEMOKINE_GENES if g in gene_names]
X = sub.X
if hasattr(X, "toarray"):
    X = X.toarray()
idx_list = [gene_names.index(g) for g in avail]
chemo_raw = X[:, idx_list].mean(axis=1)

if len(coords) >= K_SMOOTH + 1:
    nbrs = NearestNeighbors(n_neighbors=K_SMOOTH, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    chemo_smooth = np.array([chemo_raw[indices[i]].mean() for i in range(len(chemo_raw))])
else:
    chemo_smooth = chemo_raw.copy()

lo = np.quantile(chemo_smooth, 0.60)
hi = np.quantile(chemo_smooth, 0.90)
niche_labels = np.where(chemo_smooth < lo, 0,
                        np.where(chemo_smooth < hi, 1, 2))

# Downsample for scatter (max 8000 cells)
rng = np.random.default_rng(42)
n_plot = min(8000, len(coords))
sel = rng.choice(len(coords), n_plot, replace=False)
cx, cy = coords[sel, 0], coords[sel, 1]
raw_sel = chemo_raw[sel]
smooth_sel = chemo_smooth[sel]
niche_sel = niche_labels[sel]

# =====================================================================
# Figure layout
# =====================================================================
fig = plt.figure(figsize=(18, 14))
gs_outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                              height_ratios=[1.1, 1.0, 1.1])

# ---------- Panel a: atomic unit diagram ----------
ax_a = fig.add_subplot(gs_outer[0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 4.5)
ax_a.axis("off")
ax_a.text(-0.05, 1.04, "a", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top")
ax_a.set_title("Atomic unit definition: (SAHA_Sample × batch_id)", fontsize=10, fontweight="bold", pad=8)

organ_info = [("APE", 0.5), ("COL", 3.8), ("ILE", 7.1)]
patient_labels = ["EC01", "SJ01", "TAP03", "TAP04"]
patient_colors = ["#a8ddb5", "#43a2ca", "#e08214", "#d7191c"]

for (organ, x_start), oc in zip(organ_info, ORGAN_COLORS.values()):
    # Organ label box
    rect = mpatches.FancyBboxPatch((x_start - 0.05, 0.1), 2.6, 4.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=oc, alpha=0.08,
                                    edgecolor=oc, linewidth=1.5)
    ax_a.add_patch(rect)
    ax_a.text(x_start + 1.25, 4.25, organ, ha="center", va="center",
              fontsize=11, fontweight="bold", color=oc)

    # 4 patient squares (2×2 grid)
    for pi, (pat, pc) in enumerate(zip(patient_labels, patient_colors)):
        col_idx = pi % 2
        row_idx = pi // 2
        bx = x_start + col_idx * 1.2 + 0.1
        by = 2.1 - row_idx * 1.7
        sq = mpatches.FancyBboxPatch((bx, by), 1.0, 1.4,
                                      boxstyle="round,pad=0.05",
                                      facecolor=pc, alpha=0.55,
                                      edgecolor="white", linewidth=1.0)
        ax_a.add_patch(sq)
        ax_a.text(bx + 0.5, by + 0.7, pat, ha="center", va="center",
                  fontsize=7, fontweight="bold", color="white",
                  path_effects=[pe.withStroke(linewidth=1.5, foreground="gray")])
        ax_a.text(bx + 0.5, by + 0.15, "1 slide", ha="center", va="bottom",
                  fontsize=6, color="white", alpha=0.9)

# Arrow/label explaining atomic unit
ax_a.annotate("Atomic unit =\n(sample × donor)",
              xy=(5.0, 1.5), xytext=(5.0, 0.0),
              fontsize=9, ha="center",
              arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
              color="#444")

# ---------- Panel b: niche bin workflow ----------
gs_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1],
                                         wspace=0.35)
NICHE_CMAP = matplotlib.colors.ListedColormap(["#d1e5f0", "#92c5de", "#2166ac"])
STEP_TITLES = [
    "1. Raw chemokine score\n(mean CCL19+CCL21+CXCL13)",
    f"2. KNN-smoothed\n(k={K_SMOOTH} nearest neighbors)",
    "3. Tertile-binned niches\n(Low / Mid / High)"
]
STEP_DATA = [raw_sel, smooth_sel, niche_sel]
STEP_CMAPS = ["viridis", "viridis", NICHE_CMAP]
STEP_VMINS = [None, None, -0.5]
STEP_VMAXS = [None, None, 2.5]

for si in range(3):
    ax_b = fig.add_subplot(gs_b[si])
    vmin = np.percentile(STEP_DATA[si], 2) if STEP_VMINS[si] is None else STEP_VMINS[si]
    vmax = np.percentile(STEP_DATA[si], 98) if STEP_VMAXS[si] is None else STEP_VMAXS[si]
    sc = ax_b.scatter(cx, cy, c=STEP_DATA[si], s=0.8, cmap=STEP_CMAPS[si],
                      vmin=vmin, vmax=vmax, rasterized=True, linewidths=0)
    ax_b.set_aspect("equal")
    ax_b.set_title(STEP_TITLES[si], fontsize=8, pad=4)
    ax_b.axis("off")
    cb = plt.colorbar(sc, ax=ax_b, fraction=0.05, pad=0.02, shrink=0.7)
    cb.ax.tick_params(labelsize=6)
    if si == 2:
        cb.set_ticks([0, 1, 2])
        cb.set_ticklabels(["Low", "Mid", "High"])

# Panel b label via fig.text positioned at top-left of the subplot row
pos_b = gs_outer[1].get_position(fig)
fig.text(0.01, pos_b.y1, "b", fontsize=14, fontweight="bold", va="bottom")

# Add unit label below panel b
fig.text(0.5, pos_b.y0 - 0.01,
         "Representative unit: APE_A9_4 × SJ01", ha="center",
         fontsize=8, color="#555")

# ---------- Panel c: composite score table ----------
ax_c = fig.add_subplot(gs_outer[2])
ax_c.axis("off")
ax_c.text(-0.05, 1.04, "c", transform=ax_c.transAxes,
          fontsize=14, fontweight="bold", va="top")
ax_c.set_title("Composite score formulations", fontsize=10, fontweight="bold", pad=8)

TABLE_DATA = [
    ["Score", "Positive markers", "Negative markers", "Biological interpretation"],
    ["Egress", "JCHAIN, PIGR, IGHA1", "IGHG1, IGHG2", "Secretory IgA export toward mucosa"],
    ["Plasma-fate", "IRF4, XBP1, JCHAIN, CD38", "PAX5, MS4A1", "B→plasma cell differentiation"],
    ["B-retention", "MS4A1, CD19, PAX5", "IRF4, XBP1", "Mature follicular B-cell identity"],
    ["Proliferation", "MKI67, TOP2A, PCNA", "—", "Active cell-cycle entry"],
    ["Dark-zone", "AICDA, CXCR4, MKI67", "BCL6 (lo)", "GC dark-zone somatic hypermutation"],
    ["Light-zone", "BCL6, CD83, CXCR5", "CXCR4 (lo)", "GC light-zone antigen selection"],
    ["Systemic-homing", "IGHG1, IGHG2, S1PR1", "PIGR, IGHA1", "Systemic/lymph-node homing program"],
    ["Gut-homing", "IGHA1, IGHA2, ITGA4", "IGHG1, S1PR1", "Gut-specific IgA secretory program"],
]

col_widths = [0.14, 0.28, 0.22, 0.36]
col_x = [0.01, 0.15, 0.44, 0.66]
row_height = 0.1
header_bg = "#2c3e50"
row_bgs = ["#f7f7f7", "#ffffff"]

for ri, row in enumerate(TABLE_DATA):
    y_top = 1.0 - ri * row_height
    bg = header_bg if ri == 0 else row_bgs[ri % 2]
    bg_rect = mpatches.FancyBboxPatch((0.0, y_top - row_height), 1.0, row_height,
                                       boxstyle="square,pad=0",
                                       facecolor=bg, alpha=1.0,
                                       transform=ax_c.transAxes,
                                       clip_on=False, linewidth=0)
    ax_c.add_patch(bg_rect)
    for ci, (text, cx_pos) in enumerate(zip(row, col_x)):
        color = "white" if ri == 0 else "black"
        fw = "bold" if ri == 0 or ci == 0 else "normal"
        fs = 8 if ri == 0 else 7.5
        style = "italic" if ri > 0 and ci in [1, 2] else "normal"
        ax_c.text(cx_pos + 0.01, y_top - row_height / 2, text,
                  transform=ax_c.transAxes,
                  va="center", ha="left", fontsize=fs,
                  fontweight=fw, fontstyle=style, color=color)

# Thin separator lines (use ax_c.transAxes via plot, not axhline which forbids transform)
for ri in range(1, len(TABLE_DATA) + 1):
    y = 1.0 - ri * row_height
    ax_c.plot([0.0, 1.0], [y, y], color="#cccccc", lw=0.5,
              transform=ax_c.transAxes, clip_on=False)

ax_c.set_xlim(0, 1)
ax_c.set_ylim(1.0 - len(TABLE_DATA) * row_height - 0.02, 1.02)

fig.suptitle("Supplementary Figure S10 — Methods supplementary",
             fontsize=11, y=1.005, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(
        FIG_DIR + f"/fig_S10_methods.{ext}",
        dpi=600, bbox_inches="tight", facecolor="white"
    )

print("S10 done.")
