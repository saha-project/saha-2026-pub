"""03 — Assemble Nature-style main figure.

Reads cached CSVs from outputs/concordance/ and outputs/biology/ (no h5ad
loading required after Wave 1+2 ran).

Layout (single page, ~180 × 200 mm):

  A | B
  C | D
  E | F

  A: Integrated UMAP faceted by cohort_platform, coloured by ct_major
  B: Paired Xenium↔CosMx per-gene scatter (showcase patient C1_P3)
     + bar of per-(patient, ct) Pearson r
  C: HVG + platform-centered cohort pseudobulk correlation heatmap (27×27)
     — the headline "biology > technology" panel
  D: Cell-type composition stacked bars per sample, grouped by cohort_platform
  E: IBD signature dotplot — cohort_platform × ct_major × signature
  F: C1→C2 cell-type label-transfer confusion matrix
     (recovers cell types across cohorts)

Extended Data: integration metrics (kBET / iLISI / cLISI), HVG composition,
                signature heatmap with per-cohort_platform z-score.

Outputs: outputs/figures/main/fig1_consistency.{pdf,svg,png}
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)


def panel_label(ax, label, x=-0.20, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def save_fig(fig, name, out_dir=None):
    out_dir = out_dir or config.FIG_MAIN_DIR
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    svg = os.path.join(out_dir, f"{name}.svg")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, format="pdf", dpi=1200, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"     -> {os.path.basename(pdf)}, .svg, .png")


# ── Load cached data ─────────────────────────────────────────────────────────
print("[03] loading cached panel data ...")
umap = pd.read_parquet(os.path.join(config.BIOLOGY_DIR, "umap_meta.parquet"))
paired = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                   "paired_xenium_cosmx_corr.csv"))
showcase = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                     "paired_xenium_cosmx_per_gene.csv"))
corr_hvg = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                     "cohort_pseudobulk_hvg_centered_corrmat.csv"),
                       index_col=0)
comp = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                 "composition_fractions.csv"))
sig_summary = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                        "signature_summary.csv"))
metrics = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                    "integration_metrics.csv"))
lt = pd.read_csv(os.path.join(config.BIOLOGY_DIR, "label_transfer.csv"),
                 index_col=0)
lt_summary = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                       "label_transfer_summary.csv"))
print(f"     loaded: umap {umap.shape}, paired {paired.shape}, "
      f"corr_hvg {corr_hvg.shape}, comp {comp.shape}, "
      f"signatures {sig_summary['signature'].nunique()}, "
      f"metrics {len(metrics)}, label_transfer {lt.shape}")


# ── Figure scaffold ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.5))   # ~190 × 240 mm
gs = gridspec.GridSpec(3, 2, figure=fig,
                       width_ratios=[1.15, 1.0],
                       height_ratios=[1.0, 1.1, 1.05],
                       wspace=0.40, hspace=0.65,
                       left=0.08, right=0.96, top=0.94, bottom=0.06)


# ── Panel A: UMAP faceted by cohort_platform ────────────────────────────────
print("[03] panel A: integrated UMAP ...")
gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0],
                                         wspace=0.18)
SHORT_CP = {
    "C1-Rectum-CosMx": "C1 · Rectum · CosMx",
    "C1-Rectum-Xenium": "C1 · Rectum · Xenium",
    "C2-Ileum-CosMx": "C2 · Ileum · CosMx",
}
RNG = np.random.default_rng(0)
n_sub = 60_000
sub = umap.sample(n=min(n_sub, len(umap)), random_state=0)
xlim = (umap["UMAP1"].min() - 0.5, umap["UMAP1"].max() + 0.5)
ylim = (umap["UMAP2"].min() - 0.5, umap["UMAP2"].max() + 0.5)
for i, cp in enumerate(config.COHORT_PLATFORMS):
    ax = fig.add_subplot(gs_a[0, i])
    bg = sub[sub["cohort_platform"] != cp]
    fg = sub[sub["cohort_platform"] == cp]
    ax.scatter(bg["UMAP1"], bg["UMAP2"], s=0.5, c="#dddddd",
               linewidths=0, rasterized=True)
    for ct in config.CT_MAJOR_ORDER:
        m = fg["ct_major"] == ct
        if m.sum() == 0:
            continue
        ax.scatter(fg.loc[m, "UMAP1"], fg.loc[m, "UMAP2"], s=0.6,
                   c=config.CT_PALETTE.get(ct, "#999"),
                   linewidths=0, rasterized=True)
    ax.set_title(SHORT_CP[cp], fontsize=6.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if i == 0:
        ax.set_ylabel("UMAP2", fontsize=6)
        ax.set_xlabel("UMAP1", fontsize=6)
        panel_label(ax, "a", x=-0.10)
# legend strip on the right edge of A
handles = [mpatches.Patch(color=config.CT_PALETTE[c], label=c)
           for c in config.CT_MAJOR_ORDER if c != "Unknown"]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.30, 0.635), ncol=4, fontsize=5.5,
           frameon=False, columnspacing=0.5, handlelength=0.8,
           handletextpad=0.3)


# ── Panel B: paired Xenium↔CosMx ─────────────────────────────────────────────
print("[03] panel B: paired Xenium↔CosMx ...")
gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1],
                                         wspace=0.45,
                                         width_ratios=[1.0, 1.05])
ax_b1 = fig.add_subplot(gs_b[0, 0])
ax_b2 = fig.add_subplot(gs_b[0, 1])

# B1 — gene-level scatter, showcase patient, all cell types pooled
sc = showcase.copy()
ax_b1.scatter(sc["cosmx_logcpm"], sc["xenium_logcpm"], s=1.0, alpha=0.25,
              c="#444", linewidths=0, rasterized=True)
mx = max(sc["cosmx_logcpm"].max(), sc["xenium_logcpm"].max()) * 1.05
ax_b1.plot([0, mx], [0, mx], "--", color="#bbbbbb", lw=0.6)
from scipy.stats import pearsonr, spearmanr
r_show, _ = pearsonr(sc["cosmx_logcpm"], sc["xenium_logcpm"])
rho_show, _ = spearmanr(sc["cosmx_logcpm"], sc["xenium_logcpm"])
showcase_pat = sc["patient_id"].iloc[0] if len(sc) else "?"
ax_b1.text(0.04, 0.96,
           f"Patient {showcase_pat} (I0387 inflamed)\n"
           f"Pearson r = {r_show:.2f}\n"
           f"Spearman ρ = {rho_show:.2f}",
           transform=ax_b1.transAxes, ha="left", va="top", fontsize=6)
ax_b1.set_xlabel("CosMx log1p(CPM)", fontsize=6)
ax_b1.set_ylabel("Xenium log1p(CPM)", fontsize=6)
ax_b1.set_xlim(-0.2, mx); ax_b1.set_ylim(-0.2, mx)
ax_b1.tick_params(labelsize=6)
panel_label(ax_b1, "b")

# B2 — per-(patient, ct) Spearman rho heatmap (rank-based, scale-invariant)
piv = paired.pivot(index="ct_major", columns="patient_id",
                   values="spearman_r").reindex(config.CT_MAJOR_ORDER)
piv = piv.dropna(how="all")
im = ax_b2.imshow(piv.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7,
                   aspect="auto")
ax_b2.set_xticks(range(piv.shape[1]))
ax_b2.set_xticklabels(piv.columns, fontsize=6, rotation=30, ha="right")
ax_b2.set_yticks(range(piv.shape[0]))
ax_b2.set_yticklabels(piv.index, fontsize=6)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if pd.notna(v):
            ax_b2.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=5, color="white" if abs(v) > 0.35 else "black")
cbar = fig.colorbar(im, ax=ax_b2, fraction=0.04, pad=0.02)
cbar.ax.tick_params(labelsize=5)
cbar.set_label("Spearman ρ", fontsize=6)


# ── Panel C: HVG + platform-centered cohort pseudobulk corr heatmap ─────────
print("[03] panel C: HVG-centered cohort pseudobulk corr ...")
ax_c = fig.add_subplot(gs[1, 0])
# Sort columns: by (cohort_platform, ct_major) to expose block structure
cols = list(corr_hvg.columns)
def _key(col):
    cp, ct = col.split("|")
    cp_i = config.COHORT_PLATFORMS.index(cp) if cp in config.COHORT_PLATFORMS else 99
    ct_i = config.CT_MAJOR_ORDER.index(ct) if ct in config.CT_MAJOR_ORDER else 99
    return (cp_i, ct_i)
cols_sorted = sorted(cols, key=_key)
m = corr_hvg.loc[cols_sorted, cols_sorted].values
# Mask the diagonal (always 1.0) so the off-diagonal contrast is visible
np.fill_diagonal(m, np.nan)
im = ax_c.imshow(m, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
labels = [c.split("|")[1] for c in cols_sorted]
cps = [c.split("|")[0] for c in cols_sorted]
ax_c.set_xticks(range(len(cols_sorted)))
ax_c.set_xticklabels(labels, fontsize=4.5, rotation=90)
ax_c.set_yticks(range(len(cols_sorted)))
ax_c.set_yticklabels(labels, fontsize=4.5)
# Add cohort_platform annotation lines
prev = cps[0]; start = 0
for i, cp in enumerate(cps + [None]):
    if cp != prev:
        ax_c.axhline(i - 0.5, color="black", lw=0.5)
        ax_c.axvline(i - 0.5, color="black", lw=0.5)
        # Side bar label
        col = config.COHORT_PLATFORM_PALETTE[prev]
        ax_c.add_patch(mpatches.Rectangle(
            (-1.5, start - 0.5), 0.6, i - start,
            facecolor=col, clip_on=False, edgecolor="none"))
        prev = cp
        start = i
cbar = fig.colorbar(im, ax=ax_c, fraction=0.04, pad=0.02)
cbar.set_label("Pearson r (HVG, platform-centered)", fontsize=6)
cbar.ax.tick_params(labelsize=5)
panel_label(ax_c, "c")


# ── Panel D: Cell-type composition stacked bars ─────────────────────────────
print("[03] panel D: composition stacked bars ...")
ax_d = fig.add_subplot(gs[1, 1])
piv_d = (comp.pivot_table(index="sample", columns="ct_major",
                          values="fraction", aggfunc="sum",
                          observed=True)
              .reindex(columns=config.CT_MAJOR_ORDER)
              .fillna(0))
# Sort samples by cohort_platform
sample_to_cp = comp.drop_duplicates("sample").set_index("sample")["cohort_platform"]
order = sorted(piv_d.index,
               key=lambda s: (config.COHORT_PLATFORMS.index(sample_to_cp[s]),
                              s))
piv_d = piv_d.loc[order]
bottoms = np.zeros(len(piv_d))
for ct in config.CT_MAJOR_ORDER:
    if ct not in piv_d.columns:
        continue
    vals = piv_d[ct].values
    ax_d.bar(range(len(piv_d)), vals, bottom=bottoms,
              color=config.CT_PALETTE[ct], width=0.85,
              edgecolor="white", linewidth=0.2, label=ct)
    bottoms += vals
ax_d.set_xticks(range(len(piv_d)))
ax_d.set_xticklabels(piv_d.index, fontsize=4.5, rotation=90)
ax_d.set_ylim(0, 1)
ax_d.set_ylabel("Fraction of cells", fontsize=6)
ax_d.tick_params(labelsize=5)
# Cohort_platform color strip below x-axis
for i, s in enumerate(piv_d.index):
    ax_d.add_patch(mpatches.Rectangle(
        (i - 0.42, -0.06), 0.84, 0.04,
        facecolor=config.COHORT_PLATFORM_PALETTE[sample_to_cp[s]],
        clip_on=False, edgecolor="none"))
# Cohort_platform legend (small)
cp_handles = [mpatches.Patch(color=config.COHORT_PLATFORM_PALETTE[cp],
                              label=SHORT_CP[cp])
              for cp in config.COHORT_PLATFORMS]
ax_d.legend(handles=cp_handles, loc="upper left", fontsize=4.5,
            frameon=False, handlelength=0.6, handletextpad=0.25,
            borderpad=0.1, bbox_to_anchor=(1.01, 1.0))
panel_label(ax_d, "d")


# ── Panel E: IBD signature heatmaps (3 facets, ct × signature) ──────────────
print("[03] panel E: IBD signature heatmap ...")
gs_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0],
                                         wspace=0.10)
sig_order = list(config.IBD_MARKER_PANEL.keys())
SHORT_SIG = {
    "Inflammation_TNF":   "TNF axis",
    "Inflammation_IL17":  "IL-17 axis",
    "Inflammation_IFNg":  "IFN-γ axis",
    "Treg":               "Treg",
    "Myeloid_activation": "Myeloid act.",
    "Epithelial_barrier": "Epi. barrier",
    "Stromal_inflam":     "Stromal infl.",
}
# Z-score per (signature, ct) across cohorts so each cell-type/sig combination
# has the same scale across the 3 cohort_platforms.
piv_long = sig_summary.pivot_table(
    index="cohort_platform", columns=["signature", "ct_major"],
    values="mean_score", observed=True,
)
zmat = piv_long.copy()
zmat = (zmat.sub(zmat.mean(axis=0), axis=1)
             .div(zmat.std(axis=0).replace(0, 1), axis=1))

vmax = float(np.nanpercentile(np.abs(zmat.values), 95))
for i, cp in enumerate(config.COHORT_PLATFORMS):
    ax = fig.add_subplot(gs_e[0, i])
    if cp not in zmat.index:
        ax.axis("off"); continue
    sub = (zmat.loc[cp].unstack(level="ct_major")
                       .reindex(index=sig_order)
                       .reindex(columns=config.CT_MAJOR_ORDER))
    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, fontsize=5, rotation=60, ha="right")
    ax.set_title(SHORT_CP[cp], fontsize=6, pad=2)
    if i == 0:
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([SHORT_SIG.get(s, s) for s in sub.index],
                           fontsize=5.5)
        panel_label(ax, "e", x=-0.30)
    else:
        ax.set_yticks([])
    if i == 2:
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label("z-score", fontsize=5.5)
        cbar.ax.tick_params(labelsize=4.5)


# ── Panel F: label-transfer confusion matrix (C1 → C2) ──────────────────────
print("[03] panel F: C1→C2 label transfer ...")
ax_f = fig.add_subplot(gs[2, 1])
cm = lt.reindex(index=config.CT_MAJOR_ORDER, columns=config.CT_MAJOR_ORDER)
im = ax_f.imshow(cm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax_f.set_xticks(range(len(cm.columns)))
ax_f.set_xticklabels(cm.columns, fontsize=5.5, rotation=45, ha="right")
ax_f.set_yticks(range(len(cm.index)))
ax_f.set_yticklabels(cm.index, fontsize=5.5)
ax_f.set_xlabel("Predicted (from C1)", fontsize=6)
ax_f.set_ylabel("True (C2 label)", fontsize=6)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm.values[i, j]
        if pd.notna(v) and v >= 0.05:
            ax_f.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=5, color="white" if v > 0.55 else "black")
acc = float(np.nanmean(np.diag(cm.values)))
ax_f.set_title(f"C1 → C2 recall (mean diag = {acc:.2f})", fontsize=7)
cbar = fig.colorbar(im, ax=ax_f, fraction=0.04, pad=0.02)
cbar.set_label("Recall", fontsize=6)
cbar.ax.tick_params(labelsize=5)
panel_label(ax_f, "f")


# ── Save ────────────────────────────────────────────────────────────────────
fig.suptitle("Cross-cohort, cross-platform consistency justifies integration "
             "of IBD spatial transcriptomics", fontsize=8, y=1.005)
save_fig(fig, "fig1_consistency")


# ── Extended Data: integration metrics + composition pair corr ──────────────
print("\n[03] extended data: integration metrics ...")
fig2, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
metrics_plot = metrics.dropna(subset=["kbet"])
xs = np.arange(len(metrics_plot))
axes[0].bar(xs, metrics_plot["kbet"], color="#0072B2",
             edgecolor="white", linewidth=0.4)
axes[0].axhline(config.KBET_MIN, color="grey", lw=0.6, ls="--")
axes[0].set_xticks(xs); axes[0].set_xticklabels(metrics_plot["ct_major"],
                                                  fontsize=6, rotation=45,
                                                  ha="right")
axes[0].set_ylabel("kBET acceptance", fontsize=6)
axes[0].tick_params(labelsize=5)
panel_label(axes[0], "a", x=-0.18)

axes[1].bar(xs, metrics_plot["ilisi"], color="#CC79A7",
             edgecolor="white", linewidth=0.4)
axes[1].axhline(2.0, color="grey", lw=0.6, ls="--")
axes[1].set_xticks(xs); axes[1].set_xticklabels(metrics_plot["ct_major"],
                                                  fontsize=6, rotation=45,
                                                  ha="right")
axes[1].set_ylabel("iLISI (effective batches)", fontsize=6)
axes[1].tick_params(labelsize=5)
panel_label(axes[1], "b", x=-0.18)

axes[2].bar(xs, metrics_plot["clisi"], color="#E69F00",
             edgecolor="white", linewidth=0.4)
axes[2].set_xticks(xs); axes[2].set_xticklabels(metrics_plot["ct_major"],
                                                  fontsize=6, rotation=45,
                                                  ha="right")
axes[2].set_ylabel("cLISI (cell-type purity)", fontsize=6)
axes[2].tick_params(labelsize=5)
panel_label(axes[2], "c", x=-0.18)

fig2.suptitle("Per-cell-type integration metrics on scANVI embedding",
               fontsize=8)
fig2.tight_layout()
save_fig(fig2, "extfig1_integration_metrics", out_dir=config.FIG_EXT_DIR)

print(f"\n[03] DONE")
