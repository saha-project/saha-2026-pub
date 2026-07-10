"""04 — Supplementary figure: hexbin scatter + LR conservation + spatial nhood.

Reads:
  outputs/concordance/paired_xenium_cosmx_per_gene.csv
  outputs/biology/lr_jaccard.csv, lr_rank_corr.csv, lr_top_overlap.csv
  outputs/biology/nhood_zscore_*.csv

Writes:
  outputs/figures/main/fig2_supplementary_consistency.{pdf,svg,png}

Layout (3 rows × 1 col, ~180 × 220 mm):
  A: Hexbin density of paired Xenium↔CosMx (showcase patient C1_P4)
     + per-CT Pearson r bar inset
  B: LIANA L-R conservation across cohort_platforms
     (Jaccard heatmap + rank-correlation heatmap + top-overlap dot table)
  C: Spatial neighborhood enrichment (nhood z-score) per cohort_platform
     — 3 facets, common color scale
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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)


def panel_label(ax, label, x=-0.10, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def save_fig(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    svg = os.path.join(out_dir, f"{name}.svg")
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf, format="pdf", dpi=1200, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"     -> {os.path.basename(pdf)}, .svg, .png")


SHORT_CP = {
    "C1-Rectum-CosMx":  "C1 · Rectum · CosMx",
    "C1-Rectum-Xenium": "C1 · Rectum · Xenium",
    "C2-Ileum-CosMx":   "C2 · Ileum · CosMx",
}

t0 = time.time()
print("[04] loading cached panel data ...")
showcase = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                     "paired_xenium_cosmx_per_gene.csv"))
paired = pd.read_csv(os.path.join(config.CONCORDANCE_DIR,
                                   "paired_xenium_cosmx_corr.csv"))

lr_jacc = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                    "lr_jaccard.csv"), index_col=0)
lr_rank = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                    "lr_rank_corr.csv"), index_col=0)
lr_top = pd.read_csv(os.path.join(config.BIOLOGY_DIR, "lr_top_overlap.csv"))

# Spatial nhood (optional — figure handles missing files gracefully)
nhood_files = {
    cp: os.path.join(config.BIOLOGY_DIR, f"nhood_zscore_{cp}.csv")
    for cp in config.COHORT_PLATFORMS
}
nhood_avail = {cp: os.path.exists(p) for cp, p in nhood_files.items()}
print("     nhood files present:", nhood_avail)


# ── Figure scaffold ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.5))
gs = gridspec.GridSpec(3, 1, figure=fig,
                       height_ratios=[1.0, 1.0, 1.05],
                       hspace=0.55,
                       left=0.10, right=0.94, top=0.95, bottom=0.05)


# ── Panel A: hexbin density of paired Xenium↔CosMx ─────────────────────────
print("[04] panel A: hexbin paired scatter ...")
gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                                         wspace=0.32,
                                         width_ratios=[1.05, 0.95])
ax_a1 = fig.add_subplot(gs_a[0, 0])
ax_a2 = fig.add_subplot(gs_a[0, 1])

sc = showcase.copy()
mx = max(sc["cosmx_logcpm"].max(), sc["xenium_logcpm"].max()) * 1.05
hb = ax_a1.hexbin(sc["cosmx_logcpm"], sc["xenium_logcpm"],
                   gridsize=70, mincnt=1, bins="log",
                   cmap="viridis", linewidths=0)
ax_a1.plot([0, mx], [0, mx], "--", color="white", lw=0.7, alpha=0.7)
r_show, _ = pearsonr(sc["cosmx_logcpm"], sc["xenium_logcpm"])
rho_show, _ = spearmanr(sc["cosmx_logcpm"].to_numpy(),
                         sc["xenium_logcpm"].to_numpy())
showcase_pat = sc["patient_id"].iloc[0] if len(sc) else "?"
ax_a1.text(0.04, 0.96,
           f"Patient {showcase_pat} (I0387 inflamed)\n"
           f"All cell types pooled\n"
           f"Pearson r = {r_show:.2f}\n"
           f"Spearman ρ = {rho_show:.2f}\n"
           f"n = {len(sc):,} (gene × cell type)",
           transform=ax_a1.transAxes, ha="left", va="top",
           fontsize=6, color="white",
           bbox=dict(facecolor="black", alpha=0.4, edgecolor="none",
                     boxstyle="round,pad=0.2"))
ax_a1.set_xlabel("CosMx log1p(CPM)", fontsize=7)
ax_a1.set_ylabel("Xenium log1p(CPM)", fontsize=7)
ax_a1.set_xlim(-0.1, mx); ax_a1.set_ylim(-0.1, mx)
ax_a1.tick_params(labelsize=6)
cb = fig.colorbar(hb, ax=ax_a1, fraction=0.04, pad=0.02)
cb.set_label("log10(N genes per bin)", fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_label(ax_a1, "a", x=-0.14)

# A2: per-(patient, ct) Spearman + Pearson grouped bars
ct_summary = (paired.dropna(subset=["pearson_r"])
                     .groupby("ct_major", observed=True)
                     [["pearson_r", "spearman_r"]].mean()
                     .reindex(config.CT_MAJOR_ORDER).dropna())
xs = np.arange(len(ct_summary))
w = 0.35
ax_a2.bar(xs - w / 2, ct_summary["pearson_r"], w, color="#0072B2",
           edgecolor="white", lw=0.4, label="Pearson r")
ax_a2.bar(xs + w / 2, ct_summary["spearman_r"], w, color="#CC79A7",
           edgecolor="white", lw=0.4, label="Spearman ρ")
ax_a2.axhline(0, color="black", lw=0.4)
ax_a2.set_xticks(xs)
ax_a2.set_xticklabels(ct_summary.index, fontsize=6, rotation=45, ha="right")
ax_a2.set_ylabel("Mean across 4 patients", fontsize=7)
ax_a2.set_ylim(-0.05, max(0.7, ct_summary.values.max() * 1.1))
ax_a2.legend(fontsize=6, frameon=False, loc="upper right",
              handlelength=0.8, handletextpad=0.3)
ax_a2.tick_params(labelsize=6)
ax_a2.set_title("Paired Xenium↔CosMx (C1, n=4 patients)", fontsize=7)


# ── Panel B: LIANA L-R conservation ────────────────────────────────────────
print("[04] panel B: LIANA L-R conservation ...")
gs_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0],
                                         wspace=0.55,
                                         width_ratios=[1.0, 1.0, 1.4])
ax_b1 = fig.add_subplot(gs_b[0, 0])
ax_b2 = fig.add_subplot(gs_b[0, 1])
ax_b3 = fig.add_subplot(gs_b[0, 2])

# B1 — Jaccard heatmap
cps = list(lr_jacc.columns)
short_cps = [SHORT_CP.get(c, c) for c in cps]
m = lr_jacc.values
im = ax_b1.imshow(m, cmap="Blues", vmin=0, vmax=max(0.3, np.nanmax(m)),
                   aspect="auto")
ax_b1.set_xticks(range(len(cps))); ax_b1.set_yticks(range(len(cps)))
ax_b1.set_xticklabels(short_cps, fontsize=5, rotation=30, ha="right")
ax_b1.set_yticklabels(short_cps, fontsize=5)
ax_b1.set_title("Jaccard of top-100 L–R pairs", fontsize=7)
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m[i, j]
        ax_b1.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if v < 0.4 else "white")
cb = fig.colorbar(im, ax=ax_b1, fraction=0.04, pad=0.02)
cb.ax.tick_params(labelsize=5)
panel_label(ax_b1, "b", x=-0.18)

# B2 — Spearman of -magnitude_rank on union top pairs
m = lr_rank.values
im = ax_b2.imshow(m, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
ax_b2.set_xticks(range(len(cps))); ax_b2.set_yticks(range(len(cps)))
ax_b2.set_xticklabels(short_cps, fontsize=5, rotation=30, ha="right")
ax_b2.set_yticklabels(short_cps, fontsize=5)
ax_b2.set_title("Spearman ρ of L–R strength\n(union top pairs)",
                 fontsize=7)
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m[i, j]
        if pd.notna(v):
            ax_b2.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5,
                        color="white" if abs(v) > 0.5 else "black")
cb = fig.colorbar(im, ax=ax_b2, fraction=0.04, pad=0.02)
cb.ax.tick_params(labelsize=5)

# B3 — Top conserved pairs (present in ≥2 cohort_platforms by top-100)
top_set_per_cp = (lr_top.groupby("cohort_platform")
                       .apply(lambda g: set(g["pair"].unique()))
                       .to_dict())
pair_counts = (lr_top.groupby("pair")
                     .agg(n_cps=("cohort_platform", "nunique"),
                          mean_rank=("magnitude_rank", "mean"),
                          source=("source", "first"),
                          target=("target", "first"),
                          ligand=("ligand_complex", "first"),
                          receptor=("receptor_complex", "first")))
shared = pair_counts[pair_counts["n_cps"] >= 2].sort_values("mean_rank")
shared = shared.head(20)
labels = [f"{r['source'][:5]}→{r['target'][:5]}: "
          f"{r['ligand'][:8]}|{r['receptor'][:8]}"
          for _, r in shared.iterrows()]
mat_long = (lr_top[lr_top["pair"].isin(shared.index)]
                .pivot_table(index="pair", columns="cohort_platform",
                             values="magnitude_rank", aggfunc="mean")
                .reindex(index=shared.index, columns=cps))
# Lower magnitude_rank = stronger; invert for display
disp = -np.log10(mat_long.replace(0, np.nan))
im = ax_b3.imshow(disp.values, cmap="viridis", aspect="auto")
ax_b3.set_xticks(range(len(cps)))
ax_b3.set_xticklabels(short_cps, fontsize=5, rotation=30, ha="right")
ax_b3.set_yticks(range(len(labels)))
ax_b3.set_yticklabels(labels, fontsize=4.5)
ax_b3.set_title("Top L–R pairs shared across ≥2 cohort_platforms",
                 fontsize=7)
cb = fig.colorbar(im, ax=ax_b3, fraction=0.04, pad=0.02)
cb.set_label("-log10(magnitude_rank)", fontsize=5.5)
cb.ax.tick_params(labelsize=5)


# ── Panel C: Spatial neighborhood enrichment per cohort_platform ───────────
print("[04] panel C: spatial nhood enrichment ...")
gs_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0],
                                         wspace=0.10)
all_z = []
for cp in config.COHORT_PLATFORMS:
    if not nhood_avail.get(cp):
        continue
    df = pd.read_csv(nhood_files[cp], index_col=0)
    all_z.append(df.values.flatten())
if all_z:
    arr = np.concatenate([a[~np.isnan(a)] for a in all_z])
    vmax = float(np.nanpercentile(np.abs(arr), 95))
else:
    vmax = 5.0

for i, cp in enumerate(config.COHORT_PLATFORMS):
    ax = fig.add_subplot(gs_c[0, i])
    if not nhood_avail.get(cp):
        ax.text(0.5, 0.5, f"{cp}\n(not computed)", ha="center", va="center",
                fontsize=6, transform=ax.transAxes)
        ax.axis("off")
        continue
    df = pd.read_csv(nhood_files[cp], index_col=0)
    df = df.reindex(index=config.CT_MAJOR_ORDER,
                    columns=config.CT_MAJOR_ORDER)
    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=5, rotation=60, ha="right")
    ax.set_title(SHORT_CP[cp], fontsize=7)
    if i == 0:
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=5.5)
        panel_label(ax, "c", x=-0.32)
    else:
        ax.set_yticks([])
    if i == 2:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.set_label("Neighborhood enrichment z-score\n(squidpy, 350 µm radius)",
                     fontsize=5.5)
        cb.ax.tick_params(labelsize=5)


# ── Save ───────────────────────────────────────────────────────────────────
fig.suptitle("Supplementary cross-cohort consistency: density-shaded paired "
             "expression, L–R conservation, and spatial niche enrichment",
             fontsize=8, y=0.99)
save_fig(fig, "fig2_supplementary_consistency", out_dir=config.FIG_MAIN_DIR)

print(f"\n[04] DONE in {time.time()-t0:.1f}s")
