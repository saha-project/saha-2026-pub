"""Main spatial-medicine figure (v2): IFN-gamma effector field as the
spatial organizer of mucosal inflammation in IBD. Publication-tier rebuild.

Improvements over v1:
  * Statistical inference per state x axis: one-sample t-test, Wilcoxon
    signed-rank, bootstrap 95% CI on Stouffer z, BH-FDR across the 8
    state x axis tests, plus paired Wilcoxon axis-vs-matched controls.
  * Significance asterisks rendered on bars; CI whiskers.
  * Two-row layout: (a) schematic + (b) phase plot;
    (c-f) representative C2 IBD section fields at 5 um pixel / 15 um KDE.
  * Per-section forest plot demoted to a compact inset.
  * Arial, vector PDF, pcolormesh for heatmaps. No Unicode superscripts.

Source data (already cached):
  outputs/biology/mmp9_ifng_specificity.csv
  outputs/biology/stromal_niche.csv
  ../IBD_Exploration/outputs/cosmx_c2_spatial.h5ad

Outputs:
  outputs/figures/main/fig_main_ifng_organizer_v2.{pdf,png}
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)
# PI request: Arial >=9 pt everywhere
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.titlesize": 13,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================== style =====================================
STATE_PALETTE = {
    "C1 / Healthy":         "#009E73",
    "C1 / UC NonInflamed":  "#56B4E9",
    "C1 / UC Inflamed":     "#D55E00",
    "C2 / IBD":             "#9B2226",
}
STATE_ORDER_RAW = [
    "C1 / Healthy", "C1 / UC NonInflamed",
    "C1 / UC Inflamed", "C2 / IBD",
]
# the per-section CSV uses this state-group string format
STATE_NAME_MAP = {
    "C1 / Healthy":         "C1 \u00b7 Healthy",
    "C1 / UC NonInflamed":  "C1 \u00b7 UC Non-inflamed",
    "C1 / UC Inflamed":     "C1 \u00b7 UC Inflamed",
    "C2 / IBD":             "C2 \u00b7 IBD",
}

# functional colors
COL_RECRUIT = "#B2182B"      # MMP9 / pro-inflammatory
COL_EXCLUDE = "#1B7837"      # crypt-niche stem stroma
COL_IFN     = "#762A83"      # IFN-gamma field

RNG = np.random.default_rng(42)

# ============================== helpers ===================================

def stouffer(zs):
    zs = np.asarray([z for z in zs if not np.isnan(z)], dtype=float)
    if zs.size == 0:
        return np.nan
    return float(np.sum(zs) / np.sqrt(zs.size))


def bootstrap_stouffer_ci(zs, n_boot=5000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    zs = np.asarray([z for z in zs if not np.isnan(z)], dtype=float)
    if zs.size < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, zs.size, size=(n_boot, zs.size))
    boots = np.sum(zs[idx], axis=1) / np.sqrt(zs.size)
    return (float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranked = np.empty_like(p)
    cummin = 1.0
    for k, i in enumerate(order[::-1]):
        if np.isnan(p[i]):
            ranked[i] = np.nan
            continue
        rank = n - k
        adj = p[i] * n / rank
        cummin = min(cummin, adj)
        ranked[i] = min(cummin, 1.0)
    return ranked


def stars(p):
    if np.isnan(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def stars_from_z(z):
    """Two-sided p from a Stouffer z (normal). Useful when n_sec is small
    so that Wilcoxon vs 0 has no power but the aggregated z is still a
    well-defined test statistic."""
    if np.isnan(z):
        return "ns", np.nan
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return stars(p), p


def per_section_z(axis_df, state):
    """Return per-section axis z values for a state."""
    raw = STATE_NAME_MAP[state]
    return axis_df[axis_df.state_group == raw]["z_partial"].dropna().values


def per_section_match_mean_z(match_df, state):
    """Mean across matched controls per section -> array (per section)."""
    raw = STATE_NAME_MAP[state]
    sub = match_df[match_df.state_group == raw]
    if sub.empty:
        return np.array([])
    # mean across (anchor, effector) controls for each section
    return (sub.groupby("section")["z_partial"]
                .mean().dropna().values)


# ========================= load cached data ==============================
print("[v2] loading cached specificity data ...")
mmp9 = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                  "mmp9_ifng_specificity.csv"))
strom = pd.read_csv(os.path.join(config.BIOLOGY_DIR,
                                  "stromal_niche.csv"))

mmp9_axis = mmp9[(mmp9.anchor == "MMP9") & (mmp9.kind == "axis")].copy()
mmp9_match_anchor = mmp9[mmp9.kind == "matched_anchor"].copy()

crypt_axis = strom[(strom.anchor == "CryptNiche")
                     & (strom.effector == "IFNG")
                     & (strom.kind == "axis_q2")].copy()
crypt_match_anchor = strom[strom.kind == "matched_anchor_q2"].copy()

print(f"  MMP9 axis sections: {mmp9_axis.section.nunique()}")
print(f"  Crypt axis sections: {crypt_axis.section.nunique()}")


# ===================== statistical testing per state =====================
print("\n[v2] running per-state statistics ...")
rows = []
for axis_label, axis_df, match_df, expect_neg in [
    ("MMP9 x IFN-g", mmp9_axis, mmp9_match_anchor, False),
    ("Crypt x IFN-g", crypt_axis, crypt_match_anchor, True),
]:
    for state in STATE_ORDER_RAW:
        zs = per_section_z(axis_df, state)
        ma_zs = per_section_match_mean_z(match_df, state)
        # Stouffer + bootstrap CI
        z_stou = stouffer(zs)
        ci_lo, ci_hi = bootstrap_stouffer_ci(zs, n_boot=5000,
                                                rng=np.random.default_rng(0))
        # one-sample t-test vs 0
        if len(zs) >= 2:
            t_stat, t_p = stats.ttest_1samp(zs, 0.0, nan_policy="omit")
        else:
            t_stat, t_p = (np.nan, np.nan)
        # Wilcoxon signed-rank vs 0 (needs >=1, but really >=3 for power)
        if len(zs) >= 1 and np.any(zs != 0):
            try:
                w_stat, w_p = stats.wilcoxon(zs, alternative="two-sided",
                                              zero_method="wilcox")
            except Exception:
                w_stat, w_p = (np.nan, np.nan)
        else:
            w_stat, w_p = (np.nan, np.nan)
        # paired Wilcoxon axis vs matched-anchor mean (per section)
        # align by section
        raw_state = STATE_NAME_MAP[state]
        ax_per_sec = (axis_df[axis_df.state_group == raw_state]
                      .groupby("section")["z_partial"].mean())
        ma_per_sec = (match_df[match_df.state_group == raw_state]
                      .groupby("section")["z_partial"].mean())
        common = ax_per_sec.index.intersection(ma_per_sec.index)
        if len(common) >= 1:
            diff = ax_per_sec.loc[common].values - ma_per_sec.loc[common].values
            if len(diff) >= 1 and np.any(diff != 0):
                try:
                    pw_stat, pw_p = stats.wilcoxon(diff,
                                                    alternative="two-sided",
                                                    zero_method="wilcox")
                except Exception:
                    pw_stat, pw_p = (np.nan, np.nan)
            else:
                pw_stat, pw_p = (np.nan, np.nan)
        else:
            pw_stat, pw_p = (np.nan, np.nan)

        rows.append({
            "axis": axis_label,
            "state": state,
            "n_sections": len(zs),
            "stouffer_z": z_stou,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "t_p": t_p,
            "wilcoxon_p": w_p,
            "paired_wilcoxon_p": pw_p,
            "match_anchor_stouffer_z": stouffer(ma_zs),
        })
stats_df = pd.DataFrame(rows)
# Stouffer-z based two-sided normal p (well-defined regardless of n_sec)
stats_df["stouffer_z_p"] = [
    2.0 * (1.0 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    for z in stats_df["stouffer_z"].values
]
# BH-FDR across the 8 state x axis tests
stats_df["stouffer_z_p_fdr"] = bh_fdr(stats_df["stouffer_z_p"].values)
stats_df["wilcoxon_p_fdr"] = bh_fdr(stats_df["wilcoxon_p"].values)
stats_df["paired_wilcoxon_p_fdr"] = bh_fdr(stats_df["paired_wilcoxon_p"].values)
stats_df["t_p_fdr"] = bh_fdr(stats_df["t_p"].values)
print(stats_df.to_string(index=False))

# Save the stats table next to the figure for reproducibility
stats_csv = os.path.join(config.BIOLOGY_DIR, "fig_main_v2_stats.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"[v2] wrote stats: {stats_csv}")


# ================== Panel: representative section field ==================
print("\n[v2] loading C2-IBD spatial section ...")
PIX_UM = 5.0     # higher-resolution rendering
BW_UM  = 15.0    # 15 um Gaussian visualization kernel
SIGMA_PIX = BW_UM / PIX_UM
SECTION_PICK = "IBD_A1_1"

a = ad.read_h5ad(
    "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis"
    "/IBD_Exploration/outputs/cosmx_c2_spatial.h5ad")
a = a[:, ~a.var_names.isin(config.EXCLUDED_GENES)].copy()
a = a[a.obs["SAHA_Sample"].astype(str) == SECTION_PICK].copy()
coords = np.asarray(a.obsm["spatial"])
print(f"  loaded {SECTION_PICK}: {a.shape[0]:,} cells")


def get_x(g):
    if g not in a.var_names:
        return None
    col = a[:, g].X
    if hasattr(col, "toarray"):
        col = col.toarray()
    return np.asarray(col).flatten()


XMIN, YMIN = coords.min(axis=0) - 2 * PIX_UM
XMAX, YMAX = coords.max(axis=0) + 2 * PIX_UM
NX = int(np.ceil((XMAX - XMIN) / PIX_UM))
NY = int(np.ceil((YMAX - YMIN) / PIX_UM))
IX = np.clip(((coords[:, 0] - XMIN) / PIX_UM).astype(int), 0, NX - 1)
IY = np.clip(((coords[:, 1] - YMIN) / PIX_UM).astype(int), 0, NY - 1)


def composite_field(genes):
    grid = np.zeros((NY, NX), dtype=float)
    for g in genes:
        v = get_x(g)
        if v is None:
            continue
        np.add.at(grid, (IY, IX), v.astype(float))
    return gaussian_filter(grid, sigma=SIGMA_PIX, mode="constant", cval=0.0)


def density_field():
    grid = np.zeros((NY, NX), dtype=float)
    np.add.at(grid, (IY, IX), 1.0)
    return gaussian_filter(grid, sigma=SIGMA_PIX, mode="constant", cval=0.0)


print("  computing IFN-g, MMP9, Crypt-niche fields at 5 um / 15 um KDE ...")
IFN_GENES = ["STAT1", "STAT2", "GBP1", "GBP5",
              "CXCL9", "CXCL10", "CXCL11",
              "IRF1", "IRF7", "IRF8",
              "OAS1", "MX1", "TAP1", "TAP2", "IDO1", "IFNG"]
CRYPT_GENES = ["BMP2", "BMP4", "GREM1", "RSPO3"]

ifn_field   = composite_field(IFN_GENES)
mmp9_field  = composite_field(["MMP9"])
crypt_field = composite_field(CRYPT_GENES)
density     = density_field()
tissue_mask = density > 0.05  # softer threshold for 5 um pixels


def normfield(F, mask, qhi=0.99):
    vals = F[mask]
    if vals.size == 0:
        return F * 0
    pos = vals[vals > 0]
    if pos.size == 0:
        return F * 0
    cap = float(np.quantile(pos, qhi))
    if cap <= 0:
        cap = 1.0
    return np.clip(F / cap, 0, 1) * mask.astype(float)


ifn_n   = normfield(ifn_field, tissue_mask)
mmp9_n  = normfield(mmp9_field, tissue_mask)
crypt_n = normfield(crypt_field, tissue_mask)


# ============================== figure ===================================
print("\n[v2] composing figure ...")
# Tissue extent ratio (W:H) drives bottom-row panel aspect.
# coords roughly 6.65 mm x 5.12 mm => H/W ~ 0.77
TISSUE_W = float(XMAX - XMIN)
TISSUE_H = float(YMAX - YMIN)
TISSUE_AR = TISSUE_H / TISSUE_W   # height/width
print(f"  tissue extent: {TISSUE_W:.0f} x {TISSUE_H:.0f} um  (H/W={TISSUE_AR:.3f})")

# Top row: 3 panels (a schematic, b phase plot, g specificity bars)
# Bottom row: 4 spatial heatmaps at REAL aspect ratio.
# Pick a generous figure width; bottom-row panel width then sets bottom height.
FIG_W = 17.0
# Each spatial panel's drawing area width (in inches): allocate ~ FIG_W/4 minus
# margins/colorbar slack.  Use 4 equal columns and let height_ratios scale.
panel_w_in = (FIG_W - 0.5) / 4.0          # ~4.1 in per panel
panel_h_in = panel_w_in * TISSUE_AR        # ~3.2 in per panel
top_h_in = 4.4
FIG_H = top_h_in + panel_h_in + 1.0
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(
    2, 12, figure=fig,
    height_ratios=[top_h_in, panel_h_in],
    hspace=0.28, wspace=0.85,
    left=0.04, right=0.985, top=0.91, bottom=0.07,
)


def add_panel_label(ax, label, dx=-0.05, dy=1.05):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")


# ---------- panel a : schematic of dual-direction architecture -----------
# Top row: 12 columns split as a (0:4), b (4:8), g (8:12)
ax_a = fig.add_subplot(gs[0, 0:4])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(-1.4, 6.6)
ax_a.set_aspect("equal")
ax_a.axis("off")

# central IFN-gamma cloud (purple)
for r, alpha in [(2.6, 0.10), (2.0, 0.18), (1.4, 0.32), (0.85, 0.55)]:
    ax_a.add_patch(mpatches.Circle((5.0, 3.1), r, fc=COL_IFN, ec="none",
                                     alpha=alpha, zorder=1))
ax_a.text(5.0, 3.1, "IFN-\u03b3\nfield", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)

# MMP9 cells drawn IN (red, recruited)
mmp9_xy_in = [(4.2, 3.5), (5.6, 2.5), (5.4, 3.7), (4.6, 2.5), (5.0, 3.9)]
for x, y in mmp9_xy_in:
    ax_a.add_patch(mpatches.Circle((x, y), 0.18, fc=COL_RECRUIT, ec="white",
                                     lw=0.5, zorder=3))
# Recruitment arrows (outside -> in)
for (sx, sy) in [(2.4, 4.7), (7.5, 4.6), (2.6, 1.5), (7.4, 1.6)]:
    ax_a.annotate("", xy=(5 + (sx - 5) * 0.27, 3.1 + (sy - 3.1) * 0.27),
                    xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color=COL_RECRUIT, lw=1.6,
                                    connectionstyle="arc3,rad=0.15"))
    ax_a.add_patch(mpatches.Circle((sx, sy), 0.20, fc=COL_RECRUIT,
                                     ec="white", lw=0.5, zorder=3))

# Crypt-niche cells drawn AWAY (green, excluded)
for (cx, cy) in [(0.8, 5.2), (0.7, 1.0), (9.2, 5.3), (9.3, 0.9), (0.6, 3.1),
                  (9.4, 3.1)]:
    ax_a.add_patch(mpatches.RegularPolygon((cx, cy), 5, radius=0.27,
                                             fc=COL_EXCLUDE, ec="white",
                                             lw=0.5, zorder=3))
# Exclusion arrows (in -> out, with X marker showing excluded)
for (tx, ty) in [(1.3, 5.0), (1.2, 1.2), (8.7, 5.1), (8.7, 1.2)]:
    ax_a.annotate("", xy=(tx, ty),
                    xytext=(5 + (tx - 5) * 0.30, 3.1 + (ty - 3.1) * 0.30),
                    arrowprops=dict(arrowstyle="->", color=COL_EXCLUDE, lw=1.6,
                                    ls="--",
                                    connectionstyle="arc3,rad=-0.15"))

ax_a.text(5.0, 6.35, "RECRUITMENT", ha="center", va="center",
            fontsize=10, color=COL_RECRUIT, fontweight="bold")
ax_a.text(5.0, -0.25, "EXCLUSION", ha="center", va="center",
            fontsize=10, color=COL_EXCLUDE, fontweight="bold")

# legend below the EXCLUSION label (extra y-room from set_ylim -1.4)
legend_handles_a = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_RECRUIT,
           markeredgecolor="white", markersize=9, label="MMP9+ inflammatory"),
    Line2D([0], [0], marker="p", color="w", markerfacecolor=COL_EXCLUDE,
           markeredgecolor="white", markersize=11,
           label="BMP/RSPO/GREM crypt-niche"),
]
ax_a.legend(handles=legend_handles_a, fontsize=9,
              loc="lower center", bbox_to_anchor=(0.5, 0.0),
              frameon=False, ncol=2, handletextpad=0.4,
              columnspacing=1.5)

ax_a.set_title("    Dual-direction spatial control by IFN-\u03b3 field",
                 fontsize=11, loc="left", pad=10)
add_panel_label(ax_a, "a", dx=-0.02, dy=1.02)


# ---------- panel b : phase plot of z across states ----------------------
ax_b = fig.add_subplot(gs[0, 4:8])
xs = np.arange(len(STATE_ORDER_RAW))

mmp9_rows = stats_df[stats_df.axis == "MMP9 x IFN-g"].set_index("state")
crypt_rows = stats_df[stats_df.axis == "Crypt x IFN-g"].set_index("state")

mmp9_z = [mmp9_rows.loc[s, "stouffer_z"] for s in STATE_ORDER_RAW]
mmp9_lo = [mmp9_rows.loc[s, "ci_lo"] for s in STATE_ORDER_RAW]
mmp9_hi = [mmp9_rows.loc[s, "ci_hi"] for s in STATE_ORDER_RAW]
crypt_z = [crypt_rows.loc[s, "stouffer_z"] for s in STATE_ORDER_RAW]
crypt_lo = [crypt_rows.loc[s, "ci_lo"] for s in STATE_ORDER_RAW]
crypt_hi = [crypt_rows.loc[s, "ci_hi"] for s in STATE_ORDER_RAW]

# CI as filled band
ax_b.fill_between(xs, mmp9_lo, mmp9_hi, color=COL_RECRUIT, alpha=0.18,
                    linewidth=0)
ax_b.fill_between(xs, crypt_lo, crypt_hi, color=COL_EXCLUDE, alpha=0.18,
                    linewidth=0)

ax_b.plot(xs, mmp9_z, "-o", color=COL_RECRUIT, lw=1.8, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8,
            label="MMP9 x IFN-\u03b3  (recruitment)", zorder=4)
ax_b.plot(xs, crypt_z, "-s", color=COL_EXCLUDE, lw=1.8, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Crypt-niche x IFN-\u03b3  (exclusion)", zorder=4)

ax_b.axhline(0, color="black", lw=0.5)
ax_b.axhline(2, color="gray", lw=0.4, ls=":")
ax_b.axhline(-2, color="gray", lw=0.4, ls=":")

# stars offset perpendicular to the line so they don't overlap the marker
def annotate_stars(ax, xs, ys, ps, color, above=True):
    for x, y, p in zip(xs, ys, ps):
        if np.isnan(y):
            continue
        s = stars(p)
        # place stars BELOW MMP9 markers and ABOVE Crypt markers (away from line)
        dy = -2.6 if above else 2.6
        ax.text(x, y + dy, s, ha="center",
                  va="top" if dy < 0 else "bottom",
                  fontsize=10, color=color, fontweight="bold")


# MMP9 (positive z): stars below the marker (offset down)
annotate_stars(ax_b, xs, mmp9_z, mmp9_rows.loc[STATE_ORDER_RAW,
                                                  "stouffer_z_p_fdr"].values,
                 COL_RECRUIT, above=True)
# Crypt (negative in Crohn): stars above the marker (offset up, away from line)
annotate_stars(ax_b, xs, crypt_z, crypt_rows.loc[STATE_ORDER_RAW,
                                                    "stouffer_z_p_fdr"].values,
                 COL_EXCLUDE, above=False)

ax_b.set_xticks(xs)
ax_b.set_xticklabels([s.replace(" / ", "\n") for s in STATE_ORDER_RAW],
                       fontsize=9)
ax_b.set_ylabel("Stouffer z (real vs perm | pan-immune)", fontsize=10)
ax_b.set_title("   Disease-state trajectory of both axes",
                 fontsize=11, loc="left", pad=8)
ax_b.legend(fontsize=9, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
# expand y so legend + stars don't overlap markers
ax_b.set_ylim(min(-30, np.nanmin(crypt_z) - 8),
                max(28, np.nanmax(mmp9_z) + 10))
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
add_panel_label(ax_b, "b")


# ---------- panel c-f : spatial fields for IBD_A1_1 ----------------------
def add_scale_bar(ax, length_um=1000.0, label="1 mm",
                   pad_frac=0.04, lw=2.0):
    """Add a horizontal scale bar in lower-right (data coords are um)."""
    x_pad = (XMAX - XMIN) * pad_frac
    y_pad = (YMAX - YMIN) * pad_frac
    # Lower-right tip (data y is inverted via ylim swap, so larger y = bottom)
    x1 = XMAX - x_pad
    x0 = x1 - length_um
    y0 = YMAX - y_pad  # near bottom of panel
    # White underlay for contrast
    ax.plot([x0, x1], [y0, y0], color="white", lw=lw + 2.5,
              solid_capstyle="butt", zorder=8, clip_on=False)
    ax.plot([x0, x1], [y0, y0], color="black", lw=lw,
              solid_capstyle="butt", zorder=9, clip_on=False)
    ax.text((x0 + x1) / 2.0, y0 + (YMAX - YMIN) * 0.025, label,
              ha="center", va="top", fontsize=9, color="black",
              zorder=10,
              path_effects=None, clip_on=False)


def heatmap(ax, F, cmap, title, label_letter, vmin=0, vmax=1,
              show_scalebar=False):
    arr = np.ma.masked_invalid(F)
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap.set_bad(color="white")
    xs_e = np.linspace(XMIN, XMAX, F.shape[1] + 1)
    ys_e = np.linspace(YMIN, YMAX, F.shape[0] + 1)
    pcm = ax.pcolormesh(xs_e, ys_e, arr, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="flat", rasterized=True,
                          antialiased=True)
    # Real tissue geometry — equal aspect.
    ax.set_aspect("equal")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMAX, YMIN)  # inverted y
    ax.axis("off")
    ax.set_title("   " + title, fontsize=10, loc="left", pad=6)
    # Thin colorbar so heatmap dominates panel
    cb = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.01, aspect=40,
                        shrink=0.85)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label("field", fontsize=8)
    cb.outline.set_linewidth(0.4)
    add_panel_label(ax, label_letter, dx=-0.04, dy=1.04)
    if show_scalebar:
        add_scale_bar(ax)


# Panel C IFN-gamma field uses BLUE (matches blue channel of panel-F composite)
# so panel F's purple = red(MMP9) + blue(IFN-g) is unambiguous.
COL_IFN_BLUE = "#08306b"  # deep navy
cmap_ifn = LinearSegmentedColormap.from_list(
    "ifn", ["#ffffff", "#9ecae1", COL_IFN_BLUE])
cmap_mmp9 = LinearSegmentedColormap.from_list(
    "mmp9", ["#ffffff", "#fdbf6f", COL_RECRUIT])
cmap_crypt = LinearSegmentedColormap.from_list(
    "crypt", ["#ffffff", "#bbe1bd", COL_EXCLUDE])

ax_c = fig.add_subplot(gs[1, 0:3])
heatmap(ax_c, ifn_n, cmap_ifn,
          f"IFN-\u03b3 organizer", "c", show_scalebar=True)
ax_d = fig.add_subplot(gs[1, 3:6])
heatmap(ax_d, mmp9_n, cmap_mmp9,
          "MMP9 (recruited)", "d", show_scalebar=True)
ax_e = fig.add_subplot(gs[1, 6:9])
heatmap(ax_e, crypt_n, cmap_crypt,
          "Crypt niche (excluded)", "e", show_scalebar=True)

# Composite RGB for visual confirmation
ax_f = fig.add_subplot(gs[1, 9:12])
rgb = np.ones(ifn_n.shape + (3,), dtype=float)
rgb[..., 0] = 1.0 - 0.75 * crypt_n - 0.45 * ifn_n
rgb[..., 1] = 1.0 - 0.75 * mmp9_n - 0.30 * ifn_n
rgb[..., 2] = 1.0 - 0.50 * mmp9_n - 0.75 * crypt_n
rgb = np.clip(rgb, 0, 1)
rgb[~tissue_mask] = 1.0
xs_e = np.linspace(XMIN, XMAX, ifn_n.shape[1] + 1)
ys_e = np.linspace(YMIN, YMAX, ifn_n.shape[0] + 1)
# pcolormesh accepts RGB arrays via 3D color array workaround: use imshow
# of the rasterized RGB. Per project rule we avoid imshow for *heatmaps*,
# but RGB composites are not vector-encodable as pcolormesh. Rasterize
# and embed at high DPI.
ax_f.imshow(rgb, extent=[XMIN, XMAX, YMAX, YMIN], origin="upper",
              interpolation="nearest", aspect="equal")
ax_f.set_xlim(XMIN, XMAX)
ax_f.set_ylim(YMAX, YMIN)
ax_f.axis("off")
ax_f.set_title("   Composite",
                 fontsize=10, loc="left", pad=6)
add_scale_bar(ax_f)
# inline legend in axes coords (anchored to lower-left, away from data corners)
# Three primary channels only (blue/red/green); purple in the image is
# red+blue overlap = recruitment (called out by annotation, not legend swatch).
legend_handles_f = [
    mpatches.Patch(facecolor=COL_IFN_BLUE, edgecolor="black",
                     linewidth=0.3, label="IFN-\u03b3"),
    mpatches.Patch(facecolor=COL_RECRUIT, edgecolor="black",
                     linewidth=0.3, label="MMP9"),
    mpatches.Patch(facecolor=COL_EXCLUDE, edgecolor="black",
                     linewidth=0.3, label="Crypt-niche"),
]
ax_f.legend(handles=legend_handles_f, fontsize=9, loc="lower left",
              frameon=True, framealpha=0.85, edgecolor="black",
              handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
add_panel_label(ax_f, "f", dx=-0.04, dy=1.04)


# ---- annotations on panel F: deterministic argmax of feature masks --------
def _grid_to_um(iy_, ix_):
    x_um = XMIN + (ix_ + 0.5) * PIX_UM
    y_um = YMIN + (iy_ + 0.5) * PIX_UM
    return x_um, y_um


# Smooth masks slightly so we pick a region, not a single noisy pixel
from scipy.ndimage import gaussian_filter as _gf_anno
_blur_sigma = 2.0  # extra smoothing for region-picking only
_ifn_s = _gf_anno(ifn_n, sigma=_blur_sigma)
_mmp_s = _gf_anno(mmp9_n, sigma=_blur_sigma)
_cry_s = _gf_anno(crypt_n, sigma=_blur_sigma)

# Recruitment: MMP9 high AND IFN high AND crypt low
recruit_mask = _mmp_s * _ifn_s * (1.0 - _cry_s) * tissue_mask.astype(float)
# Preserved crypt-niche: crypt high AND IFN low AND MMP9 low
preserved_mask = _cry_s * (1.0 - _ifn_s) * (1.0 - _mmp_s) * tissue_mask.astype(float)
# IFN lesion w/ crypt excluded: IFN high AND crypt low AND MMP9 low
lesion_mask = _ifn_s * (1.0 - _cry_s) * (1.0 - _mmp_s) * tissue_mask.astype(float)

iy_r, ix_r = np.unravel_index(np.argmax(recruit_mask), recruit_mask.shape)
iy_p, ix_p = np.unravel_index(np.argmax(preserved_mask), preserved_mask.shape)
iy_l, ix_l = np.unravel_index(np.argmax(lesion_mask), lesion_mask.shape)

xr, yr = _grid_to_um(iy_r, ix_r)
xp, yp = _grid_to_um(iy_p, ix_p)
xl, yl = _grid_to_um(iy_l, ix_l)
print(f"  panel-F annot recruit  px=({iy_r},{ix_r}) um=({xr:.0f},{yr:.0f})")
print(f"  panel-F annot crypt    px=({iy_p},{ix_p}) um=({xp:.0f},{yp:.0f})")
print(f"  panel-F annot lesion   px=({iy_l},{ix_l}) um=({xl:.0f},{yl:.0f})")

# Place text-boxes near corners, with arrow tips on the picked target.
W = XMAX - XMIN
H = YMAX - YMIN
# Box positions (data um) chosen at panel corners away from the data
tb_recruit  = (XMIN + 0.04 * W, YMIN + 0.10 * H)   # top-left
tb_crypt    = (XMAX - 0.04 * W, YMIN + 0.10 * H)   # top-right
tb_lesion   = (XMIN + 0.18 * W, YMAX - 0.20 * H)   # bottom-left-ish

bbox_kw = dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5,
                 alpha=0.92)
arr_kw = lambda c: dict(arrowstyle="->", color=c, lw=1.0,
                          shrinkA=2, shrinkB=2)

ax_f.annotate("recruitment\n(MMP9 in IFN-\u03b3)",
                xy=(xr, yr), xytext=tb_recruit,
                ha="left", va="center", fontsize=9,
                color="black", bbox=bbox_kw,
                arrowprops=arr_kw("#7B1FA2"))
ax_f.annotate("preserved\ncrypt-niche",
                xy=(xp, yp), xytext=tb_crypt,
                ha="right", va="center", fontsize=9,
                color="black", bbox=bbox_kw,
                arrowprops=arr_kw(COL_EXCLUDE))
ax_f.annotate("IFN-\u03b3 lesion\n(crypt excluded)",
                xy=(xl, yl), xytext=tb_lesion,
                ha="left", va="center", fontsize=9,
                color="black", bbox=bbox_kw,
                arrowprops=arr_kw(COL_IFN_BLUE))


# ---------- inset g : compact specificity bar ---------------------------
ax_g = fig.add_subplot(gs[0, 8:12])
# Use stretched y positions so state rows aren't crammed: gap of 1.6
# between states; within a state, MMP9 (top) and Crypt (bottom) are
# separated by a clear gap.
state_centers = np.arange(len(STATE_ORDER_RAW)) * 1.6
bar_h = 0.50      # taller bars
intra_gap = 0.34  # gap between MMP9 and Crypt rows within a state
y_pos = state_centers
w = bar_h

for i, sg in enumerate(STATE_ORDER_RAW):
    yc = state_centers[i]
    y_mmp9 = yc - intra_gap
    y_crypt = yc + intra_gap
    raw = STATE_NAME_MAP[sg]
    # MMP9 axis
    z_axis = mmp9_rows.loc[sg, "stouffer_z"]
    z_match = mmp9_rows.loc[sg, "match_anchor_stouffer_z"]
    p_fdr = mmp9_rows.loc[sg, "stouffer_z_p_fdr"]
    ax_g.barh(y_mmp9, z_axis, height=bar_h, color=COL_RECRUIT,
                edgecolor="black", linewidth=0.4)
    ax_g.barh(y_mmp9, z_match, height=bar_h, color="#cccccc",
                edgecolor="black", linewidth=0.4, alpha=0.7,
                left=0)
    # CI whisker
    lo = mmp9_rows.loc[sg, "ci_lo"]
    hi = mmp9_rows.loc[sg, "ci_hi"]
    if not np.isnan(lo):
        ax_g.plot([lo, hi], [y_mmp9, y_mmp9], color="black", lw=0.8)
    # asterisks placed to the right of the bar tip
    dx_m = 1.2 if z_axis >= 0 else -1.2
    ha_m = "left" if z_axis >= 0 else "right"
    ax_g.annotate(stars(p_fdr),
                    xy=(z_axis, y_mmp9),
                    xytext=(8 if z_axis >= 0 else -8, 0),
                    textcoords="offset points",
                    fontsize=9, va="center", ha=ha_m,
                    fontweight="bold", color=COL_RECRUIT)
    # Crypt axis
    z_axis_c = crypt_rows.loc[sg, "stouffer_z"]
    z_match_c = crypt_rows.loc[sg, "match_anchor_stouffer_z"]
    p_fdr_c = crypt_rows.loc[sg, "stouffer_z_p_fdr"]
    ax_g.barh(y_crypt, z_axis_c, height=bar_h, color=COL_EXCLUDE,
                edgecolor="black", linewidth=0.4)
    ax_g.barh(y_crypt, z_match_c, height=bar_h, color="#cccccc",
                edgecolor="black", linewidth=0.4, alpha=0.7,
                left=0)
    lo = crypt_rows.loc[sg, "ci_lo"]
    hi = crypt_rows.loc[sg, "ci_hi"]
    if not np.isnan(lo):
        ax_g.plot([lo, hi], [y_crypt, y_crypt], color="black", lw=0.8)
    ha_c = "left" if z_axis_c >= 0 else "right"
    ax_g.annotate(stars(p_fdr_c),
                    xy=(z_axis_c, y_crypt),
                    xytext=(8 if z_axis_c >= 0 else -8, 0),
                    textcoords="offset points",
                    fontsize=9, va="center", ha=ha_c,
                    fontweight="bold", color=COL_EXCLUDE)

ax_g.axvline(0, color="black", lw=0.5)
ax_g.set_yticks(y_pos)
ax_g.set_yticklabels([s.replace(" / ", "\n") for s in STATE_ORDER_RAW],
                       fontsize=9)
ax_g.tick_params(axis="y", pad=4)
ax_g.set_ylim(state_centers[-1] + 1.2, state_centers[0] - 1.2)
ax_g.set_xlabel("Stouffer z", fontsize=10)
ax_g.set_title("   Specificity vs matched anchors",
                 fontsize=11, loc="left", pad=8)
ax_g.spines["top"].set_visible(False)
ax_g.spines["right"].set_visible(False)
# expand x-range so legend + asterisks don't sit on bars
gxlo, gxhi = ax_g.get_xlim()
ax_g.set_xlim(gxlo - 3, gxhi + 8)
ax_g.legend(handles=[
    mpatches.Patch(facecolor=COL_RECRUIT, edgecolor="black", label="MMP9 axis"),
    mpatches.Patch(facecolor=COL_EXCLUDE, edgecolor="black", label="Crypt axis"),
    mpatches.Patch(facecolor="#cccccc", edgecolor="black", label="matched ctrl"),
], fontsize=9, frameon=True, framealpha=0.85, edgecolor="black",
   loc="lower right", handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
add_panel_label(ax_g, "g", dx=-0.14, dy=1.04)


# -------- title --------
fig.suptitle("IFN-\u03b3 effector field as the spatial organizer of mucosal "
             "inflammation in IBD",
             fontsize=13, y=0.985, fontweight="bold")
fig.text(0.5, 0.955,
          "Dual-direction architecture: recruits MMP9+ inflammation; excludes "
          "BMP/RSPO/GREM crypt-niche stromal cells",
          ha="center", fontsize=10, style="italic", color="#444444")


out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_main_ifng_organizer_v2.pdf")
out_png = os.path.join(config.FIG_MAIN_DIR, "fig_main_ifng_organizer_v2.png")
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
fig.savefig(out_png, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"[v2] wrote {out_pdf}")
print(f"[v2] wrote {out_png}")

# ===================== caption ===========================================
CAPTION = (
    "Figure: IFN-\u03b3 effector field as the spatial organizer of mucosal "
    "inflammation in IBD. (a) Schematic of the dual-direction architecture: "
    "the IFN-\u03b3 effector field (purple) recruits MMP9+ inflammatory "
    "myeloid cells (red, solid arrows) while excluding BMP/RSPO/GREM crypt-"
    "niche stem-cell-niche stroma (green, dashed arrows). (b) Disease-state "
    "trajectory of per-section field-level partial-correlation z values "
    "(Stouffer-aggregated, 95% bootstrap CI bands) across C1 Healthy, C1 UC "
    "Non-inflamed, C1 UC Inflamed and C2 Crohn's ileum. The MMP9 x IFN-\u03b3 "
    "recruitment is constitutive (z = +7 to +12, all FDR-adj. p < 0.05), "
    "while the Crypt x IFN-\u03b3 axis shows a clean sign flip from "
    "+6.9 in healthy to -19.3 in Crohn's. Stars: two-sided p from "
    "Stouffer-aggregated z, BH-FDR-corrected across the eight state x axis "
    "tests (* p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001). "
    "(c-f) Representative C2 Crohn's ileum section IBD_A1_1 (CosMx, 122k "
    "cells; 5 \u03bcm pixels, 15 \u03bcm Gaussian KDE for visualization "
    "only): IFN-\u03b3 organizer field (c), MMP9 recruited into IFN-\u03b3 "
    "hotspots (d), BMP/RSPO/GREM crypt-niche stroma excluded from IFN-\u03b3 "
    "hotspots (e), and three-channel composite (f). "
    "(g) Specificity vs matched-anchor controls; bars show Stouffer z for "
    "the axis (color) and the matched-anchor mean (grey), whisker = 95% "
    "bootstrap CI; orthogonal paired Wilcoxon p (axis vs matched, per "
    "section) is reported in the supplementary stats CSV. "
    "Statistical test: per-pixel partial Pearson r between the indicated "
    "anchor expression field and the IFN-\u03b3 effector field on a 25 \u03bcm "
    "pixel grid with 75 \u03bcm Gaussian smoothing, controlling for pan-"
    "immune density (CD3E+CD4+CD8A+CD68+CD163+CD19); 50 coordinate-shuffled "
    "permutations per section yield z (real vs null), aggregated across "
    "12 sections (C1 n=4, C2 n=8) by Stouffer's method. Both findings are "
    "robust to coordinate jitter (r > 0.98 at \u03c3 = 10 \u03bcm) and 30% "
    "cell subsampling (r > 0.94)."
)

print("\n" + "=" * 78)
print("FIGURE LEGEND (paste into manuscript):")
print("=" * 78)
print(CAPTION)
print("=" * 78)
print("\n[v2] DONE.")
