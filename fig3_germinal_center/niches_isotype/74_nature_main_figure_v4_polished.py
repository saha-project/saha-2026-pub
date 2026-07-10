"""Nature-grade main figure v4 (polished): Three-organ humoral specialization through
distinct lymphoid architectures.

Layout: 3-row × 3-column GridSpec (landscape 18×14 inches)
  Row 0: Panel A (revised anatomical thesis, cols 0-1) | Panel F (4-violin composite, col 2)
  Row 1: Panel B (corrected spatial architecture, cols 0-2)
  Row 2: Panel C (IgG class-switch, col 0) | Panel D (MEI gradient, col 1) | Panel E (JCHAIN heatmap, col 2)

Outputs:
  figures/fig_NATURE_main_v4.png  (600 dpi)
  figures/fig_NATURE_main_v4.pdf  (vector, TrueType embedded)
"""

import sys
from pathlib import Path
import warnings
import importlib.util

# ---------------------------------------------------------------------------
# Global typography & style — must be set BEFORE any artist is created
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42          # TrueType, editable in Illustrator
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI")
IG_DIR = BASE / "ig_validation"
FIG_DIR = BASE / "figures"
SCRIPTS_DIR = BASE / "scripts"

# Safe import of module whose filename starts with a digit
_spec = importlib.util.spec_from_file_location("panel_modules", SCRIPTS_DIR / "67_panel_modules.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
panel_C_igg_class_switch = _mod.panel_C_igg_class_switch

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
ORGAN_ORDER  = ["APE", "ILE", "COL"]
ISOTYPES     = ["IGHA1", "IGHG1", "IGHG2", "IGHM", "IGHD"]

# Font sizes — centralised
FS = dict(
    panel_letter=16,
    panel_title=12,
    axis_label=11,
    tick=9,
    legend=9,
    annotation=8.5,
    sig_star=9,
    suptitle=14,
    organ_name=10,
    organ_desc=8,
)


# ===========================================================================
# Panel A — revised anatomical thesis schematic
# ===========================================================================
def panel_A_anatomical_thesis(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Cecum body (tan/brown ellipse) -----------------------------------
    cecum = mpatches.Ellipse(
        (5.0, 4.5), width=2.8, height=3.2,
        facecolor="#C4A882", edgecolor="#7A5C3A", linewidth=2.0, zorder=3
    )
    ax.add_patch(cecum)
    ax.text(5.0, 4.5, "Cecum", ha="center", va="center",
            fontsize=FS['organ_name'], fontweight="bold", color="#5A3A1A", zorder=5)

    # ---- Ileum — pipe entering from upper-left ----------------------------
    ile_color = ORGAN_COLORS["ILE"]
    ile_path_x = [1.0, 2.2, 3.6]
    ile_path_y = [8.5, 7.0, 5.8]
    ax.plot(ile_path_x, ile_path_y, color=ile_color, linewidth=14,
            solid_capstyle="round", zorder=2, alpha=0.85)
    ax.plot(ile_path_x, ile_path_y, color="#5A1A1A", linewidth=14,
            solid_capstyle="round", zorder=1, alpha=0.25)
    # ILE label
    ax.annotate(
        "ILEUM\n──────────────\nClassical Peyer's patch\nIgA class-switch\nPIGR-mediated mucosal secretion",
        xy=(2.0, 7.8), xycoords="data",
        fontsize=FS['organ_desc'], ha="left", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ile_color,
                  edgecolor="#5A1A1A", alpha=0.92),
        zorder=10
    )
    ax.annotate("", xy=(2.2, 7.2), xytext=(2.0, 7.4),
                arrowprops=dict(arrowstyle="-|>", color=ile_color,
                                lw=1.2, mutation_scale=10), zorder=11)

    # ---- Appendix — curved finger from cecum bottom -----------------------
    ape_color = ORGAN_COLORS["APE"]
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatch
    ap_verts = [
        (4.7, 3.1),
        (4.2, 2.0),
        (3.8, 0.9),
        (4.1, 0.3),
    ]
    ap_codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ap_path = MPath(ap_verts, ap_codes)
    ap_patch = mpatch.PathPatch(
        ap_path, facecolor="none", edgecolor=ape_color,
        linewidth=9, capstyle="round", zorder=4
    )
    ax.add_patch(ap_patch)
    ax.scatter([4.1], [0.3], s=220, color=ape_color, zorder=5,
               edgecolor="#1A5A5A", linewidth=1.2)
    # APE label
    ax.annotate(
        "APPENDIX\n──────────────\nDiffuse B-cell reservoir\nIgG class-switch, plasmablast egress\ngut-dispersal via FcRn-mediated transport",
        xy=(4.1, 0.8), xycoords="data",
        fontsize=FS['organ_desc'], ha="right", va="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ape_color,
                  edgecolor="#1A5A5A", alpha=0.92),
        zorder=10
    )
    ax.annotate("", xy=(4.3, 1.5), xytext=(4.5, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=ape_color,
                                lw=1.2, mutation_scale=10), zorder=11)

    # ---- Colon — continues from right side of cecum upward ----------------
    col_color = ORGAN_COLORS["COL"]
    col_path_x = [6.1, 7.5, 8.8, 9.5]
    col_path_y = [5.2, 5.8, 7.0, 8.5]
    ax.plot(col_path_x, col_path_y, color=col_color, linewidth=14,
            solid_capstyle="round", zorder=2, alpha=0.85)
    ax.plot(col_path_x, col_path_y, color="#0A1A3A", linewidth=14,
            solid_capstyle="round", zorder=1, alpha=0.25)
    # COL label
    ax.annotate(
        "COLON\n──────────────\nScattered active ILFs\n+ bulk JCHAIN secretion factory",
        xy=(9.0, 7.5), xycoords="data",
        fontsize=FS['organ_desc'], ha="right", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=col_color,
                  edgecolor="#0A1A3A", alpha=0.92),
        zorder=10
    )
    ax.annotate("", xy=(8.0, 6.8), xytext=(8.4, 7.1),
                arrowprops=dict(arrowstyle="-|>", color=col_color,
                                lw=1.2, mutation_scale=10), zorder=11)

    # ---- Ileocecal valve marker ------------------------------------------
    ax.scatter([3.8], [5.6], s=60, color="white", zorder=6,
               edgecolor="#7A5C3A", linewidth=1.5)
    ax.text(3.3, 5.9, "ICV", fontsize=7, color="#7A5C3A",
            ha="center", va="bottom", style="italic")

    ax.set_title("GI humoral compartment — distinct lymphoid architectures",
                 fontsize=13, pad=8, fontweight="bold")


# ===========================================================================
# Panel D — Migration switch MEI gradient
# ===========================================================================
def panel_D_mei_gradient(ax):
    df = pd.read_csv(IG_DIR / "plasma_egress_per_unit_niche.csv")
    niche_labels = ["LP-distal", "GALT-edge", "GALT-interior"]

    stats = {}
    for organ, grp in df.groupby("organ"):
        pv = grp.pivot_table(index="unit_id", columns="niche", values="mean_egress")
        pv = pv.reindex(columns=niche_labels)
        means = [pv[c].mean() for c in niche_labels]
        sems  = [pv[c].sem() for c in niche_labels]
        stats[organ] = (means, sems)

    x = np.arange(len(niche_labels))
    for organ in ORGAN_ORDER:
        if organ not in stats:
            continue
        means, sems = stats[organ]
        color = ORGAN_COLORS[organ]
        ax.plot(x, means, marker="o", color=color, linewidth=2.2,
                markersize=8, label=organ, zorder=4)
        ax.fill_between(x,
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        color=color, alpha=0.15, zorder=3)
        delta = means[2] - means[0]
        dy = 0.015 if organ == "APE" else (-0.02 if organ == "COL" else 0.015)
        ax.annotate(
            f"Δ={delta:.2f}",
            xy=(2, means[2]), xytext=(2.12, means[2] + dy),
            fontsize=FS['annotation'], color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8)
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(niche_labels, fontsize=FS['tick'])
    ax.set_ylabel("Egress score\n(CXCR4 − CXCR5 − GPR183)", fontsize=FS['axis_label'])
    ax.set_title("Spatial egress gradient — plasma-lineage restricted",
                 fontsize=FS['panel_title'], pad=6)
    ax.legend(fontsize=FS['legend'], loc="upper left",
              handles=[Line2D([0], [0], color=ORGAN_COLORS[o], linewidth=2, label=o)
                       for o in ORGAN_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis='both', labelsize=FS['tick'])

    # Statistics box — lower-right, outside plot area
    ax.text(0.98, 0.02,
            "APE > COL: p=0.002\nAPE > ILE: p=0.030\nILE > COL: p=0.043\n"
            "(Mann-Whitney, 37 atomic units)",
            transform=ax.transAxes, fontsize=FS['sig_star'], ha="right", va="bottom",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8dc",
                      edgecolor="gray", alpha=0.9))

    ax.text(0.02, 0.97,
            "IGH+ plasma-lineage cells, n=37 atomic units",
            transform=ax.transAxes, fontsize=FS['annotation'], ha="left", va="top",
            style="italic", color="#444")


# ===========================================================================
# Panel E — JCHAIN heatmap + within-SJ01 inset
# ===========================================================================
def panel_E_jchain_heatmap(ax):
    df  = pd.read_csv(IG_DIR / "jchain_isotype_ratio_per_section.csv")
    sig = pd.read_csv(IG_DIR / "jchain_mixedeffects_pairwise.csv")
    wb  = pd.read_csv(IG_DIR / "jchain_within_batch_contrast.csv")

    pivot = (df.groupby(["organ", "isotype"])["frac_jchain_in_iso"]
               .mean()
               .unstack(level="isotype")
               .reindex(ORGAN_ORDER)[ISOTYPES])

    col_sig = {}
    for iso in ISOTYPES:
        sub = sig[(sig.isotype == iso) &
                  (sig.contrast.isin(["COL_vs_ILE", "COL_vs_APE"]))]
        any_sig = (sub["p_bonf"] < 0.05).any()
        col_sig[iso] = any_sig

    vmin = pivot.values.min()
    vmax = pivot.values.max()

    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(ISOTYPES)))
    ax.set_xticklabels(ISOTYPES, fontsize=FS['tick'], rotation=30, ha="right")
    ax.set_yticks(range(len(ORGAN_ORDER)))
    ax.set_yticklabels(ORGAN_ORDER, fontsize=FS['tick'] + 1, fontweight="bold")
    for i, organ in enumerate(ORGAN_ORDER):
        ax.get_yticklabels()[i].set_color(ORGAN_COLORS[organ])

    for ri, organ in enumerate(ORGAN_ORDER):
        for ci, iso in enumerate(ISOTYPES):
            val = pivot.loc[organ, iso]
            star = "*" if (organ == "COL" and col_sig.get(iso, False)) else ""
            ax.text(ci, ri, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=10,
                    color="white" if val > (vmin + (vmax - vmin) * 0.6) else "black",
                    fontweight="bold" if star else "normal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction JCHAIN+ in isotype+ cells", fontsize=10)
    cbar.ax.tick_params(labelsize=FS['tick'])

    # Significance stars fontsize and weight
    ax.set_title("JCHAIN secretion coupling per organ × isotype\n(15 sections, mixed-effects; * p<0.05 Bonferroni)",
                 fontsize=FS['panel_title'], pad=6)

    wb_sj01 = wb[wb.batch == "SJ01-only"]
    if len(wb_sj01) > 0:
        pivot_wb = (wb_sj01.groupby(["organ", "isotype"])["frac_jchain_in_iso"]
                    .mean()
                    .unstack(level="isotype"))
        organs_wb = [o for o in ORGAN_ORDER if o in pivot_wb.index]
        isos_wb   = [i for i in ISOTYPES if i in pivot_wb.columns]
        pivot_wb  = pivot_wb.reindex(organs_wb)[isos_wb]

        # Tighter inset — reduced padding
        axins = ax.inset_axes([0.52, 0.56, 0.44, 0.40])
        im2 = axins.imshow(pivot_wb.values, aspect="auto", cmap="Oranges",
                           vmin=pivot_wb.values.min(), vmax=pivot_wb.values.max())
        axins.set_xticks(range(len(isos_wb)))
        axins.set_xticklabels(isos_wb, fontsize=6.5, rotation=30, ha="right")
        axins.set_yticks(range(len(organs_wb)))
        axins.set_yticklabels(organs_wb, fontsize=7)
        for ii, o in enumerate(organs_wb):
            axins.get_yticklabels()[ii].set_color(ORGAN_COLORS[o])
        for ri, o in enumerate(organs_wb):
            for ci, iso in enumerate(isos_wb):
                val = pivot_wb.loc[o, iso]
                axins.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                           fontsize=6, color="black")
        axins.set_title("Within SJ01 batch only", fontsize=7, pad=2)
        plt.colorbar(im2, ax=axins, fraction=0.08, pad=0.02)


# ===========================================================================
# Panel F — 4-violin composite scores (2×2 inside the Panel F grid cell)
# ===========================================================================
def _get_unit_means(df, score):
    """Average across batches for the same sample, return per-sample means per organ."""
    sub = df[df["score_name"] == score].copy()
    unit_means = sub.groupby(["organ", "sample"])["value"].mean().reset_index()
    return unit_means


def _get_bh_stars(stats_df, score, organ_a, organ_b):
    """Return BH-corrected stars string for a pairwise comparison."""
    row = stats_df[
        (stats_df["score"] == score) &
        (stats_df["organ_A"] == organ_a) &
        (stats_df["organ_B"] == organ_b)
    ]
    if len(row) == 0:
        row = stats_df[
            (stats_df["score"] == score) &
            (stats_df["organ_A"] == organ_b) &
            (stats_df["organ_B"] == organ_a)
        ]
    if len(row) == 0:
        return "ns"
    stars = row.iloc[0]["stars_bh"]
    return str(stars) if pd.notna(stars) else "ns"


def _draw_significance_bracket(ax, x1, x2, y, h, stars, fontsize=9):
    """Draw a significance bracket between two x positions."""
    if stars == "ns":
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, color="black")
    ax.text((x1 + x2) / 2, y + h + 0.005, stars,
            ha="center", va="bottom", fontsize=fontsize, color="black",
            fontweight="bold")


def panel_F_violin_composite(ax_parent):
    """4-panel (2×2) violin grid showing GALT-interior composite cell-state scores."""
    df = pd.read_csv(IG_DIR / "galt_interior_composite_per_unit.csv")
    stats_df = pd.read_csv(IG_DIR / "galt_interior_composite_stats.csv")

    panels = [
        ("Egress",      "Egress score\n(APE wins, BH*)",        "APE > COL *"),
        ("Light-zone",  "Light-zone score\n(COL wins, BH**)",   "COL > APE **"),
        ("B-retention", "B-retention score\n(APE trend, n.s.)", "trend ns"),
        ("Gut-homing",  "Gut-homing score\n(COL wins, BH*)",    "COL > APE *"),
    ]

    ax_parent.axis("off")

    fig = ax_parent.figure
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=ax_parent.get_subplotspec(),
        hspace=0.60, wspace=0.50
    )

    for idx, (score_name, title, winner_note) in enumerate(panels):
        row_i = idx // 2
        col_i = idx % 2
        ax = fig.add_subplot(inner_gs[row_i, col_i])

        unit_means = _get_unit_means(df, score_name)

        x_labels = ["APE", "ILE", "COL"]

        violin_data = []
        for organ in x_labels:
            vals = unit_means[unit_means["organ"] == organ]["value"].dropna().values
            violin_data.append(vals)

        parts = ax.violinplot(
            violin_data,
            positions=[0, 1, 2],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.7
        )
        for pc, organ in zip(parts["bodies"], x_labels):
            pc.set_facecolor(ORGAN_COLORS[organ])
            pc.set_edgecolor("none")
            pc.set_alpha(0.45)

        # Strip dots — edge color black, linewidth 0.4
        rng = np.random.default_rng(42)
        for xi, organ in enumerate(x_labels):
            vals = unit_means[unit_means["organ"] == organ]["value"].dropna().values
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=ORGAN_COLORS[organ], s=12, alpha=0.75,
                       zorder=4, linewidths=0.4,
                       edgecolors="black")

        # Mean + 95% CI bar
        for xi, organ in enumerate(x_labels):
            vals = unit_means[unit_means["organ"] == organ]["value"].dropna().values
            if len(vals) < 2:
                continue
            mean_v = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(xi, mean_v, yerr=ci, fmt="D",
                        color="black", markersize=4, linewidth=1.2,
                        capsize=3, zorder=6)

        # Significance brackets
        all_vals = np.concatenate(violin_data)
        y_max = all_vals.max() if len(all_vals) > 0 else 1.0
        y_min = all_vals.min() if len(all_vals) > 0 else 0.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        bracket_offset = y_range * 0.12
        bracket_h = y_range * 0.04
        current_y = y_max + bracket_offset * 0.5

        comparisons = [
            ("APE", "COL", 0, 2),
            ("APE", "ILE", 0, 1),
            ("ILE", "COL", 1, 2),
        ]
        for org_a, org_b, xi_a, xi_b in comparisons:
            stars = _get_bh_stars(stats_df, score_name, org_a, org_b)
            if stars not in ("ns", ""):
                _draw_significance_bracket(ax, xi_a, xi_b, current_y, bracket_h,
                                           stars, fontsize=FS['sig_star'])
                current_y += bracket_offset * 0.9

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(x_labels, fontsize=FS['tick'], fontweight="bold")
        for tick, organ in zip(ax.get_xticklabels(), x_labels):
            tick.set_color(ORGAN_COLORS[organ])

        # Score title fontsize 10
        ax.set_title(title, fontsize=10, pad=4, fontweight="bold")
        ax.tick_params(axis="y", labelsize=FS['tick'])
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

        ax.set_ylim(y_min - y_range * 0.1, current_y + bracket_h * 2)

        ax.text(0.97, 0.97, winner_note, transform=ax.transAxes,
                fontsize=FS['annotation'], ha="right", va="top", style="italic",
                color="#555555")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel F super-title
    ax_parent.text(
        0.5, 1.01,
        "GALT-interior composite cell-state scores\n(IGH+ cells, per atomic unit)",
        transform=ax_parent.transAxes, ha="center", va="bottom",
        fontsize=9, fontweight="bold", color="#222"
    )


# ===========================================================================
# Main figure assembly
# ===========================================================================
def build_figure():
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45, wspace=0.30,
        height_ratios=[1.0, 1.4, 1.0],
        width_ratios=[1.2, 1.2, 1.0],
    )

    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.04, right=0.98)

    # ---- Panel A (revised anatomy) — row 0, cols 0-1 ------------------------
    ax_A = fig.add_subplot(gs[0, 0:2])
    panel_A_anatomical_thesis(ax_A)
    ax_A.text(-0.02, 1.04, "A", transform=ax_A.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Panel F (4-violin composite) — row 0, col 2 -------------------------
    ax_F = fig.add_subplot(gs[0, 2])
    panel_F_violin_composite(ax_F)
    ax_F.text(-0.08, 1.04, "F", transform=ax_F.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Panel B (spatial architecture) — row 1, cols 0-2 -------------------
    ax_B = fig.add_subplot(gs[1, 0:3])
    ax_B.axis("off")
    fig_b_path = FIG_DIR / "fig_B_spatial_architecture_corrected.png"
    if fig_b_path.exists():
        img = mpimg.imread(str(fig_b_path))
        ax_B.imshow(img, aspect="equal")
        ax_B.set_title(
            "Corrected spatial architecture — representative atomic units "
            "(APE: diffuse B-cell foci | ILE: Peyer's patch | COL: scattered active ILFs)",
            fontsize=FS['panel_title'], pad=6
        )
    else:
        ax_B.text(0.5, 0.5,
                  "[Panel B] fig_B_spatial_architecture_corrected.png not found",
                  transform=ax_B.transAxes, ha="center", va="center",
                  fontsize=11, color="gray")
    ax_B.text(-0.01, 1.04, "B", transform=ax_B.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Panel C (IgG class-switch) — row 2, col 0 --------------------------
    ax_C = fig.add_subplot(gs[2, 0])
    panel_C_igg_class_switch(ax_C)
    # Apply Arial-consistent axis label sizes post-call
    ax_C.title.set_fontsize(FS['panel_title'])
    ax_C.xaxis.label.set_fontsize(FS['axis_label'])
    ax_C.yaxis.label.set_fontsize(FS['axis_label'])
    ax_C.tick_params(axis='both', labelsize=FS['tick'])
    # Thinner bar edges
    for patch in ax_C.patches:
        patch.set_linewidth(0.6)
    ax_C.text(-0.12, 1.04, "C", transform=ax_C.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Panel D (MEI gradient) — row 2, col 1 ------------------------------
    ax_D = fig.add_subplot(gs[2, 1])
    panel_D_mei_gradient(ax_D)
    ax_D.text(-0.12, 1.04, "D", transform=ax_D.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Panel E (JCHAIN heatmap) — row 2, col 2 ---------------------------
    ax_E = fig.add_subplot(gs[2, 2])
    panel_E_jchain_heatmap(ax_E)
    ax_E.text(-0.12, 1.04, "E", transform=ax_E.transAxes,
              fontsize=FS['panel_letter'], fontweight="bold", va="top")

    # ---- Figure super-title -------------------------------------------------
    fig.suptitle(
        "Three-organ humoral specialization through distinct lymphoid architectures",
        fontsize=FS['suptitle'], fontweight="bold", y=0.97
    )

    return fig


# ===========================================================================
# Save
# ===========================================================================
def main():
    print("Building Nature main figure v4 (polished)...")
    fig = build_figure()

    out_png = FIG_DIR / "fig_NATURE_main_v4.png"
    out_pdf = FIG_DIR / "fig_NATURE_main_v4.pdf"

    fig.savefig(str(out_png), dpi=600, bbox_inches="tight", facecolor="white")
    print(f"  Saved PNG: {out_png}  ({out_png.stat().st_size // 1024} KB)")

    fig.savefig(str(out_pdf), bbox_inches="tight", facecolor="white")
    print(f"  Saved PDF: {out_pdf}  ({out_pdf.stat().st_size // 1024} KB)")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
