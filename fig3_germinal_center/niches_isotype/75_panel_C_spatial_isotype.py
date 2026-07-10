"""Spatial visualization of IgG vs IgA plasma-cell distribution per organ.

Cross-validates the snPATHO finding (APE plasma cells IgG-biased, ILE IgA-biased)
in spatial CosMx data, and extends it to COL which lacks snPATHO data.

For each representative atomic unit:
- Classify each cell as Pure-IgG (IGHG1+ or IGHG2+, IGHA1 absent),
  Pure-IgA (IGHA1+ only), Both (mixed), or Neither (background)
- Plot spatially with two-color overlay (red=IgG, blue=IgA)
- Compute per-niche fractions to quantify the bias by tissue zone

Outputs (figures/):
- fig_C_spatial_isotype_distribution.{png,pdf}

CSV (ig_validation/):
- spatial_isotype_per_unit_niche.csv
"""

from pathlib import Path
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8

CLEAN_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned")
OUT_DIR = Path(__file__).resolve().parent.parent / "ig_validation"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

ORGANS = ["APE", "ILE", "COL"]
ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
NICHE_GENES = ["CCL19", "CCL21", "CXCL13"]
K_NICHE = 20
LAYER = "normalized"

REP_UNITS = {
    "APE": ("APE_A9_4", "SJ01"),
    "ILE": ("ILE_A1_3", "SJ01"),
    "COL": ("COL_A1_4", "EC02"),
}

IGG_COLOR = "#D62728"
IGA_COLOR = "#1F77B4"
BOTH_COLOR = "#9467BD"
BG_COLOR = "#dddddd"


def gvec(a, gene):
    if gene not in a.var_names:
        return None
    j = a.var_names.get_loc(gene)
    M = a.layers[LAYER] if LAYER in a.layers else a.X
    col = M[:, j]
    return np.asarray(col.toarray() if sp.issparse(col) else col).ravel()


def classify_isotype(a):
    igg1 = gvec(a, "IGHG1"); igg2 = gvec(a, "IGHG2")
    iga1 = gvec(a, "IGHA1")
    iga_pos = (iga1 > 0) if iga1 is not None else np.zeros(a.n_obs, dtype=bool)
    igg_pos = np.zeros(a.n_obs, dtype=bool)
    if igg1 is not None: igg_pos |= (igg1 > 0)
    if igg2 is not None: igg_pos |= (igg2 > 0)
    label = np.full(a.n_obs, "Neither", dtype=object)
    label[iga_pos & ~igg_pos] = "Pure-IgA"
    label[igg_pos & ~iga_pos] = "Pure-IgG"
    label[iga_pos & igg_pos] = "Both"
    return label, igg_pos, iga_pos


def smooth_score(coords, score, k):
    if len(coords) < k + 1:
        return score
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = nn.kneighbors(coords)
    return score[idx[:, 1:]].mean(axis=1)


def niche_per_cell(a, coords):
    chemo = np.zeros(a.n_obs)
    for g in NICHE_GENES:
        e = gvec(a, g)
        if e is not None:
            chemo += e
    chemo_smooth = smooth_score(coords, chemo, K_NICHE)
    qs = np.quantile(chemo_smooth, [0.6, 0.9])
    bins = np.full(a.n_obs, "LP-distal", dtype=object)
    bins[chemo_smooth >= qs[0]] = "GALT-edge"
    bins[chemo_smooth >= qs[1]] = "GALT-interior"
    return bins


def load_unit(organ):
    sample, batch = REP_UNITS[organ]
    a = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
    m = (a.obs["SAHA_Sample"] == sample) & (a.obs["batch_id"] == batch)
    a_u = a[m].copy()
    coords = np.column_stack([
        a_u.obs["x_centroid"].to_numpy(float),
        a_u.obs["y_centroid"].to_numpy(float),
    ])
    valid = np.isfinite(coords).all(axis=1)
    a_u = a_u[valid]
    coords = coords[valid]
    label, igg_pos, iga_pos = classify_isotype(a_u)
    niches = niche_per_cell(a_u, coords)
    return dict(coords=coords, label=label, igg=igg_pos, iga=iga_pos,
                niches=niches, sample=sample, batch=batch)


def per_unit_per_niche_isotype():
    rows = []
    for organ in ORGANS:
        a = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
        for sample in sorted(a.obs["SAHA_Sample"].unique()):
            for batch in sorted(a.obs["batch_id"].unique()):
                m = (a.obs["SAHA_Sample"] == sample) & (a.obs["batch_id"] == batch)
                if m.sum() < 2000:
                    continue
                a_u = a[m].copy()
                coords = np.column_stack([
                    a_u.obs["x_centroid"].to_numpy(float),
                    a_u.obs["y_centroid"].to_numpy(float),
                ])
                valid = np.isfinite(coords).all(axis=1)
                a_u = a_u[valid]
                coords = coords[valid]
                label, igg_pos, iga_pos = classify_isotype(a_u)
                niches = niche_per_cell(a_u, coords)
                for niche in ["LP-distal", "GALT-edge", "GALT-interior"]:
                    m2 = niches == niche
                    n_total = int(m2.sum())
                    n_pure_igg = int((label[m2] == "Pure-IgG").sum())
                    n_pure_iga = int((label[m2] == "Pure-IgA").sum())
                    n_both = int((label[m2] == "Both").sum())
                    n_igh_pos = n_pure_igg + n_pure_iga + n_both
                    if n_igh_pos < 10:
                        continue
                    rows.append(dict(
                        organ=organ, sample=sample, batch=batch,
                        unit_id=f"{sample}__{batch}", niche=niche,
                        n_cells=n_total, n_igh_pos=n_igh_pos,
                        frac_pure_igg=n_pure_igg / n_igh_pos,
                        frac_pure_iga=n_pure_iga / n_igh_pos,
                        frac_both=n_both / n_igh_pos,
                        igg_iga_ratio=(n_pure_igg / n_pure_iga) if n_pure_iga > 0 else np.nan,
                    ))
    return pd.DataFrame(rows)


def main():
    print("Loading representative atomic units...")
    rep = {organ: load_unit(organ) for organ in ORGANS}

    print("Computing per-niche isotype fractions per atomic unit...")
    df = per_unit_per_niche_isotype()
    df.to_csv(OUT_DIR / "spatial_isotype_per_unit_niche.csv", index=False)
    print(f"  Wrote {OUT_DIR / 'spatial_isotype_per_unit_niche.csv'} ({len(df)} rows)")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.35, wspace=0.25)

    for ci, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[0, ci])
        d = rep[organ]
        coords = d["coords"]; lab = d["label"]
        m_neither = lab == "Neither"
        m_igg = lab == "Pure-IgG"
        m_iga = lab == "Pure-IgA"
        m_both = lab == "Both"
        ax.scatter(coords[m_neither, 0], coords[m_neither, 1],
                   s=0.15, c=BG_COLOR, alpha=0.45, rasterized=True)
        ax.scatter(coords[m_iga, 0], coords[m_iga, 1],
                   s=1.0, c=IGA_COLOR, alpha=0.85, edgecolor="none",
                   rasterized=True, label="Pure-IgA")
        ax.scatter(coords[m_igg, 0], coords[m_igg, 1],
                   s=1.0, c=IGG_COLOR, alpha=0.85, edgecolor="none",
                   rasterized=True, label="Pure-IgG")
        ax.scatter(coords[m_both, 0], coords[m_both, 1],
                   s=0.8, c=BOTH_COLOR, alpha=0.55, edgecolor="none",
                   rasterized=True, label="Both")

        n_igg = int(m_igg.sum()); n_iga = int(m_iga.sum())
        ratio = n_igg / max(n_iga, 1)
        ax.set_title(f"{organ}  ({d['sample']} × {d['batch']})\n"
                     f"Pure-IgG n={n_igg:,} | Pure-IgA n={n_iga:,} | "
                     f"IgG:IgA = {ratio:.2f}",
                     fontsize=10, color=ORGAN_COLORS[organ])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.axis("off")
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        bar_len = 500
        bx = xmin + 0.05 * (xmax - xmin)
        by = ymin + 0.04 * (ymax - ymin)
        ax.plot([bx, bx + bar_len], [by, by], color="black", linewidth=2)
        ax.text(bx + bar_len / 2, by - 0.03 * (ymax - ymin),
                "500 µm", ha="center", va="top", fontsize=8)

    handles = [
        mpatches.Patch(facecolor=IGA_COLOR, label="Pure-IgA (IGHA1+)"),
        mpatches.Patch(facecolor=IGG_COLOR, label="Pure-IgG (IGHG1/2+)"),
        mpatches.Patch(facecolor=BOTH_COLOR, label="Both"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0.965), frameon=False)

    # Bottom row: per-niche IgG vs IgA fractions across organs
    metrics = [("frac_pure_igg", "Pure-IgG fraction (of IGH+ cells)", IGG_COLOR),
               ("frac_pure_iga", "Pure-IgA fraction (of IGH+ cells)", IGA_COLOR)]
    for mi, (col, ylabel, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, mi])
        niche_order = ["LP-distal", "GALT-edge", "GALT-interior"]
        x = np.arange(len(niche_order))
        bar_w = 0.27
        for oi, organ in enumerate(ORGANS):
            sub = df[df.organ == organ]
            means = []; sems = []
            for n in niche_order:
                vals = sub[sub.niche == n][col].dropna().values
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            xpos = x + (oi - 1) * bar_w
            ax.bar(xpos, means, bar_w, yerr=sems,
                   color=ORGAN_COLORS[organ], alpha=0.85,
                   edgecolor="black", linewidth=0.6, label=organ,
                   error_kw=dict(linewidth=0.8, capsize=2))
        ax.set_xticks(x)
        ax.set_xticklabels(niche_order, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(("Pure-IgG enrichment per niche × organ" if mi == 0
                      else "Pure-IgA enrichment per niche × organ"),
                     fontsize=11)
        if mi == 0:
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    # Right bottom: cross-organ IgG fraction in GALT-interior with stats
    ax = fig.add_subplot(gs[1, 2])
    sub_gi = df[df.niche == "GALT-interior"]
    organ_vals = {organ: sub_gi[sub_gi.organ == organ]["frac_pure_igg"].dropna().values
                  for organ in ORGANS}
    pos = {"APE": 0, "ILE": 1, "COL": 2}
    for organ, vals in organ_vals.items():
        x = pos[organ] + np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x, vals, color=ORGAN_COLORS[organ], s=50, alpha=0.85,
                   edgecolor="black", linewidth=0.6)
        ax.hlines(vals.mean(), pos[organ] - 0.25, pos[organ] + 0.25,
                  color=ORGAN_COLORS[organ], linewidth=2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(ORGANS, fontsize=10)
    ax.set_ylabel("Pure-IgG fraction in GALT-interior\n(per atomic unit)", fontsize=10)
    ax.set_title("APE: Pure-IgG enrichment in GALT-interior", fontsize=11)

    pairs = [("APE", "ILE"), ("APE", "COL"), ("ILE", "COL")]
    ymax = max(np.concatenate([v for v in organ_vals.values()])) + 0.05
    bracket_h = ymax + 0.02
    for ai, (a_, b_) in enumerate(pairs):
        if len(organ_vals[a_]) > 2 and len(organ_vals[b_]) > 2:
            u, p = stats.mannwhitneyu(organ_vals[a_], organ_vals[b_],
                                      alternative="two-sided")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            x1, x2 = pos[a_], pos[b_]
            yy = bracket_h + ai * 0.04
            ax.plot([x1, x1, x2, x2], [yy - 0.005, yy, yy, yy - 0.005],
                    color="black", linewidth=0.8)
            ax.text((x1 + x2) / 2, yy + 0.005,
                    f"{stars} p={p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(None, bracket_h + 0.18)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Spatial validation of organ-specific isotype bias — Pure-IgG vs Pure-IgA plasma cells per niche",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(FIG_DIR / "fig_C_spatial_isotype_distribution.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_C_spatial_isotype_distribution.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_C_spatial_isotype_distribution_hires.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    print(f"\nWrote figures/fig_C_spatial_isotype_distribution.{{png,pdf}}")

    print("\n=== Per-organ pure-IgG fraction in GALT-interior (atomic units) ===")
    for organ, vals in organ_vals.items():
        if len(vals) > 0:
            print(f"  {organ}: mean={vals.mean():.3f}  SD={vals.std():.3f}  n={len(vals)}")
    print("\n=== Pairwise Mann-Whitney (GALT-interior pure-IgG fraction) ===")
    for a_, b_ in pairs:
        if len(organ_vals[a_]) > 2 and len(organ_vals[b_]) > 2:
            u, p = stats.mannwhitneyu(organ_vals[a_], organ_vals[b_],
                                      alternative="two-sided")
            print(f"  {a_} vs {b_}: p={p:.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
