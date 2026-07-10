"""Single tight Panel C for the main figure.

3 spatial maps (APE/ILE/COL — Pure-IgG vs Pure-IgA cells)
+ small per-atomic-unit summary (GALT-interior IgG fraction with pairwise stats)
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
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

CLEAN_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned")
OUT_DIR = Path(__file__).resolve().parent.parent / "ig_validation"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

ORGANS = ["APE", "ILE", "COL"]
ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
NICHE_GENES = ["CCL19", "CCL21", "CXCL13"]
K_NICHE = 20
LAYER = "normalized"

REP = {
    "APE": ("APE_A9_4", "SJ01"),
    "ILE": ("ILE_A1_3", "SJ01"),
    "COL": ("COL_A1_4", "EC02"),
}

IGG_COLOR = "#D62728"
IGA_COLOR = "#1F77B4"
BG_COLOR = "#dddddd"


def gvec(a, gene):
    if gene not in a.var_names:
        return None
    j = a.var_names.get_loc(gene)
    M = a.layers[LAYER] if LAYER in a.layers else a.X
    col = M[:, j]
    return np.asarray(col.toarray() if sp.issparse(col) else col).ravel()


def classify_isotype(a):
    igg1 = gvec(a, "IGHG1"); igg2 = gvec(a, "IGHG2"); iga1 = gvec(a, "IGHA1")
    iga_pos = (iga1 > 0) if iga1 is not None else np.zeros(a.n_obs, dtype=bool)
    igg_pos = np.zeros(a.n_obs, dtype=bool)
    if igg1 is not None: igg_pos |= (igg1 > 0)
    if igg2 is not None: igg_pos |= (igg2 > 0)
    return igg_pos, iga_pos


def smooth(coords, score, k):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = nn.kneighbors(coords)
    return score[idx[:, 1:]].mean(axis=1)


def niches(a, coords):
    chemo = np.zeros(a.n_obs)
    for g in NICHE_GENES:
        e = gvec(a, g)
        if e is not None: chemo += e
    cs = smooth(coords, chemo, K_NICHE)
    qs = np.quantile(cs, [0.6, 0.9])
    nb = np.full(a.n_obs, "LP-distal", dtype=object)
    nb[cs >= qs[0]] = "GALT-edge"
    nb[cs >= qs[1]] = "GALT-interior"
    return nb


def load_unit(organ):
    sample, batch = REP[organ]
    a = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
    m = (a.obs["SAHA_Sample"] == sample) & (a.obs["batch_id"] == batch)
    a_u = a[m].copy()
    coords = np.column_stack([a_u.obs["x_centroid"].to_numpy(float),
                              a_u.obs["y_centroid"].to_numpy(float)])
    valid = np.isfinite(coords).all(axis=1)
    a_u = a_u[valid]; coords = coords[valid]
    igg, iga = classify_isotype(a_u)
    return dict(coords=coords, igg=igg, iga=iga, sample=sample, batch=batch,
                niches=niches(a_u, coords))


def per_unit_galt_interior():
    rows = []
    for organ in ORGANS:
        a = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
        for sample in sorted(a.obs["SAHA_Sample"].unique()):
            for batch in sorted(a.obs["batch_id"].unique()):
                m = (a.obs["SAHA_Sample"] == sample) & (a.obs["batch_id"] == batch)
                if m.sum() < 2000: continue
                a_u = a[m].copy()
                coords = np.column_stack([a_u.obs["x_centroid"].to_numpy(float),
                                          a_u.obs["y_centroid"].to_numpy(float)])
                valid = np.isfinite(coords).all(axis=1)
                a_u = a_u[valid]; coords = coords[valid]
                igg, iga = classify_isotype(a_u)
                nb = niches(a_u, coords)
                m_gi = nb == "GALT-interior"
                pure_igg = igg & ~iga
                pure_iga = iga & ~igg
                n_igh = int((pure_igg & m_gi).sum() + (pure_iga & m_gi).sum() +
                            (igg & iga & m_gi).sum())
                if n_igh < 20: continue
                rows.append(dict(organ=organ, unit_id=f"{sample}__{batch}",
                                 frac_pure_igg=float((pure_igg & m_gi).sum() / n_igh)))
    return pd.DataFrame(rows)


def main():
    np.random.seed(42)
    rep = {o: load_unit(o) for o in ORGANS}
    df = per_unit_galt_interior()

    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.85], wspace=0.20)

    for ci, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[0, ci])
        d = rep[organ]
        coords = d["coords"]; igg = d["igg"]; iga = d["iga"]
        pure_igg = igg & ~iga
        pure_iga = iga & ~igg
        both = igg & iga
        neither = ~(igg | iga)

        ax.scatter(coords[neither, 0], coords[neither, 1],
                   s=0.10, c=BG_COLOR, alpha=0.5, rasterized=True)
        ax.scatter(coords[pure_iga, 0], coords[pure_iga, 1],
                   s=0.7, c=IGA_COLOR, alpha=0.85, edgecolor="none",
                   rasterized=True)
        ax.scatter(coords[pure_igg, 0], coords[pure_igg, 1],
                   s=0.7, c=IGG_COLOR, alpha=0.85, edgecolor="none",
                   rasterized=True)
        ax.scatter(coords[both, 0], coords[both, 1],
                   s=0.5, c="#9467BD", alpha=0.4, edgecolor="none",
                   rasterized=True)

        n_igg = int(pure_igg.sum()); n_iga = int(pure_iga.sum())
        ax.set_title(f"{organ}\n{int(n_igg):,} pure-IgG | {int(n_iga):,} pure-IgA",
                     fontsize=11, color=ORGAN_COLORS[organ], pad=4,
                     fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(False)

        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        bx = xmin + 0.05 * (xmax - xmin)
        by = ymin + 0.04 * (ymax - ymin)
        ax.plot([bx, bx + 500], [by, by], color="black", linewidth=2)
        ax.text(bx + 250, by - 0.025 * (ymax - ymin),
                "500 µm", ha="center", va="top", fontsize=8)

    ax_stat = fig.add_subplot(gs[0, 3])
    pos = {"APE": 0, "ILE": 1, "COL": 2}
    organ_vals = {organ: df[df.organ == organ]["frac_pure_igg"].values
                  for organ in ORGANS}
    for organ, vals in organ_vals.items():
        x = pos[organ] + np.random.uniform(-0.13, 0.13, size=len(vals))
        ax_stat.scatter(x, vals, color=ORGAN_COLORS[organ], s=55, alpha=0.85,
                        edgecolor="black", linewidth=0.6, zorder=3)
        ax_stat.hlines(vals.mean(), pos[organ] - 0.27, pos[organ] + 0.27,
                       color=ORGAN_COLORS[organ], linewidth=2.5, zorder=2)
    pairs = [("APE", "ILE"), ("APE", "COL"), ("ILE", "COL")]
    ymax = max(np.concatenate(list(organ_vals.values()))) + 0.04
    for ai, (a_, b_) in enumerate(pairs):
        u, p = stats.mannwhitneyu(organ_vals[a_], organ_vals[b_],
                                  alternative="two-sided")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        x1, x2 = pos[a_], pos[b_]
        yy = ymax + ai * 0.045
        ax_stat.plot([x1, x1, x2, x2], [yy - 0.005, yy, yy, yy - 0.005],
                     color="black", linewidth=0.8)
        ax_stat.text((x1 + x2) / 2, yy + 0.005,
                     f"{stars}  p={p:.4f}",
                     ha="center", va="bottom", fontsize=8.5)
    ax_stat.set_xticks([0, 1, 2])
    ax_stat.set_xticklabels(ORGANS, fontsize=10, fontweight="bold")
    for tl, organ in zip(ax_stat.get_xticklabels(), ORGANS):
        tl.set_color(ORGAN_COLORS[organ])
    ax_stat.set_ylabel("Pure-IgG fraction\n(of IGH+ cells in GALT-interior)",
                       fontsize=10)
    ax_stat.set_title("Per atomic unit", fontsize=11, fontweight="bold")
    ax_stat.set_ylim(0, ymax + 3 * 0.045 + 0.04)
    ax_stat.grid(axis="y", alpha=0.25)

    handles = [
        mpatches.Patch(facecolor=IGG_COLOR, label="Pure-IgG (IGHG1/2+)"),
        mpatches.Patch(facecolor=IGA_COLOR, label="Pure-IgA (IGHA1+)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.40, 1.02), frameon=False)

    fig.savefig(FIG_DIR / "fig_C_panel.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(FIG_DIR / "fig_C_panel.pdf", bbox_inches="tight",
                facecolor="white")
    print(f"Wrote {FIG_DIR / 'fig_C_panel.png'}")


if __name__ == "__main__":
    main()
