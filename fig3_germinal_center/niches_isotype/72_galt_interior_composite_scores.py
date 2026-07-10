"""GALT-interior composite cell-state scores across APE, ILE, COL.

For each IGH+ cell (any IGHA1/IGHG1/IGHG2/IGHM/IGHD > 0) in GALT-interior
niches per atomic unit (SAHA_Sample × batch_id), computes 8 z-scored composite
scores WITHIN each atomic unit, then reports per-unit means.

Niche definition mirrored from scripts/69_plasma_egress_colocalization.py:
  chemokine = CCL19 + CCL21 + CXCL13, smoothed over k=20 NN
  tertile bins: GALT-edge at q0.6, GALT-interior at q0.9

Scores:
  1. Egress:              z(CXCR4) - z(CXCR5) - z(GPR183)
  2. Plasma-fate:         z(IRF4) + z(XBP1) + z(CD38) + z(CD27)
  3. B-retention:         z(MS4A1) + z(CD19) + z(CXCR5) + z(GPR183)
  4. Proliferation:       z(MKI67)
  5. Dark zone:           z(CXCR4) + z(MKI67)
  6. Light zone:          z(CD83) + z(CD86) + z(CD80)
  7. Systemic homing:     z(CXCR3) + z(SELL)
  8. Gut homing:          z(CCR10)

Statistical testing: pairwise Mann-Whitney (two-sided) + BH correction (24 tests).
Cohen's d per pair per score.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import itertools

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEAN_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned")
BASE_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI")
OUT_DIR = BASE_DIR / "ig_validation"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORGANS = ["APE", "COL", "ILE"]
ISOTYPES = ["IGHA1", "IGHG1", "IGHG2", "IGHM", "IGHD"]
NICHE_GENES = ["CCL19", "CCL21", "CXCL13"]
K_NICHE = 20
LAYER = "normalized"
ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
MIN_UNIT_CELLS = 2000       # atomic unit minimum total cells
MIN_IGH_INTERIOR = 20       # minimum IGH+ GALT-interior cells per unit

SCORE_DEFS = {
    "Egress":          (["CXCR4"],               ["CXCR5", "GPR183"]),
    "Plasma-fate":     (["IRF4", "XBP1", "CD38", "CD27"], []),
    "B-retention":     (["MS4A1", "CD19", "CXCR5", "GPR183"], []),
    "Proliferation":   (["MKI67"],               []),
    "Dark-zone":       (["CXCR4", "MKI67"],       []),
    "Light-zone":      (["CD83", "CD86", "CD80"], []),
    "Systemic-homing": (["CXCR3", "SELL"],        []),
    "Gut-homing":      (["CCR10"],                []),
}
SCORE_NAMES = list(SCORE_DEFS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gvec(adata, gene):
    """Extract dense expression vector for gene from normalized layer."""
    if gene not in adata.var_names:
        return None
    j = adata.var_names.get_loc(gene)
    M = adata.layers[LAYER] if LAYER in adata.layers else adata.X
    col = M[:, j]
    return np.asarray(col.toarray() if sp.issparse(col) else col).ravel()


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def smooth_score(coords, score, k):
    if len(coords) < k + 1:
        return score.copy()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = nn.kneighbors(coords)
    return score[idx[:, 1:]].mean(axis=1)


def cohens_d(a, b):
    """Pooled-SD Cohen's d."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else 0.0


def p_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Per-unit processing
# ---------------------------------------------------------------------------
def process_unit(adata_unit):
    """
    Returns dict mapping score_name -> mean_score within GALT-interior IGH+ cells,
    plus n_igh_interior and n_total.
    Returns None if unit should be skipped.
    """
    coords = np.column_stack([
        adata_unit.obs["x_centroid"].to_numpy(float),
        adata_unit.obs["y_centroid"].to_numpy(float),
    ])
    valid = np.isfinite(coords).all(axis=1)
    if valid.sum() < 500:
        return None
    coords = coords[valid]
    av = adata_unit[valid]

    # --- Chemokine niche ---
    chemo = np.zeros(av.n_obs)
    for g in NICHE_GENES:
        e = gvec(av, g)
        if e is not None:
            chemo += e
    chemo_smooth = smooth_score(coords, chemo, K_NICHE)
    qs = np.quantile(chemo_smooth, [0.6, 0.9])
    niches = np.full(av.n_obs, "LP-distal", dtype=object)
    niches[chemo_smooth >= qs[0]] = "GALT-edge"
    niches[chemo_smooth >= qs[1]] = "GALT-interior"

    # --- IGH+ mask ---
    igh_mask = np.zeros(av.n_obs, dtype=bool)
    for iso in ISOTYPES:
        e = gvec(av, iso)
        if e is not None:
            igh_mask |= (e > 0)

    # GALT-interior IGH+ mask
    gi_igh = (niches == "GALT-interior") & igh_mask
    if gi_igh.sum() < MIN_IGH_INTERIOR:
        return None

    # --- z-score all genes WITHIN this unit (all cells) ---
    gene_z = {}
    all_genes = set()
    for pos, neg in SCORE_DEFS.values():
        all_genes.update(pos)
        all_genes.update(neg)
    for gene in all_genes:
        e = gvec(av, gene)
        gene_z[gene] = zscore(e) if e is not None else np.zeros(av.n_obs)

    # --- Compute composite scores and take mean within GALT-interior IGH+ ---
    result = {"n_igh_interior": int(gi_igh.sum()), "n_total": int(av.n_obs)}
    for score_name, (pos_genes, neg_genes) in SCORE_DEFS.items():
        score = np.zeros(av.n_obs)
        for g in pos_genes:
            score += gene_z[g]
        for g in neg_genes:
            score -= gene_z[g]
        result[score_name] = float(score[gi_igh].mean())

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_rows = []

    for organ in ORGANS:
        print(f"\n=== Loading {organ} ===")
        adata = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
        print(f"  Total cells: {adata.n_obs:,}")

        unit_count = 0
        skip_count = 0
        for sample in sorted(adata.obs["SAHA_Sample"].unique()):
            for batch in sorted(adata.obs["batch_id"].unique()):
                m = (adata.obs["SAHA_Sample"] == sample) & (adata.obs["batch_id"] == batch)
                if m.sum() < MIN_UNIT_CELLS:
                    continue
                print(f"  {sample} x {batch}: n={m.sum():,}", end=" ... ", flush=True)
                res = process_unit(adata[m].copy())
                if res is None:
                    skip_count += 1
                    print("skipped (< 20 IGH+ GALT-interior)")
                    continue
                row = {
                    "organ": organ,
                    "sample": sample,
                    "batch": batch,
                    "unit_id": f"{sample}__{batch}",
                    "n_igh_interior": res["n_igh_interior"],
                    "n_total": res["n_total"],
                }
                for sn in SCORE_NAMES:
                    row[sn] = res[sn]
                all_rows.append(row)
                unit_count += 1
                print(f"ok (n_igh_interior={res['n_igh_interior']})")

        print(f"  => {unit_count} units passed, {skip_count} skipped")

    df_wide = pd.DataFrame(all_rows)

    # --- Long format for CSV ---
    id_cols = ["organ", "sample", "batch", "unit_id", "n_igh_interior", "n_total"]
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=SCORE_NAMES,
        var_name="score_name",
        value_name="value",
    )
    df_long.to_csv(OUT_DIR / "galt_interior_composite_per_unit.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'galt_interior_composite_per_unit.csv'}")

    # --- Stats: pairwise Mann-Whitney + BH ---
    pairs = [("APE", "COL"), ("APE", "ILE"), ("ILE", "COL")]
    organ_data = {
        organ: {sn: df_wide.loc[df_wide.organ == organ, sn].dropna().values
                for sn in SCORE_NAMES}
        for organ in ORGANS
    }

    stat_rows = []
    raw_pvals = []
    test_keys = []

    for sn in SCORE_NAMES:
        for a, b in pairs:
            va = organ_data[a][sn]
            vb = organ_data[b][sn]
            if len(va) >= 3 and len(vb) >= 3:
                _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            else:
                p = np.nan
            d = cohens_d(va, vb)
            raw_pvals.append(p)
            test_keys.append((sn, a, b))
            stat_rows.append({
                "score": sn,
                "organ_A": a,
                "organ_B": b,
                "n_A": len(va),
                "n_B": len(vb),
                "mean_A": float(np.mean(va)) if len(va) > 0 else np.nan,
                "mean_B": float(np.mean(vb)) if len(vb) > 0 else np.nan,
                "raw_p": p,
                "cohens_d": d,
            })

    # BH correction over all 24 tests
    valid_idx = [i for i, p in enumerate(raw_pvals) if not np.isnan(p)]
    bh_pvals = np.full(len(raw_pvals), np.nan)
    if valid_idx:
        valid_raw = [raw_pvals[i] for i in valid_idx]
        _, bh_corrected, _, _ = multipletests(valid_raw, method="fdr_bh")
        for ii, vi in enumerate(valid_idx):
            bh_pvals[vi] = bh_corrected[ii]

    for i, row in enumerate(stat_rows):
        row["bh_p"] = bh_pvals[i]
        row["stars_bh"] = p_stars(bh_pvals[i]) if not np.isnan(bh_pvals[i]) else "n/a"

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(OUT_DIR / "galt_interior_composite_stats.csv", index=False)
    print(f"Wrote {OUT_DIR / 'galt_interior_composite_stats.csv'}")

    # --- Print summary ---
    print("\n=== Unit counts per organ ===")
    for organ in ORGANS:
        n = (df_wide.organ == organ).sum()
        print(f"  {organ}: {n} units")

    print("\n=== BH-significant scores (p_bh < 0.05) ===")
    sig = df_stats[df_stats.bh_p < 0.05][["score", "organ_A", "organ_B", "mean_A", "mean_B", "raw_p", "bh_p", "cohens_d", "stars_bh"]]
    if len(sig) == 0:
        print("  None")
    else:
        print(sig.to_string(index=False))

    print("\n=== Per-score mean per organ ===")
    means = df_wide.groupby("organ")[SCORE_NAMES].mean()
    print(means.round(3).to_string())

    # -----------------------------------------------------------------------
    # Figure 1: 2×4 violin grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    # Build BH lookup: (score, A, B) -> bh_p
    bh_lookup = {(r.score, r.organ_A, r.organ_B): r.bh_p for _, r in df_stats.iterrows()}

    for idx, sn in enumerate(SCORE_NAMES):
        ax = axes[idx]
        organ_vals = [organ_data[o][sn] for o in ORGANS]

        # Violin
        parts = ax.violinplot(
            [v for v in organ_vals],
            positions=range(len(ORGANS)),
            showmedians=False, showextrema=False,
        )
        for pi, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(list(ORGAN_COLORS.values())[pi])
            pc.set_alpha(0.45)
            pc.set_edgecolor("none")

        # Strip + mean + 95% CI
        for pi, (organ, vals) in enumerate(zip(ORGANS, organ_vals)):
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(pi + jitter, vals, s=22, color=ORGAN_COLORS[organ],
                       alpha=0.75, zorder=3, linewidths=0)
            mean_v = np.mean(vals)
            ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.errorbar(pi, mean_v, yerr=ci, fmt="o", color="black",
                        markersize=6, capsize=4, linewidth=1.8, zorder=5)

        # Significance brackets
        y_max = max(np.max(v) if len(v) > 0 else 0 for v in organ_vals)
        y_range = y_max - min(np.min(v) if len(v) > 0 else 0 for v in organ_vals)
        step = y_range * 0.12 if y_range > 0 else 0.1

        bracket_pairs = [(0, 1, "APE", "COL"), (0, 2, "APE", "ILE"), (1, 2, "ILE", "COL")]
        current_top = y_max + step * 0.5
        for xi, xj, oa, ob in bracket_pairs:
            bh_p = bh_lookup.get((sn, oa, ob), np.nan)
            if np.isnan(bh_p):
                continue
            star = p_stars(bh_p)
            bh_top = current_top + step
            ax.plot([xi, xi, xj, xj],
                    [current_top, bh_top, bh_top, current_top],
                    lw=0.9, color="black")
            ax.text((xi + xj) / 2, bh_top + step * 0.1, star,
                    ha="center", va="bottom", fontsize=8,
                    color="black" if star != "ns" else "#888888")
            current_top = bh_top + step * 0.5

        ax.set_xticks(range(len(ORGANS)))
        ax.set_xticklabels(ORGANS, fontsize=9)
        ax.set_title(sn, fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean composite score\n(GALT-interior, IGH+)", fontsize=7)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "GALT-interior composite cell-state scores — IGH+ cells (SAHA CosMx)\n"
        "Per-atomic-unit means; brackets = BH-adjusted Mann-Whitney",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_galt_interior_composite_scores.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_galt_interior_composite_scores.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR / 'fig_galt_interior_composite_scores.png'}")

    # -----------------------------------------------------------------------
    # Figure 2: 8×3 heatmap
    # -----------------------------------------------------------------------
    hmap_data = np.zeros((len(SCORE_NAMES), len(ORGANS)))
    for ci, organ in enumerate(ORGANS):
        for ri, sn in enumerate(SCORE_NAMES):
            vals = organ_data[organ][sn]
            hmap_data[ri, ci] = np.mean(vals) if len(vals) > 0 else np.nan

    fig2, ax2 = plt.subplots(figsize=(5.5, 7))
    vext = np.nanmax(np.abs(hmap_data))
    im = ax2.imshow(hmap_data, cmap="RdBu_r", vmin=-vext, vmax=vext, aspect="auto")

    # Annotate cells
    for ri in range(len(SCORE_NAMES)):
        for ci in range(len(ORGANS)):
            val = hmap_data[ri, ci]
            if np.isnan(val):
                txt = "NaN"
                col = "gray"
            else:
                txt = f"{val:.3f}"
                col = "white" if abs(val) > vext * 0.6 else "black"

            # Add stars for BH-significant pairs involving this organ
            # Mark score rows that have any significant pair
            sig_mark = ""
            for a, b in pairs:
                bh_p = bh_lookup.get((SCORE_NAMES[ri], a, b), np.nan)
                if not np.isnan(bh_p) and bh_p < 0.05:
                    sig_mark = "*"
                    break
            ax2.text(ci, ri, txt, ha="center", va="center",
                     fontsize=9, color=col, fontweight="bold")

    ax2.set_xticks(range(len(ORGANS)))
    ax2.set_xticklabels(ORGANS, fontsize=11, fontweight="bold")
    ax2.set_yticks(range(len(SCORE_NAMES)))
    ax2.set_yticklabels(SCORE_NAMES, fontsize=10)
    ax2.set_title(
        "Mean composite score per organ\n(GALT-interior IGH+ cells)",
        fontsize=11,
    )

    # Add BH-significance row markers
    for ri, sn in enumerate(SCORE_NAMES):
        any_sig = any(
            not np.isnan(bh_lookup.get((sn, a, b), np.nan)) and bh_lookup.get((sn, a, b)) < 0.05
            for a, b in pairs
        )
        if any_sig:
            ax2.text(len(ORGANS) - 0.35, ri, "BH*", ha="left", va="center",
                     fontsize=8, color="#CC3333", fontweight="bold")

    cbar = fig2.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label("Mean composite score", fontsize=9)

    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig_galt_interior_heatmap.png", dpi=300, bbox_inches="tight")
    fig2.savefig(FIG_DIR / "fig_galt_interior_heatmap.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {FIG_DIR / 'fig_galt_interior_heatmap.png'}")


if __name__ == "__main__":
    main()
