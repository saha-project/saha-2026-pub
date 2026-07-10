"""Plasma-lineage-restricted egress colocalization per atomic unit.

Rebuilds the spatial egress story so it's cell-type-specific (not composition
topology). Within IGH+ cells (any of IGHA1/IGHG1/IGHG2/IGHM/IGHD > 0) of
each atomic unit (SAHA_Sample × batch_id), computes:

- Egress score per cell = z(CXCR4) − z(CXCR5) − z(GPR183)
  where CXCR4+ = DZ / plasmablast-migration, CXCR5/GPR183 = B-cell retention
- Gut-homing score = z(CCR10)
- Systemic score = z(CXCR3) + z(SELL)

Then per atomic unit, aggregates these scores per niche bin
(LP-distal / GALT-edge / GALT-interior from existing chemokine niche
definition).

This makes the migration claim about IDENTIFIED plasma-lineage cells, not
tissue cytology.

Outputs (ig_validation/):
- plasma_egress_per_unit_niche.csv — long format per (unit, niche, score)
- plasma_egress_per_organ.csv — organ-aggregated means + SEM
- figures/fig_D_plasma_egress_spatial.{png,pdf}

Data inputs:
- CosMx cleaned h5ads
- ig_validation/spatial_atomic_unit_summary.csv — atomic unit list
"""

from pathlib import Path
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

warnings.filterwarnings("ignore", category=FutureWarning)

CLEAN_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Integration/Integration/processed/cleaned")
OUT_DIR = Path(__file__).resolve().parent.parent / "ig_validation"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

ORGANS = ["APE", "COL", "ILE"]
ISOTYPES = ["IGHA1", "IGHG1", "IGHG2", "IGHM", "IGHD"]
NICHE_GENES = ["CCL19", "CCL21", "CXCL13"]
K_NICHE = 20
LAYER = "normalized"
ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}
NICHE_ORDER = ["LP-distal", "GALT-edge", "GALT-interior"]


def gvec(a_view, gene):
    if gene not in a_view.var_names:
        return None
    j = a_view.var_names.get_loc(gene)
    M = a_view.layers[LAYER] if LAYER in a_view.layers else a_view.X
    col = M[:, j]
    return np.asarray(col.toarray() if sp.issparse(col) else col).ravel()


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def smooth_score(coords, score, k):
    if len(coords) < k + 1:
        return score
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, idx = nn.kneighbors(coords)
    return score[idx[:, 1:]].mean(axis=1)


def process_atomic_unit(a_unit):
    """Return dict with per-niche scores restricted to IGH+ cells."""
    coords = np.column_stack([
        a_unit.obs["x_centroid"].to_numpy(float),
        a_unit.obs["y_centroid"].to_numpy(float),
    ])
    valid = np.isfinite(coords).all(axis=1)
    if valid.sum() < 500:
        return None
    coords = coords[valid]
    a_v = a_unit[valid]

    chemo = np.zeros(a_v.n_obs)
    for g in NICHE_GENES:
        e = gvec(a_v, g)
        if e is not None:
            chemo += e
    chemo_smooth = smooth_score(coords, chemo, K_NICHE)
    qs = np.quantile(chemo_smooth, [0.6, 0.9])
    niches = np.full(a_v.n_obs, "LP-distal", dtype=object)
    niches[chemo_smooth >= qs[0]] = "GALT-edge"
    niches[chemo_smooth >= qs[1]] = "GALT-interior"

    igh_mask = np.zeros(a_v.n_obs, dtype=bool)
    for iso in ISOTYPES:
        e = gvec(a_v, iso)
        if e is not None:
            igh_mask |= (e > 0)
    if igh_mask.sum() < 30:
        return None

    # Compute scores
    cxcr4 = gvec(a_v, "CXCR4")
    cxcr5 = gvec(a_v, "CXCR5")
    gpr183 = gvec(a_v, "GPR183")
    ccr10 = gvec(a_v, "CCR10")
    cxcr3 = gvec(a_v, "CXCR3")
    sell = gvec(a_v, "SELL")

    egress = np.zeros(a_v.n_obs)
    if cxcr4 is not None:
        egress += zscore(cxcr4)
    if cxcr5 is not None:
        egress -= zscore(cxcr5)
    if gpr183 is not None:
        egress -= zscore(gpr183)

    gut_homing = zscore(ccr10) if ccr10 is not None else np.zeros(a_v.n_obs)
    systemic = np.zeros(a_v.n_obs)
    if cxcr3 is not None:
        systemic += zscore(cxcr3)
    if sell is not None:
        systemic += zscore(sell)

    rows = []
    for niche in NICHE_ORDER:
        m = (niches == niche) & igh_mask
        n = int(m.sum())
        if n < 10:
            rows.append(dict(
                niche=niche, n_igh_in_niche=n,
                mean_egress=np.nan, mean_gut=np.nan,
                mean_systemic=np.nan,
                mean_cxcr4=np.nan, mean_cxcr5=np.nan,
            ))
            continue
        rows.append(dict(
            niche=niche, n_igh_in_niche=n,
            mean_egress=float(egress[m].mean()),
            mean_gut=float(gut_homing[m].mean()),
            mean_systemic=float(systemic[m].mean()),
            mean_cxcr4=float(cxcr4[m].mean()) if cxcr4 is not None else np.nan,
            mean_cxcr5=float(cxcr5[m].mean()) if cxcr5 is not None else np.nan,
        ))
    return rows


def main():
    all_rows = []
    for organ in ORGANS:
        print(f"\n=== {organ} ===")
        a = ad.read_h5ad(CLEAN_DIR / f"{organ}_RNA_cleaned.h5ad")
        for sample in sorted(a.obs["SAHA_Sample"].unique()):
            for batch in sorted(a.obs["batch_id"].unique()):
                m = (a.obs["SAHA_Sample"] == sample) & (a.obs["batch_id"] == batch)
                if m.sum() < 2000:
                    continue
                a_unit = a[m].copy()
                print(f"  {sample} × {batch}: n={a_unit.n_obs:,}", end=" ", flush=True)
                rows = process_atomic_unit(a_unit)
                if rows is None:
                    print("skipped")
                    continue
                for r in rows:
                    r["organ"] = organ
                    r["sample"] = sample
                    r["batch"] = batch
                    r["unit_id"] = f"{sample}__{batch}"
                all_rows.extend(rows)
                print("ok")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "plasma_egress_per_unit_niche.csv", index=False)

    # Organ-level aggregation
    agg = df.groupby(["organ", "niche"]).agg(
        mean_egress=("mean_egress", "mean"),
        sem_egress=("mean_egress", lambda x: np.std(x.dropna(), ddof=1) / np.sqrt(x.dropna().size)),
        n_units=("mean_egress", lambda x: x.dropna().size),
        mean_gut=("mean_gut", "mean"),
        sem_gut=("mean_gut", lambda x: np.std(x.dropna(), ddof=1) / np.sqrt(x.dropna().size)),
        mean_systemic=("mean_systemic", "mean"),
        sem_systemic=("mean_systemic", lambda x: np.std(x.dropna(), ddof=1) / np.sqrt(x.dropna().size)),
    ).reset_index()
    agg.to_csv(OUT_DIR / "plasma_egress_per_organ.csv", index=False)

    print("\n=== PLASMA-LINEAGE-RESTRICTED EGRESS SCORE (CXCR4 − CXCR5 − GPR183) ===")
    pivot = df.pivot_table(index="niche", columns="organ",
                           values="mean_egress", aggfunc="mean")
    pivot = pivot.reindex(NICHE_ORDER)
    print(pivot.round(3))

    print("\n=== GUT-HOMING SCORE (CCR10) in IGH+ cells ===")
    pivot_g = df.pivot_table(index="niche", columns="organ",
                             values="mean_gut", aggfunc="mean")
    print(pivot_g.reindex(NICHE_ORDER).round(3))

    print("\n=== Gradient (GALT-interior − LP-distal) per organ ===")
    for organ in ORGANS:
        sub = df[(df.organ == organ)]
        lp = sub[sub.niche == "LP-distal"]["mean_egress"].dropna()
        gi = sub[sub.niche == "GALT-interior"]["mean_egress"].dropna()
        print(f"  {organ}: egress gradient = {gi.mean() - lp.mean():.3f} (n_units={len(gi)})")
        gl = sub[sub.niche == "LP-distal"]["mean_gut"].dropna()
        gg = sub[sub.niche == "GALT-interior"]["mean_gut"].dropna()
        print(f"       gut-homing gradient = {gg.mean() - gl.mean():.3f}")

    # Cross-organ statistics on egress gradient
    org_grads = {}
    for organ in ORGANS:
        sub = df[df.organ == organ]
        pivot_per_unit = sub.pivot_table(index="unit_id", columns="niche",
                                         values="mean_egress")
        if "GALT-interior" in pivot_per_unit.columns and "LP-distal" in pivot_per_unit.columns:
            grads = (pivot_per_unit["GALT-interior"] - pivot_per_unit["LP-distal"]).dropna()
            org_grads[organ] = grads.to_numpy()

    print("\n=== Egress gradient per atomic unit (IGH+ restricted) ===")
    for o, vals in org_grads.items():
        print(f"  {o}: mean={vals.mean():.3f} SD={vals.std():.3f} n={len(vals)}")
    if "APE" in org_grads and "COL" in org_grads:
        u, p = stats.mannwhitneyu(org_grads["APE"], org_grads["COL"],
                                  alternative="greater")
        print(f"  APE > COL: p={p:.4f}")
    if "APE" in org_grads and "ILE" in org_grads:
        u, p = stats.mannwhitneyu(org_grads["APE"], org_grads["ILE"],
                                  alternative="greater")
        print(f"  APE > ILE: p={p:.4f}")
    if "ILE" in org_grads and "COL" in org_grads:
        u, p = stats.mannwhitneyu(org_grads["ILE"], org_grads["COL"],
                                  alternative="greater")
        print(f"  ILE > COL: p={p:.4f}")

    # ---- Figure: three-part Panel D -----
    snp_h = pd.read_csv(OUT_DIR / "snpatho_plasma_homing_receptors.csv")
    snp_h = snp_h[snp_h["homing_axis"].str.startswith("single:")].copy()
    snp_h["gene"] = snp_h["genes"]

    REP_UNITS = {
        "APE": ("APE_A9_4", "SJ01"),
        "COL": ("COL_A1_4", "EC02"),
        "ILE": ("ILE_A1_3", "SJ01"),
    }

    def load_rep_unit(organ):
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
        chemo = np.zeros(a_u.n_obs)
        for g in NICHE_GENES:
            e = gvec(a_u, g)
            if e is not None:
                chemo += e
        chemo_smooth = smooth_score(coords, chemo, K_NICHE)
        qs = np.quantile(chemo_smooth, [0.6, 0.9])
        niches = np.full(a_u.n_obs, "LP-distal", dtype=object)
        niches[chemo_smooth >= qs[0]] = "GALT-edge"
        niches[chemo_smooth >= qs[1]] = "GALT-interior"
        igh_mask = np.zeros(a_u.n_obs, dtype=bool)
        for iso in ISOTYPES:
            e = gvec(a_u, iso)
            if e is not None:
                igh_mask |= (e > 0)
        cxcr4 = gvec(a_u, "CXCR4")
        cxcr5 = gvec(a_u, "CXCR5")
        gpr183 = gvec(a_u, "GPR183")
        egress = np.zeros(a_u.n_obs)
        if cxcr4 is not None: egress += zscore(cxcr4)
        if cxcr5 is not None: egress -= zscore(cxcr5)
        if gpr183 is not None: egress -= zscore(gpr183)
        return dict(coords=coords, niches=niches, igh=igh_mask, egress=egress,
                    sample=sample, batch=batch)

    rep_data = {organ: load_rep_unit(organ) for organ in ORGANS}

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.32, wspace=0.25)

    # Top row: 3 spatial maps
    for ci, organ in enumerate(ORGANS):
        ax = fig.add_subplot(gs[0, ci])
        d = rep_data[organ]
        coords = d["coords"]; igh = d["igh"]; eg = d["egress"]
        ax.scatter(coords[~igh, 0], coords[~igh, 1],
                   s=0.15, c="#eeeeee", alpha=0.5, rasterized=True)
        if igh.sum() > 0:
            vmax = np.percentile(np.abs(eg[igh]), 95)
            vmax = max(vmax, 1.0)
            sc = ax.scatter(coords[igh, 0], coords[igh, 1],
                            c=eg[igh], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            s=0.9, alpha=0.85, edgecolor='none', rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("Egress score\n(CXCR4−CXCR5−GPR183)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{organ}  ({d['sample']} × {d['batch']})\n"
                     f"IGH+ cells, n={int(igh.sum()):,}",
                     fontsize=10, color=ORGAN_COLORS[organ])

        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        bar_len = 500
        bx = xmin + 0.05 * (xmax - xmin)
        by = ymin + 0.08 * (ymax - ymin)
        ax.plot([bx, bx + bar_len], [by, by], color="black", linewidth=2.5)
        ax.text(bx + bar_len / 2, by - 0.02 * (ymax - ymin),
                "500 µm", ha="center", va="top", fontsize=7)

    axL = fig.add_subplot(gs[1, 0:2])
    axR = fig.add_subplot(gs[1, 2])

    # Left: MEI (egress score) per niche, IGH+ restricted, per organ
    for organ in ORGANS:
        sub = agg[agg.organ == organ].set_index("niche").reindex(NICHE_ORDER)
        x = np.arange(len(NICHE_ORDER))
        y = sub["mean_egress"].values
        yerr = sub["sem_egress"].values
        axL.plot(x, y, "o-", color=ORGAN_COLORS[organ],
                 linewidth=2, markersize=9, label=organ)
        axL.fill_between(x, y - yerr, y + yerr, color=ORGAN_COLORS[organ], alpha=0.2)
    axL.set_xticks(range(len(NICHE_ORDER)))
    axL.set_xticklabels(NICHE_ORDER, fontsize=10)
    axL.set_ylabel("Egress score (CXCR4 − CXCR5 − GPR183)\nwithin IGH+ plasma-lineage cells",
                   fontsize=10)
    axL.set_title("Spatial egress gradient — plasma-lineage restricted",
                  fontsize=11)
    axL.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    axL.legend(fontsize=9, loc="upper left", framealpha=0.9)
    axL.grid(axis="y", alpha=0.25)

    txt = ("IGH+ mask: cells expressing any of\nIGHA1/IGHG1/IGHG2/IGHM/IGHD\n"
           "Aggregated across atomic units\n(SAHA_Sample × batch_id)")
    axL.annotate(txt, xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=8, ha="left", va="bottom", style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8dc",
                           edgecolor="gray", alpha=0.85))

    # Pairwise Mann-Whitney at GALT-interior — per-unit egress score
    gi_vals = {}
    for organ in ORGANS:
        sub = df[(df.organ == organ) & (df.niche == "GALT-interior")]
        gi_vals[organ] = sub["mean_egress"].dropna().values

    pairs = [("APE", "COL"), ("APE", "ILE"), ("ILE", "COL")]
    pair_tests = {}
    for a_, b_ in pairs:
        if len(gi_vals[a_]) > 2 and len(gi_vals[b_]) > 2:
            u, p = stats.mannwhitneyu(gi_vals[a_], gi_vals[b_],
                                      alternative="two-sided")
            pair_tests[(a_, b_)] = p

    def p_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    ymax = max(agg["mean_egress"].max() + 0.1, 0.4)
    bracket_heights = [ymax + 0.05, ymax + 0.12, ymax + 0.19]
    for (a_, b_), bh in zip(pairs, bracket_heights):
        if (a_, b_) not in pair_tests:
            continue
        p = pair_tests[(a_, b_)]
        y_a = np.mean(gi_vals[a_]) if len(gi_vals[a_]) > 0 else 0
        y_b = np.mean(gi_vals[b_]) if len(gi_vals[b_]) > 0 else 0
        axL.plot([2.0, 2.0, 2.0, 2.0], [bh - 0.01, bh, bh, bh - 0.01],
                 color="black", linewidth=0.8)
        axL.plot([1.95, 2.05], [bh, bh], color="black", linewidth=0.8)
        axL.text(2.15, bh, f"{a_} vs {b_}: {p_stars(p)} (p={p:.3f})",
                 ha="left", va="center", fontsize=8)

    for organ in ORGANS:
        if organ in org_grads:
            grads = org_grads[organ]
            axL.annotate(f"Δ={grads.mean():.2f}", xy=(0.05, 0.97 - 0.05 * ORGANS.index(organ)),
                         xycoords="axes fraction", fontsize=8,
                         color=ORGAN_COLORS[organ],
                         ha="left", va="top", fontweight="bold")

    axL.set_ylim(None, bracket_heights[-1] + 0.08)
    axL.set_xlim(-0.3, 3.2)

    # Right: snPATHO homing receptors in plasma cells — APE vs ILE
    receptors = ["ITGA4", "ITGB7", "CCR9", "CCR10", "CXCR3"]
    snp_sub = snp_h[snp_h["gene"].isin(receptors)].sort_values(
        by="gene", key=lambda s: s.map({g: i for i, g in enumerate(receptors)})
    )
    ape_vals = snp_sub["ape_mean"].tolist()
    ile_vals = snp_sub["ile_mean"].tolist()
    ds = snp_sub["d"].tolist()
    x = np.arange(len(receptors))
    w = 0.38
    b1 = axR.bar(x - w/2, ape_vals, w, color=ORGAN_COLORS["APE"],
                 alpha=0.85, edgecolor="black", linewidth=0.8, label="APE")
    b2 = axR.bar(x + w/2, ile_vals, w, color=ORGAN_COLORS["ILE"],
                 alpha=0.85, edgecolor="black", linewidth=0.8, label="ILE")
    for i, d in enumerate(ds):
        if abs(d) > 0.2:
            y = max(ape_vals[i], ile_vals[i]) + 0.05
            axR.text(i, y, f"d={d:.2f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold",
                     color="#228B22" if d > 0 else "#8B0000")
    axR.set_xticks(x)
    axR.set_xticklabels(receptors, fontsize=10)
    axR.set_ylabel("Mean normalized expression\n(snPATHO, plasma cells only)", fontsize=10)
    axR.set_title("Gut-dispersal receptor bias — APE plasma cells",
                  fontsize=11)
    axR.set_ylim(0, max(max(ape_vals), max(ile_vals)) * 1.25)
    axR.legend(fontsize=9, loc="upper right", framealpha=0.9)
    axR.grid(axis="y", alpha=0.25)

    axR.annotate("APE plasma cells are equipped for\ngut-wide dispersal (α4β7, CCR9, CCR10)",
                 xy=(0.02, 0.97), xycoords="axes fraction",
                 fontsize=8, ha="left", va="top", style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8dc",
                           edgecolor="gray", alpha=0.85))

    fig.suptitle("APE plasma cells combine spatial egress commitment with gut-dispersal molecular signature",
                 fontsize=12, y=1.02, fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fig_D_plasma_egress_spatial.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_D_plasma_egress_spatial.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_D_plasma_egress_spatial_hires.png", dpi=600, bbox_inches="tight")
    print(f"\nWrote {FIG_DIR / 'fig_D_plasma_egress_spatial.png'}")


if __name__ == "__main__":
    main()
