"""IBD signature derived from C1 (Healthy vs Inflamed) → tested on C2.

Demonstrates that integration enables cross-cohort disease inference:
  1. C1 cohort has clinical states (Healthy / Non-inflamed / Inflamed UC)
  2. Per-cell-type, find genes upregulated in C1 Inflamed vs Healthy
  3. Score that "C1 IBD signature" on each cell in C2 (all IBD)
  4. Show the signature is elevated in C2 IBD cells (matched cell types)
     vs C1 Healthy cells (negative control)

Outputs:
  outputs/biology/c1_ibd_signature_per_ct.csv
  outputs/biology/ibd_signature_transfer.csv
  outputs/figures/main/fig_ibd_transfer.{pdf,png}
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _sig_stars(p):
    if not np.isfinite(p): return "n.s."
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)

# Use top-N most up-regulated genes as the signature, per CT
N_SIG = 30
MIN_CELLS = 50

print("[ibd] loading ...")
a = ad.read_h5ad(config.ADATA_META)
a = a[:, ~a.var_names.isin(config.EXCLUDED_GENES)].copy()

# Restrict to ct_majors present in both C1 and C2 with enough cells
common_cts = ["Epithelial", "Mesenchymal", "Myeloid", "T/NK", "B", "Plasma"]

# ── (1) Derive C1 IBD signature per CT (Inflamed vs Healthy) ───────────
print("\n[ibd] deriving C1 IBD signature (Inflamed vs Healthy) per CT ...")
c1 = a[a.obs["dataset"].isin(["cosmx_c1", "xenium_c1"])].copy()
c1.obs["state"] = c1.obs["disease_state"].astype(str)
c1 = c1[c1.obs["state"].isin(["Healthy", "Inflamed"])].copy()
print(f"     C1 cells with Healthy/Inflamed labels: {c1.shape[0]:,}")
print(f"     by state: {c1.obs['state'].value_counts().to_dict()}")

sigs = {}     # ct -> list of top-N up genes in Inflamed
sig_rows = []
for ct in common_cts:
    sub = c1[c1.obs["ct_major"].astype(str) == ct].copy()
    n_h = (sub.obs["state"] == "Healthy").sum()
    n_i = (sub.obs["state"] == "Inflamed").sum()
    if n_h < MIN_CELLS or n_i < MIN_CELLS:
        print(f"     skip {ct}: Healthy={n_h}, Inflamed={n_i}")
        continue
    sub.obs["state"] = pd.Categorical(sub.obs["state"], ["Healthy", "Inflamed"])
    sc.tl.rank_genes_groups(sub, groupby="state", method="wilcoxon",
                            groups=["Inflamed"], reference="Healthy",
                            use_raw=False, n_genes=sub.shape[1])
    rg = sub.uns["rank_genes_groups"]
    names = list(rg["names"]["Inflamed"])
    logfc = list(rg["logfoldchanges"]["Inflamed"])
    pvals_adj = list(rg["pvals_adj"]["Inflamed"])
    scores = list(rg["scores"]["Inflamed"])
    # Filter: significant + up-regulated
    df_de = pd.DataFrame(dict(gene=names, logfc=logfc, padj=pvals_adj, score=scores))
    df_de = df_de[(df_de["padj"] < 0.05) & (df_de["logfc"] > 0)]
    df_de = df_de.sort_values("score", ascending=False).head(N_SIG)
    sigs[ct] = df_de["gene"].tolist()
    for _, r in df_de.iterrows():
        sig_rows.append(dict(ct=ct, gene=r["gene"], logfc=r["logfc"],
                              padj=r["padj"], score=r["score"]))
    print(f"     {ct}: {len(sigs[ct])} signature genes "
          f"(Healthy n={n_h}, Inflamed n={n_i}); top: {sigs[ct][:5]}")

pd.DataFrame(sig_rows).to_csv(
    os.path.join(config.BIOLOGY_DIR, "c1_ibd_signature_per_ct.csv"), index=False)

# ── (2) Score the signature on every cell in C1+C2 ──────────────────────
print("\n[ibd] scoring signature on all cells ...")
score_rows = []
for ct, gene_list in sigs.items():
    if len(gene_list) < 5:
        continue
    sc.tl.score_genes(a, gene_list=gene_list,
                      score_name=f"C1_IBD_sig_{ct.replace('/', '_')}",
                      random_state=0)

# Build long-format per-cell scores per (ct, cohort_platform, disease_state)
def _state_label(row):
    """Return harmonized disease/state label for plotting."""
    cp = str(row["cohort_platform"])
    ds = str(row["disease_state"])
    if cp.startswith("C2"):
        return "C2 · IBD"
    if ds == "Healthy":
        return "C1 · Healthy"
    if ds == "Non-inflamed":
        return "C1 · UC Non-inflamed"
    if ds == "Inflamed":
        return "C1 · UC Inflamed"
    return None

a.obs["state_group"] = a.obs.apply(_state_label, axis=1)
print(f"     state group counts: {a.obs['state_group'].value_counts().to_dict()}")

# For plot: for each ct, collect mean signature score per (state_group)
plot_rows = []
for ct in sigs:
    col = f"C1_IBD_sig_{ct.replace('/', '_')}"
    if col not in a.obs.columns: continue
    sub = a[(a.obs["ct_major"].astype(str) == ct)
            & a.obs["state_group"].notna()].copy()
    for sg in ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]:
        sg_mask = sub.obs["state_group"] == sg
        if sg_mask.sum() < 20: continue
        plot_rows.append(dict(ct=ct, state_group=sg,
                                n=int(sg_mask.sum()),
                                mean=float(sub.obs.loc[sg_mask, col].mean()),
                                median=float(sub.obs.loc[sg_mask, col].median())))
score_df = pd.DataFrame(plot_rows)
score_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                              "ibd_signature_transfer.csv"), index=False)
print("\n  Mean C1-derived IBD signature per (ct, state_group):")
piv = score_df.pivot(index="ct", columns="state_group", values="mean")
print(piv.round(3).to_string())

# ── (2b) Pairwise Mann-Whitney across state_groups per CT ───────────────
print("\n[ibd] pairwise Mann-Whitney across state_groups per CT ...")
STATE_ORDER = ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]
MIN_N = 20
stat_rows = []
# Cache per-cell score arrays per (ct, state_group) so we don't repeatedly filter
score_arrays = {}
for ct in sigs:
    col = f"C1_IBD_sig_{ct.replace('/', '_')}"
    if col not in a.obs.columns: continue
    sub_mask = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
    sub_obs = a.obs.loc[sub_mask, [col, "state_group"]]
    for sg in STATE_ORDER:
        vals = sub_obs.loc[sub_obs["state_group"] == sg, col].values
        if len(vals) >= MIN_N:
            score_arrays[(ct, sg)] = vals
    for g1, g2 in combinations(STATE_ORDER, 2):
        x = score_arrays.get((ct, g1)); y = score_arrays.get((ct, g2))
        if x is None or y is None: continue
        try:
            u, p = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            u, p = np.nan, np.nan
        stat_rows.append(dict(
            ct=ct, group1=g1, group2=g2,
            n1=len(x), n2=len(y),
            median1=float(np.median(x)), median2=float(np.median(y)),
            delta_median=float(np.median(y) - np.median(x)),
            U=float(u) if np.isfinite(u) else np.nan,
            p_raw=float(p) if np.isfinite(p) else np.nan))
stat_df = pd.DataFrame(stat_rows)
if len(stat_df):
    ok = stat_df["p_raw"].notna()
    stat_df.loc[ok, "p_bh"] = multipletests(stat_df.loc[ok, "p_raw"],
                                             method="fdr_bh")[1]
    stat_df["sig"] = stat_df["p_bh"].apply(_sig_stars)
stat_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                             "ibd_signature_transfer_pairwise_stats.csv"),
               index=False)
print(stat_df.round(4).to_string(index=False))

# Quick lookup: (ct, g1, g2) -> BH p
pmap = {(r.ct, r.group1, r.group2): r.p_bh
        for r in stat_df.itertuples() if pd.notna(r.p_bh)}

# ── (3) Render: violin per CT × state_group ─────────────────────────────
n_ct = len(sigs)
fig, axes = plt.subplots(2, int(np.ceil(n_ct / 2)),
                          figsize=(3.0 * int(np.ceil(n_ct / 2)), 5.5))
axes = axes.ravel()

state_palette = {
    "C1 · Healthy":          "#009E73",
    "C1 · UC Non-inflamed":  "#56B4E9",
    "C1 · UC Inflamed":      "#D55E00",
    "C2 · IBD":              "#E69F00",
}

for i, ct in enumerate(sigs):
    ax = axes[i]
    col = f"C1_IBD_sig_{ct.replace('/', '_')}"
    if col not in a.obs.columns:
        ax.axis("off"); continue
    sub = a[(a.obs["ct_major"].astype(str) == ct)
             & a.obs["state_group"].notna()].copy()
    data = []; labels = []; colors = []
    for sg in ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]:
        m = (sub.obs["state_group"] == sg).values
        if m.sum() < 20: continue
        data.append(sub.obs.loc[m, col].values)
        labels.append(sg)
        colors.append(state_palette[sg])
    parts = ax.violinplot(data, positions=range(len(data)),
                            showmeans=False, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor("black")
        pc.set_linewidth(0.4)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" · ", "\n") for l in labels],
                       fontsize=5, rotation=0)
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.set_title(f"{ct} — n_sig={len(sigs[ct])}", fontsize=7)
    ax.set_ylabel("C1-IBD signature score", fontsize=6)
    ax.tick_params(labelsize=5)

    # Significance brackets for key comparisons:
    #   C1·Healthy vs C1·UC Inflamed  (derivation validation)
    #   C1·Healthy vs C2·IBD          (cross-cohort transfer claim)
    key_pairs = [("C1 · Healthy", "C1 · UC Inflamed"),
                 ("C1 · Healthy", "C2 · IBD")]
    idx = {lab: j for j, lab in enumerate(labels)}
    all_vals = np.concatenate(data) if len(data) else np.array([0.0])
    y_max = np.percentile(all_vals, 99.5)
    y_min = np.percentile(all_vals, 0.5)
    y_span = y_max - y_min if y_max > y_min else 1.0
    bracket_h = y_span * 0.06
    y_cursor = y_max + y_span * 0.05
    for g1, g2 in key_pairs:
        if g1 not in idx or g2 not in idx: continue
        p = pmap.get((ct, g1, g2))
        if p is None: continue
        x1, x2 = idx[g1], idx[g2]
        ax.plot([x1, x1, x2, x2],
                [y_cursor, y_cursor + bracket_h,
                 y_cursor + bracket_h, y_cursor],
                lw=0.5, color="black")
        ax.text((x1 + x2) / 2, y_cursor + bracket_h,
                _sig_stars(p), ha="center", va="bottom",
                fontsize=6, color="black")
        y_cursor += y_span * 0.14
    ax.set_ylim(y_min - y_span * 0.05, y_cursor + y_span * 0.05)

# Hide unused
for j in range(n_ct, len(axes)):
    axes[j].axis("off")

fig.suptitle(f"C1-derived IBD signature (Inflamed vs Healthy, top {N_SIG} up-genes)\n"
             "scored on all integrated cells: C1 controls vs C2 IBD",
             fontsize=8)
fig.tight_layout()
out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_ibd_transfer.pdf")
out_png = os.path.join(config.FIG_MAIN_DIR, "fig_ibd_transfer.png")
fig.savefig(out_pdf, format="pdf", dpi=1200, bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n     -> {os.path.basename(out_pdf)}, .png")

# Headline: how many CTs show C2 IBD score > C1 Healthy score?
print(f"\n[ibd] HEADLINE — C1 IBD signature in C2 vs C1 Healthy:")
piv2 = piv.copy()
if "C2 · IBD" in piv2.columns and "C1 · Healthy" in piv2.columns:
    diff = piv2["C2 · IBD"] - piv2["C1 · Healthy"]
    print(diff.round(3).to_string())
    print(f"\n  CTs where C2 IBD > C1 Healthy: "
           f"{int((diff > 0).sum())}/{int(diff.notna().sum())}")
