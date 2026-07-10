"""Deeper IBD biology: recovered T/NK+B signatures, composition shifts, and
Hallmark pathway scoring per (ct, state_group).

Three panels in a single figure:
  A  Relaxed per-CT C1 Inflamed-vs-Healthy signatures scored on C1 controls
     vs C2 IBD (extends _ibd_signature_transfer.py to T/NK and B).
  B  Composition shifts — patient-level proportion of each CT within each
     state_group, with Mann-Whitney tests vs C1 Healthy (BH-FDR).
  C  Hallmark pathway heatmap — mean score per (CT, state_group) for key
     IBD-relevant gene sets, with significance dots (vs C1 Healthy, BH-FDR).

Outputs:
  outputs/biology/ibd_deep_signatures_relaxed.csv
  outputs/biology/ibd_deep_signature_violin_stats.csv
  outputs/biology/ibd_deep_composition_proportions.csv
  outputs/biology/ibd_deep_composition_stats.csv
  outputs/biology/ibd_deep_pathway_scores.csv
  outputs/biology/ibd_deep_pathway_stats.csv
  outputs/figures/main/fig_ibd_biology.{pdf,png}
"""
from __future__ import annotations
import os, sys
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)


# ── Parameters ──────────────────────────────────────────────────────────
COMMON_CTS = ["Epithelial", "Mesenchymal", "Myeloid", "T/NK", "B", "Plasma"]
MIN_CELLS_DE = 50       # per group for DE
MIN_CELLS_SCORE = 20    # per (ct, state_group) for scoring / stats
N_SIG = 30              # top-N signature length

STATE_ORDER = ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]
STATE_PALETTE = {
    "C1 · Healthy":          "#009E73",
    "C1 · UC Non-inflamed":  "#56B4E9",
    "C1 · UC Inflamed":      "#D55E00",
    "C2 · IBD":              "#E69F00",
}


# Hallmark gene sets (MSigDB v7.5+, IBD-relevant subset).  Only symbols are
# used; genes not present in the panel are silently dropped by scanpy.
HALLMARK = {
    "TNF_NFKB": [
        "TNF", "TNFAIP3", "NFKB1", "NFKB2", "NFKBIA", "RELB", "REL", "RELA",
        "CXCL1", "CXCL2", "CXCL3", "CXCL10", "CCL2", "CCL20",
        "ICAM1", "VCAM1", "SOD2", "PTGS2", "IL1B", "IL6", "TNFRSF1A",
        "TNFRSF1B", "TNFAIP2", "BIRC3", "NFKBIE", "JUNB", "FOS", "EGR1",
    ],
    "IFN_GAMMA": [
        "IFNG", "IFNGR1", "IFNGR2", "STAT1", "STAT2", "IRF1", "IRF7", "IRF8",
        "GBP1", "GBP4", "GBP5", "CXCL9", "CXCL10", "CXCL11", "CCL5", "IDO1",
        "HLA-A", "HLA-B", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
        "PSMB8", "PSMB9", "TAP1", "TAP2", "PSMA3", "OAS1", "ISG15",
    ],
    "INFLAMMATORY": [
        "IL1B", "IL6", "IL18", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
        "CCL5", "CCR7", "CD14", "PTGS2", "NOS2", "LYZ", "S100A8", "S100A9",
        "FCGR3A", "FCGR2A", "IFITM1", "IFITM3", "ICAM1", "SELL", "ITGAX",
        "TLR2", "TLR4", "MYD88", "NLRP3", "CASP1",
    ],
    "COMPLEMENT": [
        "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "C5", "C6", "C7",
        "CFB", "CFH", "CFI", "CFD", "C1R", "C1S", "C2", "SERPING1", "CD55",
        "CD46", "CD59", "ITGAM", "ITGB2", "VSIG4",
    ],
    "IL6_JAK_STAT3": [
        "IL6", "IL6R", "IL6ST", "JAK1", "JAK2", "STAT3", "SOCS3", "SOCS1",
        "CRP", "SAA1", "SAA2", "HP", "A2M", "TNFRSF12A", "OSMR", "TIMP1",
        "PIM1", "BCL3", "MAP3K8", "IFI30",
    ],
    "HYPOXIA": [
        "HIF1A", "VEGFA", "SLC2A1", "SLC2A3", "LDHA", "PGK1", "PFKFB3",
        "ENO1", "ENO2", "ALDOA", "HK2", "PDK1", "BNIP3", "BNIP3L",
        "DDIT4", "ERO1A", "EGLN3",
    ],
    "EMT": [
        "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2",
        "COL1A1", "COL1A2", "COL3A1", "MMP2", "MMP9", "TGFB1", "TGFB2",
        "ACTA2", "S100A4", "SPARC", "POSTN", "FAP",
    ],
    "APOPTOSIS": [
        "BAX", "BAK1", "BCL2", "BCL2L1", "MCL1", "BID", "BAD", "BIK",
        "CASP3", "CASP7", "CASP8", "CASP9", "CASP10", "FAS", "FASLG",
        "TNFRSF10A", "TNFRSF10B", "XIAP", "BIRC3", "DIABLO",
    ],
}


# ── Helpers ─────────────────────────────────────────────────────────────
def _sig_stars(p):
    if p is None or not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."


def _state_label(row):
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


def _bh(pvals):
    pv = np.asarray(pvals, dtype=float)
    ok = np.isfinite(pv)
    out = np.full_like(pv, np.nan)
    if ok.any():
        out[ok] = multipletests(pv[ok], method="fdr_bh")[1]
    return out


# ── Load ────────────────────────────────────────────────────────────────
print("[deep] loading ...")
a = ad.read_h5ad(config.ADATA_META)
a = a[:, ~a.var_names.isin(config.EXCLUDED_GENES)].copy()
a.obs["state_group"] = a.obs.apply(_state_label, axis=1)
print(f"     state_group: {a.obs['state_group'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════
#  Panel A — Relaxed per-CT IBD signatures (rescues T/NK and B)
# ═══════════════════════════════════════════════════════════════════════
print("\n[deep/A] deriving relaxed C1 Inflamed-vs-Healthy signatures ...")
c1 = a[a.obs["dataset"].isin(["cosmx_c1", "xenium_c1"])].copy()
c1.obs["state"] = c1.obs["disease_state"].astype(str)
c1 = c1[c1.obs["state"].isin(["Healthy", "Inflamed"])].copy()

sig_genes = {}     # ct -> list
sig_rows = []
for ct in COMMON_CTS:
    sub = c1[c1.obs["ct_major"].astype(str) == ct].copy()
    n_h = int((sub.obs["state"] == "Healthy").sum())
    n_i = int((sub.obs["state"] == "Inflamed").sum())
    if n_h < MIN_CELLS_DE or n_i < MIN_CELLS_DE:
        print(f"     skip {ct}: Healthy={n_h}, Inflamed={n_i}")
        continue
    sub.obs["state"] = pd.Categorical(sub.obs["state"], ["Healthy", "Inflamed"])
    sc.tl.rank_genes_groups(sub, groupby="state", method="wilcoxon",
                             groups=["Inflamed"], reference="Healthy",
                             use_raw=False, n_genes=sub.shape[1])
    rg = sub.uns["rank_genes_groups"]
    de = pd.DataFrame(dict(
        gene=list(rg["names"]["Inflamed"]),
        logfc=list(rg["logfoldchanges"]["Inflamed"]),
        padj=list(rg["pvals_adj"]["Inflamed"]),
        score=list(rg["scores"]["Inflamed"]),
    ))
    # Strict filter first
    strict = de[(de["padj"] < 0.05) & (de["logfc"] > 0)]
    relaxed_used = False
    if len(strict) >= 10:
        chosen = strict.sort_values("score", ascending=False).head(N_SIG)
        mode = "strict(padj<0.05 & lfc>0)"
    else:
        # Relax: up-regulated (logfc > 0), top-N by score regardless of padj
        chosen = de[de["logfc"] > 0].sort_values("score", ascending=False).head(N_SIG)
        relaxed_used = True
        mode = "relaxed(lfc>0, top-N by score)"
    sig_genes[ct] = chosen["gene"].tolist()
    for _, r in chosen.iterrows():
        sig_rows.append(dict(ct=ct, mode=mode, relaxed=relaxed_used,
                              gene=r["gene"], logfc=r["logfc"],
                              padj=r["padj"], score=r["score"]))
    print(f"     {ct}: n_sig={len(sig_genes[ct])}  mode={mode}  "
          f"top: {sig_genes[ct][:5]}")

pd.DataFrame(sig_rows).to_csv(
    os.path.join(config.BIOLOGY_DIR, "ibd_deep_signatures_relaxed.csv"),
    index=False)

# Score signatures on full AnnData
print("\n[deep/A] scoring signatures on all cells ...")
for ct, genes in sig_genes.items():
    if len(genes) >= 5:
        sc.tl.score_genes(a, gene_list=genes,
                          score_name=f"sigR_{ct.replace('/', '_')}",
                          random_state=0)

# Collect per-cell scores into one long frame (for plot + stats)
def _score_col(ct): return f"sigR_{ct.replace('/', '_')}"
violin_stats_rows = []
score_cache = {}   # (ct, sg) -> np.array
for ct in sig_genes:
    col = _score_col(ct)
    if col not in a.obs.columns: continue
    sub_mask = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
    obs = a.obs.loc[sub_mask, [col, "state_group"]]
    for sg in STATE_ORDER:
        v = obs.loc[obs["state_group"] == sg, col].values
        if len(v) >= MIN_CELLS_SCORE:
            score_cache[(ct, sg)] = v
    for g1, g2 in combinations(STATE_ORDER, 2):
        x = score_cache.get((ct, g1)); y = score_cache.get((ct, g2))
        if x is None or y is None: continue
        try:
            u, p = mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            u, p = np.nan, np.nan
        violin_stats_rows.append(dict(
            ct=ct, group1=g1, group2=g2, n1=len(x), n2=len(y),
            median1=float(np.median(x)), median2=float(np.median(y)),
            U=float(u) if np.isfinite(u) else np.nan,
            p_raw=float(p) if np.isfinite(p) else np.nan))
vdf = pd.DataFrame(violin_stats_rows)
if len(vdf):
    vdf["p_bh"] = _bh(vdf["p_raw"].values)
    vdf["sig"] = vdf["p_bh"].apply(_sig_stars)
vdf.to_csv(os.path.join(config.BIOLOGY_DIR,
                          "ibd_deep_signature_violin_stats.csv"), index=False)
vmap = {(r.ct, r.group1, r.group2): r.p_bh
        for r in vdf.itertuples() if pd.notna(r.p_bh)}


# ═══════════════════════════════════════════════════════════════════════
#  Panel B — Composition shifts (patient-level proportions)
# ═══════════════════════════════════════════════════════════════════════
print("\n[deep/B] composition — (patient × platform)-level CT proportions ...")
# Use (patient_id, cohort_platform) as replicate: Dulai P1 profiled on both
# CosMx and Xenium contributes 2 replicates, etc. Gives enough n for Wilcoxon
# while staying biological (one patient-sample per replicate).
comp = a.obs[["patient_id", "cohort_platform",
              "state_group", "ct_major"]].dropna().copy()
comp = comp[comp["ct_major"].astype(str).isin(COMMON_CTS)]
comp = comp[comp["state_group"].isin(STATE_ORDER)]
comp["replicate"] = (comp["patient_id"].astype(str)
                      + "|" + comp["cohort_platform"].astype(str))

counts = (comp.groupby(["replicate", "state_group", "ct_major"],
                         observed=True)
                .size().reset_index(name="n"))
# Zero-fill CTs absent in a replicate
present = counts[["replicate", "state_group"]].drop_duplicates()
grid = (present.assign(key=1)
        .merge(pd.DataFrame({"ct_major": COMMON_CTS, "key": 1}), on="key")
        .drop(columns="key"))
counts = grid.merge(counts, on=["replicate", "state_group", "ct_major"],
                     how="left").fillna({"n": 0})
totals = (counts.groupby(["replicate", "state_group"], observed=True)["n"]
                .sum().reset_index(name="total"))
counts = counts.merge(totals, on=["replicate", "state_group"])
counts = counts[counts["total"] >= 50].copy()   # drop tiny replicates
counts["frac"] = counts["n"] / counts["total"]
print(f"     replicates per state: "
      f"{counts[['replicate','state_group']].drop_duplicates()['state_group'].value_counts().to_dict()}")
counts.to_csv(os.path.join(config.BIOLOGY_DIR,
                             "ibd_deep_composition_proportions.csv"),
              index=False)

# stats: for each CT, compare C1 Healthy vs {C1 Non-inflamed, C1 Inflamed, C2 IBD}
comp_stats_rows = []
for ct in COMMON_CTS:
    ref = counts[(counts["ct_major"] == ct)
                   & (counts["state_group"] == "C1 · Healthy")]["frac"].values
    for sg in ["C1 · UC Non-inflamed", "C1 · UC Inflamed", "C2 · IBD"]:
        other = counts[(counts["ct_major"] == ct)
                         & (counts["state_group"] == sg)]["frac"].values
        if len(ref) < 2 or len(other) < 2:
            continue
        try:
            u, p = mannwhitneyu(ref, other, alternative="two-sided")
        except ValueError:
            u, p = np.nan, np.nan
        comp_stats_rows.append(dict(
            ct=ct, ref="C1 · Healthy", other=sg,
            n_ref=len(ref), n_other=len(other),
            median_ref=float(np.median(ref)),
            median_other=float(np.median(other)),
            delta=float(np.median(other) - np.median(ref)),
            U=float(u) if np.isfinite(u) else np.nan,
            p_raw=float(p) if np.isfinite(p) else np.nan))
cdf = pd.DataFrame(comp_stats_rows)
if len(cdf):
    cdf["p_bh"] = _bh(cdf["p_raw"].values)
    cdf["sig"] = cdf["p_bh"].apply(_sig_stars)
cdf.to_csv(os.path.join(config.BIOLOGY_DIR,
                          "ibd_deep_composition_stats.csv"), index=False)
print(cdf.round(4).to_string(index=False) if len(cdf) else "     no comp stats")


# ═══════════════════════════════════════════════════════════════════════
#  Panel C — Hallmark pathway scoring
# ═══════════════════════════════════════════════════════════════════════
print("\n[deep/C] Hallmark pathway scoring ...")
var_names = set(a.var_names)
path_used = {}     # name -> subset of genes present
for name, genes in HALLMARK.items():
    present = [g for g in genes if g in var_names]
    if len(present) >= 5:
        path_used[name] = present
        sc.tl.score_genes(a, gene_list=present,
                          score_name=f"path_{name}", random_state=0)
        print(f"     {name}: {len(present)}/{len(genes)} genes in panel")
    else:
        print(f"     skip {name}: only {len(present)} genes in panel")

# Mean score per (ct, state_group) + Mann-Whitney vs C1 Healthy
path_score_rows = []
path_stats_rows = []
for pname in path_used:
    col = f"path_{pname}"
    for ct in COMMON_CTS:
        sub_mask = ((a.obs["ct_major"].astype(str) == ct)
                     & a.obs["state_group"].notna())
        obs = a.obs.loc[sub_mask, [col, "state_group"]]
        group_vals = {}
        for sg in STATE_ORDER:
            v = obs.loc[obs["state_group"] == sg, col].values
            if len(v) >= MIN_CELLS_SCORE:
                group_vals[sg] = v
                path_score_rows.append(dict(
                    pathway=pname, ct=ct, state_group=sg,
                    n=len(v), mean=float(np.mean(v)),
                    median=float(np.median(v))))
        ref = group_vals.get("C1 · Healthy")
        if ref is None: continue
        for sg, other in group_vals.items():
            if sg == "C1 · Healthy": continue
            try:
                u, p = mannwhitneyu(ref, other, alternative="two-sided")
            except ValueError:
                u, p = np.nan, np.nan
            path_stats_rows.append(dict(
                pathway=pname, ct=ct, ref="C1 · Healthy", other=sg,
                n_ref=len(ref), n_other=len(other),
                mean_ref=float(np.mean(ref)),
                mean_other=float(np.mean(other)),
                delta=float(np.mean(other) - np.mean(ref)),
                U=float(u) if np.isfinite(u) else np.nan,
                p_raw=float(p) if np.isfinite(p) else np.nan))
psdf = pd.DataFrame(path_score_rows)
psdf.to_csv(os.path.join(config.BIOLOGY_DIR,
                           "ibd_deep_pathway_scores.csv"), index=False)
pstdf = pd.DataFrame(path_stats_rows)
if len(pstdf):
    pstdf["p_bh"] = _bh(pstdf["p_raw"].values)
    pstdf["sig"] = pstdf["p_bh"].apply(_sig_stars)
pstdf.to_csv(os.path.join(config.BIOLOGY_DIR,
                            "ibd_deep_pathway_stats.csv"), index=False)

# Cross-cohort pathway direction concordance: does the same pathway × CT
# go up in both C1-Inflamed and C2-IBD (both referenced to C1-Healthy)?
from scipy.stats import pearsonr, spearmanr
if len(pstdf):
    conc = (pstdf.pivot_table(index=["pathway", "ct"],
                                columns="other", values="delta")
                 .reset_index())
    if "C1 · UC Inflamed" in conc.columns and "C2 · IBD" in conc.columns:
        conc = conc.rename(columns={"C1 · UC Inflamed": "delta_C1_inflamed",
                                      "C2 · IBD":         "delta_C2_IBD"})
        conc = conc.dropna(subset=["delta_C1_inflamed", "delta_C2_IBD"])
        conc.to_csv(os.path.join(config.BIOLOGY_DIR,
                                  "ibd_deep_pathway_concordance.csv"),
                    index=False)
        if len(conc) >= 3:
            r_pear, p_pear = pearsonr(conc["delta_C1_inflamed"],
                                         conc["delta_C2_IBD"])
            r_spear, p_spear = spearmanr(conc["delta_C1_inflamed"],
                                           conc["delta_C2_IBD"])
            concordance_r = (r_pear, p_pear, r_spear, p_spear)
            # same-sign rate (directional concordance)
            same_sign = ((conc["delta_C1_inflamed"] > 0)
                          == (conc["delta_C2_IBD"] > 0)).mean()
            print(f"     pathway-direction concordance: "
                  f"Pearson r={r_pear:.2f} p={p_pear:.1e}, "
                  f"Spearman ρ={r_spear:.2f}, same-sign={same_sign:.0%}, "
                  f"n={len(conc)}")
        else:
            concordance_r = None
    else:
        conc = None; concordance_r = None
else:
    conc = None; concordance_r = None


# ═══════════════════════════════════════════════════════════════════════
#  Render the composite figure
# ═══════════════════════════════════════════════════════════════════════
print("\n[deep] rendering figure ...")
n_ct_sig = len(sig_genes)
fig = plt.figure(figsize=(13, 10))
gs = gridspec.GridSpec(3, max(n_ct_sig, 6), figure=fig,
                       height_ratios=[1.0, 1.0, 1.3], hspace=0.75, wspace=0.55)

# ---- Panel A: violins per CT (row 0) ----------------------------------
for i, ct in enumerate(sig_genes):
    ax = fig.add_subplot(gs[0, i])
    col = _score_col(ct)
    if col not in a.obs.columns:
        ax.axis("off"); continue
    sub_mask = ((a.obs["ct_major"].astype(str) == ct)
                 & a.obs["state_group"].notna())
    obs = a.obs.loc[sub_mask, [col, "state_group"]]
    data, labels, colors = [], [], []
    for sg in STATE_ORDER:
        v = obs.loc[obs["state_group"] == sg, col].values
        if len(v) < MIN_CELLS_SCORE: continue
        data.append(v); labels.append(sg); colors.append(STATE_PALETTE[sg])
    if not data:
        ax.axis("off"); continue
    parts = ax.violinplot(data, positions=range(len(data)),
                           showmeans=False, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor("black")
        pc.set_linewidth(0.4)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" · ", "\n") for l in labels], fontsize=5)
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.set_title(f"{ct} — n_sig={len(sig_genes[ct])}", fontsize=7)
    if i == 0: ax.set_ylabel("C1-IBD signature", fontsize=6)
    ax.tick_params(labelsize=5)

    # significance brackets: Healthy vs Inflamed, Healthy vs C2·IBD
    idx = {lab: j for j, lab in enumerate(labels)}
    allv = np.concatenate(data)
    y_hi = np.percentile(allv, 99.5)
    y_lo = np.percentile(allv, 0.5)
    span = (y_hi - y_lo) or 1.0
    bh = span * 0.06
    y = y_hi + span * 0.05
    for g1, g2 in [("C1 · Healthy", "C1 · UC Inflamed"),
                   ("C1 · Healthy", "C2 · IBD")]:
        if g1 not in idx or g2 not in idx: continue
        p = vmap.get((ct, g1, g2))
        if p is None: continue
        x1, x2 = idx[g1], idx[g2]
        ax.plot([x1, x1, x2, x2], [y, y + bh, y + bh, y],
                lw=0.5, color="black")
        ax.text((x1 + x2) / 2, y + bh, _sig_stars(p),
                ha="center", va="bottom", fontsize=6)
        y += span * 0.14
    ax.set_ylim(y_lo - span * 0.05, y + span * 0.05)

# ---- Panel B: composition stacked bars (row 1) ------------------------
ax_b = fig.add_subplot(gs[1, :3])
# mean fraction across patients, per state_group × ct
comp_mean = (counts.groupby(["state_group", "ct_major"],
                                observed=True)["frac"].mean()
                  .unstack(fill_value=0))
comp_mean = comp_mean.reindex(STATE_ORDER).reindex(columns=COMMON_CTS, fill_value=0)
bottom = np.zeros(len(comp_mean))
for ct in COMMON_CTS:
    vals = comp_mean[ct].values
    ax_b.bar(range(len(comp_mean)), vals, bottom=bottom,
             color=config.CT_PALETTE.get(ct, "#888"),
             edgecolor="white", lw=0.4, label=ct)
    bottom += vals
ax_b.set_xticks(range(len(comp_mean)))
ax_b.set_xticklabels([l.replace(" · ", "\n") for l in comp_mean.index],
                     fontsize=6)
ax_b.set_ylabel("Mean replicate-level fraction", fontsize=7)
ax_b.set_ylim(0, 1)
ax_b.set_title("Composition by state group\n"
                "(patient×platform replicates; n=2/2/4/8 — qualitative only;\n"
                " C2 is Ileum vs C1 Rectum — tissue confounds C1↔C2 shifts)",
                fontsize=7)
ax_b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=5, frameon=False, title="Cell type", title_fontsize=6)

# Panel D — Cross-cohort pathway direction concordance scatter
ax_bs = fig.add_subplot(gs[1, 3:])
if conc is not None and concordance_r is not None:
    # color by pathway
    path_colors = dict(zip(list(HALLMARK.keys()),
                            plt.cm.tab10(np.linspace(0, 1, len(HALLMARK)))))
    x_vals = conc["delta_C1_inflamed"].values
    y_vals = conc["delta_C2_IBD"].values
    for _, row in conc.iterrows():
        ax_bs.scatter(row["delta_C1_inflamed"], row["delta_C2_IBD"],
                      color=path_colors[row["pathway"]],
                      edgecolor="black", lw=0.3, s=24,
                      label=row["pathway"])
    # unity line + zero axes
    lim = max(np.abs(np.r_[x_vals, y_vals])) * 1.1
    ax_bs.plot([-lim, lim], [-lim, lim], color="grey", lw=0.5, ls="--")
    ax_bs.axhline(0, color="black", lw=0.3)
    ax_bs.axvline(0, color="black", lw=0.3)
    ax_bs.set_xlim(-lim, lim); ax_bs.set_ylim(-lim, lim)
    ax_bs.set_xlabel("Δ pathway score\n(C1 Inflamed − C1 Healthy)", fontsize=6)
    ax_bs.set_ylabel("Δ pathway score\n(C2 IBD − C1 Healthy)", fontsize=6)
    r_pear, p_pear, r_spear, p_spear = concordance_r
    same_sign = ((conc["delta_C1_inflamed"] > 0)
                  == (conc["delta_C2_IBD"] > 0)).mean()
    ax_bs.set_title(f"Cross-cohort pathway×CT concordance\n"
                      f"Pearson r={r_pear:.2f} (p={p_pear:.1e})  ·  "
                      f"same-sign {same_sign:.0%}  ·  n={len(conc)}",
                      fontsize=7)
    # dedup legend by pathway
    handles, labels_ = ax_bs.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_):
        if l not in seen: seen[l] = h
    ax_bs.legend(list(seen.values()), list(seen.keys()),
                 fontsize=5, frameon=False, loc="center left",
                 bbox_to_anchor=(1.02, 0.5), title="Hallmark pathway",
                 title_fontsize=6)
    ax_bs.tick_params(labelsize=5)
else:
    ax_bs.axis("off")

# ---- Panel C: Hallmark pathway heatmap (row 2) ------------------------
ax_c = fig.add_subplot(gs[2, :])
if len(pstdf):
    # rows = (pathway, ct), cols = state group (non-Healthy)
    pstdf["row"] = pstdf["pathway"] + " | " + pstdf["ct"]
    heat_delta = pstdf.pivot_table(index="row", columns="other", values="delta")
    heat_sig = pstdf.pivot_table(index="row", columns="other", values="sig",
                                   aggfunc="first")
    col_order = [c for c in ["C1 · UC Non-inflamed", "C1 · UC Inflamed",
                               "C2 · IBD"] if c in heat_delta.columns]
    # row order: group by pathway (HALLMARK order), then CT (COMMON_CTS order)
    row_order = [f"{p} | {ct}" for p in HALLMARK
                   for ct in COMMON_CTS
                   if f"{p} | {ct}" in heat_delta.index]
    heat_delta = heat_delta.loc[row_order, col_order]
    heat_sig = heat_sig.loc[row_order, col_order]
    vmax = np.nanmax(np.abs(heat_delta.values)) or 1e-3
    im = ax_c.imshow(heat_delta.values, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, aspect="auto")
    ax_c.set_xticks(range(len(col_order)))
    ax_c.set_xticklabels(col_order, fontsize=6)
    ax_c.set_yticks(range(len(row_order)))
    ax_c.set_yticklabels(row_order, fontsize=5)
    for i in range(heat_delta.shape[0]):
        for j in range(heat_delta.shape[1]):
            s = heat_sig.values[i, j]
            if isinstance(s, str) and s and s != "n.s.":
                ax_c.text(j, i, s, ha="center", va="center",
                           color="black", fontsize=5)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Δ mean pathway score vs C1 Healthy", fontsize=6)
    ax_c.set_title("Hallmark pathway activation vs C1 Healthy (Mann-Whitney, BH-FDR)",
                    fontsize=8)
else:
    ax_c.axis("off")

fig.suptitle("Deep IBD biology: recovered signatures, composition shifts, "
              "and Hallmark pathway activation", fontsize=9)
out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_ibd_biology.pdf")
out_png = os.path.join(config.FIG_MAIN_DIR, "fig_ibd_biology.png")
fig.savefig(out_pdf, format="pdf", dpi=1200, bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"     -> {out_pdf}\n     -> {out_png}")


# ── Headlines ───────────────────────────────────────────────────────────
print("\n[deep] HEADLINES:")
if len(vdf):
    sig_hits = vdf[(vdf["p_bh"] < 0.05)
                   & (vdf["group1"] == "C1 · Healthy")
                   & (vdf["group2"].isin(["C1 · UC Inflamed", "C2 · IBD"]))]
    print(f"  Panel A: {len(sig_hits)} CT×comparison with FDR<0.05 "
          f"for signature elevation vs Healthy")
if len(cdf):
    print(f"  Panel B: composition shifts FDR<0.05: "
          f"{int((cdf['p_bh'] < 0.05).sum())}/{len(cdf)}")
if len(pstdf):
    print(f"  Panel C: pathway shifts FDR<0.05: "
          f"{int((pstdf['p_bh'] < 0.05).sum())}/{len(pstdf)}")
if conc is not None and concordance_r is not None:
    r_pear, p_pear, r_spear, p_spear = concordance_r
    same_sign = ((conc["delta_C1_inflamed"] > 0)
                  == (conc["delta_C2_IBD"] > 0)).mean()
    print(f"  Panel D: pathway×CT direction concordance — "
          f"Pearson r={r_pear:.2f} (p={p_pear:.1e}), "
          f"same-sign {same_sign:.0%}, n={len(conc)}")
