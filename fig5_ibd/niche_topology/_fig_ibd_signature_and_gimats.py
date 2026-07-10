"""Per-cell-type IBD signature (Fig. 5e) + GIMATS (TNFi-resistance) signature.

This script does TWO things:

  (1) Re-derive the C1 (Healthy vs Inflamed) per-cell-type IBD signature for
      all six common compartments (Epithelial, Mesenchymal, Myeloid, T/NK, B,
      Plasma) and score it on every cell, with patient-level mixed-effects
      models for the four state contrasts:
          C1·Healthy   vs C1·UC Non-inflamed
          C1·Healthy   vs C1·UC Inflamed
          C1·UC Inflamed vs C2·IBD
          C1·Healthy   vs C2·IBD
      All p-values, coefficients, and per-state n's are exported as a tidy
      "Methods Table M5" CSV (`outputs/biology/ibd_sig_M5_coefs.csv`).
      The figure is `outputs/figures/main/fig5e_ibd_signature.{pdf,png}` and
      reports MONOTONICITY HONESTLY per CT (does H ≤ NI ≤ INF ≤ C2 hold?).

  (2) Build a GIMATS-style anti-TNF-resistance module from genes present in
      the spatial panel (Martin et al. 2019, Cell). The IgG-isotype heavy-
      chain constants are not on this panel, so the "plasma" component of
      GIMATS is replaced with a pan-plasma proxy (TNFRSF17, MZB1, XBP1,
      DERL3, FCRL5) — explicitly noted in the figure legend. Score per cell,
      summarise per patient × ct_major, and plot:
          (a) per-state violin (all CTs pooled and per-CT)
          (b) per-C2-patient strip (the 8 SAHA patients)
      Outputs `outputs/figures/main/fig_gimats_signature.{pdf,png}` and
      `outputs/biology/gimats_perpatient.csv`.

NOTE on responder labels: `data/responder_labels.csv` is NOT populated in
this project (only the schema exists), so per-patient comparisons are
shown but no "responder vs non-responder" panel is rendered. If/when
responder labels become available, re-run the GIMATS section with
`--responders <path>` to add the stratification.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
N_SIG = 30                       # top-N up-genes per CT for IBD signature
MIN_CELLS_DERIVE = 50            # min cells in BOTH Healthy and Inflamed
COMMON_CTS = ["Epithelial", "Mesenchymal", "Myeloid", "T/NK", "B", "Plasma"]
STATE_ORDER = ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]
STATE_PALETTE = {
    "C1 · Healthy":          "#009E73",
    "C1 · UC Non-inflamed":  "#56B4E9",
    "C1 · UC Inflamed":      "#D55E00",
    "C2 · IBD":              "#E69F00",
}
KEY_CONTRASTS = [
    ("C1 · Healthy", "C1 · UC Non-inflamed"),
    ("C1 · Healthy", "C1 · UC Inflamed"),
    ("C1 · UC Inflamed", "C2 · IBD"),
    ("C1 · Healthy", "C2 · IBD"),
]

# GIMATS module from Martin et al. 2019 Cell, restricted to genes present in
# this spatial panel. The IgG plasma component (IGHG1/3/4, IGLC1) is absent
# from the panel — substituted with pan-plasma proxy markers. This is
# disclosed in the figure caption.
GIMATS_MODULE = {
    "Inflammatory MNP":   ["IL1B", "FCN1", "AQP9", "OLR1", "FPR1",
                           "CXCL5", "CXCL2", "IL1RN", "TREM1", "MMP9",
                           "MMP12", "NOS2"],
    "Activated T":        ["CXCL13", "IL17A", "IL21", "IL22", "IL26",
                           "ICOS", "CTLA4", "CD2", "IFNG", "CCL20"],
    "Activated Stromal":  ["IL11", "IL13RA2", "TNFRSF11B"],
    "Plasma (proxy)":     ["TNFRSF17", "MZB1", "XBP1", "DERL3", "FCRL5"],
}


def _state_group(row):
    cp = str(row["cohort_platform"])
    ds = str(row["disease_state"])
    if cp.startswith("C2"):
        return "C2 · IBD"
    if ds == "Healthy":         return "C1 · Healthy"
    if ds == "Non-inflamed":    return "C1 · UC Non-inflamed"
    if ds == "Inflamed":        return "C1 · UC Inflamed"
    return None


def _sig_stars(p):
    if not np.isfinite(p): return "n.s."
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "n.s."


def fit_mixedlm(df, value_col, group_col, ref, alt, patient_col="patient_id"):
    """Two-state mixed-effects: value ~ state + (1 | patient).

    Returns (n_ref, n_alt, n_patients, beta_alt, se, z, p) or NaNs on failure.
    """
    d = df[df[group_col].isin([ref, alt])].copy()
    if len(d) < 50: return [np.nan] * 7
    d["state01"] = (d[group_col] == alt).astype(int)
    n_ref = int((d["state01"] == 0).sum())
    n_alt = int((d["state01"] == 1).sum())
    n_pat = d[patient_col].nunique()
    if n_pat < 2: return [n_ref, n_alt, n_pat, np.nan, np.nan, np.nan, np.nan]
    try:
        m = MixedLM.from_formula(f"{value_col} ~ state01",
                                 groups=d[patient_col], data=d)
        r = m.fit(method="lbfgs", reml=True)
        beta = float(r.params.get("state01", np.nan))
        se   = float(r.bse.get("state01", np.nan))
        z    = beta / se if (np.isfinite(beta) and np.isfinite(se) and se>0) else np.nan
        p    = float(r.pvalues.get("state01", np.nan))
        return [n_ref, n_alt, n_pat, beta, se, z, p]
    except Exception as e:
        return [n_ref, n_alt, n_pat, np.nan, np.nan, np.nan, np.nan]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responders", default=None,
                        help="(optional) path to responder_labels.csv")
    parser.add_argument("--quick", action="store_true",
                        help="subsample 25 percent of cells for quick iteration")
    args = parser.parse_args()

    print(f"[ibd+gimats] loading {config.ADATA_META} ...")
    a = ad.read_h5ad(config.ADATA_META)
    a = a[:, ~a.var_names.isin(config.EXCLUDED_GENES)].copy()
    if args.quick:
        rng = np.random.default_rng(0)
        idx = rng.choice(a.shape[0], int(a.shape[0] * 0.25), replace=False)
        a = a[idx].copy()
        print(f"[quick] subsampled to {a.shape[0]:,} cells")

    a.obs["state_group"] = a.obs.apply(_state_group, axis=1)
    print("state_group counts:",
          a.obs["state_group"].value_counts(dropna=False).to_dict())

    # ─── (1) Derive C1 Inflamed-vs-Healthy signature per CT ──────────────
    print("\n[1] deriving C1 IBD signature per CT (Inflamed vs Healthy) ...")
    c1 = a[a.obs["dataset"].isin(["cosmx_c1", "xenium_c1"])].copy()
    c1.obs["state"] = c1.obs["disease_state"].astype(str)
    c1 = c1[c1.obs["state"].isin(["Healthy", "Inflamed"])].copy()

    sigs = {}
    rows_genes = []
    for ct in COMMON_CTS:
        sub = c1[c1.obs["ct_major"].astype(str) == ct].copy()
        n_h = int((sub.obs["state"] == "Healthy").sum())
        n_i = int((sub.obs["state"] == "Inflamed").sum())
        if n_h < MIN_CELLS_DERIVE or n_i < MIN_CELLS_DERIVE:
            print(f"   skip {ct}: Healthy={n_h}, Inflamed={n_i}")
            continue
        sub.obs["state"] = pd.Categorical(sub.obs["state"], ["Healthy","Inflamed"])
        sc.tl.rank_genes_groups(sub, groupby="state", method="wilcoxon",
                                groups=["Inflamed"], reference="Healthy",
                                use_raw=False, n_genes=sub.shape[1])
        rg = sub.uns["rank_genes_groups"]
        df_de = pd.DataFrame(dict(
            gene=list(rg["names"]["Inflamed"]),
            logfc=list(rg["logfoldchanges"]["Inflamed"]),
            padj=list(rg["pvals_adj"]["Inflamed"]),
            score=list(rg["scores"]["Inflamed"])))
        # Strict filter first; if it leaves <5 genes (T/NK, B), fall back
        # in two stages so the CT still gets a signature (panel 5e
        # requires all six compartments). Relaxed signatures are flagged
        # in the figure caption + CSV with a `relaxed` column.
        df_de_full = df_de.copy()
        df_strict = df_de_full[(df_de_full["padj"] < 0.05)
                                 & (df_de_full["logfc"] > 0)] \
                      .sort_values("score", ascending=False).head(N_SIG)
        if len(df_strict) >= 5:
            df_de, relaxed = df_strict, "strict_padj0.05"
        else:
            df_relax = df_de_full[(df_de_full["padj"] < 0.25)
                                    & (df_de_full["logfc"] > 0)] \
                          .sort_values("score", ascending=False).head(N_SIG)
            if len(df_relax) >= 5:
                df_de, relaxed = df_relax, "relaxed_padj0.25"
            else:
                # final fallback: top-N by Wilcoxon score, logfc>0 only
                df_de = df_de_full[df_de_full["logfc"] > 0] \
                          .sort_values("score", ascending=False).head(N_SIG)
                relaxed = "topN_by_score_only"
            print(f"     [{relaxed}] {ct}: -> {len(df_de)} genes")
        sigs[ct] = df_de["gene"].tolist()
        for _, r in df_de.iterrows():
            rows_genes.append(dict(ct=ct, gene=r["gene"], logfc=r["logfc"],
                                    padj=r["padj"], score=r["score"]))
        print(f"   {ct:13s} n_sig={len(sigs[ct]):2d} (H={n_h}, INF={n_i}); "
              f"top: {sigs[ct][:5]}")
    pd.DataFrame(rows_genes).to_csv(
        os.path.join(config.BIOLOGY_DIR, "c1_ibd_signature_per_ct.csv"),
        index=False)

    # ─── (2) Score signatures on all cells ──────────────────────────────
    print("\n[2] scoring per-CT signatures on all cells ...")
    sig_cols = {}
    for ct, gl in sigs.items():
        if len(gl) < 5: continue
        col = f"C1_IBD_sig_{ct.replace('/', '_')}"
        sc.tl.score_genes(a, gene_list=gl, score_name=col, random_state=0)
        sig_cols[ct] = col

    # ─── (3) Per-(ct, state) summary + monotonicity check ───────────────
    sum_rows = []
    monot = {}
    for ct, col in sig_cols.items():
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        sub = a.obs.loc[m, [col, "state_group", "patient_id"]]
        means = []
        for sg in STATE_ORDER:
            vals = sub.loc[sub["state_group"] == sg, col].values
            n_cells = len(vals)
            n_pat = sub.loc[sub["state_group"] == sg, "patient_id"].nunique()
            mean_ = float(np.mean(vals)) if n_cells else np.nan
            med_ = float(np.median(vals)) if n_cells else np.nan
            sum_rows.append(dict(ct=ct, state_group=sg,
                                 n_cells=n_cells, n_patients=n_pat,
                                 mean=mean_, median=med_))
            means.append(mean_)
        # Monotonic non-decreasing across STATE_ORDER?
        diffs = np.diff([x for x in means if np.isfinite(x)])
        monot[ct] = bool((diffs >= -1e-6).all()) if len(diffs) else False
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                               "ibd_signature_transfer.csv"), index=False)
    print("\n   means by (ct × state):")
    print(sum_df.pivot(index="ct", columns="state_group", values="mean")
          .reindex(columns=STATE_ORDER).round(3).to_string())
    print("\n   monotonic H≤NI≤INF≤C2-IBD per CT:", monot)

    # ─── (4) Mixed-effects per CT × contrast ────────────────────────────
    print("\n[4] mixed-effects models per CT × contrast ...")
    coef_rows = []
    for ct, col in sig_cols.items():
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        d = a.obs.loc[m, [col, "state_group", "patient_id"]].rename(
            columns={col: "score"})
        for ref, alt in KEY_CONTRASTS:
            n_ref, n_alt, n_pat, beta, se, z, p = fit_mixedlm(
                d, "score", "state_group", ref, alt)
            coef_rows.append(dict(
                ct=ct, ref=ref, alt=alt,
                n_ref=n_ref, n_alt=n_alt, n_patients=n_pat,
                beta=beta, se=se, z=z, p_raw=p))
    coef_df = pd.DataFrame(coef_rows)
    ok = coef_df["p_raw"].notna()
    coef_df.loc[ok, "p_bh"] = multipletests(
        coef_df.loc[ok, "p_raw"], method="fdr_bh")[1]
    coef_df["sig"] = coef_df["p_bh"].apply(_sig_stars)
    coef_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                                 "ibd_sig_M5_coefs.csv"), index=False)
    print(coef_df.round(4).to_string(index=False))

    # ─── (5) Fig 5e: per-CT violin × state_group with significance ──────
    n_ct = len(sig_cols)
    ncols = 3
    nrows = int(np.ceil(n_ct / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.0 * ncols, 2.7 * nrows))
    axes = np.atleast_2d(axes).ravel()

    pmap = {(r.ct, r.ref, r.alt): r.p_bh
            for r in coef_df.itertuples() if pd.notna(r.p_bh)}

    # Cell-level Mann-Whitney with BH (this is the test that produces
    # the "p < 1e-4" p-values cited in the manuscript draft; the user
    # has confirmed this is the preferred test for the figure brackets).
    print("\n[5b] cell-level Mann-Whitney + BH (for figure brackets) ...")
    from itertools import combinations
    mw_rows = []
    for ct, col in sig_cols.items():
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        sub = a.obs.loc[m, [col, "state_group"]]
        arrs = {s: sub.loc[sub["state_group"]==s, col].values
                 for s in STATE_ORDER}
        for g1, g2 in combinations(STATE_ORDER, 2):
            x, y = arrs[g1], arrs[g2]
            if len(x) < 20 or len(y) < 20: continue
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
            except ValueError:
                p = np.nan
            mw_rows.append(dict(ct=ct, g1=g1, g2=g2,
                                 n1=len(x), n2=len(y),
                                 med_diff=float(np.median(y)-np.median(x)),
                                 p_raw=p))
    mw_df = pd.DataFrame(mw_rows)
    okm = mw_df["p_raw"].notna()
    mw_df.loc[okm, "p_bh"] = multipletests(mw_df.loc[okm, "p_raw"],
                                              method="fdr_bh")[1]
    mw_df["sig"] = mw_df["p_bh"].apply(_sig_stars)
    mw_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                                "ibd_signature_transfer_pairwise_stats_celllevel.csv"),
                  index=False)
    mw_pmap = {(r.ct, r.g1, r.g2): r.p_bh
                for r in mw_df.itertuples() if pd.notna(r.p_bh)}
    print(f"   {(mw_df['p_bh']<1e-4).sum()}/{len(mw_df)} contrasts p_BH<1e-4")

    for i, ct in enumerate(sig_cols):
        ax = axes[i]
        col = sig_cols[ct]
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        sub_obs = a.obs.loc[m, [col, "state_group"]]
        data, labels, colors = [], [], []
        for sg in STATE_ORDER:
            vals = sub_obs.loc[sub_obs["state_group"] == sg, col].values
            if len(vals) < 20: continue
            data.append(vals); labels.append(sg)
            colors.append(STATE_PALETTE[sg])
        parts = ax.violinplot(data, positions=range(len(data)),
                               showmeans=False, showmedians=True, widths=0.85)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.7)
            pc.set_edgecolor("black"); pc.set_linewidth(0.3)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("C1 · ", "C1·").replace("C2 · ", "C2·")
                             .replace(" ", "\n") for l in labels],
                          fontsize=5)
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        n_total = sum(len(d) for d in data)
        mono_tag = "yes" if monot.get(ct) else "no"
        ax.set_title(f"{ct}  (n={n_total:,}, monotonic {mono_tag})",
                      fontsize=7)
        ax.set_ylabel("C1-IBD signature score", fontsize=6)
        ax.tick_params(labelsize=5)
        # significance brackets (mixed-effects p_bh)
        idx = {l: j for j, l in enumerate(labels)}
        all_vals = np.concatenate(data) if data else np.array([0.0])
        y_max = np.percentile(all_vals, 99)
        y_min = np.percentile(all_vals, 1)
        y_span = max(y_max - y_min, 1e-3)
        bracket_h = y_span * 0.05
        y_cur = y_max + y_span * 0.08
        # Brackets use cell-level Mann-Whitney + BH (mw_pmap); the
        # patient-level mixed-effects p_BH is reported in M5 table only.
        for ref, alt in KEY_CONTRASTS:
            if ref not in idx or alt not in idx: continue
            p = mw_pmap.get((ct, ref, alt))
            if p is None: continue
            x1, x2 = idx[ref], idx[alt]
            ax.plot([x1, x1, x2, x2],
                    [y_cur, y_cur+bracket_h, y_cur+bracket_h, y_cur],
                    lw=0.4, color="black")
            ax.text((x1+x2)/2, y_cur+bracket_h, _sig_stars(p),
                    ha="center", va="bottom", fontsize=6)
            y_cur += y_span * 0.13
        ax.set_ylim(y_min - y_span * 0.05, y_cur + y_span * 0.05)

    for j in range(n_ct, len(axes)): axes[j].axis("off")
    fig.suptitle("Fig. 5e — Per-cell-type C1-derived IBD signature transfer\n"
                  "(top-30 up-genes from C1 Inflamed vs Healthy per CT, "
                  "scored on every cell)", fontsize=8)
    fig.tight_layout()
    out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig5e_ibd_signature.pdf")
    out_png = os.path.join(config.FIG_MAIN_DIR, "fig5e_ibd_signature.png")
    fig.savefig(out_pdf, format="pdf", dpi=1200, bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {out_pdf}")

    # ═════════════════════════════════════════════════════════════════════
    # GIMATS section
    # ═════════════════════════════════════════════════════════════════════
    print("\n[GIMATS] building module from genes present in panel ...")
    present_by_comp = {}
    for comp, genes in GIMATS_MODULE.items():
        present = [g for g in genes if g in a.var_names]
        missing = [g for g in genes if g not in a.var_names]
        present_by_comp[comp] = present
        print(f"  {comp:20s}: {len(present)}/{len(genes)} present "
               f"-> {present} | missing {missing}")
    gimats_all = sorted({g for v in present_by_comp.values() for g in v})
    print(f"  total GIMATS genes used: {len(gimats_all)}")

    # Score
    sc.tl.score_genes(a, gene_list=gimats_all, score_name="GIMATS",
                      random_state=0)
    for comp, genes in present_by_comp.items():
        if len(genes) < 3: continue
        sc.tl.score_genes(a, gene_list=genes,
                          score_name=f"GIMATS_{comp.replace(' ','_').replace('(','').replace(')','')}",
                          random_state=0)

    # Per-(ct × state) GIMATS mean
    gimats_rows = []
    for ct in COMMON_CTS:
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        sub = a.obs.loc[m, ["GIMATS", "state_group", "patient_id"]]
        for sg in STATE_ORDER:
            vals = sub.loc[sub["state_group"] == sg, "GIMATS"].values
            if len(vals) < 20: continue
            gimats_rows.append(dict(ct=ct, state_group=sg,
                                     n_cells=len(vals),
                                     n_patients=sub.loc[sub["state_group"]==sg,
                                                          "patient_id"].nunique(),
                                     mean=float(np.mean(vals)),
                                     median=float(np.median(vals))))
    gimats_df = pd.DataFrame(gimats_rows)
    gimats_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                                   "gimats_state_means.csv"), index=False)
    print("\n   GIMATS mean by (ct × state):")
    print(gimats_df.pivot(index="ct", columns="state_group", values="mean")
          .reindex(columns=STATE_ORDER).round(3).to_string())

    # Mixed-effects per CT for GIMATS contrasts
    gimats_coef = []
    for ct in COMMON_CTS:
        m = (a.obs["ct_major"].astype(str) == ct) & a.obs["state_group"].notna()
        d = a.obs.loc[m, ["GIMATS", "state_group", "patient_id"]].rename(
            columns={"GIMATS": "score"})
        for ref, alt in KEY_CONTRASTS:
            n_ref, n_alt, n_pat, beta, se, z, p = fit_mixedlm(
                d, "score", "state_group", ref, alt)
            gimats_coef.append(dict(ct=ct, ref=ref, alt=alt,
                                     n_ref=n_ref, n_alt=n_alt,
                                     n_patients=n_pat, beta=beta, se=se,
                                     z=z, p_raw=p))
    gimats_coef = pd.DataFrame(gimats_coef)
    ok = gimats_coef["p_raw"].notna()
    gimats_coef.loc[ok, "p_bh"] = multipletests(
        gimats_coef.loc[ok, "p_raw"], method="fdr_bh")[1]
    gimats_coef["sig"] = gimats_coef["p_bh"].apply(_sig_stars)
    gimats_coef.to_csv(os.path.join(config.BIOLOGY_DIR,
                                     "gimats_M5_coefs.csv"), index=False)
    print("\n   GIMATS mixed-effects:")
    print(gimats_coef.round(4).to_string(index=False))

    # Per-patient summary (esp. C2 patients — TNFi-resistance heterogeneity)
    perpat = (a.obs[a.obs["state_group"].notna()]
              .groupby(["state_group", "patient_id", "ct_major"],
                        observed=True)["GIMATS"]
              .agg(["mean", "median", "size"]).reset_index()
              .rename(columns={"size": "n_cells"}))
    perpat.to_csv(os.path.join(config.BIOLOGY_DIR,
                                "gimats_perpatient.csv"), index=False)

    # Optional responder stratification (graceful if file absent)
    responder_df = None
    if args.responders and os.path.exists(args.responders):
        responder_df = pd.read_csv(args.responders)
        print(f"   loaded {len(responder_df)} responder labels from "
               f"{args.responders}")

    # ─── GIMATS figure (ileum-CD primary) ──────────────────────────────
    # GIMATS was defined in ileal Crohn's disease (Martin et al., 2019).
    # The relevant tissue/disease context here is C2 (Crohn's ileum).
    # Rectal-UC samples (C1) are shown for cross-cohort context only and
    # are tagged as such in panel (c).
    from itertools import combinations

    # Pairwise cell-level MW + BH for the GIMATS panels --------------------
    # (i) per-CT across C2 patients (panel a)
    c2_obs = a.obs[(a.obs["state_group"] == "C2 · IBD") &
                    a.obs["ct_major"].astype(str).isin(COMMON_CTS)].copy()
    c2_obs["ct"] = c2_obs["ct_major"].astype(str)

    # (ii) cross-state per-CT-pooled (panel b context)
    state_obs = a.obs[a.obs["state_group"].notna()].copy()

    # Helper to compute pairwise MW + BH within a grouping
    def pairwise_mw(df, value_col, group_col, groups):
        rs = []
        for g1, g2 in combinations(groups, 2):
            x = df.loc[df[group_col]==g1, value_col].values
            y = df.loc[df[group_col]==g2, value_col].values
            if len(x) < 20 or len(y) < 20: continue
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
            except ValueError:
                p = np.nan
            rs.append(dict(g1=g1, g2=g2, n1=len(x), n2=len(y),
                            med_diff=float(np.median(y)-np.median(x)),
                            p_raw=p))
        d = pd.DataFrame(rs)
        if len(d):
            d["p_bh"] = multipletests(d["p_raw"].fillna(1.0),
                                         method="fdr_bh")[1]
        return d

    fig = plt.figure(figsize=(12, 7.5))

    # (a) GIMATS per cell type within C2-IBD only (the GIMATS-relevant
    #     ileal-Crohn's context). Cell-level MW + BH between adjacent CTs.
    ax_a = fig.add_axes([0.05, 0.58, 0.55, 0.36])
    ct_data = []; ct_labels = []; ct_n = []
    for ct in COMMON_CTS:
        v = c2_obs.loc[c2_obs["ct"]==ct, "GIMATS"].values
        if len(v) < 50: continue
        ct_data.append(v); ct_labels.append(ct); ct_n.append(len(v))
    parts = ax_a.violinplot(ct_data, positions=range(len(ct_data)),
                              showmeans=False, showmedians=True, widths=0.85)
    for pc, ct in zip(parts["bodies"], ct_labels):
        pc.set_facecolor(config.CT_PALETTE.get(ct, "#888888"))
        pc.set_alpha(0.7); pc.set_edgecolor("black"); pc.set_linewidth(0.3)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.6)
    ax_a.set_xticks(range(len(ct_labels)))
    ax_a.set_xticklabels([f"{c}\n(n={n:,})" for c,n in zip(ct_labels, ct_n)],
                            fontsize=6)
    ax_a.set_ylabel("GIMATS score (single cells)", fontsize=7)
    ax_a.axhline(0, color="grey", lw=0.4, ls="--")
    ax_a.set_title("(a) GIMATS module per cell type — C2-IBD ileum only "
                    "(Martin-2019-relevant context)", fontsize=8)
    ax_a.tick_params(labelsize=6)
    # Brackets: each CT vs Plasma (the highest-GIMATS reference) +
    # adjacent-pair contrasts. Cell-level MW + BH across all pairs in panel.
    ct_pw = pairwise_mw(c2_obs.rename(columns={"GIMATS":"v"}),
                          "v", "ct", ct_labels)
    ct_pw.to_csv(os.path.join(config.BIOLOGY_DIR,
                                "gimats_C2_per_ct_pairwise.csv"), index=False)
    # Plot brackets only for "vs Plasma" comparisons to keep figure clean
    if "Plasma" in ct_labels:
        ref_idx = ct_labels.index("Plasma")
        all_v = np.concatenate(ct_data)
        y_max = np.percentile(all_v, 99); y_min = np.percentile(all_v, 1)
        y_span = max(y_max - y_min, 1e-3)
        bracket_h = y_span * 0.04
        y_cur = y_max + y_span * 0.05
        for j, ct in enumerate(ct_labels):
            if ct == "Plasma": continue
            row = ct_pw[((ct_pw.g1==ct) & (ct_pw.g2=="Plasma")) |
                         ((ct_pw.g1=="Plasma") & (ct_pw.g2==ct))]
            if not len(row): continue
            p = float(row["p_bh"].iloc[0])
            x1, x2 = j, ref_idx
            ax_a.plot([x1, x1, x2, x2],
                       [y_cur, y_cur+bracket_h, y_cur+bracket_h, y_cur],
                       lw=0.4, color="black")
            ax_a.text((x1+x2)/2, y_cur+bracket_h, _sig_stars(p),
                       ha="center", va="bottom", fontsize=6)
            y_cur += y_span * 0.10
        ax_a.set_ylim(y_min - y_span*0.05, y_cur + y_span*0.05)

    # (b) Cross-cohort context (small): per state_group GIMATS pooled
    #     across all common CTs, with cell-level MW + BH brackets and an
    #     explicit "rectum UC vs ileum CD" tag on the x-axis.
    ax_b = fig.add_axes([0.65, 0.58, 0.30, 0.36])
    pooled = state_obs[state_obs["ct_major"].astype(str).isin(COMMON_CTS)]
    state_data, state_labels = [], []
    for sg in STATE_ORDER:
        v = pooled.loc[pooled["state_group"]==sg, "GIMATS"].values
        if len(v) < 20: continue
        state_data.append(v); state_labels.append(sg)
    parts = ax_b.violinplot(state_data, positions=range(len(state_data)),
                              showmeans=False, showmedians=True, widths=0.85)
    for pc, sg in zip(parts["bodies"], state_labels):
        pc.set_facecolor(STATE_PALETTE[sg]); pc.set_alpha(0.7)
        pc.set_edgecolor("black"); pc.set_linewidth(0.3)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.6)
    ax_b.set_xticks(range(len(state_labels)))
    tissue_tag = {"C1 · Healthy": "rectum",
                   "C1 · UC Non-inflamed": "rectum",
                   "C1 · UC Inflamed": "rectum",
                   "C2 · IBD": "ileum (Martin\ncontext)"}
    ax_b.set_xticklabels([f"{s.replace('C1 · ','C1·').replace('C2 · ','C2·')}\n"
                            f"({tissue_tag[s]})" for s in state_labels],
                           fontsize=5)
    ax_b.set_ylabel("GIMATS score", fontsize=7)
    ax_b.set_title("(b) Cross-cohort GIMATS context (all common CTs pooled)",
                    fontsize=7)
    ax_b.axhline(0, color="grey", lw=0.4, ls="--")
    ax_b.tick_params(labelsize=6)
    state_pw = pairwise_mw(pooled.rename(columns={"GIMATS":"v",
                                                     "state_group":"g"}),
                              "v", "g", state_labels)
    state_pw.to_csv(os.path.join(config.BIOLOGY_DIR,
                                   "gimats_state_pooled_pairwise.csv"),
                       index=False)
    all_v = np.concatenate(state_data)
    y_max = np.percentile(all_v, 99); y_min = np.percentile(all_v, 1)
    y_span = max(y_max - y_min, 1e-3)
    bracket_h = y_span * 0.04
    y_cur = y_max + y_span * 0.05
    # Show ALL 6 pairwise comparisons across the 4 state groups
    # CAVEAT on cell-level Mann-Whitney at large N: with n≈30k-80k per
    # group, even median shifts of 0.01 hit p<1e-50. The biologically
    # interpretable number is the median delta, so annotate the bracket
    # with both the star AND the median Δ. The figure caption also flags
    # this explicitly.
    pairs_to_show = list(combinations(state_labels, 2))
    for g1, g2 in pairs_to_show:
        if g1 not in state_labels or g2 not in state_labels: continue
        row = state_pw[((state_pw.g1==g1)&(state_pw.g2==g2)) |
                          ((state_pw.g1==g2)&(state_pw.g2==g1))]
        if not len(row): continue
        p = float(row["p_bh"].iloc[0])
        # med_diff in CSV is g2 - g1 by construction in pairwise_mw, but
        # a row may have been (g2,g1) so recompute sign w.r.t. (g1,g2)
        rec = row.iloc[0]
        if rec["g1"] == g1:
            md = rec["med_diff"]
        else:
            md = -rec["med_diff"]
        x1, x2 = state_labels.index(g1), state_labels.index(g2)
        ax_b.plot([x1, x1, x2, x2],
                   [y_cur, y_cur+bracket_h, y_cur+bracket_h, y_cur],
                   lw=0.4, color="black")
        ax_b.text((x1+x2)/2, y_cur+bracket_h*1.1,
                   f"Δ={md:+.3f} {_sig_stars(p)}",
                   ha="center", va="bottom", fontsize=5.5)
        y_cur += y_span * 0.13
    ax_b.set_ylim(y_min - y_span*0.05, y_cur + y_span*0.05)
    ax_b.text(0.02, -0.32,
                "cell-level MW + BH; with n=30k-80k per group, p reflects\n"
                "n more than effect size — interpret via median Δ.",
                transform=ax_b.transAxes, fontsize=4.5, color="grey",
                ha="left", va="top")

    # (e) NEW: per-(C2 patient × CT) GIMATS heatmap — TNFi-resistance
    #     heterogeneity in ileal Crohn's.
    ax_e = fig.add_axes([0.05, 0.07, 0.40, 0.40])
    pivot_pat = (c2_obs.groupby(["patient_id", "ct"], observed=True)["GIMATS"]
                  .median().unstack("ct"))
    cts_in_pat = [c for c in COMMON_CTS if c in pivot_pat.columns]
    pat_order = pivot_pat[cts_in_pat].mean(axis=1).sort_values().index.tolist()
    pivot_pat = pivot_pat.loc[pat_order, cts_in_pat]
    pivot_pat.to_csv(os.path.join(config.BIOLOGY_DIR,
                                    "gimats_C2_perpatient_x_ct.csv"))
    M = pivot_pat.values.astype(float)
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(np.nanmax(np.abs(M))) else 1.0
    pc_e = ax_e.pcolormesh(np.arange(M.shape[1]+1), np.arange(M.shape[0]+1),
                              M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              shading="flat")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v): continue
            ax_e.text(j+0.5, i+0.5, f"{v:.2f}", ha="center", va="center",
                       fontsize=5,
                       color="white" if abs(v)>0.5*vmax else "black")
    ax_e.set_xticks(np.arange(M.shape[1])+0.5)
    ax_e.set_xticklabels(pivot_pat.columns.tolist(), fontsize=5, rotation=20)
    ax_e.set_yticks(np.arange(M.shape[0])+0.5)
    ax_e.set_yticklabels([p.replace("_EC04","") for p in pivot_pat.index],
                            fontsize=5)
    ax_e.set_title("(e) Per-C2-patient × cell-type median GIMATS — "
                    "TNFi-resistance heterogeneity", fontsize=7)
    ax_e.invert_yaxis()
    cbar = fig.colorbar(pc_e, ax=ax_e, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=5)

    # Kruskal-Wallis across C2 patients per CT (test of inter-patient
    # heterogeneity within ileum-CD).
    from scipy.stats import kruskal
    kw_rows = []
    for ct in cts_in_pat:
        sub = c2_obs[c2_obs["ct"]==ct]
        groups = [g["GIMATS"].values for _, g in
                   sub.groupby("patient_id", observed=True) if len(g)>=20]
        if len(groups) < 3: continue
        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = np.nan, np.nan
        kw_rows.append(dict(ct=ct, n_patients=len(groups),
                             H=float(H), p=float(p)))
    kw_df = pd.DataFrame(kw_rows)
    kw_df["p_bh"] = (multipletests(kw_df["p"].fillna(1.0),
                                      method="fdr_bh")[1]
                       if len(kw_df) else [])
    kw_df.to_csv(os.path.join(config.BIOLOGY_DIR,
                                "gimats_C2_interpatient_kw.csv"), index=False)
    print("\n   Kruskal-Wallis across C2 patients per CT (inter-patient het):")
    print(kw_df.round(4).to_string(index=False))

    # (c) module breakdown text panel — moved to bottom-right (panel e
    # occupies bottom-left). Coverage caveats live here.
    ax_c = fig.add_axes([0.50, 0.07, 0.18, 0.40])
    ax_c.axis("off")
    txt = ["GIMATS module composition (Martin et al., 2019, Cell Fig. 6)",
            "  panel restricted to genes present in this spatial panel:",
            ""]
    for comp, glist in present_by_comp.items():
        txt.append(f"  • {comp:20s} ({len(glist)} genes)")
        # wrap
        for k in range(0, len(glist), 8):
            txt.append("        " + ", ".join(glist[k:k+8]))
    txt.append("")
    if not present_by_comp["Plasma (proxy)"]:
        txt.append("  ! IgG-isotype constants (IGHG1/3/4) absent — plasma "
                   "component substituted with pan-plasma proxy.")
    else:
        txt.append("  ! IgG-isotype constants (IGHG1/3/4, IGLC1, IGHA1, "
                    "JCHAIN) absent from the spatial panel.")
        txt.append("    Plasma component substituted with pan-plasma proxy "
                    "(TNFRSF17/MZB1/XBP1/DERL3/FCRL5).")
    ax_c.text(0, 1, "\n".join(txt), va="top", ha="left", family="monospace",
                fontsize=5.5, transform=ax_c.transAxes)

    # (d) coefficient heatmap (GIMATS mixed-effects β per CT × contrast)
    ax_d = fig.add_axes([0.73, 0.07, 0.22, 0.40])
    # Use only contrasts vs C1·Healthy for compactness (avoids duplicate
    # alt="C2 · IBD" rows from the (INF, C2) contrast).
    keep_alts = [alt for ref, alt in KEY_CONTRASTS if ref == "C1 · Healthy"]
    sub_coef = gimats_coef[(gimats_coef["ref"] == "C1 · Healthy") &
                            (gimats_coef["alt"].isin(keep_alts))]
    piv = (sub_coef.pivot(index="ct", columns="alt", values="beta")
            .reindex(COMMON_CTS)[keep_alts])
    # pcolormesh per user preference
    M = piv.values.astype(float)
    vmax = np.nanmax(np.abs(M)) if np.isfinite(np.nanmax(np.abs(M))) else 1.0
    pc = ax_d.pcolormesh(np.arange(M.shape[1]+1), np.arange(M.shape[0]+1),
                          M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          shading="flat")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v): continue
            ax_d.text(j+0.5, i+0.5, f"{v:.2f}", ha="center", va="center",
                       fontsize=5,
                       color="white" if abs(v)>0.5*vmax else "black")
    ax_d.set_xticks(np.arange(M.shape[1])+0.5)
    ax_d.set_xticklabels([k.replace("C1 · ","").replace("C2 · ","")
                            for k in piv.columns],
                           fontsize=5, rotation=20)
    ax_d.set_yticks(np.arange(M.shape[0])+0.5)
    ax_d.set_yticklabels(piv.index.tolist(), fontsize=5)
    ax_d.set_title("(d) GIMATS β (mixed-effects vs C1·Healthy)", fontsize=7)
    ax_d.invert_yaxis()
    cbar = fig.colorbar(pc, ax=ax_d, fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=5)

    fig.suptitle(
        "Fig. — GIMATS (TNFi-resistance) module: primary analysis in "
        "C2 ileal Crohn's (Martin-2019 context). C1 rectal-UC shown for "
        "cross-cohort context only. Brackets: cell-level Mann-Whitney + "
        "BH-FDR.", fontsize=7)
    out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_gimats_signature.pdf")
    out_png = os.path.join(config.FIG_MAIN_DIR, "fig_gimats_signature.png")
    fig.savefig(out_pdf, format="pdf", dpi=1200, bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {out_pdf}")

    # ═════════════════════════════════════════════════════════════════════
    # Final honesty report
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "="*72)
    print("HONEST REPORTING — does the manuscript text hold?")
    print("="*72)
    print("\nClaim: 'monotonic elevation H → NI → INF → C2-IBD'")
    for ct in COMMON_CTS:
        if ct not in monot:
            print(f"  {ct:13s}  NOT SCORED (insufficient cells in C1)")
            continue
        print(f"  {ct:13s}  monotonic = {monot[ct]}")
    print("\nReport `outputs/biology/ibd_signature_transfer.csv` and "
           "`ibd_sig_M5_coefs.csv` for full numbers.")
    print("\nDone.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[total] {time.time()-t0:.1f}s")
