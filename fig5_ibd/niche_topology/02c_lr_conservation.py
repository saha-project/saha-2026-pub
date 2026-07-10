"""02c — LIANA L-R conservation across cohort_platforms.

Reuses upstream LIANA outputs (do NOT recompute):
  C1-Rectum-CosMx:  IBD_Exploration/outputs/phase4/concordance/liana_shared_cosmx_c1.csv
  C1-Rectum-Xenium: IBD_Exploration/outputs/phase4/concordance/liana_shared_xenium_c1.csv
  C2-Ileum-CosMx:   IBD_Exploration/outputs/phase4/responder/liana_saha_IBD_*.csv
                     (8 sections; aggregated by mean magnitude_rank)

Outputs:
  outputs/biology/lr_topN.csv             (per cohort_platform top 100 L-R pairs)
  outputs/biology/lr_jaccard.csv          (3 × 3 Jaccard of top-100 pairs)
  outputs/biology/lr_rank_corr.csv        (3 × 3 Spearman of magnitude_rank
                                            on the union of top pairs)
  outputs/biology/lr_top_overlap.csv      (the union top-pairs with their per-CP
                                            ranks — for the figure heatmap)

Run: python scripts/02c_lr_conservation.py
"""
from __future__ import annotations

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()

EXP_PHASE4 = config.EXP_PHASE4
LIANA_PATHS = {
    "C1-Rectum-CosMx":
        os.path.join(EXP_PHASE4, "concordance", "liana_shared_cosmx_c1.csv"),
    "C1-Rectum-Xenium":
        os.path.join(EXP_PHASE4, "concordance", "liana_shared_xenium_c1.csv"),
}
C2_GLOB = os.path.join(EXP_PHASE4, "responder", "liana_saha_IBD_*.csv")
TOPN = 100


def _pair_key(df):
    return (df["source"].astype(str) + "|"
            + df["target"].astype(str) + "|"
            + df["ligand_complex"].astype(str) + "|"
            + df["receptor_complex"].astype(str))


def _load_one(path: str, cp: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pair"] = _pair_key(df)
    # magnitude_rank: smaller = stronger interaction in liana_py
    df = df[["pair", "source", "target",
             "ligand_complex", "receptor_complex",
             "magnitude_rank", "specificity_rank", "lr_means"]]
    df["cohort_platform"] = cp
    return df


t0 = time.time()
frames = []
for cp, p in LIANA_PATHS.items():
    if not os.path.exists(p):
        print(f"     missing {p}; skip {cp}")
        continue
    df = _load_one(p, cp)
    print(f"     {cp}: {len(df):,} rows from {os.path.basename(p)}")
    frames.append(df)

# C2: aggregate 8 SAHA sections
c2_files = sorted(glob.glob(C2_GLOB))
print(f"     C2 files: {len(c2_files)}")
c2_concat = []
for p in c2_files:
    d = pd.read_csv(p)
    d["pair"] = _pair_key(d)
    d = d[["pair", "source", "target",
           "ligand_complex", "receptor_complex",
           "magnitude_rank", "specificity_rank", "lr_means"]]
    d["section"] = os.path.basename(p).replace("liana_saha_", "").replace(".csv", "")
    c2_concat.append(d)
c2_long = pd.concat(c2_concat, ignore_index=True)
# Aggregate: mean magnitude_rank across sections (smaller = stronger)
c2_agg = (c2_long.groupby(["pair", "source", "target",
                            "ligand_complex", "receptor_complex"],
                            as_index=False)
                  .agg(magnitude_rank=("magnitude_rank", "mean"),
                       specificity_rank=("specificity_rank", "mean"),
                       lr_means=("lr_means", "mean"),
                       n_sections=("section", "nunique")))
c2_agg["cohort_platform"] = "C2-Ileum-CosMx"
print(f"     C2 aggregated: {len(c2_agg):,} unique pairs")
frames.append(c2_agg)

all_lr = pd.concat(frames, ignore_index=True, sort=False)
print(f"     combined LR table: {len(all_lr):,} rows")

# ── Top-N per cohort_platform ───────────────────────────────────────────────
top = (all_lr.sort_values("magnitude_rank")
              .groupby("cohort_platform")
              .head(TOPN)
              .reset_index(drop=True))
top.to_csv(os.path.join(config.BIOLOGY_DIR, "lr_topN.csv"), index=False)

# ── Jaccard of top-N sets ───────────────────────────────────────────────────
sets = {cp: set(g["pair"]) for cp, g in top.groupby("cohort_platform")}
cps_present = list(sets.keys())
jacc = pd.DataFrame(index=cps_present, columns=cps_present, dtype=float)
for a in cps_present:
    for b in cps_present:
        u = len(sets[a] | sets[b])
        i = len(sets[a] & sets[b])
        jacc.loc[a, b] = i / u if u else np.nan
jacc.to_csv(os.path.join(config.BIOLOGY_DIR, "lr_jaccard.csv"))
print("\n     Top-100 L-R pair Jaccard (cohort_platform × cohort_platform):")
print(jacc.round(3).to_string())

# ── Spearman of magnitude_rank on the union of top-N pairs ──────────────────
union_pairs = set().union(*sets.values())
print(f"\n     Union top pairs: {len(union_pairs)}")
ranks = (all_lr[all_lr["pair"].isin(union_pairs)]
            .pivot_table(index="pair", columns="cohort_platform",
                         values="magnitude_rank", aggfunc="mean"))
# Use negative rank so Spearman positive = concordant strength
ranks = -ranks
spear = pd.DataFrame(index=cps_present, columns=cps_present, dtype=float)
for a in cps_present:
    for b in cps_present:
        sub = ranks[[a, b]].dropna()
        if len(sub) < 5:
            spear.loc[a, b] = np.nan
            continue
        # sub may be a DataFrame with duplicate cols if a==b -> use iloc
        x = sub.iloc[:, 0].to_numpy(dtype=float).ravel()
        y = sub.iloc[:, -1].to_numpy(dtype=float).ravel()
        if a == b:
            spear.loc[a, b] = 1.0
            continue
        res = spearmanr(x, y)
        try:
            rho = float(res.statistic)
        except (TypeError, AttributeError, ValueError):
            rho = float(np.atleast_1d(res)[0])
        spear.loc[a, b] = rho
spear.to_csv(os.path.join(config.BIOLOGY_DIR, "lr_rank_corr.csv"))
print("\n     L-R rank correlation (union top pairs, Spearman of -magnitude_rank):")
print(spear.round(3).to_string())

# ── Long-format top union pairs for the figure ──────────────────────────────
overlap_long = (all_lr[all_lr["pair"].isin(union_pairs)]
                  [["pair", "source", "target", "ligand_complex",
                    "receptor_complex", "magnitude_rank",
                    "cohort_platform"]])
overlap_long.to_csv(os.path.join(config.BIOLOGY_DIR,
                                  "lr_top_overlap.csv"), index=False)

print(f"\n[02c] DONE in {time.time()-t0:.1f}s")
