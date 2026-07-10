"""02b — Spatial neighborhood enrichment per cohort_platform.

Loads the three spatial h5ads from ../IBD_Exploration/outputs/, computes
squidpy.gr.nhood_enrichment per sample/section, then averages z-scores per
cohort_platform. Output:

  outputs/biology/nhood_zscore_C1-Rectum-CosMx.csv  (9 × 9)
  outputs/biology/nhood_zscore_C1-Rectum-Xenium.csv (9 × 9)
  outputs/biology/nhood_zscore_C2-Ileum-CosMx.csv   (9 × 9)
  outputs/biology/nhood_zscore_long.csv             (long-format combined)

Sub-samples large samples to ≤30,000 cells each (radius graph at 350 µm
becomes expensive otherwise).

Run: python scripts/02b_spatial_nhood.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
config.ensure_output_dirs()

EXP_OUT = os.path.join(config.IBD_EXP_ROOT, "outputs")
SPATIAL_H5AD = {
    "C1-Rectum-CosMx":  os.path.join(EXP_OUT, "cosmx_c1_spatial.h5ad"),
    "C1-Rectum-Xenium": os.path.join(EXP_OUT, "xenium_c1_spatial.h5ad"),
    "C2-Ileum-CosMx":   os.path.join(EXP_OUT, "cosmx_c2_spatial.h5ad"),
}
SAMPLE_KEY = {
    "C1-Rectum-CosMx":  "sample",
    "C1-Rectum-Xenium": "sample",
    "C2-Ileum-CosMx":   "section_ID",
}
RADIUS_UM = 350
N_PER_SAMPLE_MAX = 30_000
RNG = np.random.default_rng(0)

t0 = time.time()
all_long = []

for cp, h5_path in SPATIAL_H5AD.items():
    print(f"\n[02b] {cp} -> loading {os.path.basename(h5_path)} ...")
    # Load only obs + obsm['spatial'] to keep memory down
    a = ad.read_h5ad(h5_path)
    print(f"      n_cells={a.shape[0]:,}; samples in '{SAMPLE_KEY[cp]}' = "
          f"{a.obs[SAMPLE_KEY[cp]].nunique()}")

    # Ensure ct_major is categorical with full ordering
    a.obs["ct_major"] = pd.Categorical(
        a.obs["ct_major"].astype(str), categories=config.CT_MAJOR_ORDER,
        ordered=True,
    )

    samples = a.obs[SAMPLE_KEY[cp]].unique()
    z_per_sample = []
    for samp in samples:
        m = (a.obs[SAMPLE_KEY[cp]] == samp).values
        n = int(m.sum())
        if n < 200:
            print(f"      skip {samp} (n={n})")
            continue
        sub = a[m].copy()
        if sub.shape[0] > N_PER_SAMPLE_MAX:
            keep_idx = RNG.choice(sub.shape[0], size=N_PER_SAMPLE_MAX,
                                   replace=False)
            sub = sub[keep_idx].copy()
        # Build radius spatial graph
        sq.gr.spatial_neighbors(sub, coord_type="generic",
                                  radius=RADIUS_UM, delaunay=False)
        sq.gr.nhood_enrichment(sub, cluster_key="ct_major",
                                seed=0, show_progress_bar=False)
        z = sub.uns["ct_major_nhood_enrichment"]["zscore"]
        # Restore full CT ordering (squidpy may drop empty cats)
        present = list(sub.obs["ct_major"].cat.categories)
        z_df = pd.DataFrame(z, index=present, columns=present)
        z_df = z_df.reindex(index=config.CT_MAJOR_ORDER,
                            columns=config.CT_MAJOR_ORDER)
        z_per_sample.append(z_df)
        print(f"      {samp}: n={sub.shape[0]:,} -> nhood OK")

    if not z_per_sample:
        print(f"      WARN: no usable samples for {cp}; skipping")
        continue

    # Average across samples (NaNs ignored)
    avg = sum(d.fillna(0) * (~d.isna()).astype(int) for d in z_per_sample)
    counts = sum((~d.isna()).astype(int) for d in z_per_sample)
    avg = avg / counts.replace(0, 1)
    avg = avg.where(counts > 0)

    out_path = os.path.join(config.BIOLOGY_DIR,
                             f"nhood_zscore_{cp.replace('/', '_')}.csv")
    avg.to_csv(out_path)
    print(f"      -> {os.path.basename(out_path)}")

    long = (avg.reset_index(names="ct_a")
                .melt(id_vars="ct_a", var_name="ct_b", value_name="zscore"))
    long["cohort_platform"] = cp
    all_long.append(long)

if all_long:
    pd.concat(all_long, ignore_index=True).to_csv(
        os.path.join(config.BIOLOGY_DIR, "nhood_zscore_long.csv"), index=False
    )
    print(f"\n[02b] long-format combined CSV written.")

print(f"\n[02b] DONE in {time.time()-t0:.1f}s")
