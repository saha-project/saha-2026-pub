"""Redraw Fig 1d (and ED Fig 1a) from the saved UMAP coordinates in the Source Data.

The coordinates are the published ones (obsm['X_umap'] of the all-organ RNA object, computed on
X_harmony; see fig1d_joint_umap.py), so this reproduces the figure exactly; a fresh UMAP run would
not, since UMAP layouts vary across machines and library versions.

Usage: python plot_fig1d_from_coordinates.py Fig1_d_umap_coordinates_RNA.parquet [out_prefix]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

src = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "fig1d"
palette = json.loads((Path(__file__).parent / "fig1d_palette.json").read_text())
df = pd.read_parquet(src)

for col, tag in [("Insitutype_Broad", "celltype"), ("SAHA_organcode", "organ")]:  # Fig 1d, ED Fig 1a
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = df[col].astype(str).map(palette[col]).fillna("lightgray")
    ax.scatter(df.umap_1, df.umap_2, c=colors, s=0.05, linewidths=0, rasterized=True)
    for name, c in palette[col].items():
        ax.scatter([], [], c=c, s=20, label=name)
    ax.legend(frameon=False, fontsize=7, markerscale=1, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_axis_off()
    fig.savefig(f"{prefix}_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
