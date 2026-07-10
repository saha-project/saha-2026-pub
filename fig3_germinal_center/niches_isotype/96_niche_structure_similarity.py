"""
96_niche_structure_similarity.py
Quantify similarity of the 8 UTAG niche "structures" (cell-type composition
vectors) two ways:

  A. Cross-organ cosine similarity PER niche
     -> is the same niche built from the same cells in ILE/COL/APE?
  B. Bray-Curtis dissimilarity BETWEEN niches (organ-averaged composition)
     -> how distinct are the 8 niches from each other?

Input : utag_niche_analysis/niche_composition_per_organ.csv
        (niche, organ, cell_type, count, frac, n_niche_cells)
Output: utag_niche_analysis/niche_structure_similarity_crossorgan.csv
        utag_niche_analysis/niche_structure_dissimilarity_between.csv

Run: conda run -n saha python scripts/96_niche_structure_similarity.py
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import braycurtis, cosine, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "utag_niche_analysis"
FIG = BASE / "figures"
df = pd.read_csv(OUT / "niche_composition_per_organ.csv")

# composition matrix: rows = (organ, niche), cols = cell_type, values = frac.
# missing cell types in a niche -> 0 (absent component, correct for both metrics)
comp = (df.pivot_table(index=["organ", "niche"], columns="cell_type",
                       values="frac", fill_value=0.0)
          .sort_index())
niches = sorted(df["niche"].unique())
organs = sorted(df["organ"].unique())

# ── A. cross-organ cosine similarity per niche ────────────────────────
rows = []
for n in niches:
    vecs = {o: comp.loc[(o, n)].values for o in organs if (o, n) in comp.index}
    sims = [1 - cosine(vecs[a], vecs[b]) for a, b in combinations(vecs, 2)]
    rows.append({"niche": n, "n_organs": len(vecs),
                 "mean_crossorgan_cosine": np.mean(sims) if sims else np.nan})
cross = pd.DataFrame(rows).sort_values("mean_crossorgan_cosine", ascending=False)
cross.to_csv(OUT / "niche_structure_similarity_crossorgan.csv", index=False)

# ── B. Bray-Curtis between niches (organ-averaged composition) ─────────
avg = comp.groupby("niche").mean()            # organ-averaged composition per niche
avg = avg.div(avg.sum(axis=1), axis=0)        # renormalize to sum=1
bc = pd.DataFrame(index=niches, columns=niches, dtype=float)
for a, b in combinations(niches, 2):
    d = braycurtis(avg.loc[a].values, avg.loc[b].values)
    bc.loc[a, b] = bc.loc[b, a] = d
np.fill_diagonal(bc.values, 0.0)
bc.to_csv(OUT / "niche_structure_dissimilarity_between.csv")
mean_bc = bc.values[np.triu_indices(len(niches), k=1)].mean()

# ── report ────────────────────────────────────────────────────────────
print(f"8-niche UTAG map | {len(organs)} organs {organs}\n")
print("A. cross-organ COSINE similarity per niche (1=identical composition):")
for _, r in cross.iterrows():
    print(f"   {r['niche']:24s} {r['mean_crossorgan_cosine']:.3f}  "
          f"(n_organ={int(r['n_organs'])})")
print(f"   {'MEAN across niches':24s} {cross['mean_crossorgan_cosine'].mean():.3f}\n")
print(f"B. mean Bray-Curtis DISSIMILARITY between niches: {mean_bc:.3f} "
      f"(0=identical, 1=no shared cells)")

# ── companion figure: 8x8 Bray-Curtis heatmap + cross-organ cosine bars ─
order = [niches[i] for i in leaves_list(linkage(squareform(bc.values), "average"))]
bc_o = bc.loc[order, order]
nice = lambda s: s.replace("_", " ")

fig, (axh, axb) = plt.subplots(1, 2, figsize=(11, 5),
                               gridspec_kw={"width_ratios": [1.25, 1]})

im = axh.imshow(bc_o.values, cmap="magma_r", vmin=0, vmax=1)
axh.set_xticks(range(len(order))); axh.set_yticks(range(len(order)))
axh.set_xticklabels([nice(n) for n in order], rotation=45, ha="right", fontsize=7)
axh.set_yticklabels([nice(n) for n in order], fontsize=7)
for i in range(len(order)):
    for j in range(len(order)):
        v = bc_o.values[i, j]
        axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                 color="white" if v > 0.55 else "black")
axh.set_title(f"Bray-Curtis dissimilarity between niches\n(mean {mean_bc:.2f})",
              fontsize=9)
fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04, label="dissimilarity")

cb = cross.dropna(subset=["mean_crossorgan_cosine"]).sort_values(
    "mean_crossorgan_cosine")
colors = ["#4c9f70" if n == 3 else "#d98b5f" for n in cb["n_organs"]]
axb.barh([nice(n) for n in cb["niche"]], cb["mean_crossorgan_cosine"],
         color=colors, edgecolor="white")
for y, (v, n) in enumerate(zip(cb["mean_crossorgan_cosine"], cb["n_organs"])):
    axb.text(v + 0.01, y, f"{v:.2f} (n={int(n)})", va="center", fontsize=7)
axb.set_xlim(0, 1); axb.set_xlabel("cross-organ cosine similarity", fontsize=8)
axb.set_title("Per-niche conservation across organs\n"
              "green=3 organs, orange=2 organs (partial)", fontsize=9)
axb.tick_params(labelsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_niche_structure_similarity.{ext}",
                dpi=200, bbox_inches="tight")
print(f"\nFigure: figures/fig_niche_structure_similarity.png/pdf")

# ── self-check ────────────────────────────────────────────────────────
assert (cross["mean_crossorgan_cosine"].dropna().between(0, 1)).all()
assert 0 <= mean_bc <= 1
assert cosine(avg.loc[niches[0]].values, avg.loc[niches[0]].values) < 1e-9  # self-sim=1
print("\nOK: metrics in [0,1], self-similarity=1.")
