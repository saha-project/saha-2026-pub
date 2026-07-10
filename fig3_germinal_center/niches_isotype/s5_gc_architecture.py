"""S5 — Spatial GC architecture (5 panels — NEGATIVE FINDING)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8

ORGAN_COLORS = {"APE": "#3D8B8B", "COL": "#1B3A5C", "ILE": "#8B3A3A"}

DATA = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI/ig_validation"
OUT  = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI/figures"

summary  = pd.read_csv(f"{DATA}/spatial_atomic_unit_summary.csv")
gc_unit  = pd.read_csv(f"{DATA}/gc_spatial_segregation_per_unit.csv")
gc_organ = pd.read_csv(f"{DATA}/gc_spatial_segregation_per_organ.csv")

ORGANS = ["APE", "COL", "ILE"]

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.42)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, :])   # full bottom row for text panel

# ── panel a: bivariate Moran's I per unit, strip per organ ───────────────────
for oi, org in enumerate(ORGANS):
    sub = summary[summary["organ"] == org]["biv_moran_dz_lz"].dropna()
    if sub.empty:
        continue
    jitter = np.random.RandomState(42).uniform(-0.18, 0.18, len(sub))
    ax_a.scatter(oi + jitter, sub, color=ORGAN_COLORS[org], s=22, alpha=0.75, zorder=3)
    ax_a.hlines(sub.median(), oi - 0.25, oi + 0.25, color=ORGAN_COLORS[org], lw=2)

# pairwise Kruskal-Wallis
groups = [summary[summary["organ"] == o]["biv_moran_dz_lz"].dropna().values for o in ORGANS]
if all(len(g) > 1 for g in groups):
    H, p_kw = stats.kruskal(*groups)
    ax_a.text(0.97, 0.97, f"KW p={p_kw:.3f}", transform=ax_a.transAxes,
              ha="right", va="top", fontsize=8, color="gray")

ax_a.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
ax_a.set_xticks(range(len(ORGANS)))
ax_a.set_xticklabels(ORGANS, fontsize=11)
ax_a.set_ylabel("Bivariate Moran's I (DZ vs LZ)", fontsize=9)
ax_a.set_title("DZ/LZ spatial co-segregation per atomic unit", fontsize=10)

# -- panel b: Ripley's L12(r)-r curves per organ --
radii_cols = ["L12_minus_r_50um", "L12_minus_r_100um", "L12_minus_r_200um", "L12_minus_r_500um"]
radii_vals = [50, 100, 200, 500]

for org in ORGANS:
    sub = gc_unit[gc_unit["organ"] == org][radii_cols].dropna()
    if sub.empty:
        continue
    means = sub.mean().values
    sems  = sub.sem().values
    ax_b.plot(radii_vals, means, color=ORGAN_COLORS[org], lw=1.8, label=org, marker="o", markersize=5)
    ax_b.fill_between(radii_vals, means - sems, means + sems,
                      color=ORGAN_COLORS[org], alpha=0.15)

ax_b.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
ax_b.set_xlabel("Radius r (µm)", fontsize=9)
ax_b.set_ylabel("L12(r) - r (mean +/- SEM)", fontsize=9)
ax_b.set_title("Ripley's L12(r)-r: DZ-LZ co-clustering", fontsize=10)
ax_b.legend(fontsize=8, frameon=False)

# ── panel c: Morisita-Horn overlap violin + strip ─────────────────────────────
for oi, org in enumerate(ORGANS):
    sub = gc_unit[gc_unit["organ"] == org]["morisita_horn"].dropna()
    if sub.empty:
        continue
    if len(sub) >= 3:
        parts = ax_c.violinplot(sub, positions=[oi], widths=0.5,
                                 showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(ORGAN_COLORS[org])
            pc.set_alpha(0.5)
            pc.set_edgecolor("none")
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
    jitter = np.random.RandomState(7).uniform(-0.1, 0.1, len(sub))
    ax_c.scatter(oi + jitter, sub, color=ORGAN_COLORS[org], s=18, alpha=0.7, zorder=3)

# pairwise Mann-Whitney with annotation
organ_data = {o: gc_unit[gc_unit["organ"] == o]["morisita_horn"].dropna().values for o in ORGANS}
pairs = [("APE","ILE"), ("COL","ILE"), ("APE","COL")]
ymax = gc_unit["morisita_horn"].max() * 1.05
step = ymax * 0.08
for pi, (o1, o2) in enumerate(pairs):
    d1, d2 = organ_data.get(o1, []), organ_data.get(o2, [])
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        x1, x2 = ORGANS.index(o1), ORGANS.index(o2)
        y = ymax + pi * step
        ax_c.plot([x1, x2], [y, y], color="gray", lw=0.8)
        ax_c.text((x1+x2)/2, y, star, ha="center", va="bottom", fontsize=8)

ax_c.set_xticks(range(len(ORGANS)))
ax_c.set_xticklabels(ORGANS, fontsize=11)
ax_c.set_ylabel("Morisita-Horn overlap index", fontsize=9)
ax_c.set_title("DZ/LZ Morisita-Horn overlap (0=segregated, 1=mixed)", fontsize=10)

# ── panel d: NN-state ratio violin + strip ────────────────────────────────────
for oi, org in enumerate(ORGANS):
    sub = gc_unit[gc_unit["organ"] == org]["nn_lz_fraction"].dropna()
    if sub.empty:
        continue
    if len(sub) >= 3:
        parts = ax_d.violinplot(sub, positions=[oi], widths=0.5,
                                 showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(ORGAN_COLORS[org])
            pc.set_alpha(0.5)
            pc.set_edgecolor("none")
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
    jitter = np.random.RandomState(13).uniform(-0.1, 0.1, len(sub))
    ax_d.scatter(oi + jitter, sub, color=ORGAN_COLORS[org], s=18, alpha=0.7, zorder=3)

ax_d.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5, label="LZ=DZ (0.5)")
ax_d.set_xticks(range(len(ORGANS)))
ax_d.set_xticklabels(ORGANS, fontsize=11)
ax_d.set_ylabel("NN LZ fraction", fontsize=9)
ax_d.set_title("Nearest-neighbor LZ state ratio per unit", fontsize=10)
ax_d.legend(fontsize=8, frameon=False)

# ── panel e: caption text panel (negative finding) ───────────────────────────
ax_e.axis("off")
caption = (
    "Healthy human gastrointestinal organs do not exhibit classical Peyer's-patch\n"
    "dark-zone / light-zone (DZ/LZ) polarization at CosMx spatial transcriptomics resolution.\n\n"
    "Panels a-d show bivariate Moran's I, Ripley's L12(r)-r cross-function, Morisita-Horn\n"
    "overlap index, and nearest-neighbor LZ fraction across 37 atomic units spanning\n"
    "appendix (APE), colon (COL), and ileum (ILE). None of the metrics show statistically\n"
    "significant organ-level DZ/LZ spatial segregation. This is consistent with the absence\n"
    "of organised secondary lymphoid follicle architecture in healthy adult GI mucosa\n"
    "at the resolution captured by CosMx (~50 µm cell-to-cell distance).\n\n"
    "Negative result reported for transparency; GC architecture analysis was pre-registered\n"
    "as a secondary aim of this atlas study."
)
ax_e.text(0.05, 0.90, caption, transform=ax_e.transAxes,
          fontsize=10, va="top", ha="left",
          bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor="#CCCCCC", lw=0.8),
          fontfamily="sans-serif")
ax_e.set_title("Supplementary note: negative DZ/LZ finding", fontsize=10, loc="left")

# ── panel labels ─────────────────────────────────────────────────────────────
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d, ax_e], list("abcde")):
    ax.text(-0.04, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

fig.patch.set_facecolor("white")
fig.savefig(f"{OUT}/fig_S5_gc_architecture.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_S5_gc_architecture.pdf", bbox_inches="tight", facecolor="white")
print("S5 saved.")
