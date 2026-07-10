"""S3 — IgG class-switch validation (4 panels)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

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

per_donor   = pd.read_csv(f"{DATA}/snpatho_plasma_per_donor_isotype.csv")
homing      = pd.read_csv(f"{DATA}/snpatho_plasma_homing_receptors.csv")
iso_niche   = pd.read_csv(f"{DATA}/spatial_isotype_per_unit_niche.csv")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.45)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ── panel a: snPATHO per-donor isotype fractions heatmap ─────────────────────
iso_cols = [c for c in per_donor.columns if c.startswith("frac_")]
iso_labels = [c.replace("frac_", "") for c in iso_cols]

# sort donors: APE first then ILE
df_a = per_donor.sort_values(["organ", "donor"]).reset_index(drop=True)
mat = df_a[iso_cols].values.astype(float)
row_labels = [f"{row['organ']}-D{row['donor']}" for _, row in df_a.iterrows()]

im = ax_a.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax_a.set_xticks(range(len(iso_labels)))
ax_a.set_xticklabels(iso_labels, fontsize=8, rotation=40, ha="right")
ax_a.set_yticks(range(len(row_labels)))
ax_a.set_yticklabels(row_labels, fontsize=7)
ax_a.set_title("snPATHO plasma: per-donor isotype fractions", fontsize=10)
plt.colorbar(im, ax=ax_a, shrink=0.7, label="Fraction positive")

# organ separators
organ_order = df_a["organ"].values
for i in range(1, len(organ_order)):
    if organ_order[i] != organ_order[i-1]:
        ax_a.axhline(i - 0.5, color="white", lw=1.5)

# ── panel b: homing receptor bar chart APE vs ILE ────────────────────────────
# filter to single-gene rows (no semicolons in genes)
single = homing[~homing["genes"].str.contains(";", na=False)].copy()
single = single.sort_values("ape_mean", ascending=True)

y = np.arange(len(single))
width = 0.35
ax_b.barh(y - width/2, single["ape_mean"], height=width, color=ORGAN_COLORS["APE"], label="APE")
ax_b.barh(y + width/2, single["ile_mean"], height=width, color=ORGAN_COLORS["ILE"], label="ILE")
ax_b.set_yticks(y)
ax_b.set_yticklabels(single["genes"], fontsize=8)
ax_b.set_xlabel("Mean expression", fontsize=9)
ax_b.set_title("snPATHO homing receptor expression (single genes)", fontsize=10)
ax_b.axvline(0, color="black", lw=0.6)
ax_b.legend(fontsize=8, frameon=False)

# ── panel c: spatial pure-IgG / pure-IgA / Both stacked bars per niche × organ ─
NICHES = ["LP-distal", "GALT-edge", "GALT-interior"]
ORGANS = ["APE", "COL", "ILE"]

niche_colors = {
    "frac_pure_igg": "#C0392B",
    "frac_both":     "#8E44AD",
    "frac_pure_iga": "#2980B9",
}
niche_labels = {"frac_pure_igg": "Pure IgG", "frac_both": "Both", "frac_pure_iga": "Pure IgA"}

grp = iso_niche.groupby(["organ", "niche"])[list(niche_colors.keys())].mean().reset_index()

n_groups = len(ORGANS) * len(NICHES)
x = np.arange(n_groups)
x_ticks, x_tick_labels = [], []

xi = 0
for org_i, org in enumerate(ORGANS):
    for niche in NICHES:
        sub = grp[(grp["organ"] == org) & (grp["niche"] == niche)]
        if sub.empty:
            xi += 1
            continue
        row = sub.iloc[0]
        bottom = 0
        for col in ["frac_pure_igg", "frac_both", "frac_pure_iga"]:
            val = row[col]
            ax_c.bar(xi, val, bottom=bottom, color=niche_colors[col],
                     width=0.7, linewidth=0, label=niche_labels[col] if xi == 0 else "")
            bottom += val
        x_ticks.append(xi)
        x_tick_labels.append(niche.replace("GALT-", "G-").replace("LP-distal", "LP"))
        xi += 1
    if org_i < len(ORGANS) - 1:
        ax_c.axvline(xi - 0.5, color="gray", lw=0.6, ls="--", alpha=0.5)

ax_c.set_xticks(x_ticks)
ax_c.set_xticklabels(x_tick_labels, fontsize=7, rotation=30, ha="right")

# organ labels below
for oi, org in enumerate(ORGANS):
    mid = oi * len(NICHES) + len(NICHES)/2 - 0.5
    ax_c.text(mid, -0.12, org, transform=ax_c.get_xaxis_transform(),
              ha="center", fontsize=9, fontweight="bold", color=ORGAN_COLORS[org])

ax_c.set_ylabel("Mean fraction", fontsize=9)
ax_c.set_title("IgG/IgA spatial classification by niche × organ", fontsize=10)
handles, labels = ax_c.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_c.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=False)

# ── panel d: pure-IgG fraction heatmap rows=units columns=niches ─────────────
pivot_d = iso_niche.pivot_table(index="unit_id", columns="niche",
                                 values="frac_pure_igg", aggfunc="mean")
# sort rows by organ
unit_organ = iso_niche[["unit_id","organ"]].drop_duplicates().set_index("unit_id")
pivot_d = pivot_d.loc[unit_organ.index]
pivot_d = pivot_d.reindex(columns=NICHES)

# color rows by organ
row_organs = unit_organ.loc[pivot_d.index, "organ"].values

im2 = ax_d.imshow(pivot_d.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.8)
ax_d.set_xticks(range(len(NICHES)))
ax_d.set_xticklabels(NICHES, fontsize=8, rotation=30, ha="right")
ax_d.set_yticks(range(len(pivot_d)))
ax_d.set_yticklabels(pivot_d.index, fontsize=5)
ax_d.set_title("Pure-IgG fraction per unit × niche", fontsize=10)
plt.colorbar(im2, ax=ax_d, shrink=0.7, label="Fraction pure IgG")

# color ytick labels by organ
for tick, organ in zip(ax_d.get_yticklabels(), row_organs):
    tick.set_color(ORGAN_COLORS.get(organ, "black"))

# organ separator lines
for i in range(1, len(row_organs)):
    if row_organs[i] != row_organs[i-1]:
        ax_d.axhline(i - 0.5, color="white", lw=1.5)

# ── panel labels ─────────────────────────────────────────────────────────────
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], list("abcd")):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

fig.patch.set_facecolor("white")
fig.savefig(f"{OUT}/fig_S3_igg_classswitch.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_S3_igg_classswitch.pdf", bbox_inches="tight", facecolor="white")
print("S3 saved.")
