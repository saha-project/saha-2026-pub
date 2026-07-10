"""
S9 — Cross-organ donor-level breakdown
Panel a: Per-donor JCHAIN coupling per isotype per organ (strip plot)
Panel b: Per-donor IgG pure fraction in GALT-interior IGH+ cells per organ
Panel c: Per-donor egress score across niches
"""
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
ORGANS = ["APE", "ILE", "COL"]
DONORS = ["EC01", "SJ01", "TAP03"]
DONOR_MARKERS = {"EC01": "o", "SJ01": "s", "TAP03": "^"}
DONOR_COLORS = {"EC01": "#555555", "SJ01": "#999999", "TAP03": "#CCCCCC"}

BASE = "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/Across-GI"
FIG_DIR = BASE + "/figures"
DATA_DIR = BASE + "/ig_validation"

jchain_iso = pd.read_csv(DATA_DIR + "/jchain_isotype_ratio_per_section.csv")
per_unit = pd.read_csv(DATA_DIR + "/galt_interior_composite_per_unit.csv")
niche_jchain = pd.read_csv(DATA_DIR + "/jchain_niche_conditional_per_section.csv")

# Extract donor from batch column (section = e.g. APE_A9_1 — use batch from per_unit)
# jchain_iso has 'section' column — parse donor from it via per_unit mapping
# per_unit has: organ, sample, batch, unit_id
# section in jchain_iso is like APE_A9_1 → match to sample in per_unit
unit_to_batch = per_unit[["organ", "sample", "batch"]].drop_duplicates()
unit_to_batch = unit_to_batch.rename(columns={"sample": "section"})

jchain_iso = jchain_iso.merge(unit_to_batch, on=["organ", "section"], how="left")
jchain_iso["donor"] = jchain_iso["batch"]

niche_jchain = niche_jchain.merge(unit_to_batch, on=["organ", "section"], how="left")
niche_jchain["donor"] = niche_jchain["batch"]

# Panel b: IgG pure fraction per donor per organ (from jchain_iso: pure IgG = IGHG1+IGHG2)
igg_df = jchain_iso[jchain_iso["isotype"].isin(["IGHG1", "IGHG2"])].copy()
igg_by_section = igg_df.groupby(["organ", "section", "donor"])["frac_jchain_in_iso"].mean().reset_index()
igg_by_section.rename(columns={"frac_jchain_in_iso": "igg_frac"}, inplace=True)

# Panel c: Egress score per niche per donor (from niche_jchain: niche = GALT-interior as proxy)
# Use GALT-interior IGHA1 frac_jchain as egress proxy
egress_niche = niche_jchain[niche_jchain["isotype"] == "IGHA1"].copy()
egress_niche = egress_niche[egress_niche["niche"] == "GALT-interior"]

# Also pull egress score from per_unit (direct)
egress_pu = per_unit[per_unit["score_name"] == "Egress"].copy()
# per_unit has batch directly as donor
egress_pu["donor"] = egress_pu["batch"]

# --- Figure ---
fig = plt.figure(figsize=(16, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

ISOTYPES = ["IGHA1", "IGHG1", "IGHG2", "IGHM"]

# --- Panel a: JCHAIN coupling per isotype per organ, per donor ---
ax_a = fig.add_subplot(gs[0])

jchain_agg = jchain_iso.groupby(["organ", "isotype", "donor"])["frac_jchain_in_iso"].mean().reset_index()

x_pos = 0
xticks = []
xlabels = []

for iso_i, iso in enumerate(ISOTYPES):
    for org_i, organ in enumerate(ORGANS):
        sub = jchain_agg[(jchain_agg["isotype"] == iso) & (jchain_agg["organ"] == organ)]
        xc = x_pos + org_i * 0.35
        for _, row in sub.iterrows():
            donor = row["donor"]
            ax_a.scatter(xc, row["frac_jchain_in_iso"],
                         color=ORGAN_COLORS[organ],
                         marker=DONOR_MARKERS.get(donor, "o"),
                         s=45, zorder=3, alpha=0.9,
                         edgecolors="white", linewidths=0.4)
        # mean line
        if len(sub) > 0:
            ax_a.hlines(sub["frac_jchain_in_iso"].mean(),
                        xc - 0.12, xc + 0.12, lw=1.5,
                        colors=ORGAN_COLORS[organ], zorder=4)
        xticks.append(xc)
        xlabels.append(f"{organ}\n{iso}" if org_i == 1 else "")

    x_pos += len(ORGANS) * 0.35 + 0.25

ax_a.set_xticks([])
# Add isotype labels centered under each group
group_centers = []
x_pos = 0
for iso in ISOTYPES:
    gc = x_pos + (len(ORGANS) - 1) * 0.35 / 2
    group_centers.append(gc)
    x_pos += len(ORGANS) * 0.35 + 0.25

ax_a.set_xticks(group_centers)
ax_a.set_xticklabels(ISOTYPES, fontsize=8)
ax_a.set_ylabel("JCHAIN coupling\n(frac. JCHAIN+ in isotype+)", fontsize=8)
ax_a.set_title("a  JCHAIN coupling per isotype\nper organ × donor", fontsize=9, fontweight="bold")

# Legend for organs (colored bars) and donors (markers)
from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], marker=DONOR_MARKERS[d], color="gray",
                        markerfacecolor=DONOR_COLORS[d], markersize=6,
                        label=d, linewidth=0) for d in DONORS]
ax_a.legend(handles=legend_elems, fontsize=7, loc="upper right", frameon=False)

# --- Panel b: IgG fraction per donor per organ ---
ax_b = fig.add_subplot(gs[1])

for org_i, organ in enumerate(ORGANS):
    sub = igg_by_section[igg_by_section["organ"] == organ]
    for _, row in sub.iterrows():
        donor = row["donor"]
        ax_b.scatter(org_i + 1, row["igg_frac"],
                     color=ORGAN_COLORS[organ],
                     marker=DONOR_MARKERS.get(donor, "o"),
                     s=55, zorder=3, alpha=0.9,
                     edgecolors="white", linewidths=0.5)
    if len(sub) > 0:
        ax_b.hlines(sub["igg_frac"].mean(),
                    org_i + 0.7, org_i + 1.3, lw=2.0,
                    colors=ORGAN_COLORS[organ], zorder=4)

ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(ORGANS, fontsize=9)
ax_b.set_ylabel("IgG JCHAIN coupling fraction", fontsize=8)
ax_b.set_title("b  IgG class-switch\nper donor × organ", fontsize=9, fontweight="bold")

legend_elems_b = [Line2D([0], [0], marker=DONOR_MARKERS[d], color="gray",
                          markerfacecolor=DONOR_COLORS[d], markersize=6,
                          label=d, linewidth=0) for d in DONORS]
ax_b.legend(handles=legend_elems_b, fontsize=7, loc="upper right", frameon=False)

# --- Panel c: Egress score per niche per donor ---
ax_c = fig.add_subplot(gs[2])

NICHES = ["LP-distal", "GALT-edge", "GALT-interior"]
NICHE_X = {n: i + 1 for i, n in enumerate(NICHES)}

# Use niche_jchain IGHA1 frac as egress proxy
for organ in ORGANS:
    sub_o = egress_niche[egress_niche["organ"] == organ]
    for _, row in sub_o.iterrows():
        donor = row["donor"]
        x = NICHE_X.get(row["niche"], 3)
        ax_c.scatter(x, row["frac_jchain_in_iso"],
                     color=ORGAN_COLORS[organ],
                     marker=DONOR_MARKERS.get(donor, "o"),
                     s=45, zorder=3, alpha=0.9,
                     edgecolors="white", linewidths=0.4)

# Also overlay mean lines per organ across GALT-interior
for organ in ORGANS:
    sub_all = niche_jchain[(niche_jchain["organ"] == organ) & (niche_jchain["isotype"] == "IGHA1")]
    for niche_label, sub_n in sub_all.groupby("niche"):
        x = NICHE_X.get(niche_label, None)
        if x is None:
            continue
        ax_c.hlines(sub_n["frac_jchain_in_iso"].mean(),
                    x - 0.2, x + 0.2, lw=1.5,
                    colors=ORGAN_COLORS[organ], zorder=4)

ax_c.set_xticks(list(NICHE_X.values()))
ax_c.set_xticklabels([n.replace("-", "\n") for n in NICHES], fontsize=7)
ax_c.set_ylabel("IgA JCHAIN coupling\n(egress proxy)", fontsize=8)
ax_c.set_title("c  Egress gradient per niche\nper donor × organ", fontsize=9, fontweight="bold")

legend_elems_c = [Line2D([0], [0], color=ORGAN_COLORS[o], lw=2, label=o) for o in ORGANS]
ax_c.legend(handles=legend_elems_c, fontsize=7, loc="upper right", frameon=False)

fig.suptitle("Supplementary Figure S9 — Cross-organ donor-level breakdown",
             fontsize=11, y=1.01, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(
        FIG_DIR + f"/fig_S9_donor_breakdown.{ext}",
        dpi=600, bbox_inches="tight", facecolor="white"
    )

print("S9 done.")
