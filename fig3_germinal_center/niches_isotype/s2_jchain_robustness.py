"""S2 — JCHAIN axis robustness (6 panels: a–f)"""
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

per_section  = pd.read_csv(f"{DATA}/jchain_qc_per_section.csv")
lib_size     = pd.read_csv(f"{DATA}/jchain_qc_library_size.csv")
downsampled  = pd.read_csv(f"{DATA}/jchain_qc_downsampled.csv")
sahacr       = pd.read_csv(f"{DATA}/jchain_qc_sahacr_replication.csv")
pairwise     = pd.read_csv(f"{DATA}/jchain_mixedeffects_pairwise.csv")
sensitivity  = pd.read_csv(f"{DATA}/jchain_mixedeffects_sensitivity.csv")

ISOTYPES = ["IGHA1", "IGHA2", "IGHG1", "IGHG2", "IGHM", "IGHD"]
ORGANS   = ["APE", "COL", "ILE"]

fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.42)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, 0])
ax_f = fig.add_subplot(gs[2, 1])

# ── panel a: per-section JCHAIN+ fraction strip plot (organ × isotype) ────────
isos_present = sorted(per_section["isotype"].unique())
n_iso = len(isos_present)
n_org = len(ORGANS)
group_width = 0.8
bar_w = group_width / n_org

for ii, iso in enumerate(isos_present):
    for oi, org in enumerate(ORGANS):
        sub = per_section[(per_section["isotype"] == iso) & (per_section["organ"] == org)]
        if sub.empty:
            continue
        x_center = ii + (oi - (n_org-1)/2) * bar_w
        ys = sub["frac_jchain_in_iso"].values
        ax_a.scatter([x_center]*len(ys), ys, color=ORGAN_COLORS[org],
                     s=18, alpha=0.75, zorder=3)
        ax_a.hlines(np.median(ys), x_center - bar_w*0.4, x_center + bar_w*0.4,
                    color=ORGAN_COLORS[org], lw=1.5, zorder=4)

ax_a.set_xticks(range(n_iso))
ax_a.set_xticklabels(isos_present, fontsize=8, rotation=30, ha="right")
ax_a.set_ylabel("JCHAIN+ fraction in isotype", fontsize=9)
ax_a.set_title("Per-section JCHAIN+ fraction by organ × isotype", fontsize=10)
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=ORGAN_COLORS[o],
              markersize=6, label=o) for o in ORGANS]
ax_a.legend(handles=leg, fontsize=8, frameon=False)

# ── panel b: library-size profile ─────────────────────────────────────────────
groups = ["all", "IGH+"]
grp_colors = {"all": "#BBBBBB", "IGH+": "#444444"}
metrics = ["med_total_counts", "med_n_genes"]
metric_labels = ["Median total counts", "Median n genes"]
n_org = len(ORGANS)
x_grp = np.arange(n_org)
width = 0.35

ax_b2 = ax_b.twinx()
ax_b.spines["right"].set_visible(False)
ax_b2.spines["top"].set_visible(False)

for gi, grp in enumerate(groups):
    sub = lib_size[lib_size["group"] == grp].set_index("organ")
    offset = (gi - 0.5) * width
    vals_counts = [sub.loc[o, "med_total_counts"] if o in sub.index else 0 for o in ORGANS]
    vals_genes  = [sub.loc[o, "med_n_genes"]      if o in sub.index else 0 for o in ORGANS]
    if gi == 0:
        ax_b.bar(x_grp + offset, vals_counts, width=width*0.9,
                 color=[ORGAN_COLORS[o] for o in ORGANS], alpha=0.5,
                 label="all – counts", linewidth=0)
        ax_b2.plot(x_grp + offset, vals_genes, 's--', color="gray",
                   markersize=5, label="all – n_genes")
    else:
        ax_b.bar(x_grp + offset, vals_counts, width=width*0.9,
                 color=[ORGAN_COLORS[o] for o in ORGANS], alpha=1.0,
                 label="IGH+ – counts", linewidth=0)
        ax_b2.plot(x_grp + offset, vals_genes, 's-', color="black",
                   markersize=5, label="IGH+ – n_genes")

ax_b.set_xticks(x_grp)
ax_b.set_xticklabels(ORGANS, fontsize=10)
ax_b.set_ylabel("Median total counts", fontsize=9)
ax_b2.set_ylabel("Median n genes", fontsize=9)
ax_b.set_title("Library-size profile: all vs IGH+ cells", fontsize=10)
lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False, loc="upper left")

# ── panel c: downsampling heatmap ────────────────────────────────────────────
# pivot: rows = isotype, cols = target_depth, values = frac_jchain_in_iso per organ (mean)
ds = downsampled.copy()
isos_ds = sorted(ds["isotype"].unique())
depths   = sorted(ds["target_depth"].unique())

# mean over organs
pivot = ds.groupby(["isotype", "target_depth"])["frac_jchain_in_iso"].mean().unstack("target_depth")
pivot = pivot.reindex(isos_ds)
im = ax_c.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
ax_c.set_xticks(range(len(depths)))
ax_c.set_xticklabels([str(d) for d in pivot.columns], fontsize=8, rotation=45, ha="right")
ax_c.set_yticks(range(len(isos_ds)))
ax_c.set_yticklabels(pivot.index, fontsize=8)
ax_c.set_xlabel("Target depth (downsampled)", fontsize=9)
ax_c.set_title("JCHAIN+ fraction after downsampling (mean over organs)", fontsize=10)
plt.colorbar(im, ax=ax_c, shrink=0.8, label="JCHAIN+ fraction")

# ── panel d: SAHA_CR replication heatmap ─────────────────────────────────────
pivot_cr = sahacr.pivot_table(index="section", columns="isotype",
                               values="frac_jchain_in_iso", aggfunc="mean")
im2 = ax_d.imshow(pivot_cr.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
ax_d.set_xticks(range(len(pivot_cr.columns)))
ax_d.set_xticklabels(pivot_cr.columns, fontsize=8, rotation=35, ha="right")
ax_d.set_yticks(range(len(pivot_cr.index)))
ax_d.set_yticklabels(pivot_cr.index, fontsize=7)
ax_d.set_title("SAHA_CR replication: JCHAIN+ fractions", fontsize=10)
plt.colorbar(im2, ax=ax_d, shrink=0.8, label="JCHAIN+ fraction")

# ── panel e: forest plot mixed-effects ORs ────────────────────────────────────
full = pairwise[pairwise["dataset"] == "full"].copy() if "dataset" in pairwise.columns else pairwise.copy()
isos_f    = sorted(full["isotype"].unique())
contrasts = sorted(full["contrast"].unique())
contrast_colors = {"APE_vs_ILE": "#3D8B8B", "COL_vs_ILE": "#1B3A5C", "COL_vs_APE": "#8B5C3A"}

y_ticks = []
y_labels = []
y = 0
for iso in isos_f:
    sub = full[full["isotype"] == iso]
    for _, row in sub.iterrows():
        color = contrast_colors.get(row["contrast"], "#666666")
        ax_e.plot([row["OR_CI_lo"], row["OR_CI_hi"]], [y, y], color=color, lw=1.5)
        ax_e.scatter([row["OR"]], [y], color=color, s=30, zorder=3)
        y_ticks.append(y)
        y_labels.append(f"{iso} {row['contrast']}")
        y += 1
    y += 0.5  # gap between isotypes

ax_e.axvline(1.0, color="black", lw=0.8, ls="--")
ax_e.set_yticks(y_ticks)
ax_e.set_yticklabels(y_labels, fontsize=6)
ax_e.set_xlabel("Odds Ratio (95% CI)", fontsize=9)
ax_e.set_title("Mixed-effects ORs: full dataset", fontsize=10)
from matplotlib.lines import Line2D
leg_e = [Line2D([0],[0], color=v, lw=2, label=k) for k, v in contrast_colors.items()]
ax_e.legend(handles=leg_e, fontsize=7, frameon=False)

# ── panel f: sensitivity vs full comparison ───────────────────────────────────
sens = sensitivity.copy()
merged = full.merge(sens, on=["isotype", "contrast"], suffixes=("_full", "_sens"))
isos_s = sorted(merged["isotype"].unique())
n = len(merged)
xs = np.arange(n)
ax_f.scatter(merged["OR_full"], merged["OR_sens"],
             c=[ORGAN_COLORS.get(c.split("_")[0], "#666666") for c in merged["contrast"]],
             s=30, alpha=0.8)
lims = [min(merged[["OR_full","OR_sens"]].min()), max(merged[["OR_full","OR_sens"]].max())]
ax_f.plot(lims, lims, "k--", lw=0.8, label="identity")
ax_f.set_xlabel("OR (full dataset)", fontsize=9)
ax_f.set_ylabel("OR (sensitivity: COL_A1_2/A1_6 dropped)", fontsize=9)
ax_f.set_title("Sensitivity analysis: full vs reduced model ORs", fontsize=10)
ax_f.legend(fontsize=8, frameon=False)

# ── panel labels ─────────────────────────────────────────────────────────────
for ax, letter in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("abcdef")):
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

fig.patch.set_facecolor("white")
fig.savefig(f"{OUT}/fig_S2_jchain_robustness.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_S2_jchain_robustness.pdf", bbox_inches="tight", facecolor="white")
print("S2 saved.")
