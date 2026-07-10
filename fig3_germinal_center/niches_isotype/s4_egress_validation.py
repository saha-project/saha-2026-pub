"""S4 — Egress validation and detail (3 panels)"""
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

egress     = pd.read_csv(f"{DATA}/plasma_egress_per_unit_niche.csv")
commitment = pd.read_csv(f"{DATA}/snpatho_plasma_commitment_no_jchain.csv")
summary    = pd.read_csv(f"{DATA}/spatial_atomic_unit_summary.csv")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.45)

ax_a = fig.add_subplot(gs[0, :])   # full top row
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])

ORGANS = ["APE", "COL", "ILE"]
NICHES = ["LP-distal", "GALT-edge", "GALT-interior"]
niche_offsets = {"LP-distal": -0.28, "GALT-edge": 0.0, "GALT-interior": 0.28}
niche_markers = {"LP-distal": "o", "GALT-edge": "s", "GALT-interior": "^"}
niche_alphas  = {"LP-distal": 0.5, "GALT-edge": 0.7, "GALT-interior": 0.9}

# ── panel a: egress score violin + strip per organ × niche ───────────────────
x_positions = {org: i for i, org in enumerate(ORGANS)}
violin_width = 0.22

for niche in NICHES:
    for org in ORGANS:
        sub = egress[(egress["organ"] == org) & (egress["niche"] == niche)]["mean_egress"].dropna()
        if sub.empty:
            continue
        xpos = x_positions[org] + niche_offsets[niche]
        # violin
        if len(sub) >= 3:
            parts = ax_a.violinplot(sub, positions=[xpos], widths=violin_width,
                                     showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(ORGAN_COLORS[org])
                pc.set_alpha(niche_alphas[niche])
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.5)
        # strip
        jitter = np.random.RandomState(42).uniform(-0.04, 0.04, len(sub))
        ax_a.scatter(xpos + jitter, sub, color=ORGAN_COLORS[org],
                     s=12, alpha=0.6, marker=niche_markers[niche], zorder=3)

ax_a.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
ax_a.set_xticks(range(len(ORGANS)))
ax_a.set_xticklabels(ORGANS, fontsize=11)
ax_a.set_ylabel("Mean egress score", fontsize=10)
ax_a.set_title("Plasma egress score per atomic unit × niche", fontsize=11)
from matplotlib.lines import Line2D
leg_niche = [Line2D([0],[0], marker=niche_markers[n], color='gray',
                    linestyle='none', markersize=6, label=n, alpha=0.8) for n in NICHES]
ax_a.legend(handles=leg_niche, fontsize=8, frameon=False, loc="upper right")

# ── panel b: snPATHO commitment bar chart APE vs ILE ─────────────────────────
scores = commitment["score"].tolist()
n = len(scores)
x = np.arange(n)
width = 0.35

ape_vals = commitment["ape_mean"].values
ile_vals = commitment["ile_mean"].values

bars_ape = ax_b.bar(x - width/2, ape_vals, width, color=ORGAN_COLORS["APE"], label="APE", linewidth=0)
bars_ile = ax_b.bar(x + width/2, ile_vals, width, color=ORGAN_COLORS["ILE"], label="ILE", linewidth=0)

# add significance stars
for i, row in commitment.iterrows():
    p = row["p"]
    if p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    elif p < 0.05:
        star = "*"
    else:
        star = "ns"
    ymax = max(row["ape_mean"], row["ile_mean"])
    ymax = max(ymax, 0)
    ax_b.text(i, ymax + abs(ymax)*0.05 + 1e-9, star,
              ha="center", va="bottom", fontsize=8, color="black")

ax_b.axhline(0, color="black", lw=0.6)
ax_b.set_xticks(x)
score_labels = [s.replace("_", "\n") for s in scores]
ax_b.set_xticklabels(score_labels, fontsize=7, ha="center")
ax_b.set_ylabel("Mean composite score", fontsize=9)
ax_b.set_title("snPATHO plasma commitment: with vs without JCHAIN", fontsize=10)
ax_b.legend(fontsize=8, frameon=False)

# ── panel c: per-batch MEI gradient (APE > ILE > COL within SJ01) ────────────
# filter to SJ01 batch; show mei_gradient per organ
sj01 = summary[summary["batch"] == "SJ01"].copy()
non_sj01 = summary[summary["batch"] != "SJ01"].copy()

for batch_df, marker, alpha, zorder in [(non_sj01, "o", 0.35, 2), (sj01, "s", 0.9, 3)]:
    for org in ORGANS:
        sub = batch_df[batch_df["organ"] == org]["mei_gradient"].dropna()
        if sub.empty:
            continue
        xpos = ORGANS.index(org)
        jitter = np.random.RandomState(99).uniform(-0.15, 0.15, len(sub))
        lbl = f"{org} SJ01" if alpha > 0.5 else None
        ax_c.scatter(xpos + jitter, sub, color=ORGAN_COLORS[org],
                     s=18, alpha=alpha, marker=marker, zorder=zorder)

# SJ01 organ medians
for org in ORGANS:
    sub = sj01[sj01["organ"] == org]["mei_gradient"].dropna()
    if sub.empty:
        continue
    xpos = ORGANS.index(org)
    ax_c.hlines(sub.median(), xpos - 0.25, xpos + 0.25,
                color=ORGAN_COLORS[org], lw=2.0, zorder=4)

ax_c.set_xticks(range(len(ORGANS)))
ax_c.set_xticklabels(ORGANS, fontsize=10)
ax_c.set_ylabel("MEI gradient (GALT-int − LP-distal)", fontsize=9)
ax_c.set_title("MEI gradient per batch (SJ01 highlighted)", fontsize=10)
ax_c.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)

from matplotlib.lines import Line2D
leg_c = [Line2D([0],[0], marker='s', color='gray', linestyle='none', markersize=6, label="SJ01"),
         Line2D([0],[0], marker='o', color='gray', linestyle='none', markersize=6, alpha=0.4, label="Other batches")]
ax_c.legend(handles=leg_c, fontsize=8, frameon=False)

# ── panel labels ─────────────────────────────────────────────────────────────
for ax, letter in zip([ax_a, ax_b, ax_c], list("abc")):
    ax.text(-0.04, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

fig.patch.set_facecolor("white")
fig.savefig(f"{OUT}/fig_S4_egress_validation.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/fig_S4_egress_validation.pdf", bbox_inches="tight", facecolor="white")
print("S4 saved.")
