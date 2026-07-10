"""Re-test the existing main paper finding (MMP9⁺ Myeloid → IFN-γ niche)
using spatial-field partial correlation + matched anchor + matched
effector controls.

Existing finding (from project_ibd_cohort_comp memory):
  MMP9⁺ Myeloid creates a localized IFN-γ paracrine niche specifically
  in active IBD mucosa, reproducible across cohorts and tissues.

Tested previously by: per-anchor distance-bin gradient of Hallmark IFN-γ
score on non-anchor neighbours (random-cell null was implicit).
Risk: anchor cells live in immune-rich regions where IFN-γ pathway
genes are also enriched -> apparent gradient may reflect general
"immune-rich neighbourhood" rather than MMP9-driven IFN-γ paracrine.

Specificity test here (per CosMx section):
  1. Smoothed transcript-density field for:
       - MMP9 (the anchor cytokine)
       - IFN-γ effector composite (sum of STAT1, GBP1, CXCL9, CXCL10,
         CXCL11, IRF1, IRF7, IRF8, OAS1, MX1, TAP1, TAP2, IDO1, IFNG,
         GBP5, STAT2)
       - pan-immune density (CD3E + CD4 + CD8A + CD68 + CD163 + CD19)
  2. Partial Pearson r between MMP9 field and IFN-γ-effector field,
     controlling for pan-immune field, restricted to in-tissue pixels.
  3. Permutation null: shuffle effector weights across cells.  z-score
     and one-sided p per section.
  4. MATCHED ANCHOR CONTROLS: replace MMP9 with another myeloid-enriched
     gene (CD68, CD163, ITGAX, ITGAM, TREM2, FCGR3A) and rerun.  If
     MMP9 z is higher than these matched-anchor zs, MMP9-specific.
  5. MATCHED EFFECTOR CONTROLS: replace IFN-γ-effector composite with
     other-axis inflammatory cytokine fields (TNF, IL1B, IL6, IL10,
     CXCL2). If MMP9 z is higher with IFN-γ axis than with these,
     effector-specific.

Outputs:
  outputs/biology/mmp9_ifng_specificity.csv
  outputs/biology/mmp9_ifng_specificity_summary.csv
  outputs/figures/main/fig_mmp9_ifng_specificity.{pdf,png}
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd
import anndata as ad
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.ensure_output_dirs()
plt.rcParams.update(config.NATURE_RCPARAMS)
warnings.filterwarnings("ignore", category=FutureWarning)

PIX_UM = 25.0
BW_UM = 75.0
SIGMA_PIX = BW_UM / PIX_UM
MIN_CELL_DENSITY = 0.1
N_PERM = 50

PAN_IMMUNE_GENES = ["CD3E", "CD4", "CD8A", "CD68", "CD163", "CD19"]

# IFN-γ effector composite = sum of these gene fields
IFNG_EFFECTOR_GENES = ["STAT1", "STAT2", "GBP1", "GBP5",
                        "CXCL9", "CXCL10", "CXCL11",
                        "IRF1", "IRF7", "IRF8",
                        "OAS1", "MX1", "TAP1", "TAP2", "IDO1", "IFNG"]

# Matched anchor controls (other myeloid-enriched genes besides MMP9)
MATCHED_ANCHORS = ["CD68", "CD163", "ITGAX", "ITGAM", "TREM2", "FCGR3A"]

# Matched effector controls (other inflammatory axes, NOT IFN-γ)
MATCHED_EFFECTORS = {
    "TNF":   ["TNF"],
    "IL1B":  ["IL1B"],
    "IL6":   ["IL6"],
    "IL10":  ["IL10"],
    "CXCL2": ["CXCL2"],
    "IL18":  ["IL18"],
}

STATE_PALETTE = {
    "C1 · Healthy":          "#009E73",
    "C1 · UC Non-inflamed":  "#56B4E9",
    "C1 · UC Inflamed":      "#D55E00",
    "C2 · IBD":              "#E69F00",
}
STATE_ORDER = ["C1 · Healthy", "C1 · UC Non-inflamed",
               "C1 · UC Inflamed", "C2 · IBD"]


def _state_label(row):
    cp = str(row["cohort_platform"])
    ds = str(row["disease_state"])
    if cp.startswith("C2"): return "C2 · IBD"
    if ds == "Healthy": return "C1 · Healthy"
    if ds == "Non-inflamed": return "C1 · UC Non-inflamed"
    if ds == "Inflamed": return "C1 · UC Inflamed"
    return None


def _get_X(a, gene):
    if gene not in a.var_names: return None
    col = a[:, gene].X
    if hasattr(col, "toarray"): col = col.toarray()
    return np.asarray(col).flatten()


print("[mmp9-spec] loading meta annotations ...")
meta = ad.read_h5ad(config.ADATA_META, backed="r")
meta_obs = meta.obs[["cohort_platform", "patient_id", "ct_major",
                       "disease_state"]].copy()
meta_obs["state_group"] = meta_obs.apply(_state_label, axis=1)
meta.file.close()


def load_spatial(h5_path, label):
    print(f"\n[mmp9-spec] loading {h5_path} ({label}) ...")
    t0 = time.time()
    a = ad.read_h5ad(h5_path)
    a = a[:, ~a.var_names.isin(config.EXCLUDED_GENES)].copy()
    common = a.obs.index.intersection(meta_obs.index)
    a = a[common].copy()
    overlap = [c for c in meta_obs.columns if c in a.obs.columns]
    if overlap: a.obs = a.obs.drop(columns=overlap)
    a.obs = a.obs.join(meta_obs.loc[common])
    coords = np.asarray(a.obsm["spatial"])
    a.obs["x"] = coords[:, 0]
    a.obs["y"] = coords[:, 1]
    if label == "C1":
        a.obs["section"] = a.obs["sample"].astype(str)
    else:
        a.obs["section"] = a.obs["SAHA_Sample"].astype(str)
    print(f"   shape: {a.shape}, sections: {a.obs['section'].nunique()}, "
          f"loaded in {time.time()-t0:.1f}s")
    return a


a_c1 = load_spatial(
    "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_Exploration/outputs/cosmx_c1_spatial.h5ad", "C1")
a_c2 = load_spatial(
    "/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/2025_SAHA/Analysis/IBD_Exploration/outputs/cosmx_c2_spatial.h5ad", "C2")


# ── Field utilities ────────────────────────────────────────────────
def make_grid(coords, pix_um=PIX_UM, pad=2):
    xmin, ymin = coords.min(axis=0) - pad * pix_um
    xmax, ymax = coords.max(axis=0) + pad * pix_um
    nx = int(np.ceil((xmax - xmin) / pix_um))
    ny = int(np.ceil((ymax - ymin) / pix_um))
    return xmin, ymin, nx, ny


def cell_to_grid(coords, weights, xmin, ymin, nx, ny, pix_um=PIX_UM):
    ix = np.clip(((coords[:, 0] - xmin) / pix_um).astype(int), 0, nx - 1)
    iy = np.clip(((coords[:, 1] - ymin) / pix_um).astype(int), 0, ny - 1)
    grid = np.zeros((ny, nx), dtype=float)
    np.add.at(grid, (iy, ix), weights)
    return grid


def smoothed(grid):
    return gaussian_filter(grid, sigma=SIGMA_PIX, mode="constant", cval=0.0)


def pearson_masked(x, y, mask):
    a = x[mask]; b = y[mask]
    if a.size < 50: return np.nan
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(field_a, field_b, field_c, mask):
    r_ab = pearson_masked(field_a, field_b, mask)
    r_ac = pearson_masked(field_a, field_c, mask)
    r_bc = pearson_masked(field_b, field_c, mask)
    if any(np.isnan([r_ab, r_ac, r_bc])): return np.nan
    den = np.sqrt(max(1 - r_ac**2, 1e-9) * max(1 - r_bc**2, 1e-9))
    return float((r_ab - r_ac * r_bc) / den)


def composite_field(a, gene_list, loc, coords, xmin, ymin, nx, ny):
    """Sum of smoothed transcript fields for the listed genes."""
    out = np.zeros((ny, nx), dtype=float)
    used = []
    for g in gene_list:
        v = _get_X(a, g)
        if v is None: continue
        out = out + cell_to_grid(coords, v[loc].astype(float),
                                   xmin, ymin, nx, ny)
        used.append(g)
    return smoothed(out), used


def composite_weights(a, gene_list, loc):
    """Concatenated weights for permuting a composite (we permute each gene
    individually then sum, to preserve per-gene marginals)."""
    return [(g, _get_X(a, g)[loc].astype(float))
            for g in gene_list if _get_X(a, g) is not None]


def perm_composite_field(weights_list, rng, coords, xmin, ymin, nx, ny):
    out = np.zeros((ny, nx), dtype=float)
    for g, w in weights_list:
        wp = rng.permutation(w)
        out = out + cell_to_grid(coords, wp, xmin, ymin, nx, ny)
    return smoothed(out)


# ── Per-section computation ───────────────────────────────────────
def section_specificity(a, label):
    out_rows = []
    obs = a.obs

    for sec, sub in obs.groupby("section"):
        loc = a.obs.index.get_indexer(sub.index)
        coords = np.column_stack([sub["x"].values, sub["y"].values])
        if coords.shape[0] < 500: continue
        sg = str(sub["state_group"].iloc[0])
        xmin, ymin, nx, ny = make_grid(coords)

        cell_grid = cell_to_grid(coords, np.ones(coords.shape[0]),
                                  xmin, ymin, nx, ny)
        cell_smooth = smoothed(cell_grid)
        mask = cell_smooth > MIN_CELL_DENSITY
        if mask.sum() < 200: continue

        # Pan-immune control field
        pan, _ = composite_field(a, PAN_IMMUNE_GENES, loc, coords,
                                   xmin, ymin, nx, ny)

        # Anchor and matched-anchor fields (single-gene)
        anchor_genes_all = ["MMP9"] + MATCHED_ANCHORS
        anchor_fields = {}
        anchor_weights = {}
        for g in anchor_genes_all:
            v = _get_X(a, g)
            if v is None: continue
            w = v[loc].astype(float)
            anchor_weights[g] = w
            anchor_fields[g] = smoothed(cell_to_grid(coords, w,
                                                      xmin, ymin, nx, ny))

        # Effector composite: IFN-γ
        ifng_field, _ = composite_field(a, IFNG_EFFECTOR_GENES, loc,
                                          coords, xmin, ymin, nx, ny)
        ifng_weights = composite_weights(a, IFNG_EFFECTOR_GENES, loc)

        # Matched effector fields (single-gene)
        eff_fields = {"IFNG_axis": ifng_field}
        eff_weights = {"IFNG_axis": ifng_weights}
        for ename, glist in MATCHED_EFFECTORS.items():
            f, _ = composite_field(a, glist, loc, coords, xmin, ymin, nx, ny)
            eff_fields[ename] = f
            eff_weights[ename] = composite_weights(a, glist, loc)

        rng = np.random.default_rng(hash((sec, label)) & 0xffffffff)

        # Walk anchors x effectors
        for anchor in anchor_genes_all:
            if anchor not in anchor_fields: continue
            for eff_name, eff_f in eff_fields.items():
                r_real = partial_corr(anchor_fields[anchor], eff_f,
                                        pan, mask)
                if np.isnan(r_real): continue
                # Permutation null on effector
                w_list = eff_weights[eff_name]
                perms = []
                for _ in range(N_PERM):
                    f_perm = perm_composite_field(w_list, rng, coords,
                                                     xmin, ymin, nx, ny)
                    rp = partial_corr(anchor_fields[anchor], f_perm,
                                        pan, mask)
                    if not np.isnan(rp): perms.append(rp)
                perms = np.array(perms)
                if perms.size < 5: continue
                mu, sd = perms.mean(), perms.std()
                z = (r_real - mu) / sd if sd > 0 else np.nan
                p = float((np.sum(perms >= r_real) + 1) / (perms.size + 1))
                kind = ("axis" if anchor == "MMP9" and eff_name == "IFNG_axis"
                         else ("matched_anchor" if anchor != "MMP9" and eff_name == "IFNG_axis"
                                else ("matched_effector" if anchor == "MMP9" and eff_name != "IFNG_axis"
                                       else "matched_both")))
                out_rows.append({
                    "cohort": label, "section": sec,
                    "patient_id": str(sub["patient_id"].iloc[0]),
                    "state_group": sg,
                    "anchor": anchor, "effector": eff_name,
                    "kind": kind,
                    "partial_r_real": r_real,
                    "partial_r_null_mean": float(mu),
                    "partial_r_null_sd": float(sd),
                    "z_partial": z, "p_perm": p,
                    "n_pixels": int(mask.sum()),
                })
        print(f"   {label} {sec}: {coords.shape[0]:,} cells, "
              f"{mask.sum():,} tissue pixels")
    return pd.DataFrame(out_rows)


print("\n[mmp9-spec] computing per-section partial corrs + matched controls ...")
t0 = time.time()
df = pd.concat([section_specificity(a_c1, "C1"),
                 section_specificity(a_c2, "C2")], ignore_index=True)
print(f"\n[mmp9-spec] computed in {time.time()-t0:.1f}s, "
      f"{len(df)} (section x pair) records")

df.to_csv(os.path.join(config.BIOLOGY_DIR, "mmp9_ifng_specificity.csv"),
           index=False)


# ── Aggregate per state ─────────────────────────────────────────
def stouffer(zs):
    zs = np.array([z for z in zs if not np.isnan(z)])
    if zs.size == 0: return np.nan
    return float(np.sum(zs) / np.sqrt(zs.size))


# Per state, summarize each (anchor, effector) pair across sections
summary_rows = []
for (sg, anchor, effector, kind), sub in df.groupby(["state_group",
                                                          "anchor",
                                                          "effector",
                                                          "kind"]):
    summary_rows.append({
        "state_group": sg, "anchor": anchor, "effector": effector,
        "kind": kind,
        "n_sections": sub["section"].nunique(),
        "z_stouffer": stouffer(sub["z_partial"].values),
        "r_mean": float(np.nanmean(sub["partial_r_real"].values)),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(config.BIOLOGY_DIR,
                              "mmp9_ifng_specificity_summary.csv"), index=False)


# ── Specificity comparison: MMP9 x IFNG_axis vs matched controls ────
print("\n[mmp9-spec] === Specificity test ===")
print("\n  per-state z (Stouffer across sections):")
for sg in STATE_ORDER:
    sub_state = summary[summary["state_group"] == sg]
    if sub_state.empty: continue
    z_axis = float(sub_state[(sub_state["anchor"] == "MMP9") &
                                (sub_state["effector"] == "IFNG_axis")]
                    ["z_stouffer"].values[0]) if not sub_state[(sub_state["anchor"] == "MMP9") &
                                                                  (sub_state["effector"] == "IFNG_axis")].empty else np.nan
    z_match_anchor = sub_state[(sub_state["kind"] == "matched_anchor")]["z_stouffer"].values
    z_match_eff = sub_state[(sub_state["kind"] == "matched_effector")]["z_stouffer"].values
    z_match_anchor_mean = float(np.nanmean(z_match_anchor)) if z_match_anchor.size else np.nan
    z_match_eff_mean = float(np.nanmean(z_match_eff)) if z_match_eff.size else np.nan

    spec_anchor = (z_axis - z_match_anchor_mean
                    if not np.isnan(z_axis) and not np.isnan(z_match_anchor_mean)
                    else np.nan)
    spec_eff = (z_axis - z_match_eff_mean
                 if not np.isnan(z_axis) and not np.isnan(z_match_eff_mean)
                 else np.nan)

    verdict_anchor = "YES" if spec_anchor > 0.5 and z_axis > 2 else "no"
    verdict_eff = "YES" if spec_eff > 0.5 and z_axis > 2 else "no"

    print(f"  {sg:25s}  z(MMP9 x IFN-γ) = {z_axis:+6.2f}  "
          f"z(matched-anchor mean) = {z_match_anchor_mean:+6.2f}  "
          f"z(matched-effector mean) = {z_match_eff_mean:+6.2f}")
    print(f"      anchor-spec Δz = {spec_anchor:+6.2f}  "
          f"({verdict_anchor})    "
          f"effector-spec Δz = {spec_eff:+6.2f}  ({verdict_eff})")

print("\n  Per matched-anchor breakdown (z for each anchor x IFNG_axis):")
sub = summary[(summary["effector"] == "IFNG_axis")]
piv = sub.pivot_table(index="anchor", columns="state_group",
                       values="z_stouffer").reindex(columns=STATE_ORDER)
piv = piv.reindex(["MMP9"] + MATCHED_ANCHORS)
print(piv.to_string())

print("\n  Per matched-effector breakdown (z for MMP9 x each effector):")
sub = summary[(summary["anchor"] == "MMP9")]
piv = sub.pivot_table(index="effector", columns="state_group",
                       values="z_stouffer").reindex(columns=STATE_ORDER)
print(piv.to_string())


# ── Figure ─────────────────────────────────────────────────────────
print("\n[mmp9-spec] building figure ...")
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30,
                        left=0.08, right=0.99, top=0.93, bottom=0.10)

# A: per-state z bars: axis pair vs mean of matched anchor controls
ax_a = fig.add_subplot(gs[0, 0])
xs = np.arange(len(STATE_ORDER)); w = 0.35
for i, sg in enumerate(STATE_ORDER):
    sub_state = summary[summary["state_group"] == sg]
    if sub_state.empty: continue
    z_axis = float(sub_state[(sub_state["anchor"] == "MMP9") &
                                (sub_state["effector"] == "IFNG_axis")]
                    ["z_stouffer"].values[0]) if not sub_state[(sub_state["anchor"] == "MMP9") &
                                                                  (sub_state["effector"] == "IFNG_axis")].empty else np.nan
    z_match = sub_state[sub_state["kind"] == "matched_anchor"]["z_stouffer"].values
    z_match_mean = float(np.nanmean(z_match)) if z_match.size else np.nan
    ax_a.bar([i - w/2], [z_axis], width=w,
              color=STATE_PALETTE[sg], edgecolor="black", linewidth=0.5,
              label="MMP9 × IFN-γ" if i == 0 else None)
    ax_a.bar([i + w/2], [z_match_mean], width=w,
              color="#999999", edgecolor="black", linewidth=0.5,
              label="other myeloid × IFN-γ (mean)" if i == 0 else None)
    # Individual matched anchor zs as dots
    ax_a.scatter(np.full(z_match.size, i + w/2)
                  + np.random.default_rng(i).uniform(-0.06, 0.06, z_match.size),
                  z_match, s=18, c="black", alpha=0.75,
                  edgecolors="white", linewidths=0.4)
ax_a.axhline(0, color="gray", lw=0.5)
ax_a.axhline(2, color="red", lw=0.4, ls=":")
ax_a.set_xticks(xs); ax_a.set_xticklabels(
    [s.replace("· ", "\n") for s in STATE_ORDER], fontsize=8)
ax_a.set_ylabel("Per-section partial-r z (Stouffer)\n(real vs perm | pan-immune)")
ax_a.set_title("A  Anchor specificity:\nMMP9 vs other myeloid markers, "
                "× IFN-γ effector composite", fontsize=10, loc="left")
ax_a.legend(fontsize=8, frameon=False, loc="best")

# B: per-state z bars: axis pair vs mean of matched effector controls
ax_b = fig.add_subplot(gs[0, 1])
for i, sg in enumerate(STATE_ORDER):
    sub_state = summary[summary["state_group"] == sg]
    if sub_state.empty: continue
    z_axis = float(sub_state[(sub_state["anchor"] == "MMP9") &
                                (sub_state["effector"] == "IFNG_axis")]
                    ["z_stouffer"].values[0]) if not sub_state[(sub_state["anchor"] == "MMP9") &
                                                                  (sub_state["effector"] == "IFNG_axis")].empty else np.nan
    z_match = sub_state[sub_state["kind"] == "matched_effector"]["z_stouffer"].values
    z_match_mean = float(np.nanmean(z_match)) if z_match.size else np.nan
    ax_b.bar([i - w/2], [z_axis], width=w,
              color=STATE_PALETTE[sg], edgecolor="black", linewidth=0.5,
              label="MMP9 × IFN-γ" if i == 0 else None)
    ax_b.bar([i + w/2], [z_match_mean], width=w,
              color="#999999", edgecolor="black", linewidth=0.5,
              label="MMP9 × other effector (mean)" if i == 0 else None)
    ax_b.scatter(np.full(z_match.size, i + w/2)
                  + np.random.default_rng(i).uniform(-0.06, 0.06, z_match.size),
                  z_match, s=18, c="black", alpha=0.75,
                  edgecolors="white", linewidths=0.4)
ax_b.axhline(0, color="gray", lw=0.5)
ax_b.axhline(2, color="red", lw=0.4, ls=":")
ax_b.set_xticks(xs); ax_b.set_xticklabels(
    [s.replace("· ", "\n") for s in STATE_ORDER], fontsize=8)
ax_b.set_ylabel("Per-section partial-r z (Stouffer)")
ax_b.set_title("B  Effector specificity:\nIFN-γ axis vs other inflammatory "
                "axes, × MMP9", fontsize=10, loc="left")
ax_b.legend(fontsize=8, frameon=False, loc="best")

# C: anchor heatmap (rows = anchors, cols = states)
ax_c = fig.add_subplot(gs[1, 0])
sub = summary[(summary["effector"] == "IFNG_axis")]
piv = sub.pivot_table(index="anchor", columns="state_group",
                       values="z_stouffer").reindex(columns=STATE_ORDER)
piv = piv.reindex(["MMP9"] + MATCHED_ANCHORS)
mat = piv.values
ax_c.pcolormesh(np.arange(piv.shape[1] + 1), np.arange(piv.shape[0] + 1),
                 mat, cmap="RdBu_r", vmin=-6, vmax=6,
                 edgecolors="white", linewidth=0.3)
ax_c.invert_yaxis()
ax_c.set_xticks(np.arange(piv.shape[1]) + 0.5)
ax_c.set_yticks(np.arange(piv.shape[0]) + 0.5)
ax_c.set_xticklabels([s.replace("· ", "\n") for s in piv.columns],
                      fontsize=8)
ax_c.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = mat[i, j]
        if np.isnan(v): continue
        ax_c.text(j + 0.5, i + 0.5, f"{v:+.1f}", ha="center", va="center",
                   fontsize=7, color="white" if abs(v) > 3 else "black")
ax_c.set_title("C  Anchor breakdown: each myeloid marker × IFN-γ effector\n"
                "(Stouffer z; > 2 = significant)", fontsize=10, loc="left")

# D: effector heatmap (rows = effectors, cols = states)
ax_d = fig.add_subplot(gs[1, 1])
sub = summary[(summary["anchor"] == "MMP9")]
piv = sub.pivot_table(index="effector", columns="state_group",
                       values="z_stouffer").reindex(columns=STATE_ORDER)
piv = piv.reindex(["IFNG_axis"] + list(MATCHED_EFFECTORS.keys()))
mat = piv.values
ax_d.pcolormesh(np.arange(piv.shape[1] + 1), np.arange(piv.shape[0] + 1),
                 mat, cmap="RdBu_r", vmin=-6, vmax=6,
                 edgecolors="white", linewidth=0.3)
ax_d.invert_yaxis()
ax_d.set_xticks(np.arange(piv.shape[1]) + 0.5)
ax_d.set_yticks(np.arange(piv.shape[0]) + 0.5)
ax_d.set_xticklabels([s.replace("· ", "\n") for s in piv.columns],
                      fontsize=8)
ax_d.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = mat[i, j]
        if np.isnan(v): continue
        ax_d.text(j + 0.5, i + 0.5, f"{v:+.1f}", ha="center", va="center",
                   fontsize=7, color="white" if abs(v) > 3 else "black")
ax_d.set_title("D  Effector breakdown: MMP9 × each inflammatory axis\n"
                "(Stouffer z; > 2 = significant)", fontsize=10, loc="left")

fig.suptitle("MMP9⁺ Myeloid → IFN-γ niche specificity test "
             "(spatial fields, partial corr | pan-immune)",
             fontsize=11, y=0.985)
out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_mmp9_ifng_specificity.pdf")
out_png = os.path.join(config.FIG_MAIN_DIR, "fig_mmp9_ifng_specificity.png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=180)
print(f"[mmp9-spec] wrote {out_pdf}")
print(f"[mmp9-spec] wrote {out_png}")
print("\n[mmp9-spec] DONE.")
