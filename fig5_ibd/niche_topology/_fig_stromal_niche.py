"""Stromal / fibroblast spatial niche test (proposal B), spatial-field
formulation with partial-correlation + matched-anchor controls.

Three biological questions:
  Q1.  Does the activated-fibroblast field (FAP+POSTN+PDPN+THY1)
       spatially follow the IFN-γ inflammation field in IBD,
       beyond what pan-immune density predicts and beyond what
       generic mesenchymal markers predict?

  Q2.  Does the crypt-niche stromal field (BMP2+BMP4+GREM1+RSPO3 -
       the canonical stem-cell-niche signalling stroma) get spatially
       *excluded* from inflammation in IBD (negative partial corr)?

  Q3.  Are the activated-fibroblast and crypt-niche stromal fields
       spatially anti-correlated (i.e., distinct compartments) -
       and does this anti-correlation strengthen in IBD?

Same field/partial-correlation/matched-control framework as
_fig_mmp9_ifng_specificity.py and _fig_il2317_specificity.py.
This is a CELL-INDEPENDENT test (transcript fields, no ct_major).

Outputs:
  outputs/biology/stromal_niche.csv
  outputs/biology/stromal_niche_summary.csv
  outputs/figures/main/fig_stromal_niche.{pdf,png}
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

ACT_FIBRO_GENES   = ["FAP", "POSTN", "PDPN", "THY1"]
CRYPT_NICHE_GENES = ["BMP2", "BMP4", "GREM1", "RSPO3"]
ECM_GENES         = ["LOX", "LOXL2", "MMP14", "COL4A1", "COL11A1"]
IFNG_EFFECTOR_GENES = ["STAT1", "STAT2", "GBP1", "GBP5",
                        "CXCL9", "CXCL10", "CXCL11",
                        "IRF1", "IRF7", "IRF8",
                        "OAS1", "MX1", "TAP1", "TAP2", "IDO1", "IFNG"]

# Matched stromal anchor controls (other mesenchymal markers, not the
# activated-fibroblast or crypt-niche signature)
MATCHED_STROMAL = {
    "PDGFRA":   ["PDGFRA"],
    "PDGFRB":   ["PDGFRB"],
    "ENG":      ["ENG"],
    "WNT5A":    ["WNT5A"],
    "SFRP1":    ["SFRP1"],
}

# Matched effector controls (other inflammatory cytokine axes)
MATCHED_EFFECTORS = {
    "TNF":   ["TNF"],
    "IL1B":  ["IL1B"],
    "IL6":   ["IL6"],
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


print("[stromal] loading meta annotations ...")
meta = ad.read_h5ad(config.ADATA_META, backed="r")
meta_obs = meta.obs[["cohort_platform", "patient_id", "ct_major",
                       "disease_state"]].copy()
meta_obs["state_group"] = meta_obs.apply(_state_label, axis=1)
meta.file.close()


def load_spatial(h5_path, label):
    print(f"\n[stromal] loading {h5_path} ({label}) ...")
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
    out = np.zeros((ny, nx), dtype=float)
    for g in gene_list:
        v = _get_X(a, g)
        if v is None: continue
        out = out + cell_to_grid(coords, v[loc].astype(float),
                                   xmin, ymin, nx, ny)
    return smoothed(out)


def composite_weights(a, gene_list, loc):
    return [(g, _get_X(a, g)[loc].astype(float))
            for g in gene_list if _get_X(a, g) is not None]


def perm_composite_field(weights_list, rng, coords, xmin, ymin, nx, ny):
    out = np.zeros((ny, nx), dtype=float)
    for g, w in weights_list:
        wp = rng.permutation(w)
        out = out + cell_to_grid(coords, wp, xmin, ymin, nx, ny)
    return smoothed(out)


# ── Per-section computation ───────────────────────────────────────
def section_stromal(a, label):
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

        pan = composite_field(a, PAN_IMMUNE_GENES, loc, coords,
                                xmin, ymin, nx, ny)

        # Build all anchor / effector composite fields
        fields = {}
        weights_map = {}
        spec = {
            "ActFibro":     ACT_FIBRO_GENES,
            "CryptNiche":   CRYPT_NICHE_GENES,
            "ECM":          ECM_GENES,
            "IFNG":         IFNG_EFFECTOR_GENES,
        }
        for ename, glist in spec.items():
            fields[ename] = composite_field(a, glist, loc, coords,
                                              xmin, ymin, nx, ny)
            weights_map[ename] = composite_weights(a, glist, loc)
        for ename, glist in MATCHED_STROMAL.items():
            fields[ename] = composite_field(a, glist, loc, coords,
                                              xmin, ymin, nx, ny)
            weights_map[ename] = composite_weights(a, glist, loc)
        for ename, glist in MATCHED_EFFECTORS.items():
            fields[ename] = composite_field(a, glist, loc, coords,
                                              xmin, ymin, nx, ny)
            weights_map[ename] = composite_weights(a, glist, loc)

        rng = np.random.default_rng(hash((sec, label)) & 0xffffffff)

        def measure(anchor_name, effector_name, kind):
            if anchor_name not in fields or effector_name not in fields:
                return None
            r_real = partial_corr(fields[anchor_name], fields[effector_name],
                                    pan, mask)
            if np.isnan(r_real): return None
            wlist = weights_map[effector_name]
            perms = []
            for _ in range(N_PERM):
                f_perm = perm_composite_field(wlist, rng, coords,
                                                xmin, ymin, nx, ny)
                rp = partial_corr(fields[anchor_name], f_perm, pan, mask)
                if not np.isnan(rp): perms.append(rp)
            perms = np.array(perms)
            if perms.size < 5: return None
            mu, sd = perms.mean(), perms.std()
            z = (r_real - mu) / sd if sd > 0 else np.nan
            p = float((np.sum(perms >= r_real) + 1) / (perms.size + 1))
            return {
                "cohort": label, "section": sec,
                "patient_id": str(sub["patient_id"].iloc[0]),
                "state_group": sg,
                "anchor": anchor_name, "effector": effector_name,
                "kind": kind,
                "partial_r_real": r_real,
                "partial_r_null_mean": float(mu),
                "partial_r_null_sd": float(sd),
                "z_partial": z, "p_perm": p,
                "n_pixels": int(mask.sum()),
            }

        # Q1: ActFibro x IFNG (axis), and matched stromal anchors x IFNG
        for anchor, kind in [("ActFibro", "axis_q1")] + \
                              [(s, "matched_anchor_q1") for s in MATCHED_STROMAL]:
            r = measure(anchor, "IFNG", kind)
            if r: out_rows.append(r)
        # Matched effector for Q1 (ActFibro x other inflammatory axes)
        for eff in MATCHED_EFFECTORS:
            r = measure("ActFibro", eff, "matched_effector_q1")
            if r: out_rows.append(r)

        # Q2: CryptNiche x IFNG
        for anchor, kind in [("CryptNiche", "axis_q2")] + \
                              [(s, "matched_anchor_q2") for s in MATCHED_STROMAL]:
            r = measure(anchor, "IFNG", kind)
            if r: out_rows.append(r)
        for eff in MATCHED_EFFECTORS:
            r = measure("CryptNiche", eff, "matched_effector_q2")
            if r: out_rows.append(r)

        # Q3: ActFibro x CryptNiche
        r = measure("ActFibro", "CryptNiche", "axis_q3")
        if r: out_rows.append(r)
        for s in MATCHED_STROMAL:
            r = measure(s, "CryptNiche", "matched_anchor_q3")
            if r: out_rows.append(r)

        # Bonus: ActFibro x ECM
        r = measure("ActFibro", "ECM", "bonus_actfibro_ecm")
        if r: out_rows.append(r)

        print(f"   {label} {sec}: {coords.shape[0]:,} cells, "
              f"{mask.sum():,} tissue pixels")
    return pd.DataFrame(out_rows)


print("\n[stromal] computing per-section partial corrs ...")
t0 = time.time()
df = pd.concat([section_stromal(a_c1, "C1"),
                 section_stromal(a_c2, "C2")], ignore_index=True)
print(f"\n[stromal] computed in {time.time()-t0:.1f}s, "
      f"{len(df)} (section x pair) records")

df.to_csv(os.path.join(config.BIOLOGY_DIR, "stromal_niche.csv"), index=False)


def stouffer(zs):
    zs = np.array([z for z in zs if not np.isnan(z)])
    if zs.size == 0: return np.nan
    return float(np.sum(zs) / np.sqrt(zs.size))


# Per-state summary
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
                              "stromal_niche_summary.csv"), index=False)


def get_z(state_group, anchor, effector):
    s = summary[(summary["state_group"] == state_group) &
                 (summary["anchor"] == anchor) &
                 (summary["effector"] == effector)]
    return float(s["z_stouffer"].values[0]) if not s.empty else np.nan


def matched_mean(state_group, kind):
    s = summary[(summary["state_group"] == state_group) &
                 (summary["kind"] == kind)]
    return float(np.nanmean(s["z_stouffer"].values)) if not s.empty else np.nan


print("\n[stromal] === Q1: ActFibro × IFN-γ (specificity) ===")
print("  state                       z_axis  z_match-anchor  z_match-effector  Δz_anchor  Δz_eff")
for sg in STATE_ORDER:
    z_axis = get_z(sg, "ActFibro", "IFNG")
    z_ma = matched_mean(sg, "matched_anchor_q1")
    z_me = matched_mean(sg, "matched_effector_q1")
    print(f"  {sg:25s}  {z_axis:+6.2f}    {z_ma:+6.2f}        {z_me:+6.2f}        "
          f"{(z_axis-z_ma):+6.2f}    {(z_axis-z_me):+6.2f}")

print("\n[stromal] === Q2: CryptNiche × IFN-γ (expecting NEGATIVE z = exclusion) ===")
print("  state                       z_axis  z_match-anchor  z_match-effector  Δz_anchor  Δz_eff")
for sg in STATE_ORDER:
    z_axis = get_z(sg, "CryptNiche", "IFNG")
    z_ma = matched_mean(sg, "matched_anchor_q2")
    z_me = matched_mean(sg, "matched_effector_q2")
    print(f"  {sg:25s}  {z_axis:+6.2f}    {z_ma:+6.2f}        {z_me:+6.2f}        "
          f"{(z_axis-z_ma):+6.2f}    {(z_axis-z_me):+6.2f}")

print("\n[stromal] === Q3: ActFibro × CryptNiche (expecting NEGATIVE z = compartments) ===")
print("  state                       z_axis  z_match-anchor (other stroma × CryptNiche)  Δz")
for sg in STATE_ORDER:
    z_axis = get_z(sg, "ActFibro", "CryptNiche")
    z_ma = matched_mean(sg, "matched_anchor_q3")
    print(f"  {sg:25s}  {z_axis:+6.2f}    {z_ma:+6.2f}                                {(z_axis-z_ma):+6.2f}")

print("\n[stromal] === Bonus: ActFibro × ECM remodeling ===")
for sg in STATE_ORDER:
    z = get_z(sg, "ActFibro", "ECM")
    print(f"  {sg:25s}  z = {z:+6.2f}")


# ── Figure ─────────────────────────────────────────────────────────
print("\n[stromal] building figure ...")
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4,
                        left=0.07, right=0.99, top=0.93, bottom=0.10)


def panel_q(ax, qkind_axis, qkind_match_anchor, anchor_axis_label,
             effector_label, title, expect_negative=False):
    xs = np.arange(len(STATE_ORDER)); w = 0.35
    for i, sg in enumerate(STATE_ORDER):
        z_axis = get_z(sg, anchor_axis_label, effector_label)
        z_ma = matched_mean(sg, qkind_match_anchor)
        ax.bar([i - w/2], [z_axis], width=w,
                color=STATE_PALETTE[sg], edgecolor="black", linewidth=0.5,
                label=(f"{anchor_axis_label} × {effector_label}"
                       if i == 0 else None))
        ax.bar([i + w/2], [z_ma], width=w,
                color="#999999", edgecolor="black", linewidth=0.5,
                label=("matched-anchor mean" if i == 0 else None))
        # Individual matched anchor zs
        sub_match = summary[(summary["state_group"] == sg) &
                              (summary["kind"] == qkind_match_anchor)]
        zs = sub_match["z_stouffer"].dropna().values
        ax.scatter(np.full(zs.size, i + w/2)
                    + np.random.default_rng(i).uniform(-0.06, 0.06, zs.size),
                    zs, s=18, c="black", alpha=0.7,
                    edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    if expect_negative:
        ax.axhline(-2, color="red", lw=0.4, ls=":")
    else:
        ax.axhline(2, color="red", lw=0.4, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("· ", "\n") for s in STATE_ORDER], fontsize=7)
    ax.set_ylabel("Stouffer z (real vs perm | pan-immune)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")


ax_a = fig.add_subplot(gs[0, 0])
panel_q(ax_a, "axis_q1", "matched_anchor_q1", "ActFibro", "IFNG",
         "A  Q1: activated fibroblast × IFN-γ axis\n"
         "vs other stromal markers × IFN-γ")
ax_b = fig.add_subplot(gs[0, 1])
panel_q(ax_b, "axis_q2", "matched_anchor_q2", "CryptNiche", "IFNG",
         "B  Q2: crypt-niche stroma × IFN-γ axis\n"
         "vs other stromal markers × IFN-γ", expect_negative=True)
ax_c = fig.add_subplot(gs[0, 2])
panel_q(ax_c, "axis_q3", "matched_anchor_q3", "ActFibro", "CryptNiche",
         "C  Q3: ActFibro × CryptNiche\n"
         "vs other stromal × CryptNiche", expect_negative=True)

# D: ActFibro × {IFNG, TNF, IL1B, IL6} per state
ax_d = fig.add_subplot(gs[1, 0])
effectors_q1 = ["IFNG", "TNF", "IL1B", "IL6"]
xs = np.arange(len(effectors_q1)); w = 0.20
for i, eff in enumerate(effectors_q1):
    for j, sg in enumerate(STATE_ORDER):
        z = get_z(sg, "ActFibro", eff)
        ax_d.bar([i + (j - 1.5) * w], [z], width=w,
                  color=STATE_PALETTE[sg],
                  edgecolor="black", linewidth=0.4,
                  label=(sg if i == 0 else None))
ax_d.axhline(0, color="gray", lw=0.5)
ax_d.axhline(2, color="red", lw=0.4, ls=":")
ax_d.set_xticks(xs); ax_d.set_xticklabels(effectors_q1, fontsize=8)
ax_d.set_ylabel("Stouffer z")
ax_d.set_title("D  ActFibro × cytokine axes (effector specificity)",
                fontsize=10, loc="left")
ax_d.legend(fontsize=6, frameon=False, loc="best")

# E: CryptNiche × {IFNG, TNF, IL1B, IL6}
ax_e = fig.add_subplot(gs[1, 1])
for i, eff in enumerate(effectors_q1):
    for j, sg in enumerate(STATE_ORDER):
        z = get_z(sg, "CryptNiche", eff)
        ax_e.bar([i + (j - 1.5) * w], [z], width=w,
                  color=STATE_PALETTE[sg],
                  edgecolor="black", linewidth=0.4)
ax_e.axhline(0, color="gray", lw=0.5)
ax_e.axhline(-2, color="red", lw=0.4, ls=":")
ax_e.set_xticks(xs); ax_e.set_xticklabels(effectors_q1, fontsize=8)
ax_e.set_ylabel("Stouffer z")
ax_e.set_title("E  CryptNiche × cytokine axes (NEG = exclusion)",
                fontsize=10, loc="left")

# F: ActFibro × ECM
ax_f = fig.add_subplot(gs[1, 2])
xs = np.arange(len(STATE_ORDER)); w = 0.5
for i, sg in enumerate(STATE_ORDER):
    z = get_z(sg, "ActFibro", "ECM")
    ax_f.bar([i], [z], width=w, color=STATE_PALETTE[sg],
              edgecolor="black", linewidth=0.5)
ax_f.axhline(0, color="gray", lw=0.5)
ax_f.axhline(2, color="red", lw=0.4, ls=":")
ax_f.set_xticks(xs); ax_f.set_xticklabels(
    [s.replace("· ", "\n") for s in STATE_ORDER], fontsize=7)
ax_f.set_ylabel("Stouffer z")
ax_f.set_title("F  ActFibro × ECM remodeling axis", fontsize=10, loc="left")

fig.suptitle("Stromal niche spatial test "
             "(transcript fields, partial corr | pan-immune)",
             fontsize=11, y=0.985)
out_pdf = os.path.join(config.FIG_MAIN_DIR, "fig_stromal_niche.pdf")
out_png = os.path.join(config.FIG_MAIN_DIR, "fig_stromal_niche.png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=180)
print(f"[stromal] wrote {out_pdf}")
print(f"[stromal] wrote {out_png}")
print("\n[stromal] DONE.")
