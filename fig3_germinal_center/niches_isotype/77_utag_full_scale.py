"""
77_utag_full_scale.py
Full-scale UTAG-framed analysis of the plasma-vascular niche:

Task 1: Port UTAG niches to 15-section data (via classifier trained on 4 shared sections)
Task 2: Refine the 4-section UTAG analysis with clean narrative
Task 3: Add spatial maps of UTAG niches in each organ's representative section
Task 4: Cross-validate: check that 15-section UTAG proxy matches 4-section UTAG
         on shared sections

Run: conda run -n saha python scripts/77_utag_full_scale.py
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR  = BASE_DIR / "figures"
OUT_DIR  = BASE_DIR / "utag_niche_analysis"
OUT_DIR.mkdir(exist_ok=True)
CLEANED_DIR = Path("/Users/jiwoonpark/Dropbox (Personal)/2025- MasonLab/"
                   "2025_SAHA/Analysis/Integration/Integration/processed/cleaned")

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.5,
    'legend.fontsize': 5.5, 'legend.frameon': False,
})

OC = {'APE': '#3D8B8B', 'COL': '#1B3A5C', 'ILE': '#8B3A3A'}
OL = {'APE': 'Appendix', 'COL': 'Colon', 'ILE': 'Ileum'}
ORGANS = ['APE', 'COL', 'ILE']
PX_TO_UM = 0.1203
RADIUS_PX = int(100 / PX_TO_UM)  # 100µm for niche-scale assignment

UTAG_NICHES = ['crypt_adjacent_immune', 'epithelial', 'lymphoid_structure',
               'germinal_center', 'mucosa', 'muscularis_mucosa',
               'submucosa', 'muscularis_externa']

NICHE_COLORS = {
    'crypt_adjacent_immune': '#FFB74D',
    'epithelial': '#81C784',
    'lymphoid_structure': '#9575CD',
    'germinal_center': '#E91E63',
    'mucosa': '#64B5F6',
    'muscularis_mucosa': '#8D6E63',
    'submucosa': '#90A4AE',
    'muscularis_externa': '#5D4037',
}

print("=" * 72)
print("FULL-SCALE UTAG NICHE ANALYSIS (15 sections)")
print("=" * 72)

# ═══ Task 1 setup: load analysis_ready.h5ad (has UTAG labels) ═══════════
print("\nLoading analysis_ready.h5ad (has UTAG_names)...")
ar = sc.read_h5ad(BASE_DIR / "analysis_ready.h5ad")
ar_organs = ar.obs['SAHA_organcode'].astype(str).values
ar_sections = ar.obs['SAHA_Sample'].astype(str).values
ar_broad_map = {'Bcell': 'Lymphoid-B', 'LymphoidStructure_Bcells': 'Lymphoid-B',
                'IgA_plasma': 'Lymphoid-Plasma', 'IgG_plasma': 'Lymphoid-Plasma',
                'Cytotoxic_CD8': 'Lymphoid-T', 'CD4': 'Lymphoid-T',
                'Naive_T': 'Lymphoid-T', 'Exhausted_CD4': 'Lymphoid-T',
                'EpiTcell_mixed': 'Lymphoid-T', 'Mixed_Lymphoids': 'Lymphoid-Other',
                'Mixed_Immune': 'Lymphoid-Other', 'Myeloids': 'Myeloid',
                'Mast': 'Myeloid',
                'Fibroblasts': 'Mesenchymal-Stromal',
                'SmoothMuscle': 'Mesenchymal-Stromal',
                'Endothelial': 'Endothelial', 'Lymphatic_Endo': 'Endothelial',
                'bottom_crypts': 'Epithelial', 'top_crypts': 'Epithelial',
                'Paneth_Stem': 'Epithelial', 'ileum_crypts': 'Epithelial',
                'Low_quality': 'Other', 'Glial': 'Neuronal-Glial'}

ar_broad = np.array([ar_broad_map.get(c, 'Other')
                      for c in ar.obs['new_clusters'].astype(str).values])
ar_utag = ar.obs['UTAG_names'].astype(str).values
ar_coords = ar.obs[['CenterX_global_px', 'CenterY_global_px']].values.astype(float)

ALL_CT = ['Endothelial', 'Epithelial', 'Erythroid', 'Lymphoid-B', 'Lymphoid-DC',
          'Lymphoid-Other', 'Lymphoid-Plasma', 'Lymphoid-T', 'Mesenchymal-Stromal',
          'Myeloid', 'Neuronal-Glial', 'Stem']

ct_to_idx = {ct: i for i, ct in enumerate(ALL_CT)}


def compute_neighborhood_comp(broad_arr, coords_arr, sections_arr, radius_px):
    """Compute neighborhood composition (12-dim) for all cells."""
    n = len(broad_arr)
    comp = np.zeros((n, len(ALL_CT)), dtype=np.float32)
    for sec in np.unique(sections_arr):
        sm = sections_arr == sec
        sec_idx = np.where(sm)[0]
        sec_coords = coords_arr[sec_idx]
        sec_broad = broad_arr[sec_idx]

        if len(sec_idx) < 50:
            continue

        tree = cKDTree(sec_coords)
        nbs_list = tree.query_ball_point(sec_coords, r=radius_px)

        for i, nbs in enumerate(nbs_list):
            if len(nbs) < 3:
                continue
            nb_types = sec_broad[nbs]
            for ct in ALL_CT:
                comp[sec_idx[i], ct_to_idx[ct]] = (nb_types == ct).sum()

        # Normalize
        row_sums = comp[sec_idx].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        comp[sec_idx] = comp[sec_idx] / row_sums

    return comp


print(f"  Computing neighborhoods on analysis_ready ({len(ar_broad):,} cells)...")
t0 = time.time()
ar_comp = compute_neighborhood_comp(ar_broad, ar_coords, ar_sections, RADIUS_PX)
print(f"  Done in {time.time()-t0:.0f}s")


# ═══ Train classifier: neighborhood → UTAG_names ══════════════════════
print("\nTraining classifier (neighborhood → UTAG_names)...")
valid_mask = (ar_comp.sum(axis=1) > 0.9) & (ar_utag != 'nan')
X_train = ar_comp[valid_mask]
y_train = ar_utag[valid_mask]

t0 = time.time()
clf = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
print(f"  Train accuracy: {train_acc:.3f}  [{time.time()-t0:.0f}s]")

# Cross-validate on held-out cells per section
from sklearn.model_selection import cross_val_score
# Use 5-fold for speed
cv_scores = cross_val_score(clf, X_train[:20000], y_train[:20000], cv=3)
print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ═══ Apply classifier to all 15 sections ═════════════════════════════
print("\n" + "=" * 72)
print("APPLYING UTAG CLASSIFIER TO ALL 15 SECTIONS")
print("=" * 72)

all_cell_records = []
for org in ORGANS:
    t0 = time.time()
    a = sc.read_h5ad(CLEANED_DIR / f"{org}_RNA_cleaned.h5ad")
    broad = a.obs['cell_type'].astype(str).values
    subtypes = a.obs['cell_subtype'].astype(str).values
    sections = a.obs['SAHA_Sample'].astype(str).values
    coords = a.obs[['CenterX_global_px', 'CenterY_global_px']].values.astype(float)

    print(f"\n  {org}: {a.shape[0]:,} cells")
    comp = compute_neighborhood_comp(broad, coords, sections, RADIUS_PX)

    # Predict niche
    predicted = clf.predict(comp)

    for i in range(a.shape[0]):
        all_cell_records.append({
            'organ': org, 'section': sections[i], 'cell_idx': i,
            'broad': broad[i], 'subtype': subtypes[i],
            'predicted_niche': predicted[i],
            'coord_x': coords[i, 0], 'coord_y': coords[i, 1],
        })

    del a
    print(f"    Predicted niches [{time.time()-t0:.0f}s]")

pred_df = pd.DataFrame(all_cell_records)
pred_df.to_csv(OUT_DIR / "predicted_UTAG_niches_15sec.csv", index=False)
print(f"\nSaved predictions: {len(pred_df):,} rows")


# ═══ Task 4: Validate on shared sections ═══════════════════════════════
print("\n" + "=" * 72)
print("VALIDATION: shared sections (4) — predicted vs actual UTAG")
print("=" * 72)

# Map analysis_ready cells to cleaned via coordinates (approximate match)
shared_sections = set(ar_sections) & set(pred_df['section'].unique())
print(f"\n  Shared sections: {shared_sections}")

# For each shared section, for each actual UTAG niche, compute:
#   What fraction of cells in that niche get classified correctly?
# Use section-level composition check (not cell-level matching)

for sec in sorted(shared_sections):
    # Actual niche fractions
    sm = ar_sections == sec
    actual_fracs = pd.Series(ar_utag[sm]).value_counts(normalize=True)

    # Predicted niche fractions (in cleaned, same section)
    pm = pred_df['section'] == sec
    pred_fracs = pred_df[pm]['predicted_niche'].value_counts(normalize=True)

    print(f"\n  {sec}:")
    all_niches = sorted(set(actual_fracs.index) | set(pred_fracs.index))
    print(f"  {'Niche':25s} {'Actual %':>10s} {'Predicted %':>12s}")
    for n in all_niches:
        a_pct = actual_fracs.get(n, 0) * 100
        p_pct = pred_fracs.get(n, 0) * 100
        print(f"  {n:25s} {a_pct:9.1f}% {p_pct:11.1f}%")


# ═══ Task 2 (refined): crypt-adjacent-immune composition across 15 sections ══
print("\n" + "=" * 72)
print("15-SECTION CRYPT-ADJACENT-IMMUNE COMPOSITION")
print("=" * 72)

cai = pred_df[pred_df['predicted_niche'] == 'crypt_adjacent_immune']

# Cell type composition per organ
cai_comp = {}
for org in ORGANS:
    org_cai = cai[cai['organ'] == org]
    if len(org_cai) < 50:
        continue

    broad_counts = org_cai['broad'].value_counts(normalize=True) * 100
    cai_comp[org] = broad_counts

    print(f"\n  {org} crypt-adjacent-immune (n={len(org_cai):,}):")
    for bt, pct in broad_counts.head(8).items():
        print(f"    {bt:25s} {pct:5.1f}%")


# ═══ Task 3: Spatial maps of UTAG niches in each organ ═══════════════════
print("\n" + "=" * 72)
print("BUILDING COMPREHENSIVE FIGURE")
print("=" * 72)

# Representative sections
REP_SECS = {'APE': 'APE_A9_3', 'COL': 'COL_A1_5', 'ILE': 'ILE_A1_3'}

fig = plt.figure(figsize=(7.5, 10))
gs = gridspec.GridSpec(4, 3, figure=fig,
                       left=0.07, right=0.96, top=0.97, bottom=0.04,
                       hspace=0.60, wspace=0.40,
                       height_ratios=[1.4, 1, 1, 1])

# ── (a) Spatial maps of UTAG niches per organ ─────────────────────────
print("  Panel a: spatial maps...")
for oi, org in enumerate(ORGANS):
    ax = fig.add_subplot(gs[0, oi])
    sec = REP_SECS[org]
    sec_cells = pred_df[pred_df['section'] == sec]

    coords = sec_cells[['coord_x', 'coord_y']].values * PX_TO_UM
    coords -= coords.min(axis=0)

    for niche in UTAG_NICHES:
        mask = sec_cells['predicted_niche'].values == niche
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                        s=0.12, c=NICHE_COLORS[niche], alpha=0.7,
                        rasterized=True,
                        label=niche.replace('_', ' ') if oi == 0 else None)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{OL[org]} ({sec})',
                  fontsize=7, fontweight='bold', color=OC[org])

    # Scale bar
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bar_x = xmin + (xmax - xmin) * 0.05
    bar_y = ymax - (ymax - ymin) * 0.05
    ax.plot([bar_x, bar_x + 500], [bar_y, bar_y], 'k-', lw=1)
    ax.text(bar_x + 250, bar_y - (ymax - ymin) * 0.03,
            '500 µm', ha='center', fontsize=5)

    if oi == 0:
        ax.text(-0.05, 1.05, 'a', transform=ax.transAxes,
                 fontsize=14, fontweight='bold')

# Legend below
leg_elements = [Patch(facecolor=NICHE_COLORS[n], label=n.replace('_', ' '))
                for n in UTAG_NICHES]
fig.legend(handles=leg_elements, loc='lower center',
           bbox_to_anchor=(0.5, 0.72), ncol=4, fontsize=5.5,
           handletextpad=0.3, columnspacing=0.8)


# ── (b) Niche frequency per organ (15-section) ────────────────────────
print("  Panel b: niche frequency per organ...")
ax_b = fig.add_subplot(gs[1, :])
niche_org = pd.crosstab(pred_df['organ'], pred_df['predicted_niche'],
                         normalize='index') * 100
niche_org = niche_org[UTAG_NICHES].reindex(index=ORGANS)

x = np.arange(len(UTAG_NICHES))
w = 0.26
for oi, org in enumerate(ORGANS):
    vals = niche_org.loc[org].values
    ax_b.bar(x + (oi - 1) * w, vals, w, color=OC[org], alpha=0.85,
              label=OL[org], edgecolor='white', linewidth=0.3)

ax_b.set_xticks(x)
ax_b.set_xticklabels([n.replace('_', ' ') for n in UTAG_NICHES],
                      rotation=35, ha='right', fontsize=6)
ax_b.set_ylabel('% of cells per organ', fontsize=6.5)
ax_b.set_title('UTAG niche frequency across organs (15 sections)',
               fontweight='bold')
ax_b.legend(fontsize=6, loc='upper right')
ax_b.text(-0.04, 1.08, 'b', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold')


# ── (c) Crypt-adjacent-immune composition ─────────────────────────────
print("  Panel c: crypt-adj-immune composition...")
ax_c = fig.add_subplot(gs[2, :])

# Top 8 cell types in CAI niche across organs
all_cts = set()
for org in ORGANS:
    if org in cai_comp:
        all_cts.update(cai_comp[org].head(10).index)

# Order by APE fraction
cts_ordered = sorted(all_cts,
                      key=lambda c: -(cai_comp.get('APE', pd.Series()).get(c, 0)))[:10]

x = np.arange(len(cts_ordered))
w = 0.26
for oi, org in enumerate(ORGANS):
    vals = [cai_comp.get(org, pd.Series()).get(c, 0) for c in cts_ordered]
    ax_c.bar(x + (oi - 1) * w, vals, w, color=OC[org], alpha=0.85,
              label=OL[org], edgecolor='white', linewidth=0.3)

ax_c.set_xticks(x)
ax_c.set_xticklabels([c.replace('Lymphoid-', 'Ly-').replace('Mesenchymal-', 'Mes-')
                      for c in cts_ordered],
                      rotation=25, ha='right', fontsize=6)
ax_c.set_ylabel('% of cells in\ncrypt-adjacent-immune niche', fontsize=6.5)
ax_c.set_title('Cellular composition of the crypt-adjacent-immune niche per organ',
                fontweight='bold')
ax_c.legend(fontsize=6, loc='upper right')
ax_c.text(-0.04, 1.08, 'c', transform=ax_c.transAxes,
          fontsize=14, fontweight='bold')


# ── (d) Plasma cell localization per organ ─────────────────────────────
print("  Panel d: plasma localization...")
ax_d = fig.add_subplot(gs[3, 0:2])

# For each organ, where are plasma cells located (by UTAG niche)?
plasma_dist = {}
for org in ORGANS:
    plasma_cells = pred_df[(pred_df['organ'] == org) &
                            (pred_df['broad'] == 'Lymphoid-Plasma')]
    if len(plasma_cells) < 10:
        continue
    dist = plasma_cells['predicted_niche'].value_counts(normalize=True) * 100
    plasma_dist[org] = dist

x = np.arange(len(UTAG_NICHES))
w = 0.26
for oi, org in enumerate(ORGANS):
    if org in plasma_dist:
        vals = [plasma_dist[org].get(n, 0) for n in UTAG_NICHES]
        ax_d.bar(x + (oi - 1) * w, vals, w, color=OC[org], alpha=0.85,
                  label=OL[org], edgecolor='white', linewidth=0.3)

ax_d.set_xticks(x)
ax_d.set_xticklabels([n.replace('_', ' ') for n in UTAG_NICHES],
                      rotation=35, ha='right', fontsize=6)
ax_d.set_ylabel('% of plasma cells', fontsize=6.5)
ax_d.set_title('Plasma cell localization across niches (15 sections)',
                fontweight='bold')
ax_d.legend(fontsize=6, loc='upper right')
ax_d.text(-0.06, 1.08, 'd', transform=ax_d.transAxes,
          fontsize=14, fontweight='bold')


# ── (e) Endothelial coupling ─────────────────────────────────────────
print("  Panel e: endothelial coupling...")
ax_e = fig.add_subplot(gs[3, 2])

endo_dist = {}
for org in ORGANS:
    endo_cells = pred_df[(pred_df['organ'] == org) &
                          (pred_df['broad'] == 'Endothelial')]
    if len(endo_cells) < 10:
        continue
    dist = endo_cells['predicted_niche'].value_counts(normalize=True) * 100
    endo_dist[org] = dist

show_niches = ['lymphoid_structure', 'crypt_adjacent_immune', 'submucosa',
               'muscularis_mucosa']
x = np.arange(len(show_niches))
w = 0.26
for oi, org in enumerate(ORGANS):
    if org in endo_dist:
        vals = [endo_dist[org].get(n, 0) for n in show_niches]
        ax_e.bar(x + (oi - 1) * w, vals, w, color=OC[org], alpha=0.85,
                  label=OL[org], edgecolor='white', linewidth=0.3)

ax_e.set_xticks(x)
ax_e.set_xticklabels([n.replace('_', ' ') for n in show_niches],
                      rotation=35, ha='right', fontsize=6)
ax_e.set_ylabel('% of endothelial cells', fontsize=6.5)
ax_e.set_title('Endothelial localization', fontweight='bold', fontsize=7)
ax_e.legend(fontsize=5, loc='upper right')
ax_e.text(-0.15, 1.08, 'e', transform=ax_e.transAxes,
          fontsize=14, fontweight='bold')


# Save
for ext in ['pdf', 'png']:
    fig.savefig(FIG_DIR / f"fig_utag_full_scale.{ext}",
                dpi=300, bbox_inches='tight')
    print(f"  Saved: figures/fig_utag_full_scale.{ext}")
plt.close()

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
