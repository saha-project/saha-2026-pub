"""v4: LN-GC4 vs APE-GC2 focused comparison — radial gradients, cross-organ overlay, DE + pathways."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.gc_utils import (
    load_adata, compute_fractional_radius, radial_bin_profile,
    nature_rcparams, apply_nature_style, GC_CELL_TYPE_PALETTE, PIXEL_SIZE_UM,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

FOCUS_GCS = ['LN-GC4', 'APE-GC2']
GC_MARKERS = ['CXCR4', 'CXCR5', 'CD83', 'CXCL13', 'IRF4', 'MYC', 'BCL2', 'CD38', 'MS4A1', 'CD19', 'CD79A']

GC_GENE_SETS = {
    'B_cell_activation': ['CD19', 'CD79A', 'MS4A1', 'CD38', 'IRF4', 'CXCR5', 'CD83', 'TNFRSF13B', 'TNFRSF13C', 'CD40'],
    'GC_DZ_program': ['CXCR4', 'BCL2', 'MYC', 'FOXO1', 'BACH2', 'E2F1', 'PCNA', 'MKI67', 'TOP2A'],
    'GC_LZ_program': ['CXCR5', 'CD83', 'IRF4', 'CD38', 'NFKB1', 'NFKB2', 'REL', 'RELA'],
    'Cytokine_signaling': ['IL21R', 'IL4R', 'IL6', 'IL10', 'IFNG', 'TGFB1', 'TNF', 'TNFSF13B', 'IL2RG'],
    'Apoptosis': ['BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1', 'FAS', 'FASLG', 'CASP3', 'CASP8', 'BID'],
    'Antigen_presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DPA1', 'HLA-DPB1', 'CD74', 'B2M', 'TAP1', 'TAP2'],
    'PC_differentiation': ['IRF4', 'XBP1', 'SDC1', 'MZB1', 'IGHG1', 'IGHG2', 'IGHA1', 'JCHAIN'],
}

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)

adata.obs['gc_id'] = obs_assign['gc_id'].values
adata.obs['gc_member'] = obs_assign['gc_member'].values

# ============================================================
# Part 1: Radial gradients for LN-GC4 and APE-GC2
# ============================================================
print("\n=== Radial Gradients (LN-GC4 vs APE-GC2) ===")

frac_radius = compute_fractional_radius(obs_assign)
obs_assign['frac_radius'] = frac_radius
obs_assign['cell_types_insitutype'] = adata.obs['cell_types_insitutype'].values

# Subset to focus GCs
focus_mask = obs_assign['gc_id'].isin(FOCUS_GCS) & (obs_assign['gc_member'] == True)
df_focus = obs_assign[focus_mask].copy()
print(f"  LN-GC4: {(df_focus['gc_id'] == 'LN-GC4').sum()} cells")
print(f"  APE-GC2: {(df_focus['gc_id'] == 'APE-GC2').sum()} cells")

profiles = radial_bin_profile(df_focus, gc_id_col='gc_id', n_bins=5)
profiles.to_csv(os.path.join(OUTPUTS_DIR, 'radial_profiles_v4.csv'), index=False)

BIN_ORDER = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
bin_centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
CELL_TYPE_ORDER = ['GCBC', 'NBC/MBC', 'FDC', 'PC', 'TC_CD4', 'TC_CD8', 'macro', 'DC', 'PDC', 'ILC', 'gran', 'mono', 'mast']
TOP_TYPES = ['GCBC', 'FDC', 'NBC/MBC', 'PC', 'TC_CD4', 'TC_CD8']

plt.rcParams.update(nature_rcparams())

# --- Figure: Stacked bar + line for each GC ---
fig_rad, axes_rad = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)

for col_idx, gc_id in enumerate(FOCUS_GCS):
    gc_prof = profiles[profiles['gc_id'] == gc_id]
    n_cells = int(df_focus[df_focus['gc_id'] == gc_id].shape[0])

    # Stacked bar
    ax_bar = axes_rad[0, col_idx]
    bottom = np.zeros(5)
    for ct in CELL_TYPE_ORDER:
        ct_data = gc_prof[gc_prof['cell_type'] == ct].set_index('bin').reindex(BIN_ORDER).fillna(0)
        vals = ct_data['proportion'].values
        ax_bar.bar(range(5), vals, bottom=bottom, color=GC_CELL_TYPE_PALETTE.get(ct, '#AAA'), width=0.8)
        bottom += vals
    ax_bar.set_xticks(range(5))
    ax_bar.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=4, rotation=45, ha='right')
    ax_bar.set_ylabel('Proportion', fontsize=6)
    ax_bar.set_title(f'{gc_id} (n={n_cells:,})', fontsize=7)
    ax_bar.set_ylim(0, 1)
    apply_nature_style(ax_bar)

    # Line plot
    ax_line = axes_rad[1, col_idx]
    for ct in TOP_TYPES:
        ct_data = gc_prof[gc_prof['cell_type'] == ct].set_index('bin').reindex(BIN_ORDER).fillna(0)
        ax_line.plot(bin_centers, ct_data['proportion'].values, color=GC_CELL_TYPE_PALETTE.get(ct, '#AAA'),
                     lw=1.2, marker='o', markersize=2, label=ct)
    ax_line.set_xticks(bin_centers)
    ax_line.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=4, rotation=45, ha='right')
    ax_line.set_ylabel('Proportion', fontsize=6)
    ax_line.set_xlabel('Fractional radius', fontsize=6)
    if col_idx == 1:
        ax_line.legend(fontsize=4, ncol=1, frameon=False, loc='upper right')
    apply_nature_style(ax_line)

axes_rad[0, 0].text(-0.15, 1.05, 'A', transform=axes_rad[0, 0].transAxes, fontsize=8, fontweight='bold', va='top')
axes_rad[0, 1].text(-0.15, 1.05, 'B', transform=axes_rad[0, 1].transAxes, fontsize=8, fontweight='bold', va='top')
axes_rad[1, 0].text(-0.15, 1.05, 'C', transform=axes_rad[1, 0].transAxes, fontsize=8, fontweight='bold', va='top')
axes_rad[1, 1].text(-0.15, 1.05, 'D', transform=axes_rad[1, 1].transAxes, fontsize=8, fontweight='bold', va='top')

legend_handles = [mpatches.Patch(color=GC_CELL_TYPE_PALETTE.get(ct, '#AAA'), label=ct) for ct in CELL_TYPE_ORDER]
fig_rad.legend(handles=legend_handles, loc='lower center', ncol=7, fontsize=4,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

fig_rad.savefig(os.path.join(FIGURES_DIR, '03_radial_gradients_v4.png'), dpi=300, bbox_inches='tight')
fig_rad.savefig(os.path.join(FIGURES_DIR, '03_radial_gradients_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 03_radial_gradients_v4.png/pdf")

# --- Cross-organ overlay for LN-GC4 vs APE-GC2 ---
fig_cross, axes_cross = plt.subplots(2, 3, figsize=(7.2, 4.5), constrained_layout=True)
plt.rcParams.update(nature_rcparams())

STYLES = {
    'LN-GC4': {'color': '#2A6365', 'ls': '-', 'label': 'LN-GC4'},
    'APE-GC2': {'color': '#8B0000', 'ls': '--', 'label': 'APE-GC2'},
}

for ct_idx, ct in enumerate(TOP_TYPES):
    r, c = divmod(ct_idx, 3)
    ax = axes_cross[r, c]
    for gc_id, style in STYLES.items():
        ct_data = profiles[(profiles['gc_id'] == gc_id) & (profiles['cell_type'] == ct)]
        ct_data = ct_data.set_index('bin').reindex(BIN_ORDER).fillna(0)
        ax.plot(bin_centers, ct_data['proportion'].values, color=style['color'],
                ls=style['ls'], lw=1.2, marker='o', markersize=2, label=style['label'])
    ax.set_title(ct, fontsize=7, color=GC_CELL_TYPE_PALETTE.get(ct, '#333'))
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=4, rotation=45, ha='right')
    ax.set_ylabel('Proportion', fontsize=5)
    ax.set_xlabel('Fractional radius', fontsize=5)
    if ct_idx == 0:
        ax.legend(fontsize=4, frameon=False)
    apply_nature_style(ax)

fig_cross.savefig(os.path.join(FIGURES_DIR, '03_radial_cross_organ_v4.png'), dpi=300, bbox_inches='tight')
fig_cross.savefig(os.path.join(FIGURES_DIR, '03_radial_cross_organ_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 03_radial_cross_organ_v4.png/pdf")

# ============================================================
# Part 2: Subsetted DE — GCBC in LN-GC4 vs APE-GC2 only
# ============================================================
print("\n=== DE: GCBC in LN-GC4 vs APE-GC2 ===")

gcbc_mask = (
    adata.obs['gc_id'].isin(FOCUS_GCS) &
    (adata.obs['cell_types_insitutype'] == 'GCBC')
)
adata_gcbc = adata[gcbc_mask].copy()
adata_gcbc.obs['organ'] = adata_gcbc.obs['SAHA_organcode'].values
print(f"  LN-GC4 GCBC: {(adata_gcbc.obs['organ'] == 'LN').sum()}")
print(f"  APE-GC2 GCBC: {(adata_gcbc.obs['organ'] == 'APE').sum()}")

sc.pp.normalize_total(adata_gcbc, target_sum=1e4)
sc.pp.log1p(adata_gcbc)

sc.tl.rank_genes_groups(adata_gcbc, groupby='organ', groups=['APE'], reference='LN',
                        method='wilcoxon', use_raw=False)

result = sc.get.rank_genes_groups_df(adata_gcbc, group='APE')
result = result.rename(columns={'names': 'gene', 'logfoldchanges': 'log2fc',
                                'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
result['neg_log10_padj'] = -np.log10(result['pval_adj'].clip(lower=1e-300))
result['abs_log2fc'] = result['log2fc'].abs()

result.to_csv(os.path.join(OUTPUTS_DIR, 'de_results_GCBC_v4.csv'), index=False)

sig = result[(result['pval_adj'] < 0.05) & (result['log2fc'].abs() > 0.5)]
print(f"  Significant: {len(sig)} (APE-up: {(sig['log2fc'] > 0).sum()}, LN-up: {(sig['log2fc'] < 0).sum()})")

# Top hits
print("\n  Top 10 by |log2fc|:")
for _, r in result.nlargest(10, 'abs_log2fc').iterrows():
    d = "APE↑" if r['log2fc'] > 0 else "LN↑"
    print(f"    {r['gene']:12s} log2fc={r['log2fc']:+.2f} padj={r['pval_adj']:.2e} {d}")

# --- Volcano plot ---
print("\nGenerating v4 volcano...")
plt.rcParams.update(nature_rcparams())
fig_v, ax_v = plt.subplots(1, 1, figsize=(4.5, 4.0))

colors_v = np.full(len(result), '#CCCCCC')
sig_up = (result['pval_adj'] < 0.05) & (result['log2fc'] > 0.5)
sig_down = (result['pval_adj'] < 0.05) & (result['log2fc'] < -0.5)
colors_v[sig_up.values] = '#8B0000'
colors_v[sig_down.values] = '#2A6365'

ax_v.scatter(result['log2fc'], result['neg_log10_padj'], c=colors_v, s=3,
             linewidths=0, rasterized=True, alpha=0.7)
ax_v.axhline(-np.log10(0.05), color='#999', lw=0.3, ls='--')
ax_v.axvline(0.5, color='#999', lw=0.3, ls='--')
ax_v.axvline(-0.5, color='#999', lw=0.3, ls='--')

labeled = set()
for _, r in result.nlargest(8, 'abs_log2fc').iterrows():
    ax_v.annotate(r['gene'], (r['log2fc'], r['neg_log10_padj']),
                  fontsize=3.5, ha='center', va='bottom', textcoords='offset points', xytext=(0, 2))
    labeled.add(r['gene'])
for gene in GC_MARKERS:
    if gene in labeled:
        continue
    row = result[result['gene'] == gene]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    ax_v.annotate(r['gene'], (r['log2fc'], r['neg_log10_padj']),
                  fontsize=3.5, ha='center', va='bottom', color='#555', fontstyle='italic',
                  textcoords='offset points', xytext=(0, 2))

ax_v.set_xlabel('log₂FC (APE-GC2 vs LN-GC4)', fontsize=7)
ax_v.set_ylabel('-log₁₀(adj. p-value)', fontsize=7)
ax_v.set_title('GCBC DE: APE-GC2 vs LN-GC4', fontsize=7)
apply_nature_style(ax_v)

fig_v.savefig(os.path.join(FIGURES_DIR, '04_gcbc_volcano_v4.png'), dpi=300, bbox_inches='tight')
fig_v.savefig(os.path.join(FIGURES_DIR, '04_gcbc_volcano_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 04_gcbc_volcano_v4.png/pdf")

# ============================================================
# Part 3: Pathway enrichment on v4 DE results
# ============================================================
print("\n=== Pathway enrichment (v4) ===")

sig_up_genes = set(result[(result['pval_adj'] < 0.05) & (result['log2fc'] > 0.5)]['gene'])
sig_down_genes = set(result[(result['pval_adj'] < 0.05) & (result['log2fc'] < -0.5)]['gene'])
all_panel = set(adata.var_names)
N_panel = len(all_panel)

enrichment_records = []
for direction, sig_genes, dir_label in [('APE_up', sig_up_genes, 'Up in APE-GC2'),
                                         ('LN_up', sig_down_genes, 'Up in LN-GC4')]:
    n_sig = len(sig_genes)
    for pw_name, pw_genes in GC_GENE_SETS.items():
        pw_in_panel = set(pw_genes) & all_panel
        if not pw_in_panel:
            continue
        overlap = sig_genes & pw_in_panel
        a = len(overlap)
        b = n_sig - a
        c = len(pw_in_panel) - a
        d = N_panel - a - b - c
        odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
        enrichment_records.append({
            'direction': dir_label, 'pathway': pw_name,
            'n_pathway_in_panel': len(pw_in_panel), 'n_overlap': a,
            'overlap_genes': ', '.join(sorted(overlap)) if overlap else '',
            'odds_ratio': round(odds, 2), 'pval': pval,
        })

enr_df = pd.DataFrame(enrichment_records).sort_values('pval')
enr_df.to_csv(os.path.join(OUTPUTS_DIR, 'pathway_enrichment_v4.csv'), index=False)
print("Saved: pathway_enrichment_v4.csv")

print("\n  Pathway results (v4):")
for _, r in enr_df.iterrows():
    sig_mark = " *" if r['pval'] < 0.05 else ""
    print(f"    {r['direction']:18s} {r['pathway']:25s} overlap={r['n_overlap']}/{r['n_pathway_in_panel']} "
          f"OR={r['odds_ratio']:.1f} p={r['pval']:.3f}{sig_mark}")

# --- Pathway barplot ---
plt.rcParams.update(nature_rcparams())
fig_pw, ax_pw = plt.subplots(1, 1, figsize=(4.5, 3.5))
enr_df['neg_log10_p'] = -np.log10(enr_df['pval'].clip(lower=1e-10))
enr_df['label'] = enr_df['direction'] + ': ' + enr_df['pathway']
pw_sorted = enr_df.sort_values('neg_log10_p', ascending=True)

colors_pw = ['#8B0000' if 'APE' in d else '#2A6365' for d in pw_sorted['direction']]
ax_pw.barh(range(len(pw_sorted)), pw_sorted['neg_log10_p'].values, color=colors_pw, height=0.6)
ax_pw.set_yticks(range(len(pw_sorted)))
ax_pw.set_yticklabels(pw_sorted['label'].values, fontsize=3.5)
ax_pw.axvline(-np.log10(0.05), color='#999', lw=0.3, ls='--')
ax_pw.set_xlabel('-log₁₀(p)', fontsize=6)
ax_pw.set_title('Pathway enrichment: APE-GC2 vs LN-GC4 (Fisher\'s exact)', fontsize=7)
apply_nature_style(ax_pw)

fig_pw.savefig(os.path.join(FIGURES_DIR, '04_pathway_barplot_v4.png'), dpi=300, bbox_inches='tight')
fig_pw.savefig(os.path.join(FIGURES_DIR, '04_pathway_barplot_v4.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("Saved: 04_pathway_barplot_v4.png/pdf")

print("\n=== v4 comparison complete ===")
