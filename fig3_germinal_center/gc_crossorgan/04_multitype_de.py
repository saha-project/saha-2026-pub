"""Phase 4 Plan 02: Multi-cell-type DE, dot plots, and pathway enrichment."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style, GC_CELL_TYPE_PALETTE

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
EXCLUDE_GCS = {'LN-GC1', 'LN-GC2'}

GC_MARKERS = ['CXCR4', 'CXCR5', 'CD83', 'CXCL13', 'IRF4', 'MYC', 'BCL2', 'CD38',
              'MS4A1', 'CD19', 'CD79A']

# Curated GC-relevant gene sets (restricted to what may be in 889-gene panel)
GC_GENE_SETS = {
    'B_cell_activation': ['CD19', 'CD79A', 'MS4A1', 'CD38', 'IRF4', 'CXCR5', 'CD83', 'TNFRSF13B', 'TNFRSF13C', 'CD40'],
    'GC_DZ_program': ['CXCR4', 'BCL2', 'MYC', 'FOXO1', 'BACH2', 'E2F1', 'PCNA', 'MKI67', 'TOP2A'],
    'GC_LZ_program': ['CXCR5', 'CD83', 'IRF4', 'CD38', 'NFKB1', 'NFKB2', 'REL', 'RELA'],
    'Cytokine_signaling': ['IL21R', 'IL4R', 'IL6', 'IL10', 'IFNG', 'TGFB1', 'TNF', 'TNFSF13B', 'IL2RG'],
    'Apoptosis': ['BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1', 'FAS', 'FASLG', 'CASP3', 'CASP8', 'BID'],
    'Class_switch_SHM': ['AICDA', 'UNG', 'MSH2', 'MSH6', 'EXO1', 'APEX1', 'POLH', 'POLD1'],
    'Antigen_presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DPA1', 'HLA-DPB1', 'CD74', 'B2M', 'TAP1', 'TAP2'],
    'PC_differentiation': ['IRF4', 'XBP1', 'SDC1', 'MZB1', 'IGHG1', 'IGHG2', 'IGHA1', 'JCHAIN'],
}

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)

adata.obs['gc_id'] = obs_assign['gc_id'].values
adata.obs['gc_member'] = obs_assign['gc_member'].values

# === Part 1: Multi-cell-type DE ===
DE_GROUPS = [
    ('FDC', 'FDC'),
    ('TC_CD4', 'TC_CD4'),
    ('GC_all', None),  # all GC-member cells pooled
]

for label, cell_type in DE_GROUPS:
    print(f"\n=== DE: {label} ===")

    if cell_type:
        mask = (
            (adata.obs['gc_member'] == True) &
            (adata.obs['cell_types_insitutype'] == cell_type) &
            (~adata.obs['gc_id'].isin(EXCLUDE_GCS))
        )
    else:
        mask = (
            (adata.obs['gc_member'] == True) &
            (~adata.obs['gc_id'].isin(EXCLUDE_GCS))
        )

    adata_sub = adata[mask].copy()
    adata_sub.obs['organ'] = adata_sub.obs['SAHA_organcode'].values

    n_ln = (adata_sub.obs['organ'] == 'LN').sum()
    n_ape = (adata_sub.obs['organ'] == 'APE').sum()
    print(f"  LN: {n_ln}, APE: {n_ape}")

    if n_ln < 5 or n_ape < 5:
        print(f"  WARNING: Too few cells (<5) in one group. Skipping DE for {label}.")
        # Save empty results
        pd.DataFrame(columns=['gene', 'log2fc', 'pval', 'pval_adj', 'score']).to_csv(
            os.path.join(OUTPUTS_DIR, f'de_results_{label}.csv'), index=False)
        continue

    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    sc.tl.rank_genes_groups(adata_sub, groupby='organ', groups=['APE'], reference='LN',
                            method='wilcoxon', use_raw=False)

    result = sc.get.rank_genes_groups_df(adata_sub, group='APE')
    result = result.rename(columns={
        'names': 'gene', 'logfoldchanges': 'log2fc',
        'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score',
    })

    csv_path = os.path.join(OUTPUTS_DIR, f'de_results_{label}.csv')
    result.to_csv(csv_path, index=False)

    sig = result[(result['pval_adj'] < 0.05) & (result['log2fc'].abs() > 0.5)]
    print(f"  Significant (padj<0.05, |log2fc|>0.5): {len(sig)}")
    print(f"  Saved: {csv_path}")

# === Part 2: Dot plot 1 — Cell type x GC marker matrix ===
print("\n=== Generating dot plot: cell types x GC markers ===")
plt.rcParams.update(nature_rcparams())

# Subset to GC members, exclude small GCs
gc_mask = (adata.obs['gc_member'] == True) & (~adata.obs['gc_id'].isin(EXCLUDE_GCS))
adata_gc = adata[gc_mask].copy()
adata_gc.obs['organ'] = adata_gc.obs['SAHA_organcode'].values

# Create organ_celltype grouping
adata_gc.obs['organ_ct'] = adata_gc.obs['organ'].astype(str) + '_' + adata_gc.obs['cell_types_insitutype'].astype(str)

# Normalize for visualization
sc.pp.normalize_total(adata_gc, target_sum=1e4)
sc.pp.log1p(adata_gc)

# Filter to markers present in panel
markers_present = [g for g in GC_MARKERS if g in adata_gc.var_names]
print(f"  Markers in panel: {markers_present}")

# Cell types to show
CT_ORDER = ['GCBC', 'FDC', 'NBC/MBC', 'PC', 'TC_CD4']
groups_order = []
for organ in ['LN', 'APE']:
    for ct in CT_ORDER:
        key = f'{organ}_{ct}'
        if key in adata_gc.obs['organ_ct'].values:
            groups_order.append(key)

# Build manual dot plot: compute mean expr and pct expressing per group
import scipy.sparse as sp

adata_gc_sub = adata_gc[adata_gc.obs['cell_types_insitutype'].isin(CT_ORDER)].copy()
adata_gc_sub.obs['organ_ct'] = (adata_gc_sub.obs['organ'].astype(str) + '_' +
                                 adata_gc_sub.obs['cell_types_insitutype'].astype(str))

groups_order_filtered = []
for organ in ['LN', 'APE']:
    for ct in CT_ORDER:
        key = f'{organ}_{ct}'
        if key in adata_gc_sub.obs['organ_ct'].values:
            groups_order_filtered.append(key)

dot_data = []
for grp in groups_order_filtered:
    mask = adata_gc_sub.obs['organ_ct'] == grp
    X_sub = adata_gc_sub[mask].X
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    for gi, gene in enumerate(markers_present):
        gene_idx = list(adata_gc_sub.var_names).index(gene)
        vals = X_sub[:, gene_idx]
        mean_expr = float(vals.mean())
        pct = float((vals > 0).mean())
        dot_data.append({'group': grp, 'gene': gene, 'mean_expr': mean_expr, 'pct': pct})

dot_df = pd.DataFrame(dot_data)

fig_dp, ax_dp = plt.subplots(1, 1, figsize=(6, 3.5))
for i, grp in enumerate(groups_order_filtered):
    for j, gene in enumerate(markers_present):
        row = dot_df[(dot_df['group'] == grp) & (dot_df['gene'] == gene)].iloc[0]
        size = row['pct'] * 200
        color_val = row['mean_expr']
        ax_dp.scatter(j, i, s=size, c=[[color_val]], cmap='Reds', vmin=0,
                     vmax=dot_df['mean_expr'].quantile(0.95), edgecolors='#333', linewidths=0.3)

ax_dp.set_xticks(range(len(markers_present)))
ax_dp.set_xticklabels(markers_present, fontsize=5, rotation=45, ha='right')
ax_dp.set_yticks(range(len(groups_order_filtered)))
ax_dp.set_yticklabels(groups_order_filtered, fontsize=5)
ax_dp.set_title('GC marker expression by cell type and organ', fontsize=7)
apply_nature_style(ax_dp)

fig_dp.savefig(os.path.join(FIGURES_DIR, '04_dotplot_markers.png'), dpi=300, bbox_inches='tight')
fig_dp.savefig(os.path.join(FIGURES_DIR, '04_dotplot_markers.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: 04_dotplot_markers.png/pdf")

# === Part 3: Dot plot 2 — GCBC only, LN vs APE ===
print("\n=== Generating dot plot: GCBC LN vs APE ===")
gcbc_mask = (
    (adata.obs['gc_member'] == True) &
    (adata.obs['cell_types_insitutype'] == 'GCBC') &
    (~adata.obs['gc_id'].isin(EXCLUDE_GCS))
)
adata_gcbc = adata[gcbc_mask].copy()
adata_gcbc.obs['organ'] = adata_gcbc.obs['SAHA_organcode'].values
sc.pp.normalize_total(adata_gcbc, target_sum=1e4)
sc.pp.log1p(adata_gcbc)

# Manual GCBC dot plot
dot_data2 = []
for organ in ['LN', 'APE']:
    mask = adata_gcbc.obs['organ'] == organ
    X_sub = adata_gcbc[mask].X
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    for gene in markers_present:
        gene_idx = list(adata_gcbc.var_names).index(gene)
        vals = X_sub[:, gene_idx]
        dot_data2.append({'group': organ, 'gene': gene,
                          'mean_expr': float(vals.mean()), 'pct': float((vals > 0).mean())})

dot_df2 = pd.DataFrame(dot_data2)

fig_dp2, ax_dp2 = plt.subplots(1, 1, figsize=(5, 1.8))
for i, organ in enumerate(['LN', 'APE']):
    for j, gene in enumerate(markers_present):
        row = dot_df2[(dot_df2['group'] == organ) & (dot_df2['gene'] == gene)].iloc[0]
        ax_dp2.scatter(j, i, s=row['pct'] * 200, c=[[row['mean_expr']]], cmap='Reds',
                      vmin=0, vmax=dot_df2['mean_expr'].quantile(0.95),
                      edgecolors='#333', linewidths=0.3)

ax_dp2.set_xticks(range(len(markers_present)))
ax_dp2.set_xticklabels(markers_present, fontsize=5, rotation=45, ha='right')
ax_dp2.set_yticks([0, 1])
ax_dp2.set_yticklabels(['LN', 'APE'], fontsize=6)
ax_dp2.set_title('GCBC: GC marker expression (LN vs APE)', fontsize=7)
apply_nature_style(ax_dp2)

fig_dp2.savefig(os.path.join(FIGURES_DIR, '04_dotplot_gcbc_organs.png'), dpi=300, bbox_inches='tight')
fig_dp2.savefig(os.path.join(FIGURES_DIR, '04_dotplot_gcbc_organs.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: 04_dotplot_gcbc_organs.png/pdf")

# === Part 4: Pathway enrichment (Fisher's exact) ===
print("\n=== Pathway enrichment (Fisher's exact) ===")

# Load GCBC DE results
gcbc_de = pd.read_csv(os.path.join(OUTPUTS_DIR, 'de_results_GCBC.csv'))
sig_genes_up = set(gcbc_de[(gcbc_de['pval_adj'] < 0.05) & (gcbc_de['log2fc'] > 0.5)]['gene'])
sig_genes_down = set(gcbc_de[(gcbc_de['pval_adj'] < 0.05) & (gcbc_de['log2fc'] < -0.5)]['gene'])
all_panel_genes = set(adata.var_names)
N_panel = len(all_panel_genes)

print(f"  Sig up (APE): {len(sig_genes_up)}, Sig down (LN): {len(sig_genes_down)}")

enrichment_records = []
for direction, sig_genes, dir_label in [('APE_up', sig_genes_up, 'Up in APE'),
                                         ('LN_up', sig_genes_down, 'Up in LN')]:
    n_sig = len(sig_genes)
    for pathway_name, pathway_genes in GC_GENE_SETS.items():
        # Restrict to genes in panel
        pathway_in_panel = set(pathway_genes) & all_panel_genes
        if len(pathway_in_panel) == 0:
            continue

        overlap = sig_genes & pathway_in_panel

        # 2x2 contingency: overlap, sig-not-pathway, pathway-not-sig, neither
        a = len(overlap)
        b = n_sig - a
        c = len(pathway_in_panel) - a
        d = N_panel - a - b - c

        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative='greater')

        enrichment_records.append({
            'direction': dir_label,
            'pathway': pathway_name,
            'n_pathway_in_panel': len(pathway_in_panel),
            'n_overlap': a,
            'overlap_genes': ', '.join(sorted(overlap)) if overlap else '',
            'odds_ratio': round(odds_ratio, 2),
            'pval': pval,
            'genes_in_panel': ', '.join(sorted(pathway_in_panel)),
        })

enrichment_df = pd.DataFrame(enrichment_records).sort_values('pval')
enrichment_df.to_csv(os.path.join(OUTPUTS_DIR, 'pathway_enrichment.csv'), index=False)
print(f"  Saved: pathway_enrichment.csv")

# Print results
print("\n  Pathway enrichment results:")
for _, r in enrichment_df.iterrows():
    sig_marker = " *" if r['pval'] < 0.05 else ""
    print(f"    {r['direction']:12s} {r['pathway']:25s} "
          f"overlap={r['n_overlap']}/{r['n_pathway_in_panel']} "
          f"OR={r['odds_ratio']:.1f} p={r['pval']:.3f}{sig_marker}")

# === Pathway barplot ===
print("\n=== Generating pathway barplot ===")
plt.rcParams.update(nature_rcparams())

fig_pw, ax_pw = plt.subplots(1, 1, figsize=(4.5, 3.5))

# Show all pathways, color by direction
enr_plot = enrichment_df.copy()
enr_plot['neg_log10_p'] = -np.log10(enr_plot['pval'].clip(lower=1e-10))
enr_plot['label'] = enr_plot['direction'] + ': ' + enr_plot['pathway']

# Sort by -log10(p)
enr_plot = enr_plot.sort_values('neg_log10_p', ascending=True)

colors = ['#8B0000' if 'APE' in d else '#2A6365' for d in enr_plot['direction']]
ax_pw.barh(range(len(enr_plot)), enr_plot['neg_log10_p'].values, color=colors, height=0.6)
ax_pw.set_yticks(range(len(enr_plot)))
ax_pw.set_yticklabels(enr_plot['label'].values, fontsize=4)
ax_pw.axvline(-np.log10(0.05), color='#999', lw=0.3, ls='--', label='p=0.05')
ax_pw.set_xlabel('-log₁₀(p-value)', fontsize=6)
ax_pw.set_title('Pathway enrichment (Fisher\'s exact, 889-gene background)', fontsize=7)
ax_pw.legend(fontsize=5, frameon=False)
apply_nature_style(ax_pw)

fig_pw.savefig(os.path.join(FIGURES_DIR, '04_pathway_barplot.png'), dpi=300, bbox_inches='tight')
fig_pw.savefig(os.path.join(FIGURES_DIR, '04_pathway_barplot.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: 04_pathway_barplot.png/pdf")

print("\nMulti-type DE + dot plots + pathways complete.")
