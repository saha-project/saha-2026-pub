"""Phase 4 Plan 01: GCBC cross-organ DE + volcano plot."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.gc_utils import load_adata, nature_rcparams, apply_nature_style

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
EXCLUDE_GCS = {'LN-GC1', 'LN-GC2'}

print("Loading data...")
adata = load_adata()
obs_assign = pd.read_csv(os.path.join(OUTPUTS_DIR, 'gc_obs_assignments.csv'), index_col=0)
obs_assign.index = obs_assign.index.astype(str)

# Merge GC assignments
adata.obs['gc_id'] = obs_assign['gc_id'].values
adata.obs['gc_member'] = obs_assign['gc_member'].values

# Subset: GCBC within GCs, excluding small LN GCs
gcbc_mask = (
    (adata.obs['gc_member'] == True) &
    (adata.obs['cell_types_insitutype'] == 'GCBC') &
    (~adata.obs['gc_id'].isin(EXCLUDE_GCS))
)
adata_gcbc = adata[gcbc_mask].copy()

# Add organ label for DE grouping
adata_gcbc.obs['organ'] = adata_gcbc.obs['SAHA_organcode'].values
print(f"GCBC cells for DE: LN={((adata_gcbc.obs['organ']=='LN').sum())}, APE={((adata_gcbc.obs['organ']=='APE').sum())}")

# Normalize a copy for DE (raw counts → log-normalized)
# adata.X is raw counts — confirmed in Phase 1
sc.pp.normalize_total(adata_gcbc, target_sum=1e4)
sc.pp.log1p(adata_gcbc)

# Run Wilcoxon rank-sum DE: APE vs LN
print("Running Wilcoxon rank-sum DE...")
sc.tl.rank_genes_groups(adata_gcbc, groupby='organ', groups=['APE'], reference='LN',
                        method='wilcoxon', use_raw=False)

# Extract results
result = sc.get.rank_genes_groups_df(adata_gcbc, group='APE')
result = result.rename(columns={
    'names': 'gene',
    'logfoldchanges': 'log2fc',
    'pvals': 'pval',
    'pvals_adj': 'pval_adj',
    'scores': 'score',
})

# scanpy reports natural log fold changes — convert to log2
result['log2fc'] = result['log2fc'] / np.log(2) * np.log(np.e)
# Actually scanpy already uses log2 internally in recent versions, but let's verify
# The logfoldchanges from rank_genes_groups are base-e, so convert:
# Actually in scanpy >= 1.9, logfoldchanges are log2. Let's keep as-is and check range
print(f"log2fc range: [{result['log2fc'].min():.2f}, {result['log2fc'].max():.2f}]")

result['neg_log10_padj'] = -np.log10(result['pval_adj'].clip(lower=1e-300))

# Add pct expressing
pct_ape = (adata_gcbc[adata_gcbc.obs['organ'] == 'APE'].X.toarray() > 0).mean(axis=0)
pct_ln = (adata_gcbc[adata_gcbc.obs['organ'] == 'LN'].X.toarray() > 0).mean(axis=0)
gene_to_pct_ape = dict(zip(adata_gcbc.var_names, pct_ape.flat))
gene_to_pct_ln = dict(zip(adata_gcbc.var_names, pct_ln.flat))
result['pct_APE'] = result['gene'].map(gene_to_pct_ape)
result['pct_LN'] = result['gene'].map(gene_to_pct_ln)

result.to_csv(os.path.join(OUTPUTS_DIR, 'de_results_GCBC.csv'), index=False)
print(f"Saved: de_results_GCBC.csv ({len(result)} genes)")

# Significant genes
sig = result[(result['pval_adj'] < 0.05) & (result['log2fc'].abs() > 0.5)]
print(f"Significant (padj<0.05, |log2fc|>0.5): {len(sig)} genes")
print(f"  Up in APE: {(sig['log2fc'] > 0).sum()}")
print(f"  Up in LN: {(sig['log2fc'] < 0).sum()}")

# Top hits
print("\nTop 10 by |log2fc|:")
top10 = result.nlargest(10, 'log2fc', keep='first')
top10_neg = result.nsmallest(10, 'log2fc', keep='first')
for _, r in pd.concat([top10, top10_neg]).iterrows():
    direction = "APE↑" if r['log2fc'] > 0 else "LN↑"
    print(f"  {r['gene']:12s} log2fc={r['log2fc']:+.2f} padj={r['pval_adj']:.2e} {direction}")

# === Volcano plot ===
print("\nGenerating volcano plot...")
plt.rcParams.update(nature_rcparams())

KNOWN_GC_MARKERS = ['CXCR4', 'CXCR5', 'CD83', 'IRF4', 'MYC', 'BCL2', 'CD38', 'CXCL13', 'MS4A1', 'CD19', 'CD79A']

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))

# Color by significance
colors = np.full(len(result), '#CCCCCC')  # not significant
sig_up = (result['pval_adj'] < 0.05) & (result['log2fc'] > 0.5)
sig_down = (result['pval_adj'] < 0.05) & (result['log2fc'] < -0.5)
colors[sig_up.values] = '#8B0000'   # APE up — dark red (appendix color)
colors[sig_down.values] = '#2A6365' # LN up — teal (LN color)

ax.scatter(result['log2fc'], result['neg_log10_padj'],
           c=colors, s=4, linewidths=0, rasterized=True, alpha=0.7)

# Threshold lines
ax.axhline(-np.log10(0.05), color='#999', lw=0.3, ls='--')
ax.axvline(0.5, color='#999', lw=0.3, ls='--')
ax.axvline(-0.5, color='#999', lw=0.3, ls='--')

# Label top genes by |log2fc|
result['abs_log2fc'] = result['log2fc'].abs()
top_by_fc = result.nlargest(10, 'abs_log2fc')

labeled_genes = set()
for _, r in top_by_fc.iterrows():
    ax.annotate(r['gene'], (r['log2fc'], r['neg_log10_padj']),
                fontsize=4, ha='center', va='bottom',
                textcoords='offset points', xytext=(0, 3))
    labeled_genes.add(r['gene'])

# Label known GC markers (if not already labeled)
for gene in KNOWN_GC_MARKERS:
    if gene in labeled_genes:
        continue
    gene_row = result[result['gene'] == gene]
    if len(gene_row) == 0:
        continue
    r = gene_row.iloc[0]
    ax.annotate(r['gene'], (r['log2fc'], r['neg_log10_padj']),
                fontsize=4, ha='center', va='bottom', color='#555555',
                textcoords='offset points', xytext=(0, 3),
                fontstyle='italic')
    labeled_genes.add(gene)

ax.set_xlabel('log₂ fold change (APE vs LN)', fontsize=7)
ax.set_ylabel('-log₁₀(adjusted p-value)', fontsize=7)
ax.set_title('GCBC Differential Expression: Appendix vs Lymph Node', fontsize=7)

# Add annotation for sides
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[1] * 0.85, ylim[1] * 0.95, 'APE ↑', fontsize=6, color='#8B0000', ha='center', va='top')
ax.text(xlim[0] * 0.85, ylim[1] * 0.95, 'LN ↑', fontsize=6, color='#2A6365', ha='center', va='top')

apply_nature_style(ax)

png_path = os.path.join(FIGURES_DIR, '04_gcbc_volcano.png')
pdf_path = os.path.join(FIGURES_DIR, '04_gcbc_volcano.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("\nGCBC DE + volcano complete.")
