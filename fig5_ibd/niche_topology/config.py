"""Configuration for IBD_cohort_comp — cross-cohort/cross-platform comparison.

All paths absolute. Reuses palettes/conventions from sibling IBD_Exploration project.
"""
import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.dirname(PROJ_ROOT)
IBD_INT_ROOT = os.path.join(ANALYSIS_ROOT, "IBD_integration")
IBD_EXP_ROOT = os.path.join(ANALYSIS_ROOT, "IBD_Exploration")

# ── Source inputs (read-only) ────────────────────────────────────────────────
INT_OUTPUTS = os.path.join(IBD_INT_ROOT, "outputs")
ADATA_C1HEAVY = os.path.join(INT_OUTPUTS, "adata_c1heavy.h5ad")
ADATA_FULLSCALE = os.path.join(INT_OUTPUTS, "adata_fullscale_preprocessed.h5ad")
DULAI_META_CSV = os.path.join(INT_OUTPUTS, "dulai_percell_meta.csv")
SAHA_META_CSV = os.path.join(INT_OUTPUTS, "saha_c2_percell_meta.csv")
LABEL_MAPPING_CSV = os.path.join(INT_OUTPUTS, "label_mapping_final.csv")

# Reuse upstream concordance / LIANA artefacts where possible
EXP_PHASE4 = os.path.join(IBD_EXP_ROOT, "outputs", "phase4")
EXP_CONCORDANCE_DIR = os.path.join(EXP_PHASE4, "concordance")
EXP_RESPONDER_DIR = os.path.join(EXP_PHASE4, "responder")
RESPONDER_LABELS_CSV = os.path.join(IBD_EXP_ROOT, "data", "responder_labels.csv")

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs")
ADATA_META = os.path.join(OUTPUT_DIR, "adata_c1heavy_meta.h5ad")
CONCORDANCE_DIR = os.path.join(OUTPUT_DIR, "concordance")
BIOLOGY_DIR = os.path.join(OUTPUT_DIR, "biology")
FIG_MAIN_DIR = os.path.join(OUTPUT_DIR, "figures", "main")
FIG_EXT_DIR = os.path.join(OUTPUT_DIR, "figures", "extended")
DATA_DIR = os.path.join(PROJ_ROOT, "data")

# ── Cohort / platform labels (locked orderings) ──────────────────────────────
COHORT_PLATFORMS = ["C1-Rectum-CosMx", "C1-Rectum-Xenium", "C2-Ileum-CosMx"]
DATASET_TO_CP = {
    "cosmx_c1": "C1-Rectum-CosMx",
    "xenium_c1": "C1-Rectum-Xenium",
    "cosmx_c2": "C2-Ileum-CosMx",
}

# Dulai 4 patients (paired across CosMx + Xenium); block IDs from dulai_percell_meta.csv
DULAI_PATIENTS = {
    1: {"block": "I0400", "disease": "Healthy", "disease_state": "Healthy"},
    2: {"block": "I0355", "disease": "UC", "disease_state": "Non-inflamed"},
    3: {"block": "I0380", "disease": "UC", "disease_state": "Inflamed"},
    4: {"block": "I0387", "disease": "UC", "disease_state": "Inflamed"},
}

# C2 SAHA sections (per outputs/saha_c2_percell_meta.csv)
SAHA_SECTIONS = [
    "IBD_A1_1", "IBD_A1_2", "IBD_A1_3", "IBD_A1_4", "IBD_A1_5", "IBD_A1_6",
    "IBD_A2_1", "IBD_A2_2",
]

CT_MAJOR_ORDER = [
    "Epithelial", "Mesenchymal", "Endothelial", "Neuron",
    "Myeloid", "T/NK", "B", "Plasma", "Unknown",
]

# ── Palettes (Okabe-Ito colorblind-safe; mirrors IBD_Exploration) ───────────
CT_PALETTE = {
    "B":             "#1f77b4",
    "Endothelial":   "#ff7f0e",
    "Epithelial":    "#2ca02c",
    "Mesenchymal":   "#d62728",
    "Myeloid":       "#9467bd",
    "Neuron":        "#bcbd22",
    "Plasma":        "#17becf",
    "T/NK":          "#e377c2",
    "Unknown":       "#aaaaaa",
}
COHORT_PLATFORM_PALETTE = {
    "C1-Rectum-CosMx":  "#0072B2",
    "C1-Rectum-Xenium": "#CC79A7",
    "C2-Ileum-CosMx":   "#E69F00",
}
PLATFORM_PALETTE = {"cosmx": "#0072B2", "xenium": "#CC79A7"}
DISEASE_STATE_PALETTE = {
    "Healthy":      "#009E73",
    "Non-inflamed": "#56B4E9",
    "Inflamed":     "#D55E00",
    "IBD":          "#D55E00",
}
RESPONDER_PALETTE = {
    "Responder":     "#009E73",
    "Non-responder": "#D55E00",
    "Pending":       "#999999",
    "IBD-no-label":  "#cccccc",
    "Healthy":       "#bbbbbb",
}

# ── IBD biology gene sets ────────────────────────────────────────────────────
# CXCL8 and MUC2 are EXCLUDED per IBD_Exploration invariant (cross-platform
# normalization issues — see IBD_Exploration/scripts/04_concordance.py).
EXCLUDED_GENES = {"CXCL8", "MUC2"}

IBD_MARKER_PANEL = {
    "Inflammation_TNF":   ["TNF", "TNFRSF1A", "TNFRSF1B", "NFKBIA", "RELB"],
    "Inflammation_IL17":  ["IL17A", "IL17F", "IL17RA", "IL23A", "IL22"],
    "Inflammation_IFNg":  ["IFNG", "STAT1", "GBP1", "CXCL9", "CXCL10", "IDO1"],
    "Treg":               ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "Myeloid_activation": ["S100A8", "S100A9", "LYZ", "CD68", "CD163"],
    "Epithelial_barrier": ["EPCAM", "KRT20", "REG3A", "DEFA5", "DEFB1"],
    "Stromal_inflam":     ["IL11", "IL13RA2", "TNFRSF11B", "CHI3L1", "COL1A1"],
}

CT_MARKER_DICT = {
    "Epithelial":  ["EPCAM", "KRT20"],
    "T/NK":        ["CD3E", "CD8A", "CD4", "FOXP3", "NCAM1"],
    "Myeloid":     ["CD68", "LYZ", "S100A8"],
    "B":           ["CD79A", "MS4A1"],
    "Plasma":      ["MZB1", "IGHG1", "SDC1"],
    "Mesenchymal": ["COL1A1", "VIM", "ACTA2"],
    "Endothelial": ["PECAM1", "VWF"],
    "Inflammation": ["TNF", "IL1B", "FOXP3"],
}

# ── Embedding key ────────────────────────────────────────────────────────────
EMBEDDING_KEY = "X_scANVI_c1heavy"

# ── Verification thresholds (Nyquist-style) ──────────────────────────────────
PAIRED_PEARSON_MIN = 0.7      # paired Xenium↔CosMx pseudobulk per (patient, ct)
PAIRED_FRAC_PASS_PER_PATIENT = 6 / 9   # ≥6/9 cell types must pass
PAIRED_PATIENTS_PASS = 3       # ≥3/4 patients must pass
KBET_MIN = 0.5
ILISI_MIN = 0.6

# ── Matplotlib rcParams (Nature style; mirrors IBD_Exploration) ──────────────
NATURE_RCPARAMS = {
    "pdf.fonttype":   42,           # TrueType (editable text in PDF)
    "ps.fonttype":    42,
    "svg.fonttype":   "none",
    "font.family":    "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":      7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi":     300,
    "savefig.bbox":   "tight",
}


def ensure_output_dirs():
    """Create all output directories if missing. Idempotent."""
    for d in [OUTPUT_DIR, CONCORDANCE_DIR, BIOLOGY_DIR,
              FIG_MAIN_DIR, FIG_EXT_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_output_dirs()
    print("config.py OK — outputs at:", OUTPUT_DIR)
