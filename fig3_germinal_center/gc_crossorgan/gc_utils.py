"""
gc_utils.py — Shared utility module for SAHA Cross-Organ GC Analysis.

Import pattern for notebooks (from notebooks/ directory):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.gc_utils import load_adata, get_gene_or_proxy, nature_rcparams

DATA CONTRACT (established in Phase 1):
- adata.X = raw integer counts (use for DE)
- layers['normalized'] = normalized expression (use for visualization)
- obsm['spatial'] = global pixel coordinates (use for spatial clustering)
- PIXEL_SIZE_UM = 0.120 um/pixel (confirmed from data)
- AICDA, BCL6, PRDM1 absent from 889-gene panel — check gene_availability.json
"""

import os
import sys
import json
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default path to the preprocessed h5ad file (resolved relative to this file).
DATA_PATH: str = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "Data", "preprocessed",
        "adata_combined_appendix_lymph_node.h5ad",
    )
)

#: Confirmed CosMx pixel size in micrometers per pixel (median across 21,823 cells).
PIXEL_SIZE_UM: float = 0.120

#: 300 um GC radius in pixels at PIXEL_SIZE_UM = 0.120 um/px.
GC_RADIUS_300UM_PX: float = 2500.0

#: 13-category cell type palette aligned to the SAHA project color scheme.
#: B_cells family anchored at #F096C1 (pink), T_cells at #8C554B (brown),
#: Myeloid at #C33334 (red), DC at #6BC836 (green), Plasma at #7F7F7F (gray),
#: Other_Lymphoid at #53DBE1 (teal).
#: Keys match adata.obs['cell_types_insitutype'].unique() exactly.
GC_CELL_TYPE_PALETTE: Dict[str, str] = {
    # B cell lineage (pinks / roses)
    'GCBC':    '#F096C1',  # germinal center B cells — anchor pink
    'NBC/MBC': '#D95F8C',  # naive/memory B — deeper rose
    'PC':      '#7F7F7F',  # plasma cells — gray (SAHA anchor)
    # T cell lineage (browns / oranges)
    'TC_CD4':  '#8C554B',  # CD4 T — anchor brown
    'TC_CD8':  '#C47B5A',  # CD8 T — warm sienna
    # Myeloid (reds)
    'macro':   '#C33334',  # macrophages — anchor red
    'mono':    '#E85D5D',  # monocytes — lighter red
    'gran':    '#A02020',  # granulocytes — dark red
    'mast':    '#F4A460',  # mast cells — sandy orange
    # Dendritic cells (greens)
    'FDC':     '#4CAF50',  # follicular DC — mid green
    'DC':      '#6BC836',  # conventional DC — anchor green
    'PDC':     '#2E7D32',  # plasmacytoid DC — dark green
    # Innate lymphoid (teals)
    'ILC':     '#53DBE1',  # innate lymphoid — anchor teal
}


# ---------------------------------------------------------------------------
# Function 1: load_adata
# ---------------------------------------------------------------------------

def load_adata(data_path: Optional[str] = None) -> ad.AnnData:
    """Load and schema-validate the SAHA combined appendix/lymph node AnnData object.

    Parameters
    ----------
    data_path:
        Path to the .h5ad file. Defaults to DATA_PATH (resolved relative to
        this module's location). Pass an explicit path for testing or alternate datasets.

    Returns
    -------
    ad.AnnData
        Validated AnnData with n_obs==21823, n_vars==889.

    Raises
    ------
    AssertionError
        If any schema assertion fails (cell count, gene count, required keys).
    FileNotFoundError
        If the h5ad file does not exist at the resolved path.
    """
    if data_path is None:
        data_path = DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"h5ad file not found: {data_path}\n"
            "Check that DATA_PATH points to the preprocessed data directory."
        )

    adata = ad.read_h5ad(data_path)

    # Cell and gene count assertions
    assert adata.n_obs == 21823, (
        f"Expected 21823 cells, got {adata.n_obs}. "
        "Schema mismatch — wrong h5ad file?"
    )
    assert adata.n_vars == 889, (
        f"Expected 889 genes, got {adata.n_vars}. "
        "Schema mismatch — wrong h5ad file?"
    )

    # Required obsm key (global pixel coordinates)
    assert 'spatial' in adata.obsm, (
        "Missing obsm['spatial'] — expected global pixel coordinates (CenterX/Y_global_px)."
    )

    # Required layers
    assert 'raw' in adata.layers, (
        "Missing layers['raw'] — expected raw integer count matrix."
    )
    assert 'normalized' in adata.layers, (
        "Missing layers['normalized'] — expected normalized expression matrix."
    )

    # Required obs column
    assert 'cell_types_insitutype' in adata.obs.columns, (
        "Missing obs column 'cell_types_insitutype' — InSituType cell type labels required."
    )

    # Coordinate system guard: global coords span > 10,000 px; local FOV coords do not
    x_span = adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()
    assert x_span > 10000, (
        f"obsm['spatial'] x-span is only {x_span:.0f} px — expected > 10000 px for global coords. "
        "May be using local FOV coordinates instead of global pixel coordinates."
    )

    return adata


# ---------------------------------------------------------------------------
# Function 2: load_gene_availability
# ---------------------------------------------------------------------------

def load_gene_availability(outputs_dir: Optional[str] = None) -> dict:
    """Load the gene availability report from outputs/gene_availability.json.

    Parameters
    ----------
    outputs_dir:
        Directory containing gene_availability.json. Defaults to the outputs/
        directory adjacent to the project root (resolved relative to this file).

    Returns
    -------
    dict
        Gene entries with keys: present (bool), role (str), proxy (str|None), note (str|None).

    Raises
    ------
    FileNotFoundError
        If gene_availability.json is not found in outputs_dir. Run Plan 02
        (01_validate_environment.ipynb) to generate it.
    """
    if outputs_dir is None:
        outputs_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "outputs")
        )

    json_path = os.path.join(outputs_dir, "gene_availability.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"gene_availability.json not found at: {json_path}\n"
            "Run 01_validate_environment.ipynb (Plan 02) to generate this file."
        )

    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Function 3: get_gene_or_proxy
# ---------------------------------------------------------------------------

def get_gene_or_proxy(gene: str, gene_availability: dict) -> Optional[str]:
    """Return the gene to use for analysis: the gene itself, a proxy, or None.

    Parameters
    ----------
    gene:
        Gene symbol to look up (e.g., 'BCL2', 'AICDA').
    gene_availability:
        Dict loaded from gene_availability.json via load_gene_availability().

    Returns
    -------
    str or None
        - The gene itself if present in the panel.
        - A proxy gene name (with warning) if gene is absent but proxy is defined.
        - None (with warning) if gene is absent and no proxy is available.

    Raises
    ------
    KeyError
        If the gene is not in gene_availability at all (programming error — add it to
        gene_availability.json or check the gene symbol).
    """
    if gene not in gene_availability:
        raise KeyError(
            f"Gene '{gene}' not found in gene_availability dict. "
            "Check the gene symbol or update gene_availability.json."
        )

    entry = gene_availability[gene]

    if entry['present']:
        return gene

    proxy = entry.get('proxy')
    if proxy:
        print(
            f"WARNING: {gene} is absent from the 889-gene panel — using proxy {proxy}",
            file=sys.stderr,
        )
        return proxy

    print(
        f"WARNING: {gene} is absent from the 889-gene panel and has no proxy — returning None",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Function 4: load_metadata
# ---------------------------------------------------------------------------

def load_metadata(outputs_dir: Optional[str] = None) -> dict:
    """Load analysis metadata from outputs/metadata.json.

    Parameters
    ----------
    outputs_dir:
        Directory containing metadata.json. Defaults to the outputs/ directory
        adjacent to the project root (resolved relative to this file).

    Returns
    -------
    dict
        Metadata with keys such as pixel_size_um, gc_radius_300um_px, layer_state, etc.

    Raises
    ------
    FileNotFoundError
        If metadata.json is not found. Run 01_validate_environment.ipynb to generate it.
    """
    if outputs_dir is None:
        outputs_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "outputs")
        )

    json_path = os.path.join(outputs_dir, "metadata.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"metadata.json not found at: {json_path}\n"
            "Run 01_validate_environment.ipynb (Plan 02) to generate this file."
        )

    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Function 5: nature_rcparams
# ---------------------------------------------------------------------------

def nature_rcparams() -> dict:
    """Return a matplotlib rcParams dict for Nature submission figure style.

    Returns a NEW dict each call — caller applies via::

        plt.rcParams.update(nature_rcparams())

    Does NOT mutate the global plt.rcParams.

    Returns
    -------
    dict
        Matplotlib rcParams for Nature style: 7pt fonts, 0.5pt line widths,
        no top/right spines, 300 DPI, TrueType fonts (pdf.fonttype=42).
    """
    return {
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    }


# ---------------------------------------------------------------------------
# Function 6: apply_nature_style
# ---------------------------------------------------------------------------

def apply_nature_style(ax=None) -> None:
    """Apply Nature-style spine cleanup to a matplotlib Axes.

    Removes the top and right spines, and sets the remaining spine linewidths
    to 0.5 pt (matching Nature's axis style guidelines).

    Does NOT change font sizes — call plt.rcParams.update(nature_rcparams())
    before creating the figure to set font sizes globally.

    Parameters
    ----------
    ax:
        Matplotlib Axes object. If None, uses plt.gca().
    """
    if ax is None:
        ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Function 7: estimate_dbscan_eps
# ---------------------------------------------------------------------------

def estimate_dbscan_eps(
    coords: np.ndarray,
    k: int = 10,
) -> Tuple[float, np.ndarray]:
    """Estimate DBSCAN eps via k-NN elbow method.

    Computes the k-th nearest-neighbor distance for each point, sorts
    descending, and finds the elbow (maximum second derivative) as eps.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Spatial pixel coordinates.
    k : int
        Number of nearest neighbors. Should equal min_samples - 1.

    Returns
    -------
    eps_estimate : float
        Distance at the elbow of the sorted k-NN curve (pixels).
    knn_distances : np.ndarray
        Sorted k-th nearest-neighbor distances, descending.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    knn_distances = np.sort(distances[:, k])[::-1]
    d2 = np.diff(np.diff(knn_distances))
    elbow_idx = int(np.argmax(d2)) + 1
    eps_estimate = float(knn_distances[elbow_idx])
    return eps_estimate, knn_distances


# ---------------------------------------------------------------------------
# Function 8: detect_gcs
# ---------------------------------------------------------------------------

def detect_gcs(
    adata: 'ad.AnnData',
    organ_sample_id: str,
    min_samples: int = 10,
) -> Tuple[np.ndarray, 'pd.Series', float]:
    """Run DBSCAN on GCBC+FDC cells for one organ to detect GC clusters.

    Uses data-driven eps calibration via estimate_dbscan_eps. Each organ
    is processed independently (coordinate spaces are disjoint).

    Parameters
    ----------
    adata : AnnData
        Full combined dataset (21823 cells).
    organ_sample_id : str
        Value in adata.obs['SAHA_Sample'] ('LN_A1_3' or 'APE_A9_3').
    min_samples : int
        DBSCAN min_samples. Default 10.

    Returns
    -------
    gc_labels : np.ndarray, shape (n_seed_cells,)
        DBSCAN cluster labels. -1 = noise.
    seed_mask : pd.Series (bool)
        Boolean Series indexed like adata.obs, True for GCBC+FDC in this organ.
    eps_used : float
        The eps value applied (pixels).
    """
    seed_mask = (
        (adata.obs['SAHA_Sample'] == organ_sample_id) &
        (adata.obs['cell_types_insitutype'].isin(['GCBC', 'FDC']))
    )
    coords = adata.obsm['spatial'][seed_mask.values]

    eps_used, _ = estimate_dbscan_eps(coords, k=min_samples - 1)

    db = DBSCAN(eps=eps_used, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    gc_labels = db.fit_predict(coords)

    n_clusters = len(set(gc_labels)) - (1 if -1 in gc_labels else 0)
    print(f"  [{organ_sample_id}] eps={eps_used:.1f} px "
          f"({eps_used * PIXEL_SIZE_UM:.1f} um), "
          f"min_samples={min_samples} -> {n_clusters} GC clusters detected")

    return gc_labels, seed_mask, eps_used


# ---------------------------------------------------------------------------
# Function 9: compute_gc_boundary
# ---------------------------------------------------------------------------

def compute_gc_boundary(
    coords: np.ndarray,
    method: str = 'convex_hull',
) -> 'MultiPoint':
    """Compute GC boundary polygon from cell coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Pixel coordinates of cells within one DBSCAN cluster.
    method : str
        'convex_hull' (default) or 'alpha_shape' (not yet implemented).

    Returns
    -------
    shapely.geometry.Polygon
        Boundary polygon in pixel coordinates.
    """
    if len(coords) < 3:
        raise ValueError(f"Need at least 3 points for a polygon; got {len(coords)}")
    if method == 'convex_hull':
        return MultiPoint(coords).convex_hull
    raise NotImplementedError(
        "Alpha shape not implemented. Add scipy Delaunay approach if convex hull "
        "produces incorrect boundaries for concave GCs."
    )


# ---------------------------------------------------------------------------
# Function 10: assign_cells_to_gcs
# ---------------------------------------------------------------------------

def assign_cells_to_gcs(
    adata: 'ad.AnnData',
    gc_polygons: Dict[str, 'object'],
) -> 'pd.DataFrame':
    """Assign ALL cells to GC boundaries via point-in-polygon.

    Cells outside all GC boundaries receive gc_id=NaN, gc_member=False,
    distance_to_centroid=NaN. Cells inside get the gc_id and distance
    to centroid.

    Parameters
    ----------
    adata : AnnData
        Full combined dataset.
    gc_polygons : dict
        {gc_id (str): shapely.Polygon} for all detected GCs.

    Returns
    -------
    pd.DataFrame
        Index matches adata.obs_names. Columns: gc_id, gc_member, distance_to_centroid.
    """
    all_coords = adata.obsm['spatial']
    result = pd.DataFrame(
        {
            'gc_id': pd.array([pd.NA] * adata.n_obs, dtype='object'),
            'gc_member': False,
            'distance_to_centroid': np.nan,
        },
        index=adata.obs_names,
    )

    for gc_id, polygon in gc_polygons.items():
        xs = all_coords[:, 0]
        ys = all_coords[:, 1]

        # shapely 2.x vectorized path
        try:
            from shapely.vectorized import contains as _contains
            inside = _contains(polygon, xs, ys)
        except ImportError:
            inside = np.array([
                polygon.contains(Point(x, y)) for x, y in zip(xs, ys)
            ])

        unassigned = result['gc_member'].values == False
        newly_inside = inside & unassigned

        centroid_xy = np.array(polygon.centroid.coords[0])
        distances = np.linalg.norm(all_coords - centroid_xy, axis=1)

        result.loc[newly_inside, 'gc_id'] = gc_id
        result.loc[newly_inside, 'gc_member'] = True
        result.loc[newly_inside, 'distance_to_centroid'] = distances[newly_inside]

    return result


# ---------------------------------------------------------------------------
# Function 11: build_gc_registry
# ---------------------------------------------------------------------------

def build_gc_registry(
    gc_polygons: Dict[str, 'object'],
    gc_cluster_data: Dict[str, dict],
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> 'pd.DataFrame':
    """Build gc_registry DataFrame from detected GC polygons.

    Parameters
    ----------
    gc_polygons : dict
        {gc_id: shapely.Polygon}
    gc_cluster_data : dict
        {gc_id: {'organ': str, 'coords': np.ndarray, 'n_gcbc': int,
                 'n_fdc': int, 'eps_used': float}}
    pixel_size_um : float
        Microns per pixel.

    Returns
    -------
    pd.DataFrame
        Columns: gc_id, organ, centroid_x, centroid_y, n_cells, n_gcbc,
        n_fdc, diameter_um, flagged, eps_used, convex_hull_area_um2.
    """
    from scipy.spatial.distance import pdist

    records = []
    for gc_id, polygon in gc_polygons.items():
        data = gc_cluster_data[gc_id]
        centroid = polygon.centroid
        coords = data['coords']
        n_cells = len(coords)

        if n_cells >= 2:
            dists = pdist(coords)
            diameter_px = float(dists.max()) if len(dists) > 0 else 0.0
        else:
            diameter_px = 0.0

        area_um2 = polygon.area * (pixel_size_um ** 2)

        records.append({
            'gc_id':                gc_id,
            'organ':                data['organ'],
            'centroid_x':           float(centroid.x),
            'centroid_y':           float(centroid.y),
            'n_cells':              n_cells,
            'n_gcbc':               data['n_gcbc'],
            'n_fdc':                data['n_fdc'],
            'diameter_um':          round(diameter_px * pixel_size_um, 2),
            'flagged':              bool(n_cells < 50 or n_cells > 2000),
            'eps_used':             round(data['eps_used'], 2),
            'convex_hull_area_um2': round(area_um2, 2),
        })

    df = pd.DataFrame(records).sort_values(['organ', 'gc_id']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Function 12: compute_fractional_radius
# ---------------------------------------------------------------------------

def compute_fractional_radius(obs_assignments: 'pd.DataFrame') -> 'pd.Series':
    """Compute fractional radius (0-1) for each cell within its GC.

    For each GC, normalizes distance_to_centroid by the max distance in that GC.
    Non-GC-member cells get NaN.

    Parameters
    ----------
    obs_assignments : pd.DataFrame
        DataFrame with columns gc_id, gc_member, distance_to_centroid.
        Index should be str (matching adata.obs_names).

    Returns
    -------
    pd.Series
        Fractional radius in [0, 1] for GC members, NaN for non-members.
    """
    gc_mem = obs_assignments[obs_assignments['gc_member'] == True].copy()
    gc_max = gc_mem.groupby('gc_id')['distance_to_centroid'].max()
    gc_mem = gc_mem.join(gc_max.rename('gc_max_dist'), on='gc_id')
    gc_mem['gc_max_dist'] = gc_mem['gc_max_dist'].replace(0, np.nan)
    gc_mem['frac_radius'] = gc_mem['distance_to_centroid'] / gc_mem['gc_max_dist']
    gc_mem['frac_radius'] = gc_mem['frac_radius'].fillna(0.0)
    result = pd.Series(np.nan, index=obs_assignments.index, dtype=float)
    result.update(gc_mem['frac_radius'])
    return result


# ---------------------------------------------------------------------------
# Function 13: radial_bin_profile
# ---------------------------------------------------------------------------

def radial_bin_profile(
    df_gc: 'pd.DataFrame',
    frac_radius_col: str = 'frac_radius',
    cell_type_col: str = 'cell_types_insitutype',
    gc_id_col: str = 'gc_id',
    n_bins: int = 5,
) -> 'pd.DataFrame':
    """Compute cell type proportions in radial bins for each GC.

    Parameters
    ----------
    df_gc : pd.DataFrame
        Subset of obs for GC members, must have frac_radius column.
    frac_radius_col : str
        Column with fractional radius values in [0, 1].
    cell_type_col : str
        Column with cell type labels.
    gc_id_col : str
        Column with GC identifiers.
    n_bins : int
        Number of radial bins. Default 5.

    Returns
    -------
    pd.DataFrame
        Columns: gc_id, bin, cell_type, count, proportion.
        Proportions per (gc_id, bin) sum to 1.0.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)]
    df = df_gc.copy()
    df['bin'] = pd.cut(df[frac_radius_col], bins=bins, labels=labels, include_lowest=True)
    all_cell_types = sorted(df[cell_type_col].unique())
    records = []
    for (gc_id, bin_label), grp in df.groupby([gc_id_col, 'bin'], observed=True):
        total = len(grp)
        ct_counts = grp[cell_type_col].value_counts()
        for ct in all_cell_types:
            cnt = int(ct_counts.get(ct, 0))
            records.append({
                'gc_id': gc_id,
                'bin': str(bin_label),
                'cell_type': ct,
                'count': cnt,
                'proportion': cnt / total if total > 0 else 0.0,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Function 14: classify_dzlz_by_expression
# ---------------------------------------------------------------------------

def classify_dzlz_by_expression(
    adata: 'ad.AnnData',
    cell_mask,
    cxcr4_threshold: float = None,
    cd83_threshold: float = None,
    cxcr5_threshold: float = None,
) -> 'pd.Series':
    """Classify cells as DZ, LZ, or intermediate by marker expression.

    Uses CXCR4 (DZ marker) and CD83/CXCR5 (LZ markers) from the normalized
    expression layer. DZ assignment takes priority over LZ.

    Parameters
    ----------
    adata : AnnData
        Full or subset AnnData with layers['normalized'].
    cell_mask : array-like (bool)
        Boolean mask selecting cells to classify.
    cxcr4_threshold, cd83_threshold, cxcr5_threshold : float, optional
        Expression thresholds. If None, uses median of positive-expressing cells.

    Returns
    -------
    pd.Series
        Values: 'DZ', 'LZ', or 'intermediate'. Index = obs_names of masked cells.
    """
    import scipy.sparse as sp

    def _get_gene(adata_sub, gene, layer='normalized'):
        if gene not in adata_sub.var_names:
            return np.zeros(adata_sub.n_obs)
        idx = list(adata_sub.var_names).index(gene)
        mat = adata_sub.layers[layer]
        col = mat[:, idx]
        if sp.issparse(col):
            col = np.asarray(col.todense()).ravel()
        return np.asarray(col).ravel()

    def _auto_threshold(vals):
        pos = vals[vals > 0]
        return float(np.median(pos)) if len(pos) > 0 else 0.1

    subset = adata[cell_mask]
    cxcr4 = _get_gene(subset, 'CXCR4')
    cd83 = _get_gene(subset, 'CD83')
    cxcr5 = _get_gene(subset, 'CXCR5')

    t_cxcr4 = cxcr4_threshold if cxcr4_threshold is not None else _auto_threshold(cxcr4)
    t_cd83 = cd83_threshold if cd83_threshold is not None else _auto_threshold(cd83)
    t_cxcr5 = cxcr5_threshold if cxcr5_threshold is not None else _auto_threshold(cxcr5)

    labels = np.full(len(cxcr4), 'intermediate', dtype=object)
    lz_mask = (cd83 > t_cd83) | (cxcr5 > t_cxcr5)
    dz_mask = (cxcr4 > t_cxcr4)
    labels[lz_mask] = 'LZ'
    labels[dz_mask] = 'DZ'  # DZ takes priority

    return pd.Series(labels, index=subset.obs_names)


# ---------------------------------------------------------------------------
# Smoke test / __main__ block
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("gc_utils.py — smoke test")
    print(f"  DATA_PATH: {DATA_PATH}")
    print(f"  PIXEL_SIZE_UM: {PIXEL_SIZE_UM} um/px")
    print(f"  GC_RADIUS_300UM_PX: {GC_RADIUS_300UM_PX} px")
    print()
    print("Loading AnnData (this may take a few seconds)...")
    adata = load_adata()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  layer keys: {list(adata.layers.keys())}")
    print(f"  obs columns (first 5): {list(adata.obs.columns[:5])}")
    print()
    print("nature_rcparams() check...")
    rc = nature_rcparams()
    assert rc['pdf.fonttype'] == 42
    assert rc['font.size'] == 7
    assert rc['axes.linewidth'] == 0.5
    print(f"  pdf.fonttype={rc['pdf.fonttype']}, font.size={rc['font.size']}, axes.linewidth={rc['axes.linewidth']}")
    print()
    # Verify new Phase 2 exports
    from src.gc_utils import (
        estimate_dbscan_eps, detect_gcs, compute_gc_boundary,
        assign_cells_to_gcs, build_gc_registry, GC_CELL_TYPE_PALETTE,
    )
    assert len(GC_CELL_TYPE_PALETTE) == 13, f"Expected 13 palette entries, got {len(GC_CELL_TYPE_PALETTE)}"
    print(f"  GC_CELL_TYPE_PALETTE: {len(GC_CELL_TYPE_PALETTE)} cell types")
    print("  All new function exports verified.")
    print()
    print("Smoke test PASSED")
