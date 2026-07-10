"""Phase 4 Plan 06 — Restore raw integer counts to spatial h5ads.

Root-cause fix for VERIFICATION.md Gap 1. Reads the upstream integration h5ad
(adata_fullscale_preprocessed.h5ad) which has layers['counts'] already, and
copies those counts into the three per-cohort spatial h5ads while preserving:
  - adata.X (log1p-normalized, for visualization)
  - adata.obs columns
  - adata.obsm['spatial']
  - adata.obsp (if present — squidpy graphs)
  - adata.var_names

Invariants asserted per cohort:
  - layers['counts'].dtype.kind in ('i', 'u', 'f') AND values int-like
  - X.max() < 20 (log1p-bounded)
  - X.shape == layers['counts'].shape
  - 'CXCL8' not in var_names AND 'MUC2' not in var_names

Idempotent: re-running after success is a no-op that only re-verifies invariants.
"""
import os
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

UPSTREAM_H5AD = os.path.join(
    config.IBD_INT_ROOT, "outputs", "adata_fullscale_preprocessed.h5ad"
)

COHORTS = [
    ("cosmx_c1", config.COSMX_C1_H5AD, config.EXPECTED_COSMX_C1_CELLS),
    ("xenium_c1", config.XENIUM_C1_H5AD, config.EXPECTED_XENIUM_C1_CELLS),
    ("cosmx_c2", config.COSMX_C2_H5AD, config.EXPECTED_COSMX_C2_CELLS),
]

FORBIDDEN_GENES = ("CXCL8", "MUC2")


def _to_dense(mat):
    """Materialize a sparse or array-like matrix to a dense ndarray."""
    if sp.issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _assert_counts_layer_valid(adata, cohort, sample_cap=10_000):
    """Assert layers['counts'] is present, integer-valued, and .X remains log1p-bounded.

    Uses a sampled subset of rows so the check stays tractable on 437K-cell cohorts.
    """
    assert "counts" in adata.layers, (
        f"{cohort}: layers['counts'] missing after restore"
    )

    n = min(sample_cap, adata.n_obs)
    counts_sample = _to_dense(adata.layers["counts"][:n])
    assert np.all(counts_sample % 1 == 0), (
        f"{cohort}: counts contain non-integer values in first {n} cells"
    )

    x_sample = _to_dense(adata.X[:n])
    x_max = float(x_sample.max())
    assert x_max < 20, (
        f"{cohort}: .X max={x_max} not log1p-bounded (first {n} cells)"
    )

    assert adata.X.shape == adata.layers["counts"].shape, (
        f"{cohort}: .X {adata.X.shape} != counts {adata.layers['counts'].shape}"
    )

    for g in FORBIDDEN_GENES:
        assert g not in adata.var_names, (
            f"{cohort}: forbidden gene {g} in var_names"
        )


def restore_counts_for_cohort(cohort, target_path, expected_n_obs, upstream):
    """Copy layers['counts'] from upstream into target h5ad, preserving .X / obs / obsm / obsp.

    Idempotent: if target already has layers['counts'], only verifies invariants.
    """
    print(f"\n[{cohort}] Loading target {target_path}")
    target = ad.read_h5ad(target_path)
    assert target.n_obs == expected_n_obs, (
        f"{cohort}: target has {target.n_obs} cells, expected {expected_n_obs}"
    )
    print(
        f"[{cohort}] target shape: {target.shape}; obs cols: {len(target.obs.columns)}"
    )

    if "counts" in target.layers:
        print(
            f"[{cohort}] SKIP — layers['counts'] already present; "
            f"verifying invariants only"
        )
        _assert_counts_layer_valid(target, cohort)
        return

    # Subset upstream to this cohort by obs['dataset']
    up_sub = upstream[upstream.obs["dataset"] == cohort].copy()
    print(
        f"[{cohort}] upstream subset: {up_sub.n_obs} cells "
        f"(expected {expected_n_obs})"
    )
    assert up_sub.n_obs == expected_n_obs, (
        f"{cohort}: upstream has {up_sub.n_obs} cells in dataset={cohort}, "
        f"expected {expected_n_obs} — cohort cell counts diverged from integration"
    )

    # Reindex upstream by target obs_names (exact set equality first)
    missing = set(target.obs_names) - set(up_sub.obs_names)
    assert not missing, (
        f"{cohort}: {len(missing)} target cells missing from upstream"
    )
    up_sub = up_sub[target.obs_names].copy()

    # var_names must match in order (same 2,551-gene panel)
    assert list(target.var_names) == list(up_sub.var_names), (
        f"{cohort}: var_names mismatch — upstream {up_sub.n_vars} vs "
        f"target {target.n_vars}"
    )

    counts = up_sub.layers["counts"]
    # Preserve sparsity — coerce to csr_matrix if dense and mostly zero
    if not sp.issparse(counts):
        # Count zeros on a sample to decide
        sample = counts if counts.size < 1_000_000 else counts[:5000]
        if (np.asarray(sample) == 0).mean() > 0.5:
            counts = sp.csr_matrix(counts)

    target.layers["counts"] = counts
    print(
        f"[{cohort}] counts layer attached: dtype={target.layers['counts'].dtype}, "
        f"sparse={sp.issparse(target.layers['counts'])}"
    )

    _assert_counts_layer_valid(target, cohort)

    # Atomic write: temp path + rename
    tmp = target_path + ".tmp_restore.h5ad"
    target.write_h5ad(tmp)
    os.replace(tmp, target_path)
    print(f"[{cohort}] wrote {target_path}")


def main():
    assert os.path.exists(UPSTREAM_H5AD), (
        f"Upstream h5ad missing: {UPSTREAM_H5AD}. "
        f"Expected at ../IBD_integration/outputs/adata_fullscale_preprocessed.h5ad"
    )
    print(f"Loading upstream {UPSTREAM_H5AD} ...")
    upstream = ad.read_h5ad(UPSTREAM_H5AD)
    print(
        f"Upstream: {upstream.shape}, layers={list(upstream.layers.keys())}, "
        f"datasets={sorted(upstream.obs['dataset'].unique())}"
    )
    assert "counts" in upstream.layers, "Upstream missing layers['counts']"

    for cohort, path, expected_n in COHORTS:
        restore_counts_for_cohort(cohort, path, expected_n, upstream)

    print("\nAll three spatial h5ads have layers['counts'] restored.")


if __name__ == "__main__":
    main()
