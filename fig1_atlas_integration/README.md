# Fig 1 — cross-organ reference (CosMx RNA + protein)

Fig 1d and Extended Data Fig 1 (joint UMAP), per-organ UMAPs (Fig 1e, ED Fig 2).

## Files

- `plot_fig1d_from_coordinates.py` + `fig1d_palette.json` — redraws Fig 1d (cell type) and ED Fig 1a
  (organ) from the published UMAP coordinates in the Source Data
  (`Fig1_d_umap_coordinates_RNA.parquet`; protein: `Fig1_protein_umap_coordinates_PRT.parquet`).
  This is the exact figure; start here.
- `fig1d_joint_umap.py` — how those coordinates were made: runnable extract of the joint RNA UMAP
  (Harmony on `batch`, neighbours on `X_harmony`), parameters unchanged from
  `as_run/merge_all_organs_RNA.py`. A re-run gives an equivalent but not identical layout.
- `as_run/` — the scripts as they were run, unchanged. They were executed interactively, section by
  section, so they do not run top to bottom (later sections are exploratory and some reuse variable
  names from an interactive session).

## Order in which `as_run/` was used

1. `merge_all_objects_v4.py` (RNA) and `merge_all_objects_PRT.py` (protein) — per-sample QC and
   squidpy processing, merge per organ, Harmony on `batch`, per-organ UMAP.
   `merge_all_objects_v3.py` is the previous version of the RNA step, kept because some per-organ
   objects may predate v4.
2. `postmerge_processing.R` — InSituType cell typing with organ reference profiles.
   It sources `helpers/AtoMx_Sobj_Processing_helpers.R` (adjust the relative `source()` path).
   Reference profiles are NanoString's public Cell Profile Library, e.g.
   [`Human/Adult/Gut_HCA.RData`](https://github.com/Nanostring-Biostats/CellProfileLibrary/tree/master/Human/Adult);
   replace the absolute `load()` path with your local copy.
3. `merge_all_organs_RNA.py` / `merge_all_organs_PRT.py` — merge all organs, Harmony, joint UMAP.

## Environment

Python: scanpy, anndata, harmonypy, squidpy, scipy. R: InSituType, Seurat.
