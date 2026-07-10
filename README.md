# saha-2026-pub — reproducible code for the SAHA flagship atlas paper

Analysis code for *"SAHA: a multimodal subcellular spatial atlas of human anatomy."*
A frozen, citable snapshot that reproduces the figures of this paper. Organized by figure.

- `fig3_germinal_center/` — Fig 3 + Extended Data Fig 4
  - `niches_isotype/` (Across-GI): UTAG niches, Ig-isotype maps, pure-IgG fraction, plasma-lineage egress (Fig 3a-f)
  - `gc_crossorgan/` (Cross-organ): LN-GC vs GALT-APE-GC — GC detection, DZ/LZ, radial, DE (Fig 3g-l, ED Fig 4)
- `fig5_ibd/` — Fig 5 (IBD niche topology)
  - `integration/`, `niche_topology/`, `build_inputs/`

## Relationship to other SAHA repos
- **`SAHA`** — the org's general/evergreen landing repo (kept separate; not this paper).
- **`saha-scimap`**, **`saha-maxfuse`**, CONCH pipeline — component/method repos owned by collaborators; they cover Fig 2 (CONCH histology) and Fig 4 (RNA–protein / MaxFuse, scimap).
- **`saha-data-access`** — portal / programmatic access.

## Data
Large objects (h5ad / rds / zarr) are on **Zenodo** — not in this repo. Per-figure-panel
**Source Data** tables accompany the manuscript (Nature Source Data requirement), staged
separately, not committed here.

Fig 1 (atlas integration) is produced elsewhere; Fig 2 and Fig 4 are the collaborator repos above.
