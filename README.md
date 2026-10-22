# saha-2026-pub — reproducible code for the SAHA flagship atlas paper

Analysis code for *"SAHA: a multimodal subcellular spatial atlas of human anatomy."*
A frozen, citable snapshot that reproduces the figures of this paper. Organized by figure.

- `fig1_atlas_integration/` — Fig 1d–e, Extended Data Figs 1–2 (Harmony joint and per-organ UMAPs, InSituType)
- `fig3_germinal_center/` — Fig 3 + Extended Data Fig 4
  - `niches_isotype/` (Across-GI): UTAG niches, Ig-isotype maps, pure-IgG fraction, plasma-lineage egress (Fig 3a-f)
  - `gc_crossorgan/` (Cross-organ): LN-GC vs GALT-APE-GC — GC detection, DZ/LZ, radial, DE (Fig 3g-l, ED Fig 4)
- `fig5_ibd/` — Fig 5 (IBD niche topology)
  - `integration/`, `niche_topology/`, `build_inputs/`

## Relationship to other SAHA repos
- **`SAHA`** — the org's general/evergreen landing repo (kept separate; not this paper).
- **Fig 2 (whole-slide H&E histology)** — [`rendeirolab/LazySlide`](https://github.com/rendeirolab/LazySlide) (CONCH / CellViT pipeline; also Fig 5c–d IBD histology). Maintained by the Rendeiro lab (CeMM).
- **Fig 4 (RNA–protein integration)** — [`saha-project/saha-maxfuse`](https://github.com/saha-project/saha-maxfuse) (MaxFuse cell matching) and [`saha-project/saha-scimap`](https://github.com/saha-project/saha-scimap) (protein gating / annotation).
- **`saha-data-access`** — portal / programmatic access.

## Data
Large objects (SpatialData zarr) are not in this repo. They are openly available at
`s3://saha-public` (endpoint `https://s3.wasabisys.com`, no credentials; see
[`saha-data-access`](https://github.com/saha-project/saha-data-access)) and move to the AWS Open Data
Program and Zenodo at publication. Per-figure-panel **Source Data** tables accompany the manuscript.

Fig 2 and Fig 4 live in the repos linked above.
