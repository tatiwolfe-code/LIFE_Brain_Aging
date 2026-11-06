# 4_data_descriptor_generation — figures & table data

<!-- ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders ***** -->
<!-- ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com ***** -->

Scripts that operate on the **finished LIFE BIDS dataset** to build the
manuscript **figures (Figs 1–4 and S1–S4)** and to compute the **table values**
(written as CSV). No manuscript text, legends, references, or `.docx` are
produced here.

## Setup

```bash
export LIFE_BASE=/path/to/LIFE          # folder containing bids_dataset_LIFE/
export FREESURFER_HOME=/Applications/freesurfer/7.4.1   # structural stages only
pip install -r ../../requirements.txt   # numpy pandas scipy statsmodels matplotlib nibabel
```

Paths resolve from `config.py` (via `LIFE_BASE`). All outputs land under
`bids_dataset_LIFE/derivatives/technical_validation/` (figures under `.../figures`
when run standalone; table CSVs under `.../tables`).

## Figure builders

| Script | Builds |
|---|---|
| `make_figure1.py` | **Figure 1** — structural QC & morphometry |
| `make_figure2.py` | **Figure 2** — white-matter T2* & WMH |
| `make_figure3.py` | **Figure 3** — rs-fMRI QC |
| `make_figure4.py` | **Figure 4** — DWI QC (panel A from `assets/Fig4A.jpg`: illustrative only, rendered in DSI Studio (Hou, 2026) from cohort-level LIFE `.fib` files, n = 57; no script produces it) |
| `make_figures_noise.py` | **Figures S1–S4** — noise floor (structural, T2*, BOLD, DWI) |

## Table-value builders (CSV, exact manuscript values)

| Script | Builds |
|---|---|
| `build_tables.py` | **Table 2** (phenotyping) |
| `build_reliability_tables.py` | **Table 2** (reliability panel — merged into Table 2) |

(Each writes a `*_note.txt` sidecar with the table note. **Table 1** is built by
`../2_for_data_release/build_table1_demographics.py`.)

## Analyses that compute the plotted / tabulated values (from BIDS)

| Script | Produces |
|---|---|
| `01_structural_validation.py` | Euler numbers, eTIV, volume/thickness × age (Fig 1) |
| `02_wmh_validation.py` | WMH volume & WMH–age (Fig 2) |
| `03_rsfmri_validation.py` | FD, tSNR, network structure (Fig 3) |
| `04_dwi_validation.py` | per-shell SNR, residuals (Fig 4) |
| `05_behavioral_validation.py` | cognitive/behavioral descriptives & α (Tables 2, 3) |
| `06_dti_wm_from_qsdr.py`, `07_b0_motion.py` | whole-WM DTI metrics, inter-b0 motion (Fig 4) |
| `cohort_fd.py`, `cohort_tsnr.py` | FD/tSNR cohort data + MNI maps (Fig. 3) |
| `t2star_regional/extract_regional_t2star.py` | Regional T2* per FreeSurfer structure (Fig. 2B) |
| `t2star_regional/make_regional_t2star_figure.py` | Panel B drawing core; also builds the standalone version |
| `make_group_tissue_masks.py`, `mk_tsnr_atlas_masks.sh` | group GM/WM masks (Fig 3C) |
| `noise_floor.py` | per-modality noise-floor CSVs (Figs S1–S4) |
| `cogbehav_data.py` | cognitive/behavioral data helpers (imported by the table builder) |

## Shared infrastructure (imported, not run alone)

`config.py` (paths, phenotype, stats helpers) · `make_figures.py` (drawing cores)
· `make_dwi_figures.py` · `make_fd_plot.py` · `make_tsnr_plot.py`
· `make_t2star_panel.py` · `navia.csv` (T2* colormap).

## Notes

- **External tools:** FreeSurfer 7.x (structural) and FSL 6.x `mcflirt` (FD /
  inter-b0 motion). DWI QC/DTI inputs for Figure 4 come from
  `../2_for_data_release/{15_dwi_qc_python.py, 13_dti_from_src.py}`.
- Analyses **guard** missing inputs (skip with a clear message), so partial
  pipelines run cleanly.
- Covariate policy: education was dropped (non-significant in every model);
  age and sex (and eTIV for volumes) are the covariates used.
