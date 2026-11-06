# LIFE Data Release — analysis & figure/table generation code

Code that **operates on the finished LIFE BIDS dataset (OpenNeuro ds009071)** to build the
**manuscript figures (Figs 1–4 and S1–S4)** and to **compute the manuscript
tables (Tables 1 and 2 + an instrument-reliability table), written as CSV**. 
This repository is the code companion to that data release. It contains the
figure-building scripts and the analyses that compute figure/table values from
the BIDS data. Tables are emitted as **CSV** (universal format) with the exact manuscript
values.

Author: **Tatiana Wolfe, Ph.D.** · tati.wolfe@gmail.com · v2026.11.06

---

## How to cite

This code is released under the **MIT License** (see `LICENSE`). It is free to
use, modify, and redistribute.

If you use this code, or data produced with it, please cite the LIFE dataset:

> Wolfe, T., Calvert, M. L., Isaac, L., Clontz, S., Shell, A., Kelly, E.,
> Dunn, L. B., Koscik, T., & James, G. A. *A multimodal neuroimaging and
> cognitive flexibility dataset in aging adults: The Lifespan Imaging and
> Function in Elders (LIFE) Study.* OpenNeuro.
> doi:10.18112/openneuro.ds009071.v1.0.0

`CITATION.cff` carries the same citation in machine-readable form; GitHub's
"Cite this repository" button and most reference managers read it directly.

Citation is requested, not required by the license.

---

## 1. Requirements

**Python** ≥ 3.10 with:

```bash
pip install -r requirements.txt
```

(numpy, pandas, scipy, statsmodels, matplotlib, nibabel, python-docx)

**External neuroimaging tools** (needed only for the stages noted below):

| Tool | Used by |
|---|---|
| FreeSurfer 7.x (`recon-all`, `asegstats2table`, `aparcstats2table`) | structural derivatives, Euler numbers |
| FSL 6.x (`mcflirt`) | framewise displacement, inter-b0 motion |
| DSI Studio | DWI QC (NDC, bad slices), GQI reconstruction, DTI-from-SRC |

## 2. Point the code at your data (`LIFE_BASE`)

**Set the dataset path once.** Every script in `LIFE_code/` reads a single variable,
`LIFE_BASE` — the folder that *contains* `bids_dataset_LIFE/` (so the BIDS
dataset is at `$LIFE_BASE/bids_dataset_LIFE`). Edit the one line in
[`set_path.sh`](set_path.sh) and source it before running anything:

```bash
# edit LIFE_BASE in set_path.sh, then:
source set_path.sh
```

Equivalently, export it yourself:

```bash
export LIFE_BASE=/path/to/LIFE            # folder that contains bids_dataset_LIFE/
export FREESURFER_HOME=/Applications/freesurfer/7.4.1   # structural stages only
```

If `LIFE_BASE` is unset, each script falls back to auto-detecting the repo root
(useful only when the code sits inside the dataset tree).

Expected BIDS layout under `$LIFE_BASE`:

```
LIFE/
  bids_dataset_LIFE/
    participants.tsv
    phenotype/
    sub-*/{anat,func,dwi}/
    derivatives/
      freesurfer/  samseg/  dsi_studio_gqi/
      technical_validation/{structural,wmh,rsfmri,dwi,t2star,behavioral,noise_floor}/
```

## 3. Repository layout

```
set_path.sh                    edit LIFE_BASE here (single dataset-path entry)
LIFE_code/                          folders are numbered in run order
  1_freesurfer/                recon-all driver (operates on BIDS T1w)
  3_for_data_release/          DWI/DTI QC (Figure 4) + Table 1 (CSV)
  2_reliability/               noise-floor reliability analysis
  4_data_descriptor_generation/  figure builders (Figs 1–4, S1–S4), the analyses
                                 that compute figure/table values, and the CSV
                                 table builders (flat package — scripts import
                                 `config.py` as siblings)
```

## 4. What generates what (manuscript deliverable → code)

**Figures** — built end-to-end by this release:

| Figure | Builder | Drawing/data dependencies |
|---|---|---|
| **Figure 1** (structural QC & morphometry) | `4_data_descriptor_generation/make_figure1.py` | `make_figures.py`, `config.py`; data ← `01_structural_validation.py` |
| **Figure 2** (WM T2* & WMH) | `4_data_descriptor_generation/make_figure2.py` | `make_figures.py`, `make_t2star_panel.py` (panel A), `t2star_regional/make_regional_t2star_figure.py` (panel B); data ← `t2star_regional/extract_regional_t2star.py`, `02_wmh_validation.py` |
| **Figure 3** (rs-fMRI QC) | `4_data_descriptor_generation/make_figure3.py` | `make_fd_plot.py`, `make_tsnr_plot.py`; data ← `cohort_fd.py`, `cohort_tsnr.py`, `make_group_tissue_masks.py`, `03_rsfmri_validation.py` |
| **Figure 4** (DWI QC) | `4_data_descriptor_generation/make_figure4.py` | `make_figures.py`, `make_dwi_figures.py`, `4_data_descriptor_generation/assets/Fig4A.jpg` (panel A render, see below); data ← `3_for_data_release/15_dwi_qc_python.py`, `3_for_data_release/13_dti_from_src.py`, `06_dti_wm_from_qsdr.py`, `07_b0_motion.py`, `04_dwi_validation.py` |
| **Figures S1–S4** (noise floor) | `4_data_descriptor_generation/make_figures_noise.py` | `noise_floor.py`, `make_figures.py`, `make_t2star_panel.py` |

**Tables** — built from the BIDS data with the **exact manuscript values** and
written as **CSV** (each with a `*_note.txt` sidecar carrying the table note).
Outputs land in `derivatives/technical_validation/tables/`.

| Table | Builder (CSV) | Output file |
|---|---|---|
| **Table 1** (demographics + health/SES) | `3_for_data_release/build_table1_demographics.py` | `Table1_demographics.csv` |
| **Table 2** (phenotyping: descriptives + age) — merged with reliability below into the manuscript's single Table 2 | `4_data_descriptor_generation/build_tables.py` | `Table2_phenotyping.csv` |
| **Table 2 (reliability)** — the reliability portion of the merged Table 2 | `4_data_descriptor_generation/build_reliability_tables.py` | `Table2_reliability.csv` (+ `nihtb_reliability.csv`, `loneliness_reliability.csv`) |

**Not included** (text / references / Word formatting), by design: figure-legend
and supplement-narrative generators, `.docx` table formatters, and pipeline
orchestration wrappers.

## 5. Reproduce

`source set_path.sh` first (Section 2). Folders are numbered in run order.

**Stage 1 — build derivatives from BIDS** (the three folders are independent;
any order, or in parallel):

```bash
# 1_freesurfer  → structural derivatives (recon-all): Fig 1, Table 2
python LIFE_code/1_freesurfer/3_freesurfer_analysis.py

# 3_for_data_release  → DWI QC + DTI + item-level alpha: Fig 4, Table 2 (reliability)
python LIFE_code/3_for_data_release/13_dti_from_src.py
python LIFE_code/3_for_data_release/15_dwi_qc_python.py
python LIFE_code/3_for_data_release/11_dwi_qc_dsistudio.py

# 2_reliability  → standalone noise-floor reliability (reads raw BIDS)
python LIFE_code/2_reliability/4_noise_floor_reliability.py
```

**Stage 2 — compute the figure/table values** (`4_data_descriptor_generation/`):

Panel A of Figure 4 is the one figure element no script produces: a DSI
Studio rendering of the cohort whole-brain tractogram colored by quantitative
anisotropy, beside axial slices of the cohort spin distribution function, made
in the DSI Studio GUI (Hou release, 2026) from cohort-level LIFE `.fib` files
(n = 57). **It is illustrative only.** Nothing in the Technical Validation is
measured from it, and no statistic in the manuscript derives from it; it is
there to show what the released diffusion data support. The render used in the
published figure ships with the code at
`LIFE_code/4_data_descriptor_generation/assets/Fig4A.jpg`, and
`make_figure4.py` picks it up automatically, so the figure reproduces from a
clean checkout. A replacement can be dropped into the figures directory under
any of the names in `TRACT_CANDIDATES`, which takes precedence over the
shipped copy. If neither is present the script draws a labeled placeholder
box rather than failing.

```bash
for s in 01_structural_validation 02_wmh_validation 03_rsfmri_validation \
         04_dwi_validation 05_behavioral_validation 06_dti_wm_from_qsdr \
         07_b0_motion cohort_fd cohort_tsnr \
         make_group_tissue_masks noise_floor; do
  python LIFE_code/4_data_descriptor_generation/$s.py
done
# Regional T2* for Figure 2B (its own subdirectory; needs FreeSurfer on PATH)
python LIFE_code/4_data_descriptor_generation/t2star_regional/extract_regional_t2star.py
# (mk_tsnr_atlas_masks.sh is a helper for the tSNR masks, if needed)
```

**Stage 3 — build the deliverables:**

```bash
# figures
python LIFE_code/4_data_descriptor_generation/make_figure1.py
python LIFE_code/4_data_descriptor_generation/make_figure2.py
python LIFE_code/4_data_descriptor_generation/make_figure3.py
python LIFE_code/4_data_descriptor_generation/make_figure4.py
python LIFE_code/4_data_descriptor_generation/make_figures_noise.py
# tables (CSV, exact manuscript values, under derivatives/.../tables/)
python LIFE_code/4_data_descriptor_generation/build_tables.py
python LIFE_code/4_data_descriptor_generation/build_reliability_tables.py
python LIFE_code/3_for_data_release/build_table1_demographics.py
```

Analyses **guard** missing inputs (skip with a clear message), so partial
pipelines run cleanly. Table 1 needs no upstream compute and can run anytime.

## 6. Notes on scope

- The manuscript describes an idealized pipeline (Schaefer-400 / NODDI /
  TractSeg / LST); the released derivatives use **DSI Studio GQI + FreeSurfer
  DKT + SAMSEG**. Scripts consume what exists and skip the rest.
- This repository contains **only code that operates on the finished BIDS
  dataset**. The upstream pipeline that *builds* BIDS — DICOM→BIDS conversion,
  behavioral organization, and data-dictionary generation — is deliberately
  **not** included.
- Code for the separate cognitive-flexibility/connectometry manuscript
  (`brain-behavior/`, `connectometry/`, `dsi_diffusion/`) is **not** included —
  this repository is scoped to the data-descriptor content only.

## 7. License & citation

Released under the **MIT License** — free to use, modify and redistribute.
See `LICENSE` for the terms.

Citation is **requested, not required**: if you use this code, please cite the
LIFE dataset (doi:10.18112/openneuro.ds009071.v1.0.0). See *How to cite* above,
or `CITATION.cff` for the machine-readable form.
