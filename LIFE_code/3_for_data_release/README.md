# LIFE data-release scripts (BIDS-operating QC subset)

Scripts that operate on the **finished LIFE BIDS dataset** to compute the
diffusion QC and reliability values used by the manuscript figures. Paths
resolve to the `LIFE/` root automatically, or override with
`export LIFE_BASE=/path/to/LIFE`.

> The upstream BIDS-building pipeline (DICOM→BIDS, behavioral organization,
> phenotype/data-dictionary generation) and the `.docx` table/supplement
> formatters are **not** part of this release.

## Scripts

| Script | Produces | Reads |
|---|---|---|
| `build_table1_demographics.py` | **Table 1** values → `derivatives/technical_validation/tables/Table1_demographics.csv` (universal CSV, not .docx) | released `participants.tsv` + BIDS `phenotype/` (education) |
| `15_dwi_qc_python.py` | DWI QC (per-shell SNR, NDC, bad slices) → `derivatives/technical_validation/dwi/dwi_qc_python.csv` (feeds **Figure 4**) | BIDS `dwi/` |
| `11_dwi_qc_dsistudio.py` | DWI QC via DSI Studio (cross-check) | BIDS `dwi/` / DSI Studio SRC |
| `13_dti_from_src.py` | DTI metrics (FA/MD) → DWI derivatives (feeds **Figure 4**) | DSI Studio SRC (from BIDS `dwi/`) |
