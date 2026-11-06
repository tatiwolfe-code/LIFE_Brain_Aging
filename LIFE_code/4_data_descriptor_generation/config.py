#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Shared configuration and helper functions for the LIFE Data Release
Technical Validation analyses.

This module centralises:
  - dataset paths (auto-detected from this file's location, override with
    the LIFE_BASE environment variable),
  - subject discovery from participants.tsv,
  - phenotype loading (NIH Toolbox + REDCap),
  - FreeSurfer sourcing for shell-based tools,
  - small statistics helpers (partial correlation, ANCOVA, Cronbach's alpha)
    used across the validation scripts.

All Technical Validation scripts import from here so that paths and
conventions stay consistent.

Design notes
------------
* Single-session study: no ses- labels anywhere.
* The manuscript describes an idealised pipeline (Schaefer-400 / NODDI /
  TractSeg / LST). The data actually on disk use DSI Studio GQI +
  FreeSurfer DKT + SAMSEG. Each validation script consumes the derivatives
  that exist and *guards* (skips with a clear message) any analysis whose
  inputs have not yet been generated, so the same script becomes fully
  functional once those derivatives are produced.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
# BASE_DIR resolves to .../LIFE regardless of machine:
#   this file lives at  LIFE/Data manuscript/code/config.py
#   parents[2]          -> LIFE
# Override with:  export LIFE_BASE=/path/to/LIFE
BASE_DIR = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
BIDS_DIR = BASE_DIR / "bids_dataset_LIFE"
DERIV_DIR = BIDS_DIR / "derivatives"
PHENO_DIR = BIDS_DIR / "phenotype"

FS_SUBJECTS_DIR = DERIV_DIR / "freesurfer"
SAMSEG_DIR = DERIV_DIR / "samseg"
DSI_DIR = DERIV_DIR / "dsi_studio_gqi"

# All Technical Validation outputs land here, one sub-folder per domain.
TV_DIR = DERIV_DIR / "technical_validation"

PARTICIPANTS_TSV = BIDS_DIR / "participants.tsv"

# ── Cohort definition ────────────────────────────────────────────────
# The released cohort is the older adults.
AGE_MIN = 60

# Technical Validation exclusion.
#
# sub-007 is an extreme outlier on BOTH FreeSurfer-derived structural quality
# measures: topological defect count and white-matter lesion (WMH) burden.
# It is therefore dropped from the FreeSurfer-based Technical Validation
# analyses and from Figures 1 and 2, which are reported on 56 of the 57
# released reconstructions.
#
# Scope: this applies ONLY to the FreeSurfer/structural and WMH analyses.
# It does NOT apply to the resting-state fMRI or diffusion analyses, which
# are reported on their full subsamples (rs-fMRI n = 56, limited by sub-062
# having no resting-state run; DWI n = 57).
#
# The participant's data are released in full; only these validation
# statistics exclude it.
STRUCTURAL_EXCLUDE = frozenset({"sub-007"})


def older_ids(pheno, age_min: int = AGE_MIN, exclude=frozenset()) -> set[str]:
    """IDs of the released older cohort, minus any excluded participants.

    `pheno` is a participant-indexed frame carrying an `age` column (e.g. the
    output of load_phenotype()). Pass exclude=STRUCTURAL_EXCLUDE for the
    FreeSurfer-based structural and WMH analyses; leave it at the default for
    every other modality.
    """
    age = pd.to_numeric(pheno["age"], errors="coerce")
    return set(pheno.index[age >= age_min]) - set(exclude)

# FreeSurfer install on the analysis machine (matches the existing
# freesurfer_analysis.py convention). Override with FREESURFER_HOME.
FREESURFER_HOME = Path(
    os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1")
)
FS_SETUP = FREESURFER_HOME / "SetUpFreeSurfer.sh"


# =====================================================================
#  Output directory helper
# =====================================================================

def outdir(domain: str) -> Path:
    """Return (and create) the technical-validation output dir for a domain."""
    d = TV_DIR / domain
    d.mkdir(parents=True, exist_ok=True)
    return d


# =====================================================================
#  Subject discovery + participants table
# =====================================================================

def load_participants() -> pd.DataFrame:
    """Load participants.tsv; index by participant_id. 'n/a' -> NaN."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    df["participant_id"] = df["participant_id"].astype(str)
    return df.set_index("participant_id", drop=False)


def subjects_with(modality_flag: str | None = None) -> list[str]:
    """
    Return sorted subject IDs. If modality_flag is one of the participants.tsv
    availability columns ('anat','dwi','func',...), keep only subjects flagged 1.
    """
    df = load_participants()
    if modality_flag and modality_flag in df.columns:
        df = df[df[modality_flag] == 1]
    return sorted(df["participant_id"].tolist())


def subject_dirs(bids_dir: Path = BIDS_DIR, subject: str | None = None):
    """Yield (sub_label, sub_dir) for every sub-* directory on disk."""
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if sub_dir.is_dir() and (subject is None or sub_dir.name == subject):
            yield sub_dir.name, sub_dir


# =====================================================================
#  Phenotype loading
# =====================================================================

def load_nih_toolbox() -> pd.DataFrame:
    p = PHENO_DIR / "nih_toolbox.tsv"
    df = pd.read_csv(p, sep="\t", na_values=["n/a", "N/A", ""])
    return df.set_index("participant_id", drop=False)


def load_redcap() -> pd.DataFrame:
    p = PHENO_DIR / "redcap_questionnaires.tsv"
    df = pd.read_csv(p, sep="\t", na_values=["n/a", "N/A", ""])
    return df.set_index("participant_id", drop=False)


def load_phenotype() -> pd.DataFrame:
    """
    Merge participants + NIH Toolbox + REDCap into one subject-level frame.
    'education' and 'phq' are surfaced from REDCap (dcf_education, phq_total)
    because they are used as covariates throughout the validation scripts.
    """
    part = load_participants()
    nih = load_nih_toolbox()
    red = load_redcap()

    df = part.join(nih.drop(columns=["participant_id"], errors="ignore"),
                   how="left")
    df = df.join(red.drop(columns=["participant_id"], errors="ignore"),
                 how="left")

    # Convenience aliases used as covariates in the analyses.
    if "dcf_education" in df.columns:
        df["education"] = pd.to_numeric(df["dcf_education"], errors="coerce")
    if "phq_total" in df.columns:
        df["phq"] = pd.to_numeric(df["phq_total"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    # Numeric sex code for regression (M=1, F=0).
    df["sex_code"] = df["sex"].map({"M": 1, "F": 0})
    return df


# =====================================================================
#  FreeSurfer shell wrapper (for Euler numbers, stats2table, etc.)
# =====================================================================

def run_fs_cmd(cmd_str: str, threads: int = 4, capture: bool = True):
    """Run a shell command with FreeSurfer sourced. Returns CompletedProcess."""
    wrapped = (
        f"export FREESURFER_HOME={FREESURFER_HOME}; "
        f"source {FS_SETUP}; "
        f"export SUBJECTS_DIR={FS_SUBJECTS_DIR}; "
        f"export OMP_NUM_THREADS={threads}; "
        f"{cmd_str}"
    )
    return subprocess.run(
        ["bash", "-lc", wrapped],
        capture_output=capture, text=True,
    )


# =====================================================================
#  Statistics helpers
# =====================================================================

def partial_corr(df: pd.DataFrame, x: str, y: str, covars: list[str],
                 method: str = "pearson") -> dict:
    """
    Partial correlation of x and y controlling for `covars`, computed as the
    correlation of the residuals of x~covars and y~covars.
    Returns dict(r, p, n).
    """
    from scipy import stats
    cols = [x, y] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < len(covars) + 3:
        return {"r": np.nan, "p": np.nan, "n": n}

    C = np.column_stack([np.ones(n)] + [d[c].values for c in covars])

    def resid(v):
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta

    rx = resid(d[x].values.astype(float))
    ry = resid(d[y].values.astype(float))
    if method == "spearman":
        rx = stats.rankdata(rx)
        ry = stats.rankdata(ry)
    r, _ = stats.pearsonr(rx, ry)
    # p-value adjusted for the number of covariates.
    dfree = n - len(covars) - 2
    if dfree <= 0 or abs(r) >= 1:
        return {"r": r, "p": np.nan, "n": n}
    t = r * np.sqrt(dfree / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), dfree)
    return {"r": float(r), "p": float(p), "n": int(n)}


def ancova(df: pd.DataFrame, dv: str, predictor: str,
           covars: list[str]) -> dict:
    """
    OLS ANCOVA: dv ~ predictor + covars. Returns the predictor's
    coefficient, t, p, partial eta^2, and n. Requires statsmodels.
    """
    import statsmodels.formula.api as smf

    cols = [dv, predictor] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < len(cols) + 2:
        return {"beta": np.nan, "t": np.nan, "p": np.nan,
                "partial_eta2": np.nan, "n": len(d)}
    terms = " + ".join([predictor] + list(covars)) if covars else predictor
    model = smf.ols(f"Q('{dv}') ~ {terms}", data=d).fit()
    beta = model.params.get(predictor, np.nan)
    t = model.tvalues.get(predictor, np.nan)
    p = model.pvalues.get(predictor, np.nan)
    # partial eta^2 = t^2 / (t^2 + df_resid)
    peta2 = (t**2) / (t**2 + model.df_resid) if np.isfinite(t) else np.nan
    return {"beta": float(beta), "t": float(t), "p": float(p),
            "partial_eta2": float(peta2), "n": int(len(d))}


def cronbach_alpha(item_matrix: np.ndarray) -> float:
    """
    Cronbach's alpha for an (n_subjects x n_items) matrix. Rows with any NaN
    are dropped listwise. Returns NaN if < 2 items or < 2 rows remain.
    """
    X = np.asarray(item_matrix, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n_items = X.shape[1]
    if n_items < 2 or X.shape[0] < 2:
        return float("nan")
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((n_items / (n_items - 1)) * (1 - item_var.sum() / total_var))


def describe(series: pd.Series) -> dict:
    """mean/SD/min/max/n for a numeric series (NaNs dropped)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {"mean": float(s.mean()), "sd": float(s.std(ddof=1)),
            "min": float(s.min()), "max": float(s.max()), "n": int(len(s))}


def banner(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


if __name__ == "__main__":
    # Smoke test: print resolved paths and sample sizes.
    print(banner("LIFE Technical Validation — config"))
    print(f"BASE_DIR : {BASE_DIR}")
    print(f"BIDS_DIR : {BIDS_DIR}  (exists={BIDS_DIR.exists()})")
    print(f"DERIV_DIR: {DERIV_DIR}  (exists={DERIV_DIR.exists()})")
    print(f"TV_DIR   : {TV_DIR}")
    try:
        part = load_participants()
        print(f"participants.tsv: N={len(part)}")
        for m in ["anat", "dwi", "func"]:
            if m in part.columns:
                print(f"  with {m}: {int((part[m] == 1).sum())}")
    except Exception as e:  # noqa: BLE001
        print(f"participants.tsv not loaded: {e}")
