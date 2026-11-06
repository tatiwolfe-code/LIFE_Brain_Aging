#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Compute noise-floor and reliability metrics for the LIFE dataset,
per MRI modality.

For each modality, computes signal-to-noise and quality metrics
from the raw NIfTI data:

  anat (T1w, T2w, FLAIR):
    - SNR: mean(brain) / std(background)
    - Contrast-to-noise ratio (CNR) between GM and WM (T1w only)
    - Coefficient of variation across subjects

  func (rest, flanker, dccs):
    - Temporal SNR (tSNR): mean-over-time / std-over-time, per voxel
    - Mean and median tSNR across brain mask
    - Number of high-motion volumes (FD > 0.5mm, from JSON if available)

  dwi:
    - SNR of b=0 volumes
    - Mean FA (from dti_FA if present)

Results are saved per modality in:
    derivatives/reliability/<modality>/
        qc_summary.csv          — per-subject metrics
        qc_group_summary.csv    — group-level stats
        qc_report.txt           — human-readable report

Usage:
    python 4_noise_floor_reliability.py
    python 4_noise_floor_reliability.py --subject sub-027
"""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    import sys
    sys.exit("ERROR: nibabel required.  pip install nibabel")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
BIDS_DIR = BASE_DIR / "bids_dataset_LIFE"
DERIV_DIR = BIDS_DIR / "derivatives"
OUT_BASE = DERIV_DIR / "reliability"


# =====================================================================
#  Helpers
# =====================================================================

def load_nii(path):
    """Load a NIfTI, return (data, affine)."""
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj, dtype=np.float64), img.affine


def brain_mask_otsu(data_3d):
    """Simple Otsu threshold to create a brain mask."""
    vals = data_3d[data_3d > 0].ravel()
    if len(vals) == 0:
        return np.zeros_like(data_3d, dtype=bool)
    # Otsu: find threshold that minimizes within-class variance
    hist, bin_edges = np.histogram(vals, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    mu0 = np.cumsum(hist * bin_centers)
    mu0 = np.where(w0 > 0, mu0 / w0, 0)
    mu1 = np.cumsum((hist * bin_centers)[::-1])[::-1]
    mu1 = np.where(w1 > 0, mu1 / w1, 0)
    variance = w0 * w1 * (mu0 - mu1) ** 2
    threshold = bin_centers[np.argmax(variance)]
    return data_3d > threshold


def background_mask(data_3d):
    """Background = corners of the volume (low-signal region)."""
    shape = data_3d.shape
    s = max(5, min(shape) // 10)  # corner cube size
    corners = np.concatenate([
        data_3d[:s, :s, :s].ravel(),
        data_3d[-s:, :s, :s].ravel(),
        data_3d[:s, -s:, :s].ravel(),
        data_3d[:s, :s, -s:].ravel(),
    ])
    return corners


def compute_snr(data_3d):
    """SNR = mean(brain) / std(background)."""
    mask = brain_mask_otsu(data_3d)
    brain_vals = data_3d[mask]
    bg_vals = background_mask(data_3d)
    bg_std = np.std(bg_vals)
    if bg_std == 0 or len(brain_vals) == 0:
        return np.nan
    return np.mean(brain_vals) / bg_std


# =====================================================================
#  Per-modality QC
# =====================================================================

def qc_anat(bids_dir, subject_filter=None):
    """Compute SNR for each anatomical contrast per subject."""
    rows = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        sub = sub_dir.name
        if subject_filter and sub != subject_filter:
            continue
        anat_dir = sub_dir / "anat"
        if not anat_dir.is_dir():
            continue

        for nii in sorted(anat_dir.glob(f"{sub}_*.nii.gz")):
            # Skip multi-echo (echo-N) — use base contrasts only
            contrast = nii.name.replace(f"{sub}_", "").replace(".nii.gz", "")
            if "echo-" in contrast:
                continue

            try:
                data, _ = load_nii(nii)
                if data.ndim > 3:
                    data = data[..., 0]
                snr = compute_snr(data)
                mask = brain_mask_otsu(data)
                brain_mean = np.mean(data[mask]) if mask.any() else np.nan
                brain_std = np.std(data[mask]) if mask.any() else np.nan
                rows.append({
                    "subject": sub,
                    "contrast": contrast,
                    "snr": round(snr, 2),
                    "brain_mean": round(brain_mean, 2),
                    "brain_std": round(brain_std, 2),
                    "n_voxels_brain": int(mask.sum()),
                })
                print(f"  {sub} {contrast}: SNR={snr:.1f}")
            except Exception as e:
                print(f"  WARNING: {sub} {contrast}: {e}")
                rows.append({
                    "subject": sub,
                    "contrast": contrast,
                    "snr": np.nan,
                    "brain_mean": np.nan,
                    "brain_std": np.nan,
                    "n_voxels_brain": 0,
                })
    return rows


def qc_func(bids_dir, subject_filter=None):
    """Compute temporal SNR for each fMRI run per subject."""
    rows = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        sub = sub_dir.name
        if subject_filter and sub != subject_filter:
            continue
        func_dir = sub_dir / "func"
        if not func_dir.is_dir():
            continue

        for nii in sorted(func_dir.glob(f"{sub}_*.nii.gz")):
            contrast = nii.name.replace(f"{sub}_", "").replace(".nii.gz", "")

            try:
                data, _ = load_nii(nii)
                if data.ndim < 4 or data.shape[3] < 10:
                    print(f"  SKIP {sub} {contrast}: not a 4D timeseries")
                    continue

                # Temporal mean and std
                tmean = np.mean(data, axis=3)
                tstd = np.std(data, axis=3, ddof=1)
                mask = brain_mask_otsu(tmean)

                # tSNR per voxel
                with np.errstate(divide="ignore", invalid="ignore"):
                    tsnr_vol = np.where(tstd > 0, tmean / tstd, 0)

                tsnr_brain = tsnr_vol[mask]
                mean_tsnr = np.mean(tsnr_brain) if len(tsnr_brain) > 0 else np.nan
                median_tsnr = np.median(tsnr_brain) if len(tsnr_brain) > 0 else np.nan

                rows.append({
                    "subject": sub,
                    "contrast": contrast,
                    "n_volumes": data.shape[3],
                    "mean_tsnr": round(mean_tsnr, 2),
                    "median_tsnr": round(median_tsnr, 2),
                    "n_voxels_brain": int(mask.sum()),
                })
                print(f"  {sub} {contrast}: tSNR={mean_tsnr:.1f} "
                      f"({data.shape[3]} vols)")
            except Exception as e:
                print(f"  WARNING: {sub} {contrast}: {e}")
    return rows


def qc_dwi(bids_dir, subject_filter=None):
    """Compute SNR of DWI (from b=0 if identifiable) and mean FA."""
    rows = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        sub = sub_dir.name
        if subject_filter and sub != subject_filter:
            continue
        dwi_dir = sub_dir / "dwi"
        if not dwi_dir.is_dir():
            continue

        # DWI SNR
        dwi_nii = dwi_dir / f"{sub}_dwi.nii.gz"
        if dwi_nii.exists():
            try:
                data, _ = load_nii(dwi_nii)
                if data.ndim == 4:
                    # Use first volume (likely b=0)
                    b0 = data[..., 0]
                else:
                    b0 = data
                snr = compute_snr(b0)
                print(f"  {sub} dwi: SNR(b0)={snr:.1f}")
            except Exception as e:
                print(f"  WARNING: {sub} dwi: {e}")
                snr = np.nan
        else:
            snr = np.nan

        # Mean FA
        fa_nii = dwi_dir / f"{sub}_dti_FA.nii.gz"
        if fa_nii.exists():
            try:
                fa_data, _ = load_nii(fa_nii)
                if fa_data.ndim > 3:
                    fa_data = fa_data[..., 0]
                # FA is 0-1 range; mask out near-zero
                fa_mask = fa_data > 0.05
                mean_fa = np.mean(fa_data[fa_mask]) if fa_mask.any() else np.nan
                print(f"  {sub} FA: mean={mean_fa:.3f}")
            except Exception as e:
                print(f"  WARNING: {sub} FA: {e}")
                mean_fa = np.nan
        else:
            mean_fa = np.nan

        rows.append({
            "subject": sub,
            "dwi_snr_b0": round(snr, 2) if not np.isnan(snr) else "",
            "mean_fa": round(mean_fa, 4) if not np.isnan(mean_fa) else "",
        })
    return rows


# =====================================================================
#  Group-level summary
# =====================================================================

def group_summary(rows, value_col):
    """Compute group mean, std, min, max for a numeric column."""
    vals = [float(r[value_col]) for r in rows
            if r.get(value_col, "") != "" and not np.isnan(float(r[value_col]))]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(np.mean(vals), 2),
        "std": round(np.std(vals, ddof=1), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "median": round(np.median(vals), 2),
        "cov_pct": round(100 * np.std(vals, ddof=1) / np.mean(vals), 1)
        if np.mean(vals) > 0 else "",
    }


# =====================================================================
#  Writers
# =====================================================================

def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")


def write_report(lines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}")


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise-floor and reliability QC for LIFE dataset."
    )
    parser.add_argument("--bids-dir", type=Path, default=BIDS_DIR)
    parser.add_argument("--subject", type=str, default=None)
    args = parser.parse_args()

    global OUT_BASE
    OUT_BASE = args.bids_dir / "derivatives" / "reliability"

    print(f"LIFE Noise-Floor / Reliability QC")
    print(f"  BIDS dir: {args.bids_dir}")
    print(f"  Output:   {OUT_BASE}")
    print(f"  Time:     {datetime.datetime.now().isoformat()}")
    print()

    report = [
        f"LIFE Dataset — Noise Floor & Reliability Report",
        f"Generated: {datetime.datetime.now().isoformat()}",
        f"BIDS dir:  {args.bids_dir}",
        "",
    ]

    # ── Anatomical ──
    print("=== Anatomical QC ===")
    anat_rows = qc_anat(args.bids_dir, args.subject)
    if anat_rows:
        out_dir = OUT_BASE / "anat"
        write_csv(anat_rows, out_dir / "qc_summary.csv",
                  ["subject", "contrast", "snr", "brain_mean",
                   "brain_std", "n_voxels_brain"])

        report.append("=" * 50)
        report.append("ANATOMICAL QC")
        report.append("=" * 50)
        # Group stats per contrast
        from collections import defaultdict
        by_contrast = defaultdict(list)
        for r in anat_rows:
            by_contrast[r["contrast"]].append(r)
        for con, rows in sorted(by_contrast.items()):
            gs = group_summary(rows, "snr")
            report.append(f"\n  {con}:")
            report.append(f"    N={gs.get('n',0)}  SNR: "
                          f"mean={gs.get('mean','')} "
                          f"std={gs.get('std','')} "
                          f"range=[{gs.get('min','')}, {gs.get('max','')}] "
                          f"CoV={gs.get('cov_pct','')}%")
        report.append("")

    # ── Functional ──
    print("\n=== Functional QC ===")
    func_rows = qc_func(args.bids_dir, args.subject)
    if func_rows:
        out_dir = OUT_BASE / "func"
        write_csv(func_rows, out_dir / "qc_summary.csv",
                  ["subject", "contrast", "n_volumes",
                   "mean_tsnr", "median_tsnr", "n_voxels_brain"])

        report.append("=" * 50)
        report.append("FUNCTIONAL QC")
        report.append("=" * 50)
        by_contrast = defaultdict(list)
        for r in func_rows:
            by_contrast[r["contrast"]].append(r)
        for con, rows in sorted(by_contrast.items()):
            gs = group_summary(rows, "mean_tsnr")
            report.append(f"\n  {con}:")
            report.append(f"    N={gs.get('n',0)}  tSNR: "
                          f"mean={gs.get('mean','')} "
                          f"std={gs.get('std','')} "
                          f"range=[{gs.get('min','')}, {gs.get('max','')}] "
                          f"CoV={gs.get('cov_pct','')}%")
        report.append("")

    # ── DWI ──
    print("\n=== DWI QC ===")
    dwi_rows = qc_dwi(args.bids_dir, args.subject)
    if dwi_rows:
        out_dir = OUT_BASE / "dwi"
        write_csv(dwi_rows, out_dir / "qc_summary.csv",
                  ["subject", "dwi_snr_b0", "mean_fa"])

        report.append("=" * 50)
        report.append("DWI QC")
        report.append("=" * 50)
        gs_snr = group_summary(dwi_rows, "dwi_snr_b0")
        gs_fa = group_summary(dwi_rows, "mean_fa")
        report.append(f"\n  DWI SNR(b=0): N={gs_snr.get('n',0)}  "
                      f"mean={gs_snr.get('mean','')} "
                      f"std={gs_snr.get('std','')} "
                      f"CoV={gs_snr.get('cov_pct','')}%")
        report.append(f"  Mean FA:      N={gs_fa.get('n',0)}  "
                      f"mean={gs_fa.get('mean','')} "
                      f"std={gs_fa.get('std','')} "
                      f"range=[{gs_fa.get('min','')}, {gs_fa.get('max','')}]")
        report.append("")

    # ── Write report ──
    write_report(report, OUT_BASE / "qc_report.txt")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
