#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Pure-Python DWI quality control (nibabel only; NO DSI Studio required) — an
alternative to the DSI Studio native QC that avoids the binary/version issue
and runs anywhere the raw BIDS DWI is available.

Per subject, from sub-XXX/dwi/{nii.gz,bval,bvec}:
  - per-shell SNR (median over a brain mask of mean/std across shell volumes)
  - NDC-style index: mean correlation between each DWI volume and its
    nearest gradient-direction neighbor within the same shell (higher = more
    internally consistent; low outliers flag poor scans)
  - bad-slice count: axial slices whose in-mask mean signal is a strong low
    outlier vs the same slice across other same-shell volumes (signal dropout)

Output -> derivatives/technical_validation/dwi/
  dwi_qc_python.csv (+ printed group summary)

Usage: python 15_dwi_qc_python.py
"""
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

BASE = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
for _c in glob.glob(str(BASE / "**" / "code" / "config.py"), recursive=True):
    sys.path.insert(0, os.path.dirname(_c))
    break
import config as C  # noqa: E402

SHELLS = (0, 650, 1350, 2000)
B0_TOL = 100


def shell_of(b):
    return min(SHELLS, key=lambda s: abs(b - s))


def qc_subject(nii, bval, bvec, bad_z=5.0):
    data = np.asarray(nib.load(str(nii)).dataobj, dtype=np.float32)
    bvals = np.loadtxt(bval)
    bvecs = np.loadtxt(bvec)
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if data.ndim != 4 or data.shape[3] != len(bvals):
        return None
    b0 = data[..., bvals < B0_TOL].mean(3)
    thr = np.percentile(b0[b0 > 0], 40) if (b0 > 0).any() else 0
    mask = b0 > thr
    rec = {}

    # per-shell SNR
    for s in SHELLS:
        sel = np.array([abs(b - s) <= B0_TOL and shell_of(b) == s for b in bvals])
        if sel.sum() < 2:
            continue
        v = data[..., sel]
        m, sd = v.mean(3), v.std(3, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(sd > 0, m / sd, 0.0)
        rec[f"snr_b{s}"] = float(np.median(snr[mask]))

    # NDC-style nearest-direction neighbor correlation (non-b0 volumes)
    dwi_idx = [i for i in range(len(bvals)) if bvals[i] >= B0_TOL]
    cors = []
    for i in dwi_idx:
        best, bj = -2.0, None
        for j in dwi_idx:
            if j == i or shell_of(bvals[j]) != shell_of(bvals[i]):
                continue
            dot = abs(float(np.dot(bvecs[i], bvecs[j])))
            if dot > best:
                best, bj = dot, j
        if bj is not None:
            cors.append(np.corrcoef(data[..., i][mask], data[..., bj][mask])[0, 1])
    rec["ndc"] = float(np.mean(cors)) if cors else np.nan

    # bad-slice / dropout count (per shell): a slice counts as bad only if its
    # in-mask mean is BOTH a strong low outlier (robust z < -bad_z) AND a real
    # signal drop (< 0.7x the slice's median), on slices with enough brain
    # coverage (>=200 in-mask voxels) so edge slices are not flagged.
    bad = 0
    nz = data.shape[2]
    cover = np.array([int(mask[:, :, z].sum()) for z in range(nz)])
    valid_z = cover >= 200
    for s in SHELLS:
        sel = np.where([shell_of(b) == s and b >= B0_TOL for b in bvals])[0]
        if len(sel) < 3:
            continue
        sm = np.array([[data[:, :, z, v][mask[:, :, z]].mean()
                        if valid_z[z] else np.nan
                        for z in range(nz)] for v in sel])
        med = np.nanmedian(sm, axis=0)
        mad = np.nanmedian(np.abs(sm - med), axis=0) + 1e-6
        zsc = (sm - med) / (1.4826 * mad)
        drop = sm < 0.7 * med
        bad += int(np.nansum((zsc < -bad_z) & drop))
    rec["n_bad_slices"] = bad
    return rec


def main():
    out = C.outdir("dwi")
    rows = []
    for sub, sdir in C.subject_dirs():
        d = sdir / "dwi"
        nii = d / f"{sub}_dwi.nii.gz"
        bval = d / f"{sub}_dwi.bval"
        bvec = d / f"{sub}_dwi.bvec"
        if not (nii.exists() and bval.exists() and bvec.exists()):
            continue
        rec = qc_subject(nii, bval, bvec)
        if rec:
            rec["participant_id"] = sub
            rows.append(rec)
    if not rows:
        print("No raw DWI found.")
        return
    df = pd.DataFrame(rows)
    df = df[["participant_id"] + [c for c in df.columns if c != "participant_id"]]
    df.to_csv(out / "dwi_qc_python.csv", index=False)
    print(f"DWI QC (pure Python, n={len(df)}):")
    for c in ["snr_b0", "snr_b650", "snr_b1350", "snr_b2000", "ndc"]:
        if c in df:
            s = C.describe(df[c])
            print(f"  {c:<10} mean={s['mean']:.3f} SD={s['sd']:.3f} "
                  f"range=[{s['min']:.3f},{s['max']:.3f}]")
    if "n_bad_slices" in df:
        print(f"  bad slices: total={int(df['n_bad_slices'].sum())}, "
              f"max/subject={int(df['n_bad_slices'].max())}, "
              f"subjects with >0: {int((df['n_bad_slices'] > 0).sum())}")
    print(f"written -> {out / 'dwi_qc_python.csv'}")


if __name__ == "__main__":
    main()
