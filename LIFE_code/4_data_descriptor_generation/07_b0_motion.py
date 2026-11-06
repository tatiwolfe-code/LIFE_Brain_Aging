#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Head motion from the interspersed b0 volumes (raw-data QC).
===========================================================
The LIFE DWI acquisition interleaves 7 b = 0 volumes through the 97-volume
run. Rigid-body registering those b0 images to each other with FSL mcflirt
recovers head displacement across the scan directly from the *raw* data, with
no eddy/topup pass and no dependence on any downstream reconstruction.

This is an undersampled estimate: it samples motion at 7 time points rather
than every volume, so it captures drift and large events but will miss motion
that occurs and reverses within a b0 interval. It is a lower bound on total
motion, and must be described as inter-b0 displacement rather than as
volume-wise framewise displacement (Power et al., 2012).

Outputs -> derivatives/technical_validation/dwi/b0_motion.csv
    mean_rel_disp_mm  mean RMS displacement between successive b0 volumes
    max_rel_disp_mm   largest single between-b0 displacement
    mean_abs_disp_mm  mean RMS displacement from the first b0
    max_abs_disp_mm   largest displacement from the first b0
    fd_b0_mm          Power-style displacement on the b0 series (50 mm radius)
    max_trans_mm, max_rot_deg   peak translation / rotation from reference
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

RAW_DIR = Path(os.environ.get("LIFE_DWI_NIFTI", C.BASE_DIR / "LIFE_dwi_nifti"))
FSLDIR = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))
RADIUS = 50.0  # mm, standard head radius for rotation -> displacement


def dwi_sources():
    """{subject: (nifti, bval)} for every participant with a DWI run.

    The released layout is sub-<ID>/dwi/sub-<ID>_dwi.nii.gz, so that is what
    is read. A flat directory of sub-<ID>_dwi.nii.gz is still accepted (set
    LIFE_DWI_NIFTI) because the volumes were staged that way on the authoring
    machine -- but the BIDS tree is preferred, so the script runs against the
    dataset as published. The staging directory was missing one released
    participant and carried four who are not in the release, which is why the
    published SNR and motion statistics were n = 56 rather than 57.
    """
    out = {}
    for n in sorted(C.BIDS_DIR.glob("sub-*/dwi/sub-*_dwi.nii.gz")):
        sub = n.name.split("_")[0]
        b = n.with_name(f"{sub}_dwi.bval")
        if b.exists():
            out[sub] = (n, b)
    if not out and RAW_DIR.is_dir():
        for n in sorted(RAW_DIR.glob("sub-*_dwi.nii.gz")):
            sub = n.name.split("_")[0]
            b = RAW_DIR / f"{sub}_dwi.bval"
            if b.exists():
                out[sub] = (n, b)
    return out


def _env():
    e = dict(os.environ)
    e["FSLDIR"] = str(FSLDIR)
    e["PATH"] = f"{FSLDIR / 'bin'}:{e.get('PATH', '')}"
    e["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    return e


def motion_one(args) -> dict | None:
    sub, dwi, bval = args
    if not (dwi.exists() and bval.exists()):
        return None
    b = np.loadtxt(bval)
    sel = b < 100
    if sel.sum() < 3:
        return None
    tmp = tempfile.mkdtemp(prefix=f"b0mot_{sub}_")
    try:
        img = nib.load(str(dwi))
        idx = np.flatnonzero(sel)
        b0 = np.stack([np.asarray(img.dataobj[..., int(i)], dtype=np.float32)
                       for i in idx], axis=-1)
        b0f = os.path.join(tmp, "b0s.nii.gz")
        nib.save(nib.Nifti1Image(b0, img.affine, img.header), b0f)
        r = subprocess.run(
            [str(FSLDIR / "bin" / "mcflirt"), "-in", b0f, "-out",
             os.path.join(tmp, "b0s_mc"), "-refvol", "0",
             "-plots", "-rmsabs", "-rmsrel"],
            capture_output=True, text=True, env=_env())
        par = os.path.join(tmp, "b0s_mc.par")
        if not os.path.exists(par):
            print(f"  ! {sub}: mcflirt produced no .par ({r.stderr.strip()[:80]})")
            return None
        p = np.loadtxt(par)                       # rot xyz (rad), trans xyz (mm)
        rot, tr = p[:, :3], p[:, 3:6]
        d = np.abs(np.diff(tr, axis=0)).sum(axis=1) + \
            RADIUS * np.abs(np.diff(rot, axis=0)).sum(axis=1)
        rec = {"participant_id": sub, "n_b0": int(sel.sum()),
               "fd_b0_mm": float(d.mean()),
               "max_trans_mm": float(np.abs(tr).max()),
               "max_rot_deg": float(np.degrees(np.abs(rot).max()))}
        for key, f in [("abs", "b0s_mc_abs.rms"), ("rel", "b0s_mc_rel.rms")]:
            fp = os.path.join(tmp, f)
            if os.path.exists(fp):
                v = np.atleast_1d(np.loadtxt(fp))
                rec[f"mean_{key}_disp_mm"] = float(v.mean())
                rec[f"max_{key}_disp_mm"] = float(v.max())
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    src = dwi_sources()
    items = [(s_, n, bv) for s_, (n, bv) in sorted(src.items())]
    root = items[0][1].parent.parent.parent if items else RAW_DIR
    print(f"b0 motion estimation over {len(items)} subjects  (source: {root})")
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", 6))) as ex:
        for rec in ex.map(motion_one, items):
            if rec:
                rows.append(rec)
                print(f"  {rec['participant_id']}  FD={rec['fd_b0_mm']:.3f} mm  "
                      f"max abs={rec.get('max_abs_disp_mm', float('nan')):.3f} mm")
    df = pd.DataFrame(rows)
    out = C.outdir("dwi") / "b0_motion.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  (n={len(df)})")
    if len(df):
        for c in ["fd_b0_mm", "mean_rel_disp_mm", "mean_abs_disp_mm", "max_abs_disp_mm"]:
            if c in df:
                print(f"  {c:18} M = {df[c].mean():.3f}, SD = {df[c].std(ddof=1):.3f}, "
                      f"max = {df[c].max():.3f}")


if __name__ == "__main__":
    main()
