#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Cohort framewise-displacement (FD) timeseries for the fMRI motion figure.

For every older-cohort participant (age >= 60) with a resting-state run, obtain
the 4-D BOLD (local if present, else convert the 'rest' series from Box), run
FSL mcflirt, and compute the standard Power et al. (2012) frame-to-frame FD
(|dx|+|dy|+|dz| + 50*(|d_pitch|+|d_roll|+|d_yaw|), rotations in rad -> mm at a
50 mm radius). FD[0] is set to 0.

Writes one file per subject to derivatives/technical_validation/rsfmri/
fd_timeseries/<sub>_fd.txt (resumable). Run in the background.
"""
import sys, os, glob, re, shutil, subprocess
import os
from pathlib import Path
from collections import defaultdict
import numpy as np, nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

BOXROOT = Path(os.path.expanduser(
    "~/Library/CloudStorage/Box-Box/LIFE_Brain_protocol/LIFE DATA"))
PROBE = "/Applications/freesurfer/7.4.1/bin/mri_probedicom"
DCM2NIIX = "/opt/homebrew/bin/dcm2niix"
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
MCFLIRT = f"{FSLDIR}/bin/mcflirt"
# Working directory for intermediate volumes (mcflirt output, any DICOM
# conversions). Defaults to a folder beside the derivatives so the script runs
# on any machine; override with LIFE_FD_SCRATCH.
SCRATCH = Path(os.environ.get(
    "LIFE_FD_SCRATCH",
    C.DERIV_DIR / "technical_validation" / "rsfmri" / "work"))
BOLDDIR = SCRATCH / "bold"; BOLDDIR.mkdir(parents=True, exist_ok=True)
GAPFILL = SCRATCH / "gapfill" / "nifti"
OUT = C.DERIV_DIR / "technical_validation" / "rsfmri" / "fd_timeseries"
OUT.mkdir(parents=True, exist_ok=True)
FSLENV = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ",
              PATH=f"{FSLDIR}/bin:" + os.environ.get("PATH", ""))


def rest_bold(sid):
    for p in [GAPFILL / f"{sid}_task-rest_bold.nii.gz", BOLDDIR / f"{sid}_rest.nii.gz",
              C.BASE_DIR / "bids_dataset_LIFE" / sid / "func" / f"{sid}_task-rest_bold.nii.gz"]:
        if p.exists():
            return p
    life = sid.replace("sub-", "LIFE_")
    dcms = glob.glob(str(BOXROOT / life / "MRI" / "**" / "*.dcm"), recursive=True)
    if not dcms:
        return None
    ser = defaultdict(list)
    for f in dcms:
        m = re.search(r"\.MR\._\.(\d+)\.\d+\.", os.path.basename(f))
        if m:
            ser[int(m.group(1))].append(f)
    cand = None
    for snum, files in sorted(ser.items(), key=lambda kv: -len(kv[1])):
        if len(files) < 100:
            continue
        d = subprocess.run([PROBE, "--i", files[0], "--t", "0008", "103e"],
                           capture_output=True, text=True).stdout.strip().lower()
        if d == "rest" or (("rest" in d) and "flanker" not in d and "dccs" not in d):
            cand = (snum, files); break
    if cand is None:
        return None
    raw = BOLDDIR / f"{sid}_rest_dcm"; raw.mkdir(exist_ok=True)
    for f in cand[1]:
        shutil.copy(f, raw)
    subprocess.run([DCM2NIIX, "-z", "y", "-o", str(BOLDDIR), "-f", f"{sid}_rest", str(raw)],
                   capture_output=True, text=True)
    shutil.rmtree(raw, ignore_errors=True)
    out = list(BOLDDIR.glob(f"{sid}_rest*.nii.gz"))
    return out[0] if out else None


def power_fd(par):
    d = np.diff(par, axis=0)
    fd = np.sum(np.abs(d[:, 3:]), 1) + 50.0 * np.sum(np.abs(d[:, :3]), 1)
    return np.concatenate([[0.0], fd])


def process(sid):
    outp = OUT / f"{sid}_fd.txt"
    if outp.exists():
        return "cached"
    bold = rest_bold(sid)
    if bold is None:
        return "no rest BOLD"
    img = nib.load(str(bold))
    if img.ndim < 4 or img.shape[3] < 20:
        return f"not 4D/too short ({img.shape})"
    base = BOLDDIR / f"{sid}_mcf"
    r = subprocess.run([MCFLIRT, "-in", str(bold), "-out", str(base), "-plots", "-refvol", "4"],
                       env=FSLENV, capture_output=True, text=True)
    par = base.with_suffix(".par")
    if not par.exists():
        return f"mcflirt failed: {r.stderr[-160:]}"
    fd = power_fd(np.loadtxt(par))
    np.savetxt(outp, fd, fmt="%.5f")
    # clean big intermediates
    for x in BOLDDIR.glob(f"{sid}_mcf*"):
        x.unlink()
    return f"ok ({len(fd)} vols, meanFD={fd.mean():.3f})"


def main():
    ph = C.load_phenotype()
    older = [i for i in ph.index if ph.loc[i, "age"] >= 60]
    for k, sid in enumerate(older):
        try:
            msg = process(sid)
        except Exception as e:  # noqa: BLE001
            msg = f"error: {e}"
        print(f"[{k+1}/{len(older)}] {sid}: {msg}", flush=True)
    done = len(list(OUT.glob("*_fd.txt")))
    print(f"COHORT_FD_DONE: {done} subjects with FD timeseries -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
