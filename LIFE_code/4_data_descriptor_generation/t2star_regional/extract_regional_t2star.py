#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Mask-based regional T2* for the LIFE cohort.

Rationale
---------
The Figure 2 T2* panels sample four hand-placed squares on one axial slice.
That shows four numbers, three of them null, and says nothing about how
broadly T2* is stable across the brain in late life. This script replaces the
hand-placed boxes with anatomically defined FreeSurfer (aseg) masks, so the
age-stability claim is made over whole structures rather than over single
slices, and the exceptions (caudate, and any other iron-rich nucleus) stand
out against that background.

Method
------
For each participant with a six-echo MEGRE and a FreeSurfer reconstruction:

  1. SynthStrip the first echo to get a brain mask (same tool as Figure 2).
  2. Fit T2* voxelwise by ordinary least squares on log signal across the
     echo train, keeping R^2 of that fit.
  3. Rigidly register (6 dof, `mri_coreg`) the stripped first echo to the
     participant's own FreeSurfer brainmask.mgz.
  4. Pull aseg.mgz INTO the native MEGRE grid with the inverse of that
     transform, nearest-neighbor. The quantitative map is never resampled:
     interpolating T2* across a 3 mm slice gap would manufacture values.
  5. Erode each label IN PLANE by one voxel to limit partial-volume
     contamination. The erosion is in-plane only because the voxels are
     anisotropic (1 x 1 x 3 mm): a 3D erosion would strip a whole 3 mm slice
     off the top and bottom of every structure, which costs ~75% of the
     caudate and ~83% of the corpus callosum for no gain in tissue purity.
  6. Take the MEDIAN T2* over the surviving voxels of each label.

A structure is reported only when enough of it is actually inside the MEGRE
slab (`*_cov`, measured BEFORE erosion and QC, against MIN_COVERAGE) and
enough voxels survive erosion and fit QC (`*_nvox`, against MIN_VOX). The two
are kept apart on purpose: coverage is a statement about anatomy reaching the
slab, retention (`*_qc`) a statement about the fit. The slab is
40 x 3 mm, so inferior structures are not guaranteed to be covered; the
per-structure `*_cov` and `*_nvox` columns record what was available in each
participant, and the figure script drops structures that fail the cohort-level
coverage criterion rather than silently averaging over partial anatomy.

Outputs (under derivatives/technical_validation/t2star_regional/):
  regional_t2star_per_subject.csv  — median T2*, voxel count and coverage per
                                     structure, one row per participant
  regional_t2star_qc.txt           — per-subject log, including failures

Usage:
  python3 extract_regional_t2star.py              # whole cohort
  python3 extract_regional_t2star.py --subject sub-009
  python3 extract_regional_t2star.py --force      # ignore cached intermediates
  python3 extract_regional_t2star.py --jobs 8     # 8 participants at a time
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

# ── configuration ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402  (path must be set first)

BASE = C.BASE_DIR
# Where the imaging actually lives. On the authoring machine the released
# BIDS tree sits beside the working tree rather than under it, so this is
# overridable; everywhere else it is the BIDS directory config points to.
BIDS = Path(os.environ.get("LIFE_BIDS", C.BIDS_DIR))
SUBJECTS_DIR = Path(os.environ.get("LIFE_FS_SUBJECTS", C.FS_SUBJECTS_DIR))
# Outputs go beside every other Technical Validation derivative.
OUT = Path(os.environ.get(
    "LIFE_T2STAR_REGIONAL_OUT",
    C.DERIV_DIR / "technical_validation" / "t2star_regional"))
WORK = Path(os.environ.get("LIFE_T2STAR_REGIONAL_WORK", OUT / "work"))
FS_HOME = os.environ.get("LIFE_FS_HOME", "/Applications/freesurfer/7.4.1")
COREG_THREADS = int(os.environ.get("LIFE_COREG_THREADS", "2"))

AGE_MIN = 60            # released cohort is the older adults

# Fit QC, applied voxelwise before any label is summarized.
T2S_LO, T2S_HI = 5.0, 150.0   # ms; outside this a mono-exponential fit on a
                              # 2.68-65 ms echo train is not interpretable
MIN_R2 = 0.70                 # drop voxels the echo train does not explain
MIN_VOX = 20                  # label needs this many surviving voxels
MIN_COVERAGE = 0.50           # ...and this fraction of the label inside the slab

# aseg labels, grouped anatomically (NOT by expected effect).
# FreeSurferColorLUT.txt: 2/41 cerebral WM, 3/42 cerebral cortex,
# 251-255 CC subdivisions, 10/49 thalamus, 11/50 caudate, 12/51 putamen,
# 13/52 pallidum, 17/53 hippocampus, 18/54 amygdala, 26/58 accumbens,
# 28/60 ventral DC, 7/46 cerebellar WM, 8/47 cerebellar cortex, 16 brainstem.
STRUCTURES = [
    # name                    aseg ids            anatomical block
    ("cerebral_wm_l",         (2,),               "Cerebral white matter"),
    ("cerebral_wm_r",         (41,),              "Cerebral white matter"),
    ("corpus_callosum",       (251, 252, 253, 254, 255), "Cerebral white matter"),
    ("cerebral_cortex_l",     (3,),               "Cerebral cortex"),
    ("cerebral_cortex_r",     (42,),              "Cerebral cortex"),
    ("thalamus_l",            (10,),              "Deep gray matter"),
    ("thalamus_r",            (49,),              "Deep gray matter"),
    ("caudate_l",             (11,),              "Deep gray matter"),
    ("caudate_r",             (50,),              "Deep gray matter"),
    ("putamen_l",             (12,),              "Deep gray matter"),
    ("putamen_r",             (51,),              "Deep gray matter"),
    ("pallidum_l",            (13,),              "Deep gray matter"),
    ("pallidum_r",            (52,),              "Deep gray matter"),
    ("hippocampus_l",         (17,),              "Medial temporal"),
    ("hippocampus_r",         (53,),              "Medial temporal"),
    ("amygdala_l",            (18,),              "Medial temporal"),
    ("amygdala_r",            (54,),              "Medial temporal"),
    ("cerebellar_cortex_l",   (8,),               "Cerebellum"),
    ("cerebellar_cortex_r",   (47,),              "Cerebellum"),
    ("cerebellar_wm_l",       (7,),               "Cerebellum"),
    ("cerebellar_wm_r",       (46,),              "Cerebellum"),
    ("brainstem",             (16,),              "Brainstem"),
]

# In-plane 6-neighborhood (a cross in x/y, nothing through-plane): the
# erosion structuring element for the anisotropic 1 x 1 x 3 mm MEGRE grid.
INPLANE = np.zeros((3, 3, 3), dtype=bool)
INPLANE[1, :, 1] = True
INPLANE[:, 1, 1] = True

ENV = dict(os.environ, FREESURFER_HOME=FS_HOME,
           SUBJECTS_DIR=str(SUBJECTS_DIR),
           PATH=f"{FS_HOME}/bin:" + os.environ.get("PATH", ""))


# ── helpers ──────────────────────────────────────────────────────────
def sh(cmd, log):
    """Run a FreeSurfer command; return True on success and log failures."""
    r = subprocess.run([str(c) for c in cmd], env=ENV,
                       capture_output=True, text=True)
    if r.returncode:
        log.append(f"    FAILED: {Path(str(cmd[0])).name} -> "
                   f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no stderr'}")
        return False
    return True


def _echo_num(path, marker):
    """Echo index from a filename, tolerating a dcm2niix letter suffix.

    dcm2niix writes a trailing letter when a series needs disambiguating
    (e.g. "_e1a.nii.gz"), so a bare int() on the token raises ValueError.
    """
    tok = os.path.basename(path).split(marker)[-1].split(".")[0]
    m = re.match(r"(\d+)", tok)
    return int(m.group(1)) if m else 10 ** 6


def echo_paths(sid):
    """The participant's MEGRE echoes, in echo order, from the release."""
    import glob
    for pattern, marker in (
            (BIDS / sid / "anat" / f"{sid}_echo-*_MEGRE.nii.gz", "_echo-"),
            (BIDS / sid / "anat" / f"{sid}_ME-T2_echo-*.nii.gz", "echo-")):
        p = sorted(glob.glob(str(pattern)), key=lambda x: _echo_num(x, marker))
        if p:
            return p
    return []


def echo_times(paths):
    """Echo times in ms, read from the BIDS sidecars."""
    te = []
    for p in paths:
        j = Path(str(p).replace(".nii.gz", ".json"))
        if not j.exists():
            return None
        t = json.load(open(j)).get("EchoTime")
        if t is None:
            return None
        te.append(float(t) * 1000.0)        # BIDS stores seconds
    return np.asarray(te)


def fit_t2star(paths, te, brain):
    """Voxelwise mono-exponential T2* (ms) and the fit's R^2.

    Ordinary least squares on log signal: log S = log S0 - TE / T2*.
    """
    S = np.stack([np.asarray(nib.load(p).dataobj, dtype=np.float64)
                  for p in paths], axis=-1)
    ok = brain & (S > 0).all(axis=-1)
    y = np.log(S[ok])                                  # (nvox, nTE)
    X = np.column_stack([np.ones_like(te), te])        # (nTE, 2)
    beta, *_ = np.linalg.lstsq(X, y.T, rcond=None)     # (2, nvox)
    pred = (X @ beta).T
    ss_res = ((y - pred) ** 2).sum(axis=1)
    ss_tot = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
        t2s = -1.0 / beta[1]
    vol = np.full(brain.shape, np.nan)
    r2v = np.full(brain.shape, np.nan)
    vol[ok] = t2s
    r2v[ok] = r2
    return vol, r2v


def subject_rows(sid, force, log):
    """Median T2* per structure for one participant, or None if unusable."""
    paths = echo_paths(sid)
    if len(paths) < 4:
        log.append(f"    skipped: {len(paths)} echoes (need >= 4)")
        return None
    te = echo_times(paths)
    if te is None or not np.all(np.diff(te) > 0):
        log.append("    skipped: unusable EchoTime metadata")
        return None

    e1 = paths[0]
    bmask = SUBJECTS_DIR / sid / "mri" / "brainmask.mgz"
    aseg = SUBJECTS_DIR / sid / "mri" / "aseg.mgz"
    if not (bmask.exists() and aseg.exists()):
        log.append("    skipped: no FreeSurfer brainmask/aseg")
        return None

    WORK.mkdir(parents=True, exist_ok=True)
    mask_f = WORK / f"{sid}_e1_brainmask.nii.gz"
    strip_f = WORK / f"{sid}_e1_brain.nii.gz"
    lta_f = WORK / f"{sid}_e1_to_fs.lta"
    aseg_f = WORK / f"{sid}_aseg_in_megre.nii.gz"

    # 1. brain mask on the first echo
    if force or not mask_f.exists():
        if not sh([f"{FS_HOME}/bin/mri_synthstrip", "-i", e1,
                   "-m", mask_f, "-o", strip_f], log):
            return None
    # 2. rigid registration of that echo to the participant's own T1 space
    if force or not lta_f.exists():
        if not sh([f"{FS_HOME}/bin/mri_coreg", "--mov", strip_f, "--ref", bmask,
                   "--reg", lta_f, "--dof", "6",
                   "--threads", str(COREG_THREADS)], log):
            return None
    # 3. aseg into the native MEGRE grid (nearest neighbor; the T2* map,
    #    which is the quantitative one, is left on its own grid)
    if force or not aseg_f.exists():
        if not sh([f"{FS_HOME}/bin/mri_vol2vol", "--mov", aseg, "--targ", e1,
                   "--lta-inv", lta_f, "--nearest", "--o", aseg_f], log):
            return None

    brain = np.asarray(nib.load(mask_f).dataobj) > 0
    t2s, r2 = fit_t2star(paths, te, brain)
    lab = np.asarray(nib.load(aseg_f).dataobj).astype(np.int32)
    lab_fs = np.asarray(nib.load(aseg).dataobj).astype(np.int32)

    # Voxel-volume ratio, so "coverage" compares like with like: the aseg in
    # FreeSurfer's conformed space is 1 mm^3, the MEGRE voxel is larger.
    vox_megre = float(np.prod(nib.load(e1).header.get_zooms()[:3]))
    vox_fs = float(np.prod(nib.load(aseg).header.get_zooms()[:3]))

    good = np.isfinite(t2s) & (t2s > T2S_LO) & (t2s < T2S_HI) & (r2 >= MIN_R2)
    row = {"participant_id": sid}
    for name, ids, _blk in STRUCTURES:
        m = np.isin(lab, ids) & brain
        # Coverage: how much of the structure reaches the slab at all. Measured
        # before erosion and before fit QC, so it answers "is this anatomy in
        # the field of view", not "did the fit work".
        n_fs = int(sum((lab_fs == i).sum() for i in ids))
        cov = (int(m.sum()) * vox_megre) / (n_fs * vox_fs) if n_fs else np.nan
        # In-plane erosion (see module docstring).
        me = ndimage.binary_erosion(m, structure=INPLANE) if m.any() else m
        use = me if me.sum() >= MIN_VOX else m      # tiny labels: no erosion
        sel = use & good
        n = int(sel.sum())
        row[f"{name}"] = float(np.median(t2s[sel])) if (
            n >= MIN_VOX and cov >= MIN_COVERAGE) else np.nan
        row[f"{name}_nvox"] = n
        row[f"{name}_cov"] = round(float(cov), 4) if n_fs else np.nan
        row[f"{name}_qc"] = round(n / int(use.sum()), 4) if use.sum() else np.nan
        row[f"{name}_eroded"] = bool(me.sum() >= MIN_VOX)
    row["n_echoes"] = len(paths)
    row["te_min_ms"], row["te_max_ms"] = round(te[0], 2), round(te[-1], 2)
    row["brain_vox"] = int(brain.sum())
    row["status"] = "ok"
    kept = sum(1 for n, _, _ in STRUCTURES if np.isfinite(row[n]))
    log.append(f"    ok: {kept}/{len(STRUCTURES)} structures reported")
    return row


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", help="run one participant only")
    ap.add_argument("--force", action="store_true",
                    help="recompute cached masks/registrations")
    ap.add_argument("--jobs", type=int, default=1,
                    help="participants to process concurrently (each uses "
                         "LIFE_COREG_THREADS registration threads)")
    args = ap.parse_args()

    part = pd.read_csv(C.PARTICIPANTS_TSV if C.PARTICIPANTS_TSV.exists()
                       else BIDS / "participants.tsv", sep="\t")
    part["participant_id"] = part["participant_id"].astype(str)
    age = pd.to_numeric(part.get("age"), errors="coerce")
    cohort = part.loc[age >= AGE_MIN, "participant_id"].tolist()
    ages = dict(zip(part["participant_id"], age))

    sids = [args.subject] if args.subject else cohort
    OUT.mkdir(parents=True, exist_ok=True)

    def one(idx_sid):
        """Process a participant; collect its own log lines so that running
        several at once does not interleave them."""
        i, sid = idx_sid
        mylog = [f"[{i}/{len(sids)}] {sid}"]
        try:
            r = subject_rows(sid, args.force, mylog)
        except Exception as e:                       # keep the cohort running
            mylog.append(f"    ERROR: {type(e).__name__}: {e}")
            r = None
        if r is None:
            r = {"participant_id": sid, "status": "failed"}
        else:
            r["age"] = ages.get(sid, np.nan)
        print("\n".join(mylog), flush=True)
        return r, mylog

    jobs = max(1, args.jobs)
    if jobs == 1:
        out = [one(x) for x in enumerate(sids, 1)]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            out = list(ex.map(one, enumerate(sids, 1)))
    rows = [r for r, _ in out]
    log = [line for _, ml in out for line in ml]

    df = pd.DataFrame(rows)
    front = ["participant_id", "age", "status", "n_echoes",
             "te_min_ms", "te_max_ms", "brain_vox"]
    cols = [c for c in front if c in df.columns] + \
           [c for c in df.columns if c not in front]
    df = df[cols]
    csv = OUT / "regional_t2star_per_subject.csv"
    df.to_csv(csv, index=False)

    ok = (df.get("status") == "ok").sum()
    hdr = [f"Mask-based regional T2*  ({ok}/{len(df)} participants usable)",
           f"  BIDS:          {BIDS}",
           f"  FreeSurfer:    {SUBJECTS_DIR}",
           f"  fit QC:        {T2S_LO}-{T2S_HI} ms, R^2 >= {MIN_R2}",
           f"  label QC:      >= {MIN_VOX} voxels and >= {MIN_COVERAGE:.0%} coverage",
           ""]
    (OUT / "regional_t2star_qc.txt").write_text("\n".join(hdr + log) + "\n")
    print(f"\nwrote {csv}  ({ok}/{len(df)} usable)")
    if ok:
        print("cohort coverage by structure (median across participants):")
        for name, _ids, _blk in STRUCTURES:
            c = pd.to_numeric(df.get(f"{name}_cov"), errors="coerce")
            q = pd.to_numeric(df.get(f"{name}_qc"), errors="coerce")
            v = pd.to_numeric(df.get(name), errors="coerce")
            print(f"  {name:22s} cov={c.median():5.2f}  qc={q.median():5.2f}  "
                  f"reported n={v.notna().sum():2d}")


if __name__ == "__main__":
    main()
