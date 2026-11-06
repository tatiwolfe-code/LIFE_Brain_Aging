#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Cohort temporal-SNR (tSNR) in gray and white matter for the fMRI tSNR figure.

For every older-cohort participant (age >= 60) with a resting-state run:
  1. motion-correct the raw 4-D BOLD (FSL mcflirt, refvol 4);
  2. brain-extract the mean EPI and the T1w (FreeSurfer SynthStrip);
  3. segment the T1w into CSF/GM/WM partial-volume maps (FSL FAST);
  4. register mean EPI -> T1w (FSL FLIRT, boundary-based cost) and
     T1w -> MNI152 2 mm (FLIRT, 12 dof);
  5. build tissue masks in EPI space --
       GM  = pve_GM > 0.5, restricted to cerebrum (MNI brain minus
             cerebellum and brainstem);
       WM  = pve_WM > 0.7, restricted to the union of the MNI structural
             lobes/insula/basal ganglia, with subcortical gray nuclei
             removed and one voxel eroded -> deep cerebral WM;
  6. compute voxelwise tSNR (mean / SD of the residuals after removing a
     quadratic trend) for the whole run, and a sliding-window tSNR
     timeseries (WINDOW volumes, STEP volumes) summarized as the median
     across mask voxels;
  7. resample the whole-run tSNR map into MNI space for the group average.

FreeSurfer recon-all segmentations are not available for this cohort on the
analysis laptop and mri_synthseg cannot run (the bundled TensorFlow requires
AVX, absent on Apple Silicon), so tissue classes come from FSL FAST on the
T1w. Validated against the real FreeSurfer aparc+aseg in sub-007:
median GM tSNR 47.9 (FreeSurfer 45.7) and median WM tSNR 67.1
(FreeSurfer 65.9); the WM mask is 98.5% FreeSurfer cerebral-WM voxels.

Outputs -> derivatives/technical_validation/rsfmri/
  tsnr_tissue_per_subject.csv          median/IQR/SD/mean per tissue
  tsnr_windows/<sub>_tsnr-{gm,wm}.txt  sliding-window timeseries
  tsnr_mni/<sub>_tsnr_mni.nii.gz       whole-run tSNR map in MNI 2 mm
Resumable: subjects with all outputs present are skipped. Run in background.
"""
import os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# ── external tools ───────────────────────────────────────────────────
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
FS_HOME = os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1")
# One worker per subject, so each tool is capped to a few threads: SynthStrip
# (PyTorch) and the BLAS calls otherwise each grab every core and thrash.
THREADS = os.environ.get("LIFE_TSNR_THREADS", "3")
FSLENV = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ",
              FREESURFER_HOME=FS_HOME, SUBJECTS_DIR=str(C.FS_SUBJECTS_DIR),
              OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
              VECLIB_MAXIMUM_THREADS=THREADS, NUMEXPR_NUM_THREADS=THREADS,
              PATH=f"{FSLDIR}/bin:{FS_HOME}/bin:" + os.environ.get("PATH", ""))
MNI_BRAIN = f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain"

# ── paths ────────────────────────────────────────────────────────────
# Working directories for intermediates. Defaults sit beside the derivatives
# so the script runs on any machine; override with the env vars named.
SCRATCH = Path(os.environ.get(
    "LIFE_TSNR_SCRATCH",
    C.DERIV_DIR / "technical_validation" / "rsfmri" / "work" / "tsnr"))
BOLDDIR = Path(os.environ.get(
    "LIFE_BOLD_SCRATCH",
    C.DERIV_DIR / "technical_validation" / "rsfmri" / "work" / "bold"))
# sub-050 and sub-058 were converted during the structural gap-fill, not by
# cohort_fd.py, so their rest BOLD may live in a separate folder.
GAPFILL = Path(os.environ.get(
    "LIFE_GAPFILL_DIR",
    C.DERIV_DIR / "technical_validation" / "rsfmri" / "work" / "gapfill"))
T1DIR = SCRATCH / "t1"
MNIDIR = SCRATCH / "mni"          # cerebrum_gm, cerebrum_wm, subcort (mk_mni_masks.sh)
WORK = SCRATCH / "work"

OUT = C.DERIV_DIR / "technical_validation" / "rsfmri"
WINOUT = OUT / "tsnr_windows"
SPECOUT = OUT / "eigenspectra"
MNIOUT = OUT / "tsnr_mni"
for d in (WORK, WINOUT, MNIOUT, SPECOUT):
    d.mkdir(parents=True, exist_ok=True)

# ── analysis parameters ──────────────────────────────────────────────
SCHEMA = 3         # bump to force recomputation of cached rows
WINDOW = 60        # volumes per sliding window (48 s at TR = 0.8 s).
                   # A 24 s window cannot contain fluctuations below
                   # ~0.04 Hz, which is most of the resting-state band,
                   # and it hides the gray/white tSNR difference
                   # (WM-GM: -1.6 at 24 s, +0.8 at 48 s, +3.3 at 96 s,
                   # against +4.1 for the whole run).
STEP = 5           # volumes between window centers
DETREND_ORDER = 2  # quadratic drift removed before the SD
GM_PVE_THR = 0.5
WM_PVE_THR = 0.7


def rest_bold(sid):
    """Locate this participant's resting-state BOLD in either scratch source."""
    for c in (BOLDDIR / f"{sid}_rest.nii.gz",
              GAPFILL / f"{sid}_task-rest_bold.nii.gz"):
        if c.exists():
            return c
    return None


def all_rest_subjects():
    """Every subject label with a resting-state BOLD in either source."""
    a = {p.name.split("_")[0] for p in BOLDDIR.glob("sub-*_rest.nii.gz")}
    b = {p.name.split("_")[0] for p in GAPFILL.glob("sub-*_task-rest_bold.nii.gz")}
    return sorted(a | b)


def sh(cmd, **kw):
    return subprocess.run(cmd, env=FSLENV, capture_output=True, text=True, **kw)


def detrended_tsnr(Y):
    """tSNR per row of a (voxels x time) array: mean / SD of the residuals
    after removing a polynomial trend of order DETREND_ORDER."""
    n = Y.shape[1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    X = np.column_stack([t ** k for k in range(DETREND_ORDER + 1)]).astype(np.float32)
    mu = Y.mean(1)
    beta, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    resid = Y.T - X @ beta
    sd = resid.std(0, ddof=X.shape[1])
    return np.where(sd > 0, mu / sd, 0.0).astype(np.float32)


def summarize(v):
    q1, q3 = np.percentile(v, [25, 75])
    return {"median": float(np.median(v)), "iqr": float(q3 - q1),
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def pca_spectrum(Y):
    """Eigenvalues of the temporal covariance of a (voxels x time) array after
    quadratic detrending and per-voxel variance normalization (the same
    normalization MELODIC applies before its PCA)."""
    n = Y.shape[1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    X = np.column_stack([t ** k for k in range(DETREND_ORDER + 1)]).astype(np.float32)
    beta, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
    R = (Y.T - X @ beta).T                       # voxels x time, detrended
    sd = R.std(1, ddof=DETREND_ORDER + 1)
    keep = sd > 0
    R = R[keep] / sd[keep, None]                 # unit-variance voxels
    cov = (R.T @ R) / R.shape[0]                 # time x time
    lam = np.linalg.eigvalsh(cov.astype(np.float64))[::-1]
    return np.clip(lam, 1e-12, None), int(R.shape[0])


def minka_mdl(lam, n_samples):
    """Minka (2000) BIC/MDL model order for probabilistic PCA. `lam` are the
    eigenvalues (descending) of a p x p covariance from n_samples observations.
    Voxels are spatially correlated, so n_samples is optimistic and this
    estimate runs high; reported alongside the participation ratio."""
    p = len(lam)
    best_k, best_score = 1, -np.inf
    for k in range(1, p - 1):
        tail = lam[k:]
        sigma2 = tail.mean()
        if sigma2 <= 0:
            break
        ll = -0.5 * n_samples * (np.log(lam[:k]).sum() + (p - k) * np.log(sigma2)
                                 + p * (1.0 + np.log(2 * np.pi)))
        m_k = p * k - k * (k + 1) / 2.0 + 1 + k
        score = ll - 0.5 * m_k * np.log(n_samples)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def dimensionality(lam, n_samples):
    """Effective dimensionality of a covariance eigenspectrum."""
    tot = lam.sum()
    cum = np.cumsum(lam) / tot
    return {"pr": float(tot ** 2 / (lam ** 2).sum()),      # participation ratio
            "mdl": int(minka_mdl(lam, n_samples)),
            "n50": int(np.searchsorted(cum, 0.50) + 1),
            "n90": int(np.searchsorted(cum, 0.90) + 1)}


def preprocess(sid, w):
    """Motion-correct, segment, register. Returns paths dict or raises."""
    bold = rest_bold(sid)
    t1 = T1DIR / f"{sid}_T1w.nii.gz"
    if bold is None:
        raise FileNotFoundError("no rest BOLD")
    if not t1.exists():
        raise FileNotFoundError("no T1w")

    mcf = w / f"{sid}_mcf.nii.gz"
    if not mcf.exists():
        r = sh(["mcflirt", "-in", str(bold), "-out", str(w / f"{sid}_mcf"),
                "-refvol", "4", "-plots"])
        if not mcf.exists():
            raise RuntimeError(f"mcflirt failed: {r.stderr[-200:]}")
    mean = w / f"{sid}_meanepi.nii.gz"
    if not mean.exists():
        sh(["fslmaths", str(mcf), "-Tmean", str(mean)])
    epibrain = w / f"{sid}_meanepi_brain.nii.gz"
    epimask = w / f"{sid}_meanepi_brain_mask.nii.gz"
    if not epimask.exists():
        r = sh(["mri_synthstrip", "-i", str(mean), "-o", str(epibrain), "-m", str(epimask)])
        if not epimask.exists():
            raise RuntimeError(f"SynthStrip (EPI) failed: {r.stderr[-200:]}")
    t1brain = w / f"{sid}_T1_brain.nii.gz"
    if not t1brain.exists():
        r = sh(["mri_synthstrip", "-i", str(t1), "-o", str(t1brain)])
        if not t1brain.exists():
            raise RuntimeError(f"SynthStrip (T1) failed: {r.stderr[-200:]}")
    pve2 = w / f"{sid}_fast_pve_2.nii.gz"
    if not pve2.exists():
        sh(["fast", "-t", "1", "-n", "3", "-o", str(w / f"{sid}_fast"), str(t1brain)])
        if not pve2.exists():
            raise RuntimeError("FAST produced no pve maps")

    # EPI -> T1: 6-dof initialization, then boundary-based refinement
    e2t = w / f"{sid}_epi2t1.mat"
    if not e2t.exists():
        wmseg = w / f"{sid}_wmseg.nii.gz"
        sh(["fslmaths", str(pve2), "-thr", "0.5", "-bin", str(wmseg)])
        sh(["flirt", "-in", str(epibrain), "-ref", str(t1brain), "-dof", "6",
            "-omat", str(w / f"{sid}_init.mat")])
        sh(["flirt", "-in", str(epibrain), "-ref", str(t1), "-dof", "6",
            "-cost", "bbr", "-wmseg", str(wmseg), "-init", str(w / f"{sid}_init.mat"),
            "-schedule", f"{FSLDIR}/etc/flirtsch/bbr.sch", "-omat", str(e2t)])
        if not e2t.exists():
            raise RuntimeError("BBR registration failed")
    t2e = w / f"{sid}_t12epi.mat"
    if not t2e.exists():
        sh(["convert_xfm", "-omat", str(t2e), "-inverse", str(e2t)])

    # T1 -> MNI (12 dof) and the MNI -> EPI composition
    t2m = w / f"{sid}_t12mni.mat"
    if not t2m.exists():
        sh(["flirt", "-in", str(t1brain), "-ref", MNI_BRAIN, "-dof", "12",
            "-omat", str(t2m)])
        if not t2m.exists():
            raise RuntimeError("T1->MNI registration failed")
    m2t = w / f"{sid}_mni2t1.mat"
    m2e = w / f"{sid}_mni2epi.mat"
    e2m = w / f"{sid}_epi2mni.mat"
    if not m2e.exists():
        sh(["convert_xfm", "-omat", str(m2t), "-inverse", str(t2m)])
        sh(["convert_xfm", "-omat", str(m2e), "-concat", str(t2e), str(m2t)])
    if not e2m.exists():
        sh(["convert_xfm", "-omat", str(e2m), "-concat", str(t2m), str(e2t)])

    # tissue and atlas masks resampled into EPI space
    for src, name in [(w / f"{sid}_fast_pve_1", "pve1"), (w / f"{sid}_fast_pve_2", "pve2")]:
        dst = w / f"{sid}_{name}_epi.nii.gz"
        if not dst.exists():
            sh(["flirt", "-in", str(src), "-ref", str(epibrain), "-applyxfm",
                "-init", str(t2e), "-interp", "trilinear", "-out", str(dst)])
    for m in ("cerebrum_gm", "cerebrum_wm", "subcort"):
        dst = w / f"{sid}_{m}_epi.nii.gz"
        if not dst.exists():
            sh(["flirt", "-in", str(MNIDIR / m), "-ref", str(epibrain), "-applyxfm",
                "-init", str(m2e), "-interp", "trilinear", "-out", str(dst)])
    return {"mcf": mcf, "epibrain": epibrain, "epimask": epimask, "epi2mni": e2m}


def process(sid):
    w = WORK / sid
    w.mkdir(parents=True, exist_ok=True)
    win_gm = WINOUT / f"{sid}_tsnr-gm.txt"
    win_wm = WINOUT / f"{sid}_tsnr-wm.txt"
    mni_map = MNIOUT / f"{sid}_tsnr_mni.nii.gz"
    csv_row = w / f"{sid}_summary.npy"
    if win_gm.exists() and win_wm.exists() and mni_map.exists() and csv_row.exists():
        cached = np.load(csv_row, allow_pickle=True).item()
        if cached.get("schema") == SCHEMA:
            return sid, cached, "cached"

    p = preprocess(sid, w)
    ld = lambda f: np.asarray(nib.load(str(w / f)).dataobj)
    brain = ld(f"{sid}_meanepi_brain_mask.nii.gz") > 0
    pve1, pve2 = ld(f"{sid}_pve1_epi.nii.gz"), ld(f"{sid}_pve2_epi.nii.gz")
    cg = ld(f"{sid}_cerebrum_gm_epi.nii.gz") > 0.5
    cw = ld(f"{sid}_cerebrum_wm_epi.nii.gz") > 0.5
    sub = ld(f"{sid}_subcort_epi.nii.gz") > 0.3

    gm = (pve1 > GM_PVE_THR) & cg & brain
    wm = ndimage.binary_erosion((pve2 > WM_PVE_THR) & cw & (~sub) & brain, iterations=1)
    if gm.sum() < 500 or wm.sum() < 100:
        raise RuntimeError(f"masks too small (GM={gm.sum()}, WM={wm.sum()})")

    img = nib.load(str(p["mcf"]))
    data = np.asarray(img.dataobj, dtype=np.float32)
    n_vols = data.shape[3]

    row = {"participant_id": sid, "n_vols": n_vols, "schema": SCHEMA,
           "n_voxels_gm": int(gm.sum()), "n_voxels_wm": int(wm.sum())}
    for name, mask in (("gm", gm), ("wm", wm)):
        v = detrended_tsnr(data[mask].astype(np.float32))
        for k, val in summarize(v).items():
            row[f"{name}_{k}"] = val

    # sliding-window tSNR (median across mask voxels), window centers in volumes
    centers, series = [], {"gm": [], "wm": []}
    for start in range(0, n_vols - WINDOW + 1, STEP):
        chunk = data[..., start:start + WINDOW]
        centers.append(start + WINDOW / 2.0)
        for name, mask in (("gm", gm), ("wm", wm)):
            series[name].append(float(np.median(detrended_tsnr(chunk[mask].astype(np.float32)))))
    for name, path in (("gm", win_gm), ("wm", win_wm)):
        np.savetxt(path, np.column_stack([centers, series[name]]), fmt="%.4f",
                   header="window_center_vol median_tsnr")
    row["gm_window_mean"] = float(np.mean(series["gm"]))
    row["wm_window_mean"] = float(np.mean(series["wm"]))

    # BOLD signal dimensionality from the PCA eigenspectrum (whole brain and GM)
    for name, mask in (("brain", brain & (data.mean(3) > 0)), ("gm", gm)):
        lam, n_vox = pca_spectrum(data[mask].astype(np.float32))
        for k, val in dimensionality(lam, n_vox).items():
            row[f"dim_{name}_{k}"] = val
        if name == "brain":
            np.savetxt(SPECOUT / f"{sid}_eigenspectrum.txt", lam, fmt="%.8e",
                       header=f"eigenvalues of the temporal covariance, n_voxels={n_vox}")

    # whole-run tSNR map -> MNI 2 mm for the group average
    full = np.zeros(data.shape[:3], dtype=np.float32)
    inbrain = brain & (data.mean(3) > 0)
    full[inbrain] = detrended_tsnr(data[inbrain].astype(np.float32))
    tmp = w / f"{sid}_tsnr_epi.nii.gz"
    nib.save(nib.Nifti1Image(full, img.affine), str(tmp))
    sh(["flirt", "-in", str(tmp), "-ref", MNI_BRAIN, "-applyxfm",
        "-init", str(p["epi2mni"]), "-interp", "trilinear", "-out", str(mni_map)])

    np.save(csv_row, row, allow_pickle=True)
    # keep the motion-corrected series and transforms for reuse
    (w / f"{sid}_tsnr_epi.nii.gz").unlink(missing_ok=True)
    return sid, row, f"ok (GM {row['gm_median']:.1f}, WM {row['wm_median']:.1f}, {n_vols} vols)"


def main():
    import pandas as pd
    ph = C.load_phenotype()
    older = {i for i in ph.index if ph.loc[i, "age"] >= 60}
    subs = sorted(set(all_rest_subjects()) & older)
    subs = [s for s in subs if (T1DIR / f"{s}_T1w.nii.gz").exists()]
    print(f"{len(subs)} subjects with rest BOLD + T1w", flush=True)

    rows = []
    n_workers = int(os.environ.get("LIFE_TSNR_WORKERS", "6"))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process, s): s for s in subs}
        for k, f in enumerate(futures):
            sid = futures[f]
            try:
                sid, row, msg = f.result()
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                msg = f"FAILED: {e}"
            print(f"[{k+1}/{len(subs)}] {sid}: {msg}", flush=True)

    if rows:
        df = pd.DataFrame(rows).sort_values("participant_id")
        df.to_csv(OUT / "tsnr_tissue_per_subject.csv", index=False)
        print(f"\nWrote {OUT/'tsnr_tissue_per_subject.csv'} (n={len(df)})")
        print(f"  GM median tSNR: {df['gm_median'].mean():.1f} +/- {df['gm_median'].std():.1f}")
        print(f"  WM median tSNR: {df['wm_median'].mean():.1f} +/- {df['wm_median'].std():.1f}")
        if "dim_brain_pr" in df.columns:
            print(f"  dimensionality (participation ratio, whole brain): "
                  f"{df['dim_brain_pr'].mean():.1f} +/- {df['dim_brain_pr'].std():.1f}")
            print(f"  dimensionality (Minka MDL, whole brain): "
                  f"{df['dim_brain_mdl'].mean():.1f} +/- {df['dim_brain_mdl'].std():.1f}")
    print("COHORT_TSNR_DONE", flush=True)


if __name__ == "__main__":
    main()
