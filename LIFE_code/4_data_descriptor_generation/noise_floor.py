#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Noise-floor quantification for every imaging modality in the LIFE release.

In magnitude MR images the measured signal cannot decay to zero: rectifying
complex Gaussian noise leaves a Rayleigh-distributed floor whose mean is
sigma * sqrt(pi/2). Measurements at or near that level carry no tissue
information, so the distance between tissue signal and the floor bounds what any
downstream model can recover. This module estimates that distance per subject
and per modality.

Estimating sigma
----------------
No single estimator is valid for all five acquisitions, so each modality uses
the estimator its data actually support, and every estimator that *is* available
for a modality is written out so the choices can be cross-checked (see
`noise_floor_estimators_*.csv` and Figure S1):

  air        Rayleigh sigma from the median background intensity. Valid only
             where the reconstruction leaves the background untouched; the
             median (not the mean) is used so N/2 ghosts and residual signal
             outside the head do not inflate it.
  repeat     SD across the seven b = 0 repeats of the DWI series, measured
             inside the brain. Assumption-free, and the reference against which
             the other estimators are judged.
  thermal    Height of the white-noise plateau at the top of the BOLD temporal
             power spectrum, measured inside the brain. Also assumption-free.
  laplacian  MAD of the response to a 2-D Laplacian kernel inside the brain
             (Immerkaer 1996 combined with a robust MAD scale). Independent of
             the background, but only trustworthy where anatomy is well sampled
             relative to the voxel; it is used for the 1 mm structurals and
             validated against `air` on the 0.9 mm GRE, not applied to the
             2-2.4 mm EPI where anatomical structure contaminates it.

The T1w and FLAIR reconstructions zero a large share of the background (the
64-channel adaptive combine plus prescan normalize), so `air` is unusable there
and `laplacian` is the primary estimator. The EPI and GRE backgrounds are
intact, so `air` (DWI, GRE) or the direct in-brain estimators (DWI, BOLD) are
primary. Note that `air` measures noise outside the head and so misses the
g-factor amplification that parallel imaging produces in the brain interior;
that is why BOLD reports its temporal thermal estimate rather than `air`.

Metrics
-------
  sigma        noise SD from the modality's primary estimator
  floor        sigma * sqrt(pi/2), the expected magnitude for zero true signal
  snr          median in-brain signal / sigma
  snr_p10      10th-percentile in-brain signal / sigma (the low tail is what
               reaches the floor first)
  frac_floor   fraction of in-brain voxels at or below 2 * sigma, where the
               Rician bias on the magnitude still exceeds ~10 %

and the modality-specific axis along which signal decays toward the floor:

  DWI    SNR per shell (b = 0, 650, 1350, 2000 s/mm^2)
  GRE    SNR per echo of the 6-echo MET2_myelinSHIFT train, plus the TE at
         which the fitted decay reaches SNR 5 and SNR 2 - the TE horizon beyond
         which the acquisition cannot see
  BOLD   thermal plateau of the temporal spectrum, which caps achievable tSNR,
         reported as a tSNR ceiling and as % signal change against the
         ~0.5-1 % of a typical BOLD response
  T1w /  spatial SNR, split by FLAIR protocol (the release contains both a 2-D
  FLAIR  T2_FLAIR and a 3-D SAG T2 FLAIR SPACE variant)

All sequences use parallel imaging and the EPI additionally use multiband, so
noise is spatially non-stationary and a single global sigma is a summary rather
than a per-voxel truth. The estimates are comparable across subjects within a
modality, which is what the technical-validation figures use them for.

Outputs -> derivatives/technical_validation/noise_floor/*.csv

Usage:
    python3 noise_floor.py [dwi|gre|bold|struct|summary|all]
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.DERIV_DIR / "technical_validation" / "noise_floor"
OUT.mkdir(parents=True, exist_ok=True)

# Raw sources. The BIDS tree on this machine holds only a few subjects in full;
# the cohort-wide GRE / BOLD / structural conversions live in the session scratch
# directories, and the full DWI cohort sits beside the BIDS tree.
RAW_DIR = Path(os.environ.get("LIFE_DWI_NIFTI",
                              C.BASE_DIR / "LIFE_dwi_nifti"))
DWI_DIR = RAW_DIR  # kept for backward compatibility

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

# Working directories holding the per-modality conversions this module reads.
# Defaults sit beside the derivatives so the module runs on any machine;
# override individually with the env vars named. The DWI arm reads the
# released BIDS tree directly (see dwi_sources) and needs none of these.
SCRATCH = Path(os.environ.get(
    "LIFE_SCRATCH", C.DERIV_DIR / "technical_validation" / "noise_floor" / "work"))
GRE_DIR = Path(os.environ.get("LIFE_GRE_DIR", SCRATCH / "t2star"))
BOLD_DIR = Path(os.environ.get("LIFE_BOLD_DIR", SCRATCH / "bold"))
STRUCT_DIR = Path(os.environ.get("LIFE_STRUCT_DIR", SCRATCH / "struct"))

# Rayleigh constants: for background (zero true signal) magnitude data,
#   median = sigma * sqrt(2 ln 2),  mean = sigma * sqrt(pi/2).
RAYLEIGH_MEDIAN = np.sqrt(2 * np.log(2))     # 1.1774
RAYLEIGH_MEAN = np.sqrt(np.pi / 2)           # 1.2533
NORM_MAD = 0.6744897501960817                # median |N(0,1)|
# A voxel counts as noise-floor dominated below this multiple of sigma.
FLOOR_K = 2.0
# 2-D Laplacian of Immerkaer (1996); its noise gain is sqrt(sum(K^2)) = 6.
LAPLACE_K = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
LAPLACE_GAIN = 6.0


# =====================================================================
#  Masks
# =====================================================================

def otsu(vals: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold of a 1-D array (skimage is not installed here)."""
    v = vals[np.isfinite(vals)]
    hist, edges = np.histogram(np.clip(v, 0, np.percentile(v, 99.5)), bins=nbins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w = np.cumsum(hist).astype(float)
    m = np.cumsum(hist * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_between = (w / w[-1]) * (1 - w / w[-1]) * (m / w - (m[-1] - m) / (w[-1] - w)) ** 2
    var_between[~np.isfinite(var_between)] = 0
    return float(centers[int(np.argmax(var_between))])


def brain_mask(vol: np.ndarray) -> np.ndarray:
    """Coarse head mask: Otsu, largest connected component, fill, erode.

    Deliberately conservative — it only has to give a stable interior tissue
    sample, not an anatomically exact brain.
    """
    v = np.nan_to_num(vol.astype(np.float32))
    m = ndimage.binary_opening(v > otsu(v.ravel()), iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    # Pull in from the surface so skull and partial-volume edges stay out.
    return ndimage.binary_erosion(m, iterations=3)


def background_mask(vol: np.ndarray, brain: np.ndarray, edge: int = 3) -> np.ndarray:
    """Air voxels: outside a generously dilated head mask, away from the FOV rim
    (filtered reconstructions and wrap-around leave non-noise structure there)."""
    bg = ~ndimage.binary_dilation(brain, iterations=8)
    keep = np.zeros_like(bg)
    keep[edge:-edge, edge:-edge, edge:-edge] = True
    return bg & keep


# =====================================================================
#  Sigma estimators
# =====================================================================

def sigma_air(vol: np.ndarray, bg: np.ndarray) -> float:
    """Rayleigh sigma from the median background intensity."""
    b = vol[bg]
    b = b[np.isfinite(b)]
    if b.size < 500:
        return float("nan")
    return float(np.median(b) / RAYLEIGH_MEDIAN)


def background_zero_frac(vol: np.ndarray, bg: np.ndarray) -> float:
    """Share of background voxels the reconstruction has set to exactly zero.

    A non-trivial value means the background has been suppressed and `sigma_air`
    is not measuring thermal noise any more.
    """
    b = vol[bg]
    return float((b == 0).mean()) if b.size else float("nan")


def sigma_laplacian(vol: np.ndarray, brain: np.ndarray) -> float:
    """MAD of a 2-D Laplacian response inside the brain (background-independent).

    Applied slice-wise so through-plane anisotropy does not enter, and taken as a
    median absolute value so the minority of voxels sitting on tissue edges do
    not dominate.
    """
    v = np.nan_to_num(vol.astype(np.float32))
    resp = np.empty_like(v)
    for z in range(v.shape[2]):
        resp[:, :, z] = ndimage.convolve(v[:, :, z], LAPLACE_K, mode="nearest")
    inner = ndimage.binary_erosion(brain, iterations=2)
    if inner.sum() < 500:
        return float("nan")
    return float(np.median(np.abs(resp[inner])) / NORM_MAD / LAPLACE_GAIN)


# =====================================================================
#  Shared metrics
# =====================================================================

def floor_metrics(vol: np.ndarray, brain: np.ndarray, sigma: float) -> dict:
    """Median / low-tail in-brain signal, SNR, and the noise-floor fraction."""
    t = vol[brain]
    t = t[np.isfinite(t)]
    if t.size == 0 or not np.isfinite(sigma) or sigma <= 0:
        return {"signal": np.nan, "signal_p10": np.nan, "snr": np.nan,
                "snr_p10": np.nan, "frac_floor": np.nan}
    med = float(np.median(t))
    p10 = float(np.percentile(t, 10))
    return {"signal": med, "signal_p10": p10, "snr": med / sigma,
            "snr_p10": p10 / sigma, "frac_floor": float((t <= FLOOR_K * sigma).mean())}


def decay_horizon(tes: np.ndarray, snr: np.ndarray, level: float) -> float:
    """TE (ms) at which a mono-exponential fit to SNR(TE) reaches `level`.

    The echo train stops at 65 ms, well above the noise floor, so the crossing is
    an extrapolation of the fitted decay: the TE beyond which this acquisition
    could not distinguish signal from noise even if it were sampled.
    """
    tes, snr = np.asarray(tes, float), np.asarray(snr, float)
    ok = np.isfinite(tes) & np.isfinite(snr) & (snr > 0)
    if ok.sum() < 3:
        return float("nan")
    slope, intercept = np.polyfit(tes[ok], np.log(snr[ok]), 1)
    if slope >= 0:
        return float("nan")
    return float((np.log(level) - intercept) / slope)


def subj_from(path: Path) -> str:
    return path.name.split("_")[0]


# Tidy sigma-estimator rows for the Figure S1 validation panel, kept per
# modality because the modality runs are launched concurrently and a single
# shared file would race.
ESTIMATORS: dict[str, list[dict]] = {}


def log_estimators(sub: str, modality: str, **kw) -> None:
    for name, val in kw.items():
        if val is not None and np.isfinite(val):
            ESTIMATORS.setdefault(modality, []).append(
                {"participant_id": sub, "modality": modality,
                 "estimator": name, "sigma": float(val)})


# =====================================================================
#  DWI  — primary sigma: b0 repeats
# =====================================================================

def run_dwi() -> pd.DataFrame:
    src = dwi_sources()
    print(f"[dwi] {len(src)} subjects")
    rows = []
    for k, (sub, (f, bval_f)) in enumerate(sorted(src.items()), 1):
        if not bval_f.exists():
            print(f"  {sub}: no bval, skipped")
            continue
        # Round to the nominal shells so per-volume deviations collapse.
        shells = np.round(np.loadtxt(bval_f) / 50.0) * 50.0
        data = np.asanyarray(nib.load(str(f)).dataobj, dtype=np.float32)

        b0 = data[..., np.where(shells == 0)[0]]
        brain = brain_mask(b0.mean(axis=-1))
        bg = background_mask(b0.mean(axis=-1), brain)

        s_air = float(np.median([sigma_air(b0[..., i], bg) for i in range(b0.shape[-1])]))
        s_lap = sigma_laplacian(b0[..., 0], brain)
        # Assumption-free: the spread across identical b0 acquisitions.
        s_rep = (float(np.median(b0.std(axis=-1, ddof=1)[brain]))
                 if b0.shape[-1] >= 3 else float("nan"))
        sigma = s_rep if np.isfinite(s_rep) else s_air
        z_frac = background_zero_frac(b0[..., 0], bg)
        log_estimators(sub, "DWI", repeat=s_rep, air=s_air, laplacian=s_lap)

        for b in np.unique(shells):
            idx = np.where(shells == b)[0]
            # Metrics are per single diffusion volume — that is what the model
            # actually sees — so the floor fraction is the median over the
            # shell's volumes rather than a property of their average.
            m = floor_metrics(data[..., idx].mean(axis=-1) if idx.size > 1
                              else data[..., idx[0]], brain, sigma)
            ff = float(np.median([(data[..., i][brain] <= FLOOR_K * sigma).mean()
                                  for i in idx]))
            rows.append({
                "participant_id": sub, "shell": float(b), "n_dirs": int(idx.size),
                "sigma": sigma, "sigma_air": s_air, "sigma_repeat": s_rep,
                "sigma_laplacian": s_lap, "bg_zero_frac": z_frac,
                "floor": sigma * RAYLEIGH_MEAN,
                "signal": m["signal"], "signal_p10": m["signal_p10"],
                "snr": m["snr"], "snr_p10": m["snr_p10"], "frac_floor": ff,
            })
        if k % 10 == 0 or k == len(src):
            print(f"  {k}/{len(src)}", flush=True)
        del data
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_floor_dwi.csv", index=False)
    print(f"[dwi] wrote noise_floor_dwi.csv ({df.participant_id.nunique()} subjects)")
    return df


# =====================================================================
#  Multi-echo GRE (T2*)  — primary sigma: air
# =====================================================================

def gre_echo_times(sub: str) -> np.ndarray:
    """Echo times (ms) from the sidecars, falling back to the protocol values."""
    tes = []
    for e in range(1, 7):
        js = GRE_DIR / f"{sub}_ME-T2_e{e}.json"
        try:
            tes.append(float(json.load(open(js))["EchoTime"]) * 1000.0)
        except Exception:
            tes.append(np.nan)
    tes = np.array(tes)
    return tes if np.all(np.isfinite(tes)) else np.array(
        [2.68, 8.82, 16.58, 24.35, 40.0, 65.0])


def run_gre() -> pd.DataFrame:
    subs = sorted({p.name.split("_ME-T2")[0]
                   for p in GRE_DIR.glob("sub-*_ME-T2_e1.nii.gz")})
    print(f"[gre] {len(subs)} subjects")
    rows = []
    for k, sub in enumerate(subs, 1):
        echoes = [GRE_DIR / f"{sub}_ME-T2_e{e}.nii.gz" for e in range(1, 7)]
        if not all(p.exists() for p in echoes):
            print(f"  {sub}: incomplete echo train, skipped")
            continue
        shapes = {nib.load(str(p)).shape[:3] for p in echoes}
        if len(shapes) > 1:
            print(f"  {sub}: echoes disagree on geometry {sorted(shapes)}, skipped")
            continue
        tes = gre_echo_times(sub)
        e1 = np.asanyarray(nib.load(str(echoes[0])).dataobj, dtype=np.float32)

        bm_f = GRE_DIR / f"{sub}_brainmask.nii.gz"
        brain = (ndimage.binary_erosion(
                    np.asanyarray(nib.load(str(bm_f)).dataobj) > 0, iterations=2)
                 if bm_f.exists() else brain_mask(e1))
        bg = background_mask(e1, brain)

        # Thermal noise does not depend on TE, so sigma is estimated once from the
        # first echo, where nothing has yet decayed into the floor.
        sigma = sigma_air(e1, bg)
        s_lap = sigma_laplacian(e1, brain)
        z_frac = background_zero_frac(e1, bg)
        log_estimators(sub, "ME-GRE", air=sigma, laplacian=s_lap)

        snrs, block = [], []
        for e, (p, te) in enumerate(zip(echoes, tes), 1):
            m = floor_metrics(np.asanyarray(nib.load(str(p)).dataobj, dtype=np.float32),
                              brain, sigma)
            snrs.append(m["snr"])
            block.append({
                "participant_id": sub, "echo": e, "te_ms": float(te),
                "sigma": sigma, "sigma_air": sigma, "sigma_laplacian": s_lap,
                "bg_zero_frac": z_frac, "floor": sigma * RAYLEIGH_MEAN,
                "signal": m["signal"], "signal_p10": m["signal_p10"],
                "snr": m["snr"], "snr_p10": m["snr_p10"],
                "frac_floor": m["frac_floor"],
            })
        te5 = decay_horizon(tes, np.array(snrs), 5.0)
        te2 = decay_horizon(tes, np.array(snrs), 2.0)
        for r in block:
            r["te_snr5"], r["te_snr2"] = te5, te2
        rows.extend(block)
        if k % 10 == 0 or k == len(subs):
            print(f"  {k}/{len(subs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_floor_gre.csv", index=False)
    print(f"[gre] wrote noise_floor_gre.csv ({df.participant_id.nunique()} subjects)")
    return df


# =====================================================================
#  rs-fMRI BOLD  — primary sigma: temporal thermal plateau
# =====================================================================

def run_bold() -> pd.DataFrame:
    files = sorted(BOLD_DIR.glob("sub-*_rest.nii.gz"))
    print(f"[bold] {len(files)} subjects")
    rows, spectra = [], []
    for k, f in enumerate(files, 1):
        sub = subj_from(f)
        img = nib.load(str(f))
        tr = float(img.header.get_zooms()[3])
        if not (0.1 < tr < 10):
            js = f.with_name(f.name.replace(".nii.gz", ".json"))
            tr = float(json.load(open(js))["RepetitionTime"]) if js.exists() else np.nan
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        nt = data.shape[-1]
        mean_vol = data.mean(axis=-1)
        brain = brain_mask(mean_vol)
        bg = background_mask(mean_vol, brain)

        s_air = sigma_air(data[..., nt // 2], bg)
        z_frac = background_zero_frac(data[..., nt // 2], bg)
        # Spatial metrics come from a single volume, not the temporal mean:
        # averaging 450 volumes would shrink the noise by sqrt(450) and put the
        # floor fraction out of step with the per-volume figure reported for DWI.
        sm = floor_metrics(data[..., nt // 2], brain, s_air)

        Y = data[brain].astype(np.float64)
        mu = Y.mean(axis=1)
        keep = mu > (FLOOR_K * s_air if np.isfinite(s_air) else 0)
        Y, mu = Y[keep], mu[keep]
        # Remove mean, linear and quadratic drift: scanner drift is not noise and
        # would otherwise land in the temporal SD.
        t = np.linspace(-1, 1, nt)
        X = np.column_stack([np.ones(nt), t, t ** 2])
        R = Y - (X @ np.linalg.lstsq(X, Y.T, rcond=None)[0]).T
        sd_total = R.std(axis=1, ddof=X.shape[1])

        # White (thermal) noise appears as a flat plateau at the top of the
        # spectrum, above the physiological and BOLD bands. Its height gives the
        # noise-floor variance; the remaining temporal variance is structured.
        psd = (np.abs(np.fft.rfft(R, axis=1)) ** 2) / nt
        freqs = np.fft.rfftfreq(nt, d=tr)
        hi = freqs >= 0.75 * freqs[-1]                # top quarter of the band
        plateau = np.median(psd[:, hi], axis=1)       # median: robust to spikes
        sd_thermal = np.sqrt(np.maximum(plateau * (len(freqs) - 1) * 2 / nt, 0))
        log_estimators(sub, "BOLD", thermal=float(np.median(sd_thermal)),
                       air=s_air, laplacian=sigma_laplacian(data[..., nt // 2], brain))

        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr = np.where(sd_total > 0, mu / sd_total, np.nan)
            ceiling = np.where(sd_thermal > 0, mu / sd_thermal, np.nan)
            th_frac = np.clip(sd_thermal ** 2 / np.maximum(sd_total ** 2, 1e-12), 0, 1)

        # Cohort-mean spectrum, normalized per subject so it can be averaged.
        pm = psd.mean(axis=0)
        spectra.append(pd.DataFrame({"participant_id": sub, "freq": freqs,
                                     "psd_norm": pm / np.median(pm[hi])}))

        rows.append({
            "participant_id": sub, "n_volumes": nt, "tr": tr,
            "sigma": float(np.median(sd_thermal)), "sigma_air": s_air,
            "bg_zero_frac": z_frac,
            "floor": float(np.median(sd_thermal)) * RAYLEIGH_MEAN,
            "signal": sm["signal"], "snr_spatial": sm["snr"],
            "frac_floor": sm["frac_floor"],
            "tsnr": float(np.nanmedian(tsnr)),
            "tsnr_thermal_ceiling": float(np.nanmedian(ceiling)),
            "thermal_pct_signal": float(np.nanmedian(100 * sd_thermal / mu)),
            "total_pct_signal": float(np.nanmedian(100 * sd_total / mu)),
            "thermal_var_frac": float(np.nanmedian(th_frac)),
            "n_vox": int(keep.sum()),
        })
        print(f"  {k}/{len(files)} {sub} tSNR={rows[-1]['tsnr']:.1f} "
              f"ceiling={rows[-1]['tsnr_thermal_ceiling']:.1f}", flush=True)
        del data, Y, R, psd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_floor_bold.csv", index=False)
    pd.concat(spectra, ignore_index=True).to_csv(
        OUT / "noise_floor_bold_spectra.csv", index=False)
    print(f"[bold] wrote noise_floor_bold.csv ({len(df)} subjects)")
    return df


# =====================================================================
#  Structural T1w / FLAIR  — primary sigma: laplacian
# =====================================================================

def run_struct() -> pd.DataFrame:
    rows = []
    for tag in ("T1w", "FLAIR"):
        files = sorted(STRUCT_DIR.glob(f"sub-*_{tag}.nii.gz"))
        print(f"[struct] {tag}: {len(files)} subjects")
        for k, f in enumerate(files, 1):
            sub = subj_from(f)
            img = nib.load(str(f))
            vol = np.asanyarray(img.dataobj, dtype=np.float32)
            if vol.ndim > 3:
                vol = vol[..., 0]
            brain = brain_mask(vol)
            bg = background_mask(vol, brain)
            sigma = sigma_laplacian(vol, brain)
            s_air = sigma_air(vol, bg)
            log_estimators(sub, tag, laplacian=sigma, air=s_air)
            m = floor_metrics(vol, brain, sigma)
            t = vol[brain]
            # The release carries two FLAIR protocols; keep them separable.
            js = f.with_name(f.name.replace(".nii.gz", ".json"))
            seq = ""
            if js.exists():
                d = json.load(open(js))
                seq = d.get("SeriesDescription") or d.get("ProtocolName") or ""
            rows.append({
                "participant_id": sub, "contrast": tag, "sequence": seq,
                "voxel_mm": float(np.prod(img.header.get_zooms()[:3])),
                "sigma": sigma, "sigma_laplacian": sigma, "sigma_air": s_air,
                "bg_zero_frac": background_zero_frac(vol, bg),
                "floor": sigma * RAYLEIGH_MEAN,
                "signal": m["signal"], "signal_p10": m["signal_p10"],
                "snr": m["snr"], "snr_p10": m["snr_p10"],
                # The bright tissue peak (WM in T1w) is the conventional SNR
                # reference; report it alongside the median.
                "signal_bright": float(np.percentile(t, 90)),
                "snr_bright": float(np.percentile(t, 90) / sigma) if sigma else np.nan,
                "frac_floor": m["frac_floor"],
            })
            if k % 15 == 0 or k == len(files):
                print(f"  {k}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_floor_struct.csv", index=False)
    print("[struct] wrote noise_floor_struct.csv")
    return df


# =====================================================================
#  Summary
# =====================================================================

def summarize() -> pd.DataFrame:
    def q(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return dict(n=len(s), median=s.median(), iqr_lo=s.quantile(.25),
                    iqr_hi=s.quantile(.75), mean=s.mean(),
                    sd=s.std(ddof=1) if len(s) > 1 else np.nan)

    rows = []
    p = OUT / "noise_floor_struct.csv"
    if p.exists():
        d = pd.read_csv(p)
        for (tag, seq), g in d.groupby(["contrast", "sequence"]):
            lab = tag if tag == "T1w" else f"{tag} ({seq})"
            rows += [{"modality": lab, "metric": "SNR (median tissue / sigma)", **q(g.snr)},
                     {"modality": lab, "metric": "fraction of brain at noise floor", **q(g.frac_floor)}]
    p = OUT / "noise_floor_gre.csv"
    if p.exists():
        d = pd.read_csv(p)
        for e, g in d.groupby("echo"):
            rows.append({"modality": "ME-GRE",
                         "metric": f"SNR echo {e} (TE {g.te_ms.iloc[0]:.2f} ms)", **q(g.snr)})
        first = d.groupby("participant_id").first()
        rows += [{"modality": "ME-GRE", "metric": "TE horizon at SNR = 5 (ms)", **q(first.te_snr5)},
                 {"modality": "ME-GRE", "metric": "TE horizon at SNR = 2 (ms)", **q(first.te_snr2)}]
    p = OUT / "noise_floor_dwi.csv"
    if p.exists():
        d = pd.read_csv(p)
        for b, g in d.groupby("shell"):
            rows += [{"modality": "DWI", "metric": f"SNR b = {int(b)}", **q(g.snr)},
                     {"modality": "DWI", "metric": f"SNR b = {int(b)} (10th pct voxels)", **q(g.snr_p10)},
                     {"modality": "DWI", "metric": f"fraction at noise floor b = {int(b)}", **q(g.frac_floor)}]
    p = OUT / "noise_floor_bold.csv"
    if p.exists():
        d = pd.read_csv(p)
        for col, lab in [("snr_spatial", "spatial SNR"), ("tsnr", "tSNR"),
                         ("tsnr_thermal_ceiling", "tSNR thermal ceiling"),
                         ("thermal_pct_signal", "thermal noise (% signal)"),
                         ("total_pct_signal", "total temporal noise (% signal)"),
                         ("thermal_var_frac", "thermal fraction of temporal variance")]:
            rows.append({"modality": "BOLD", "metric": lab, **q(d[col])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_floor_summary.csv", index=False)
    print(df.to_string(index=False))
    return df


def flush_estimators() -> None:
    for modality, rows in ESTIMATORS.items():
        f = OUT / f"noise_floor_estimators_{modality.replace('-', '').lower()}.csv"
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"[est] wrote {f.name} ({len(rows)} rows)")


def load_estimators() -> pd.DataFrame:
    """Concatenate the per-modality estimator files written by each run."""
    fs = sorted(OUT.glob("noise_floor_estimators_*.csv"))
    return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
            if fs else pd.DataFrame())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("dwi", "all"):
        run_dwi()
    if what in ("gre", "all"):
        run_gre()
    if what in ("struct", "all"):
        run_struct()
    if what in ("bold", "all"):
        run_bold()
    flush_estimators()
    if what in ("summary", "all"):
        summarize()
