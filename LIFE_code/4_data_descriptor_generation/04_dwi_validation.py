#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Technical Validation — Diffusion-Weighted Imaging
==================================================
Implements the "Diffusion-Weighted Imaging" bullets:

  1. Per-shell SNR — signal-to-noise for each b-value shell
     (b = 0/650/1350/2000), averaged across participants.
  2. Preprocessing residuals — FSL eddy outlier-slice count and mean motion
     (RMS) estimates (read from eddy outputs if present).
  3. DTI metric distributions — mean +/- SD of whole white-matter FA and MD
     (expected FA ~0.40-0.50; MD ~0.70-0.80 x10^-3 mm^2/s in older adults).
  4. Age effects on DTI metrics — partial correlations of WM-mean FA/MD with
     age (control sex). Expect lower FA, higher MD with age.
  5. NODDI validation — mean +/- SD of ICVF/ISOVF/ODI; ICVF-FA correlation
     (expect r>0.60) and ICVF age effect surviving FA covariate.
  6. Connectome density & weight distributions — edge density (proportion of
     non-zero edges) and streamline-weight distribution across participants.
  7. Tract-specific consistency — along-tract FA/ICVF profiles (TractSeg),
     if bundle metrics are available.

Data reality
------------
The LIFE DWI derivatives are DSI Studio GQI connectomes
(derivatives/dsi_studio_gqi/<sub>/dwi/*.connectivity.mat). Per-shell SNR and
connectome density are computed from those + the raw DWI. DTI (FSL dtifit),
NODDI (AMICO), and TractSeg analyses run against standard output paths if
present, and are otherwise skipped with a clear message.

Outputs  ->  derivatives/technical_validation/dwi/
  per_shell_snr.csv, eddy_residuals.csv, dti_wm_metrics_dtifit.csv,
  dti_age_effects_dtifit.csv, noddi_metrics.csv, connectome_density.csv,
  tract_profiles.csv, dwi_validation_report.txt
"""

import argparse
import gzip
import io

import numpy as np
import pandas as pd

import config as C

SHELL_TOL = 100  # b-values within +/-tol are grouped into one shell


# =====================================================================
#  1. Per-shell SNR
# =====================================================================

def shell_of(bval, shells=(0, 650, 1350, 2000)):
    return min(shells, key=lambda s: abs(bval - s))


def per_shell_snr(subjects):
    """
    For each shell, SNR = mean over brain voxels of
    (mean-across-shell-volumes / std-across-shell-volumes).
    Brain mask = signal above the 30th percentile of the mean b0.
    """
    import nibabel as nib
    rows = []
    for sub, sub_dir in subjects:
        dwi = sub_dir / "dwi" / f"{sub}_dwi.nii.gz"
        bval_f = sub_dir / "dwi" / f"{sub}_dwi.bval"
        if not (dwi.exists() and bval_f.exists()):
            continue
        bvals = np.loadtxt(bval_f)
        img = nib.load(str(dwi))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 4 or data.shape[3] != len(bvals):
            continue
        b0_mean = data[..., bvals < SHELL_TOL].mean(axis=3)
        thr = np.percentile(b0_mean[b0_mean > 0], 30) if (b0_mean > 0).any() else 0
        mask = b0_mean > thr
        rec = {"participant_id": sub}
        for shell in (0, 650, 1350, 2000):
            sel = np.array([abs(shell_of(b) - shell) < 1 and abs(b - shell) <= SHELL_TOL
                            for b in bvals])
            if sel.sum() == 0:
                rec[f"snr_b{shell}"] = np.nan
                continue
            vols = data[..., sel]
            if vols.shape[3] < 2:
                # single volume (e.g. some b0 schemes): spatial SNR proxy
                sig = vols[..., 0][mask]
                rec[f"snr_b{shell}"] = float(sig.mean() /
                                             (sig.std() + 1e-6))
                continue
            m = vols.mean(axis=3)
            s = vols.std(axis=3, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(s > 0, m / s, 0.0)
            rec[f"snr_b{shell}"] = float(np.median(snr[mask]))
        rows.append(rec)
    return pd.DataFrame(rows)


# =====================================================================
#  2. Eddy preprocessing residuals
# =====================================================================

def eddy_residuals(subjects):
    """Read FSL eddy outlier report + motion RMS if eddy outputs exist."""
    rows = []
    base = C.DERIV_DIR / "dwi"
    if not base.is_dir():
        return pd.DataFrame()
    for sub, _ in subjects:
        sdir = base / sub
        if not sdir.is_dir():
            continue
        rec = {"participant_id": sub}
        # outlier map: *.eddy_outlier_map (rows=vols, cols=slices; 1=outlier)
        omap = list(sdir.rglob("*eddy_outlier_map"))
        if omap:
            arr = np.loadtxt(omap[0], skiprows=1)
            rec["n_outlier_slices"] = int(arr.sum())
            rec["pct_outlier_slices"] = float(arr.mean() * 100)
        # motion: *.eddy_movement_rms (col2 = relative RMS)
        rms = list(sdir.rglob("*eddy_movement_rms"))
        if rms:
            m = np.loadtxt(rms[0])
            rec["mean_rel_motion_rms"] = float(m[:, 1].mean()) if m.ndim == 2 else np.nan
        if len(rec) > 1:
            rows.append(rec)
    return pd.DataFrame(rows)


# =====================================================================
#  3 & 4. DTI metrics
# =====================================================================

def wm_mean_from_map(map_path, mask_path=None):
    import nibabel as nib
    img = nib.load(str(map_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if mask_path and mask_path.exists():
        m = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    else:
        m = data > 0  # nonzero as coarse WM proxy
    vals = data[m]
    return float(np.nanmean(vals)) if vals.size else np.nan


def dti_metrics(subjects):
    """
    Whole-WM mean FA/MD from FSL dtifit outputs if present:
      derivatives/dwi/<sub>/<sub>_FA.nii.gz, <sub>_MD.nii.gz
      (+ optional <sub>_wm_mask.nii.gz).
    """
    rows = []
    base = C.DERIV_DIR / "dwi"
    for sub, _ in subjects:
        sdir = base / sub if base.is_dir() else None
        if sdir is None or not sdir.is_dir():
            continue
        fa = next(iter(sdir.rglob("*_FA.nii*")), None)
        md = next(iter(sdir.rglob("*_MD.nii*")), None)
        mask = next(iter(sdir.rglob("*_wm_mask.nii*")), None)
        if fa is None and md is None:
            continue
        rec = {"participant_id": sub}
        if fa:
            rec["wm_mean_FA"] = wm_mean_from_map(fa, mask)
        if md:
            rec["wm_mean_MD"] = wm_mean_from_map(md, mask)
        rows.append(rec)
    return pd.DataFrame(rows)


# =====================================================================
#  5. NODDI metrics (AMICO)
# =====================================================================

def noddi_metrics(subjects):
    """
    Whole-WM mean ICVF/ISOVF/ODI from AMICO outputs if present:
      derivatives/amico/<sub>/AMICO/NODDI/FIT_{ICVF,ISOVF,OD}.nii.gz
    """
    rows = []
    base = C.DERIV_DIR / "amico"
    if not base.is_dir():
        return pd.DataFrame()
    for sub, _ in subjects:
        sdir = base / sub
        if not sdir.is_dir():
            continue
        rec = {"participant_id": sub}
        for key, pat in [("ICVF", "*ICVF*.nii*"), ("ISOVF", "*ISOVF*.nii*"),
                         ("ODI", "*OD*.nii*")]:
            f = next(iter(sdir.rglob(pat)), None)
            if f:
                rec[f"wm_mean_{key}"] = wm_mean_from_map(f)
        if len(rec) > 1:
            rows.append(rec)
    return pd.DataFrame(rows)


# =====================================================================
#  6. Connectome density & weights (DSI Studio GQI)
# =====================================================================

def _load_mat_gz(path):
    """Load a (possibly gzipped) MATLAB .mat and return dict."""
    from scipy.io import loadmat
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return loadmat(io.BytesIO(f.read()))
    return loadmat(str(path))


def connectome_density(subjects, weight="count"):
    """
    Edge density (proportion non-zero off-diagonal edges) and streamline
    weight distribution from DSI Studio GQI .connectivity.mat files.
    """
    rows = []
    for sub, _ in subjects:
        sdir = C.DSI_DIR / sub / "dwi"
        if not sdir.is_dir():
            continue
        mats = list(sdir.glob(
            f"*{weight}.pass.connectivity.mat"))
        if not mats:
            mats = list(sdir.glob("*count*connectivity.mat"))
        if not mats:
            continue
        try:
            d = _load_mat_gz(mats[0])
        except Exception:  # noqa: BLE001
            continue
        M = d.get("connectivity")
        if M is None:
            # fall back to first 2-D numeric array
            for v in d.values():
                if isinstance(v, np.ndarray) and v.ndim == 2 and \
                        v.shape[0] == v.shape[1]:
                    M = v
                    break
        if M is None:
            continue
        M = np.asarray(M, dtype=float)
        n = M.shape[0]
        off = ~np.eye(n, dtype=bool)
        nonzero = (M[off] > 0)
        rows.append({"participant_id": sub, "n_nodes": n,
                     "edge_density": float(nonzero.mean()),
                     "mean_weight": float(M[off][nonzero].mean()) if nonzero.any() else 0.0,
                     "total_streamlines": float(M[off].sum() / 2)})
    return pd.DataFrame(rows)


# =====================================================================
#  7. Tract-specific consistency (TractSeg)
# =====================================================================

def tract_profiles(subjects):
    """
    Along-tract FA/ICVF profiles from TractSeg 'Tractometry' outputs if
    present: derivatives/tractseg/<sub>/Tractometry.csv (bundle x position).
    Returns group mean profile per bundle.
    """
    base = C.DERIV_DIR / "tractseg"
    if not base.is_dir():
        return pd.DataFrame()
    frames = []
    for sub, _ in subjects:
        f = base / sub / "Tractometry.csv"
        if f.exists():
            df = pd.read_csv(f, sep=";")
            df["participant_id"] = sub
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    bundle_cols = [c for c in allp.columns if c != "participant_id"]
    return allp.groupby(level=0)[bundle_cols].mean() if not allp.empty else pd.DataFrame()


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--weight", default="count",
                    help="connectome edge weight for density (count/qa/mean_length)")
    args = ap.parse_args()

    out = C.outdir("dwi")
    report = [C.banner("DIFFUSION-WEIGHTED IMAGING — Technical Validation")]

    subjects = [(s, d) for s, d in C.subject_dirs()
                if args.subject is None or s == args.subject]
    pheno = C.load_phenotype()

    # ── 1. Per-shell SNR ──
    snr = per_shell_snr(subjects)
    snr.to_csv(out / "per_shell_snr.csv", index=False)
    report.append(C.banner("1. Per-shell SNR"))
    if not snr.empty:
        for shell in (0, 650, 1350, 2000):
            col = f"snr_b{shell}"
            if col in snr and snr[col].notna().any():
                d = C.describe(snr[col])
                report.append(f"  b={shell:<5} SNR mean={d['mean']:.1f} "
                              f"SD={d['sd']:.1f} n={d['n']}")
    else:
        report.append("  No raw DWI + bval found.")

    # ── 2. Eddy residuals ──
    eddy = eddy_residuals(subjects)
    if not eddy.empty:
        eddy.to_csv(out / "eddy_residuals.csv", index=False)
        report.append(C.banner("2. Preprocessing residuals (FSL eddy)"))
        if "pct_outlier_slices" in eddy:
            d = C.describe(eddy["pct_outlier_slices"])
            report.append(f"  outlier slices: mean={d['mean']:.2f}% "
                          f"range=[{d['min']:.2f},{d['max']:.2f}]%")
        if "mean_rel_motion_rms" in eddy:
            d = C.describe(eddy["mean_rel_motion_rms"])
            report.append(f"  relative motion RMS: mean={d['mean']:.3f} mm")
    else:
        report.append("\n[2] No FSL eddy outputs under derivatives/dwi/; "
                      "skipped.")

    # ── 3 & 4. DTI ── (FSL dtifit maps if present; otherwise see
    # for_data_release/13_dti_from_src.py, which reconstructs DTI from SRC)
    dti = dti_metrics(subjects)
    report.append(C.banner("3. DTI metric distributions (whole WM)"))
    if not dti.empty:
        # Written under a _dtifit suffix on purpose. 06_dti_wm_from_qsdr.py
        # writes dti_wm_metrics.csv / dti_age_effects.csv from the GQI/QSDR
        # reconstruction, and those are the tables Figure 4D and the
        # manuscript use. This block is the alternative FSL dtifit route; if
        # both wrote the same filenames, whichever script ran last would
        # silently decide which numbers the figure shows.
        dti.to_csv(out / "dti_wm_metrics_dtifit.csv", index=False)
        for col, unit in [("wm_mean_FA", ""), ("wm_mean_MD", " x10^-3 mm^2/s")]:
            if col in dti:
                d = C.describe(dti[col])
                report.append(f"  {col}: mean={d['mean']:.3f} "
                              f"SD={d['sd']:.3f}{unit} n={d['n']}")
        # age effects
        report.append(C.banner("4. Age effects on DTI (expect FA down, MD up)"))
        m = dti.set_index("participant_id").join(pheno)
        covars = [c for c in ["sex_code"] if c in m.columns]
        age_rows = []
        for col in ["wm_mean_FA", "wm_mean_MD"]:
            if col in m.columns:
                res = C.partial_corr(m, col, "age", covars)
                res["metric"] = col
                age_rows.append(res)
                report.append(f"  {col} vs age: partial r={res['r']:+.3f} "
                              f"p={res['p']:.3f} n={res['n']}")
        pd.DataFrame(age_rows).to_csv(out / "dti_age_effects_dtifit.csv",
                                      index=False)
    else:
        report.append("  No FSL dtifit FA/MD maps under derivatives/dwi/; "
                      "run dtifit on the b0+b650 shells to enable.")

    # ── 5. NODDI ──
    noddi = noddi_metrics(subjects)
    report.append(C.banner("5. NODDI validation"))
    if not noddi.empty:
        noddi.to_csv(out / "noddi_metrics.csv", index=False)
        for col in ["wm_mean_ICVF", "wm_mean_ISOVF", "wm_mean_ODI"]:
            if col in noddi:
                d = C.describe(noddi[col])
                report.append(f"  {col}: mean={d['mean']:.3f} SD={d['sd']:.3f} "
                              f"n={d['n']}")
        # ICVF-FA correlation
        if not dti.empty and "wm_mean_ICVF" in noddi:
            merged = noddi.merge(dti, on="participant_id")
            if "wm_mean_FA" in merged and len(merged) > 3:
                from scipy import stats
                mm = merged[["wm_mean_ICVF", "wm_mean_FA"]].dropna()
                r, p = stats.pearsonr(mm["wm_mean_ICVF"], mm["wm_mean_FA"])
                report.append(f"  ICVF-FA correlation: r={r:+.3f} p={p:.3f} "
                              f"(expect r>0.60)")
    else:
        report.append("  No AMICO/NODDI outputs under derivatives/amico/; "
                      "run AMICO to enable.")

    # ── 6. Connectome density ──
    dens = connectome_density(subjects, args.weight)
    report.append(C.banner("6. Connectome density & weight distributions"))
    if not dens.empty:
        dens.to_csv(out / "connectome_density.csv", index=False)
        d = C.describe(dens["edge_density"])
        report.append(f"  edge density: mean={d['mean']:.3f} SD={d['sd']:.3f} "
                      f"range=[{d['min']:.3f},{d['max']:.3f}] n={d['n']}")
        dw = C.describe(dens["total_streamlines"])
        report.append(f"  total streamlines: mean={dw['mean']:.0f} "
                      f"SD={dw['sd']:.0f}")
    else:
        report.append("  No DSI Studio .connectivity.mat found.")

    # ── 7. Tract profiles ──
    tracts = tract_profiles(subjects)
    if not tracts.empty:
        tracts.to_csv(out / "tract_profiles.csv")
        report.append(C.banner("7. Tract-specific consistency (TractSeg)"))
        report.append(f"  Group-mean along-tract profiles saved "
                      f"({tracts.shape[1]} bundle columns).")
    else:
        report.append("\n[7] No TractSeg Tractometry outputs; skipped.")

    (out / "dwi_validation_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
