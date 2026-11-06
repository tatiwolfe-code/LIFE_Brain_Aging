#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Whole-white-matter QA / FA / MD (+ RD, AD-free) from the released QSDR FIB files.
=================================================================================
The LIFE diffusion derivatives are DSI Studio QSDR reconstructions
(derivatives/connectometry/qsdr_fib/sub-*_desc-qsdr_dwi.fib.gz), already in
ICBM152 2 mm template space and carrying, per voxel:

    fa0     spin distribution function peak  -> quantitative anisotropy (QA)
    dti_fa  tensor fractional anisotropy     -> FA
    md, rd  tensor mean / radial diffusivity -> 10^-3 mm^2/s
    R2      QSDR goodness-of-fit to template (registration QC)

Metrics are averaged inside each subject's own white-matter mask (FA > 0.2),
matching the convention already used by for_data_release/13_dti_from_src.py.
This is a whole-WM average, NOT a TBSS skeleton projection.

Note on QA: DSI Studio scales QA within each subject to that subject's maximum
SDF peak, so QA here is the same subject-normalized quantity used by DSI Studio
connectometry; it is unitless and comparable across subjects only in that sense.

Outputs -> derivatives/technical_validation/dwi/
    dti_wm_metrics.csv     per-subject QA/FA/MD/RD, WM voxel count, QSDR R2
    dti_age_effects.csv    partial correlations with age (control sex)
"""
from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

QSDR_DIR = C.DERIV_DIR / "connectometry" / "qsdr_fib"

# White-matter mask threshold. FA > 0.2 admits ~910 cm3 per participant, about
# twice adult WM volume (~450-500 cm3), so it is heavily contaminated with gray
# matter and partial-volume voxels and drags FA down. FA > 0.3 yields ~555 cm3,
# an anatomically plausible WM compartment. Averaging FA inside an FA-thresholded
# mask is upward-biased by construction; the threshold sweep below documents the
# dependence, and an anatomical (FreeSurfer wmparc) mask would remove it.
FA_THR = 0.3
FA_SWEEP = (0.20, 0.25, 0.30, 0.35)


def _decode(mat, key):
    """Un-scale a byte-packed DSI Studio volume back to native units."""
    v = np.asarray(mat[key], dtype=np.float32).ravel()
    slope = float(np.asarray(mat[f"{key}.slope"]).ravel()[0])
    inter = float(np.asarray(mat[f"{key}.inter"]).ravel()[0])
    return v * slope + inter


def wm_metrics(fib_path: Path) -> dict | None:
    with gzip.open(fib_path, "rb") as fh:
        mat = sio.loadmat(io.BytesIO(fh.read()))
    if "dti_fa" not in mat:
        return None
    fa = _decode(mat, "dti_fa")
    wm = fa > FA_THR
    if wm.sum() == 0:
        return None
    vox_cc = float(np.prod(np.asarray(mat["voxel_size"]).ravel())) / 1000.0
    rec = {
        "wm_mean_FA": float(fa[wm].mean()),
        "wm_voxels": int(wm.sum()),
        "wm_volume_cc": float(wm.sum() * vox_cc),
        "fa_threshold": FA_THR,
        "qsdr_R2": float(np.asarray(mat["R2"]).ravel()[0]),
    }
    for key, out in [("fa0", "wm_mean_QA"), ("md", "wm_mean_MD"), ("rd", "wm_mean_RD")]:
        if key in mat:
            rec[out] = float(_decode(mat, key)[wm].mean())
    # threshold sensitivity, so the mask choice is documented rather than assumed
    for t in FA_SWEEP:
        w = fa > t
        if w.sum() == 0:
            continue
        rec[f"FA_t{int(t * 100)}"] = float(fa[w].mean())
        rec[f"vol_cc_t{int(t * 100)}"] = float(w.sum() * vox_cc)
        if "md" in mat:
            rec[f"MD_t{int(t * 100)}"] = float(_decode(mat, "md")[w].mean())
    return rec


def main():
    rows = []
    for fib in sorted(QSDR_DIR.glob("sub-*_desc-qsdr_dwi.fib.gz")):
        sub = fib.name.split("_")[0]
        rec = wm_metrics(fib)
        if rec is None:
            print(f"  ! {sub}: no dti_fa in FIB, skipped")
            continue
        rows.append({"participant_id": sub, **rec})
        print(f"  {sub}  QA={rec.get('wm_mean_QA', float('nan')):.3f} "
              f"FA={rec['wm_mean_FA']:.3f} MD={rec.get('wm_mean_MD', float('nan')):.3f} "
              f"R2={rec['qsdr_R2']:.2f}")
    df = pd.DataFrame(rows)
    out = C.outdir("dwi")
    df.to_csv(out / "dti_wm_metrics.csv", index=False)
    print(f"\nwrote {out / 'dti_wm_metrics.csv'}  (n={len(df)})")

    # ---- age effects (older cohort, control sex) ----
    pheno = C.load_phenotype()
    m = df.merge(pheno.reset_index(drop=True), on="participant_id", how="left")
    m = m[pd.to_numeric(m["age"], errors="coerce") >= 60]
    covars = [c for c in ["sex_code"] if c in m.columns]
    eff = []
    for metric in ["wm_mean_QA", "wm_mean_FA", "wm_mean_MD", "wm_mean_RD"]:
        if metric not in m:
            continue
        r = C.partial_corr(m, metric, "age", covars)
        eff.append({"metric": metric, "covariates": "+".join(covars), **r})
        print(f"  {metric}: r = {r['r']:+.3f}, p = {r['p']:.4f}, n = {r['n']}")
    pd.DataFrame(eff).to_csv(out / "dti_age_effects.csv", index=False)
    print(f"wrote {out / 'dti_age_effects.csv'}")


if __name__ == "__main__":
    main()
