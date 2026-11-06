#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Whole-white-matter DTI metrics (FA, MD, AD, RD) + age effects, reconstructed
uniformly from the released DSI Studio SRC files.

The pre-computed .fib.gz in this dataset are in inconsistent formats (native
FA 0-1 vs byte-scaled 0-255), so DTI is instead reconstructed fresh from each
subject's .src.gz (DSI Studio --action=rec --method=1), exported to NIfTI, and
averaged within a WM mask (FA > 0.2). MD/AD/RD are in 10^-3 mm^2/s. This is an
approximate whole-WM average, not a TBSS skeleton.

Outputs -> derivatives/technical_validation/dwi/
  dti_wm_metrics.csv, dti_age_effects.csv

Requires DSI Studio (set DSI_STUDIO, else probes /Applications).
Usage:
  python 13_dti_from_src.py              # use the released .src.gz
  python 13_dti_from_src.py --from-raw   # rebuild SRC from raw BIDS DWI first
                                         # (uniform; use if released SRC/FIB are
                                         #  inconsistent across subjects)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

BASE = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
for _c in glob.glob(str(BASE / "**" / "code" / "config.py"), recursive=True):
    sys.path.insert(0, os.path.dirname(_c))
    break
import config as C  # noqa: E402


def find_binary():
    env = os.environ.get("DSI_STUDIO")
    if env and Path(env).exists():
        return env
    for c in ["/Applications/dsi_studio1.app/Contents/MacOS/dsi_studio",
              "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
              "/Applications/dsi_studio_hou.app/Contents/MacOS/dsi_studio"]:
        if Path(c).exists():
            return c
    return "dsi_studio"


def dti_one(ds, src, tmp, fa_thr=0.2):
    out = os.path.join(tmp, "dti")
    subprocess.run([ds, "--action=rec", f"--source={src}", "--method=1",
                    f"--output={out}"], capture_output=True, text=True)
    fib = glob.glob(out + "*.fib.gz")
    if not fib:
        return None
    subprocess.run([ds, "--action=exp", f"--source={fib[0]}",
                    "--export=fa,md,ad,rd"], capture_output=True, text=True)

    def load(metric):
        g = glob.glob(f"{fib[0]}.{metric}.nii.gz")
        return np.asarray(nib.load(g[0]).dataobj, dtype=np.float32) if g else None

    fa = load("fa")
    if fa is None:
        return None
    wm = fa > fa_thr
    if wm.sum() == 0:
        return None
    rec = {"wm_mean_FA": float(fa[wm].mean())}
    for m, dst in [("md", "wm_mean_MD"), ("ad", "wm_mean_AD"), ("rd", "wm_mean_RD")]:
        v = load(m)
        if v is not None:
            rec[dst] = float(v[wm].mean())
    return rec


def build_src(ds, sub, tmp):
    """Build a fresh SRC from the raw BIDS DWI (uniform across subjects)."""
    d = C.BIDS_DIR / sub / "dwi"
    nii, bval, bvec = (d / f"{sub}_dwi.nii.gz", d / f"{sub}_dwi.bval",
                       d / f"{sub}_dwi.bvec")
    if not (nii.exists() and bval.exists() and bvec.exists()):
        return None
    src = os.path.join(tmp, f"{sub}.src.gz")
    subprocess.run([ds, "--action=src", f"--source={nii}", f"--bval={bval}",
                    f"--bvec={bvec}", f"--output={src}"],
                   capture_output=True, text=True)
    return src if os.path.exists(src) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild SRC from raw BIDS DWI (uniform across subjects)")
    args = ap.parse_args()
    ds = find_binary()
    out = C.outdir("dwi")
    rows = []
    subs = ([s for s, _ in C.subject_dirs()] if args.from_raw
            else [d.name for d in sorted(C.DSI_DIR.glob("sub-*"))])
    print(f"mode: {'from-raw' if args.from_raw else 'released-src'} | {len(subs)} subjects")
    for sub in subs:
        tmp = tempfile.mkdtemp(prefix="dti_")
        try:
            if args.from_raw:
                src = build_src(ds, sub, tmp)
            else:
                srcs = list((C.DSI_DIR / sub / "dwi").glob("*_dwi.src.gz"))
                src = str(srcs[0]) if srcs else None
            rec = dti_one(ds, src, tmp) if src else None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if rec:
            rec["participant_id"] = sub
            rows.append(rec)
    dti = pd.DataFrame(rows)
    if dti.empty:
        print("No DTI computed (no SRC files or DSI Studio not found).")
        return
    dti = dti[["participant_id"] + [c for c in dti.columns if c != "participant_id"]]
    dti.to_csv(out / "dti_wm_metrics.csv", index=False)
    print(f"DTI whole-WM metrics (n={len(dti)}):")
    for col in ["wm_mean_FA", "wm_mean_MD", "wm_mean_AD", "wm_mean_RD"]:
        if col in dti:
            d = C.describe(dti[col])
            print(f"  {col}: mean={d['mean']:.3f} SD={d['sd']:.3f}")

    ph = C.load_phenotype()
    m = dti.set_index("participant_id").join(ph)
    covs = [c for c in ["sex_code", "education"] if c in m.columns]
    ar = []
    for col in ["wm_mean_FA", "wm_mean_MD"]:
        r = C.partial_corr(m, col, "age", covs)
        r["metric"] = col
        ar.append(r)
        print(f"  {col} vs age: partial r={r['r']:+.3f} p={r['p']:.3f} n={r['n']}")
    pd.DataFrame(ar).to_csv(out / "dti_age_effects.csv", index=False)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
