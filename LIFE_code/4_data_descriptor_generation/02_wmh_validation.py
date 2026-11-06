#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Technical Validation — T2-FLAIR / White Matter Hyperintensities
================================================================
Implements the "T2-FLAIR / White Matter Hyperintensities" bullets:

  1. WMH lesion volume and count — total lesion volume (mm^3, raw and
     eTIV-corrected) and lesion-count distributions (mean, SD, range;
     expected positively skewed, increasing with age).
  2. WMH-age correlation — Pearson & Spearman of WMH volume/count with age.
  3. WMH-cognition association — partial correlations of WMH with the
     available executive-function measures (DCCS, Flanker), controlling for
     age and sex. Expect negative associations.

Lesion source (auto-detected, in priority order)
------------------------------------------------
  a) LST probability maps  derivatives/lst/<sub>/ples_*.nii(.gz)   (--lst-thr)
  b) SAMSEG lesion         derivatives/samseg/<sub>/samseg.stats  ("Lesions")
  c) FreeSurfer proxy      derivatives/freesurfer/<sub>/stats/aseg.stats
                           ("WM-hypointensities"), used as a fallback proxy
                           and flagged as such in the report.

Lesion *count* (number of connected components) is computed only for voxelwise
sources (LST / SAMSEG seg.mgz); it is left NaN for the aseg.stats proxy.

Outputs  ->  derivatives/technical_validation/wmh/
  wmh_per_subject.csv, wmh_age_correlation.csv,
  wmh_cognition_partialcorr.csv, wmh_validation_report.txt
"""

import argparse
import re

import numpy as np
import pandas as pd

import config as C

# Executive-function measures available in nih_toolbox.tsv (manuscript asks
# for fluid composite / processing speed / executive; only these exist here).
COG_MEASURES = [
    ("dccs_age_corrected_standard_score", "DCCS (cognitive flexibility)"),
    ("flanker_age_corrected_standard_score", "Flanker (inhibition)"),
]


def _voxel_volume_mm3(img) -> float:
    z = img.header.get_zooms()[:3]
    return float(np.prod(z))


def _lesion_from_lst(sub, thr):
    import nibabel as nib
    lst_dir = C.DERIV_DIR / "lst" / sub
    if not lst_dir.is_dir():
        return None
    maps = list(lst_dir.glob("ples_*.nii*"))
    if not maps:
        return None
    img = nib.load(str(maps[0]))
    data = np.asarray(img.dataobj)
    mask = data > thr
    vol = float(mask.sum()) * _voxel_volume_mm3(img)
    count = _connected_count(mask)
    return {"wmh_volume_mm3": vol, "wmh_count": count, "source": "LST"}


def _lesion_from_samseg(sub):
    import nibabel as nib
    stats = C.SAMSEG_DIR / sub / "samseg.stats"
    seg = C.SAMSEG_DIR / sub / "seg.mgz"
    vol = np.nan
    if stats.exists():
        for line in stats.read_text().splitlines():
            # "# Measure Lesions, ..., <volume>, mm^3"
            if "Lesion" in line:
                m = re.search(r"([-\d.]+),\s*mm\^3", line)
                if m:
                    vol = float(m.group(1))
    count = np.nan
    if seg.exists():
        try:
            img = nib.load(str(seg))
            data = np.asarray(img.dataobj)
            # SAMSEG lesion label = 99 (default lesion label)
            mask = data == 99
            if mask.any():
                if not np.isfinite(vol):
                    vol = float(mask.sum()) * _voxel_volume_mm3(img)
                count = _connected_count(mask)
        except Exception:  # noqa: BLE001
            pass
    if not np.isfinite(vol):
        return None
    return {"wmh_volume_mm3": vol, "wmh_count": count, "source": "SAMSEG"}


def _lesion_from_aseg_proxy(sub):
    stats = C.FS_SUBJECTS_DIR / sub / "stats" / "aseg.stats"
    if not stats.exists():
        return None
    for line in stats.read_text().splitlines():
        if "WM-hypointensities" in line and not line.startswith("#"):
            parts = line.split()
            # aseg.stats columns: Index SegId NVoxels Volume_mm3 StructName ...
            try:
                vol = float(parts[3])
                return {"wmh_volume_mm3": vol, "wmh_count": np.nan,
                        "source": "aseg_WMhypo_proxy"}
            except (IndexError, ValueError):
                return None
    return None


def _connected_count(mask):
    """Number of 26-connected components in a boolean 3D mask."""
    try:
        from scipy.ndimage import label
        _, n = label(mask, structure=np.ones((3, 3, 3)))
        return int(n)
    except Exception:  # noqa: BLE001
        return np.nan


def _etiv(sub):
    stats = C.FS_SUBJECTS_DIR / sub / "stats" / "aseg.stats"
    if not stats.exists():
        return np.nan
    for line in stats.read_text().splitlines():
        if "EstimatedTotalIntraCranialVol" in line:
            m = re.search(r"([-\d.]+),\s*mm\^3", line)
            if m:
                return float(m.group(1))
    return np.nan


def collect_wmh(subjects, lst_thr):
    rows = []
    for sub in subjects:
        rec = (_lesion_from_lst(sub, lst_thr)
               or _lesion_from_samseg(sub)
               or _lesion_from_aseg_proxy(sub))
        if rec is None:
            continue
        etiv = _etiv(sub)
        rec["participant_id"] = sub
        rec["eTIV"] = etiv
        rec["wmh_volume_etiv_corr"] = (rec["wmh_volume_mm3"] / etiv * 1e6
                                       if np.isfinite(etiv) and etiv else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--lst-thr", type=float, default=0.5,
                    help="LST probability threshold for lesion voxels")
    args = ap.parse_args()

    out = C.outdir("wmh")
    report = [C.banner("T2-FLAIR / WHITE MATTER HYPERINTENSITIES — "
                       "Technical Validation")]

    pheno = C.load_phenotype()
    subjects = ([args.subject] if args.subject
                else [s for s, _ in C.subject_dirs()])

    wmh = collect_wmh(subjects, args.lst_thr)
    if wmh.empty:
        report.append("\nNo WMH source found (LST / SAMSEG lesion / aseg "
                      "WM-hypointensities). Run SAMSEG with --lesion or LST.")
        (out / "wmh_validation_report.txt").write_text("\n".join(report))
        print("\n".join(report))
        return

    wmh = wmh.set_index("participant_id")

    # Restrict to the Technical Validation cohort: the released older adults,
    # less the structural outlier (see C.STRUCTURAL_EXCLUDE). WMH is derived
    # from FreeSurfer, so it carries the same exclusion as the surface and
    # volume measures. Skipped when --subject is given.
    if not args.subject:
        keep = C.older_ids(pheno, C.AGE_MIN, exclude=C.STRUCTURAL_EXCLUDE)
        n_before = len(wmh)
        dropped = sorted(set(wmh.index) & set(C.STRUCTURAL_EXCLUDE))
        wmh = wmh.loc[[i for i in wmh.index if i in keep]]
        report.append(f"Restricted to the released older cohort "
                      f"(age >= {C.AGE_MIN}): {n_before} -> {len(wmh)} subjects")
        if dropped:
            report.append(f"  excluded as structural outlier(s): {', '.join(dropped)}")

    wmh.to_csv(out / "wmh_per_subject.csv")
    src = wmh["source"].unique().tolist()
    report.append(f"Subjects with WMH data: {len(wmh)} | source(s): {src}")
    if any("proxy" in s for s in src):
        report.append("  NOTE: aseg WM-hypointensities is a coarse proxy; "
                      "prefer SAMSEG --lesion or LST for the final release.")

    # ── 1. Distributions ──
    report.append(C.banner("1. WMH volume & count distributions"))
    for col in ["wmh_volume_mm3", "wmh_volume_etiv_corr", "wmh_count"]:
        if col in wmh and wmh[col].notna().any():
            d = C.describe(wmh[col])
            skew = pd.to_numeric(wmh[col], errors="coerce").dropna().skew()
            report.append(f"  {col:<24} mean={d['mean']:.2f} SD={d['sd']:.2f} "
                          f"range=[{d['min']:.2f},{d['max']:.2f}] "
                          f"skew={skew:+.2f} n={d['n']}")

    # ── 2. WMH-age correlation ──
    from scipy import stats
    df = wmh.join(pheno[["age"]], how="left")
    age_rows = []
    report.append(C.banner("2. WMH-age correlation (expect positive, r>=0.30)"))
    for col in ["wmh_volume_etiv_corr", "wmh_count"]:
        d = df[[col, "age"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 4:
            continue
        pr, pp = stats.pearsonr(d[col], d["age"])
        sr, sp = stats.spearmanr(d[col], d["age"])
        age_rows.append({"measure": col, "pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "spearman_p": sp, "n": len(d)})
        report.append(f"  {col:<24} Pearson r={pr:+.3f} (p={pp:.3f}) | "
                      f"Spearman rho={sr:+.3f} (p={sp:.3f}) n={len(d)}")
    pd.DataFrame(age_rows).to_csv(out / "wmh_age_correlation.csv", index=False)

    # ── 3. WMH-cognition association ──
    report.append(C.banner("3. WMH-cognition partial correlations "
                           "(control age, sex; expect negative)"))
    covars = [c for c in ["age", "sex_code"] if c in pheno.columns]
    dfc = wmh.join(pheno, how="left")
    cog_rows = []
    for col, cog_label in COG_MEASURES:
        if col not in dfc.columns:
            continue
        res = C.partial_corr(dfc, "wmh_volume_etiv_corr", col, covars)
        res.update({"wmh_measure": "wmh_volume_etiv_corr", "cognition": col})
        cog_rows.append(res)
        report.append(f"  WMH vs {cog_label:<26} partial r={res['r']:+.3f} "
                      f"p={res['p']:.3f} n={res['n']}")
    pd.DataFrame(cog_rows).to_csv(out / "wmh_cognition_partialcorr.csv",
                                  index=False)

    (out / "wmh_validation_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
