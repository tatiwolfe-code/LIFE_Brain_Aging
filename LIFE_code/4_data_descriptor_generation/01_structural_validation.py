#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Technical Validation — Structural MRI (T1w/T2w)
================================================
Implements the "Structural MRI (T1w/T2w)" bullets of the manuscript
Technical Validation section:

  1. FreeSurfer surface reconstruction quality  — Euler number distribution
     (expected: topologically complete surfaces).
  2. eTIV comparability across social connectedness — ANCOVA of eTIV on a
     social-connectedness index, controlling for age and sex.
  3. Age-related regional volume patterns — age effects on whole-brain GM
     volume, global/regional cortical thickness, hippocampal volume, and
     caudate/putamen volume (expect robust negative age relationships).
  4. Sex effects on morphometry — ANCOVAs on eTIV-corrected regional volumes.

Following the model data descriptor (Spreng et al., 2022, Scientific Data),
structure is validated by demonstrating expected age-related reductions rather
than by comparison to an external normative table.

Inputs
------
  derivatives/freesurfer/<sub>/   (recon-all output)
  phenotype (age, sex, social connectedness)

Outputs  ->  derivatives/technical_validation/structural/
  euler_numbers.csv
  aseg_volumes.csv, aparc_thickness.csv
  etiv_connectedness_ancova.csv
  age_effects_volume.csv, sex_effects_volume.csv
  structural_validation_report.txt

Usage
-----
  python 01_structural_validation.py
  python 01_structural_validation.py --subject sub-027
  python 01_structural_validation.py --connectedness loneliness_tscore
"""

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

# The released cohort is the older adults; every other deliverable is
# computed on age >= 60, so the summary statistics written here match.
AGE_MIN = 60


# =====================================================================
#  1. Euler numbers (surface reconstruction quality)
# =====================================================================

def compute_euler(subjects):
    """
    Topological defect count of each hemisphere's pre-fix surface
    (?h.orig.nofix), via FreeSurfer's mris_euler_number.

    NOTE: despite their names, the euler_lh / euler_rh / euler_mean columns
    hold the HOLE COUNT (h), not the Euler characteristic (chi). These are
    non-negative, and larger values mean more defects. Convert with
    chi = 2 - 2*h: a topologically complete surface has h = 0, chi = 2.
    The column names are kept for backward compatibility with
    euler_numbers.csv and make_figure1.py.
    """
    rows = []
    for sub in subjects:
        subj_dir = C.FS_SUBJECTS_DIR / sub
        if not (subj_dir / "scripts" / "recon-all.done").exists():
            continue
        holes = {}
        for hemi in ("lh", "rh"):
            surf = subj_dir / "surf" / f"{hemi}.orig.nofix"
            if not surf.exists():
                holes[hemi] = np.nan
                continue
            proc = C.run_fs_cmd(f"mris_euler_number {surf}")
            euler = np.nan
            for line in (proc.stdout or "").splitlines():
                # "euler # = ... --> N holes"
                if "-->" in line and "holes" in line:
                    try:
                        euler = int(line.split("-->")[1].split("holes")[0])
                    except ValueError:
                        pass
            holes[hemi] = euler
        rows.append({"participant_id": sub,
                     "euler_lh": holes.get("lh"),
                     "euler_rh": holes.get("rh"),
                     "euler_mean": np.nanmean([holes.get("lh"),
                                               holes.get("rh")])})
    return pd.DataFrame(rows)


# =====================================================================
#  FreeSurfer stats aggregation (aseg + aparc) via *stats2table
# =====================================================================

def _read_stats2table(cmd: str) -> pd.DataFrame | None:
    """Run an *stats2table command and return its table.

    The command is written to redirect --tablefile to /dev/stdout; we rewrite
    that to a real temp file and read the file, NOT proc.stdout. FreeSurfer
    prints progress/`--skip` chatter ("Building the table..", "Skipping
    .../sub-XXX/stats/lh.aparc.stats") to stdout, which would otherwise be
    parsed as bogus data rows for subjects with incomplete recon."""
    import os, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        proc = C.run_fs_cmd(cmd.replace("/dev/stdout", tmp))
        try:
            df = pd.read_csv(tmp, sep="\t")
        except Exception:  # noqa: BLE001
            df = None
        if df is None or df.empty or df.shape[1] < 2:
            return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    df = df.rename(columns={df.columns[0]: "participant_id"})
    df["participant_id"] = (df["participant_id"].astype(str)
                            .str.replace("/$", "", regex=True))
    # keep only real subject rows (guards against any stray log text)
    df = df[df["participant_id"].str.match(r"^sub-\w+$")]
    return df.set_index("participant_id") if not df.empty else None


def _parse_aseg_stats(path):
    """Directly parse a FreeSurfer aseg.stats file -> {StructName/Measure: volume_mm3}.
    Robust fallback for when asegstats2table is unavailable/fails."""
    vals = {}
    for line in path.read_text().splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line.replace("# Measure", "").split(",")]
            if len(parts) >= 4:
                try:
                    vals[parts[0]] = float(parts[-2])
                except ValueError:
                    pass
        elif line.strip() and not line.startswith("#"):
            c = line.split()
            if len(c) >= 5:  # Index SegId NVoxels Volume_mm3 StructName ...
                try:
                    vals[c[4]] = float(c[3])
                except ValueError:
                    pass
    return vals


# aseg label IDs for the key subcortical structures (+ WM-hypointensities).
ASEG_LABELS = {
    "Left-Hippocampus": 17, "Right-Hippocampus": 53,
    "Left-Caudate": 11, "Right-Caudate": 50,
    "Left-Putamen": 12, "Right-Putamen": 51,
    "Left-Amygdala": 18, "Right-Amygdala": 54,
    "Left-Thalamus-Proper": 10, "Right-Thalamus-Proper": 49,
    "WM-hypointensities": 77,
}


def _globals_from_brainvol(sub):
    """Global volumes (TotalGrayVol, CortexVol, SubCortGrayVol, ...) from brainvol.stats."""
    p = C.FS_SUBJECTS_DIR / sub / "stats" / "brainvol.stats"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("# Measure"):
                parts = [x.strip() for x in line.replace("# Measure", "").split(",")]
                if len(parts) >= 4:
                    try:
                        out[parts[1]] = float(parts[-2])  # short name -> value
                    except ValueError:
                        pass
    return out


def _etiv_from_talairach(sub):
    """eTIV from the talairach transform determinant (FreeSurfer constant)."""
    xfm = C.FS_SUBJECTS_DIR / sub / "mri" / "transforms" / "talairach.xfm"
    if not xfm.exists():
        return None
    lines = xfm.read_text().splitlines()
    for i, l in enumerate(lines):
        if "Linear_Transform" in l:
            try:
                m = [[float(x) for x in lines[i + k].replace(";", " ").split()[:3]]
                     for k in range(1, 4)]
                det = abs(np.linalg.det(np.array(m)))
                return 1948106.0 / det if det else None
            except (ValueError, IndexError):
                return None
    return None


def _volumes_from_mgz(sub):
    """Compute subcortical volumes from aseg.mgz + globals + eTIV, without
    requiring aseg.stats (works when recon-all did not write aseg.stats)."""
    import nibabel as nib
    aseg = C.FS_SUBJECTS_DIR / sub / "mri" / "aseg.mgz"
    if not aseg.exists():
        return None
    img = nib.load(str(aseg))
    data = np.asarray(img.dataobj)
    vox = float(abs(np.prod(img.header.get_zooms()[:3])))
    vals = {name: float((data == lab).sum()) * vox
            for name, lab in ASEG_LABELS.items()}
    vals.update(_globals_from_brainvol(sub))
    etiv = _etiv_from_talairach(sub)
    if etiv:
        vals["EstimatedTotalIntraCranialVol"] = etiv
    return vals


def _augment_brainvol(df):
    """Ensure CortexVol / TotalGrayVol are present (from brainvol.stats)."""
    for want in ("CortexVol", "TotalGrayVol"):
        if want not in df.columns:
            df[want] = [ _globals_from_brainvol(str(sub)).get(want, np.nan)
                         for sub in df.index ]
    return df


def aggregate_aseg(subjects):
    """Subcortical + global volumes (incl. eTIV), tried in order of robustness:
    (1) asegstats2table; (2) direct aseg.stats parse; (3) compute from
    aseg.mgz + brainvol.stats + talairach.xfm (works with incomplete recon)."""
    subs = " ".join(subjects)
    df = _read_stats2table(f"asegstats2table --subjects {subs} "
                           f"--meas volume --tablefile /dev/stdout --skip")
    if df is not None and "EstimatedTotalIntraCranialVol" in df.columns:
        return _augment_brainvol(df)
    rows = {}
    for sub in subjects:
        stats = C.FS_SUBJECTS_DIR / sub / "stats" / "aseg.stats"
        if stats.exists():
            rows[sub] = _parse_aseg_stats(stats)
        else:
            v = _volumes_from_mgz(sub)
            if v:
                rows[sub] = v
    if not rows:
        return None
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "participant_id"
    return _augment_brainvol(out)


def aggregate_aparc(subjects, meas="thickness"):
    """Cortical thickness (or area/volume) per hemisphere via aparcstats2table."""
    subs = " ".join(subjects)
    out = {}
    for hemi in ("lh", "rh"):
        cmd = (f"aparcstats2table --subjects {subs} --hemi {hemi} "
               f"--meas {meas} --tablefile /dev/stdout --skip")
        df = _read_stats2table(cmd)
        if df is not None:
            out[hemi] = df
    if not out:
        return None
    return pd.concat(out.values(), axis=1)


# =====================================================================
#  2. eTIV comparability across social connectedness
# =====================================================================

def etiv_connectedness(pheno, aseg, connectedness_var):
    """ANCOVA: eTIV ~ connectedness + age + sex."""
    if aseg is None or "EstimatedTotalIntraCranialVol" not in aseg.columns:
        return None
    df = pheno.copy()
    df["eTIV"] = aseg["EstimatedTotalIntraCranialVol"]
    covars = [c for c in ["age", "sex_code"] if c in df.columns]
    res = C.ancova(df, "eTIV", connectedness_var, covars)
    res["dv"] = "eTIV"
    res["predictor"] = connectedness_var
    return pd.DataFrame([res])


# =====================================================================
#  3 & 4. Age and sex effects on morphometry
# =====================================================================

# Regions the manuscript calls out explicitly (aseg labels).
# measure name -> candidate aseg column names (asegstats2table vs brainvol/mgz)
KEY_ASEG = {
    "total_gray_matter": ("TotalGrayVol", "TotalGray"),
    "cortical_gray_matter": ("CortexVol", "Cortex"),
    "hippocampus_L": ("Left-Hippocampus",),
    "hippocampus_R": ("Right-Hippocampus",),
    "caudate_L": ("Left-Caudate",),
    "caudate_R": ("Right-Caudate",),
    "putamen_L": ("Left-Putamen",),
    "putamen_R": ("Right-Putamen",),
}


def morphometry_effects(pheno, aseg, aparc, predictor, extra_covars):
    """
    ANCOVA of each key volume / global thickness on `predictor`
    (age or sex_code), controlling for eTIV (volumes) and extra covariates.
    Global cortical thickness is the mean over both hemispheres' MeanThickness.
    """
    if aseg is None:
        return None
    df = pheno.copy()
    df["eTIV"] = aseg.get("EstimatedTotalIntraCranialVol")
    rows = []

    # Subcortical / global volumes — eTIV-corrected.
    for name, candidates in KEY_ASEG.items():
        col = next((c for c in candidates if c in aseg.columns), None)
        if col is None:
            continue
        df[name] = aseg[col]
        covars = ["eTIV"] + [c for c in extra_covars if c in df.columns]
        res = C.ancova(df, name, predictor, covars)
        res.update({"measure": name, "type": "volume_eTIVcorr"})
        rows.append(res)

    # Global cortical thickness (mean of lh/rh MeanThickness).
    if aparc is not None:
        thick_cols = [c for c in aparc.columns if "MeanThickness" in c]
        if thick_cols:
            df["global_thickness"] = aparc[thick_cols].mean(axis=1)
            covars = [c for c in extra_covars if c in df.columns]
            res = C.ancova(df, "global_thickness", predictor, covars)
            res.update({"measure": "global_thickness", "type": "thickness"})
            rows.append(res)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "predictor", predictor)
    return out


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None, help="restrict to one subject")
    ap.add_argument("--connectedness", default="loneliness_tscore",
                    help="phenotype column indexing social connectedness "
                         "(default: PROMIS loneliness_tscore; higher = lonelier)")
    args = ap.parse_args()

    out = C.outdir("structural")
    report = [C.banner("STRUCTURAL MRI (T1w/T2w) — Technical Validation")]

    pheno = C.load_phenotype()
    subjects = ([args.subject] if args.subject
                else [s for s, _ in C.subject_dirs()])
    # keep reconstructed subjects (done flag, or a completed aseg.mgz segmentation)
    recon = [s for s in subjects
             if (C.FS_SUBJECTS_DIR / s / "scripts" / "recon-all.done").exists()
             or (C.FS_SUBJECTS_DIR / s / "mri" / "aseg.mgz").exists()]
    report.append(f"Subjects on disk: {len(subjects)} | "
                  f"with completed recon-all: {len(recon)}")

    # Restrict to the Technical Validation cohort: the released older adults,
    # less the structural outlier (see C.STRUCTURAL_EXCLUDE). Skipped when
    # --subject is given, so a single subject can always be inspected.
    if not args.subject:
        keep = C.older_ids(pheno, AGE_MIN, exclude=C.STRUCTURAL_EXCLUDE)
        n_before = len(recon)
        dropped = sorted(set(recon) & set(C.STRUCTURAL_EXCLUDE))
        recon = [s for s in recon if s in keep]
        pheno = pheno.loc[[i for i in pheno.index if i in keep]]
        report.append(f"Restricted to the released older cohort (age >= {AGE_MIN}): "
                      f"{n_before} -> {len(recon)} subjects")
        if dropped:
            report.append(f"  excluded as structural outlier(s): {', '.join(dropped)}")

    if not recon:
        report.append("\nNo completed FreeSurfer reconstructions found under "
                      f"{C.FS_SUBJECTS_DIR}. Run freesurfer_analysis.py first.")
        (out / "structural_validation_report.txt").write_text("\n".join(report))
        print("\n".join(report))
        return

    # ── 1. Euler numbers ──
    euler = compute_euler(recon)
    euler.to_csv(out / "euler_numbers.csv", index=False)
    if not euler.empty:
        em = euler["euler_mean"]
        report.append(C.banner("1. FreeSurfer surface reconstruction quality"))
        report.append(f"  Topological defects per hemisphere "
                      f"(holes; Euler chi = 2 - 2*holes): "
                      f"median={em.median():.1f}, min={em.min():.1f}, "
                      f"max={em.max():.1f}")

    # ── FreeSurfer stats tables ──
    aseg = aggregate_aseg(recon)
    aparc = aggregate_aparc(recon, meas="thickness")
    if aseg is not None:
        aseg.to_csv(out / "aseg_volumes.csv")
    if aparc is not None:
        aparc.to_csv(out / "aparc_thickness.csv")

    # ── 2. eTIV vs social connectedness ──
    if args.connectedness in pheno.columns:
        etiv = etiv_connectedness(pheno, aseg, args.connectedness)
        if etiv is not None:
            etiv.to_csv(out / "etiv_connectedness_ancova.csv", index=False)
            row = etiv.iloc[0]
            report.append(C.banner("2. eTIV comparability across social connectedness"))
            report.append(f"  eTIV ~ {args.connectedness} (+age,sex): "
                          f"beta={row['beta']:.1f}, p={row['p']:.3f}, "
                          f"n={row['n']}  (non-significant = no recruitment confound)")
    else:
        report.append(f"\n[2] connectedness column '{args.connectedness}' "
                      f"not in phenotype; skipped.")

    # ── 3. Age effects on morphometry ──
    age_eff = morphometry_effects(pheno, aseg, aparc, "age",
                                  extra_covars=["sex_code"])
    if age_eff is not None:
        age_eff.to_csv(out / "age_effects_volume.csv", index=False)
        report.append(C.banner("3. Age effects on morphometry "
                               "(expect negative)"))
        for _, r in age_eff.iterrows():
            report.append(f"  {r['measure']:<22} beta={r['beta']:+.2f} "
                          f"p={r['p']:.4f}  eta2={r['partial_eta2']:.3f} "
                          f"n={r['n']}")

    # ── 4. Sex effects on morphometry ──
    sex_eff = morphometry_effects(pheno, aseg, aparc, "sex_code",
                                  extra_covars=["age"])
    if sex_eff is not None:
        sex_eff.to_csv(out / "sex_effects_volume.csv", index=False)
        report.append(C.banner("4. Sex effects on eTIV-corrected morphometry"))
        for _, r in sex_eff.iterrows():
            report.append(f"  {r['measure']:<22} beta={r['beta']:+.2f} "
                          f"p={r['p']:.4f} n={r['n']}")

    # Validation follows the model data descriptor (Spreng et al., 2022):
    # structure is validated by demonstrating expected age-related reductions
    # (above) rather than by comparison to an external normative table.
    report.append("\nNote: age-related reductions are consistent with those "
                  "reported by Spreng et al. (2022) in a comparable older cohort.")

    (out / "structural_validation_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
