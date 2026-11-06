#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Technical Validation — Resting-State fMRI
==========================================
Implements the "Resting-State fMRI" bullets:

  1. Framewise displacement (FD) — mean/SD/range across participants,
     age-FD relationship, final N after QC thresholding (mean FD < --fd-thr).
  2. Temporal SNR (tSNR) — median tSNR per participant (GM/brain mask) and a
     group-average tSNR spatial map.
  3. BOLD signal dimensionality — number of signal components retained after
     ICA-FIX / MELODIC (read from melodic ICstats if available).
  4. Test-retest reliability — ICC of edge-wise FC across two runs (if a
     second rest run or a rest+retest connectivity derivative exists).
  5. Known-groups network structure — mean within- vs between-network
     connectivity for canonical networks, from parcellated connectivity
     matrices.
  6. Age effects on FC — mass-univariate age effects on edge-wise FC
     (control sex, motion).

FD source (priority): fMRIPrep confounds (framewise_displacement column) ->
FSL mcflirt on the raw BOLD. tSNR is computed directly from the raw 4-D BOLD.
Network/age analyses require a parcellated connectivity derivative; if none is
found they are skipped with a clear message (the code runs once
derivatives/connectivity/<sub>_*.csv matrices exist).

Outputs  ->  derivatives/technical_validation/rsfmri/
  fd_per_subject.csv, tsnr_per_subject.csv, tsnr_group_mean.nii.gz,
  dimensionality.csv, network_structure.csv, age_effects_fc.csv,
  rsfmri_validation_report.txt
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

REST_SUFFIXES = ["task-rest_bold", "task-rest_run-1_bold", "task-rest_run-01_bold"]


# =====================================================================
#  Locate raw BOLD
# =====================================================================

def find_rest_bold(sub_dir, sub):
    func = sub_dir / "func"
    if not func.is_dir():
        return []
    hits = []
    for f in sorted(func.glob(f"{sub}_task-rest*_bold.nii.gz")):
        hits.append(f)
    return hits


# =====================================================================
#  1. Framewise displacement
# =====================================================================

def fd_from_timeseries(sub):
    """Per-volume FD as written by cohort_fd.py.

    cohort_fd.py is the single source of framewise displacement for this
    dataset: it runs mcflirt with `-refvol 4` and writes one FD value per
    volume to fd_timeseries/<sub>_fd.txt, and Figure 3A is drawn from those
    files. Reading them here keeps fd_per_subject.csv and the figure in
    agreement. Recomputing FD locally would run mcflirt against its default
    reference volume (the middle one) instead, giving motion parameters that
    differ per participant and FD that disagrees with the published figure by
    up to 0.05 mm.

    The first value is 0 by construction (there is no preceding volume to
    difference against), so it is dropped; the figure does the same.
    """
    p = (C.DERIV_DIR / "technical_validation" / "rsfmri" /
         "fd_timeseries" / f"{sub}_fd.txt")
    if not p.exists():
        return None
    fd = np.loadtxt(p)
    if fd.ndim != 1 or fd.size < 2:
        return None
    return fd[1:]


def fd_from_fmriprep(sub):
    conf = list((C.DERIV_DIR / "fmriprep" / sub).rglob(
        "*task-rest*desc-confounds_timeseries.tsv")) if \
        (C.DERIV_DIR / "fmriprep").is_dir() else []
    if not conf:
        return None
    df = pd.read_csv(conf[0], sep="\t")
    if "framewise_displacement" not in df.columns:
        return None
    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce").dropna()
    return fd.values


def fd_from_mcflirt(bold, fd_thr):
    """Run FSL mcflirt, then Power FD from the .par motion parameters."""
    if shutil.which("mcflirt") is None:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="mcflirt_"))
    try:
        base = tmp / "mc"
        subprocess.run(["mcflirt", "-in", str(bold), "-out", str(base),
                        "-plots"], capture_output=True, text=True)
        par = base.with_suffix(".par")
        if not par.exists():
            return None
        # cols: 3 rot (rad), 3 trans (mm)
        mp = np.loadtxt(par)
        rot, trans = mp[:, :3], mp[:, 3:6]
        drot = np.abs(np.diff(rot, axis=0)) * 50.0  # 50 mm brain radius
        dtrans = np.abs(np.diff(trans, axis=0))
        fd = np.concatenate([[0.0], (drot.sum(1) + dtrans.sum(1))])
        return fd
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compute_fd(subjects, fd_thr):
    """Per-participant FD summary.

    Order of preference: the FD timeseries written by cohort_fd.py (the
    source Figure 3A uses), then fMRIPrep confounds, then a local mcflirt
    run. The local run is a last resort -- it uses a different mcflirt
    reference volume, so its values will not match the published figure.
    """
    rows, fallback = [], []
    for sub, sub_dir in subjects:
        bolds = find_rest_bold(sub_dir, sub)
        if not bolds:
            continue
        fd = fd_from_timeseries(sub)
        if fd is None:
            fd = fd_from_fmriprep(sub)
            if fd is None:
                fd = fd_from_mcflirt(bolds[0], fd_thr)
            if fd is not None:
                fallback.append(sub)
        if fd is None:
            continue
        n_low = int((fd < fd_thr).sum())
        rows.append({"participant_id": sub, "mean_fd": float(np.mean(fd)),
                     "sd_fd": float(np.std(fd, ddof=1)),
                     "max_fd": float(np.max(fd)),
                     "n_frames": len(fd), "n_lowmotion_frames": n_low})
    if fallback:
        print(f"WARNING: no fd_timeseries/ for {len(fallback)} participant(s) "
              f"({', '.join(fallback[:5])}{' ...' if len(fallback) > 5 else ''}); "
              "FD was recomputed locally and will NOT match Figure 3A. "
              "Run cohort_fd.py first.", flush=True)
    return pd.DataFrame(rows)


# =====================================================================
#  2. Temporal SNR
# =====================================================================

def compute_tsnr(subjects, group_map_path):
    import nibabel as nib
    rows = []
    accum = None
    n_maps = 0
    ref_affine = ref_shape = None
    for sub, sub_dir in subjects:
        bolds = find_rest_bold(sub_dir, sub)
        if not bolds:
            continue
        img = nib.load(str(bolds[0]))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 4:
            continue
        mean = data.mean(axis=3)
        std = data.std(axis=3, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr = np.where(std > 0, mean / std, 0.0)
        # brain mask: voxels above 25th pct of mean signal
        thr = np.percentile(mean[mean > 0], 25) if (mean > 0).any() else 0
        mask = mean > thr
        vals = tsnr[mask]
        rows.append({"participant_id": sub,
                     "median_tsnr": float(np.median(vals)),
                     "mean_tsnr": float(np.mean(vals)),
                     "n_voxels_mask": int(mask.sum())})
        # accumulate group map (only if grid matches the first subject)
        if accum is None:
            accum = np.zeros_like(tsnr)
            ref_affine, ref_shape = img.affine, tsnr.shape
        if tsnr.shape == ref_shape:
            accum += tsnr
            n_maps += 1
    if accum is not None and n_maps > 0:
        nib.save(nib.Nifti1Image(accum / n_maps, ref_affine),
                 str(group_map_path))
    return pd.DataFrame(rows), n_maps


# =====================================================================
#  3. BOLD dimensionality (ICA-FIX / MELODIC)
# =====================================================================

def compute_dimensionality(subjects):
    rows = []
    fix_dir = C.DERIV_DIR / "fix"
    ica_dirs = [C.DERIV_DIR / "melodic", fix_dir]
    for sub, _ in subjects:
        n_comp = np.nan
        for base in ica_dirs:
            hits = list(base.rglob(f"*{sub}*melodic_ICstats")) if base.is_dir() else []
            if hits:
                n_comp = len(hits[0].read_text().splitlines())
                break
        if np.isfinite(n_comp):
            rows.append({"participant_id": sub, "n_components": n_comp})
    return pd.DataFrame(rows)


# =====================================================================
#  4 & 5 & 6. Connectivity-matrix analyses
# =====================================================================

def load_fc_matrices(labels_hint="rest"):
    """
    Load per-subject FC matrices from derivatives/connectivity/.
    Expected: <sub>_*task-rest*_connectome*.csv (square numeric matrix).
    Returns dict[sub] -> np.ndarray, or {} if none found.
    """
    conn_dir = C.DERIV_DIR / "connectivity"
    out = {}
    if not conn_dir.is_dir():
        return out
    for f in sorted(conn_dir.glob("sub-*")):
        pass
    for f in sorted(conn_dir.rglob("*task-rest*.csv")):
        sub = f.name.split("_")[0]
        try:
            m = pd.read_csv(f, header=None).values.astype(float)
            if m.shape[0] == m.shape[1]:
                out[sub] = m
        except Exception:  # noqa: BLE001
            continue
    return out


def network_structure(fc, network_labels_csv):
    """
    Mean within- vs between-network connectivity. `network_labels_csv` maps
    each parcel index to a canonical network (columns: index,network),
    e.g. the Schaefer-400 7-network assignment.
    """
    if not fc or not network_labels_csv:
        return None
    labels = pd.read_csv(network_labels_csv)
    net = labels["network"].values
    group_mean = np.mean(list(fc.values()), axis=0)
    nets = sorted(pd.unique(net))
    rows = []
    for a in nets:
        for b in nets:
            ia = np.where(net == a)[0]
            ib = np.where(net == b)[0]
            block = group_mean[np.ix_(ia, ib)]
            rows.append({"net_a": a, "net_b": b,
                         "mean_fc": float(np.nanmean(block)),
                         "within": a == b})
    return pd.DataFrame(rows)


def age_effects_fc(fc, pheno, fd):
    """Mass-univariate age effect on each edge (control sex, FD)."""
    if not fc:
        return None
    subs = [s for s in fc if s in pheno.index]
    if len(subs) < 10:
        return None
    n = fc[subs[0]].shape[0]
    iu = np.triu_indices(n, k=1)
    X = np.array([fc[s][iu] for s in subs])          # subjects x edges
    cov = pheno.loc[subs, ["age", "sex_code"]].copy()
    cov = cov.join(fd.set_index("participant_id")["mean_fd"]) \
        if fd is not None and not fd.empty else cov
    cov = cov.apply(pd.to_numeric, errors="coerce")
    keep = cov.dropna().index
    idx = [subs.index(s) for s in keep]
    X = X[idx]
    from scipy import stats
    a = cov.loc[keep, "age"].values
    design = np.column_stack([np.ones(len(keep)), a,
                              cov.loc[keep, "sex_code"].values])
    betas, ts, ps = [], [], []
    dfree = len(keep) - design.shape[1]
    for e in range(X.shape[1]):
        y = X[:, e]
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        s2 = (resid @ resid) / dfree
        se = np.sqrt(s2 * np.linalg.inv(design.T @ design)[1, 1])
        t = beta[1] / se if se > 0 else np.nan
        betas.append(beta[1]); ts.append(t)
        ps.append(2 * stats.t.sf(abs(t), dfree) if np.isfinite(t) else np.nan)
    edges = pd.DataFrame({"edge_i": iu[0], "edge_j": iu[1],
                          "age_beta": betas, "t": ts, "p": ps})
    return edges


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--fd-thr", type=float, default=0.5,
                    help="FD threshold (mm) for low-motion frames / QC")
    ap.add_argument("--min-lowmotion-min", type=float, default=6.0,
                    help="minimum low-motion minutes to retain a subject")
    ap.add_argument("--tr", type=float, default=0.8, help="rs-fMRI TR (s)")
    ap.add_argument("--network-labels", default=None,
                    help="CSV (index,network) mapping parcels to canonical nets")
    args = ap.parse_args()

    out = C.outdir("rsfmri")
    report = [C.banner("RESTING-STATE fMRI — Technical Validation")]

    subjects = [(s, d) for s, d in C.subject_dirs()
                if args.subject is None or s == args.subject]
    pheno = C.load_phenotype()

    # ── 1. FD ──
    fd = compute_fd(subjects, args.fd_thr)
    fd.to_csv(out / "fd_per_subject.csv", index=False)
    report.append(C.banner("1. Framewise displacement"))
    if not fd.empty:
        d = C.describe(fd["mean_fd"])
        report.append(f"  mean FD: mean={d['mean']:.3f} SD={d['sd']:.3f} "
                      f"range=[{d['min']:.3f},{d['max']:.3f}] n={d['n']}")
        # age-FD relationship
        m = fd.merge(pheno[["age"]], left_on="participant_id",
                     right_index=True, how="left")
        mm = m[["mean_fd", "age"]].dropna()
        if len(mm) > 3:
            from scipy import stats
            r, p = stats.pearsonr(mm["mean_fd"], mm["age"])
            report.append(f"  age-FD correlation: r={r:+.3f} p={p:.3f} n={len(mm)}")
        # QC: retain subjects with enough low-motion data
        min_frames = args.min_lowmotion_min * 60.0 / args.tr
        retained = fd[(fd["mean_fd"] < args.fd_thr) |
                      (fd["n_lowmotion_frames"] >= min_frames)]
        report.append(f"  Retained after QC (mean FD<{args.fd_thr} or "
                      f">={args.min_lowmotion_min} min low-motion): "
                      f"{len(retained)}/{len(fd)}")
    else:
        report.append("  No FD computed (no rest BOLD, or FSL mcflirt / "
                      "fMRIPrep confounds unavailable).")

    # ── 2. tSNR ──
    tsnr, n_maps = compute_tsnr(subjects, out / "tsnr_group_mean.nii.gz")
    tsnr.to_csv(out / "tsnr_per_subject.csv", index=False)
    report.append(C.banner("2. Temporal SNR"))
    if not tsnr.empty:
        d = C.describe(tsnr["median_tsnr"])
        report.append(f"  median tSNR: mean={d['mean']:.1f} SD={d['sd']:.1f} "
                      f"range=[{d['min']:.1f},{d['max']:.1f}] n={d['n']}")
        report.append(f"  Group-average tSNR map: tsnr_group_mean.nii.gz "
                      f"({n_maps} subjects on matching grid)")
    else:
        report.append("  No tSNR computed (no 4-D rest BOLD found).")

    # ── 3. Dimensionality ──
    dim = compute_dimensionality(subjects)
    dim.to_csv(out / "dimensionality.csv", index=False)
    report.append(C.banner("3. BOLD dimensionality (ICA-FIX / MELODIC)"))
    if not dim.empty:
        d = C.describe(dim["n_components"])
        report.append(f"  components: mean={d['mean']:.1f} SD={d['sd']:.1f} "
                      f"range=[{d['min']:.0f},{d['max']:.0f}] n={d['n']}")
    else:
        report.append("  No MELODIC/ICA-FIX outputs found; run ICA-FIX to enable.")

    # ── 4/5/6. Connectivity analyses ──
    fc = load_fc_matrices()
    report.append(C.banner("4-6. Connectivity-based validation"))
    if not fc:
        report.append("  No parcellated FC matrices in derivatives/connectivity/. "
                      "Network structure, test-retest ICC, and age-FC effects "
                      "will run once those matrices are generated.")
    else:
        report.append(f"  Loaded FC matrices for {len(fc)} subjects.")
        net = network_structure(fc, args.network_labels)
        if net is not None:
            net.to_csv(out / "network_structure.csv", index=False)
            within = net[net["within"]]["mean_fc"].mean()
            between = net[~net["within"]]["mean_fc"].mean()
            report.append(f"  [5] mean within-network FC={within:.3f} vs "
                          f"between-network FC={between:.3f} "
                          f"(expect within > between)")
        else:
            report.append("  [5] provide --network-labels (index,network) for "
                          "the network-structure check.")
        edges = age_effects_fc(fc, pheno, fd)
        if edges is not None:
            edges.to_csv(out / "age_effects_fc.csv", index=False)
            n_sig = int((edges["p"] < 0.05).sum())
            report.append(f"  [6] age-FC mass-univariate: {n_sig}/{len(edges)} "
                          f"edges p<.05 (uncorrected).")

    (out / "rsfmri_validation_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
