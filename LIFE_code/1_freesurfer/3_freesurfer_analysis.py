#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Run FreeSurfer structural analysis on the LIFE BIDS dataset.

Per subject:
  1. recon-all  on the T1w      -> derivatives/freesurfer/<sub>/
  2. run_samseg on the FLAIR    -> derivatives/samseg/<sub>/

Single-session study — no ses- labels.

Skipping:
  - Subjects whose T1w (for recon-all) or FLAIR (for samseg) is missing
  - Runs that already completed (recon-all: scripts/recon-all.done;
    samseg: seg.mgz present)

Usage:
    python freesurfer_analysis.py
    python freesurfer_analysis.py --subject sub-027
    python freesurfer_analysis.py --stage recon
    python freesurfer_analysis.py --stage samseg
    python freesurfer_analysis.py --threads 8
"""

import argparse
import datetime
import os
import subprocess
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
BIDS_DIR = BASE_DIR / "bids_dataset_LIFE"
DERIV_DIR = BIDS_DIR / "derivatives"
FS_SUBJECTS_DIR = DERIV_DIR / "freesurfer"
SAMSEG_DIR = DERIV_DIR / "samseg"
LOG_DIR = BASE_DIR / "LIFE_code" / "freesurfer" / "logs"

FREESURFER_HOME = Path(os.environ.get("FREESURFER_HOME",
                                     "/Applications/freesurfer/7.4.1"))
FS_SETUP = FREESURFER_HOME / "SetUpFreeSurfer.sh"


# =====================================================================
#  Logging
# =====================================================================

def _log(msg, log_lines):
    print(msg)
    log_lines.append(msg)


# =====================================================================
#  BIDS discovery
# =====================================================================

def discover_subjects(bids_dir, subject_filter=None):
    """Yield (sub_label, anat_dir) for every subject with an anat/ folder."""
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        if subject_filter and sub_dir.name != subject_filter:
            continue
        anat = sub_dir / "anat"
        if anat.is_dir():
            yield sub_dir.name, anat


def find_t1w(anat_dir, sub):
    p = anat_dir / f"{sub}_T1w.nii.gz"
    return p if p.exists() else None


def find_flair(anat_dir, sub):
    p = anat_dir / f"{sub}_FLAIR.nii.gz"
    return p if p.exists() else None


# =====================================================================
#  Shell wrapper that sources FreeSurfer
# =====================================================================

def run_fs_cmd(cmd_str, log_file, threads):
    """Run a shell command with FreeSurfer sourced."""
    wrapped = (
        f"export FREESURFER_HOME={FREESURFER_HOME}; "
        f"source {FS_SETUP}; "
        f"export OMP_NUM_THREADS={threads}; "
        f"{cmd_str}"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as lf:
        lf.write(f"\n\n===== {datetime.datetime.now().isoformat()} =====\n")
        lf.write(f"CMD: {cmd_str}\n\n")
        lf.flush()
        proc = subprocess.run(
            ["bash", "-lc", wrapped],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return proc


# =====================================================================
#  recon-all
# =====================================================================

def recon_all_subject(sub, t1, threads, log_lines):
    """Run recon-all -all on a subject's T1w."""
    fs_out = FS_SUBJECTS_DIR / sub
    done_flag = fs_out / "scripts" / "recon-all.done"

    if done_flag.exists():
        _log(f"    [recon-all] SKIP {sub} (already done)", log_lines)
        return True

    # If a partial run exists, resume
    resume = fs_out.exists() and not done_flag.exists()

    FS_SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd_log = LOG_DIR / f"recon-all_{sub}.log"

    if resume:
        _log(f"    [recon-all] RESUME {sub}", log_lines)
        cmd = (
            f"recon-all -all "
            f"-s {sub} "
            f"-sd {FS_SUBJECTS_DIR} "
            f"-cw256 "
            f"-openmp {threads} -parallel"
        )
    else:
        _log(f"    [recon-all] START  {sub}  (T1w: {t1.name})", log_lines)
        cmd = (
            f"recon-all -all "
            f"-s {sub} "
            f"-i {t1} "
            f"-sd {FS_SUBJECTS_DIR} "
            f"-cw256 "
            f"-openmp {threads} -parallel"
        )

    proc = run_fs_cmd(cmd, cmd_log, threads)
    ok = done_flag.exists()
    status = "OK" if ok else f"FAIL (rc={proc.returncode})"
    _log(f"    [recon-all] {status}  {sub}   log: {cmd_log}", log_lines)
    return ok


# =====================================================================
#  SAMSEG (FLAIR)
# =====================================================================

def samseg_subject(sub, flair, threads, log_lines):
    """Run run_samseg on the FLAIR volume."""
    out_dir = SAMSEG_DIR / sub
    seg_file = out_dir / "seg.mgz"

    if seg_file.exists():
        _log(f"    [samseg]    SKIP {sub} (already done)", log_lines)
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_log = LOG_DIR / f"samseg_{sub}.log"

    _log(f"    [samseg]    START  {sub}  (FLAIR: {flair.name})", log_lines)
    cmd = (
        f"run_samseg "
        f"--input {flair} "
        f"--output {out_dir} "
        f"--threads {threads}"
    )

    proc = run_fs_cmd(cmd, cmd_log, threads)
    ok = seg_file.exists()
    status = "OK" if ok else f"FAIL (rc={proc.returncode})"
    _log(f"    [samseg]    {status}  {sub}   log: {cmd_log}", log_lines)
    return ok


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer recon-all and SAMSEG on LIFE dataset."
    )
    parser.add_argument("--bids-dir", type=Path, default=BIDS_DIR)
    parser.add_argument("--subject", type=str, default=None,
                        help="Process a single subject (e.g. sub-027)")
    parser.add_argument("--stage", choices=["recon", "samseg", "both"],
                        default="both",
                        help="Which stage to run (default: both)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads per job (default: 8 for M2 Max)")
    args = parser.parse_args()

    log_lines = [
        f"LIFE FreeSurfer Analysis — {datetime.datetime.now()}",
        f"BIDS dir:   {args.bids_dir}",
        f"Stage:      {args.stage}",
        f"Threads:    {args.threads}",
        "",
    ]

    # Check FreeSurfer
    if not FREESURFER_HOME.exists():
        _log(f"ERROR: FREESURFER_HOME not found: {FREESURFER_HOME}", log_lines)
        _log("Update the path in the script or install FreeSurfer.", log_lines)
        return

    recon_ok = recon_fail = samseg_ok = samseg_fail = 0
    recon_skip = samseg_skip = 0

    for sub, anat_dir in discover_subjects(args.bids_dir, args.subject):
        _log(f"\n{'='*60}", log_lines)
        _log(f"Subject: {sub}", log_lines)

        # ── recon-all ──
        if args.stage in ("recon", "both"):
            t1 = find_t1w(anat_dir, sub)
            if t1:
                ok = recon_all_subject(sub, t1, args.threads, log_lines)
                if ok:
                    recon_ok += 1
                else:
                    recon_fail += 1
            else:
                _log(f"    [recon-all] SKIP {sub} (no T1w)", log_lines)
                recon_skip += 1

        # ── samseg ──
        if args.stage in ("samseg", "both"):
            flair = find_flair(anat_dir, sub)
            if flair:
                ok = samseg_subject(sub, flair, args.threads, log_lines)
                if ok:
                    samseg_ok += 1
                else:
                    samseg_fail += 1
            else:
                _log(f"    [samseg]    SKIP {sub} (no FLAIR)", log_lines)
                samseg_skip += 1

    # ── Summary ──
    _log(f"\n{'='*60}", log_lines)
    _log("Summary:", log_lines)
    if args.stage in ("recon", "both"):
        _log(f"  recon-all:  OK={recon_ok}  FAIL={recon_fail}  SKIP={recon_skip}",
             log_lines)
    if args.stage in ("samseg", "both"):
        _log(f"  samseg:     OK={samseg_ok}  FAIL={samseg_fail}  SKIP={samseg_skip}",
             log_lines)

    # Write log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"freesurfer_log_{ts}.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
