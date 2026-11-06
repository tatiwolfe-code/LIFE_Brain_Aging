#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Compute DSI Studio native diffusion QC for every subject and aggregate.

Runs `dsi_studio --action=qc` on each subject's SRC file, which reports the
neighboring-DWI correlation (NDC) and the number of bad slices, plus DWI count,
max b-value, and resolution. Results are aggregated to:
    derivatives/technical_validation/dwi/dwi_qc_dsistudio.csv

NDC is DSI Studio's primary data-quality metric (higher = better; scans are
flagged when NDC is a low outlier, e.g., < mean - 3 SD). Bad slices count
signal-dropout slices detected during SRC creation.

Set the binary with DSI_STUDIO; default probes common macOS app locations.
Usage:  python 11_dwi_qc_dsistudio.py
"""
import csv, glob, os, subprocess
import os
from pathlib import Path

BASE = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
DSI_DIR = BASE / "bids_dataset_LIFE" / "derivatives" / "dsi_studio_gqi"
OUT = BASE / "bids_dataset_LIFE" / "derivatives" / "technical_validation" / "dwi"


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


def main():
    ds = find_binary()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(DSI_DIR.glob("sub-*/dwi/*_dwi.src.gz")):
        sub = src.name.split("_")[0]
        subprocess.run([ds, "--action=qc", f"--source={src}"],
                       capture_output=True, text=True)
        qc = src.parent / (src.name.replace(".src.gz", ".qc.txt"))
        if not qc.exists():
            continue
        lines = qc.read_text().strip().splitlines()
        if len(lines) < 2:
            continue
        f = lines[1].split("\t")
        rows.append({"participant_id": sub, "dwi_count": f[3].strip(),
                     "max_bval": f[4].strip(), "ndc": f[5].strip(),
                     "bad_slices": f[6].strip()})
    if rows:
        with open(OUT / "dwi_qc_dsistudio.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        ndc = [float(r["ndc"]) for r in rows]
        bad = [int(float(r["bad_slices"])) for r in rows]
        import statistics as st
        print(f"n={len(rows)} | NDC mean={st.mean(ndc):.3f} "
              f"sd={st.pstdev(ndc):.3f} min={min(ndc):.3f} max={max(ndc):.3f} | "
              f"bad slices: {sum(bad)} total across subjects")
        print(f"written: {OUT / 'dwi_qc_dsistudio.csv'}")
    else:
        print("No SRC files found under", DSI_DIR)


if __name__ == "__main__":
    main()
