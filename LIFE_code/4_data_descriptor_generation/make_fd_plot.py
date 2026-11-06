#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Cohort framewise-displacement (FD) trajectory for the resting-state fMRI.

Standard Power et al. (2012) frame-to-frame FD (mm) per volume, one very thin
line per subject (forming the cohort spread), with the group mean and its 95%
confidence interval overlaid as thicker lines/band.

Reads derivatives/technical_validation/rsfmri/fd_timeseries/*_fd.txt
(from cohort_fd.py). Output -> 0 Data manuscript/figures/fig_fmri_fd_timeseries.png
"""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
FDDIR = C.DERIV_DIR / "technical_validation" / "rsfmri" / "fd_timeseries"
ACCENT = "#4C72B0"; INK = "#2b2b2b"; CRIMSON = "#C44E52"


def load():
    """Return (x, M, n): volume numbers and an n x L matrix of per-subject FD,
    short runs (e.g. sub-061, 354 vols) padded with NaN."""
    files = sorted(glob.glob(str(FDDIR / "*_fd.txt")))
    series = [np.loadtxt(f) for f in files]
    L = int(np.median([len(s) for s in series]))          # common length (450)
    M = np.full((len(series), L), np.nan)
    for i, s in enumerate(series):
        k = min(len(s), L)
        M[i, :k] = s[:k]
    return np.arange(1, L + 1), M, len(series)


def draw(ax, x, M, title="Resting-state framewise displacement",
         title_size=12, stats_y=0.79):
    """Draw the FD trajectory on `ax`: thin per-subject lines, cohort mean and
    its 95% CI. Shared by the standalone figure and the Figure 4 composite."""
    n = len(M)
    cnt = np.sum(np.isfinite(M), 0)                         # subjects per volume
    mean = np.nanmean(M, 0)
    ci = 1.96 * np.nanstd(M, 0, ddof=1) / np.sqrt(cnt)      # 95% CI of the mean
    for row in M:                                           # thin per-subject lines (NaN = gap)
        ax.plot(x, row, color=ACCENT, lw=0.35, alpha=0.25, zorder=1)
    ax.fill_between(x, mean - ci, mean + ci, color=CRIMSON, alpha=0.30, zorder=2,
                    label="95% CI of mean")
    ax.plot(x, mean, color=CRIMSON, lw=2.0, zorder=3, label="Cohort mean")
    ax.set_xlabel("Volume number")
    ax.set_ylabel("Framewise displacement (mm)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    # summary stats below the legend: per-subject mean FD, SD, and 95% CI of the mean
    per_subj = np.nanmean(M[:, 1:], axis=1)          # exclude vol 1 (FD=0)
    mfd, sfd = np.mean(per_subj), np.std(per_subj, ddof=1)
    lo, hi = mfd - 1.96 * sfd / np.sqrt(n), mfd + 1.96 * sfd / np.sqrt(n)
    ax.text(0.985, stats_y,
            f"mean FD = {mfd:.3f} mm (SD = {sfd:.3f})\n95% CI [{lo:.3f}, {hi:.3f}] mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=title_size, fontweight="bold")
    return mean


def main():
    x, M, n = load()
    L = M.shape[1]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    mean = draw(ax, x, M)
    fig.tight_layout()
    out = FIG / "fig_fmri_fd_timeseries.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"| n={n} L={L} grandmeanFD={mean.mean():.3f} "
          f"ymax(99pct)={np.nanpercentile(M,99):.2f}")


if __name__ == "__main__":
    main()
