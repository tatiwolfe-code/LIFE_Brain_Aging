#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Cohort sliding-window temporal-SNR (tSNR) trajectories for the resting-state
fMRI, in gray matter and in deep cerebral white matter.

tSNR is recomputed in a moving 60-volume window (48 s at TR = 0.8 s, stepped
every 5 volumes) and summarized as the median across mask voxels, giving one
trajectory per participant. Very thin lines form the cohort spread; the group
mean and its 95% confidence interval are overlaid.

Reads derivatives/technical_validation/rsfmri/tsnr_windows/*_tsnr-{gm,wm}.txt
and tsnr_tissue_per_subject.csv (from cohort_tsnr.py).
Output -> 0 Data manuscript/figures/fig_fmri_tsnr_timeseries.png
"""
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
TSNRDIR = C.DERIV_DIR / "technical_validation" / "rsfmri" / "tsnr_windows"
CSV = C.DERIV_DIR / "technical_validation" / "rsfmri" / "tsnr_tissue_per_subject.csv"
ACCENT = "#4C72B0"; INK = "#2b2b2b"; CRIMSON = "#C44E52"
TISSUE = {"gm": ("Gray Matter BOLD tSNR", "#4C72B0"),
          "wm": ("White Matter BOLD tSNR", "#55A868")}


def load(tissue):
    files = sorted(glob.glob(str(TSNRDIR / f"*_tsnr-{tissue}.txt")))
    if not files:
        return None, None, []
    subs = [Path(f).name.split("_")[0] for f in files]
    series = [np.loadtxt(f) for f in files]
    L = int(np.median([len(s) for s in series]))
    x = None
    M = np.full((len(series), L), np.nan)
    for i, s in enumerate(series):
        k = min(len(s), L)
        M[i, :k] = s[:k, 1]
        if len(s) >= L and x is None:
            x = s[:L, 0]
    return x, M, subs


def draw(ax, x, M, color, label, show_n=False):
    n = len(M)
    cnt = np.sum(np.isfinite(M), 0)
    mean = np.nanmean(M, 0)
    ci = 1.96 * np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
    for row in M:
        ax.plot(x, row, color=color, lw=0.35, alpha=0.25, zorder=1)
    ax.fill_between(x, mean - ci, mean + ci, color=CRIMSON, alpha=0.30, zorder=2,
                    label="95% CI of mean")
    ax.plot(x, mean, color=CRIMSON, lw=2.0, zorder=3, label="Cohort mean")
    ax.set_ylabel("Median tSNR")
    ax.set_xlim(x.min(), x.max())
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
    # per-subject mean of the windowed trajectory: cohort mean, SD, 95% CI.
    # n is carried by the caption and Methods, not repeated on every panel.
    per_subj = np.nanmean(M, axis=1)
    m, sd = np.mean(per_subj), np.std(per_subj, ddof=1)
    lo, hi = m - 1.96 * sd / np.sqrt(n), m + 1.96 * sd / np.sqrt(n)
    txt = f"mean = {m:.1f} (SD = {sd:.1f})\n95% CI [{lo:.1f}, {hi:.1f}]"
    if show_n:
        txt = f"n = {n}   " + txt
    ax.text(0.985, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))
    return per_subj


def main():
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True)
    stats = {}
    for ax, tissue in zip(axes, ("gm", "wm")):
        x, M, subs = load(tissue)
        if M is None:
            ax.text(0.5, 0.5, f"no {tissue.upper()} windows found",
                    transform=ax.transAxes, ha="center")
            continue
        label, color = TISSUE[tissue]
        stats[tissue] = draw(ax, x, M, color, label)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    axes[-1].set_xlabel("Window center (volume number)")
    fig.suptitle("Resting-state temporal SNR across the run",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = FIG / "fig_fmri_tsnr_timeseries.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    msg = "  ".join(f"{t.upper()}: mean={v.mean():.1f} SD={v.std(ddof=1):.1f} n={len(v)}"
                    for t, v in stats.items())
    print("wrote", out, "|", msg)
    if CSV.exists():
        df = pd.read_csv(CSV)
        for t in ("gm", "wm"):
            if f"{t}_median" in df.columns:
                d = C.describe(df[f"{t}_median"])
                print(f"  whole-run {t.upper()} median tSNR: mean={d['mean']:.1f} "
                      f"SD={d['sd']:.1f} range=[{d['min']:.1f},{d['max']:.1f}] n={d['n']}")


if __name__ == "__main__":
    main()
