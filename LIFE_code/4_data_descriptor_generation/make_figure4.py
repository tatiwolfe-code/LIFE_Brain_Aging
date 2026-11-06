#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Figure 4 — Diffusion-weighted imaging quality.

  A  White matter connectivity in aging: cohort tractography and SDF maps.
     Drop a PNG at figures/Fig4A_tractography.png and it is embedded
     automatically; otherwise a labeled placeholder box is drawn.
  B  Per-shell SNR (b = 0/650/1350/2000), violins + RM-ANOVA / Friedman.
  C  Preprocessing residuals — outlier slices, neighboring DWI correlation,
     and head motion estimated from the interspersed b0 volumes.
  D  Whole white-matter QA, FA and MD distributions.

Older cohort (age >= 60) throughout.
Out -> 0 Data manuscript/figures/Figure4.png (300 dpi)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import make_figures as MF
from make_figures import INK, ORANGE, TV, FIG, pfmt, stars
from make_dwi_figures import (COVARS, load_shell_snr, older,
                              partial_r, rm_anova, statbox, violin)

FS = 7.5          # stat-box font size for the composite
# The panel-A render, whatever it was saved as. First match wins.
#
# Panel A is a DSI Studio rendering: a cohort whole-brain tractogram colored
# by QA, beside axial slices of the cohort SDF, made in the DSI Studio GUI
# (Hou release, 2026) from cohort-level LIFE .fib files (n = 57). It is
# illustrative only -- no statistic in the manuscript derives from it. It is
# not produced by any script here, so a copy ships with the code under
# assets/ and is used when the figures directory has none. Without it this
# script silently draws a labeled placeholder box instead of failing, which
# is easy to miss; shipping the render means `make_figure4.py` reproduces the
# published figure from a clean checkout.
TRACT_CANDIDATES = ["Fig4A.png", "Fig4A.jpg", "Fig4A.jpeg", "Fig4A.tif", "Fig4A.tiff",
                    "Fig4A_tractography.png", "Fig4A_tractography.jpg"]
ASSETS = Path(__file__).resolve().parent / "assets"
TRACT_PNG = next(
    (d / c for d in (FIG, ASSETS) for c in TRACT_CANDIDATES if (d / c).exists()),
    FIG / TRACT_CANDIDATES[0])


def qa_bar_column(path):
    """Column index of the left edge of the QA color bar inside the panel-A
    render: the tall, narrow, strongly colored component in the upper left
    (the brains are grayscale, the streamlines are wide). Returns None if it
    cannot be found, in which case the panel is left where it is."""
    try:
        from scipy import ndimage
        im = mpimg.imread(str(path)).astype(float)
        if im.max() > 1.5:
            im = im / 255.0
        h, w = im.shape[:2]
        mask = (im[..., :3].max(2) - im[..., :3].min(2)) > 0.35
        mask[:, int(w * 0.25):] = False
        mask[int(h * 0.55):, :] = False
        lab, _ = ndimage.label(mask)
        best = None
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            ys, xs = sl
            hh, ww = ys.stop - ys.start, xs.stop - xs.start
            if hh > 0.06 * h and ww < 0.05 * w and hh / max(ww, 1) > 3:
                if best is None or xs.start < best:
                    best = xs.start
        return best
    except Exception:                                    # noqa: BLE001
        return None


# ──────────────────────────────────────────────────────────────────────
#  A — tractography
# ──────────────────────────────────────────────────────────────────────

def panel_a(ax):
    ax.set_xticks([]); ax.set_yticks([])
    if TRACT_PNG.exists():
        ax.imshow(mpimg.imread(str(TRACT_PNG)))
        ax.set_anchor("W")          # flush left within the axes box
        for sp in ax.spines.values():
            sp.set_visible(False)
        return
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linestyle("--")
        sp.set_edgecolor("#b0b0b0"); sp.set_linewidth(1.2)
    ax.set_facecolor("#fafafa")
    ax.text(0.5, 0.5, "tractography rendering\nto be inserted\n\n"
                      f"(drop a PNG at\n{TRACT_PNG.name})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color="#9a9a9a", linespacing=1.6)


# ──────────────────────────────────────────────────────────────────────
#  B — per-shell SNR
# ──────────────────────────────────────────────────────────────────────

def panel_b(ax, pheno):
    snr = older(load_shell_snr(), pheno)
    cols = ["snr_b0", "snr_b650", "snr_b1350", "snr_b2000"]
    labels = ["b = 0", "b = 650", "b = 1350", "b = 2000"]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    mat = snr[cols].apply(pd.to_numeric, errors="coerce").dropna().values
    arrs = violin(ax, [mat[:, i] for i in range(mat.shape[1])], labels,
                  "SNR (median in-brain signal / σ)", colors=palette)
    a = rm_anova(mat)
    chi2, p_fr = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
    span = mat.max() - mat.min()
    ax.set_ylim(mat.min() - 0.10 * span, mat.max() + 0.40 * span)
    for i, dd in enumerate(arrs):
        ax.text(i + 1, dd.max() + 0.035 * span,
                f"{dd.mean():.2f} ± {dd.std(ddof=1):.2f}",
                ha="center", va="bottom", fontsize=FS, color=INK)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Diffusion shell (s/mm²)")
    statbox(ax, (
                 f"RM-ANOVA F({a['df1']},{a['df2']}) = {a['F']:.1f}, {pfmt(a['p'])}\n"
                 f"partial η² = {a['eta2p']:.2f}\n"
                 f"Friedman χ²({a['df1']}) = {chi2:.1f}, {pfmt(p_fr)}"),
            loc="upper right", fs=FS)


# ──────────────────────────────────────────────────────────────────────
#  C — preprocessing residuals
# ──────────────────────────────────────────────────────────────────────

def panel_c(axes, pheno):
    qc = older(pd.read_csv(TV / "dwi/dwi_qc_python.csv"), pheno)
    qc = qc.merge(pd.read_csv(TV / "dwi/b0_motion.csv"), on="participant_id", how="left")
    specs = [("n_bad_slices", "Outlier (bad) slices per scan", "#C44E52", "spearman"),
             ("ndc", "Neighboring DWI correlation (NDC)", "#4C72B0", "pearson"),
             ("mean_rel_disp_mm", "Mean inter-b0 displacement (mm)", "#55A868", "pearson")]
    for ax, (col, ylab, colr, method) in zip(axes, specs):
        d = pd.to_numeric(qc[col], errors="coerce").dropna()
        violin(ax, [d.values], [""], ylab, colors=[colr], widths=0.55)
        ax.set_xticks([])
        ax.yaxis.label.set_size(9)
        lo_ax, hi_ax = ax.get_ylim()
        if col == "n_bad_slices":
            ax.set_ylim(-0.04 * d.max(), d.max() * 1.45)
            statbox(ax, (f"M = {d.mean():.3f}, SD = {d.std(ddof=1):.3f}\n"
                         f"{(d == 0).mean() * 100:.0f}% of scans had zero outlier slices"),
                    loc="upper right", fs=FS)
        else:
            bot = lo_ax - (0.22 * (hi_ax - lo_ax) if col in ("mean_rel_disp_mm", "ndc") else 0)
            ax.set_ylim(bot, hi_ax + 0.52 * (hi_ax - lo_ax))
            _, cov = partial_r(qc, col, "age", COVARS)
            rr = C.partial_corr(qc, col, "age", cov, method=method)
            statbox(ax, (f"M = {d.mean():.3f}, SD = {d.std(ddof=1):.3f}\n"
                         f"Mdn = {np.median(d):.3f} "
                         f"[{np.percentile(d, 25):.3f}–{np.percentile(d, 75):.3f}]\n"
                         f"range = {d.min():.3f}–{d.max():.3f}\n"
                         f"age: partial r = {rr['r']:+.2f}, {pfmt(rr['p'])} {stars(rr['p'])}"),
                    loc="upper left", fs=FS)


# ──────────────────────────────────────────────────────────────────────
#  D — QA / FA / MD distributions
# ──────────────────────────────────────────────────────────────────────

def panel_d(axes, pheno):
    dti = older(pd.read_csv(TV / "dwi/dti_wm_metrics.csv"), pheno)
    specs = [("wm_mean_QA", "Quantitative anisotropy (QA)", "#8172B3"),
             ("wm_mean_FA", "Fractional anisotropy (FA)", "#4C72B0"),
             ("wm_mean_MD", "Mean diffusivity (×10⁻³ mm²/s)", "#C44E52")]
    for j, (ax, (col, ylab, colr)) in enumerate(zip(axes, specs)):
        d = pd.to_numeric(dti[col], errors="coerce").dropna()
        violin(ax, [d.values], [""], ylab, colors=[colr], widths=0.55)
        ax.set_xticks([])
        ax.yaxis.label.set_size(9)
        mu, sd, n = d.mean(), d.std(ddof=1), len(d)
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        W, pW = stats.shapiro(d)
        # No literature reference bands: the ranges are unsourced, so the
        # comparison to published cohorts is made in the Discussion instead.
        l0, h0 = ax.get_ylim()
        ax.set_ylim(l0 - 0.34 * (h0 - l0), h0)
        black = (f"M = {mu:.3f}, SD = {sd:.3f}\n"
                 f"95% CI [{mu - ci:.3f}, {mu + ci:.3f}]\n"
                 f"range = {d.min():.3f}–{d.max():.3f}")
        # Normality in orange, matching Figs 1-3: the test is named on the
        # first panel and abbreviated to "W = ..." on the rest.
        name = "Shapiro–Wilk W" if j == 0 else "W"
        orange = f"{name} = {W:.2f}, {pfmt(pW)}"
        tb = ax.text(0.03, 0.03, black, transform=ax.transAxes, ha="left",
                     va="bottom", fontsize=FS, color=INK,
                     bbox=dict(boxstyle="round,pad=0.32", fc="white",
                               ec="none", alpha=0.72))
        fig = ax.figure; fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        bb = tb.get_window_extent(rend)
        top_y = ax.transAxes.inverted().transform((bb.x0, bb.y1))[1]
        ax.text(0.03, top_y + 0.012, orange, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=FS, color=ORANGE)


def main():
    pheno = C.load_phenotype()
    fig = plt.figure(figsize=(12.4, 12.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.28, 1.0, 1.0],
                          hspace=0.42, wspace=0.55)

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    axes_c = [fig.add_subplot(gs[1, i:i + 2]) for i in (0, 2, 4)]
    axes_d = [fig.add_subplot(gs[2, i:i + 2]) for i in (0, 2, 4)]

    panel_a(ax_a)
    panel_b(ax_b, pheno)
    panel_c(axes_c, pheno)
    panel_d(axes_d, pheno)

    MF.panel_title(ax_b, "B  Per-shell SNR")
    for ax, t in zip(axes_c, ["Outlier slices", "Signal continuity", "Head motion"]):
        ax.set_title(t, fontsize=9.5, color=INK, pad=6)
    for ax, t in zip(axes_d, ["QA", "FA", "MD"]):
        ax.set_title(t, fontsize=9.5, color=INK, pad=6)

    # A, C and D head the left-hand column, so they are placed on the same
    # vertical rule as that column's y-axis labels rather than at a fixed
    # axes-fraction offset (tick-label widths differ from row to row).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def left_text_x(ax):
        """Figure-x of the leftmost y-axis text (label or tick labels) on `ax`."""
        xs = []
        if ax.yaxis.label.get_text():
            xs.append(ax.yaxis.label.get_window_extent(rend).x0)
        xs += [t.get_window_extent(rend).x0 for t in ax.get_yticklabels() if t.get_text()]
        return inv.transform((min(xs), 0))[0] if xs else None

    x_rule = min(x for x in (left_text_x(axes_c[0]), left_text_x(axes_d[0]))
                 if x is not None)

    def col_label(ax, text, dy):
        fig.text(x_rule, ax.get_position().y1 + dy, text, ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color=INK)

    col_label(ax_a, "A  White matter connectivity in aging", 0.006)

    # Slide panel A left so the QA color bar lines up with the "W" of its
    # title (the title's "A  " prefix is measured, not assumed).
    qa_col = qa_bar_column(TRACT_PNG) if TRACT_PNG.exists() else None
    if qa_col is not None:
        # Trailing spaces are dropped from a text extent, so the "A  " prefix
        # is measured by difference: width("A  W") - width("W").
        def _tw(txt):
            t = fig.text(0, 0, txt, fontsize=12, fontweight="bold")
            fig.canvas.draw()
            wd = t.get_window_extent(rend).width
            t.remove()
            return wd
        w_prefix = _tw("A  W") - _tw("W")
        x_w_disp = fig.transFigure.transform((x_rule, 0))[0] + w_prefix
        x_qa_disp = ax_a.transData.transform((qa_col, 0))[0]
        d_fig = (x_w_disp - x_qa_disp) / fig.bbox.width
        pos = ax_a.get_position()
        ax_a.set_position([pos.x0 + d_fig, pos.y0, pos.width, pos.height])
        print(f"  panel A shifted {d_fig * 100:+.2f}% of figure width "
              f"to align the QA bar with the title")
    col_label(axes_c[0], "C  Preprocessing residuals", 0.022)
    col_label(axes_d[0], "D  White-matter diffusion metrics", 0.022)

    fig.savefig(FIG / "Figure4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote Figure4  ->", FIG / "Figure4.png")
    if not TRACT_PNG.exists():
        print(f"  note: panel A is a placeholder; drop the render at {TRACT_PNG}")


if __name__ == "__main__":
    main()
