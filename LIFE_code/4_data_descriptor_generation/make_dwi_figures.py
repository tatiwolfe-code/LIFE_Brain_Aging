#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Diffusion Technical Validation figures for the LIFE data descriptor.
One figure per validation item, violin plots throughout, inferential
statistics annotated inside every panel.

  fig_dwi_snr_violin.png          per-shell SNR (b = 0/650/1350/2000)
  fig_dwi_preproc_residuals.png   preprocessing residuals (bad slices, NDC)
  fig_dwi_dti_distributions.png   whole-WM QA / FA / MD distributions
  fig_dwi_age_effects.png         partial correlations of QA / FA / MD with age

Older cohort (age >= 60) throughout, matching make_figures.py.
Output -> 0 Data manuscript/figures/*.png (300 dpi)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from make_figures import (AGE_MIN, ACCENT, FILL, INK, ORANGE, TV, FIG,
                          panel_title, pfmt, save, stars)

COVARS = ["sex_code"]


# =====================================================================
#  Violin helper
# =====================================================================

def violin(ax, groups, labels, ylabel, *, colors=None, jitter=True,
           annotate=True, fs=8.5, widths=0.75):
    """Shaded violins + jittered points + mean ± SD marker.

    `groups` is a list of 1-D arrays. Returns the cleaned arrays.
    """
    arrs = [np.asarray(pd.to_numeric(pd.Series(g), errors="coerce").dropna())
            for g in groups]
    pos = np.arange(1, len(arrs) + 1)
    colors = colors or [FILL] * len(arrs)

    parts = ax.violinplot(arrs, positions=pos, widths=widths,
                          showmeans=False, showextrema=False, showmedians=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.32)
        body.set_linewidth(1.3)

    rng = np.random.default_rng(0)
    for p, d, col in zip(pos, arrs, colors):
        if jitter:
            ax.scatter(p + rng.uniform(-0.09, 0.09, len(d)), d, s=11, color=col,
                       alpha=0.55, edgecolor="white", linewidth=0.35, zorder=3)
        mu, sd = d.mean(), d.std(ddof=1)
        ax.vlines(p, mu - sd, mu + sd, color=INK, lw=1.6, zorder=4)
        ax.plot(p, mu, "o", ms=5.5, mfc="white", mec=INK, mew=1.4, zorder=5)
        if annotate:
            ax.text(p, ax.get_ylim()[0], "", fontsize=fs)  # keep layout stable

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.4, len(arrs) + 0.6)
    return arrs


def bracket(ax, x1, x2, y, h, text, fs=8.5):
    """Significance bracket between two violin positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color=INK)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fs, color=INK)


def statbox(ax, text, loc="upper right", fs=8.5, color=INK):
    xa, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    ya, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(xa, ya, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            color=color,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="none", alpha=0.72))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def rm_anova(mat):
    """One-way repeated-measures ANOVA on an (n_subjects x k_conditions) matrix.
    Returns dict with F, p, df, partial eta^2."""
    n, k = mat.shape
    grand = mat.mean()
    ss_cond = n * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_tot = ((mat - grand) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    df_c, df_e = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_c) / (ss_err / df_e)
    return {"F": F, "p": float(stats.f.sf(F, df_c, df_e)),
            "df1": df_c, "df2": df_e, "eta2p": ss_cond / (ss_cond + ss_err), "n": n}


def partial_r(frame, x, y, covars):
    covars = [c for c in covars if c in frame.columns]
    return C.partial_corr(frame, x, y, covars), covars


def older(df, pheno):
    """Merge phenotype and restrict to the older cohort."""
    m = df.merge(pheno.reset_index(drop=True), on="participant_id",
                 how="left", suffixes=("", "_ph"))
    return m[pd.to_numeric(m["age"], errors="coerce") >= AGE_MIN].copy()


# =====================================================================
#  1. Per-shell SNR
# =====================================================================

def load_shell_snr():
    """Per-shell SNR from the noise-floor module (the same numbers as Fig. S5).

    sigma is estimated once per participant as the SD across the seven b = 0
    repeats inside the brain (assumption-free), and each shell's SNR is the
    median in-brain signal divided by that sigma. Computing SNR instead as the
    mean/SD *across volumes within a shell* is invalid for b > 0, because
    volumes differ in gradient direction and the across-volume SD then contains
    true anisotropic signal variation rather than noise alone.
    """
    nf = pd.read_csv(C.DERIV_DIR / "technical_validation/noise_floor/noise_floor_dwi.csv")
    w = nf.pivot(index="participant_id", columns="shell", values="snr")
    w.columns = [f"snr_b{int(c)}" for c in w.columns]
    return w.reset_index()


def fig_snr(pheno):
    snr = older(load_shell_snr(), pheno)
    cols = ["snr_b0", "snr_b650", "snr_b1350", "snr_b2000"]
    labels = ["b = 0", "b = 650", "b = 1350", "b = 2000"]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    d = snr[cols].apply(pd.to_numeric, errors="coerce").dropna()
    mat = d.values

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    arrs = violin(ax, [mat[:, i] for i in range(mat.shape[1])], labels,
                  "SNR (median in-brain signal / σ)", colors=palette)

    a = rm_anova(mat)
    chi2, p_fr = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])

    span = mat.max() - mat.min()
    ax.set_ylim(mat.min() - 0.10 * span, mat.max() + 0.34 * span)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Diffusion shell (s/mm²)")
    for i, dd in enumerate(arrs):
        ax.text(i + 1, dd.max() + 0.035 * span,
                f"{dd.mean():.2f} ± {dd.std(ddof=1):.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    statbox(ax, (f"n = {a['n']}\n"
                 f"RM-ANOVA F({a['df1']},{a['df2']}) = {a['F']:.1f}, {pfmt(a['p'])}\n"
                 f"partial η² = {a['eta2p']:.2f}\n"
                 f"Friedman χ²({a['df1']}) = {chi2:.1f}, {pfmt(p_fr)}"),
            loc="upper right", fs=8.5)
    panel_title(ax, "Per-shell SNR")
    save(fig, "fig_dwi_snr_violin")
    return a


# =====================================================================
#  2. Preprocessing residuals
# =====================================================================

def fig_residuals(pheno):
    qc = older(pd.read_csv(TV / "dwi/dwi_qc_python.csv"), pheno)
    mot = pd.read_csv(TV / "dwi/b0_motion.csv")
    qc = qc.merge(mot, on="participant_id", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))

    # Raw-data QC only: QSDR template fit is a property of one reconstruction
    # choice, not of the released data, so it is deliberately not shown here.
    specs = [
        ("n_bad_slices", "Outlier (bad) slices per scan", "#C44E52", "spearman"),
        ("ndc", "Neighboring DWI correlation (NDC)", "#4C72B0", "pearson"),
        ("mean_rel_disp_mm", "Mean inter-b0 displacement (mm)", "#55A868", "pearson"),
    ]
    for ax, (col, ylab, colr, method) in zip(axes, specs):
        d = pd.to_numeric(qc[col], errors="coerce").dropna()
        violin(ax, [d.values], [""], ylab, colors=[colr], widths=0.55)
        ax.set_xticks([])
        lo_ax, hi_ax = ax.get_ylim()
        if col == "n_bad_slices":
            # count variable: no negative space, headroom for the top-right legend
            ax.set_ylim(-0.04 * d.max(), d.max() * 1.42)
            txt = (f"n = {len(d)}\n"
                   f"M = {d.mean():.3f}, SD = {d.std(ddof=1):.3f}\n"
                   f"{(d == 0).mean() * 100:.0f}% of scans had zero outlier slices")
            statbox(ax, txt, loc="upper right", fs=8)
        else:
            bot = lo_ax - (0.22 * (hi_ax - lo_ax) if col in ("mean_rel_disp_mm", "ndc") else 0)
            ax.set_ylim(bot, hi_ax + 0.50 * (hi_ax - lo_ax))
            _, cov = partial_r(qc, col, "age", COVARS)
            rr = C.partial_corr(qc, col, "age", cov, method=method)
            rlab = "ρ" if method == "spearman" else "r"
            txt = (f"n = {len(d)}\n"
                   f"M = {d.mean():.3f}, SD = {d.std(ddof=1):.3f}\n"
                   f"Mdn = {np.median(d):.3f} "
                   f"[{np.percentile(d, 25):.3f}–{np.percentile(d, 75):.3f}]\n"
                   f"range = {d.min():.3f}–{d.max():.3f}\n"
                   f"age: partial {rlab} = {rr['r']:+.2f}, {pfmt(rr['p'])} {stars(rr['p'])}")
            statbox(ax, txt, loc="upper left", fs=8)

    axes[1].text(0.5, 0.03, "all scans above the suggested\nQC threshold of 0.6 (Yeh et al., 2019)",
                 transform=axes[1].transAxes, ha="center", va="bottom",
                 fontsize=8, color=ORANGE)
    panel_title(axes[0], "A  Outlier slices")
    panel_title(axes[1], "B  Signal continuity")
    panel_title(axes[2], "C  Head motion")
    mx = pd.to_numeric(qc["max_abs_disp_mm"], errors="coerce")
    axes[2].text(0.03, 0.03,
                 f"peak displacement from first b0:\nM = {mx.mean():.2f} mm, max = {mx.max():.2f} mm",
                 transform=axes[2].transAxes, ha="left", va="bottom", fontsize=8,
                 color=ORANGE)
    save(fig, "fig_dwi_preproc_residuals")


# =====================================================================
#  3. DTI metric distributions
# =====================================================================

def fig_dti_dist(pheno):
    dti = older(pd.read_csv(TV / "dwi/dti_wm_metrics.csv"), pheno)
    specs = [("wm_mean_QA", "Quantitative anisotropy (QA)", "#8172B3", ""),
             ("wm_mean_FA", "Fractional anisotropy (FA)", "#4C72B0", ""),
             ("wm_mean_MD", "Mean diffusivity (×10⁻³ mm²/s)", "#C44E52", "")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
    for ax, (col, ylab, colr, _) in zip(axes, specs):
        d = pd.to_numeric(dti[col], errors="coerce").dropna()
        violin(ax, [d.values], [""], ylab, colors=[colr], widths=0.55)
        ax.set_xticks([])
        mu, sd, n = d.mean(), d.std(ddof=1), len(d)
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        W, pW = stats.shapiro(d)
        txt = (f"n = {n}\nM = {mu:.3f}, SD = {sd:.3f}\n"
               f"95% CI [{mu - ci:.3f}, {mu + ci:.3f}]\n"
               f"range = {d.min():.3f}–{d.max():.3f}\n"
               f"Shapiro–Wilk W = {W:.2f}, {pfmt(pW)}")
        # No literature reference bands (see make_figure5.py).
        l0, h0 = ax.get_ylim()
        ax.set_ylim(l0 - 0.34 * (h0 - l0), h0)
        statbox(ax, txt, loc="lower left", fs=8)
    panel_title(axes[0], "A  QA")
    panel_title(axes[1], "B  FA")
    panel_title(axes[2], "C  MD")
    save(fig, "fig_dwi_dti_distributions")


# =====================================================================
#  4. Age effects
# =====================================================================

def fig_age_effects(pheno):
    dti = older(pd.read_csv(TV / "dwi/dti_wm_metrics.csv"), pheno)
    specs = [("wm_mean_QA", "QA (residualized)", "#8172B3"),
             ("wm_mean_FA", "FA (residualized)", "#4C72B0"),
             ("wm_mean_MD", "MD (residualized, ×10⁻³ mm²/s)", "#C44E52")]
    covars = [c for c in COVARS if c in dti.columns]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    for ax, (col, ylab, colr) in zip(axes, specs):
        d = dti[[col, "age"] + covars].apply(pd.to_numeric, errors="coerce").dropna()
        X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in covars])

        def resid(v):
            b, *_ = np.linalg.lstsq(X, v, rcond=None)
            return v - X @ b

        rx, ry = resid(d["age"].values.astype(float)), resid(d[col].values.astype(float))
        ax.scatter(rx, ry, s=28, color=colr, alpha=0.7, edgecolor="white", linewidth=0.5)
        b1, b0, r, _, _ = stats.linregress(rx, ry)
        xs = np.linspace(rx.min(), rx.max(), 100)
        ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.6)
        rs = ry - (b0 + b1 * rx)
        ssd = np.sqrt((rs ** 2).sum() / (len(rx) - 2))
        ciw = 1.96 * ssd * np.sqrt(1 / len(rx) + (xs - rx.mean()) ** 2 /
                                   ((rx - rx.mean()) ** 2).sum())
        ax.fill_between(xs, b0 + b1 * xs - ciw, b0 + b1 * xs + ciw, color=INK, alpha=0.10)
        res = C.partial_corr(d, col, "age", covars)
        n = res["n"]
        z = np.arctanh(res["r"])
        se = 1 / np.sqrt(n - len(covars) - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        ax.set_xlabel("Age (years, residualized)")
        ax.set_ylabel(ylab)
        statbox(ax, (f"partial r = {res['r']:+.2f} {stars(res['p'])}\n"
                     f"95% CI [{lo:+.2f}, {hi:+.2f}]\n{pfmt(res['p'])}, n = {n}\n"
                     f"β = {b1:+.4f} per year"),
                loc="upper right" if b1 < 0 else "lower right", fs=8.5)
    panel_title(axes[0], "A  QA × age")
    panel_title(axes[1], "B  FA × age")
    panel_title(axes[2], "C  MD × age")
    save(fig, "fig_dwi_age_effects")


def main():
    pheno = C.load_phenotype()
    print("Diffusion Technical Validation figures ->", FIG)
    fig_snr(pheno)
    fig_residuals(pheno)
    fig_dti_dist(pheno)
    fig_age_effects(pheno)


if __name__ == "__main__":
    main()
