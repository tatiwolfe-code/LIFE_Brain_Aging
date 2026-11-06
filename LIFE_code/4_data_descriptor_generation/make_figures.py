#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Generate Technical Validation figures for the LIFE data descriptor.
Distributions = shaded (filled) histograms; age effects = scatter + fit with
inferential stats; one figure per analysis. Older cohort (>=60) for age effects.
Output -> 0 Data manuscript/figures/*.png (+ .pdf)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TV = C.DERIV_DIR / "technical_validation"
FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
AGE_MIN = 60
INK = "#2b2b2b"; ACCENT = "#4C72B0"; FILL = "#4C72B0"; ORANGE = "#E1812C"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "font.family": "DejaVu Sans", "axes.edgecolor": INK, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def pfmt(p):
    return "p < 0.001" if (p == p and p < 0.001) else f"p = {p:.3f}"


def stars(p):
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def panel_title(ax, text):
    # left-aligned to (approximately) the y-axis label position
    ax.set_title(text, loc="left", x=-0.14, y=1.02, fontsize=12, fontweight="bold")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")  # 300 dpi (savefig.dpi)
    plt.close(fig)
    print("  wrote", name)


def _normal_overlay(ax, d, sw_full=True):
    """Add a thin dashed orange Gaussian fit and return the Shapiro-Wilk label.
    `sw_full` names the test ("Shapiro-Wilk W = ..."); set it False on panels
    that follow one where the test is already named, which then read "W = ..."."""
    mu, sd = d.mean(), d.std(ddof=1)
    xs = np.linspace(d.min(), d.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=ORANGE, ls="--", lw=1.1)
    W, pN = stats.shapiro(d)
    name = "Shapiro–Wilk W" if sw_full else "W"
    return f"Normal fit\n{name} = {W:.2f}, {pfmt(pN)}"


def _statbox(ax, xa, ha, blocks):
    """Stack text blocks [(text, color, fontsize), ...] downward from the top of
    the axes, measuring each so the next sits just beneath it — robust across
    panels of different sizes (used by the standalone figs and the composite)."""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    ytop = 0.96
    for txt, color, fs in blocks:
        obj = ax.text(xa, ytop, txt, transform=ax.transAxes, ha=ha, va="top",
                      fontsize=fs, color=color,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.65))
        fig.canvas.draw()
        bb = obj.get_window_extent()
        _, y0f = inv.transform((bb.x0, bb.y0))
        ytop = y0f - 0.02


def _draw_hist(ax, d, xlabel, *, bins=18, normal=False, note=None, label=None, fs=9,
               sci_y=False, extra=None, sw_full=True):
    """Draw one shaded histogram + KDE (+ optional normal fit) onto `ax`.
    sci_y forces scientific notation on the density axis (fewer leading zeros)."""
    d = np.asarray(pd.to_numeric(pd.Series(d), errors="coerce").dropna())
    ax.hist(d, bins=bins, density=True, histtype="stepfilled",
            color=FILL, alpha=0.35, edgecolor=ACCENT, linewidth=1.4)
    if len(d) > 5:
        kde = stats.gaussian_kde(d)
        xs = np.linspace(d.min(), d.max(), 200)
        ax.plot(xs, kde(xs), color=ACCENT, lw=1.6)
    sh = _normal_overlay(ax, d, sw_full) if (normal and len(d) > 3) else None
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(fs - 1)
    # n is carried by the figure legend and Methods, not repeated on the panel
    stat = (f"{label}\n" if label else "")
    stat += (f"M = {d.mean():.2f}, SD = {d.std(ddof=1):.2f}"
             f"\nrange = {d.min():.2f}–{d.max():.2f}")
    if note:
        stat += f"\n{note}"
    right_heavy = d.mean() > 0.5 * (d.min() + d.max())
    xa, ha = (0.03, "left") if right_heavy else (0.97, "right")
    blocks = [(stat, INK, fs)]
    if sh:
        blocks.append((sh, ORANGE, fs - 1))
    if extra:
        blocks.append((extra, INK, fs - 1))
    _statbox(ax, xa, ha, blocks)


def shaded_hist(data, xlabel, name, title, bins=18, note=None, normal=False):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _draw_hist(ax, data, xlabel, bins=bins, normal=normal, note=note)
    panel_title(ax, title)
    save(fig, name)


def _draw_hist_stack(axes, series, labels, xlabel, *, bins=18, normal=False, fs=8,
                     extras=None, sw_full=True):
    """Draw two stacked shaded histograms (top, bottom) onto the given axes with
    shared bin edges and x-limits; only the bottom axis gets the x-label."""
    arrs = [np.asarray(pd.to_numeric(pd.Series(s), errors="coerce").dropna()) for s in series]
    conc = np.concatenate(arrs)
    edges = np.linspace(conc.min(), conc.max(), bins + 1)
    extras = extras or [None] * len(arrs)
    for ax, d, lab, ex in zip(axes, arrs, labels, extras):
        _draw_hist(ax, d, "", bins=edges, normal=normal, label=lab, fs=fs,
                   extra=ex, sw_full=sw_full)
        ax.set_xlim(edges[0], edges[-1])
    for ax in axes[:-1]:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel(xlabel)


def shaded_hist_stack(series, labels, xlabel, name, title, bins=18, normal=False):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4.6, 4.6))
    _draw_hist_stack(axes, series, labels, xlabel, bins=bins, normal=normal)
    panel_title(axes[0], title)
    save(fig, name)


def _draw_partial(ax, frame, dv, iv, covars, xlabel, ylabel, beta, p, n, beta_unit="",
                  ann_pos=(0.97, 0.04)):
    """Added-variable plot onto `ax`: dv and iv each residualized on covars, so
    the drawn slope equals the covariate-adjusted (ANCOVA) effect."""
    d = frame[[dv, iv] + covars].apply(pd.to_numeric, errors="coerce").dropna()

    def resid(col):
        X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in covars])
        y = d[col].values.astype(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ b

    rx, ry = resid(iv), resid(dv)
    ax.scatter(rx, ry, s=26, color=ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)
    b1, b0, r, pp, se = stats.linregress(rx, ry)
    xs = np.linspace(rx.min(), rx.max(), 100)
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.6)
    resd = ry - (b0 + b1 * rx); ssd = np.sqrt((resd ** 2).sum() / (len(rx) - 2))
    ci = 1.96 * ssd * np.sqrt(1 / len(rx) + (xs - rx.mean()) ** 2 / ((rx - rx.mean()) ** 2).sum())
    ax.fill_between(xs, b0 + b1 * xs - ci, b0 + b1 * xs + ci, color=INK, alpha=0.10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ann = f"β = {beta:+.1f}{beta_unit}, {pfmt(p)}  {stars(p)}"
    ax.text(ann_pos[0], ann_pos[1], ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.65))


def partial_regression(frame, dv, iv, covars, xlabel, ylabel, name, title,
                       beta, p, n, beta_unit="", ann_pos=(0.97, 0.04)):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _draw_partial(ax, frame, dv, iv, covars, xlabel, ylabel, beta, p, n, beta_unit, ann_pos)
    panel_title(ax, title)
    save(fig, name)


def _draw_box(ax, frame, dv, groupcol, order, labels, ylabel, beta, p, n,
              eta2=None, beta_unit="", covar_note="", ylim_pad=0.0, ann_fs=8.5,
              sig_bar=False):
    """Grouped box + jittered points onto `ax`; ylim_pad expands the y-range by
    that fraction of the data span on each side (used to fill a tall panel)."""
    d = frame[[dv, groupcol]].copy()
    d[dv] = pd.to_numeric(d[dv], errors="coerce")
    d = d.dropna()
    key = d[groupcol].astype(str).str[0].str.upper()
    groups = [d[key == g][dv].values for g in order]
    bp = ax.boxplot(groups, positions=range(len(order)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(FILL); patch.set_alpha(0.35); patch.set_edgecolor(ACCENT)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(ACCENT)
    rng = np.random.default_rng(0)
    for i, gv in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(gv)), gv, s=22, color=ACCENT,
                   alpha=0.7, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(list(labels))
    ax.set_ylabel(ylabel)
    if ylim_pad:
        allv = np.concatenate(groups); lo, hi = allv.min(), allv.max(); span = hi - lo
        ax.set_ylim(lo - ylim_pad * span, hi + ylim_pad * span)
    if sig_bar and len(groups) == 2:
        # Significance bracket spanning the two groups; the stars then live on
        # the bar rather than being repeated in the stats box.
        lo, hi = ax.get_ylim()
        span = hi - lo
        y = max(np.concatenate(groups)) + 0.07 * span
        h = 0.018 * span
        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.3, color=INK,
                solid_capstyle="butt", clip_on=False)
        ax.text(0.5, y + h * 1.25, stars(p), ha="center", va="bottom",
                fontsize=ann_fs + 3, fontweight="bold", color=INK)
        ax.set_ylim(lo, max(hi, y + h + 0.13 * span))
    line2 = (f"η² = {eta2:.2f}" if eta2 is not None else "") \
        + ("" if sig_bar else f"  {stars(p)}")
    ann = f"β = {beta:+.1f}{beta_unit}, {pfmt(p)}"
    if line2.strip():
        ann += f"\n{line2}"
    if covar_note:
        ann += f"\n{covar_note}"
    ax.text(0.97, 0.03, ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ann_fs, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))


def group_box(frame, dv, groupcol, order, labels, ylabel, name, title,
              beta, p, n, eta2=None, beta_unit="", covar_note=""):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    _draw_box(ax, frame, dv, groupcol, order, labels, ylabel, beta, p, n,
              eta2=eta2, beta_unit=beta_unit, covar_note=covar_note)
    panel_title(ax, title)
    save(fig, name)


def _draw_scatter(ax, x, y, xlabel, ylabel, covar_note="", stat=None, sig_p=None, fs=9):
    """Scatter + OLS fit + 95% CI onto `ax`; annotation auto-placed opposite the
    trend. `stat` overrides the default r/p/n annotation string."""
    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    ax.scatter(df.x, df.y, s=26, color=ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)
    b1, b0, r, p, se = stats.linregress(df.x, df.y)
    xs = np.linspace(df.x.min(), df.x.max(), 100)
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.6)
    n = len(df); resid = df.y - (b0 + b1 * df.x)
    ssd = np.sqrt((resid ** 2).sum() / (n - 2))
    ci = 1.96 * ssd * np.sqrt(1 / n + (xs - df.x.mean()) ** 2 / ((df.x - df.x.mean()) ** 2).sum())
    ax.fill_between(xs, b0 + b1 * xs - ci, b0 + b1 * xs + ci, color=INK, alpha=0.10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ann = stat if stat else f"r = {r:+.2f}, {pfmt(p)}"
    if sig_p is not None:
        ann += f"  {stars(sig_p)}"
    if covar_note:
        ann += f"\n{covar_note}"
    if b1 < 0:
        xa, ya, va = 0.97, 0.96, "top"
    else:
        xa, ya, va = 0.97, 0.04, "bottom"
    ax.text(xa, ya, ann, transform=ax.transAxes, ha="right", va=va, fontsize=fs, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))


def scatter_fit(x, y, xlabel, ylabel, name, title, covar_note="", stat=None, sig_p=None):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _draw_scatter(ax, x, y, xlabel, ylabel, covar_note=covar_note, stat=stat, sig_p=sig_p)
    panel_title(ax, title)
    save(fig, name)


def main():
    ph = C.load_phenotype()
    older = set(ph.index[ph["age"] >= AGE_MIN])

    def filt(df):  # restrict a per-subject table to the older cohort
        if "participant_id" in df.columns:
            return df[df["participant_id"].isin(older)]
        return df.loc[[i for i in df.index if i in older]]

    # ---------- distributions (older cohort) ----------
    eul = filt(pd.read_csv(TV / "structural/euler_numbers.csv"))
    shaded_hist(eul["euler_mean"], "Topological defects / hemisphere",
                "fig_structural_euler", "Euler number (surface QC)", normal=True)

    wmh = filt(pd.read_csv(TV / "wmh/wmh_per_subject.csv"))
    wcol = "wmh_volume_etiv_corr" if "wmh_volume_etiv_corr" in wmh else "wmh_volume_mm3"
    shaded_hist(wmh[wcol], "WMH volume (eTIV-corrected)",
                "fig_wmh_distribution", "WMH volume")

    fd = filt(pd.read_csv(TV / "rsfmri/fd_per_subject.csv"))
    shaded_hist(fd["mean_fd"], "Mean framewise displacement (mm)",
                "fig_fmri_fd", "Framewise displacement")
    tsnr = filt(pd.read_csv(TV / "rsfmri/tsnr_per_subject.csv"))
    shaded_hist(tsnr["median_tsnr"], "Median tSNR",
                "fig_fmri_tsnr", "Temporal SNR")

    snr = filt(pd.read_csv(TV / "dwi/per_shell_snr.csv"))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for col, c in [("snr_b0", "#4C72B0"), ("snr_b650", "#55A868"),
                   ("snr_b1350", "#C44E52"), ("snr_b2000", "#8172B3")]:
        if col in snr:
            d = pd.to_numeric(snr[col], errors="coerce").dropna()
            ax.hist(d, bins=15, density=True, histtype="stepfilled", alpha=0.30,
                    color=c, edgecolor=c, linewidth=1.2, label=f"b={col[5:]}")
    ax.set_xlabel("SNR"); ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=8)
    panel_title(ax, "Per-shell SNR")
    save(fig, "fig_dwi_pershell_snr")

    qc = filt(pd.read_csv(TV / "dwi/dwi_qc_python.csv"))
    if "ndc" in qc:
        shaded_hist(qc["ndc"], "Neighboring-DWI correlation (NDC)",
                    "fig_dwi_ndc", "DWI NDC", bins=10,
                    note=("PRELIMINARY n=%d" % qc["ndc"].notna().sum()
                          if len(qc) < 30 else None))

    # ---------- brain volumes + age effects (older cohort) ----------
    aseg = pd.read_csv(TV / "structural/aseg_volumes.csv").set_index("participant_id")
    aparc = pd.read_csv(TV / "structural/aparc_thickness.csv").set_index("participant_id")
    aparc["gt"] = aparc[[c for c in aparc.columns if "MeanThickness" in c]].mean(axis=1)
    oid = [i for i in aseg.index if i in older]
    aseg_o = aseg.loc[oid]
    o = ph.loc[[i for i in older if i in ph.index]]

    # cortical gray matter (CortexVol) if present, else total gray matter
    if "CortexVol" in aseg_o.columns:
        gmcol, gmlab, gmtitle = "CortexVol", "Cortical gray-matter volume (cm³)", "Cortical gray-matter volume"
    else:
        gmcol, gmlab, gmtitle = "TotalGray", "Total gray-matter volume (cm³)", "Gray-matter volume"

    # ---------- brain-volume distributions (Spreng Fig 1 style) ----------
    if "EstimatedTotalIntraCranialVol" in aseg_o:
        shaded_hist(aseg_o["EstimatedTotalIntraCranialVol"] / 1000.0,
                    "eTIV (cm³)", "fig_vol_etiv", "Intracranial volume (eTIV)", normal=True)
    if gmcol in aseg_o:
        shaded_hist(aseg_o[gmcol] / 1000.0, gmlab, "fig_vol_graymatter", gmtitle, normal=True)
    if {"Left-Hippocampus", "Right-Hippocampus"}.issubset(aseg_o.columns):
        shaded_hist_stack([aseg_o["Right-Hippocampus"] / 1000.0,   # top panel
                           aseg_o["Left-Hippocampus"] / 1000.0],    # bottom panel
                          ["Right", "Left"], "Hippocampal volume (cm³)",
                          "fig_vol_hippocampus", "Hippocampal volume", normal=True)
    shaded_hist(aparc.loc[oid, "gt"], "Global cortical thickness (mm)",
                "fig_vol_thickness", "Cortical thickness", normal=True)

    # eTIV vs social connectedness — recruitment-confound check (adjusted for
    # age, sex). Added-variable plot; slope = ANCOVA beta.
    if "sni_total" in ph.columns and "EstimatedTotalIntraCranialVol" in aseg_o.columns:
        soc = ph.loc[oid].copy()
        soc["eTIV_cm3"] = aseg_o["EstimatedTotalIntraCranialVol"] / 1000.0
        r = C.ancova(soc, "eTIV_cm3", "sni_total", ["age", "sex_code"])
        partial_regression(soc, "eTIV_cm3", "sni_total",
                           ["age", "sex_code"],
                           "Social network index (residual)",
                           "eTIV (cm³, residual)", "fig_etiv_social",
                           "eTIV × social network size",
                           beta=r["beta"], p=r["p"], n=r["n"], beta_unit=" cm³/pt")

    # sex effect on eTIV (placeholder; adjusted for age)
    if "EstimatedTotalIntraCranialVol" in aseg_o.columns:
        sx = ph.loc[oid].copy()
        sx["eTIV_cm3"] = aseg_o["EstimatedTotalIntraCranialVol"] / 1000.0
        rs = C.ancova(sx, "eTIV_cm3", "sex_code", ["age"])
        group_box(sx, "eTIV_cm3", "sex", ["F", "M"], ["Female", "Male"],
                  "eTIV (cm³)", "fig_etiv_sex", "Sex effect on eTIV",
                  beta=rs["beta"], p=rs["p"], n=rs["n"],
                  eta2=rs.get("partial_eta2"), beta_unit=" cm³",
                  covar_note="adj. age")

    def eff(dv_series, covars):
        d = o.join(dv_series.rename("dv"))
        d["eTIV"] = aseg.get("EstimatedTotalIntraCranialVol")
        return d, C.ancova(d, "dv", "age", covars)

    d, res = eff(aseg["Left-Hippocampus"], ["eTIV", "sex_code"])
    scatter_fit(d["age"], d["dv"], "Age (years)", "Left hippocampus (mm³)",
                "fig_struct_hippocampus_age", "Hippocampal volume × age",
                covar_note="covar: eTIV, sex",
                stat=f"β = {res['beta']:.1f}, {pfmt(res['p'])}, n = {res['n']}", sig_p=res["p"])

    d, res = eff(aseg[gmcol], ["eTIV", "sex_code"])
    scatter_fit(d["age"], d["dv"], "Age (years)", gmlab.replace(" (cm³)", " (mm³)"),
                "fig_struct_gm_age", f"{gmtitle} × age",
                covar_note="covar: eTIV, sex",
                stat=f"β = {res['beta']:.0f}, {pfmt(res['p'])}, n = {res['n']}", sig_p=res["p"])

    d, res = eff(aparc["gt"], ["sex_code"])
    scatter_fit(d["age"], d["dv"], "Age (years)", "Global cortical thickness (mm)",
                "fig_struct_thickness_age", "Cortical thickness × age",
                covar_note="covar: sex",
                stat=f"β = {res['beta']:.4f}, {pfmt(res['p'])}, n = {res['n']}", sig_p=res["p"])

    # cognition × age (older cohort)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, col, lab in [(axes[0], "dccs_uncorrected_standard_score", "DCCS (uncorrected)"),
                         (axes[1], "flanker_uncorrected_standard_score", "Flanker (uncorrected)")]:
        d = o[["age", col]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(d["age"], d[col], s=26, color=ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)
        b1, b0, r, p, _ = stats.linregress(d["age"], d[col])
        xs = np.linspace(d["age"].min(), d["age"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.5)
        ax.set_xlabel("Age (years)"); ax.set_ylabel(lab)
        va = "top" if b1 < 0 else "bottom"; ya = 0.96 if b1 < 0 else 0.04
        ax.text(0.97, ya, f"r = {r:+.2f}, {pfmt(p)}  {stars(p)}",
                transform=ax.transAxes, ha="right", va=va, fontsize=9)
    panel_title(axes[0], "Cognition × age")
    save(fig, "fig_cognition_age")

    print("\nAll figures (older cohort, >=%d y) in: %s" % (AGE_MIN, FIG))


if __name__ == "__main__":
    main()
