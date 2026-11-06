#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Supplementary figures S1-S4: the noise floor of each LIFE imaging modality.

Reads the CSVs written by noise_floor.py and draws, in the same visual language
as the main-text figures (shaded distributions, boxes with jittered points,
auto-placed stat boxes):

  S1  T1w and FLAIR spatial SNR, split by FLAIR protocol
  S2  multi-echo GRE: SNR down the echo train and the TE horizon it implies
  S3  rs-fMRI BOLD: the thermal plateau, the tSNR ceiling it sets, and the
      noise floor in units of % signal change
  S4  DWI: SNR per shell and how much of the brain reaches the floor

Output -> 0 Data manuscript/figures/FigureS1..S4.png (300 dpi)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import noise_floor as N
from make_figures import (FIG, INK, ACCENT, FILL, ORANGE, panel_title, save,
                          pfmt, _draw_hist, _statbox)

NF = C.DERIV_DIR / "technical_validation" / "noise_floor"
# The 3-D SPACE FLAIR is reconstructed at 0.45 mm from a 0.9 mm acquisition.
# Interpolation correlates neighboring voxels, which deflates every
# background-free sigma estimate, so its SNR is reported but never pooled or
# compared with the other acquisitions.
INTERP_PROTO = {"FLAIR 3D SPACE", "FLAIR 3D"}
INTERP = " †"
GREEN = "#3A923A"
GRAY = "#8C8C8C"
# Modality order used everywhere, with the sigma estimator each one relies on.
MODALITIES = [("T1w", "laplacian"), ("FLAIR", "laplacian"),
              ("ME-GRE", "air"), ("BOLD", "thermal"), ("DWI", "repeat")]


# =====================================================================
#  small drawing helpers
# =====================================================================

def _box(ax, groups, labels, ylabel, color=ACCENT, fill=FILL, widths=0.55,
         points=True, log=False, seed=0):
    """Box + jittered points, styled like the main-text panels."""
    bp = ax.boxplot(groups, positions=range(len(groups)), widths=widths,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(fill); patch.set_alpha(0.35); patch.set_edgecolor(color)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(color)
    if points:
        rng = np.random.default_rng(seed)
        for i, g in enumerate(groups):
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(g)), g, s=14, color=color,
                       alpha=0.55, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    return bp


def _floor_line(ax, y=1.0, label="noise floor (SNR = 1)"):
    ax.axhline(y, color=ORANGE, ls="--", lw=1.2, zorder=1)
    ax.text(0.99, y, f" {label}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=ORANGE)


def _note(ax, text, xa=0.03, ha="left", fs=8.5, color=INK):
    _statbox(ax, xa, ha, [(text, color, fs)])


def _headroom(ax, frac=0.45):
    """Open space above the data so a top-anchored stat box does not sit on it."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(lo, hi * (10 ** (frac * np.log10(hi / lo))))
    else:
        ax.set_ylim(lo, hi + frac * (hi - lo))


def _footnote(fig, text, y=0.005):
    fig.text(0.02, y, text, ha="left", va="bottom", fontsize=7.5, color=ORANGE)


def med_iqr(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return s.median(), s.quantile(.25), s.quantile(.75), len(s)


def fmt_mi(s, unit="", d=1):
    m, lo, hi, n = med_iqr(s)
    return f"{m:.{d}f}{unit} [{lo:.{d}f}–{hi:.{d}f}]"


# =====================================================================
#  data loading
# =====================================================================

def load():
    d = {}
    for k, f in [("dwi", "noise_floor_dwi.csv"), ("gre", "noise_floor_gre.csv"),
                 ("bold", "noise_floor_bold.csv"), ("struct", "noise_floor_struct.csv"),
                 ("spectra", "noise_floor_bold_spectra.csv")]:
        p = NF / f
        d[k] = pd.read_csv(p) if p.exists() else None
        if d[k] is None:
            print(f"  ! missing {f}")
    d["est"] = N.load_estimators()
    return d


# =====================================================================
#  S1  structural T1w / FLAIR
# =====================================================================

def figure_s1(D):
    d = D["struct"]
    if d is None:
        return
    d = d.copy()
    # The release carries a 2-D and a 3-D FLAIR; their SNR is not comparable.
    d["proto"] = np.where(d.contrast == "T1w", "T1w",
                          np.where(d.sequence.str.contains("3D|SPACE", case=False, na=False),
                                   "FLAIR 3D SPACE", "FLAIR 2D"))
    order = ["T1w", "FLAIR 2D", "FLAIR 3D SPACE"]
    order = [o for o in order if (d.proto == o).any()]
    # Noise floor as a share of tissue signal: the directly interpretable form.
    d["floor_pct"] = 100 * d.floor / d.signal
    groups = [d[d.proto == o] for o in order]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    A, B, Cx, Dx = axes.ravel()

    _box(A, [g.snr.dropna().values for g in groups],
         [f"{o}{INTERP if o in INTERP_PROTO else ''}\n(n = {len(g)})"
          for o, g in zip(order, groups)],
         "SNR  (median tissue / σ)")
    A.set_yscale("log")
    _note(A, "\n".join(f"{o}: {fmt_mi(g.snr)}" for o, g in zip(order, groups))
          + "\nmedian [IQR]", xa=0.97, ha="right")
    _headroom(A, 0.30)
    panel_title(A, "A")

    _draw_hist(B, groups[0].snr, "T1w SNR", bins=16, normal=True)
    # Flag the low-SNR outliers: print each subject's number in a small font
    # just above its own histogram bar, in the ~35-50 SNR band.
    t1 = groups[0][["participant_id", "snr"]].dropna()
    low = t1[t1.snr < 50].sort_values("snr")
    if len(low):
        counts, edges = np.histogram(t1.snr.values, bins=16, density=True)
        ymax = B.get_ylim()[1]
        for _, r in low.iterrows():
            bi = min(int(np.searchsorted(edges, r.snr, side="right") - 1), len(counts) - 1)
            y = counts[max(bi, 0)] + 0.02 * ymax
            sid = str(r.participant_id).replace("sub-", "")
            B.text(r.snr, y, sid, rotation=90, ha="center", va="bottom",
                   fontsize=6, color="#444444")
    panel_title(B, "B")

    _box(Cx, [g.floor_pct.dropna().values for g in groups],
         [o for o in order], "noise floor as % of tissue signal")
    Cx.tick_params(axis="x", labelsize=9)
    _headroom(Cx, 0.32)
    _note(Cx, "\n".join(f"{o}: {fmt_mi(g.floor_pct, ' %', 2)}"
                        for o, g in zip(order, groups)), xa=0.97, ha="right")
    panel_title(Cx, "C")

    _box(Dx, [100 * g.frac_floor.dropna().values for g in groups],
         [o for o in order], "brain voxels at the noise floor (%)")
    Dx.tick_params(axis="x", labelsize=9)
    _headroom(Dx, 0.55)
    _note(Dx, "voxels with signal ≤ 2σ\n\n"
              + "\n".join(f"{o}: {fmt_mi(100*g.frac_floor, ' %', 3)}"
                          for o, g in zip(order, groups))
              + "\n\nthe structural contrasts sit far above\ntheir floor everywhere in the brain",
          xa=0.97, ha="right")
    panel_title(Dx, "D")

    save(fig, "FigureS1_noise_structural")


# =====================================================================
#  S2  multi-echo GRE (T2*)
# =====================================================================

FLOOR_SUBJECT = "sub-054"   # last-echo <=2 sigma fraction 2.51%, at the cohort median (2.53%)


def _floor_voxel_insets(Cx):
    """Two small axial insets in panel C showing the last-echo voxels at or below
    2 sigma (red) on a gray GRE slice, at ventricle level and at the skull base,
    so the reader sees the floor voxels are concentrated in the sinus-adjacent
    dropout rather than spread through the brain."""
    from scipy import ndimage
    import make_t2star_panel as MT
    try:
        paths = [str(N.GRE_DIR / f"{FLOOR_SUBJECT}_ME-T2_e{e}.nii.gz") for e in range(1, 7)]
        data = np.stack([np.asarray(MT.canon(nib.load(pp)).dataobj, np.float32)
                         for pp in paths], -1)
    except Exception:
        return                       # raw GRE not on disk: skip the inset silently
    e1, e6 = data[..., 0], data[..., -1]
    bmp = N.GRE_DIR / f"{FLOOR_SUBJECT}_brainmask.nii.gz"
    brain = (ndimage.binary_erosion(np.asarray(MT.canon(nib.load(str(bmp))).dataobj) > 0,
                                     iterations=2)
             if bmp.exists() else N.brain_mask(e1))
    bg = N.background_mask(e1, brain)
    sigma = N.sigma_air(e1, bg)                       # sigma from echo 1, as in noise_floor
    floor = brain & (e6 <= N.FLOOR_K * sigma)         # <=2 sigma at the last echo
    t2 = MT.t2star(data, brain)
    zv = MT.ventricle_slice(t2, brain)
    zb = int(np.argmax([floor[:, :, z].sum() for z in range(floor.shape[2])]))

    def draw(ax, z, label):
        bgim = MT.disp(e1[:, :, z]); bm = MT.disp(brain[:, :, z])
        fm = ndimage.binary_dilation(MT.disp(floor[:, :, z]), iterations=1)
        rr = np.where(bm.any(1))[0]; cc = np.where(bm.any(0))[0]
        r0, r1 = rr.min() - 2, rr.max() + 3; c0, c1 = cc.min() - 2, cc.max() + 3
        lo, hi = np.percentile(bgim[bm], [2, 99.5])
        ax.imshow(np.where(bm, bgim, np.nan)[r0:r1, c0:c1], cmap="gray", vmin=lo, vmax=hi)
        ov = np.zeros((r1 - r0, c1 - c0, 4)); ov[fm[r0:r1, c0:c1]] = [0.92, 0.08, 0.08, 1]
        ax.imshow(ov)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(label, fontsize=7, pad=1)

    a1 = Cx.inset_axes([0.06, 0.40, 0.24, 0.37])
    a2 = Cx.inset_axes([0.31, 0.40, 0.24, 0.37])
    draw(a1, zv, "ventricle level"); draw(a2, zb, "skull base")
    # small legend below the insets: red square = the floor voxels
    Cx.plot(0.075, 0.335, marker="s", ms=7, color="#D01010",
            transform=Cx.transAxes, clip_on=False)
    Cx.text(0.11, 0.335, "brain voxels ≤ 2σ (TE = 65 ms)", transform=Cx.transAxes,
            ha="left", va="center", fontsize=7, color=INK)


def figure_s2(D):
    d = D["gre"]
    if d is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    A, B, Cx, Dx = axes.ravel()
    tes = d.groupby("echo").te_ms.first().values
    piv = d.pivot_table(index="participant_id", columns="echo", values="snr")

    # --- A: SNR down the echo train, extrapolated to the floor -------------
    for _, row in piv.iterrows():
        A.plot(tes, row.values, color=ACCENT, alpha=0.18, lw=0.8)
    med = piv.median().values
    A.plot(tes, med, color=INK, lw=2.2, label="cohort median")
    # Mono-exponential fit continued past the last echo.
    slope, icpt = np.polyfit(tes, np.log(med), 1)
    xs = np.linspace(tes[0], 200, 100)
    A.plot(xs, np.exp(icpt + slope * xs), color=ORANGE, ls=":", lw=1.6,
           label="fitted decay, extrapolated")
    A.axvline(tes[-1], color=GRAY, lw=1.0, ls="-")
    A.text(tes[-1], 0.98, " last echo, 65 ms", transform=A.get_xaxis_transform(),
           fontsize=8, color=GRAY, va="top", ha="left")
    _floor_line(A, 1.0)
    A.axhline(5, color=ORANGE, ls="-.", lw=1.0)
    A.text(0.99, 5, " SNR = 5", transform=A.get_yaxis_transform(), ha="right",
           va="bottom", fontsize=8, color=ORANGE)
    A.set_yscale("log"); A.set_xlabel("echo time (ms)"); A.set_ylabel("SNR")
    A.legend(fontsize=8, frameon=False, loc="lower left")
    panel_title(A, "A")

    # --- B: SNR at each echo ----------------------------------------------
    _box(B, [d[d.echo == e].snr.dropna().values for e in sorted(d.echo.unique())],
         [f"{t:.1f}" for t in tes], "SNR")
    B.set_xlabel("echo time (ms)")
    B.set_yscale("log")
    last = d[d.echo == d.echo.max()].snr
    _headroom(B, 0.30)
    _note(B, f"last echo: SNR = {fmt_mi(last)}\n"
             f"at TE = {tes[-1]:.0f} ms", xa=0.97, ha="right")
    panel_title(B, "B")

    # --- C: how much brain is at the floor at each echo --------------------
    _box(Cx, [100 * d[d.echo == e].frac_floor.dropna().values
              for e in sorted(d.echo.unique())],
         [f"{t:.1f}" for t in tes], "brain voxels at the noise floor (%)")
    Cx.set_xlabel("echo time (ms)")
    _headroom(Cx, 0.30)
    e_last = d[d.echo == d.echo.max()]
    _note(Cx, "voxels with signal ≤ 2σ\n"
              f"at TE = {tes[-1]:.0f} ms: {fmt_mi(100*e_last.frac_floor, ' %', 2)}",
          xa=0.03, ha="left")
    _floor_voxel_insets(Cx)
    panel_title(Cx, "C")

    # --- D: the TE horizon the noise floor implies -------------------------
    f = d.groupby("participant_id").first()
    _box(Dx, [f.te_snr5.dropna().values, f.te_snr2.dropna().values],
         ["TE at SNR = 5", "TE at SNR = 2"], "extrapolated echo time (ms)")
    # Keep the last acquired echo inside the view: the gap between it and the
    # extrapolated horizons is the message of the panel.
    Dx.set_ylim(tes[-1] - 12, Dx.get_ylim()[1])
    _headroom(Dx, 0.28)
    Dx.axhline(tes[-1], color=GRAY, lw=1.2, ls="--")
    Dx.text(0.99, tes[-1], " acquired range ends at 65 ms",
            transform=Dx.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=GRAY)
    _note(Dx, f"SNR = 5 at {fmt_mi(f.te_snr5, ' ms', 0)}\n"
              f"SNR = 2 at {fmt_mi(f.te_snr2, ' ms', 0)}", xa=0.03, ha="left")
    panel_title(Dx, "D")

    save(fig, "FigureS2_noise_t2star")


# =====================================================================
#  S3  rs-fMRI BOLD
# =====================================================================

def figure_s3(D):
    d, sp = D["bold"], D["spectra"]
    if d is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    A, B, Cx, Dx = axes.ravel()

    # --- A: cohort temporal power spectrum, plateau highlighted ------------
    if sp is not None:
        piv = sp.pivot_table(index="freq", columns="participant_id", values="psd_norm")
        piv = piv[piv.index > 0]        # the DC bin is ~0 after detrending
        fq = piv.index.values
        m = piv.median(axis=1).values
        lo, hi = piv.quantile(.25, axis=1).values, piv.quantile(.75, axis=1).values
        A.fill_between(fq, lo, hi, color=ACCENT, alpha=0.20)
        A.plot(fq, m, color=ACCENT, lw=1.6)
        A.axvspan(0.75 * fq[-1], fq[-1], color=ORANGE, alpha=0.13)
        A.axhline(1.0, color=ORANGE, ls="--", lw=1.2)
        A.text(0.02, 1.0, " thermal plateau (noise floor)",
               transform=A.get_yaxis_transform(), ha="left", va="top",
               fontsize=8, color=ORANGE)
        A.set_yscale("log")
        A.set_xlim(0, fq[-1])
        A.set_ylim(0.5 * np.nanmin(lo), 3.0 * np.nanmax(hi))
        A.set_xlabel("frequency (Hz)")
        A.set_ylabel("power (÷ plateau)")
        _note(A, f"TR = {d.tr.median():.2f} s\n"
                 "temporal noise floor", xa=0.97, ha="right", fs=8)
    panel_title(A, "A")

    # --- B: measured tSNR against the ceiling thermal noise alone allows ---
    B.scatter(d.tsnr, d.tsnr_thermal_ceiling, s=32, color=ACCENT, alpha=0.75,
              edgecolor="white", linewidth=0.5)
    lim = [0, max(d.tsnr.max(), d.tsnr_thermal_ceiling.max()) * 1.1]
    B.plot(lim, lim, color=INK, ls="--", lw=1.2)
    B.set_xlim(lim); B.set_ylim(lim)
    B.set_xlabel("measured tSNR")
    B.set_ylabel("tSNR ceiling set by thermal noise")
    B.text(0.97, 0.03,
           f"measured  {fmt_mi(d.tsnr)}\nceiling    {fmt_mi(d.tsnr_thermal_ceiling)}\n\n"
           "points above the line: noise variance",
           transform=B.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.65))
    panel_title(B, "B")

    # --- C: the floor in units of BOLD signal change ----------------------
    _box(Cx, [d.thermal_pct_signal.dropna().values, d.total_pct_signal.dropna().values],
         ["thermal (noise floor)", "total temporal"], "noise (% signal change)")
    Cx.axhspan(0.5, 1.0, color=GREEN, alpha=0.13, zorder=0)
    Cx.text(0.99, 1.0, " typical BOLD response, 0.5–1 %",
            transform=Cx.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=GREEN)
    _note(Cx, f"thermal {fmt_mi(d.thermal_pct_signal, ' %', 2)}\n"
              f"total    {fmt_mi(d.total_pct_signal, ' %', 2)}", xa=0.03, ha="left")
    panel_title(Cx, "C")

    # --- D: how much of the temporal variance is the floor ----------------
    _draw_hist(Dx, 100 * d.thermal_var_frac, "thermal share of temporal variance (%)",
               bins=16, normal=True)
    panel_title(Dx, "D")

    save(fig, "FigureS3_noise_bold")


# =====================================================================
#  S4  DWI
# =====================================================================

def figure_s4(D):
    d = D["dwi"]
    if d is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    A, B, Cx, Dx = axes.ravel()
    shells = sorted(d.shell.unique())
    labs = [f"{int(b)}" for b in shells]

    # --- A: SNR per shell, median voxel and low tail ----------------------
    _box(A, [d[d.shell == b].snr.dropna().values for b in shells], labs, "SNR")
    A.set_xlabel("b-value (s/mm²)")
    A.set_yscale("log")
    _floor_line(A, 1.0)
    A.axhline(2, color=ORANGE, ls="-.", lw=1.0)
    A.text(0.99, 2, " SNR = 2", transform=A.get_yaxis_transform(), ha="right",
           va="bottom", fontsize=8, color=ORANGE)
    _headroom(A, 0.28)
    _note(A, "\n".join(f"b = {int(b)}: {fmt_mi(d[d.shell==b].snr)}" for b in shells)
             + f"\nn = {d.participant_id.nunique()}", xa=0.97, ha="right")
    panel_title(A, "A")

    # --- B: signal decay against the floor --------------------------------
    med = np.array([d[d.shell == b].signal.median() for b in shells])
    lo = np.array([d[d.shell == b].signal.quantile(.25) for b in shells])
    hi = np.array([d[d.shell == b].signal.quantile(.75) for b in shells])
    p10 = np.array([d[d.shell == b].signal_p10.median() for b in shells])
    floor = d.floor.median()
    B.fill_between(shells, lo, hi, color=ACCENT, alpha=0.20)
    B.plot(shells, med, "o-", color=ACCENT, lw=1.8, ms=5, label="median brain voxel")
    B.plot(shells, p10, "s--", color=GRAY, lw=1.5, ms=4.5, label="10th-percentile voxel")
    B.axhline(floor, color=ORANGE, ls="--", lw=1.4)
    B.text(0.99, floor, " noise floor  σ·√(π/2)", transform=B.get_yaxis_transform(),
           ha="right", va="bottom", fontsize=8, color=ORANGE)
    B.set_yscale("log")
    # The signal spans well under a decade, so the default log locator would
    # label only 10^2; place ticks by hand.
    B.set_yticks([floor, 50, 100, 200, 400])
    B.set_yticklabels([f"{floor:.0f}\n(floor)", "50", "100", "200", "400"])
    B.set_xlabel("b-value (s/mm²)"); B.set_ylabel("signal (a.u.)")
    B.legend(fontsize=8, frameon=False, loc="upper right")
    panel_title(B, "B")

    # --- C: the low tail is what approaches the floor ---------------------
    _box(Cx, [d[d.shell == b].snr_p10.dropna().values for b in shells], labs,
         "SNR of the 10th-percentile voxel")
    Cx.set_xlabel("b-value (s/mm²)")
    _floor_line(Cx, 1.0)
    hi_b = d[d.shell == shells[-1]]
    _headroom(Cx, 0.30)
    _note(Cx, f"at b = {int(shells[-1])} the low-signal tail\nreaches SNR "
              f"{fmt_mi(hi_b.snr_p10)},\napproaching the regime where the\n"
              "Rician floor biases the measurement", xa=0.97, ha="right")
    panel_title(Cx, "C")

    # --- D: fraction of brain at the floor --------------------------------
    _box(Dx, [100 * d[d.shell == b].frac_floor.dropna().values for b in shells],
         labs, "brain voxels at the noise floor (%)")
    Dx.set_xlabel("b-value (s/mm²)")
    _headroom(Dx, 0.35)
    _note(Dx, "voxels with signal ≤ 2σ, per volume\n\n"
              + "\n".join(f"b = {int(b)}: {fmt_mi(100*d[d.shell==b].frac_floor, ' %', 3)}"
                          for b in shells), xa=0.03, ha="left")
    panel_title(Dx, "D")

    save(fig, "FigureS4_noise_dwi")


# =====================================================================
#  captions
# =====================================================================

def write_captions(D):
    """Paste-ready captions with every number read back from the CSVs, so the
    text cannot drift from the figures."""
    L = []
    st, g, b, w = D["struct"], D["gre"], D["bold"], D["dwi"]
    L.append("SUPPLEMENTARY FIGURES — NOISE FLOOR BY MODALITY")
    L.append("=" * 78)
    L.append("")
    L.append("Definition used throughout: in magnitude MR images, rectified complex")
    L.append("Gaussian noise leaves a Rayleigh-distributed background whose mean,")
    L.append("sigma*sqrt(pi/2), is the noise floor — the value a voxel takes when the")
    L.append("true signal is zero. A voxel at or below 2*sigma is called noise-floor")
    L.append("dominated (the Rician bias on its magnitude still exceeds ~10%).")
    L.append("")

    if st is not None:
        stt = st.copy()
        stt["proto"] = np.where(stt.contrast == "T1w", "T1w",
                                np.where(stt.sequence.str.contains("3D|SPACE", case=False, na=False),
                                         "FLAIR 3D SPACE", "FLAIR 2D"))
        t1 = stt[stt.proto == "T1w"]
        f2 = stt[stt.proto == "FLAIR 2D"]
        f3 = stt[stt.proto == "FLAIR 3D SPACE"]

    if st is not None:
        L.append("Figure S1. Noise floor of the structural contrasts.")
        L.append("(A) Spatial SNR (median brain signal / sigma), median [IQR]:")
        L.append(f"      T1w             {fmt_mi(t1.snr)}  (n = {len(t1)})")
        L.append(f"      FLAIR 2-D       {fmt_mi(f2.snr)}  (n = {len(f2)})")
        L.append(f"      FLAIR 3-D SPACE {fmt_mi(f3.snr)}  (n = {len(f3)})")
        L.append("    The release contains two FLAIR protocols, which are not")
        L.append("    interchangeable and are reported separately throughout. The 3-D SPACE")
        L.append("    FLAIR is reconstructed at 0.45 mm from a 0.9 mm acquisition; that")
        L.append("    interpolation correlates neighboring voxels and deflates every")
        L.append("    background-free sigma estimate, so its SNR is marked with a dagger and")
        L.append("    is not comparable with the 2-D protocol or with the other modalities.")
        L.append("(B) Distribution of T1w SNR across the cohort with a normal fit.")
        L.append(f"(C) The noise floor as a percentage of median tissue signal: T1w")
        L.append(f"    {fmt_mi(100*t1.floor/t1.signal, ' %', 2)}, FLAIR 2-D {fmt_mi(100*f2.floor/f2.signal, ' %', 2)}.")
        L.append(f"(D) Brain voxels at or below 2*sigma: T1w {fmt_mi(100*t1.frac_floor, ' %', 3)},")
        L.append(f"    FLAIR 2-D {fmt_mi(100*f2.frac_floor, ' %', 3)}. Both structural contrasts sit far")
        L.append("    above their noise floor everywhere in the brain.")
        L.append("")

    if g is not None:
        tes = g.groupby("echo").te_ms.first().values
        e1 = g[g.echo == 1]; e6 = g[g.echo == g.echo.max()]
        gf = g.groupby("participant_id").first()
        L.append("Figure S2. Noise floor of the multi-echo GRE (T2*) acquisition.")
        L.append(f"(A) SNR down the 6-echo train (TE = {', '.join(f'{t:.2f}' for t in tes)} ms), one")
        L.append(f"    thin line per subject (n = {g.participant_id.nunique()}), cohort median in black, with the")
        L.append("    fitted mono-exponential decay extrapolated past the last acquired echo.")
        L.append(f"(B) SNR falls from {fmt_mi(e1.snr)} at TE = {tes[0]:.2f} ms to {fmt_mi(e6.snr)} at")
        L.append(f"    TE = {tes[-1]:.0f} ms — still an order of magnitude above the floor.")
        L.append(f"(C) Brain voxels at or below 2*sigma reach only {fmt_mi(100*e6.frac_floor, ' %', 2)} at the last")
        L.append("    echo, confined to CSF and sinus-adjacent susceptibility dropout.")
        L.append(f"(D) Extrapolating the fitted decay, SNR would reach 5 at TE = {fmt_mi(gf.te_snr5, ' ms', 0)}")
        L.append(f"    and 2 at TE = {fmt_mi(gf.te_snr2, ' ms', 0)}, both far beyond the {tes[-1]:.0f} ms the")
        L.append("    echo train actually samples. The long-T2 component that the released")
        L.append("    data cannot resolve is therefore limited by the sampled TE range, not")
        L.append("    by the noise floor.")
        L.append("")

    if b is not None:
        L.append("Figure S3. Noise floor of the resting-state BOLD acquisition.")
        L.append(f"(A) Cohort temporal power spectrum (n = {len(b)}, TR = {b.tr.median():.2f} s), median and IQR,")
        L.append("    normalized to the white-noise plateau. The flat top of the spectrum")
        L.append("    (shaded) is thermal noise; its height is the temporal noise floor.")
        L.append(f"(B) Measured tSNR {fmt_mi(b.tsnr)} against the ceiling thermal noise alone")
        L.append(f"    would permit, {fmt_mi(b.tsnr_thermal_ceiling)}. The gap is structured noise")
        L.append("    (physiological and motion), not the floor.")
        L.append(f"(C) The floor in BOLD units: thermal noise is {fmt_mi(b.thermal_pct_signal, ' %', 2)} of signal per")
        L.append(f"    volume against a total temporal noise of {fmt_mi(b.total_pct_signal, ' %', 2)}, both above the")
        L.append("    0.5-1 % of a typical BOLD response (shaded) — that is, a BOLD response")
        L.append("    in any single volume is smaller than the thermal noise on that volume,")
        L.append("    and is recovered only by averaging across the run.")
        L.append(f"(D) Thermal noise accounts for {fmt_mi(100*b.thermal_var_frac, ' %', 1)} of the temporal variance.")
        L.append("")

    if w is not None:
        sh = sorted(w.shell.unique())
        L.append("Figure S4. Noise floor of the multi-shell diffusion acquisition.")
        L.append(f"(A) SNR per shell (n = {w.participant_id.nunique()}):")
        for sv in sh:
            L.append(f"      b = {int(sv):>4}: {fmt_mi(w[w.shell==sv].snr)}")
        L.append("(B) Median and 10th-percentile brain signal against b-value, with the")
        L.append("    noise floor sigma*sqrt(pi/2) marked.")
        hi = w[w.shell == sh[-1]]
        L.append(f"(C) The low-signal tail approaches the floor first: at b = {int(sh[-1])} the")
        L.append(f"    10th-percentile voxel reaches SNR {fmt_mi(hi.snr_p10)}.")
        L.append("(D) Brain voxels at or below 2*sigma, per diffusion volume:")
        for sv in sh:
            L.append(f"      b = {int(sv):>4}: {fmt_mi(100*w[w.shell==sv].frac_floor, ' %', 3)}")
        L.append("")

    L.append("Caveat applying to all panels: every sequence uses parallel imaging and")
    L.append("the EPI additionally use multiband, so noise is spatially non-stationary")
    L.append("and a single global sigma summarizes rather than fully describes it. The")
    L.append("estimates are comparable across subjects within a modality.")

    out = FIG / "FigureS1-S4_noise_captions.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"  wrote {out.name}")


def main():
    D = load()
    figure_s1(D)
    figure_s2(D)
    figure_s3(D)
    figure_s4(D)
    write_captions(D)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
