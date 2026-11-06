#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Compose Figure 1 of the LIFE data descriptor as a single GridSpec figure,
re-plotting each panel via the shared drawing cores in make_figures.py (so the
panels match their standalone versions but share one coordinate system).

Layout (2 rows x 3 columns):
    A Topological defects  B eTIV distribution   C eTIV x social network size
    D Sex effect eTIV   E Hippocampus (R/L)     F Gray matter / thickness
Geometry:
  - equal panel heights within each row (GridSpec cells);
  - D is a single tall axis filling the bottom-row height (y-range padded);
  - E and F are 2-row sub-grids with identical splits, so the bottom sub-panels
    (left hippocampus, cortical thickness) share the same x-axis baseline as D;
  - within E and F the two sub-panels share a column, so their y-axes align;
  - panel letters placed on a common figure grid (aligned across rows/columns).

Run make_figures.py first is NOT required — this reads the derivative CSVs
directly. Output -> 0 Data manuscript/figures/Figure1.png (300 dpi).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import make_figures as MF  # cores + style constants + AGE_MIN + TV

FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
TITLE_FS = 12
LETTER_FS = 20
TITLE_INDENT = 0.021  # figure fraction: left content rule -> title (matches Figure 2)


def _title(ax, text, fs=TITLE_FS):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=fs, pad=6)


def main():
    ph = C.load_phenotype()
    # FreeSurfer-derived figure: drop the structural outlier (sub-007),
    # so every panel is the n = 56 Technical Validation cohort.
    older = C.older_ids(ph, exclude=C.STRUCTURAL_EXCLUDE)
    TV = MF.TV

    def col(df):
        return df[df["participant_id"].isin(older)] if "participant_id" in df.columns \
            else df.loc[[i for i in df.index if i in older]]

    eul = col(pd.read_csv(TV / "structural/euler_numbers.csv"))["euler_mean"]
    aseg = pd.read_csv(TV / "structural/aseg_volumes.csv").set_index("participant_id")
    aparc = pd.read_csv(TV / "structural/aparc_thickness.csv").set_index("participant_id")
    aparc["gt"] = aparc[[c for c in aparc.columns if "MeanThickness" in c]].mean(axis=1)
    oid = [i for i in aseg.index if i in older]
    aseg_o = aseg.loc[oid]
    gmcol = "CortexVol" if "CortexVol" in aseg_o.columns else "TotalGray"

    etiv = aseg_o["EstimatedTotalIntraCranialVol"] / 1000.0
    gm = aseg_o[gmcol] / 1000.0
    hipR = aseg_o["Right-Hippocampus"] / 1000.0
    hipL = aseg_o["Left-Hippocampus"] / 1000.0
    thick = aparc.loc[oid, "gt"]

    soc = ph.loc[oid].copy()
    soc["eTIV_cm3"] = aseg_o["EstimatedTotalIntraCranialVol"] / 1000.0
    rsoc = C.ancova(soc, "eTIV_cm3", "sni_total", ["age", "sex_code"])
    rsex = C.ancova(soc, "eTIV_cm3", "sex_code", ["age"])

    # Age effects annotated on E and F. Volumes are adjusted for head size and
    # sex (without eTIV in the model the gray-matter effect does not reach
    # significance); thickness needs no head-size term. Education was dropped:
    # it was non-significant in every model (p = 0.06-0.79).
    vol = ph.loc[oid].copy()
    vol["etiv"] = aseg_o["EstimatedTotalIntraCranialVol"]
    vol["hipR"], vol["hipL"] = aseg_o["Right-Hippocampus"], aseg_o["Left-Hippocampus"]
    vol["gm_mm3"] = aseg_o[gmcol]
    vol["thick"] = aparc.loc[oid, "gt"]
    VCOV = ["sex_code", "etiv"]

    def age_note(dv, unit, covars=VCOV, dec=1):
        r = C.ancova(vol, dv, "age", covars)
        adj = "sex, eTIV" if "etiv" in covars else "sex"
        return (f"age: β = {r['beta']:+.{dec}f} {unit}, {MF.pfmt(r['p'])}"
                f"\nη² = {r['partial_eta2']:.2f} (adj. {adj})")

    note_hipR = age_note("hipR", "mm³/y")
    note_hipL = age_note("hipL", "mm³/y")
    note_gm = age_note("gm_mm3", "mm³/y", dec=0)
    note_th = age_note("thick", "mm/y", covars=["sex_code"], dec=3)

    fig = plt.figure(figsize=(15, 9.8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.30],
                  left=0.055, right=0.99, top=0.93, bottom=0.07,
                  wspace=0.30, hspace=0.34)
    SUB_HSPACE = 0.55  # identical gap for E and F so their baselines align

    # ---- top row: three single panels (equal heights) ----
    axA = fig.add_subplot(gs[0, 0])
    MF._draw_hist(axA, eul, "Topological defects / hemisphere", normal=True)
    _title(axA, "Cortical surface quality (topological defects)")

    axB = fig.add_subplot(gs[0, 1])
    MF._draw_hist(axB, etiv, "eTIV (cm³)", normal=True, sci_y=True, sw_full=False)
    _title(axB, "Intracranial volume (eTIV)")

    axC = fig.add_subplot(gs[0, 2])
    MF._draw_partial(axC, soc, "eTIV_cm3", "sni_total", ["age", "sex_code"],
                     "Social network index (residual)", "eTIV (cm³, residual)",
                     rsoc["beta"], rsoc["p"], rsoc["n"], beta_unit=" cm³/pt",
                     ann_pos=(0.97, 0.15))
    _title(axC, "eTIV × social network size")

    # ---- bottom row ----
    # D: single tall axis filling the row height, y-range padded
    axD = fig.add_subplot(gs[1, 0])
    MF._draw_box(axD, soc, "eTIV_cm3", "sex", ["F", "M"], ["Female", "Male"],
                 "eTIV (cm³)", rsex["beta"], rsex["p"], rsex["n"],
                 eta2=rsex.get("partial_eta2"), beta_unit=" cm³",
                 covar_note="adj. age", ylim_pad=0.10, sig_bar=True)
    _title(axD, "Sex effect on eTIV")

    # E: right hippocampus atop left (shared split with F)
    subE = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=SUB_HSPACE)
    axE1 = fig.add_subplot(subE[0])
    axE2 = fig.add_subplot(subE[1], sharex=axE1)
    MF._draw_hist_stack([axE1, axE2], [hipR, hipL], ["Right", "Left"],
                        "Hippocampal volume (cm³)", normal=True,
                        extras=[note_hipR, note_hipL], sw_full=False)
    _title(axE1, "Hippocampal volume")

    # F: gray matter atop cortical thickness (same split as E)
    subF = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2], hspace=SUB_HSPACE)
    axF1 = fig.add_subplot(subF[0])
    axF2 = fig.add_subplot(subF[1])
    MF._draw_hist(axF1, gm, "Cortical gray-matter volume (cm³)", normal=True, fs=8,
                  sci_y=True, extra=note_gm, sw_full=False)
    _title(axF1, "Cortical gray-matter volume")
    MF._draw_hist(axF2, thick, "Global cortical thickness (mm)", normal=True, fs=8,
                  extra=note_th, sw_full=False)
    _title(axF2, "Cortical thickness")

    # E and F carry an extra annotation line, so lift the y-limit to keep the
    # stats box clear of the distributions.
    for ax in (axE1, axE2, axF1, axF2):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi - lo) * 1.55)

    # ---- panel letters and titles on a common rule (same scheme as Figure 2) ----
    # The letter sits on the panel's left content rule (the leftmost of its
    # y-axis label and tick labels); the title is indented from that same rule
    # by TITLE_INDENT, so the letter->title gap is identical on every panel.
    # Panels in a column share one rule so their letters line up vertically.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()

    def title_obj(ax):
        """set_title(loc="left") writes to ax._left_title, not ax.title."""
        for t in (getattr(ax, "_left_title", None), ax.title):
            if t is not None and t.get_text():
                return t
        return ax.title

    def left_text_x(ax):
        xs = []
        if ax.yaxis.label.get_text():
            xs.append(ax.yaxis.label.get_window_extent(rend).x0)
        xs += [t.get_window_extent(rend).x0 for t in ax.get_yticklabels() if t.get_text()]
        return inv.transform((min(xs), 0))[0] if xs else None

    def align_row(axes):
        """Put every title in `axes` on their highest common baseline and freeze
        it. Without this a panel carrying a scientific-notation offset text
        (the "1e-4" above the y-axis) has its title lifted by matplotlib's
        auto-positioning, so it no longer lines up with its neighbors."""
        fig.canvas.draw()
        y = max(inv.transform((0, title_obj(ax).get_window_extent(rend).y0))[1]
                for ax in axes)
        y_px = fig.transFigure.transform((0, y))[1]
        for ax in axes:
            ttl = title_obj(ax)
            ttl.set_y(ttl.get_position()[1]
                      + (y_px - ttl.get_window_extent(rend).y0)
                      / ax.get_window_extent(rend).height)
            ax._autotitlepos = False        # keep matplotlib from re-lifting it
        fig.canvas.draw()
        return y

    def indent_title(ax, x_rule=None):
        x0 = left_text_x(ax) if x_rule is None else x_rule
        ttl = title_obj(ax)
        bb = ax.get_window_extent(rend)
        x_px = fig.transFigure.transform((x0 + TITLE_INDENT, 0))[0]
        ttl.set_x((x_px - bb.x0) / bb.width)
        fig.canvas.draw()
        return x0, inv.transform((0, ttl.get_window_extent(rend).y0))[1]

    cols = [((axA, axD), ("A", "D")),
            ((axB, axE1), ("B", "E")),
            ((axC, axF1), ("C", "F"))]

    # x: one rule per column, shared by the panel above and below.
    rules = [min(left_text_x(t), left_text_x(b)) for (t, b), _ in cols]
    for ((top, bot), _), rule in zip(cols, rules):
        indent_title(top, rule)
        indent_title(bot, rule)

    # y: one baseline per row, so every letter in a row sits on the same line.
    y_top = align_row([t for (t, _), _ in cols])
    y_bot = align_row([b for (_, b), _ in cols])
    for ((top, bot), (lt, lb)), rule in zip(cols, rules):
        for ax, lab, y in ((top, lt, y_top), (bot, lb, y_bot)):
            fig.text(rule, y, lab, fontsize=LETTER_FS, fontweight="bold",
                     va="baseline", ha="left", color="#111111")

    indent_title(axF2)          # sub-panel title, no letter of its own

    out = FIG / "Figure1.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
