#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Figure 2 — White-matter tissue integrity: T2* and white-matter hyperintensities.

  Row 1: (A) three-subject proof-of-concept panel (fig_shift_t2star.png)
  Row 2: (B) regional T2* across FreeSurfer structures, drawn as split
             violins (left/right hemisphere), filled by the T2*-age
             correlation, with age and sex effects marked
  Row 3: (C) WMH lesion-volume distribution (eTIV-corrected, positively skewed)
         (D) WMH volume x age (rank-based)

Panel B replaces the former 2B/2C pair. Those sampled four hand-placed
squares on a single slice; B now summarizes whole anatomical structures from
each participant's own FreeSurfer segmentation (t2star_regional/).

Uses regional_t2star_per_subject.csv (t2star_regional/extract_regional_t2star.py)
and wmh_per_subject.csv.
Output -> 0 Data manuscript/figures/Figure2.png (300 dpi).
"""
import json
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import make_figures as MF  # shared drawing cores + AGE_MIN + TV
sys.path.insert(0, str(Path(__file__).resolve().parent / "t2star_regional"))
import make_regional_t2star_figure as RG  # panel B drawing core

FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
REGIONAL_CSV = Path(os.environ.get(
    "LIFE_T2STAR_REGIONAL_CSV",
    C.DERIV_DIR / "technical_validation" / "t2star_regional" /
    "regional_t2star_per_subject.csv"))
PARTICIPANTS = Path(os.environ.get(
    "LIFE_PARTICIPANTS", C.BIDS_DIR / "participants.tsv"))
PANEL_A = FIG / "fig_shift_t2star.png"
ACCENT = "#4C72B0"; INK = "#2b2b2b"
TITLE_FS = 15        # panel titles (B, D, E); panel A headers match
LETTER_FS = 22
TITLE_INDENT = 0.021 # figure fraction: left content rule -> title, all panels
WMH_VOL = "wmh_volume_etiv_corr"


def pf(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}"


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=22, fontweight="bold", va="top", ha="left")


def main():
    # WMH (row 3), older cohort only, same convention as the other figures
    ph = C.load_phenotype()
    # WMH is FreeSurfer-derived, so the structural outlier (sub-007) is
    # dropped here (n = 56). The T2* panels above are independent of
    # FreeSurfer and keep their own full subsample.
    older = C.older_ids(ph, exclude=C.STRUCTURAL_EXCLUDE)
    w = pd.read_csv(MF.TV / "wmh/wmh_per_subject.csv").set_index("participant_id")
    ow = w.loc[[i for i in w.index if i in older]].copy()
    ow["age"] = ph.reindex(ow.index)["age"]
    wvol = pd.to_numeric(ow[WMH_VOL], errors="coerce")
    wskew = stats.skew(wvol.dropna())
    mw = ow[[WMH_VOL, "age"]].copy()
    mw["sex_code"] = ph.reindex(ow.index)["sex_code"]
    mw = mw.apply(pd.to_numeric, errors="coerce").dropna()
    # WMH volume is strongly right-skewed, so rank-transform both variables
    # first, then residualize on sex: a partial Spearman correlation.
    rk = pd.DataFrame({"w": stats.rankdata(mw[WMH_VOL]),
                       "a": stats.rankdata(mw["age"]),
                       "sex_code": mw["sex_code"].values})
    pr = C.partial_corr(rk, "w", "a", ["sex_code"])
    wsr, wsp = pr["r"], pr["p"]

    # Sex effect on WMH burden, tested on the same ranks and adjusted for age,
    # so the WMH panels state explicitly what panel B marks per structure.
    Xs = np.column_stack([np.ones(len(rk)), rk["a"].values, rk["sex_code"].values])
    ys = rk["w"].values
    bs, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    rs = ys - Xs @ bs
    dofs = len(ys) - Xs.shape[1]
    s2s = (rs @ rs) / dofs
    ses = np.sqrt(s2s * np.linalg.inv(Xs.T @ Xs)[2, 2])
    wsex_p = 2 * stats.t.sf(abs(bs[2] / ses), dofs)
    sex_note = f"(no sex effect, {pf(wsex_p)}, adj. age)"

    if not PANEL_A.exists():
        raise SystemExit(
            f"panel A image not found: {PANEL_A}\n"
            "Run make_t2star_panel.py first -- it writes fig_shift_t2star.png,\n"
            "which this script embeds as Figure 2A.")
    imgA = mpimg.imread(PANEL_A)
    aspect = imgA.shape[1] / imgA.shape[0]      # width / height

    # explicit inch layout so A fills its width and B+C span exactly that width
    W = 15.5
    LEFT = 0.5
    CW = 14.5                                   # content width (A and B+C share it)
    AW = W                                      # A spans the full figure width
    AH = AW / aspect                            # A height (embedded 1:1)
    GAP_AB = 0.35                               # A -> bottom row
    HR = 6.0                                    # panel B height
    CBAR_GAP = 0.28                             # panel B -> its color bar
    CBAR_W = 0.15
    # Panel B carries a larger y-label and tick labels than C and D, so it
    # needs more room to their left than the shared LEFT margin provides;
    # without this its y-axis is clipped at the figure edge.
    BLEFT = LEFT + 0.50
    BW = CW - CBAR_GAP - CBAR_W - 0.55 - (BLEFT - LEFT)
    DR = 3.5                                    # WMH row height
    GAP_BD = 1.05                               # B/C row -> WMH row
    GAP_DE = 1.30                               # D -> E
    DW = (CW - GAP_DE) / 2                      # D and E each half the content width
    BOTTOM, TOP = 0.5, 0.3
    H = BOTTOM + DR + GAP_BD + HR + GAP_AB + AH + TOP
    yD = BOTTOM
    yB = BOTTOM + DR + GAP_BD
    yA = yB + HR + GAP_AB
    fig = plt.figure(figsize=(W, H))

    def ax_in(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    # ---- (A) 3-subject image ----
    axA = ax_in(0, yA, AW, AH)
    axA.imshow(imgA); axA.axis("off")

    # ---- (B) regional T2*: split violins, one structure per position ----
    axB = ax_in(BLEFT, yB, BW, HR)
    entries = RG.load_entries(csv=REGIONAL_CSV, participants=PARTICIPANTS)
    norm, cmap = RG.draw_panel(axB, entries, colorbar=False,
                               title="Regional T2* and its association with age",
                               title_fs=TITLE_FS, label_fs=11.5,
                               tick_fs=11, marker_fs=9.5, key_fs=11,
                               ylabel_fs=13, lr_fs=14)
    # The color bar gets its own axes: panel positions here are explicit
    # inches, so letting colorbar() steal space from axB would break the
    # alignment the rest of the figure depends on.
    cax = ax_in(BLEFT + BW + CBAR_GAP, yB + HR * 0.13, CBAR_W, HR * 0.74)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("T2*–age correlation (r)", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=9)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")
    cb.outline.set_linewidth(1.2)

    # ---- (C) WMH lesion-volume distribution ----
    axD = ax_in(LEFT, yD, DW, DR)
    MF._draw_hist(axD, wvol, "WMH volume (eTIV-corrected, mm³)",
                  note=f"skew = {wskew:+.2f}\n{sex_note}", normal=True,
                  sci_y=True)
    axD.set_title("WMH lesion-volume distribution", loc="left",
                  fontsize=TITLE_FS, fontweight="bold", pad=8)

    # ---- (D) WMH volume x age (skewed -> rank-based) ----
    ex0 = LEFT + DW + GAP_DE
    axE = ax_in(ex0, yD, DW, DR)
    MF._draw_scatter(axE, mw["age"], mw[WMH_VOL], "Age (years)",
                     "WMH volume (eTIV-corrected, mm³)",
                     stat=f"partial Spearman ρ = {wsr:+.2f}, {pf(wsp)}",
                     covar_note="adj. sex")
    loE, hiE = axE.get_ylim()
    axE.set_ylim(loE, hiE + 0.16 * (hiE - loE))        # headroom for the stats box
    for t in axE.texts:                                # move it to the top-right
        t.set_position((0.97, 0.96)); t.set_va("top")
    axE.set_title("WMH volume × age", loc="left",
                  fontsize=TITLE_FS, fontweight="bold", pad=8)

    # Panel letters sit on each panel's left content rule (the leftmost of its
    # y-axis label and tick labels); every title is indented from that same rule
    # by TITLE_INDENT, so the letter->title gap is identical on every panel
    # instead of tracking each panel's y-label width.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()

    def left_text_x(ax):
        """Figure-x of the leftmost y-axis text (label or tick labels) on `ax`."""
        xs = []
        if ax.yaxis.label.get_text():
            xs.append(ax.yaxis.label.get_window_extent(rend).x0)
        xs += [t.get_window_extent(rend).x0 for t in ax.get_yticklabels() if t.get_text()]
        return inv.transform((min(xs), 0))[0] if xs else None

    def title_obj(ax):
        """The Text actually carrying the title. set_title(loc="left") writes to
        ax._left_title, not ax.title (the empty center title)."""
        for t in (getattr(ax, "_left_title", None), ax.title):
            if t is not None and t.get_text():
                return t
        return ax.title

    def indent_title(ax, x_rule=None):
        """Move `ax`'s left-aligned title to TITLE_INDENT right of `x_rule`
        (its own content rule by default). Returns (rule, title baseline y)."""
        x0 = left_text_x(ax) if x_rule is None else x_rule
        ttl = title_obj(ax)
        bb = ax.get_window_extent(rend)
        x_px = fig.transFigure.transform((x0 + TITLE_INDENT, 0))[0]
        ttl.set_x((x_px - bb.x0) / bb.width)
        fig.canvas.draw()
        return x0, inv.transform((0, ttl.get_window_extent(rend).y0))[1]

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

    def letter_at_title(ax, lab, x_rule=None):
        x0, y0 = indent_title(ax, x_rule)
        fig.text(x0, y0, lab, fontsize=LETTER_FS, fontweight="bold",
                 va="baseline", ha="left")
        return x0, y0

    # B spans the content width; C and D share the row below it. B and C
    # start the left column, so they share one rule (the outer of the two).
    xL = min(left_text_x(axB), left_text_x(axD))
    rules = {id(axB): xL, id(axD): xL, id(axE): left_text_x(axE)}
    for ax in (axB, axD, axE):
        indent_title(ax, rules[id(ax)])

    for row, labs in [((axB,), ("B",)), ((axD, axE), ("C", "D"))]:
        y = align_row(row)
        for ax, lab in zip(row, labs):
            fig.text(rules[id(ax)], y, lab, fontsize=LETTER_FS,
                     fontweight="bold", va="baseline", ha="left")

    # A is an embedded image, so its "titles" are drawn inside the PNG. The
    # sub-figure records where its first header sits; letter A goes on that
    # header's baseline, at the same TITLE_INDENT used by every other panel.
    meta = FIG / "fig_shift_t2star.json"
    if meta.exists():
        m = json.load(open(meta))
        hx = m["header_left_in"] * (AW / m["fig_w_in"])
        hy = yA + m["header_baseline_in"] * (AH / m["fig_h_in"])
        fig.text(hx / W - TITLE_INDENT, hy / H, "A", fontsize=LETTER_FS,
                 fontweight="bold", va="baseline", ha="left")
    else:                                  # metadata missing: fall back to the corner
        letter(fig, xL, (yA + AH) / H, "A")

    out = FIG / "Figure2.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"| T2* n={entries[0]['n']}  WMH skew={wskew:+.2f} "
          f"Spearman rho={wsr:+.2f} {pf(wsp)} n={len(mw)}")


if __name__ == "__main__":
    main()
