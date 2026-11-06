#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Figure 3 — Resting-state fMRI quality.

  (A) framewise-displacement trajectory across the run, one thin line per
      participant with the cohort mean and its 95% CI;
  (B) sliding-window tSNR in gray matter and in deep cerebral white matter,
      side by side, drawn the same way;
  (C) group-average whole-run tSNR map in MNI152 2 mm space (axial montage,
      left) and the same six slices restricted to the group gray-matter mask
      (right).

Uses fd_timeseries/, tsnr_windows/, tsnr_mni/ and group_tissue_masks/ (from
cohort_fd.py, cohort_tsnr.py and make_group_tissue_masks.py). The per-axis drawing cores are reused
from make_fd_plot.py and make_tsnr_plot.py so the composite and the standalone
figures cannot drift apart.

Output -> 0 Data manuscript/figures/Figure3.png (300 dpi).
"""
import sys, glob
import os
from pathlib import Path
import numpy as np, pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import make_tsnr_plot as T
import make_fd_plot as F

FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
# This script does not import make_figures, so nothing else creates the
# output folder for it; savefig fails on a fresh checkout without this.
FIG.mkdir(parents=True, exist_ok=True)
RS = C.DERIV_DIR / "technical_validation" / "rsfmri"
CSV = RS / "tsnr_tissue_per_subject.csv"
MNIDIR = RS / "tsnr_mni"
MNI_BRAIN = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / \
    "data/standard/MNI152_T1_2mm_brain.nii.gz"
GM_C, WM_C = "#4C72B0", "#55A868"
INK = "#2b2b2b"
TITLE_FS = 13        # panel titles (match Figure 1's enlarged titles)
LETTER_FS = 20
TITLE_INDENT = 0.021 # figure fraction: content rule -> title (as Fig 1/2)
SLICES = [20, 28, 36, 44, 52, 60]      # axial z indices in MNI152 2 mm (2 rows x 3)
VMAX = 100                             # fixed scale, comparable across releases
MASKDIR = RS / "group_tissue_masks"
# Display consensus: show a voxel where at least this fraction of participants'
# own gray-matter masks cover it (T1->MNI is affine, so masks overlap only
# approximately and a strict 50% consensus erodes the ribbon).
CONSENSUS = 0.30


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=20, fontweight="bold", va="top", ha="left")


def _group_mean_map():
    """Mean of the per-participant whole-run tSNR maps in MNI space."""
    files = sorted(glob.glob(str(MNIDIR / "*_tsnr_mni.nii.gz")))
    if not files:
        return None, 0
    acc = None
    for f in files:
        d = np.asarray(nib.load(f).dataobj, dtype=np.float32)
        acc = d if acc is None else acc + d
    mean = acc / len(files)
    brain = np.asarray(nib.load(str(MNI_BRAIN)).dataobj) > 0
    return np.where(brain, mean, np.nan), len(files)


def _montage(vol, slices, ncol):
    tiles = [np.rot90(vol[:, :, z]) for z in slices]
    rows = [np.concatenate(tiles[i:i + ncol], axis=1)
            for i in range(0, len(tiles), ncol)]
    return np.concatenate(rows, axis=0)


def panel_c_whole(ax, mean):
    """Whole-brain group-average tSNR, six axial slices in a 2 x 3 montage."""
    im = ax.imshow(_montage(mean, SLICES, 3), cmap="magma", vmin=0, vmax=VMAX,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Group-average tSNR, MNI152 2 mm",
                 loc="left", fontsize=TITLE_FS, fontweight="bold")
    return im


def panel_c_tissue(ax, fig, mean):
    """The same six slices, restricted to the group gray-matter mask."""
    f = MASKDIR / "gm_prob_mni.nii.gz"
    if not f.exists():
        ax.text(0.5, 0.5, "run make_group_tissue_masks.py",
                transform=ax.transAxes, ha="center"); ax.axis("off"); return None
    m = np.asarray(nib.load(str(f)).dataobj) >= CONSENSUS
    im = ax.imshow(_montage(np.where(m, mean, np.nan), SLICES, 3), cmap="magma",
                   vmin=0, vmax=VMAX, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Gray matter only (\u2265{CONSENSUS:.0%} of participants' masks)",
                 loc="left", fontsize=TITLE_FS, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, shrink=0.85)
    cb.set_label("tSNR", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im


def main():
    df = pd.read_csv(CSV)
    fig = plt.figure(figsize=(13.0, 14.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15],
                  hspace=0.40, wspace=0.22,
                  left=0.075, right=0.945, top=0.955, bottom=0.050)
    axA = fig.add_subplot(gs[0, :])
    axB1 = fig.add_subplot(gs[1, 0]); axB2 = fig.add_subplot(gs[1, 1])
    axC1 = fig.add_subplot(gs[2, 0]); axC2 = fig.add_subplot(gs[2, 1])

    # A — framewise displacement
    x, M, n_fd = F.load()
    F.draw(axA, x, M, title="Framewise displacement", title_size=TITLE_FS,
           stats_y=0.72)
    axA.set_xlabel("Volume number (TR = 800 ms)")
    # lift A's title (and, with it, the letter placed on its baseline) clear of
    # the top spine and the 1.0 tick label
    axA.set_title("Framewise displacement", loc="left", fontsize=TITLE_FS,
                  fontweight="bold", pad=16)

    # B — sliding-window tSNR, gray matter and deep cerebral white matter
    ylims = []
    for ax, tissue, color in ((axB1, "gm", GM_C), (axB2, "wm", WM_C)):
        xw, Mw, _ = T.load(tissue)
        if Mw is None:
            ax.text(0.5, 0.5, f"no {tissue} windows", transform=ax.transAxes, ha="center")
            continue
        T.draw(ax, xw, Mw, color, T.TISSUE[tissue][0])
        (ax._left_title if ax._left_title.get_text() else ax.title)\
            .set_fontsize(TITLE_FS)
        ax.set_xlabel("Window center (window size = 60 volumes)")
        ylims.append(ax.get_ylim())
    if len(ylims) == 2:                      # shared scale so the tissues compare
        lo = min(a for a, _ in ylims); hi = max(b for _, b in ylims)
        axB1.set_ylim(lo, hi); axB2.set_ylim(lo, hi)
    axB1.legend(frameon=False, fontsize=8, loc="upper left")

    # C is one panel across the row: the whole-brain map and the same map
    # restricted to each tissue, on a common 0-VMAX scale.
    mean, n_maps = _group_mean_map()
    panel_c_whole(axC1, mean)
    panel_c_tissue(axC2, fig, mean)

    # Letters sit on the left content rule of the left-column panels (A, B1,
    # C1); each title is indented from its own rule by TITLE_INDENT, so the
    # letter->title gap is identical everywhere (same scheme as Figures 1-2).
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()

    def title_obj(ax):
        for t in (getattr(ax, "_left_title", None), ax.title):
            if t is not None and t.get_text():
                return t
        return ax.title

    def left_text_x(ax):
        """Figure-x of the leftmost y-axis text; for the image panels (no y
        label or ticks) fall back to the left edge of the axes."""
        xs = []
        if ax.yaxis.label.get_text():
            xs.append(ax.yaxis.label.get_window_extent(rend).x0)
        xs += [t.get_window_extent(rend).x0 for t in ax.get_yticklabels() if t.get_text()]
        px = min(xs) if xs else ax.get_window_extent(rend).x0
        return inv.transform((px, 0))[0]

    def indent_title(ax, x_rule=None):
        x0 = left_text_x(ax) if x_rule is None else x_rule
        ttl = title_obj(ax)
        bb = ax.get_window_extent(rend)
        x_px = fig.transFigure.transform((x0 + TITLE_INDENT, 0))[0]
        ttl.set_x((x_px - bb.x0) / bb.width)
        fig.canvas.draw()
        return x0, inv.transform((0, ttl.get_window_extent(rend).y0))[1]

    # left-column panels share one rule, so A/B/C letters line up vertically
    ruleL = min(left_text_x(axA), left_text_x(axB1), left_text_x(axC1))
    for ax, lab in ((axA, "A"), (axB1, "B"), (axC1, "C")):
        x0, y0 = indent_title(ax, ruleL)
        fig.text(x0, y0, lab, fontsize=LETTER_FS, fontweight="bold",
                 va="baseline", ha="left")
    for ax in (axB2, axC2):                 # right-column titles: own rule, no letter
        indent_title(ax)

    out = FIG / "Figure3.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"| FD n={n_fd}  tSNR n={len(df)}  MNI maps n={n_maps}")


if __name__ == "__main__":
    main()
