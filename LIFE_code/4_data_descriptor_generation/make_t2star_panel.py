#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
SHIFT T2* — proof-of-concept panel for the LIFE data descriptor.

Per subject: a header block (subject/age/WMH + white-matter ROI T2* values in a
2x2 grid), a T1w (top) / FLAIR (bottom) mini stack on the left, and the SHIFT
T2* map (mono-exponential over the 6 GRE echoes, SynthStrip brain extraction) at
a matched bilateral-ventricle slice on the right. Four ROIs are sampled in
anterior WM, posterior deep WM, corpus callosum, and caudate; red dotted squares
mark them on the first subject. Navia colormap (Fuderer et al. 2024); SHIFT
sequence (Wolfe et al. 2020).

Output -> 0 Data manuscript/figures/fig_shift_t2star.png
"""
import json
import sys, os, re, glob, subprocess
import os
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

HERE = Path(__file__).resolve().parent
FIG = C.BASE_DIR / "0 Data manuscript" / "figures"
BIDS = C.BASE_DIR / "bids_dataset_LIFE"
# Working directory for intermediate T2* files (brain masks, any DICOM
# conversions). Defaults to a folder beside the derivatives so a reader can
# reproduce the T2* maps without editing the code; override with LIFE_T2STAR_SCRATCH.
SCRATCH = Path(os.environ.get(
    "LIFE_T2STAR_SCRATCH",
    C.DERIV_DIR / "technical_validation" / "t2star" / "work"))
SCRATCH.mkdir(parents=True, exist_ok=True)
FS_HOME = os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/7.4.1")
NAVIA = ListedColormap(np.loadtxt(HERE / "navia.csv"), name="navia")
TES = np.array([2.68, 8.82, 16.58, 24.35, 40.0, 65.0])
HALF = 4  # ROI half-size in voxels

SUBJECTS = [
    dict(sid="sub-050", age=62, wmh=300),
    dict(sid="sub-035", age=73, wmh=771),
    dict(sid="sub-023", age=83, wmh=8248),
]

# ROI offsets (fraction of brain half-height / half-width) from the ventricle
# centroid, DISPLAY coords (row-0 = top = anterior; col-0 = left). Tune here.
ROIS = {
    "Anterior":  (-0.42, -0.14),
    "Posterior": (+0.48, -0.16),
    "Callosum":  (+0.26, -0.02),
    "Caudate":   (-0.08, -0.10),
}
GRID = [("Anterior", "Posterior"), ("Callosum", "Caudate")]  # 2x2 value layout
NAMES = {"Anterior": "Anterior", "Posterior": "Posterior",
         "Callosum": "Callosum (splenium)", "Caudate": "Caudate"}


def _echo_num(path, marker):
    """Echo index from a filename, tolerating a dcm2niix letter suffix.

    dcm2niix writes a trailing letter when a series needs disambiguating
    (e.g. "_e1a.nii.gz"), so a bare int() on the token raises ValueError and
    silently drops the subject. Take the leading digits instead; return a
    large sentinel when there are none, so such a file sorts last rather
    than killing the sort.
    """
    tok = os.path.basename(path).split(marker)[-1].split(".")[0]
    m = re.match(r"(\d+)", tok)
    return int(m.group(1)) if m else 10**6


def echo_paths(sid):
    """Six-echo MEGRE for `sid`, in echo order.

    Prefers the released BIDS files (sub-NNN_echo-N_MEGRE.nii.gz); falls back
    to the older ME-T2 naming and then to locally converted scratch files.
    """
    for pattern, marker in (
            (BIDS / sid / "anat" / f"{sid}_echo-*_MEGRE.nii.gz", "_echo-"),
            (BIDS / sid / "anat" / f"{sid}_ME-T2_echo-*.nii.gz", "echo-"),
            (SCRATCH / f"{sid}_ME-T2_e*.nii.gz", "_e")):
        p = sorted(glob.glob(str(pattern)), key=lambda x: _echo_num(x, marker))
        if p:
            return p
    return []


def canon(img):
    return nib.as_closest_canonical(img)


def disp(a):
    return np.rot90(a)


def brain_mask(sid, e1):
    mp = SCRATCH / f"{sid}_brainmask.nii.gz"
    if not mp.exists():
        subprocess.run([f"{FS_HOME}/bin/mri_synthstrip", "-i", e1, "-m", str(mp)],
                       env=dict(os.environ, FREESURFER_HOME=FS_HOME), capture_output=True)
    return np.asarray(canon(nib.load(str(mp))).dataobj) > 0


def t2star(data, brain):
    S = np.clip(data[brain], 1e-3, None)
    X = np.vstack([np.ones_like(TES), TES]).T
    beta, *_ = np.linalg.lstsq(X, np.log(S).T, rcond=None)
    vol = np.full(data.shape[:3], np.nan, np.float32)
    vol[brain] = -1.0 / beta[1]
    vol[(vol < 1) | (vol > 250)] = np.nan
    return vol


def ventricle_slice(t2, brain):
    zs = np.where(brain.any((0, 1)))[0]
    xs = np.where(brain.any((1, 2)))[0]; ys = np.where(brain.any((0, 2)))[0]
    lo, hi = zs.min(), zs.max()
    cx = slice(int(xs.min() + .30 * np.ptp(xs)), int(xs.min() + .70 * np.ptp(xs)))
    cy = slice(int(ys.min() + .30 * np.ptp(ys)), int(ys.min() + .70 * np.ptp(ys)))
    best, bc = int(lo + .6 * (hi - lo)), -1
    for z in range(int(lo + .40 * (hi - lo)), int(lo + .80 * (hi - lo)) + 1):
        c = np.nansum((t2[cx, cy, z] > 60) & (t2[cx, cy, z] < 180))
        if c > bc:
            bc, best = c, z
    return best


def roi_centers(t2d, brd):
    """Anchor ROIs to the lateral-ventricle bounding box (frontal horns / atria /
    lateral walls) so placement generalizes across ventricle sizes."""
    rr, cc = np.where(brd)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    H, W = r1 - r0, c1 - c0
    cen = np.zeros_like(brd)
    cen[int(r0 + .28 * H):int(r0 + .74 * H), int(c0 + .30 * W):int(c0 + .70 * W)] = True
    vmask = brd & cen & (t2d > 70) & (t2d < 200)
    vrows, vcols = np.where(vmask.any(1))[0], np.where(vmask.any(0))[0]
    vt, vb = vrows.min(), vrows.max()          # frontal horn (top) / atrium (bottom)
    vl = vcols.min()                            # left ventricle wall
    vc = int(np.argwhere(vmask).mean(0)[1])     # midline (ventricle centroid column)
    Hv = vb - vt
    dv = int(0.17 * Hv)                         # vertical (superior/inferior) step
    dh = max(3, int(0.05 * Hv))                 # small lateral step -> immediately lateral
    col = vl - dh                               # left column shared by Anterior/Caudate/Posterior
    return {
        "Anterior":  (vt - dv, col),            # diagonal superior-left of frontal horn
        "Caudate":   ((vt + vb) // 2, col),     # immediately lateral to ventricle center
        "Posterior": (vb + dv, col),            # diagonal inferior-left of posterior horn
        "Callosum":  (vb - int(0.05 * Hv), vc), # between the ventricles (midline, splenium)
    }


def sample(t2d, rc):
    r, c = rc
    b = t2d[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]
    b = b[np.isfinite(b)]; b = b[(b > 15) & (b < 75)]
    return np.median(b) if b.size else np.nan


def coreg_to_gre(sid, mod):
    """Resample the participant's T1w or FLAIR onto the MEGRE grid.

    The panel overlays T1w and FLAIR beside the T2* map at the same slice, so
    both must sit on the MEGRE voxel grid. This step was previously done by
    hand and its outputs lived in a scratch directory, which meant the panel
    could not be rebuilt from the released data. It is done here instead:
    a 6-dof mri_coreg of the structural onto the first echo, then
    mri_vol2vol onto that same grid. Results are cached in SCRATCH.
    """
    out = SCRATCH / f"{sid}_{mod}_in_gre.nii.gz"
    if out.exists():
        return out
    src = BIDS / sid / "anat" / f"{sid}_{mod}.nii.gz"
    echoes = echo_paths(sid)
    if not src.exists() or not echoes:
        raise FileNotFoundError(f"missing {mod} or MEGRE for {sid}")
    e1 = echoes[0]
    lta = SCRATCH / f"{sid}_{mod}_to_gre.lta"
    env = dict(os.environ, FREESURFER_HOME=FS_HOME,
               PATH=f"{FS_HOME}/bin:" + os.environ.get("PATH", ""))
    if not lta.exists():
        r = subprocess.run([f"{FS_HOME}/bin/mri_coreg", "--mov", str(src),
                            "--ref", str(e1), "--reg", str(lta),
                            "--dof", "6", "--threads", "4"],
                           env=env, capture_output=True, text=True)
        if r.returncode or not lta.exists():
            raise RuntimeError(f"mri_coreg failed for {sid} {mod}: "
                               f"{r.stderr[-300:]}")
    r = subprocess.run([f"{FS_HOME}/bin/mri_vol2vol", "--mov", str(src),
                        "--targ", str(e1), "--lta", str(lta),
                        "--o", str(out)], env=env, capture_output=True,
                       text=True)
    if r.returncode or not out.exists():
        raise RuntimeError(f"mri_vol2vol failed for {sid} {mod}: "
                           f"{r.stderr[-300:]}")
    return out


def slice_gre(path, z):
    return disp(np.asarray(canon(nib.load(str(path))).dataobj, np.float32)[:, :, z])


def win(a):
    v = a[np.isfinite(a) & (a > 0)]
    return (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0, 1)


DRAW_ALL = True  # squares on all subjects (only consulted when DRAW_ROIS)

# The hand-placed ROI squares were how regional T2* was sampled for the old
# Figure 2B/C. Those panels are now built from whole-structure FreeSurfer
# masks (t2star_regional/), so the squares no longer correspond to anything
# reported and are off by default. Set LIFE_T2STAR_DRAW_ROIS=1 to restore
# them (and the header's per-ROI values) for inspection.
DRAW_ROIS = os.environ.get("LIFE_T2STAR_DRAW_ROIS", "0").lower() \
    not in ("0", "", "false", "no")
# per-subject uniform shift of ALL boxes, DISPLAY (drow, dcol): -col = left
SUBJ_SHIFT = {"sub-050": (0, -6)}
# per-subject nudge of the splenium (Callosum) box, DISPLAY (drow, dcol): -row = up, +col = right
CALLOSUM_NUDGE = {"sub-050": (0, 5), "sub-035": (-9, 0), "sub-023": (-9, 0)}


def main():
    # Width matches Figure 2 so the panel embeds 1:1 (no font rescaling).
    W = 15.5
    HEADER_FS = 15    # matches TITLE_FS in make_figure2.py
    S = 1.50          # mini side (in); minis stack = 2*S (touching); brain fills map height 2*S
    MMGAP = 0.06
    # LEFT is Figure 2's title indent, so the headers start on the same vertical
    # rule as the titles of panels B and D; RIGHT_TARGET is the right end of the
    # panel C plots, which the color bar has to reach.
    LEFT, BOTTOM = 0.36, 0.20
    # Header position and figure height depend on whether the per-ROI value
    # block is drawn. With the squares off that block is gone, so the header
    # drops to just above the brains and the panel shortens by its height;
    # keeping the tall layout would leave a band of white space between the
    # header and the images.
    HDR_TOP = 4.62 if DRAW_ROIS else BOTTOM + 2 * S + 0.42
    H = 4.8 if DRAW_ROIS else HDR_TOP + 0.13
    RIGHT_TARGET = 15.0
    MGN = 6           # crop margin around the brain (voxels)
    ASPECT = 0.82     # fixed map width/height so all three maps are the same size

    # ---- pass 1: load and crop every subject, so the horizontal layout can be
    #      solved before anything is drawn ----
    prep = []
    for s in SUBJECTS:
        sid = s["sid"]
        paths = echo_paths(sid)
        data = np.stack([np.asarray(canon(nib.load(p)).dataobj, np.float32) for p in paths], -1)
        brain = brain_mask(sid, paths[0])
        t2 = t2star(data, brain)
        z = ventricle_slice(t2, brain)
        t2d, brd = disp(t2[:, :, z]), disp(brain[:, :, z])
        centers = roi_centers(t2d, brd)
        sdr, sdc = SUBJ_SHIFT.get(sid, (0, 0))       # uniform shift of all boxes
        centers = {n: (r + sdr, c + sdc) for n, (r, c) in centers.items()}
        dr, dc = CALLOSUM_NUDGE.get(sid, (0, 0))     # splenium-specific nudge
        centers["Callosum"] = (centers["Callosum"][0] + dr, centers["Callosum"][1] + dc)
        vals = {n: sample(t2d, rc) for n, rc in centers.items()}

        # crop the T2* map to the brain (fills the map; height = two minis), then
        # expand to a fixed width/height aspect so all maps are the same size
        rows, cols = np.where(brd.any(1))[0], np.where(brd.any(0))[0]
        r0 = max(rows.min() - MGN, 0); r1 = min(rows.max() + MGN, t2d.shape[0])
        c0 = max(cols.min() - MGN, 0); c1 = min(cols.max() + MGN, t2d.shape[1])
        bh, bw = r1 - r0, c1 - c0
        if bw < ASPECT * bh:                          # too narrow -> widen cols
            e = int((ASPECT * bh - bw) / 2)
            c0 = max(c0 - e, 0); c1 = min(c1 + e, t2d.shape[1])
        else:                                         # too wide -> add rows
            e = int((bw / ASPECT - bh) / 2)
            r0 = max(r0 - e, 0); r1 = min(r1 + e, t2d.shape[0])
        mh = 2 * S
        prep.append(dict(s=s, sid=sid, z=z, centers=centers, vals=vals,
                         t2c=t2d[r0:r1, c0:c1], r0=r0, c0=c0,
                         mh=mh, mw=mh * (c1 - c0) / (r1 - r0)))

    GW = [S + MMGAP + q["mw"] for q in prep]          # width of one subject group

    # ---- pass 2: draw at a given inter-group gap ----
    def draw(ggap):
        fig = plt.figure(figsize=(W, H))

        def ax_in(x, y, w, h):
            return fig.add_axes([x / W, y / H, w / W, h / H])

        ims, hdr0, gx = [], None, LEFT
        for k, q in enumerate(prep):
            sid, s, vals, centers = q["sid"], q["s"], q["vals"], q["centers"]
            mw, mh, r0, c0 = q["mw"], q["mh"], q["r0"], q["c0"]
            ax_fl = ax_in(gx, BOTTOM, S, S)                   # FLAIR bottom
            ax_t1 = ax_in(gx, BOTTOM + S, S, S)               # T1w top (touches FLAIR)
            for ax, mod in [(ax_t1, "T1w"), (ax_fl, "FLAIR")]:
                g = slice_gre(coreg_to_gre(sid, mod), q["z"])
                lo, hi = win(g)
                ax.imshow(g, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(0.05, 0.93, mod, transform=ax.transAxes, color="white",
                        fontsize=8, va="top", fontweight="bold")
            xmap = gx + S + MMGAP
            ax_m = ax_in(xmap, BOTTOM, mw, mh)
            im = ax_m.imshow(q["t2c"], cmap=NAVIA, vmin=0, vmax=80, aspect="auto")
            ims.append(im)
            ax_m.set_xticks([]); ax_m.set_yticks([])
            for sp in ax_m.spines.values():
                sp.set_visible(False)
            if DRAW_ROIS and (DRAW_ALL or k == 0):
                for n, (r, c) in centers.items():
                    ax_m.add_patch(Rectangle((c - c0 - HALF, r - r0 - HALF),
                                             2 * HALF, 2 * HALF,
                                             fill=False, ls=":", ec="red", lw=1.8))

            # header block above the group
            x0 = gx / W
            c2 = x0 + 2.7 / W
            t_hdr = fig.text(x0, HDR_TOP / H, f"age {s['age']} · WMH {s['wmh']:,} mm³",
                             fontsize=HEADER_FS, fontweight="bold", va="top")
            if k == 0:
                hdr0 = t_hdr
                if DRAW_ROIS:
                    fig.text(x0, (HDR_TOP - 0.35) / H, "T2* in left white matter (ms):",
                             fontsize=10.5, fontweight="bold", va="top")
            if DRAW_ROIS:
                for row, y in zip(GRID, (3.88, 3.52)):
                    for name, xx in zip(row, (x0, c2)):
                        fig.text(xx, y / H, f"{NAMES[name]} = {vals[name]:.0f}",
                                 fontsize=10.5, va="top")
            gx = xmap + mw + ggap

        cax = fig.add_axes([(gx - ggap + 0.15) / W, BOTTOM / H, 0.10 / W, 2 * S / H])
        fig.colorbar(ims[-1], cax=cax, label="T2* (ms)")
        return fig, hdr0

    # The color bar sits 0.15 in past the last group and is 0.10 in wide; its
    # ticks and axis label add a further ~0.46 in. Solve the gap that lands the
    # whole block on RIGHT_TARGET, then correct from what actually rendered.
    ggap = (RIGHT_TARGET - LEFT - sum(GW) - 0.71) / 2
    for _ in range(4):
        fig, hdr0 = draw(ggap)
        fig.canvas.draw()
        err = RIGHT_TARGET - fig.get_tightbbox(fig.canvas.get_renderer()).x1
        if abs(err) < 0.005:
            break
        plt.close(fig)
        ggap += err / 2                  # cax shifts by 2x the gap change

    # Record the first header's left edge and text baseline (in sub-figure
    # inches) so make_figure2.py can put letter A on exactly that line.
    r = fig.canvas.get_renderer()
    bb = hdr0.get_window_extent(r)
    _w, _h, _d = r.get_text_width_height_descent(
        hdr0.get_text(), hdr0.get_fontproperties(), False)
    json.dump({"header_left_in": bb.x0 / fig.dpi,
               "header_baseline_in": (bb.y1 - _h + _d) / fig.dpi,
               "fig_w_in": W, "fig_h_in": H},
              open(FIG / "fig_shift_t2star.json", "w"), indent=1)

    out = FIG / "fig_shift_t2star.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  | inter-group gap {ggap:.3f} in, "
          f"content {LEFT:.2f}-{RIGHT_TARGET:.2f} in")


if __name__ == "__main__":
    main()
