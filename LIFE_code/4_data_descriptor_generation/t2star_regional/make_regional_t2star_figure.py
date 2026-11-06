#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Regional T2* across the cohort: one panel replacing Figure 2B and 2C.

Each anatomical structure is drawn as a split violin -- left hemisphere on the
left half, right hemisphere on the right -- so the distribution and the
left/right agreement are visible at once. Structures without a hemispheric
division (corpus callosum, brainstem) are drawn as a single violin.

The fill color carries what Figure 2C used to show separately: the Pearson
correlation between T2* and age in that structure. An asterisk just above a
violin marks an age effect surviving Benjamini-Hochberg FDR across all tested
structures. Above that, a diamond marks an age-adjusted SEX effect surviving
the same correction, annotated with Cohen's d (positive = higher T2* in men).
Age is carried by color, so sex is given its own marker and its value is
printed rather than encoded.

Structures are ordered anatomically (white matter, cortex, deep gray nuclei,
medial temporal, cerebellum, brainstem), never by effect size.

Reads  regional_t2star_per_subject.csv  (extract_regional_t2star.py)
Writes figures/t2star_regional/Figure2BC_regional_t2star.png (300 dpi)
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402  (path must be set first)

BASE = C.BASE_DIR
CSV = Path(os.environ.get(
    "LIFE_T2STAR_REGIONAL_CSV",
    C.DERIV_DIR / "technical_validation" / "t2star_regional" /
    "regional_t2star_per_subject.csv"))
PARTICIPANTS = Path(os.environ.get("LIFE_PARTICIPANTS", C.PARTICIPANTS_TSV))
OUT = Path(os.environ.get("LIFE_T2STAR_REGIONAL_FIG",
                          BASE / "0_Data_Manuscript" / "figures" / "t2star_regional"))

INK = "#2b2b2b"

# (display name, left column, right column). A None column means the
# structure has no hemispheric division and is drawn as one violin.
STRUCTURES = [
    ("Cerebral\nwhite matter", "cerebral_wm_l",       "cerebral_wm_r"),
    ("Corpus\ncallosum",       "corpus_callosum",     None),
    ("Cerebral\ncortex",       "cerebral_cortex_l",   "cerebral_cortex_r"),
    ("Thalamus",               "thalamus_l",          "thalamus_r"),
    ("Caudate",                "caudate_l",           "caudate_r"),
    ("Putamen",                "putamen_l",           "putamen_r"),
    ("Pallidum",               "pallidum_l",          "pallidum_r"),
    ("Hippocampus",            "hippocampus_l",       "hippocampus_r"),
    ("Amygdala",               "amygdala_l",          "amygdala_r"),
    ("Cerebellar\ncortex",     "cerebellar_cortex_l", "cerebellar_cortex_r"),
    ("Cerebellar\nwhite matter", "cerebellar_wm_l",   "cerebellar_wm_r"),
    ("Brainstem",              "brainstem",           None),
]


def fdr(p):
    """Benjamini-Hochberg q-values, order preserved."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q


def half_violin(ax, data, pos, side, facecolor, width=0.78):
    """One half of a split violin at x = pos. side is 'l', 'r' or 'both'."""
    parts = ax.violinplot([data], positions=[pos], widths=width,
                          showextrema=False, showmedians=False)
    for b in parts["bodies"]:
        v = b.get_paths()[0].vertices
        if side == "l":
            v[:, 0] = np.clip(v[:, 0], -np.inf, pos)
        elif side == "r":
            v[:, 0] = np.clip(v[:, 0], pos, np.inf)
        b.set_facecolor(facecolor)
        b.set_edgecolor("#ffffff")
        b.set_linewidth(0.8)
        b.set_alpha(1.0)
    # median tick on the correct side
    med = float(np.median(data))
    x0, x1 = {"l": (pos - width / 2.6, pos),
              "r": (pos, pos + width / 2.6),
              "both": (pos - width / 2.6, pos + width / 2.6)}[side]
    ax.plot([x0, x1], [med, med], color=INK, lw=1.4, zorder=5,
            solid_capstyle="butt")


def load_entries(csv=None, participants=None):
    """Per-hemisphere T2* values with their age and sex statistics.

    Returned as a list of dicts, one per violin half, in the anatomical order
    of STRUCTURES. Factored out of main() so Figure 2 and the standalone
    figure draw from exactly the same numbers.
    """
    df = pd.read_csv(csv or CSV)
    df = df[df["status"] == "ok"]

    # Sex is read from participants.tsv and coded 1 = male.
    part = pd.read_csv(participants or PARTICIPANTS, sep="\t")
    df = df.merge(part[["participant_id", "sex"]], on="participant_id",
                  how="left")

    # Age association and age-adjusted sex effect for every hemisphere entry;
    # FDR applied separately within each family of tests.
    entries, p_age, p_sex = [], [], []
    for name, lcol, rcol in STRUCTURES:
        for side, col in (("l", lcol), ("r", rcol)):
            if col is None:
                continue
            s = df[["age", "sex", col]].dropna()
            r, p = stats.pearsonr(s["age"], s[col])

            # T2* ~ 1 + age + sex, so the sex effect is adjusted for age.
            male = (s["sex"].astype(str).str.upper().str[0] == "M").astype(float)
            X = np.column_stack([np.ones(len(s)), s["age"].values, male.values])
            y = s[col].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = len(s) - X.shape[1]
            s2 = (resid @ resid) / dof
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
            t = beta[2] / se
            ps = 2 * stats.t.sf(abs(t), dof)
            d = beta[2] / np.sqrt(s2)          # age-adjusted standardized diff

            entries.append({"name": name, "side": side, "col": col,
                            "vals": y, "r": r, "p": p, "n": len(s),
                            "sex_d": d, "sex_p": ps})
            p_age.append(p)
            p_sex.append(ps)
    for e, q, qs in zip(entries, fdr(p_age), fdr(p_sex)):
        e["q"] = q
        e["sex_q"] = qs
    return entries


def draw_panel(ax, entries, fig=None, colorbar=True, title=None,
               title_fs=12.5, label_fs=9.5, key=True, tick_fs=9,
               marker_fs=6.6, key_fs=8.5, ylabel_fs=11, lr_fs=11):
    """Draw the split-violin panel on `ax`.

    Used both by the standalone figure and by Figure 2, so the two cannot
    drift apart. `fig` is needed only when colorbar is True.
    """
    rmax = max(abs(e["r"]) for e in entries)
    lim = np.ceil(rmax * 10) / 10
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = matplotlib.colormaps["RdBu_r"]

    by_name = {}
    for e in entries:
        by_name.setdefault(e["name"], []).append(e)

    xpos, labels = [], []
    for i, (name, lcol, rcol) in enumerate(STRUCTURES):
        es = by_name[name]
        bilateral = rcol is not None
        for e in es:
            side = e["side"] if bilateral else "both"
            half_violin(ax, e["vals"], i, side, cmap(norm(e["r"])))
        xpos.append(i)
        labels.append(name)

    # Markers sit just above their own violin, not on a shared rule, so the
    # reader attaches them to the right half. Age star low, sex diamond above.
    ymax = max(np.max(e["vals"]) for e in entries)
    ymin = min(np.min(e["vals"]) for e in entries)
    span = ymax - ymin
    for i, (name, lcol, rcol) in enumerate(STRUCTURES):
        for e in by_name[name]:
            dx = 0 if rcol is None else (-0.19 if e["side"] == "l" else 0.19)
            vtop = float(np.max(e["vals"]))
            if e["q"] < 0.05:
                ax.text(i + dx, vtop + span * 0.018, "*", ha="center",
                        va="bottom", fontsize=marker_fs * 2.1, color=INK,
                        fontweight="bold")
            if e["sex_q"] < 0.05:
                # Value on top, marker beneath it, so the number reads as the
                # label of the violin half directly below the diamond. When
                # the same half also carries an age asterisk, the block is
                # lifted clear of it.
                ax.text(i + dx, vtop + span * (0.125 if e["q"] < 0.05
                                               else 0.062),
                        f"{e['sex_d']:+.2f}\n\u25c6", ha="center",
                        va="bottom", fontsize=marker_fs, color="#1f6f54",
                        linespacing=0.95, multialignment="center")

    # Name the halves once, above the first structure, clear of whatever
    # markers that structure carries.
    first = by_name[STRUCTURES[0][0]]
    top = max(float(np.max(e["vals"]))
              + span * (0.205 if e["sex_q"] < 0.05 and e["q"] < 0.05
                        else 0.145 if e["sex_q"] < 0.05 else 0.035)
              for e in first)
    for e in first:
        dx = -0.19 if e["side"] == "l" else 0.19
        ax.text(0 + dx, top + span * 0.025, e["side"].upper(), ha="center",
                va="bottom", fontsize=lr_fs, fontweight="bold", color=INK,
                zorder=6)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=label_fs)
    ax.set_xlim(-0.75, len(STRUCTURES) - 0.25)
    ax.set_ylim(ymin - span * 0.06, ymax + span * 0.31)
    ax.set_ylabel("T2* (ms)", fontsize=ylabel_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", color="#e9ecef", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    n = entries[0]["n"]
    ax.set_title(title or ("Regional T2* and its association with age "
                           f"(n = {n})"),
                 loc="left", fontweight="bold", fontsize=title_fs, pad=12)

    # left/right key
    if key:
        # The two symbols are drawn as their own Text objects so the key
        # shows them at exactly the color and size used on the violins;
        # a single multi-line string could not vary either.
        kx, ky, dy, ind = 0.012, 0.995, 0.047, 0.016
        ax.text(kx, ky, "Within each structure: left half = left hemisphere, "
                "right half = right hemisphere", transform=ax.transAxes,
                ha="left", va="top", fontsize=key_fs, color="#5a6168")
        ax.text(kx, ky - dy, "*", transform=ax.transAxes, ha="left",
                va="top", fontsize=marker_fs * 2.1, color=INK,
                fontweight="bold")
        ax.text(kx + ind, ky - dy, "age effect, q < 0.05 (FDR)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=key_fs, color="#5a6168")
        ax.text(kx, ky - 2 * dy, "\u25c6", transform=ax.transAxes,
                ha="left", va="top", fontsize=marker_fs, color="#1f6f54")
        ax.text(kx + ind, ky - 2 * dy,
                "age-adjusted sex effect, q < 0.05 (FDR); value is Cohen's "
                "d, positive = higher T2* in men", transform=ax.transAxes,
                ha="left", va="top", fontsize=key_fs, color="#5a6168")

    if colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.022)
        cb.set_label("T2*–age correlation (r)", fontsize=10.5,
                     fontweight="bold")
        cb.ax.tick_params(labelsize=8.5)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight("bold")
        cb.outline.set_linewidth(1.2)
    return norm, cmap


def main():
    entries = load_entries()
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    draw_panel(ax, entries, fig=fig)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "Figure2BC_regional_t2star.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)

    print("wrote", out)
    print(f"{'structure':24s}{'side':>5s}{'mean':>7s}{'r':>8s}{'p':>8s}"
          f"{'q':>8s}   {'sex_d':>8s}{'sex_q':>8s}")
    for e in entries:
        print(f"{e['name'].replace(chr(10),' '):24s}{e['side']:>5s}"
              f"{np.mean(e['vals']):7.1f}{e['r']:+8.3f}{e['p']:8.3f}{e['q']:8.3f}"
              f"{'  *' if e['q'] < 0.05 else '   '}"
              f"{e['sex_d']:+8.2f}{e['sex_q']:8.3f}"
              f"{'  D' if e['sex_q'] < 0.05 else ''}")


if __name__ == "__main__":
    main()
