#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Compute the exact values of the manuscript phenotyping tables and write them as
CSV (universal format), so the tables can be regenerated in any tool. Values are
identical to the manuscript; only the output format differs (no .docx).

Outputs -> derivatives/technical_validation/tables/
    Table2_phenotyping.csv           (+ Table2_phenotyping_note.txt)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

TV = C.DERIV_DIR / "technical_validation"
OUT = C.DERIV_DIR / "technical_validation" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
AGE_MIN = 60


def csv_table(title, headers, rows, note, fname):
    """Write a manuscript table as a tidy CSV plus a plain-text note sidecar.
    Section-heading rows (plain strings in `rows`) become a leading 'Section'
    column so the construct grouping is preserved in the flat file."""
    has_sec = any(isinstance(r, str) for r in rows)
    section, out = None, []
    for r in rows:
        if isinstance(r, str):
            section = r
            continue
        out.append(([section] + list(r)) if has_sec else list(r))
    cols = (["Section"] + list(headers)) if has_sec else list(headers)
    pd.DataFrame(out, columns=cols).to_csv(OUT / fname, index=False)
    (OUT / fname.replace(".csv", "_note.txt")).write_text(title + "\n\n" + note + "\n")
    print("  wrote", fname, f"({len(out)} rows)")


def pf(p):
    return "<.001" if (p == p and p < 0.001) else (f"{p:.3f}" if p == p else "—")


def desc_row(ph, older, col, label):
    """Descriptives and age effect, both on the older cohort (the release).

    The age effect is a zero-order SPEARMAN correlation: sex and race
    were screened against every measure and none survived FDR correction, so no
    demographic covariates are carried (see demographic_covariate_screen.csv);
    and 13 of 16 measures — plus age itself — deviate from normality, so the
    rank statistic is used throughout (see normality_and_rank_check.csv).
    """
    o = ph.loc[[i for i in older if i in ph.index]]
    d = pd.to_numeric(o[col], errors="coerce").dropna()
    if d.empty:
        return None
    r = C.partial_corr(o, col, "age", [], method="spearman")
    return [label, f"{d.mean():.2f} ({d.std(ddof=1):.2f})",
            f"{d.min():.1f}–{d.max():.1f}", str(int(len(d))),
            f"{r['r']:+.2f}" if r['r'] == r['r'] else "—", pf(r['p'])]


def main():
    ph = C.load_phenotype()
    older = ph.index[ph["age"] >= AGE_MIN]
    H = ["Measure", "M (SD)", "Range", "N", "Age \u03c1†", "p"]
    W = [3400, 1500, 1500, 700, 1100, 1100]

    # ── Table 2: merged cognitive + social/mood phenotyping, by construct ──
    SECTIONS = [
        ("Cognition — performance (NIH Toolbox)", [
            ("dccs_uncorrected_standard_score", "DCCS, uncorrected standard"),
            ("dccs_age_corrected_standard_score", "DCCS, age-corrected standard"),
            ("flanker_uncorrected_standard_score", "Flanker, uncorrected standard"),
            ("flanker_age_corrected_standard_score", "Flanker, age-corrected standard"),
        ]),
        ("Cognition — self-report", [
            ("cfi_total", "CFI total (cognitive flexibility)"),
        ]),
        ("Social connectedness", [
            ("ucla_total", "UCLA Loneliness total"),
            ("loneliness_tscore", "PROMIS Loneliness (t)"),
            ("friendship_tscore", "PROMIS Friendship (t)"),
            ("mpss_total", "MSPSS total (perceived support)"),
            ("sni_total", "SIN total (social integration & network size)"),
            ("perceived_hostility_tscore", "PROMIS Perceived Hostility (t)"),
            ("perceived_rejection_tscore", "PROMIS Perceived Rejection (t)"),
        ]),
        ("Mood and health", [
            ("phq_total", "PHQ-9 total"),
            ("bdi2_total", "BDI-II total"),
            ("cdrs_total", "CD-RISC-10 total (resilience)"),
            ("comm_total", "COMM total"),
        ]),
    ]
    rows = []
    for heading, measures in SECTIONS:
        block = [r for c, l in measures if (r := desc_row(ph, older, c, l))]
        if block:
            rows.append(heading)
            rows.extend(block)
    csv_table(f"Table 2. Cognitive, social, and mood/health phenotyping "
               f"(older cohort, \u2265{AGE_MIN} y).", H, rows,
               f"Descriptives and age effects are computed on the released older "
               f"cohort (\u2265{AGE_MIN} y). \u2020Age effect is the Spearman rank "
               f"correlation with age, used because 13 of the 16 measures — and age "
               f"itself — deviated from normality (Shapiro\u2013Wilk p < .05, surviving "
               f"FDR correction). Sex and race were screened against every "
               f"measure and none survived FDR correction, so no demographic covariates "
               f"are carried. Measures are grouped by construct; N varies by measure "
               f"because of item-level missingness. MSPSS total is the mean of the 12 "
               f"items; all other totals are sums.",
               "Table2_phenotyping.csv")

    print("\nTable CSV in:", OUT)


if __name__ == "__main__":
    main()
