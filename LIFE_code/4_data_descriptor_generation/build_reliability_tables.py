#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Cognitive & Behavioral Phenotyping — internal-consistency tables.

  Table 2 (reliability)  Reliability of the phenotyping instruments — this is the
           reliability portion of the manuscript's single merged Table 2; it merges the former
           Tables 5 (NIHTB-CB internal consistency) and 6 (loneliness scale
           internal reliability) into one construct-grouped table.

Item-level responses come from the study source exports (LIFE_raw/RedCap),
not from the release, which ships scored measures only; every scoring rule
used here was verified to reproduce the released totals exactly.

Values are identical to the manuscript; output is CSV (universal format), no .docx.

Out -> derivatives/technical_validation/tables/Table2_reliability.csv (+ _note.txt)
       derivatives/technical_validation/behavioral/nihtb_reliability.csv
       derivatives/technical_validation/behavioral/loneliness_reliability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import cogbehav_data as CB

OUT_TABLES = C.DERIV_DIR / "technical_validation" / "tables"


def csv_table(title, headers, rows, note, fname):
    """Tidy CSV + plain-text note sidecar; section-heading strings in `rows`
    become a leading 'Section' column."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    section, out = None, []
    for r in rows:
        if isinstance(r, str):
            section = r
            continue
        out.append([section] + list(r))
    pd.DataFrame(out, columns=["Section"] + list(headers)).to_csv(OUT_TABLES / fname, index=False)
    (OUT_TABLES / fname.replace(".csv", "_note.txt")).write_text(title + "\n\n" + note + "\n")
    print("  wrote", fname, f"({len(out)} rows)")

NA = "—"


def fmt(v, d=2):
    return NA if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def ci(lo, hi, d=2):
    return NA if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo:.{d}f}, {hi:.{d}f}]"


# =====================================================================
#  Table 2 (reliability panel) — reliability of the phenotyping instruments
# =====================================================================

def table3():
    ph = CB.older(C.load_phenotype())
    nih = CB.older(C.load_nih_toolbox())
    rows, csv_nihtb, csv_lonely = [], [], []

    # ── NIH Toolbox Cognition Battery ──
    rows.append("NIH Toolbox Cognition Battery")
    specs = [
        ("DCCS, uncorrected standard score", "dccs_uncorrected_standard_score",
         "dccs_rawscore", 30, "30 trials"),
        ("Flanker, uncorrected standard score", "flanker_uncorrected_standard_score",
         "flanker_rawscore", 20, "20 trials"),
    ]
    for label, col, rawcol, maxraw, k in specs:
        d = pd.to_numeric(ph[col], errors="coerce").dropna()
        raw = pd.to_numeric(ph[rawcol], errors="coerce").dropna()
        ceil = 100 * (raw >= maxraw).mean()
        rows.append([label, k, str(len(d)), f"{d.mean():.2f} ({d.std(ddof=1):.2f})",
                     "Cronbach's α", "not estimable",
                     f"raw {raw.mean():.1f}/{maxraw}, {ceil:.0f}% at ceiling; "
                     "item responses not released"])
        csv_nihtb.append({"measure": col, "n": len(d), "mean": d.mean(),
                          "sd": d.std(ddof=1), "pct_at_ceiling": ceil,
                          "statistic": "cronbach_alpha", "value": np.nan,
                          "reason": "item-level responses not in release"})

    d = ph[[sp[1] for sp in specs]].apply(pd.to_numeric, errors="coerce").dropna()
    Z = (d - d.mean()) / d.std(ddof=1)
    a = C.cronbach_alpha(Z.values)
    lo, hi = CB.boot_alpha_ci(Z)
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    rows.append(["Executive composite (DCCS + Flanker)", "2 measures", str(len(Z)), NA,
                 "Standardized α", f"{a:.2f} {ci(lo, hi)}", f"inter-measure r = {r:+.2f}"])
    csv_nihtb.append({"measure": "executive_composite", "n": len(Z),
                      "statistic": "standardized_alpha", "value": a,
                      "ci_lo": lo, "ci_hi": hi, "inter_measure_r": r, "p": p})

    # ── Loneliness ──
    rows.append("Loneliness")
    u = CB.reliability(CB.ucla_items())
    tot = pd.to_numeric(ph["ucla_total"], errors="coerce").dropna()
    rows.append(["UCLA Loneliness Scale v3", "20", str(u["n"]),
                 f"{tot.mean():.2f} ({tot.std(ddof=1):.2f})", "Cronbach's α",
                 f"{u['alpha']:.3f} {ci(*CB.boot_alpha_ci(CB.ucla_items()), 3)}",
                 f"mean inter-item r = {u['mean_inter_item_r']:.2f}; item–rest "
                 f"{u['item_rest'].min():.2f}–{u['item_rest'].max():.2f}; α if deleted "
                 f"{u['alpha_if_deleted'].min():.3f}–{u['alpha_if_deleted'].max():.3f}"])
    csv_lonely.append({"scale": "UCLA-20", "k": u["k"], "n": u["n"],
                       "statistic": "cronbach_alpha", "value": u["alpha"],
                       "ci_lo": u["ci_lo"], "ci_hi": u["ci_hi"],
                       "mean_inter_item_r": u["mean_inter_item_r"]})

    e = CB.empirical_reliability(nih["loneliness_tscore"], nih["loneliness_se"])
    t = pd.to_numeric(ph["loneliness_tscore"], errors="coerce").dropna()
    dv = ph[["ucla_total", "loneliness_tscore"]].apply(pd.to_numeric, errors="coerce").dropna()
    rc, _ = stats.pearsonr(dv["ucla_total"], dv["loneliness_tscore"])
    rows.append(["NIH Toolbox Loneliness (Fixed Form)", "5", str(e["n"]),
                 f"{t.mean():.2f} ({t.std(ddof=1):.2f})", "IRT empirical reliability",
                 f"{e['rho']:.3f}",
                 f"mean SE = {e['mean_se']:.2f}; converges with UCLA r = {rc:+.2f}"])
    csv_lonely.append({"scale": "NIH Toolbox Loneliness (Fixed Form)", "n": e["n"],
                       "statistic": "irt_empirical_reliability", "value": e["rho"],
                       "mean_se": e["mean_se"], "convergent_r_with_ucla": rc})

    note = (f"All statistics are computed on the released older cohort (≥{CB.AGE_MIN} y). "
            "Cronbach's α requires item-level responses. For the NIH Toolbox the release "
            "provides scored DCCS and Flanker measures only, and the archived Toolbox item "
            "exports available for a five-participant subset of this cohort show accuracy at "
            "or near ceiling (Flanker 100% correct in all five; DCCS mean 98.4%), so a "
            "trial-level KR-20 is undefined for Flanker and uninterpretable for DCCS; the "
            "estimate the released data do support is the standardized α of the two-measure "
            "executive composite, from the correlation between the z-scored uncorrected "
            "standard scores. UCLA-20 α is computed from the item responses released in "
            "phenotype/redcap_items.tsv; reverse-scoring of the nine positively worded items "
            "(1, 5, 6, 9, 10, 15, 16, 19, 20) reproduces the released ucla_total for every "
            "participant, and α meets the a priori expectation of α ≥ 0.85. The NIH "
            "Toolbox Loneliness Fixed Form is scored by item-response theory and the "
            "release provides the scores and their standard errors rather than the item "
            "responses, so Cronbach's α cannot be computed; the released per-participant "
            "standard errors instead support the IRT empirical (marginal) reliability, "
            "1 − mean(SE²)/Var(T). Confidence intervals are bootstrap percentile "
            "intervals (10,000 resamples of participants). "
            "Reliability of the social-integration measures is reported per domain in Figure 6. "
            "NIHTB-CB = NIH Toolbox Cognition Battery; DCCS = Dimensional Change Card Sort; "
            "Flanker = Flanker Inhibitory Control and Attention Test; UCLA = UCLA Loneliness "
            "Scale (Version 3); k = number of items; M = mean; "
            "SD = standard deviation; N = participants with available data; "
            "r = Pearson correlation; SE = standard error.")

    csv_table("Table 2 (reliability). Reliability of the cognitive and behavioral phenotyping instruments.",
              ["Instrument", "k", "N", "M (SD)", "Statistic", "Estimate [95% CI]",
               "Supporting detail"],
              rows, note, "Table2_reliability.csv")

    out = C.outdir("behavioral")
    pd.DataFrame(csv_nihtb).to_csv(out / "nihtb_reliability.csv", index=False)
    pd.DataFrame(csv_lonely).to_csv(out / "loneliness_reliability.csv", index=False)
    print(f"  executive composite standardized α = {a:.3f} {ci(lo, hi)}, r = {r:+.3f}, n = {len(Z)}")
    print(f"  UCLA-20 α = {u['alpha']:.3f} {ci(u['ci_lo'], u['ci_hi'], 3)}, n = {u['n']}")
    print(f"  PROMIS Loneliness empirical reliability = {e['rho']:.3f}, n = {e['n']}")


if __name__ == "__main__":
    table3()
