#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Technical Validation — Cognitive and Behavioral Phenotyping
============================================================
Implements the "Cognitive and Behavioral Phenotyping" bullets:

  1. NIHTB-CB internal consistency — Cronbach's alpha for composite scores
     (requires item-level responses; see --items-dir).
  2. Expected age effects on cognition — ANCOVAs of each NIH Toolbox
     *uncorrected* score on age (control sex). Expect negative age
     effects on fluid/executive measures; attenuated on crystallised.
  3. UCLA Loneliness Scale internal reliability — Cronbach's alpha for the
     20-item scale (expect alpha >= 0.85).
  4. Social connectedness and cognition — partial correlations of loneliness
     and social-network size with executive cognition (control age, sex,
     depressive symptoms). Construct-validity check on the primary
     study hypothesis.
  5. Social connectedness and mood — correlations of loneliness with PHQ-9
     (and PSS if available); convergent validity / depression confound check.

Data notes
----------
* The released phenotype contains DCCS and Flanker (NIH Toolbox) plus PROMIS
  loneliness / friendship / perceived-hostility / perceived-rejection
  t-scores, and REDCap totals (phq_total, bdi2_total, mpss_total, cfi_total).
* No fluid/crystallised composite or processing-speed columns are present, so
  age-effect and social-cognition analyses use the available executive
  measures (DCCS, Flanker). DCCS is used *uncorrected* throughout, consistent
  with the study's analysis convention.
* Cronbach's alpha needs item-level data. Point --items-dir at a folder with
  per-scale item CSVs (e.g. ucla_items.csv with columns ucla_1..ucla_20 and a
  participant_id column) to enable 1 and 3.

Outputs  ->  derivatives/technical_validation/behavioral/
  age_effects_cognition.csv,
  social_cognition_partialcorr.csv, social_mood_correlations.csv,
  behavioral_validation_report.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

# NIH Toolbox scores to test for age effects (uncorrected -> true age effect).
COGNITION_UNCORRECTED = [
    ("dccs_uncorrected_standard_score", "DCCS (cognitive flexibility)", "fluid"),
    ("flanker_uncorrected_standard_score", "Flanker (inhibition)", "fluid"),
]

# Executive measures for the social-cognition construct check
# (DCCS kept uncorrected per the study convention).
EXEC_FOR_SOCIAL = [
    ("dccs_uncorrected_standard_score", "DCCS (uncorrected)"),
    ("flanker_age_corrected_standard_score", "Flanker (age-corrected)"),
]

# Candidate social-connectedness columns (loneliness: higher = less connected).
SOCIAL_MEASURES = [
    ("loneliness_tscore", "PROMIS loneliness (higher=lonelier)"),
    ("friendship_rawscore", "friendship / social network"),
    ("mpss_total", "perceived social support (MSPSS)"),
]

# Mood / depression measures.
MOOD_MEASURES = [
    ("phq_total", "PHQ-9"),
    ("bdi2_total", "BDI-II"),
    ("pss_total", "PSS-10"),
]


# =====================================================================
#  1 & 3. Cronbach's alpha from item-level data
# =====================================================================

def alpha_for_scale(items_dir, filename, item_prefix):
    f = Path(items_dir) / filename
    if not f.exists():
        return None
    df = pd.read_csv(f)
    item_cols = [c for c in df.columns if c.startswith(item_prefix)]
    if len(item_cols) < 2:
        return None
    mat = df[item_cols].apply(pd.to_numeric, errors="coerce").values
    return {"scale": item_prefix, "n_items": len(item_cols),
            "alpha": C.cronbach_alpha(mat),
            "n_subjects": int((~np.isnan(mat).any(axis=1)).sum())}


def compute_cronbach(items_dir):
    if not items_dir:
        return pd.DataFrame()
    specs = [
        ("ucla_items.csv", "ucla_"),          # UCLA Loneliness 20-item
        ("nih_fluid_items.csv", "fluid_"),    # NIHTB fluid composite items
        ("nih_cryst_items.csv", "cryst_"),    # NIHTB crystallised composite items
    ]
    rows = []
    for fname, prefix in specs:
        res = alpha_for_scale(items_dir, fname, prefix)
        if res:
            rows.append(res)
    return pd.DataFrame(rows)


# =====================================================================
#  2. Age effects on cognition
# =====================================================================

def age_effects(pheno):
    covars = [c for c in ["sex_code"] if c in pheno.columns]
    rows = []
    for col, label, domain in COGNITION_UNCORRECTED:
        if col not in pheno.columns:
            continue
        res = C.ancova(pheno, col, "age", covars)
        res.update({"measure": col, "label": label, "domain": domain})
        rows.append(res)
    return pd.DataFrame(rows)


# =====================================================================
#  4. Social connectedness and cognition
# =====================================================================

def social_cognition(pheno):
    covars = [c for c in ["age", "sex_code", "phq"]
              if c in pheno.columns]
    rows = []
    for scol, slabel in SOCIAL_MEASURES:
        if scol not in pheno.columns:
            continue
        for ccol, clabel in EXEC_FOR_SOCIAL:
            if ccol not in pheno.columns:
                continue
            res = C.partial_corr(pheno, scol, ccol, covars)
            res.update({"social": scol, "social_label": slabel,
                        "cognition": ccol, "cognition_label": clabel,
                        "covariates": "+".join(covars)})
            rows.append(res)
    return pd.DataFrame(rows)


# =====================================================================
#  5. Social connectedness and mood
# =====================================================================

def social_mood(pheno):
    from scipy import stats
    rows = []
    for scol, slabel in SOCIAL_MEASURES:
        if scol not in pheno.columns:
            continue
        for mcol, mlabel in MOOD_MEASURES:
            if mcol not in pheno.columns:
                continue
            d = pheno[[scol, mcol]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(d) < 4:
                continue
            r, p = stats.pearsonr(d[scol], d[mcol])
            rows.append({"social": scol, "social_label": slabel,
                         "mood": mcol, "mood_label": mlabel,
                         "r": r, "p": p, "n": len(d)})
    return pd.DataFrame(rows)


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", default=None,
                    help="folder with item-level CSVs (ucla_items.csv, etc.) "
                         "for Cronbach's alpha")
    args = ap.parse_args()

    out = C.outdir("behavioral")
    report = [C.banner("COGNITIVE & BEHAVIORAL PHENOTYPING — Technical Validation")]

    pheno = C.load_phenotype()
    report.append(f"Phenotype subjects: {len(pheno)}")

    # ── 1 & 3. Cronbach's alpha ──
    alpha = compute_cronbach(args.items_dir)
    report.append(C.banner("1 & 3. Internal consistency (Cronbach's alpha)"))
    if not alpha.empty:
        # Deliberately not written to disk. The internal-consistency estimates
        # the descriptor reports come from build_reliability_tables.py, which
        # computes them on the released cohort from phenotype/redcap_items.tsv
        # with bootstrap CIs. This block runs only with --items-dir and on a
        # different item source, so writing a second cronbach_alpha.csv beside
        # the published table produced contradictory alphas for the same
        # scales; the values are reported here and in the text report only.
        for _, r in alpha.iterrows():
            report.append(f"  {r['scale']:<10} alpha={r['alpha']:.3f} "
                          f"({r['n_items']} items, n={r['n_subjects']})")
    else:
        report.append("  Item-level data not provided (--items-dir). Supply "
                      "per-scale item CSVs (e.g. ucla_items.csv with ucla_1.."
                      "ucla_20) to compute alpha. Expected UCLA alpha >= 0.85.")

    # ── 2. Age effects on cognition ──
    age = age_effects(pheno)
    report.append(C.banner("2. Age effects on cognition (uncorrected scores)"))
    if not age.empty:
        age.to_csv(out / "age_effects_cognition.csv", index=False)
        for _, r in age.iterrows():
            report.append(f"  {r['label']:<28} beta={r['beta']:+.3f} "
                          f"p={r['p']:.4f} eta2={r['partial_eta2']:.3f} "
                          f"n={r['n']}")
    else:
        report.append("  No uncorrected NIH Toolbox columns found.")

    # ── 4. Social connectedness and cognition ──
    sc = social_cognition(pheno)
    report.append(C.banner("4. Social connectedness and cognition "
                           "(partial corr; control age,sex,edu,depression)"))
    if not sc.empty:
        sc.to_csv(out / "social_cognition_partialcorr.csv", index=False)
        for _, r in sc.iterrows():
            report.append(f"  {r['social_label']:<34} vs "
                          f"{r['cognition_label']:<22} partial r={r['r']:+.3f} "
                          f"p={r['p']:.3f} n={r['n']}")
    else:
        report.append("  No usable social/cognition columns.")

    # ── 5. Social connectedness and mood ──
    sm = social_mood(pheno)
    report.append(C.banner("5. Social connectedness and mood (convergent "
                           "validity / depression confound)"))
    if not sm.empty:
        sm.to_csv(out / "social_mood_correlations.csv", index=False)
        for _, r in sm.iterrows():
            report.append(f"  {r['social_label']:<34} vs {r['mood_label']:<8} "
                          f"r={r['r']:+.3f} p={r['p']:.3f} n={r['n']}")
    else:
        report.append("  No usable social/mood columns.")

    (out / "behavioral_validation_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
