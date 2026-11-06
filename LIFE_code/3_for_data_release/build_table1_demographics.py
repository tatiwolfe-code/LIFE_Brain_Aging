#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Compute the exact values of manuscript Table 1 (LIFE sample demographics +
health/SES) and write them as CSV (universal format), no .docx. Reads only the
released BIDS data: participants.tsv (demographics + health/SES) and the BIDS
phenotype (phenotype/redcap_questionnaires.tsv, dcf_education) for education —
verified identical to the source REDCap export for all 64 participants.

Output: bids_dataset_LIFE/derivatives/technical_validation/tables/Table1_demographics.csv
"""
import csv
import os
import statistics as st
from collections import Counter
from pathlib import Path

BASE = Path(os.environ.get("LIFE_BASE", Path(__file__).resolve().parents[2]))
PART = BASE / "bids_dataset_LIFE" / "participants.tsv"
PHENO = BASE / "bids_dataset_LIFE" / "phenotype" / "redcap_questionnaires.tsv"
OUT = BASE / "bids_dataset_LIFE" / "derivatives" / "technical_validation" / "tables" / "Table1_demographics.csv"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- load data ----
rows = list(csv.DictReader(open(PART), delimiter="\t"))
N = len(rows)
col = lambda c: [r[c] for r in rows]


def msd(vals):
    v = [x for x in vals if x is not None]
    return f"{st.mean(v):.1f} ({st.stdev(v):.1f})" if len(v) > 1 else "n/a"


def pct(n):
    return f"{n} ({100 * n / N:.1f})"


# education from the released BIDS phenotype (redcap_questionnaires.tsv,
# dcf_education); verified identical to the source REDCap export for all 64.
edu_map = {r["participant_id"]: num(r.get("dcf_education"))
           for r in csv.DictReader(open(PHENO), delimiter="\t")}
edu = [edu_map.get(r["participant_id"]) for r in rows]

ages = [num(x) for x in col("age")]
sex = Counter(col("sex"))
hand = Counter(col("handedness"))
race = Counter(col("race"))
eth = Counter((x or "Not reported") for x in col("ethnicity"))
inc = Counter(col("income_bracket"))
drinkers = [num(r["alcohol_drinks_per_wk"]) for r in rows
            if r["alcohol_use"] == "Yes"]
# children with >=2 (and M, SD within that group, rounded to whole numbers)
ge2 = [num(x) for x in col("num_children")
       if num(x) is not None and num(x) >= 2]
children_ge2 = (f"{len(ge2)} ({100 * len(ge2) / N:.1f}%) "
                f"[{int(round(st.mean(ge2)))}, {int(round(st.stdev(ge2)))}]")

def yes(c):
    return sum(1 for x in col(c) if x == "Yes")

TABLE = [
    ("Characteristic", "Full sample (N = 64)", True),
    # --- demographics (13) ---
    ("Age (years), M (SD)", msd(ages), False),
    ("Age range (years)", f"{int(min(x for x in ages if x))}–{int(max(x for x in ages if x))}", False),
    ("Sex — Female, n (%)", pct(sex['F']), False),
    ("Sex — Male, n (%)", pct(sex['M']), False),
    ("Handedness — Right, n (%)", pct(hand['R']), False),
    ("Race — White, n (%)", pct(race['White']), False),
    ("Race — Black/African American, n (%)", pct(race['Black/African American']), False),
    ("Race — American Indian/Alaska Native, n (%)", pct(race['American Indian/Alaska Native']), False),
    ("Race — Not reported, n (%)", pct(race['Unknown/Not reported']), False),
    ("Ethnicity — Not Hispanic/Latino, n (%)", pct(eth['Not Hispanic/Latino']), False),
    ("Ethnicity — Hispanic/Latino, n (%)", pct(eth['Hispanic/Latino']), False),
    ("Ethnicity — Not reported, n (%)", pct(eth['Not reported']), False),
    ("Education (years), M (SD)", msd(edu), False),
    # --- SES (income merged into 3 brackets) ---
    ("Annual income — <$25,000 or not reported, n (%)",
     pct(inc['<$25,000'] + inc['n/a']), False),
    ("Annual income — $25,000–$74,999, n (%)",
     pct(inc['$25,000-$49,999'] + inc['$50,000-$74,999']), False),
    ("Annual income — ≥$75,000, n (%)",
     pct(inc['$75,000-$99,999'] + inc['>=$100,000']), False),
    ("Has children, n (%)", pct(yes('has_children')), False),
    ("Number of children ≥2, n (%) [M, SD]", children_ge2, False),
    ("Religion/spirituality (scale 1–5), M (SD)",
     msd([num(x) for x in col('religion_spirituality')]), False),
    # --- health ---
    ("Medical history — cancer, n (%)", pct(yes('hx_cancer')), False),
    ("Medical history — hypertension, n (%)", pct(yes('hypertension')), False),
    ("Medical history — diabetes, n (%)", pct(yes('diabetes')), False),
    ("Medical history — heart disease, n (%)", pct(yes('hx_heart_disease')), False),
    ("Medical history — asthma, n (%)", pct(yes('asthma')), False),
    ("Medical history — mental health treatment (history), n (%)",
     pct(yes('hx_psych_treatment')), False),
    ("Current smoker, n (%)", pct(yes('current_smoker')), False),
    ("Current alcohol use, n (%)", pct(yes('alcohol_use')), False),
    ("Alcohol — drinks/week (among users), M (SD)",
     f"{round(st.mean(drinkers))} ({round(st.stdev(drinkers))})", False),
    ("Medical cannabis, n (%)", pct(yes('medical_cannabis')), False),
]

NOTE = ("Note. N = 64 community-dwelling adults. M (SD) reported for continuous "
        "variables; n (%) for categorical variables. Health and socioeconomic "
        "items are pre-screening self-report. Annual income is annualized. "
        "Religion/spirituality is a 1–5 self-rated importance scale (higher = "
        "greater importance). Drinks/week is estimated as reported frequency "
        "× drinks per occasion, among current drinkers. Race "
        "“Not reported” denotes participants who declined to report.")

# ---- write CSV ----
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([TABLE[0][0], TABLE[0][1]])          # header row
    for a, b, _is_header in TABLE[1:]:
        w.writerow([a, b])
OUT.with_name("Table1_demographics_note.txt").write_text(
    "Table 1. Sample demographics, socioeconomic, and health characteristics.\n\n"
    + NOTE + "\n")
print("Wrote", OUT, "with", len(TABLE) - 1, "data rows")
