#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Item-level loaders and reliability helpers for the Cognitive & Behavioral
Phenotyping Technical Validation.

The *released* phenotype (bids_dataset_LIFE/phenotype/) carries scored
measures only. Item-level responses live in the study source exports:

  LIFE_raw/RedCap/LIFEBRAIN_DATA_*.csv   REDCap items (UCLA-20, MSPSS-12, ...)
  LIFE_raw/RedCap/LIFE SNI.csv           SIN 9-item inventory (pre-screening)

Both are keyed by "LIFE 0NN" / "LIFE_0NN" and map identically onto sub-0NN.
Scoring conventions implemented here were verified to reproduce the released
totals exactly (ucla_total, mpss_*_subscore, sni_total).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

RAW_REDCAP_DIR = C.BASE_DIR / "LIFE_raw" / "RedCap"

# UCLA Loneliness Scale v3: positively worded items are reverse-scored (5 - x).
UCLA_REVERSE = [1, 5, 6, 9, 10, 15, 16, 19, 20]

# MSPSS-12 subscales (Zimet et al. 1988).
MSPSS_DOMAINS = {
    "Significant other": [1, 2, 5, 10],
    "Family":            [3, 4, 8, 11],
    "Friends":           [6, 7, 9, 12],
}

# SIN inventory domains, in item order (see Supplementary Table S1).
SIN_DOMAINS = [
    "Spouse / partner", "Children", "Parents", "Parents-in-law",
    "Close relatives", "Close friends", "Neighbors",
    "Work activity", "Organized groups",
]


def _to_sub(series: pd.Series) -> pd.Series:
    """'LIFE 007' / 'LIFE_007' -> 'sub-007'."""
    num = series.astype(str).str.extract(r"(\d+)")[0]
    return num.map(lambda s: f"sub-{int(s):03d}" if pd.notna(s) else None)


# The released cohort for this descriptor is the older adults; every statistic
# in the Cognitive & Behavioral Phenotyping section is computed on them.
AGE_MIN = 60


def _released() -> set[str]:
    return set(older_ids())


def older_ids(age_min: int = AGE_MIN) -> list[str]:
    """Participant IDs of the released older cohort (age >= age_min)."""
    part = C.load_participants()
    age = pd.to_numeric(part["age"], errors="coerce")
    return sorted(part.index[age >= age_min].tolist())


def older(df: pd.DataFrame, age_min: int = AGE_MIN) -> pd.DataFrame:
    """Restrict a participant-indexed frame to the older cohort."""
    keep = set(older_ids(age_min))
    return df.loc[[i for i in df.index if i in keep]]


def load_redcap_items() -> pd.DataFrame:
    """Item-level questionnaire responses, indexed by sub-0NN.

    Read from the released phenotype table, phenotype/redcap_items.tsv, so
    every reliability statistic in the descriptor is reproducible from the
    published dataset alone. The raw REDCap export is used only as a fallback
    on the authoring machine; it is not part of the release, and it carries
    participants who are not in it, so statistics computed from it do not
    match the published cohort.
    """
    released = C.PHENO_DIR / "redcap_items.tsv"
    if released.exists():
        df = pd.read_csv(released, sep="\t")
        df = df.set_index("participant_id")
        return df.loc[[i for i in df.index if i in _released()]].copy()
    hits = sorted(RAW_REDCAP_DIR.glob("LIFEBRAIN_DATA_*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"{released} not found, and no LIFEBRAIN_DATA_*.csv in "
            f"{RAW_REDCAP_DIR} to fall back on")
    df = pd.read_csv(hits[-1], low_memory=False)
    df.index = _to_sub(df["life_part_id"])
    return df.loc[[i for i in df.index if i in _released()]].copy()


def ucla_items(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """UCLA-20 items, reverse-scored (reproduces released ucla_total exactly)."""
    df = load_redcap_items() if df is None else df
    it = df[[f"ucla_{i}" for i in range(1, 21)]].apply(pd.to_numeric, errors="coerce")
    for i in UCLA_REVERSE:
        it[f"ucla_{i}"] = 5 - it[f"ucla_{i}"]
    return it


def mspss_items(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_redcap_items() if df is None else df
    return df[[f"mpss_{i}" for i in range(1, 13)]].apply(pd.to_numeric, errors="coerce")


def sin_items() -> pd.DataFrame:
    """SIN 9-item scored inventory (reproduces released sni_total exactly).

    Taken from the released item table, where the domains are sni_1..sni_9 in
    the order of SIN_DOMAINS; the separate raw SNI export is the fallback.
    """
    released = C.PHENO_DIR / "redcap_items.tsv"
    if released.exists():
        d = pd.read_csv(released, sep="\t").set_index("participant_id")
        cols = [f"sni_{i}" for i in range(1, 10)]
        if all(c in d.columns for c in cols):
            it = d[cols].apply(pd.to_numeric, errors="coerce")
            it.columns = SIN_DOMAINS
            return it.loc[[i for i in it.index if i in _released()]]
    f = RAW_REDCAP_DIR / "LIFE SNI.csv"
    s = pd.read_csv(f)
    s.index = _to_sub(s["life_id"])
    it = s[[f"life_sin_{i}" for i in range(1, 10)]].apply(pd.to_numeric, errors="coerce")
    it.columns = SIN_DOMAINS
    return it.loc[[i for i in it.index if i in _released()]]


# =====================================================================
#  Reliability helpers
# =====================================================================

def alpha_ci(alpha: float, n: int, k: int, conf: float = 0.95) -> tuple[float, float]:
    """Feldt (1965) exact confidence interval for Cronbach's alpha."""
    if not np.isfinite(alpha) or n < 3 or k < 2:
        return (np.nan, np.nan)
    a = (1 - conf) / 2
    df1, df2 = n - 1, (n - 1) * (k - 1)
    lo = 1 - (1 - alpha) * stats.f.ppf(1 - a, df1, df2)
    hi = 1 - (1 - alpha) * stats.f.ppf(a, df1, df2)
    return (float(lo), float(hi))


N_BOOT = 10000
BOOT_SEED = 0


def boot_alpha_ci(items: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                  conf: float = 0.95) -> tuple[float, float]:
    """
    Percentile bootstrap CI for Cronbach's alpha, resampling participants.

    Preferred over Feldt (1965) here: Feldt assumes normally distributed items
    with compound symmetry, and several of these scales are markedly skewed
    (e.g. MSPSS skew -1.8), where it understates uncertainty. Each call seeds
    its own generator so results do not depend on call order.
    """
    X = items.apply(pd.to_numeric, errors="coerce").dropna().values.astype(float)
    n = len(X)
    if n < 3 or X.shape[1] < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([C.cronbach_alpha(X[rng.integers(0, n, n)]) for _ in range(n_boot)])
    a = (1 - conf) / 2
    lo, hi = np.nanpercentile(bs, [100 * a, 100 * (1 - a)])
    return (float(lo), float(hi))


def boot_empirical_ci(tscore: pd.Series, se: pd.Series, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED, conf: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for IRT empirical reliability (same resampling)."""
    d = pd.concat([pd.to_numeric(tscore, errors="coerce"),
                   pd.to_numeric(se, errors="coerce")], axis=1).dropna()
    if len(d) < 3:
        return (np.nan, np.nan)
    t, s = d.iloc[:, 0].values, d.iloc[:, 1].values
    n = len(t)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v = t[i].var(ddof=1)
        bs[b] = 1 - (s[i] ** 2).mean() / v if v > 0 else np.nan
    a = (1 - conf) / 2
    lo, hi = np.nanpercentile(bs, [100 * a, 100 * (1 - a)])
    return (float(lo), float(hi))


def reliability(items: pd.DataFrame) -> dict:
    """
    Full internal-consistency profile for an (n_subjects x k_items) frame:
    alpha + Feldt 95% CI, mean inter-item r, per-item corrected item-total
    (item-rest) correlations, and alpha-if-item-deleted.
    """
    X = items.apply(pd.to_numeric, errors="coerce").dropna().values.astype(float)
    n, k = X.shape
    a = C.cronbach_alpha(X)
    lo, hi = alpha_ci(a, n, k)
    R = np.corrcoef(X, rowvar=False)
    total = X.sum(axis=1)
    item_rest = np.array([stats.pearsonr(X[:, j], total - X[:, j])[0] for j in range(k)])
    a_del = np.array([C.cronbach_alpha(np.delete(X, j, axis=1)) for j in range(k)])
    return {
        "alpha": a, "ci_lo": lo, "ci_hi": hi, "n": n, "k": k,
        "mean_inter_item_r": float(R[np.triu_indices(k, 1)].mean()),
        "item_rest": item_rest, "alpha_if_deleted": a_del,
        "item_mean": X.mean(axis=0), "item_sd": X.std(axis=0, ddof=1),
        "labels": list(items.columns),
    }


def empirical_reliability(tscore: pd.Series, se: pd.Series) -> dict:
    """
    IRT/CAT empirical (marginal) reliability for a PROMIS-style scored measure:
        rho = 1 - mean(SE^2) / Var(T)
    This is the item-response-theory analogue of Cronbach's alpha and is the
    only internal-consistency estimate recoverable when a release ships scores
    plus their standard errors rather than item responses.
    """
    d = pd.concat([pd.to_numeric(tscore, errors="coerce"),
                   pd.to_numeric(se, errors="coerce")], axis=1).dropna()
    if len(d) < 3:
        return {"rho": np.nan, "n": len(d), "var_t": np.nan, "mean_se": np.nan}
    t, s = d.iloc[:, 0], d.iloc[:, 1]
    var_t = float(t.var(ddof=1))
    return {"rho": float(1 - (s ** 2).mean() / var_t), "n": int(len(d)),
            "var_t": var_t, "mean_se": float(s.mean())}


def spearman_brown_alpha(r: float, k: int = 2) -> float:
    """Standardized alpha for a k-component composite with mean inter-item r."""
    return float(k * r / (1 + (k - 1) * r))
