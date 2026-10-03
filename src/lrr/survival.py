"""Survival models, risk scoring, and explainability (stage 05).

Defines the four nested feature sets, the design-matrix builder (with a pooled
cluster indicator), Cox PH / Random Survival Forest / XGBoost fitters, and the
risk-score, tiering, driver, and survival-curve helpers. Heavy libraries
(lifelines, scikit-survival, xgboost, shap) are imported lazily so the pure logic
(feature sets, tiering, drivers, design matrix) is unit-testable without them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lrr import config

# --------------------------------------------------------------------------- #
# Feature sets (nested, for the ablation)
# --------------------------------------------------------------------------- #

_STARS = ["stars_all", "stars_12m", "stars_trend"]
_ENGAGEMENT = [
    "review_vol_12m",
    "review_vol_6m",
    "review_vol_3m",
    "review_velocity_slope",
    "review_velocity_ratio",
    "checkin_vol",
    "checkin_slope",
    "tip_vol",
    "tip_coverage",
    "review_length",
    "location_age_days",
    "chain_prior",
]
_TEXT = [
    "sentiment_mean",
    "sentiment_slope",
    "share_negative",
    "lexicon_complaint_rate",
]


def _topic_cols(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith("topic_")]


#: Nested feature sets. ``full`` = engagement + text + topic shares + stars.
FEATURE_SETS: dict[str, list[str]] = {
    "stars_only": list(_STARS),
    "engagement": list(_ENGAGEMENT),
    "engagement_text": list(_ENGAGEMENT) + list(_TEXT),
    "full": list(_STARS) + list(_ENGAGEMENT) + list(_TEXT),
}


def feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    """Resolve a feature-set name to the columns present in ``features``.

    Topic-share columns are appended to the ``full`` set when present.
    """
    cols = list(FEATURE_SETS[feature_set])
    if feature_set == "full":
        cols = cols + _topic_cols(features)
    return [c for c in cols if c in features.columns]


def design_matrix(features: pd.DataFrame, feature_set: str, pooled: bool = False) -> pd.DataFrame:
    """Select the feature columns, add a pooled cluster indicator when requested.

    Missing numeric values are median-imputed per column so the matrix is dense.
    """
    cols = feature_columns(features, feature_set)
    X = features[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    if pooled and "cluster" in features.columns:
        X["cluster_ff"] = (features["cluster"] == "Fast Food").astype(int)
    return X


# --------------------------------------------------------------------------- #
# Model fitters (lazy imports)
# --------------------------------------------------------------------------- #


def fit_cox(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    penalizer: float = 0.1,
):
    """Fit a lifelines CoxPHFitter. Returns the fitted model."""
    from lifelines import CoxPHFitter

    df = X.copy()
    df["duration"] = np.asarray(duration, dtype=float)
    df["event"] = np.asarray(event, dtype=int)
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(df, duration_col="duration", event_col="event")
    return model


def tune_cox_penalizer(X, duration, event, penalizers: Sequence[float] = config.COX_PENALIZERS):
    """Pick the penalizer with the best in-sample partial log-likelihood/C.

    Returns (best_model, best_penalizer). Simple selection by concordance; the
    stage-05 driver cross-validates across folds.
    """
    from lifelines import CoxPHFitter  # noqa: F401 (ensures lifelines present)

    best = None
    best_pen = penalizers[0]
    best_c = -np.inf
    for pen in penalizers:
        model = fit_cox(X, duration, event, penalizer=pen)
        c = float(model.concordance_index_)
        if c > best_c:
            best, best_pen, best_c = model, pen, c
    return best, best_pen


def fit_rsf(X: pd.DataFrame, duration, event, seed: int = config.SEED):
    """Fit a scikit-survival RandomSurvivalForest."""
    from sksurv.ensemble import RandomSurvivalForest

    y = np.array(
        list(zip(np.asarray(event, dtype=bool), np.asarray(duration, dtype=float))),
        dtype=[("event", "bool"), ("time", "float64")],
    )
    model = RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1
    )
    model.fit(X.to_numpy(), y)
    return model


def fit_xgb(X: pd.DataFrame, event, seed: int = config.SEED):
    """Fit an XGBoost classifier on the 24-month event (for comparison)."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X.to_numpy(), np.asarray(event, dtype=int))
    return model


# --------------------------------------------------------------------------- #
# Risk scoring, tiering, drivers
# --------------------------------------------------------------------------- #


def risk_tier(percentile: float, cuts: dict = config.RISK_TIER_CUTS) -> str:
    """Map a 0-100 risk percentile to a tier label.

    High >= 90, Elevated >= 75, Watch >= 50, else Low.
    """
    if percentile >= cuts["High"]:
        return "High"
    if percentile >= cuts["Elevated"]:
        return "Elevated"
    if percentile >= cuts["Watch"]:
        return "Watch"
    return "Low"


def to_percentile(risk_scores: np.ndarray) -> np.ndarray:
    """Convert risk scores to 0-100 percentiles (higher score = higher percentile)."""
    s = pd.Series(np.asarray(risk_scores, dtype=float))
    return (s.rank(pct=True) * 100.0).to_numpy()


def top_drivers(contributions: np.ndarray, feature_names: Sequence[str], k: int = 3) -> list[str]:
    """Return the ``k`` feature names with the largest absolute contribution."""
    contributions = np.asarray(contributions, dtype=float)
    order = np.argsort(-np.abs(contributions))[:k]
    return [str(feature_names[i]) for i in order]


def build_risk_scores(
    features: pd.DataFrame,
    risk_scores: np.ndarray,
    surv_12m: np.ndarray,
    surv_24m: np.ndarray,
    drivers: list[list[str]],
) -> pd.DataFrame:
    """Assemble the risk_scores artifact rows.

    Columns: business_id, chain, cluster, risk_score, risk_percentile, risk_tier,
    top_3_drivers, surv_12m, surv_24m.
    """
    pct = to_percentile(risk_scores)
    return pd.DataFrame(
        {
            "business_id": features["business_id"].to_numpy(),
            "chain": features.get("chain"),
            "cluster": features.get("cluster"),
            "risk_score": np.asarray(risk_scores, dtype=float),
            "risk_percentile": pct,
            "risk_tier": [risk_tier(p) for p in pct],
            "top_3_drivers": [", ".join(d) for d in drivers],
            "surv_12m": np.asarray(surv_12m, dtype=float),
            "surv_24m": np.asarray(surv_24m, dtype=float),
        }
    )


def survival_curves(
    cox_model,
    X: pd.DataFrame,
    business_ids: Sequence[str],
    times_days: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per-location survival curves (long form) for the app.

    Returns columns business_id, t_days, survival. Uses the Cox model's predicted
    survival function sampled at ``times_days`` (default monthly to 24 months).
    """
    if times_days is None:
        times_days = [int(30.4 * m) for m in range(0, config.HORIZON_MONTHS + 1)]
    sf = cox_model.predict_survival_function(X, times=times_days)
    rows = []
    for col, bid in zip(sf.columns, business_ids):
        for t in times_days:
            rows.append({"business_id": bid, "t_days": int(t), "survival": float(sf.loc[t, col])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Full-corpus two-tier additions (Tier 1 fit, out-of-fold stacking, final score)
# --------------------------------------------------------------------------- #

#: Tier 1 feature columns (engagement + stars + sentiment aggregates; no topics).
TIER1_FEATURE_COLUMNS: tuple[str, ...] = (
    "review_vol_12m",
    "review_vol_6m",
    "review_vol_3m",
    "tip_vol_12m",
    "tip_vol_6m",
    "tip_vol_3m",
    "checkin_vol_12m",
    "checkin_vol_6m",
    "checkin_vol_3m",
    "review_velocity_slope",
    "stars_all",
    "stars_12m",
    "stars_trend",
    "tip_vol",
    "checkin_vol",
    "sentiment_mean",
    "sentiment_slope",
)

#: Re-exported so callers write one documented rule.
FINAL_SCORE_RULE: str = config.FINAL_SCORE_RULE


def tier1_design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Dense Tier 1 design matrix (median-imputed) over the Tier 1 columns present."""
    cols = [c for c in TIER1_FEATURE_COLUMNS if c in features.columns]
    X = features[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X


def fit_tier1(
    features: pd.DataFrame,
    duration,
    event,
    penalizer: float = 0.1,
):
    """Fit the Tier 1 corpus Cox model. Returns the fitted lifelines model."""
    X = tier1_design_matrix(features)
    return fit_cox(X, duration, event, penalizer=penalizer)


def oof_tier1_scores(
    features: pd.DataFrame,
    duration,
    event,
    group: pd.Series,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> np.ndarray:
    """Out-of-fold Tier 1 risk scores (the leakage-safe stacking feature).

    Rows sharing a ``group`` always fall in the same held-out fold, and each fold's
    Tier 1 model is trained on the complement. A row's score therefore never comes
    from a model that saw its group. Returns an array aligned to ``features``.
    """
    from sklearn.model_selection import GroupKFold

    features = features.reset_index(drop=True)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    group = pd.Series(group).reset_index(drop=True).astype(str)

    scores = np.full(len(features), np.nan, dtype=float)
    k = min(n_folds, group.nunique())
    if k < 2:
        model = fit_tier1(features, duration, event)
        return model.predict_partial_hazard(tier1_design_matrix(features)).to_numpy().ravel()

    gkf = GroupKFold(n_splits=k)
    for train_idx, test_idx in gkf.split(features, groups=group):
        model = fit_tier1(features.iloc[train_idx], duration[train_idx], event[train_idx])
        X_test = tier1_design_matrix(features.iloc[test_idx])
        scores[test_idx] = model.predict_partial_hazard(X_test).to_numpy().ravel()
    return scores


def assert_oof_no_group_leak(group: pd.Series, fold_of_row: pd.Series) -> None:
    """Assert each group maps to exactly one fold (the stacking invariant).

    If any group appears in more than one fold, a row could have received a Tier 1
    score from a model trained on its own group. Raises on violation.
    """
    g = pd.DataFrame(
        {
            "group": pd.Series(group).astype(str).to_numpy(),
            "fold": pd.Series(fold_of_row).to_numpy(),
        }
    )
    bad = g.groupby("group")["fold"].nunique()
    offenders = bad[bad > 1].index.tolist()
    assert not offenders, f"Group(s) split across folds (leakage): {offenders}"


def final_score(
    risk_df: pd.DataFrame,
    tier1_col: str = "tier1_score",
    tier2_col: str = "tier2_score",
    cohort_col: str = "is_cohort",
) -> np.ndarray:
    """Apply the documented final-score rule.

    Cohort locations use ``tier2_score`` (the cluster-specific model that already
    stacks the out-of-fold Tier 1 signal); non-cohort restaurants use
    ``tier1_score``. See :data:`FINAL_SCORE_RULE`.
    """
    t1 = risk_df[tier1_col].to_numpy(dtype=float)
    if tier2_col in risk_df.columns and cohort_col in risk_df.columns:
        t2 = risk_df[tier2_col].to_numpy(dtype=float)
        is_cohort = risk_df[cohort_col].to_numpy(dtype=bool)
        return np.where(is_cohort & ~np.isnan(t2), t2, t1)
    return t1
