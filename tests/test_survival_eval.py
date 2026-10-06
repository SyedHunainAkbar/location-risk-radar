"""Tests for lrr.evaluation and lrr.survival pure logic.

No lifelines, scikit-survival, xgboost, or shap. These cover Harrell's C,
calibration, bootstrap, the ablation table, feature sets, the design matrix, risk
tiering, percentiles, and top drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lrr import evaluation
from lrr import survival as S

# --------------------------------------------------------------------------- #
# Harrell's C
# --------------------------------------------------------------------------- #


def test_harrell_c_perfect_concordance() -> None:
    # Shorter survivor has the event and the higher risk -> C = 1.0.
    duration = [1, 2, 3, 4]
    event = [1, 1, 1, 0]
    risk = [0.9, 0.8, 0.7, 0.1]
    assert evaluation.harrell_c(duration, event, risk) == pytest.approx(1.0)


def test_harrell_c_perfect_discordance() -> None:
    duration = [1, 2, 3, 4]
    event = [1, 1, 1, 0]
    risk = [0.1, 0.2, 0.3, 0.9]  # risk inverted vs survival
    assert evaluation.harrell_c(duration, event, risk) == pytest.approx(0.0)


def test_harrell_c_ties_half() -> None:
    # One comparable pair, equal risk -> 0.5.
    duration = [1, 2]
    event = [1, 0]
    risk = [0.5, 0.5]
    assert evaluation.harrell_c(duration, event, risk) == pytest.approx(0.5)


def test_harrell_c_no_comparable_pairs_is_nan() -> None:
    # No events -> nothing comparable.
    assert np.isnan(evaluation.harrell_c([1, 2, 3], [0, 0, 0], [0.1, 0.2, 0.3]))


# --------------------------------------------------------------------------- #
# Calibration, bootstrap, ablation
# --------------------------------------------------------------------------- #


def test_harrell_c_ci_brackets_point_estimate() -> None:
    # Strongly concordant: shorter survival -> higher risk. C near 1, CI brackets it.
    rng = np.random.default_rng(0)
    duration = np.arange(1, 61, dtype=float)
    event = np.ones(60, dtype=int)
    risk = -duration + rng.normal(0, 0.01, size=60)
    point = evaluation.harrell_c(duration, event, risk)
    lo, hi = evaluation.harrell_c_ci(duration, event, risk)
    assert 0.0 <= lo <= point <= hi <= 1.0
    assert point > 0.95


def test_calibration_points_shape() -> None:
    pred = np.linspace(0, 1, 100)
    obs = (pred > 0.5).astype(float)
    cal = evaluation.calibration_points(pred, obs, bins=5)
    assert list(cal.columns) == ["bin", "mean_predicted", "mean_observed", "n"]
    assert len(cal) == 5
    assert cal["n"].sum() == 100


def test_bootstrap_metric_brackets_point() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float)
    yhat = y.copy()
    from sklearn.metrics import accuracy_score

    lo, hi = evaluation.bootstrap_metric(accuracy_score, [y, yhat])
    assert lo <= 1.0 <= hi + 1e-9


def test_ablation_table_delta_vs_stars() -> None:
    metrics = {
        "stars_only": {"harrell_c": 0.55},
        "engagement": {"harrell_c": 0.65},
        "engagement_text": {"harrell_c": 0.70},
        "full": {"harrell_c": 0.72},
    }
    tbl = evaluation.ablation_table(metrics).set_index("feature_set")
    assert tbl.loc["stars_only", "delta_c_vs_stars"] == pytest.approx(0.0)
    assert tbl.loc["full", "delta_c_vs_stars"] == pytest.approx(0.17)
    # Text adds value over stars.
    assert tbl.loc["engagement_text", "harrell_c"] > tbl.loc["stars_only", "harrell_c"]


# --------------------------------------------------------------------------- #
# Feature sets and design matrix
# --------------------------------------------------------------------------- #


def test_feature_sets_are_nested() -> None:
    stars = set(S.FEATURE_SETS["stars_only"])
    full = set(S.FEATURE_SETS["full"])
    assert stars.issubset(full)
    assert set(S.FEATURE_SETS["engagement"]).issubset(set(S.FEATURE_SETS["engagement_text"]))


def _toy_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b0", "b1"],
            "chain": ["A", "B"],
            "cluster": ["Fast Food", "Non-Fast Food"],
            "stars_all": [3.0, 4.0],
            "stars_12m": [3.0, np.nan],  # exercise median impute
            "stars_trend": [0.1, -0.1],
            "topic_0": [0.5, 0.2],
        }
    )


def test_design_matrix_stars_only() -> None:
    X = S.design_matrix(_toy_features(), "stars_only", pooled=False)
    assert set(X.columns) == {"stars_all", "stars_12m", "stars_trend"}
    assert not X.isna().any().any()  # imputed


def test_design_matrix_pooled_adds_cluster_indicator() -> None:
    X = S.design_matrix(_toy_features(), "stars_only", pooled=True)
    assert "cluster_ff" in X.columns
    assert X["cluster_ff"].tolist() == [1, 0]


def test_design_matrix_full_includes_topic_cols() -> None:
    X = S.design_matrix(_toy_features(), "full", pooled=False)
    assert "topic_0" in X.columns


def _cleanable_features() -> pd.DataFrame:
    """A larger toy frame with a NaN, a constant column, and a collinear pair."""
    n = 40
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "business_id": [f"b{i}" for i in range(n)],
            "chain": ["A", "B"] * (n // 2),
            "cluster": ["Fast Food", "Non-Fast Food"] * (n // 2),
            "stars_all": base,
            "stars_12m": base + rng.normal(scale=0.01, size=n),  # ~collinear with stars_all
            "stars_trend": rng.normal(size=n),
            "topic_0": rng.normal(size=n),
            "topic_1": np.zeros(n),  # zero variance
        }
    )
    df.loc[0, "stars_trend"] = np.nan  # exercise impute + indicator
    return df


def test_clean_design_matrix_has_no_nan_inf_or_zero_variance() -> None:
    """The cleaned Cox design matrix must be finite with no constant columns."""
    X = S.design_matrix(_cleanable_features(), "full", pooled=True, clean=True)
    arr = X.to_numpy(dtype=float)
    assert not np.isnan(arr).any(), "cleaned design matrix contains NaN"
    assert not np.isinf(arr).any(), "cleaned design matrix contains inf"
    stds = X.std(ddof=0)
    assert (stds > 0).all(), f"zero-variance columns remain: {stds[stds == 0].index.tolist()}"
    # The constant topic_1 must have been dropped, and a missingness indicator added.
    assert "topic_1" not in X.columns
    assert "stars_trend_missing" in X.columns


# --------------------------------------------------------------------------- #
# Risk tiering, percentile, drivers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pct, tier",
    [
        (95, "High"),
        (90, "High"),
        (80, "Elevated"),
        (75, "Elevated"),
        (60, "Watch"),
        (50, "Watch"),
        (10, "Low"),
        (0, "Low"),
    ],
)
def test_risk_tier_cut_points(pct: float, tier: str) -> None:
    assert S.risk_tier(pct) == tier


def test_to_percentile_monotone() -> None:
    pct = S.to_percentile([1.0, 2.0, 3.0, 4.0])
    assert pct[0] < pct[-1]
    assert (pct >= 0).all() and (pct <= 100).all()


def test_top_drivers_picks_largest_abs() -> None:
    contribs = np.array([0.1, -0.9, 0.3])
    names = ["a", "b", "c"]
    drivers = S.top_drivers(contribs, names, k=2)
    assert drivers[0] == "b"  # largest magnitude
    assert set(drivers) == {"b", "c"}


def test_build_risk_scores_schema() -> None:
    feats = _toy_features()
    scores = S.build_risk_scores(
        feats,
        risk_scores=np.array([2.0, 1.0]),
        surv_12m=np.array([0.8, 0.9]),
        surv_24m=np.array([0.6, 0.85]),
        drivers=[["stars_all"], ["topic_0"]],
    )
    assert list(scores.columns) == [
        "business_id",
        "chain",
        "cluster",
        "risk_score",
        "risk_percentile",
        "risk_tier",
        "top_3_drivers",
        "surv_12m",
        "surv_24m",
    ]
    assert scores["risk_tier"].isin(["High", "Elevated", "Watch", "Low"]).all()
