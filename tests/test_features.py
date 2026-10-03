"""Tests for lrr.features: leakage, population, event/duration, OOF priors.

Pure pandas/numpy. No lifelines, scikit-survival, xgboost, or shap.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lrr import config, features

T = pd.Timestamp("2018-01-01")
HORIZON_END = T + pd.DateOffset(months=config.HORIZON_MONTHS)  # 2020-01-01


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #


def test_assert_no_leakage_passes_on_pre_T() -> None:
    df = pd.DataFrame({"business_id": ["b0"], "date": ["2017-12-31"]})
    features.assert_no_leakage(df, T)  # must not raise


def test_assert_no_leakage_raises_on_T_or_after() -> None:
    df = pd.DataFrame({"business_id": ["b0", "b1"], "date": ["2017-12-31", "2018-01-01"]})
    with pytest.raises(AssertionError):
        features.assert_no_leakage(df, T)


def test_monthly_counts_excludes_on_or_after_T() -> None:
    dates = ["2017-12-15", "2018-01-05", "2017-11-20"]
    counts = features.monthly_counts(dates, T, months=24)
    # Only the two pre-T dates count; the 2018-01-05 is dropped.
    assert counts.sum() == 2


def test_build_features_uses_only_pre_T(monkeypatch) -> None:
    # One business with reviews straddling T. The post-T review must not inflate
    # the 3-month volume.
    reviews = pd.DataFrame(
        {
            "review_id": ["r0", "r1", "r2"],
            "business_id": ["b0", "b0", "b0"],
            "stars": [1, 2, 5],
            "date": ["2017-11-01", "2017-12-01", "2018-02-01"],  # last is post-T
            "text": ["slow and cold", "rude staff", "great now"],
        }
    )
    population = pd.DataFrame({"business_id": ["b0"], "cluster": ["Fast Food"], "chain": ["X"]})
    feats = features.build_features(
        reviews=reviews,
        tips=pd.DataFrame(),
        checkins=pd.DataFrame(),
        population=population,
        T=T,
    )
    row = feats.iloc[0]
    # 3-month window [2017-10-01, 2018-01-01): both Nov and Dec count, Feb excluded.
    assert row["review_vol_3m"] == 2
    assert row["review_vol_6m"] == 2
    assert row["review_vol_12m"] == 2


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #


def test_landmark_population_inclusion_rules() -> None:
    # b0: tenured (first review 2016) and recent activity (2017-10) -> included.
    # b1: too new (first review 2017-07, within 12m of T) -> excluded.
    # b2: tenured but no activity in last 6m (last review 2017-01) -> excluded.
    reviews = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "business_id": ["b0", "b0", "b1", "b2"],
            "stars": [4, 4, 4, 4],
            "date": ["2016-01-01", "2017-10-01", "2017-07-01", "2017-01-01"],
            "text": ["x", "x", "x", "x"],
        }
    )
    locations = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2"],
            "cluster": ["Fast Food"] * 3,
            "chain": ["X", "Y", "Z"],
            "is_open": [1, 1, 1],
        }
    )
    pop = features.landmark_population(reviews, pd.DataFrame(), pd.DataFrame(), locations, T)
    assert set(pop["business_id"]) == {"b0"}


# --------------------------------------------------------------------------- #
# Event and duration on a synthetic timeline
# --------------------------------------------------------------------------- #


def test_event_duration_four_cases() -> None:
    # Four businesses with distinct outcomes relative to T=2018-01-01.
    population = pd.DataFrame(
        {
            "business_id": ["closed_in", "closed_after", "open_recent", "censored_end"],
            "is_open": [0, 0, 1, 0],
            "cluster": ["Fast Food"] * 4,
            "chain": ["A", "B", "C", "D"],
        }
    )
    last_activity = pd.Series(
        {
            "closed_in": pd.Timestamp("2018-07-01"),  # within (T, T+24m] -> event
            "closed_after": pd.Timestamp("2021-01-01"),  # after window -> censored
            "open_recent": pd.Timestamp("2019-06-01"),  # open -> censored
            "censored_end": pd.Timestamp("2025-01-01"),  # beyond DATASET_END -> cap
        }
    )
    labeled = features.label_event_duration(population, last_activity, T).set_index("business_id")

    # closed_in: event, duration = days from T to 2018-07-01.
    assert labeled.loc["closed_in", "event"] == 1
    assert labeled.loc["closed_in", "duration_days"] == (pd.Timestamp("2018-07-01") - T).days

    # closed_after: last activity after the 24m window -> not an event (censored),
    # duration capped at the horizon end (T + 24m).
    assert labeled.loc["closed_after", "event"] == 0
    assert labeled.loc["closed_after", "duration_days"] == (HORIZON_END - T).days

    # open_recent: is_open == 1 -> never an event regardless of proxy.
    assert labeled.loc["open_recent", "event"] == 0

    # censored_end: proxy beyond DATASET_END and horizon -> duration capped at
    # min(horizon_end, DATASET_END) = horizon_end here (2020-01-01 < 2022-01-19).
    cap = min(HORIZON_END, pd.Timestamp(config.DATASET_END))
    assert labeled.loc["censored_end", "duration_days"] == (cap - T).days


def test_event_requires_closed_flag() -> None:
    population = pd.DataFrame(
        {"business_id": ["b0"], "is_open": [1], "cluster": ["Fast Food"], "chain": ["A"]}
    )
    last_activity = pd.Series({"b0": pd.Timestamp("2018-06-01")})
    labeled = features.label_event_duration(population, last_activity, T)
    # Even though proxy is in the window, is_open == 1 means no event.
    assert labeled.loc[0, "event"] == 0


# --------------------------------------------------------------------------- #
# OLS slope and velocity ratio
# --------------------------------------------------------------------------- #


def test_ols_slope_known_values() -> None:
    assert features.ols_slope([0, 1, 2, 3]) == pytest.approx(1.0)
    assert features.ols_slope([3, 2, 1, 0]) == pytest.approx(-1.0)
    assert features.ols_slope([5]) == 0.0
    assert features.ols_slope([2, 2, 2]) == pytest.approx(0.0)


def test_velocity_ratio_guards() -> None:
    # Prior window empty, last window has reviews -> returns the last count.
    dates = ["2017-10-01", "2017-11-01"]  # both in last-6m before 2018-01-01
    assert features.velocity_ratio(dates, T) == pytest.approx(2.0)
    # Both windows empty -> 1.0.
    assert features.velocity_ratio([], T) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Out-of-fold chain priors
# --------------------------------------------------------------------------- #


def test_out_of_fold_priors_never_see_own_fold() -> None:
    # Chain A: fold 0 has event=1, fold 1 has event=0. The prior for a fold-0 row
    # must equal the fold-1 mean for A (0.0), not include its own fold.
    df = pd.DataFrame(
        {
            "chain": ["A", "A", "A", "A"],
            "fold": [0, 0, 1, 1],
            "event": [1, 1, 0, 0],
        }
    )
    priors = features.out_of_fold_chain_priors(df)
    # Fold-0 rows: computed from fold-1 (events 0,0) -> 0.0
    assert priors.iloc[0] == pytest.approx(0.0)
    assert priors.iloc[1] == pytest.approx(0.0)
    # Fold-1 rows: computed from fold-0 (events 1,1) -> 1.0
    assert priors.iloc[2] == pytest.approx(1.0)
    assert priors.iloc[3] == pytest.approx(1.0)
