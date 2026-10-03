"""Tests for the full-corpus two-tier upgrade.

Cover the streaming monthly panel vs a brute-force groupby, the restaurant universe
flagging, the leakage-safe out-of-fold stacking invariant, the final-score rule, and
the streaming sentiment reducer. Pure pandas/numpy and scikit-learn only; no
lifelines/scikit-survival/xgboost needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lrr import config, corpus
from lrr import features as F
from lrr import survival as S

# --------------------------------------------------------------------------- #
# Streaming monthly panel vs brute force
# --------------------------------------------------------------------------- #


def _toy_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b0", "b1", "b2"],
            "stars": [5, 3, 1, 4, 2],
            "date": ["2019-01-05", "2019-01-20", "2019-02-10", "2019-01-15", "2020-07-01"],
            "text": ["x", "y", "z", "w", "v"],  # text must never be aggregated
        }
    )


def _toy_tips() -> pd.DataFrame:
    return pd.DataFrame(
        {"business_id": ["b0", "b1", "b1"], "date": ["2019-01-09", "2019-01-22", "2019-02-02"]}
    )


def _toy_checkins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b3"],
            "checkin_at": pd.to_datetime(["2019-01-11", "2019-02-15", "2019-01-01"]),
        }
    )


def _brute_force_panel(reviews, tips, checkins, rid):
    r = reviews[reviews.business_id.isin(rid)].copy()
    r["month"] = pd.to_datetime(r.date).dt.to_period("M").astype(str)
    gr = (
        r.groupby(["business_id", "month"])
        .agg(n_reviews=("stars", "size"), mean_stars=("stars", "mean"))
        .reset_index()
    )
    t = tips[tips.business_id.isin(rid)].copy()
    t["month"] = pd.to_datetime(t.date).dt.to_period("M").astype(str)
    gt = t.groupby(["business_id", "month"]).size().rename("n_tips").reset_index()
    c = checkins[checkins.business_id.isin(rid)].copy()
    c["month"] = pd.to_datetime(c.checkin_at).dt.to_period("M").astype(str)
    gc = c.groupby(["business_id", "month"]).size().rename("n_checkins").reset_index()
    panel = gr.merge(gt, on=["business_id", "month"], how="outer").merge(
        gc, on=["business_id", "month"], how="outer"
    )
    panel[["n_reviews", "n_tips", "n_checkins"]] = (
        panel[["n_reviews", "n_tips", "n_checkins"]].fillna(0).astype(int)
    )
    return panel.sort_values(["business_id", "month"]).reset_index(drop=True)


def test_streaming_panel_matches_brute_force() -> None:
    reviews, tips, checkins = _toy_reviews(), _toy_tips(), _toy_checkins()
    rid = {"b0", "b1", "b2"}  # exclude b3 (not a restaurant)

    # Stream reviews/tips in two chunks to exercise accumulation across chunks.
    rev_chunks = [reviews.iloc[:3], reviews.iloc[3:]]
    tip_chunks = [tips.iloc[:1], tips.iloc[1:]]
    streamed = corpus.stream_activity_panel(rev_chunks, tip_chunks, checkins, rid, log=False)
    brute = _brute_force_panel(reviews, tips, checkins, rid)

    s = streamed.set_index(["business_id", "month"]).sort_index()
    b = brute.set_index(["business_id", "month"]).sort_index()
    # Same business-month cells with review activity.
    common = b.index
    for key in common:
        assert s.loc[key, "n_reviews"] == b.loc[key, "n_reviews"]
        assert s.loc[key, "n_tips"] == b.loc[key, "n_tips"]
        assert s.loc[key, "n_checkins"] == b.loc[key, "n_checkins"]
        if b.loc[key, "n_reviews"] > 0:
            assert s.loc[key, "mean_stars"] == pytest.approx(b.loc[key, "mean_stars"])
    # b3 (non-restaurant) never appears.
    assert "b3" not in streamed["business_id"].values


def test_panel_mean_stars_is_weighted_correctly() -> None:
    reviews = pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b0"],
            "stars": [5, 1, 3],
            "date": ["2019-01-01", "2019-01-02", "2019-01-03"],
            "text": ["a", "b", "c"],
        }
    )
    panel = corpus.stream_activity_panel([reviews], None, None, {"b0"}, log=False)
    row = panel[(panel.business_id == "b0") & (panel.month == "2019-01")].iloc[0]
    assert row["n_reviews"] == 3
    assert row["mean_stars"] == pytest.approx(3.0)  # (5+1+3)/3


# --------------------------------------------------------------------------- #
# Restaurant universe flagging
# --------------------------------------------------------------------------- #


def test_build_all_restaurants_flags_cohort_and_filters() -> None:
    business = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2"],
            "name": ["Chregg's", "Hardware Depot", "Chregg's"],
            "categories": ["Restaurants, Burgers", "Hardware, Shopping", "Restaurants, Fast Food"],
            "stars": [4.0, 4.5, 3.5],
            "review_count": [10, 5, 8],
            "is_open": [1, 1, 0],
            "city": ["Tempe"] * 3,
            "state": ["AZ"] * 3,
            "latitude": [33.4, 33.5, 33.6],
            "longitude": [-111.9, -112.0, -112.1],
        }
    )
    from lrr import cohort as C

    clean = C.clean_business(business)
    out = corpus.build_all_restaurants(clean, cohort_business_ids=["b0"])
    # Hardware store excluded; two restaurants kept.
    assert set(out["business_id"]) == {"b0", "b2"}
    # Both share the normalized chain "chreggs" -> Fast Food because b2 is tagged.
    assert set(out["cluster"]) == {"Fast Food"}
    assert out.set_index("business_id").loc["b0", "is_cohort"] == True  # noqa: E712
    assert out.set_index("business_id").loc["b2", "is_cohort"] == False  # noqa: E712


# --------------------------------------------------------------------------- #
# group_key and the out-of-fold stacking invariant
# --------------------------------------------------------------------------- #


def test_group_key_chained_vs_singleton() -> None:
    df = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2", "b3"],
            "chain": ["bigchain", "bigchain", "solo_a", "solo_b"],
        }
    )
    gk = F.group_key(df)
    # Chained businesses share the chain key; singletons use their business id.
    assert gk.tolist() == ["bigchain", "bigchain", "b2", "b3"]


def test_oof_stacking_invariant_no_group_in_training() -> None:
    # Simulate the fold assignment that oof_tier1_scores relies on and assert the
    # invariant that each group lands in exactly one held-out fold.
    from sklearn.model_selection import GroupKFold

    df = pd.DataFrame(
        {
            "business_id": [f"b{i}" for i in range(12)],
            "chain": (["A"] * 4) + (["B"] * 4) + ["C", "D", "E", "F"],
        }
    )
    group = F.group_key(df)
    fold_of_row = pd.Series(index=df.index, dtype=int)
    gkf = GroupKFold(n_splits=min(config.N_FOLDS, group.nunique()))
    for f, (_, test_idx) in enumerate(gkf.split(df, groups=group.astype(str))):
        fold_of_row.iloc[test_idx] = f
    # Core anti-leakage assertion used by the stacking stage.
    S.assert_oof_no_group_leak(group, fold_of_row)
    # And explicitly: for each row, its group is absent from every OTHER fold's rows.
    gmap = pd.DataFrame({"group": group.astype(str).to_numpy(), "fold": fold_of_row.to_numpy()})
    for grp, sub in gmap.groupby("group"):
        assert sub["fold"].nunique() == 1


def test_assert_oof_no_group_leak_raises_on_split_group() -> None:
    group = pd.Series(["A", "A", "B"])
    fold_of_row = pd.Series([0, 1, 1])  # group A split across folds 0 and 1
    with pytest.raises(AssertionError):
        S.assert_oof_no_group_leak(group, fold_of_row)


# --------------------------------------------------------------------------- #
# Final-score rule
# --------------------------------------------------------------------------- #


def test_final_score_rule_cohort_uses_tier2_else_tier1() -> None:
    risk = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2"],
            "is_cohort": [True, True, False],
            "tier1_score": [0.10, 0.20, 0.30],
            "tier2_score": [0.90, np.nan, 0.80],
        }
    )
    fs = S.final_score(risk)
    # b0 cohort with tier2 -> tier2 (0.90).
    assert fs[0] == pytest.approx(0.90)
    # b1 cohort but tier2 is NaN -> fall back to tier1 (0.20).
    assert fs[1] == pytest.approx(0.20)
    # b2 non-cohort -> tier1 (0.30).
    assert fs[2] == pytest.approx(0.30)


def test_final_score_rule_text_is_documented() -> None:
    assert "tier2_score for cohort" in config.FINAL_SCORE_RULE
    assert S.FINAL_SCORE_RULE == config.FINAL_SCORE_RULE


# --------------------------------------------------------------------------- #
# Streaming sentiment reducer
# --------------------------------------------------------------------------- #


def test_score_reviews_streaming_matches_brute_force() -> None:
    from lrr import sentiment as SENT

    reviews = pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b1"],
            "date": ["2019-01-01", "2019-01-15", "2019-02-01"],
            "text": ["good", "bad", "ok"],
        }
    )

    class FakeScorer:
        # Deterministic P(positive) by text so we can brute-force the mean.
        _map = {"good": 0.9, "bad": 0.1, "ok": 0.5}

        def predict_proba(self, texts):
            p = np.array([self._map[t] for t in texts])
            return np.column_stack([1 - p, p])

    out = SENT.score_reviews_streaming(
        [reviews.iloc[:2], reviews.iloc[2:]], FakeScorer(), {"b0", "b1"}
    )
    row = out[(out.business_id == "b0") & (out.month == "2019-01")].iloc[0]
    assert row["n"] == 2
    assert row["mean_p_positive"] == pytest.approx((0.9 + 0.1) / 2)
