"""Tests for lrr.sentiment: label mapping, grouped split, metrics, scoring.

These run on scikit-learn and pandas only. No spaCy model, no VADER, no
TensorFlow, no GPU, no real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lrr import config
from lrr import sentiment as S

# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #


def _toy_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(6)],
            "business_id": ["b0", "b0", "b1", "b1", "b2", "b2"],
            "stars": [1, 2, 3, 4, 5, 3],
            "text": ["bad", "poor", "meh", "good", "great", "ok"],
        }
    )


def test_label_mapping_drops_three_star() -> None:
    labeled = S.map_polarity_labels(_toy_reviews())
    # Two 3-star rows dropped -> 4 remain.
    assert len(labeled) == 4
    assert 3 not in labeled["stars"].tolist()


def test_label_mapping_boundaries() -> None:
    labeled = S.map_polarity_labels(_toy_reviews()).set_index("review_id")
    assert labeled.loc["r0", "label"] == 0  # 1 star
    assert labeled.loc["r1", "label"] == 0  # 2 star
    assert labeled.loc["r3", "label"] == 1  # 4 star
    assert labeled.loc["r4", "label"] == 1  # 5 star


def test_label_values_are_binary_ints() -> None:
    labeled = S.map_polarity_labels(_toy_reviews())
    assert set(labeled["label"].unique()).issubset({0, 1})
    assert labeled["label"].dtype.kind in ("i", "u")


# --------------------------------------------------------------------------- #
# Grouped split: no business leakage
# --------------------------------------------------------------------------- #


def _grouped_frame(n_business: int = 40, per: int = 10) -> pd.DataFrame:
    rows = []
    for b in range(n_business):
        label = int(b % 2)
        for j in range(per):
            rows.append(
                {
                    "review_id": f"r{b}_{j}",
                    "business_id": f"b{b}",
                    "stars": 5 if label else 1,
                    "label": label,
                    "text": "good food" if label else "bad food",
                }
            )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_grouped_split_has_no_business_overlap() -> None:
    df = _grouped_frame()
    train, test = S.grouped_split(df)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"]))
    # Every row of a business lands on exactly one side.
    assert len(train) + len(test) == len(df)


def test_grouped_split_is_reproducible() -> None:
    df = _grouped_frame()
    a_train, a_test = S.grouped_split(df, seed=config.SEED)
    b_train, b_test = S.grouped_split(df, seed=config.SEED)
    assert set(a_test["business_id"]) == set(b_test["business_id"])


def test_grouped_split_respects_test_size() -> None:
    df = _grouped_frame(n_business=50, per=4)
    _, test = S.grouped_split(df, test_size=0.2)
    frac = len(test) / len(df)
    assert 0.1 < frac < 0.3  # grouped split is approximate


# --------------------------------------------------------------------------- #
# Metrics and bootstrap CIs
# --------------------------------------------------------------------------- #


def test_evaluate_perfect_prediction() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    out = S.evaluate(y, y, score)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["recall_neg"] == 1.0
    assert out["recall_pos"] == 1.0
    assert out["roc_auc"] == 1.0


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    flip = rng.choice(200, size=30, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    point = S.evaluate(y_true, y_pred, y_pred.astype(float))["accuracy"]
    lo, hi = S.bootstrap_ci(y_true, y_pred, y_pred.astype(float), "accuracy")
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------- #
# Benchmark, calibration, scoring (classical only; VADER skipped if absent)
# --------------------------------------------------------------------------- #


def _separable_corpus(n: int = 60) -> pd.DataFrame:
    rows = []
    for b in range(n):
        label = int(b % 2)
        text = "delicious friendly clean fast" if label else "terrible slow dirty rude"
        rows.append(
            {
                "review_id": f"r{b}",
                "business_id": f"b{b}",
                "stars": 5 if label else 1,
                "label": label,
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def test_benchmark_and_pick_production() -> None:
    df = _separable_corpus()
    train, test = S.grouped_split(df)
    bench = S.benchmark(train, test, include_vader=False)
    # Expect 2 vectorizers x 3 classifiers = 6 rows.
    assert len(bench) == 6
    assert list(bench.columns) == list(S.BENCHMARK_COLUMNS)
    prod = S.pick_production(bench)
    assert prod["features"] in ("count", "tfidf")
    assert prod["model"] in S.CLASSIFIER_NAMES


def test_calibrate_and_score_all_outputs_probabilities() -> None:
    df = _separable_corpus(n=80)
    train, _ = S.grouped_split(df)
    scorer = S.calibrate_scorer(train, "tfidf", "LogisticRegression")
    # Score ALL rows, including a synthetic 3-star neutral row.
    all_reviews = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "review_id": "r_neutral",
                        "business_id": "b0",
                        "stars": 3,
                        "label": 0,
                        "text": "it was fine i guess",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    scored = S.score_all(all_reviews, scorer)
    assert list(scored.columns) == ["review_id", "business_id", "p_positive"]
    assert len(scored) == len(all_reviews)  # 3-star row scored too
    assert scored["p_positive"].between(0.0, 1.0).all()
