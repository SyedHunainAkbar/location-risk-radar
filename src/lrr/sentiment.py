"""Sentiment labeling, benchmarking, calibration, and scoring (LA2).

We map star ratings to polarity labels, split with grouped validation so no
business leaks across folds, benchmark a VADER baseline against bag-of-words and
TF-IDF models, report metrics with bootstrap 95% confidence intervals, choose the
production scorer by macro F1, calibrate it to emit `P(positive)`, and score every
cohort review including 3-star ones.

VADER is imported lazily so this module imports without `vaderSentiment` installed;
the trainable models depend only on scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lrr import config

# Shared benchmark column order (LA2 and LA3 both emit this).
BENCHMARK_COLUMNS: tuple[str, ...] = (
    "model",
    "features",
    "accuracy",
    "accuracy_lo",
    "accuracy_hi",
    "macro_f1",
    "macro_f1_lo",
    "macro_f1_hi",
    "recall_neg",
    "recall_pos",
    "roc_auc",
    "roc_auc_lo",
    "roc_auc_hi",
    "n_test",
)

CLASSIFIER_NAMES: tuple[str, ...] = (
    "MultinomialNB",
    "LinearSVC",
    "LogisticRegression",
)
VECTORIZER_KINDS: tuple[str, ...] = ("count", "tfidf")


# --------------------------------------------------------------------------- #
# Labeling and splitting
# --------------------------------------------------------------------------- #


def map_polarity_labels(df: pd.DataFrame, stars_col: str = "stars") -> pd.DataFrame:
    """Map stars to polarity labels; drop neutral (3-star) rows.

    1-2 stars become label 0 (negative), 4-5 stars become label 1 (positive),
    and 3-star rows are removed. Adds a ``label`` column.
    """
    out = df.copy()
    out["_stars_num"] = pd.to_numeric(out[stars_col], errors="coerce")
    keep = config.NEGATIVE_STARS + config.POSITIVE_STARS
    out = out[out["_stars_num"].isin(keep)].copy()
    out["label"] = out["_stars_num"].isin(config.POSITIVE_STARS).astype(int)
    out = out.drop(columns="_stars_num")
    return out.reset_index(drop=True)


def grouped_split(
    df: pd.DataFrame,
    group_col: str = "business_id",
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so all rows of a group stay in one fold (no cross-fold leakage).

    Uses :class:`GroupShuffleSplit` keyed on ``group_col``. Deterministic for a
    given seed.

    Returns:
        (train_df, test_df) with disjoint group values.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col].astype(str)))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #


def build_vectorizer(kind: str):
    """Return a configured CountVectorizer or TfidfVectorizer."""
    params = dict(
        ngram_range=config.NGRAM_RANGE,
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
    )
    if kind == "count":
        return CountVectorizer(**params)
    if kind == "tfidf":
        return TfidfVectorizer(**params)
    raise ValueError(f"Unknown vectorizer kind {kind!r}.")


def build_classifier(name: str):
    """Return a configured classifier by name."""
    if name == "MultinomialNB":
        return MultinomialNB()
    if name == "LinearSVC":
        return LinearSVC()
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=config.SEED)
    raise ValueError(f"Unknown classifier {name!r}.")


def _scores_from_estimator(estimator, X) -> np.ndarray:
    """Return a positive-class score for ROC AUC.

    Uses ``predict_proba`` when available, else ``decision_function`` (LinearSVC).
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


# --------------------------------------------------------------------------- #
# VADER baseline
# --------------------------------------------------------------------------- #


def vader_predict(texts) -> tuple[np.ndarray, np.ndarray]:
    """Predict polarity with VADER. Positive if compound > threshold.

    Returns (labels, compound_scores). Imports VADER lazily.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    compounds = np.array([analyzer.polarity_scores(str(t))["compound"] for t in texts])
    labels = (compounds > config.VADER_POS_THRESHOLD).astype(int)
    return labels, compounds


# --------------------------------------------------------------------------- #
# Metrics and bootstrap CIs
# --------------------------------------------------------------------------- #


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict:
    """Compute accuracy, macro F1, per-class recall, and ROC AUC."""
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_neg": float(recalls[0]),
        "recall_pos": float(recalls[1]),
        "roc_auc": float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    metric: str,
    n: int = config.BOOTSTRAP_N,
    ci: float = config.BOOTSTRAP_CI,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Bootstrap a two-sided confidence interval for one metric.

    Resamples test rows with replacement `n` times and returns the lower and
    upper percentiles for the requested metric.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = None if y_score is None else np.asarray(y_score)
    m = len(y_true)
    if m == 0:
        return float("nan"), float("nan")

    def one(idx: np.ndarray) -> float:
        yt, yp = y_true[idx], y_pred[idx]
        ys = None if y_score is None else y_score[idx]
        if metric == "accuracy":
            return accuracy_score(yt, yp)
        if metric == "macro_f1":
            return f1_score(yt, yp, average="macro", zero_division=0)
        if metric == "roc_auc":
            if ys is None or len(np.unique(yt)) < 2:
                return float("nan")
            return roc_auc_score(yt, ys)
        raise ValueError(f"Unsupported bootstrap metric {metric!r}.")

    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        val = one(idx)
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return float("nan"), float("nan")
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return float(np.percentile(stats, lo_pct)), float(np.percentile(stats, hi_pct))


def _benchmark_row(
    model: str,
    features: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict:
    base = evaluate(y_true, y_pred, y_score)
    acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, y_score, "accuracy")
    f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, y_score, "macro_f1")
    auc_lo, auc_hi = bootstrap_ci(y_true, y_pred, y_score, "roc_auc")
    return {
        "model": model,
        "features": features,
        "accuracy": base["accuracy"],
        "accuracy_lo": acc_lo,
        "accuracy_hi": acc_hi,
        "macro_f1": base["macro_f1"],
        "macro_f1_lo": f1_lo,
        "macro_f1_hi": f1_hi,
        "recall_neg": base["recall_neg"],
        "recall_pos": base["recall_pos"],
        "roc_auc": base["roc_auc"],
        "roc_auc_lo": auc_lo,
        "roc_auc_hi": auc_hi,
        "n_test": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    include_vader: bool = True,
) -> pd.DataFrame:
    """Benchmark VADER plus every vectorizer x classifier combination.

    Returns a tidy DataFrame with :data:`BENCHMARK_COLUMNS`.
    """
    rows: list[dict] = []
    X_train = train[text_col].astype(str).tolist()
    X_test = test[text_col].astype(str).tolist()
    y_train = train[label_col].to_numpy()
    y_test = test[label_col].to_numpy()

    if include_vader:
        try:
            v_pred, v_score = vader_predict(X_test)
            rows.append(_benchmark_row("VADER", "lexicon", y_test, v_pred, v_score))
        except ImportError:
            pass  # VADER optional; skip if not installed

    for kind in VECTORIZER_KINDS:
        vec = build_vectorizer(kind)
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        for name in CLASSIFIER_NAMES:
            clf = build_classifier(name)
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            y_score = _scores_from_estimator(clf, Xte)
            rows.append(_benchmark_row(name, kind, y_test, y_pred, y_score))

    return pd.DataFrame(rows, columns=list(BENCHMARK_COLUMNS))


def pick_production(benchmark_df: pd.DataFrame) -> dict:
    """Return the best row by macro F1 as a dict (features + model)."""
    best = benchmark_df.sort_values("macro_f1", ascending=False).iloc[0]
    return {
        "model": str(best["model"]),
        "features": str(best["features"]),
        "macro_f1": float(best["macro_f1"]),
    }


# --------------------------------------------------------------------------- #
# Calibration and scoring
# --------------------------------------------------------------------------- #


def calibrate_scorer(
    train: pd.DataFrame,
    features: str,
    model: str,
    text_col: str = "text",
    label_col: str = "label",
) -> Pipeline:
    """Fit a calibrated text->P(positive) pipeline.

    Wraps the chosen classifier in `CalibratedClassifierCV` so even LinearSVC
    yields probabilities. Returns a fitted sklearn Pipeline (vectorizer +
    calibrated classifier).
    """
    vec = build_vectorizer(features)
    clf = build_classifier(model)
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    pipe = Pipeline([("vectorizer", vec), ("classifier", calibrated)])
    pipe.fit(train[text_col].astype(str).tolist(), train[label_col].to_numpy())
    return pipe


def score_all(
    reviews_df: pd.DataFrame,
    scorer: Pipeline,
    text_col: str = "text",
    id_col: str = "review_id",
    group_col: str = "business_id",
) -> pd.DataFrame:
    """Score every review (including 3-star) with P(positive).

    Returns columns ``review_id, business_id, p_positive``.
    """
    texts = reviews_df[text_col].astype(str).tolist()
    proba = scorer.predict_proba(texts)[:, 1]
    return pd.DataFrame(
        {
            id_col: reviews_df[id_col].to_numpy(),
            group_col: reviews_df[group_col].to_numpy(),
            "p_positive": proba.astype(float),
        }
    )


# --------------------------------------------------------------------------- #
# Full-corpus two-tier additions: non-cohort training and transfer test
# --------------------------------------------------------------------------- #


def noncohort_training_sample(
    reviews: pd.DataFrame,
    all_restaurants: pd.DataFrame,
    n: int = config.NONCOHORT_SAMPLE_SIZE,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Draw a labeled sample from NON-cohort restaurant reviews.

    Keeps 1-2 and 4-5 star reviews (via :func:`map_polarity_labels`) belonging to
    restaurants with ``is_cohort == False``, then samples up to ``n`` rows. The
    caller splits with :func:`grouped_split` so businesses never cross folds.
    """
    noncohort_ids = set(
        all_restaurants.loc[~all_restaurants["is_cohort"], "business_id"].astype(str)
    )
    rev = reviews[reviews["business_id"].astype(str).isin(noncohort_ids)]
    labeled = map_polarity_labels(rev)
    if n and len(labeled) > n:
        labeled = labeled.sample(n=n, random_state=seed).reset_index(drop=True)
    return labeled


def transfer_evaluate(
    scorer: Pipeline,
    noncohort_test: pd.DataFrame,
    cohort_reviews: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Evaluate the scorer on (a) held-out non-cohort and (b) cohort transfer.

    Both evaluations use the shared benchmark metric set with bootstrap CIs. The
    cohort frame is labeled with :func:`map_polarity_labels` (3-star dropped) so the
    transfer metrics are comparable. Returns one row per evaluation split, tagged in
    a ``split`` column.
    """

    def _eval(df: pd.DataFrame, split: str) -> dict:
        labeled = df if label_col in df.columns else map_polarity_labels(df)
        if labeled.empty:
            return {
                "split": split,
                **{c: float("nan") for c in BENCHMARK_COLUMNS if c not in ("model", "features")},
                "model": "production",
                "features": "tfidf",
            }
        X = labeled[text_col].astype(str).tolist()
        y = labeled[label_col].to_numpy()
        proba = scorer.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = _benchmark_row("production", "tfidf", y, pred, proba)
        row["split"] = split
        return row

    rows = [
        _eval(noncohort_test, "noncohort_test"),
        _eval(cohort_reviews, "cohort_transfer"),
    ]
    cols = ["split"] + list(BENCHMARK_COLUMNS)
    return pd.DataFrame(rows)[cols]


def score_reviews_streaming(
    review_chunks,
    scorer: Pipeline,
    restaurant_ids,
    text_col: str = "text",
    id_col: str = "business_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Score reviews in streaming chunks and reduce to monthly aggregates.

    Returns per (business_id, month): ``mean_p_positive`` and ``n``. Only aggregates
    are retained; review text is never accumulated. Restaurant filtering keeps the
    output to the universe of interest.
    """
    from collections import defaultdict

    rid = set(map(str, restaurant_ids))
    acc: dict = defaultdict(lambda: [0.0, 0])  # (bid, month) -> [p_sum, n]
    for chunk in review_chunks:
        sub = chunk[chunk[id_col].astype(str).isin(rid)]
        if sub.empty:
            continue
        proba = scorer.predict_proba(sub[text_col].astype(str).tolist())[:, 1]
        months = pd.to_datetime(sub[date_col], errors="coerce").dt.to_period("M").astype(str)
        for bid, m, p in zip(sub[id_col].astype(str).to_numpy(), months.to_numpy(), proba):
            if m == "NaT":
                continue
            cell = acc[(bid, m)]
            cell[0] += float(p)
            cell[1] += 1
    rows = [
        {
            "business_id": bid,
            "month": m,
            "mean_p_positive": (psum / n) if n else float("nan"),
            "n": int(n),
        }
        for (bid, m), (psum, n) in acc.items()
    ]
    cols = ["business_id", "month", "mean_p_positive", "n"]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["business_id", "month"])
        .reset_index(drop=True)
    )
