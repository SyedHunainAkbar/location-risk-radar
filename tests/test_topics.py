"""Tests for lrr.topics.

These exercise seed topics, the diversity metric, pre-landmark topic-share
windowing, LLM labeling via the mock provider, and label overrides. No BERTopic,
UMAP, HDBSCAN, gensim, GPU, or network required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lrr import config
from lrr import topics as T

# --------------------------------------------------------------------------- #
# Seed topics
# --------------------------------------------------------------------------- #


def test_seed_topics_count_and_shape() -> None:
    assert len(T.SEED_TOPICS) == 9
    assert len(T.SEED_TOPIC_PHRASES) == 9
    assert all(isinstance(words, list) and words for words in T.SEED_TOPICS)
    # Phrases mirror the config constant exactly.
    assert T.SEED_TOPIC_PHRASES == list(config.SEED_TOPIC_PHRASES)


def test_seed_phrases_cover_core_themes() -> None:
    joined = " ".join(T.SEED_TOPIC_PHRASES).lower()
    for theme in (
        "wait",
        "staff",
        "order",
        "food",
        "clean",
        "price",
        "complaint",
        "drive",
        "reservation",
    ):
        assert theme in joined


# --------------------------------------------------------------------------- #
# Diversity metric
# --------------------------------------------------------------------------- #


def test_topic_diversity_all_unique() -> None:
    topics = [["a", "b", "c"], ["d", "e", "f"]]
    assert T.topic_diversity(topics, top_k=3) == 1.0


def test_topic_diversity_all_identical() -> None:
    topics = [["a", "b"], ["a", "b"]]
    # 2 unique / 4 total = 0.5
    assert T.topic_diversity(topics, top_k=2) == 0.5


def test_topic_diversity_empty_is_nan() -> None:
    assert np.isnan(T.topic_diversity([], top_k=5))


# --------------------------------------------------------------------------- #
# Pre-landmark topic shares
# --------------------------------------------------------------------------- #


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b0", "b1", "b1", "b2"],
            "date": [
                "2017-06-01",  # pre-landmark, in window
                "2017-09-01",  # pre-landmark, in window
                "2018-03-01",  # AFTER landmark -> excluded
                "2017-05-01",  # pre-landmark
                "2015-01-01",  # older than 365-day window -> excluded if windowed
                "2019-01-01",  # after landmark -> b2 has nothing pre-landmark
            ],
            "topic": [0, 1, 0, 0, 1, 0],
        }
    )


def test_shares_exclude_on_or_after_landmark() -> None:
    landmark = date(2018, 1, 1)
    # No window so only the strict pre-landmark filter applies.
    shares = T.compute_topic_shares(_assignments(), landmark=landmark, window_days=None)
    shares = shares.set_index("business_id")
    # b0: two pre-landmark reviews (topic 0 and 1) -> 0.5 / 0.5. The 2018-03 row drops.
    assert shares.loc["b0", "topic_0"] == pytest.approx(0.5)
    assert shares.loc["b0", "topic_1"] == pytest.approx(0.5)


def test_shares_respect_window_days() -> None:
    landmark = date(2018, 1, 1)
    shares = T.compute_topic_shares(_assignments(), landmark=landmark, window_days=365)
    shares = shares.set_index("business_id")
    # b1: 2017-05 is in window (topic 0); 2015-01 is outside -> only topic 0 counts.
    assert shares.loc["b1", "topic_0"] == pytest.approx(1.0)
    assert shares.loc["b1", "topic_1"] == pytest.approx(0.0)


def test_shares_sum_to_one_or_zero() -> None:
    landmark = date(2018, 1, 1)
    shares = T.compute_topic_shares(_assignments(), landmark=landmark, window_days=365)
    topic_cols = [c for c in shares.columns if c.startswith("topic_")]
    row_sums = shares[topic_cols].sum(axis=1)
    for s in row_sums:
        assert s == pytest.approx(1.0) or s == pytest.approx(0.0)
    # b2 has no pre-landmark reviews -> all zeros.
    b2 = shares.set_index("business_id").loc["b2", topic_cols]
    assert b2.sum() == pytest.approx(0.0)


def test_shares_cover_all_businesses() -> None:
    landmark = date(2018, 1, 1)
    shares = T.compute_topic_shares(_assignments(), landmark=landmark)
    assert set(shares["business_id"]) == {"b0", "b1", "b2"}


def test_shares_drop_outlier_topic() -> None:
    df = pd.DataFrame(
        {
            "business_id": ["b0", "b0"],
            "date": ["2017-06-01", "2017-07-01"],
            "topic": [-1, 0],  # one outlier, one real
        }
    )
    shares = T.compute_topic_shares(df, landmark=date(2018, 1, 1), window_days=None)
    assert "topic_-1" not in shares.columns
    # Only the real topic counts -> share 1.0
    assert shares.set_index("business_id").loc["b0", "topic_0"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# LLM labeling (mock provider) and overrides
# --------------------------------------------------------------------------- #


def test_label_topics_llm_mock() -> None:
    topics_df = pd.DataFrame(
        {
            "topic": [-1, 0, 1],
            "top_words": [["noise"], ["slow", "wait", "line"], ["cold", "food"]],
            "rep_docs": [["x"], ["waited forever"], ["food was cold"]],
            "size": [10, 50, 40],
        }
    )
    labeled = T.label_topics_llm(topics_df, provider="mock")
    assert labeled.loc[labeled.topic == -1, "label"].iloc[0] == "Outliers"
    # Real topics get a non-empty, short label (<= 4 words) from the mock.
    for tid in (0, 1):
        lbl = labeled.loc[labeled.topic == tid, "label"].iloc[0]
        assert lbl and len(lbl.split()) <= 4


def test_apply_label_overrides(tmp_path: Path) -> None:
    labeled = pd.DataFrame({"topic": [0, 1], "label": ["auto a", "auto b"]})
    csv = tmp_path / "topic_labels.csv"
    pd.DataFrame({"topic": [1], "label": ["Human Override"]}).to_csv(csv, index=False)
    out = T.apply_label_overrides(labeled, csv)
    assert out.set_index("topic").loc[0, "label"] == "auto a"  # untouched
    assert out.set_index("topic").loc[1, "label"] == "Human Override"  # overridden


def test_apply_label_overrides_missing_file_is_noop(tmp_path: Path) -> None:
    labeled = pd.DataFrame({"topic": [0], "label": ["auto"]})
    out = T.apply_label_overrides(labeled, tmp_path / "does_not_exist.csv")
    assert out.equals(labeled)


# --------------------------------------------------------------------------- #
# Embedding cache path (no model load)
# --------------------------------------------------------------------------- #


def test_embedding_cache_path_is_stable_and_slugged(tmp_path: Path) -> None:
    texts = ["a", "b", "c"]
    p1 = T.embedding_cache_path("Non-Fast Food", texts, tmp_path)
    p2 = T.embedding_cache_path("Non-Fast Food", texts, tmp_path)
    assert p1 == p2
    assert "non_fast_food" in p1.name
    assert p1.suffix == ".npy"
    # Different corpus -> different cache file.
    assert T.embedding_cache_path("Non-Fast Food", ["x"], tmp_path) != p1
