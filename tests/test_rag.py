"""Tests for lrr.rag: chunking, retrieval, judging, cache, index-size assert.

Embeddings are injected (no sentence-transformers), LLM calls use the mock provider
or injected fakes. No network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lrr import config
from lrr import rag as R

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def test_chunk_text_respects_word_cap() -> None:
    text = " ".join(f"w{i}" for i in range(250))
    chunks = R.chunk_text(text, max_words=120)
    assert len(chunks) == 3  # 120 + 120 + 10
    assert all(len(c.split()) <= 120 for c in chunks)


def test_chunk_text_empty() -> None:
    assert R.chunk_text("") == []
    assert R.chunk_text("   ") == []


# --------------------------------------------------------------------------- #
# Retrieval: cosine and MMR
# --------------------------------------------------------------------------- #


def _toy_index():
    # 4 orthogonal-ish vectors in 3D with metadata.
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    meta = pd.DataFrame(
        {
            "chunk_id": [0, 1, 2, 3],
            "business_id": ["b0", "b0", "b0", "b1"],
            "chain": ["A", "A", "A", "B"],
            "review_id": ["r0#0", "r1#0", "r2#0", "r3#0"],
            "date": pd.to_datetime(["2019-01-01", "2019-06-01", "2020-01-01", "2020-02-01"]),
            "stars": [1, 5, 2, 1],
            "text": ["slow service", "great food", "noisy room", "other place"],
        }
    )
    return emb, meta


def test_cosine_top_k_orders_by_similarity() -> None:
    emb, _ = _toy_index()
    q = np.array([1.0, 0.0, 0.0])
    idx = R.cosine_top_k(q, emb, k=2)
    assert idx[0] == 0  # exact match first
    assert set(idx.tolist()) == {0, 1}  # two closest to the x-axis


def test_retrieve_filters_by_business_id() -> None:
    emb, meta = _toy_index()
    q = np.array([1.0, 0.0, 0.0])
    out = R.retrieve(q, emb, meta, business_id="b1", k=5)
    assert set(out["business_id"]) == {"b1"}


def test_retrieve_filters_by_stars_and_date() -> None:
    emb, meta = _toy_index()
    q = np.array([1.0, 0.0, 0.0])
    out = R.retrieve(
        q, emb, meta, business_id="b0", k=5, max_stars=2, date_range=("2018-01-01", "2019-12-31")
    )
    # Only b0 rows with stars <= 2 and date in 2018-2019: r0#0 (stars 1, 2019-01).
    assert out["review_id"].tolist() == ["r0#0"]


def test_mmr_promotes_diversity() -> None:
    emb, meta = _toy_index()
    q = np.array([1.0, 0.0, 0.0])
    # Candidates are the two x-axis vectors (0,1) plus the y vector (2).
    cand = np.array([0, 1, 2])
    # Weight diversity more (lambda 0.3) so the near-duplicate is penalized.
    picked = R.mmr(q, emb, cand, k=2, lambda_=0.3)
    # First pick is the most relevant (0); second should be the diverse one (2),
    # not the near-duplicate (1).
    assert picked[0] == 0
    assert picked[1] == 2


# --------------------------------------------------------------------------- #
# Generation and judging
# --------------------------------------------------------------------------- #


def test_generate_answer_abstains_without_snippets() -> None:
    out = R.generate_answer("why at risk?", pd.DataFrame(), provider="mock")
    assert out == "Not enough evidence."


def test_cited_ids_and_split_sentences() -> None:
    answer = "Service is slow [r0#0]. Food is cold [r1#0]. No citation here."
    sents = R.split_sentences(answer)
    assert len(sents) == 3
    assert R.cited_ids(sents[0]) == ["r0#0"]
    assert R.cited_ids(sents[2]) == []


def test_judge_sentences_and_groundedness(monkeypatch) -> None:
    snippets = pd.DataFrame(
        {
            "review_id": ["r0#0", "r1#0"],
            "text": ["the service was very slow", "the food arrived cold"],
        }
    )
    answer = "Service is slow [r0#0]. Food is cold [r1#0]. Parking is bad."

    def fake_chat(messages, **kwargs):
        # Judge: support when the evidence mentions the claim's keyword.
        content = messages[-1]["content"].lower()
        if "slow" in content and "service" in content:
            return "supported"
        if "cold" in content and "food" in content:
            return "supported"
        return "unsupported"

    monkeypatch.setattr(R.llm, "chat", fake_chat)
    labels = R.judge_sentences(answer, snippets, provider="mock")
    # Third sentence has no citation -> unsupported automatically.
    assert len(labels) == 3
    assert labels[2][1] is False
    rate = R.groundedness_rate(labels)
    assert rate == pytest.approx(2 / 3)
    visible = R.visible_answer(labels)
    assert "Parking is bad" not in visible
    assert "Service is slow" in visible


def test_groundedness_rate_empty_is_one() -> None:
    assert R.groundedness_rate([]) == 1.0


# --------------------------------------------------------------------------- #
# Index build + size assertion
# --------------------------------------------------------------------------- #


def test_build_index_writes_float16_and_asserts_size(tmp_path) -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r0", "r1"],
            "business_id": ["b0", "b0"],
            "stars": [1, 5],
            "date": ["2019-01-01", "2020-01-01"],
            "text": ["slow and cold service", "great fresh food"],
        }
    )
    tips = pd.DataFrame({"business_id": ["b0"], "text": ["try the fries"], "date": ["2019-05-01"]})
    locations = pd.DataFrame({"business_id": ["b0"], "chain": ["A"]})

    # Inject a deterministic fake encoder (no sentence-transformers).
    def fake_encode(texts):
        return np.ones((len(texts), 8), dtype="float32")

    info = R.build_index(reviews, tips, locations, out_dir=tmp_path, encode_fn=fake_encode)
    assert info["n_chunks"] == 3  # 2 reviews + 1 tip, each short -> 1 chunk
    assert info["bytes"] < config.MAX_ARTIFACT_BYTES

    emb, meta = R.load_index(tmp_path)
    assert emb.dtype == np.float32  # loaded as float32
    assert list(meta.columns) == R.META_COLUMNS
    # Saved embeddings are float16 on disk.
    raw = np.load(tmp_path / config.RAG_EMBEDDINGS_FILE)
    assert raw.dtype == np.float16


def test_build_rag_cache_with_injected_encoder(monkeypatch, tmp_path) -> None:
    emb, meta = _toy_index()

    monkeypatch.setattr(R.llm, "chat", lambda messages, **kwargs: "Service is slow [r0#0].")

    def q_encode(_q):
        return np.array([1.0, 0.0, 0.0])

    cache = R.build_rag_cache(
        emb, meta, ["b0"], questions=["why at risk?"], query_encode_fn=q_encode, provider="mock"
    )
    assert len(cache) == 1
    assert list(cache.columns) == [
        "business_id",
        "question",
        "answer",
        "visible_answer",
        "groundedness_rate",
        "n_citations",
    ]
    assert cache.iloc[0]["business_id"] == "b0"
