"""Tests for lrr.aspects: grounding, scoring, JSON validation + retry.

No rapidfuzz required (pure-Python grounding fallback). LLM calls use the mock
provider or an injected fake. scipy is used only in the risk-contrast test.
"""

from __future__ import annotations

import numpy as np
import pytest

from lrr import aspects as A
from lrr import config

# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def test_ground_quote_accepts_verbatim_span() -> None:
    source = "The fries were hot and crispy but the line was very slow."
    assert A.ground_quote("fries were hot and crispy", source) is True


def test_ground_quote_rejects_too_few_words() -> None:
    source = "Great burgers here."
    assert A.ground_quote("great", source) is False  # one word


def test_ground_quote_rejects_ungrounded() -> None:
    source = "The staff were friendly and the food was fresh."
    assert A.ground_quote("worst sushi in the entire city", source) is False


# --------------------------------------------------------------------------- #
# Scoring math (Python computes every number)
# --------------------------------------------------------------------------- #


def test_polarity_sign() -> None:
    assert A.polarity_sign("positive") == 1
    assert A.polarity_sign("negative") == -1
    assert A.polarity_sign("neutral") == 0


def test_aspect_score_known() -> None:
    items = [
        {"quote": "a a", "polarity": "positive", "intensity": 3},  # +1.0
        {"quote": "b b", "polarity": "negative", "intensity": 3},  # -1.0
    ]
    assert A.aspect_score(items) == pytest.approx(0.0)
    items2 = [{"quote": "c c", "polarity": "positive", "intensity": 3}]
    assert A.aspect_score(items2) == pytest.approx(1.0)
    assert A.aspect_score([]) == 0.0


def test_predicted_stars_formula_and_clip() -> None:
    # All strongly positive -> score 1.0 -> stars 5.0.
    pos = [{"quote": "x x", "polarity": "positive", "intensity": 3}]
    assert A.predicted_stars(pos) == pytest.approx(5.0)
    # All strongly negative -> score -1.0 -> stars 1.0.
    neg = [{"quote": "y y", "polarity": "negative", "intensity": 3}]
    assert A.predicted_stars(neg) == pytest.approx(1.0)
    # Neutral -> 3.0.
    assert A.predicted_stars([]) == pytest.approx(3.0)
    # Mid intensity positive: score = 1/3 -> stars = 3 + 2/3.
    mid = [{"quote": "z z", "polarity": "positive", "intensity": 1}]
    assert A.predicted_stars(mid) == pytest.approx(3.0 + 2.0 / 3.0)


# --------------------------------------------------------------------------- #
# JSON schema validation and retry
# --------------------------------------------------------------------------- #


def test_parse_aspect_json_valid() -> None:
    raw = '[{"quote": "cold fries", "polarity": "negative", "intensity": 2}]'
    items = A.parse_aspect_json(raw)
    assert items == [{"quote": "cold fries", "polarity": "negative", "intensity": 2}]


def test_parse_aspect_json_strips_code_fence() -> None:
    raw = '```json\n[{"quote": "nice staff", "polarity": "positive", "intensity": 1}]\n```'
    items = A.parse_aspect_json(raw)
    assert items[0]["polarity"] == "positive"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        '[{"quote": "", "polarity": "positive", "intensity": 1}]',  # empty quote
        '[{"quote": "ok ok", "polarity": "meh", "intensity": 1}]',  # bad polarity
        '[{"quote": "ok ok", "polarity": "positive", "intensity": 9}]',  # out of range
    ],
)
def test_parse_aspect_json_invalid_raises(raw: str) -> None:
    with pytest.raises(A.AspectSchemaError):
        A.parse_aspect_json(raw)


def test_run_aspect_agent_retries_then_succeeds(monkeypatch) -> None:
    # First response invalid, second valid -> agent should recover within retries.
    calls = {"n": 0}

    def fake_chat(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "garbage, not json"
        return '[{"quote": "slow service", "polarity": "negative", "intensity": 2}]'

    monkeypatch.setattr(A.llm, "chat", fake_chat)
    items = A.run_aspect_agent("service", "the service was slow", retries=2)
    assert items[0]["quote"] == "slow service"
    assert calls["n"] == 2


def test_run_aspect_agent_exhausts_retries(monkeypatch) -> None:
    monkeypatch.setattr(A.llm, "chat", lambda messages, **kwargs: "still not json")
    with pytest.raises(A.AspectSchemaError):
        A.run_aspect_agent("food", "whatever", retries=1)


# --------------------------------------------------------------------------- #
# analyze_review drops ungrounded quotes
# --------------------------------------------------------------------------- #


def test_analyze_review_drops_ungrounded(monkeypatch) -> None:
    text = "The burger was cold and the fries were soggy."

    def fake_chat(messages, **kwargs):
        # Aspect agents return one grounded + one hallucinated quote; lead is JSON obj.
        content = messages[-1]["content"].lower()
        if "overall" in content:
            return '{"polarity": "negative", "intensity": 2}'
        return (
            '[{"quote": "burger was cold", "polarity": "negative", "intensity": 2},'
            ' {"quote": "valet parking was great", "polarity": "positive", "intensity": 3}]'
        )

    monkeypatch.setattr(A.llm, "chat", fake_chat)
    out = A.analyze_review(text, provider="mock")
    # The hallucinated quote is dropped for each of the 3 aspects.
    assert out["dropped_quotes"] == len(config.ASPECTS)
    for aspect in config.ASPECTS:
        items = out["aspects"][aspect]["items"]
        assert all("burger was cold" == it["quote"] for it in items)


# --------------------------------------------------------------------------- #
# Evaluation and risk contrast
# --------------------------------------------------------------------------- #


def test_evaluate_predicted_stars_perfect() -> None:
    pred = np.array([1.0, 2.0, 4.0, 5.0])
    out = A.evaluate_predicted_stars(pred, pred)
    assert out["mae"] == pytest.approx(0.0)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["pearson_r"] == pytest.approx(1.0)


def test_risk_contrast_detects_difference() -> None:
    high = [-0.8, -0.7, -0.9, -0.6, -0.75]
    low = [0.6, 0.7, 0.5, 0.8, 0.65]
    out = A.risk_contrast(high, low)
    assert out["p_value"] < 0.05
    assert out["median_high"] < out["median_low"]
