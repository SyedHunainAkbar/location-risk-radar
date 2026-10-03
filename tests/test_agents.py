"""Tests for the risk committee agents.

Cover placeholder enforcement (digit rejection + retry), quote grounding, the
confidence formula, and orchestrator concurrency, all with injected fake completion
functions. No network and no API keys.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest

from lrr import agents
from lrr.agents import auditor as auditor_mod
from lrr.agents import confidence as conf
from lrr.agents import quant as quant_mod
from lrr.agents import voc as voc_mod
from lrr.agents.schemas import VoCItem
from lrr.gateway import cache as _cache
from lrr.gateway import health as _health
from lrr.gateway.core import reset_gateway as _reset_gateway

# --------------------------------------------------------------------------- #
# Fake completion response helper
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_gateway(tmp_path, monkeypatch):
    # Each test gets its own LLM disk cache and a fresh gateway/health state so
    # cached responses from one test never satisfy another test's call.
    monkeypatch.setattr(_cache.config, "LLM_CACHE_DIR", tmp_path / "llm")
    _health.reset_cache()
    _reset_gateway()
    yield
    _reset_gateway()


def _fake(content: str):
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _R:
        def __init__(self, c):
            self.choices = [_Choice(c)]
            self.usage = None

    return _R(content)


# --------------------------------------------------------------------------- #
# Placeholder enforcement (numeric fidelity)
# --------------------------------------------------------------------------- #


def test_enforce_placeholders_accepts_placeholder_only() -> None:
    assert quant_mod.enforce_placeholders(
        "Risk sits at {risk_percentile} with velocity {velocity_ratio}."
    )


def test_enforce_placeholders_rejects_literal_digit() -> None:
    assert not quant_mod.enforce_placeholders("Risk sits at the 92nd percentile.")
    assert not quant_mod.enforce_placeholders("velocity fell to 0.7")


def test_run_quant_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # first output has a literal number -> rejected
            return _fake(
                json.dumps(
                    {
                        "drivers": [
                            {
                                "driver_key": "velocity_ratio",
                                "direction": "down",
                                "explanation_template": "velocity fell to 0.7",
                            }
                        ]
                    }
                )
            )
        return _fake(
            json.dumps(
                {
                    "drivers": [
                        {
                            "driver_key": "velocity_ratio",
                            "direction": "down",
                            "explanation_template": "velocity fell to {velocity_ratio}",
                        }
                    ]
                }
            )
        )

    fs = {"business_id": "b0", "velocity_ratio": "0.70"}
    out = quant_mod.run_quant(fs, retries=2, completion_fn=completion)
    assert calls["n"] == 2
    assert all(quant_mod.enforce_placeholders(d.explanation_template) for d in out.drivers)


def test_run_quant_raises_after_persistent_digits() -> None:
    def completion(**kwargs):
        return _fake(
            json.dumps(
                {
                    "drivers": [
                        {
                            "driver_key": "x",
                            "direction": "up",
                            "explanation_template": "always 5 here",
                        }
                    ]
                }
            )
        )

    with pytest.raises(ValueError):
        quant_mod.run_quant({"business_id": "b0"}, retries=1, completion_fn=completion)


def test_render_injects_python_numbers() -> None:
    fs = {"velocity_ratio": "0.70", "risk_percentile": "92.00"}
    text = quant_mod.render("velocity {velocity_ratio} at {risk_percentile} pct", fs)
    assert text == "velocity 0.70 at 92.00 pct"
    # Unknown placeholder renders n/a, never a stray brace.
    assert quant_mod.render("{missing_key}", fs) == "n/a"


def test_scan_rendered_fidelity_traces_numbers_to_factsheet() -> None:
    fs = {"velocity_ratio": "0.70", "risk_percentile": "92.00", "surv_12m": "80%"}
    # Every number present in the fact sheet -> no stray tokens.
    assert (
        quant_mod.scan_rendered_fidelity("velocity 0.70, percentile 92.00, survival 80%", fs) == []
    )
    # An invented number not in the fact sheet is flagged.
    stray = quant_mod.scan_rendered_fidelity("velocity dropped by 45 points", fs)
    assert "45" in stray


def test_numeric_fidelity_rate_counts_clean_drivers() -> None:
    fs = {"velocity_ratio": "0.70", "risk_percentile": "92.00"}
    clean = "velocity is 0.70"
    dirty = "it fell 15 percent"  # 15 is not in the fact sheet
    assert quant_mod.numeric_fidelity_rate([clean, clean], fs) == pytest.approx(1.0)
    assert quant_mod.numeric_fidelity_rate([clean, dirty], fs) == pytest.approx(0.5)
    assert quant_mod.numeric_fidelity_rate([], fs) == 1.0


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def _snips():
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "text": [
                "the service was very slow and the line never moved",
                "cold fries again and the staff were rude",
            ],
        }
    )


def test_grounding_keeps_matching_quote() -> None:
    assert voc_mod.is_grounded("service was very slow", "r1", _snips())


def test_grounding_drops_hallucinated_quote() -> None:
    assert not voc_mod.is_grounded("the valet lost my car keys", "r1", _snips())


def test_grounding_requires_matching_review_id() -> None:
    # Correct text but wrong citation id -> not grounded.
    assert not voc_mod.is_grounded("service was very slow", "r2", _snips())


def test_grounding_requires_two_words() -> None:
    assert not voc_mod.is_grounded("slow", "r1", _snips())


def test_ground_items_counts_drops() -> None:
    items = [
        VoCItem(
            quote="service was very slow",
            review_id="r1",
            issue="service_speed",
            polarity="negative",
            intensity=3,
        ),
        VoCItem(
            quote="totally made up claim here",
            review_id="r2",
            issue="food_quality",
            polarity="negative",
            intensity=2,
        ),
    ]
    g = voc_mod.ground_items(items, _snips())
    assert len(g["grounded"]) == 1
    assert g["dropped"] == 1
    assert g["n_proposed"] == 2


def test_issue_scores_and_prevalence() -> None:
    items = [
        VoCItem(
            quote="service was very slow",
            review_id="r1",
            issue="service_speed",
            polarity="negative",
            intensity=3,
        ),
        VoCItem(
            quote="the staff were rude",
            review_id="r2",
            issue="staff_attitude",
            polarity="negative",
            intensity=2,
        ),
    ]
    scores = voc_mod.issue_scores(items)
    assert scores["service_speed"] == pytest.approx(-1.0)  # -1 * 3/3
    assert scores["staff_attitude"] == pytest.approx(-2 / 3)  # -1 * 2/3
    prev = voc_mod.issue_prevalence(items)
    assert prev["service_speed"] == pytest.approx(0.5)
    assert prev["staff_attitude"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Confidence formula
# --------------------------------------------------------------------------- #


def test_confidence_formula_known_values() -> None:
    # groundedness 1.0, evidence >= target (5), grade A -> 1.0.
    assert conf.compute_confidence(1.0, 5, "A") == pytest.approx(1.0)
    # groundedness 1.0, evidence 1/5=0.2, grade A -> 0.2.
    assert conf.compute_confidence(1.0, 1, "A") == pytest.approx(0.2)
    # groundedness 0.5, evidence 2/5=0.4, grade B(0.7) -> 0.14.
    assert conf.compute_confidence(0.5, 2, "B") == pytest.approx(0.14)
    # grade C weight 0.4.
    assert conf.compute_confidence(1.0, 5, "C") == pytest.approx(0.4)


def test_confidence_clamps_and_labels() -> None:
    assert conf.compute_confidence(2.0, 100, "A") == pytest.approx(1.0)  # clamp high
    assert conf.compute_confidence(-1.0, 5, "A") == 0.0  # clamp low
    assert conf.confidence_label(0.9) == "High"
    assert conf.confidence_label(0.4) == "Medium"
    assert conf.confidence_label(0.1) == "Low"


def test_confidence_formula_text_available() -> None:
    assert "groundedness_rate" in agents.CONFIDENCE_FORMULA


# --------------------------------------------------------------------------- #
# Orchestrator concurrency + brief assembly
# --------------------------------------------------------------------------- #


def test_orchestrator_runs_agents_1_and_2_concurrently() -> None:
    # Each agent fn sleeps; if they ran sequentially total >= 0.2s, concurrently ~0.1s.
    events = []
    lock = threading.Lock()

    def quant_fn(**kwargs):
        with lock:
            events.append(("quant_start", time.time()))
        time.sleep(0.1)
        with lock:
            events.append(("quant_end", time.time()))
        return _fake(
            json.dumps(
                {
                    "drivers": [
                        {
                            "driver_key": "velocity_ratio",
                            "direction": "down",
                            "explanation_template": "velocity {velocity_ratio}",
                        }
                    ]
                }
            )
        )

    def voc_fn(**kwargs):
        with lock:
            events.append(("voc_start", time.time()))
        time.sleep(0.1)
        with lock:
            events.append(("voc_end", time.time()))
        return _fake(
            json.dumps(
                {
                    "items": [
                        {
                            "quote": "service was very slow",
                            "review_id": "r1",
                            "issue": "service_speed",
                            "polarity": "negative",
                            "intensity": 3,
                        }
                    ]
                }
            )
        )

    def auditor_fn(**kwargs):
        return _fake(
            json.dumps(
                {
                    "claims": [
                        {"claim_id": "driver_0", "verdict": "supported", "reason": "ok"},
                        {"claim_id": "issue_service_speed", "verdict": "supported", "reason": "ok"},
                    ],
                    "alternatives": [],
                    "confidence_grade": "A",
                }
            )
        )

    fs = {"business_id": "b0", "risk_tier": "High", "velocity_ratio": "0.70"}
    brief = agents.build_brief(
        fs,
        _snips(),
        {"review_volume": "low"},
        quant_fn=quant_fn,
        voc_fn=voc_fn,
        auditor_fn=auditor_fn,
    )

    # Concurrency is proven structurally: both agents START before either agent ENDS.
    # Under sequential execution the first agent would end before the second begins,
    # so this interval overlap is impossible without true concurrency. This is
    # machine-speed independent (no wall-clock threshold).
    quant_start = next(t for n, t in events if n == "quant_start")
    quant_end = next(t for n, t in events if n == "quant_end")
    voc_start = next(t for n, t in events if n == "voc_start")
    voc_end = next(t for n, t in events if n == "voc_end")
    latest_start = max(quant_start, voc_start)
    earliest_end = min(quant_end, voc_end)
    assert latest_start < earliest_end  # the two sleep windows overlap

    # Brief assembled correctly.
    assert brief.tier == "High"
    assert brief.drivers[0]["text"] == "velocity 0.70"  # Python-rendered number
    assert brief.n_evidence == 1
    assert brief.confidence == pytest.approx(0.2)  # 1.0 * 1/5 * A
    assert any("staffing" in a or "ticket" in a for a in brief.actions)


def test_orchestrator_removes_unsupported_driver_claims() -> None:
    def quant_fn(**kwargs):
        return _fake(
            json.dumps(
                {
                    "drivers": [
                        {
                            "driver_key": "velocity_ratio",
                            "direction": "down",
                            "explanation_template": "velocity {velocity_ratio}",
                        }
                    ]
                }
            )
        )

    def voc_fn(**kwargs):
        return _fake(json.dumps({"items": []}))

    def auditor_fn(**kwargs):
        return _fake(
            json.dumps(
                {
                    "claims": [
                        {"claim_id": "driver_0", "verdict": "unsupported", "reason": "no evidence"}
                    ],
                    "alternatives": [],
                    "confidence_grade": "C",
                }
            )
        )

    fs = {"business_id": "b0", "risk_tier": "Elevated", "velocity_ratio": "1.00"}
    brief = agents.build_brief(
        fs, _snips(), {}, quant_fn=quant_fn, voc_fn=voc_fn, auditor_fn=auditor_fn
    )
    assert brief.drivers == []  # unsupported driver removed
    assert "driver_0" in brief.rejected_claims


# --------------------------------------------------------------------------- #
# Auditor groundedness
# --------------------------------------------------------------------------- #


def test_auditor_groundedness_rate() -> None:
    from lrr.agents.schemas import AuditOutput

    audit = AuditOutput(
        claims=[
            {"claim_id": "a", "verdict": "supported", "reason": "x"},
            {"claim_id": "b", "verdict": "unsupported", "reason": "y"},
            {"claim_id": "c", "verdict": "supported", "reason": "z"},
        ],
        alternatives=[],
        confidence_grade="B",
    )
    assert auditor_mod.groundedness_rate(audit) == pytest.approx(2 / 3)
    assert auditor_mod.supported_claim_ids(audit) == {"a", "c"}
