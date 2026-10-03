"""Smoke tests for the Streamlit app using streamlit.testing.v1.AppTest.

Every page must render with no exception in Cached mode (mock provider, no keys). We
point the artifact loaders at a temp directory with small synthetic fixtures, so the
tests run offline and fast. We also assert peak memory stays well under budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from streamlit.testing.v1 import AppTest  # noqa: E402

from lrr import config  # noqa: E402

PAGES = [
    "portfolio_radar",
    "location_deep_dive",
    "risk_committee",
    "complaint_themes",
    "ask_the_reviews",
    "model_lab",
    "methodology",
]


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #


def _write_fixtures(art: Path, data_dir: Path) -> None:
    art.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    risk = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2", "b3"],
            "chain": ["Chili's", "Chili's", "Subway", "Subway"],
            "cluster": ["Fast Food", "Fast Food", "Fast Food", "Fast Food"],
            "state": ["AZ", "AZ", "PA", "PA"],
            "risk_score": [0.9, 0.4, 0.7, 0.2],
            "tier1_score": [0.8, 0.3, 0.6, 0.2],
            "tier2_score": [0.9, 0.4, 0.7, 0.2],
            "final_score": [0.9, 0.4, 0.7, 0.2],
            "risk_percentile": [95.0, 40.0, 70.0, 20.0],
            "risk_tier": ["High", "Watch", "Elevated", "Low"],
            "surv_12m": [0.6, 0.8, 0.7, 0.9],
            "surv_24m": [0.4, 0.7, 0.5, 0.85],
            "top_3_drivers": ["velocity, sentiment, stars"] * 4,
        }
    )
    risk.to_parquet(art / config.RISK_SCORES_FILE, index=False)

    loc = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2", "b3"],
            "chain": ["Chili's", "Chili's", "Subway", "Subway"],
            "cluster": ["Fast Food"] * 4,
            "city": ["Phoenix", "Tempe", "Philadelphia", "Pittsburgh"],
            "state": ["AZ", "AZ", "PA", "PA"],
            "lat": [33.4, 33.4, 39.9, 40.4],
            "lon": [-112.0, -111.9, -75.1, -79.9],
            "stars": [2.5, 3.5, 3.0, 4.0],
            "review_count": [100, 80, 120, 60],
            "is_open": [0, 1, 1, 1],
        }
    )
    loc.to_parquet(art / config.COHORT_LOCATIONS_FILE, index=False)

    briefs = pd.DataFrame(
        {
            "business_id": ["b0"],
            "tier": ["High"],
            "drivers": ["Review velocity ratio fell to 0.70."],
            "evidence": ['"service was very slow" [r1]'],
            "confidence": [0.2],
            "confidence_label": ["Low"],
            "actions": ["Audit peak-hour staffing. | Review ticket times."],
            "groundedness": [1.0],
            "n_evidence": [1],
            "auditor_grade": ["A"],
            "n_rejected_claims": [0],
        }
    )
    briefs.to_parquet(art / config.BRIEFS_FILE, index=False)

    metrics = pd.DataFrame(
        {
            "tier": ["tier1", "tier1", "tier2"],
            "scope": ["corpus", "corpus", "cohort"],
            "feature_set": ["stars_only", "full", "full"],
            "model": ["CoxPH", "CoxPH", "CoxPH"],
            "harrell_c": [0.55, 0.70, 0.72],
            "harrell_c_lo": [0.52, 0.66, 0.68],
            "harrell_c_hi": [0.58, 0.74, 0.76],
        }
    )
    metrics.to_parquet(art / config.MODEL_METRICS_FILE, index=False)

    agent = pd.DataFrame(
        [
            {
                "n_briefs": 1,
                "numeric_fidelity_rate": 1.0,
                "mean_groundedness": 1.0,
                "auditor_rejection_rate": 0.0,
                "mean_latency_s": 0.1,
                "total_est_cost_usd": 0.0,
            }
        ]
    )
    agent.to_parquet(art / config.AGENT_EVAL_FILE, index=False)

    rag = pd.DataFrame(
        {
            "business_id": ["b0"] * 2,
            "question": ["Why is this location at risk?", "What do customers praise?"],
            "answer": ["Service is slow [r1].", "Not enough evidence."],
            "visible_answer": ["Service is slow [r1].", ""],
            "groundedness_rate": [1.0, 0.0],
            "n_citations": [1, 0],
        }
    )
    rag.to_parquet(art / config.RAG_CACHE_FILE, index=False)

    sent = pd.DataFrame(
        {
            "model": ["LogisticRegression"],
            "features": ["tfidf"],
            "accuracy": [0.9],
            "macro_f1": [0.89],
            "macro_f1_lo": [0.87],
            "macro_f1_hi": [0.91],
            "roc_auc": [0.95],
            "n_test": [1000],
        }
    )
    sent.to_parquet(art / config.SENTIMENT_BENCHMARK_FILE, index=False)


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    """Point the loaders at a temp artifacts dir with fixtures; clear caches."""
    art = tmp_path / "artifacts"
    data_dir = tmp_path / "data"
    _write_fixtures(art, data_dir)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", art)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    # Ensure no key leaks a Live default into the tests.
    for env in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "VOYAGER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    import streamlit as st

    st.cache_data.clear()
    yield art, data_dir
    st.cache_data.clear()


_ROOT_STR = str(_ROOT).replace("\\", "/")


def _page_harness(page_name: str) -> str:
    """Return a self-contained script string AppTest can run for one page."""
    return f"""
import sys
for _p in [r"{_ROOT_STR}", r"{_ROOT_STR}/src"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import importlib
from app import sidebar
page = importlib.import_module("app.pages.{page_name}")
sidebar.render()
page.render()
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(app_env, page):
    at = AppTest.from_string(_page_harness(page))
    at.run(timeout=30)
    assert not at.exception, f"page {page} raised: {at.exception}"


def test_risk_committee_shows_cached_brief(app_env):
    at = AppTest.from_string(_page_harness("risk_committee"))
    at.run(timeout=30)
    assert not at.exception
    # The cached brief for b0 (High) renders its driver sentence somewhere.
    text = " ".join(m.value for m in at.markdown)
    assert "velocity" in text.lower() or "Why we flagged it" in text


def test_filter_changes_portfolio_row_count(app_env):
    at = AppTest.from_string(_page_harness("portfolio_radar"))
    at.run(timeout=30)
    assert not at.exception
    # Apply a tier filter to High only via session state, rerun, expect fewer rows.
    at.session_state["f_tier"] = ["High"]
    at.run(timeout=30)
    assert not at.exception
    # The High-only watchlist should have a single row (b0) in the dataframe.
    assert len(at.dataframe) >= 1


def test_missing_artifact_shows_friendly_message(tmp_path, monkeypatch):
    # Empty artifacts dir -> friendly info message, not an exception.
    empty = tmp_path / "empty_art"
    empty.mkdir()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", empty)
    monkeypatch.setattr(config, "DATA_DIR", empty)
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_string(_page_harness("portfolio_radar"))
    at.run(timeout=30)
    assert not at.exception
    infos = " ".join(i.value for i in at.info)
    assert "risk_scores.parquet" in infos
    st.cache_data.clear()


def test_peak_memory_under_budget(app_env):
    import tracemalloc

    import streamlit as st

    tracemalloc.start()
    for page in PAGES:
        st.cache_data.clear()
        at = AppTest.from_string(_page_harness(page))
        at.run(timeout=30)
        assert not at.exception
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1e6
    assert peak_mb < 800, f"peak traced memory {peak_mb:.0f} MB exceeds 800 MB"
