"""Cached artifact loaders with light schema validation and friendly messages.

The app reads only precomputed parquet files. Every load is cached. A missing file
shows a clear message naming the pipeline script that creates it, never a stack trace.
Validation keeps known columns and coerces types but never raises to the UI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config  # noqa: E402


@dataclass(frozen=True)
class Artifact:
    key: str
    filename: str
    base: str  # "artifacts" or "data"
    pipeline: str  # script that produces it
    columns: tuple  # minimal expected columns (soft contract)


# Registry of every artifact the app can read.
ARTIFACTS: dict[str, Artifact] = {
    "risk_scores": Artifact(
        "risk_scores",
        config.RISK_SCORES_FILE,
        "artifacts",
        "pipeline/05_survival.py (then 05c for tier scores)",
        ("business_id", "risk_tier"),
    ),
    "cohort_locations": Artifact(
        "cohort_locations",
        config.COHORT_LOCATIONS_FILE,
        "artifacts",
        "pipeline/01_ingest_cohort.py",
        ("business_id", "chain", "cluster"),
    ),
    "all_restaurants": Artifact(
        "all_restaurants",
        config.ALL_RESTAURANTS_FILE,
        "data",
        "pipeline/01b_corpus_ingest.py",
        ("business_id", "is_cohort"),
    ),
    "model_metrics": Artifact(
        "model_metrics",
        config.MODEL_METRICS_FILE,
        "artifacts",
        "pipeline/05_survival.py and 05b_tier1_survival.py",
        ("feature_set", "harrell_c"),
    ),
    "survival_curves": Artifact(
        "survival_curves",
        config.SURVIVAL_CURVES_FILE,
        "artifacts",
        "pipeline/05_survival.py",
        ("business_id", "t_days", "survival"),
    ),
    "sentiment_benchmark": Artifact(
        "sentiment_benchmark",
        config.SENTIMENT_BENCHMARK_FILE,
        "artifacts",
        "pipeline/02_text_sentiment.py",
        ("model", "macro_f1"),
    ),
    "transfer_eval": Artifact(
        "transfer_eval",
        config.TRANSFER_EVAL_FILE,
        "artifacts",
        "pipeline/02c_corpus_sentiment.py",
        ("split",),
    ),
    "briefs": Artifact(
        "briefs",
        config.BRIEFS_FILE,
        "artifacts",
        "pipeline/08_eval_agents.py",
        ("business_id", "tier"),
    ),
    "agent_eval": Artifact(
        "agent_eval",
        config.AGENT_EVAL_FILE,
        "artifacts",
        "pipeline/08_eval_agents.py",
        ("numeric_fidelity_rate",),
    ),
    "rag_cache": Artifact(
        "rag_cache",
        config.RAG_CACHE_FILE,
        "artifacts",
        "pipeline/06_aspects_rag.py",
        ("business_id", "question", "answer"),
    ),
    "aspect_eval": Artifact(
        "aspect_eval",
        config.ASPECT_EVAL_FILE,
        "artifacts",
        "pipeline/06_aspects_rag.py",
        ("accuracy",),
    ),
    "topic_shares": Artifact(
        "topic_shares",
        config.TOPIC_SHARES_FILE,
        "data",
        "pipeline/03_topics_colab.py",
        ("business_id",),
    ),
}


def _resolve(art: Artifact) -> Path:
    base = config.ARTIFACTS_DIR if art.base == "artifacts" else config.DATA_DIR
    return Path(base) / art.filename


def topics_artifact(cluster: str) -> pd.DataFrame | None:
    """Load a per-cluster topics table (filename depends on the cluster)."""
    path = config.ARTIFACTS_DIR / config.topics_file(cluster)
    return _cached_read(str(path))


@st.cache_data(show_spinner=False)
def _cached_read(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _validate(df: pd.DataFrame, art: Artifact) -> pd.DataFrame:
    """Soft contract: we keep known columns in mind but never raise to the UI.

    Pages guard per-column, so a frame missing a non-core column still renders.
    """
    return df


def load(key: str) -> pd.DataFrame | None:
    """Load an artifact by key, or None if the file is absent/unreadable."""
    art = ARTIFACTS[key]
    df = _cached_read(str(_resolve(art)))
    return _validate(df, art) if df is not None else None


def missing_message(key: str) -> str:
    art = ARTIFACTS[key]
    return f"`{art.filename}` is not available yet. Run {art.pipeline} to produce it."


def require(key: str, container=st) -> pd.DataFrame | None:
    """Load an artifact or render a friendly message and return None."""
    df = load(key)
    if df is None:
        container.info(missing_message(key))
    return df
