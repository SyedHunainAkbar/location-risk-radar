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


@st.cache_resource(show_spinner=False)
def _query_encoder():
    """Load the live-query embedder once and keep it as a shared Streamlit resource.

    Returns a callable ``encode(list[str]) -> np.ndarray`` backed by the CPU-only
    ONNX MiniLM in :func:`lrr.rag._encode`, or ``None`` if the backend is
    unavailable. Caching with ``st.cache_resource`` means the model loads at most
    once per app process, so repeat live queries do not pay the cold-start cost.
    """
    try:
        from lrr import rag

        # Warm the backend once so the first user query is fast.
        rag._encode(["warmup"])
        return rag._encode
    except Exception:
        return None


def encode_query(texts):
    """Embed query texts via the cached encoder, or None if unavailable."""
    encoder = _query_encoder()
    if encoder is None:
        return None
    try:
        return encoder(list(texts))
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


def _short_id(bid: str) -> str:
    return str(bid)[:6]


@st.cache_data(show_spinner=False)
def _enriched(risk_path: str, loc_path: str) -> pd.DataFrame | None:
    risk = _cached_read(risk_path)
    loc = _cached_read(loc_path)
    if loc is None:
        return risk
    base = loc.copy()
    if risk is not None:
        keep = [c for c in risk.columns if c not in ("chain", "cluster")]
        base = base.merge(risk[keep], on="business_id", how="left")
    else:
        base["risk_tier"] = None
    base["eligibility"] = base["risk_tier"].notna().map(
        {True: "Scored (active at landmark)", False: "Inactive before landmark"}
    )
    base["risk_tier"] = base["risk_tier"].fillna("Not scored")
    base["observed_status"] = pd.to_numeric(base.get("is_open"), errors="coerce").map(
        {1: "Open", 0: "Closed"}
    )
    # Several branches of one chain can share a city, so a short id suffix keeps
    # every label unique while staying readable.
    base["location_label"] = [
        f"{r.chain} \u00b7 {r.city}, {r.state} ({r.cluster}) #{_short_id(r.business_id)}"
        for r in base.itertuples()
    ]
    return base


def locations(scored_only: bool = False) -> pd.DataFrame | None:
    """All 2,054 cohort locations with city, state, cluster and risk joined.

    Unscored rows (inactive in the 6 months before the landmark) carry
    risk_tier == "Not scored". Use scored_only=True for model-based pages.
    """
    df = _enriched(
        str(_resolve(ARTIFACTS["risk_scores"])), str(_resolve(ARTIFACTS["cohort_locations"]))
    )
    if df is None:
        return None
    if scored_only and "eligibility" in df:
        df = df[df["eligibility"].str.startswith("Scored")]
    return df


def label_map(df: pd.DataFrame | None) -> dict:
    if df is None or "location_label" not in df:
        return {}
    return dict(zip(df["business_id"], df["location_label"]))
