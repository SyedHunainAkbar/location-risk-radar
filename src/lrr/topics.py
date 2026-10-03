"""Complaint topic modeling helpers (stage 03).

This module holds the lightweight, testable pieces of the topic layer: the
operational seed topics, a disk embedding cache, BERTopic/UMAP/HDBSCAN/vectorizer
builders, coherence and diversity metrics, LLM-based topic labeling via
:mod:`lrr.llm`, human label overrides, and the per-business pre-landmark topic
shares used by the survival model.

Heavy libraries (bertopic, umap-learn, hdbscan, gensim, sentence-transformers) are
imported lazily inside the functions that need them so this module imports cleanly
in a plain environment and the unit tests run without them. The offline Colab
driver (``pipeline/03_topics_colab.py``) is where those libraries actually run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lrr import config, llm

# --------------------------------------------------------------------------- #
# Seed topics (zero-shot)
# --------------------------------------------------------------------------- #

#: Keyword lists per operational complaint theme, derived from the config phrases.
#: BERTopic's zero-shot mode accepts the phrases directly; the keyword lists are
#: also handy for seeding a guided vectorizer or for coherence sanity checks.
SEED_TOPICS: list[list[str]] = [
    ["slow", "service", "wait", "long", "waiting", "minutes", "forever"],
    ["rude", "staff", "inattentive", "unfriendly", "ignored", "attitude"],
    ["wrong", "missing", "order", "forgot", "incorrect", "incomplete"],
    ["cold", "quality", "food", "stale", "bland", "undercooked", "soggy"],
    ["dirty", "clean", "cleanliness", "hygiene", "unsanitary", "bathroom"],
    ["price", "value", "expensive", "overpriced", "worth", "money"],
    ["manager", "management", "complaint", "refund", "handled", "resolve"],
    ["drive", "thru", "drive-thru", "takeout", "pickup", "window"],
    ["reservation", "seating", "seated", "table", "booking", "host"],
]

#: Flat phrase list for BERTopic's ``zeroshot_topic_list``.
SEED_TOPIC_PHRASES: list[str] = list(config.SEED_TOPIC_PHRASES)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #


def corpus_hash(texts: Sequence[str]) -> str:
    """Return a stable short hash of a corpus (order-sensitive)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embedding_cache_path(cluster: str, texts: Sequence[str], cache_dir: Path) -> Path:
    """Path for the cached embedding of a cluster's corpus."""
    slug = cluster.lower().replace(" ", "_").replace("-", "_")
    return Path(cache_dir) / f"topic_emb_{slug}_{corpus_hash(texts)}.npy"


def embed_corpus(
    texts: Sequence[str],
    cluster: str,
    cache_dir: Path = config.EMBEDDING_CACHE_DIR,
    model_name: str = config.EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed a corpus with MiniLM, caching the result to disk.

    Reuses the cached ``.npy`` when the corpus is unchanged (same hash). The
    encoder is imported lazily so this module imports without sentence-transformers.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = embedding_cache_path(cluster, texts, cache_dir)
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer  # lazy

    encoder = SentenceTransformer(model_name)
    emb = encoder.encode(list(texts), show_progress_bar=False, convert_to_numpy=True).astype(
        "float32"
    )
    np.save(path, emb)
    return emb


# --------------------------------------------------------------------------- #
# BERTopic component builders (lazy imports)
# --------------------------------------------------------------------------- #


def build_umap(seed: int = config.SEED):
    """Build the UMAP reducer with the locked configuration."""
    from umap import UMAP  # lazy

    return UMAP(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        metric=config.UMAP_METRIC,
        random_state=seed,
    )


def build_hdbscan(min_cluster_size: int):
    """Build the HDBSCAN clusterer for a given minimum cluster size."""
    from hdbscan import HDBSCAN  # lazy

    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def build_vectorizer():
    """CountVectorizer with English stop words and bigrams for c-TF-IDF."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(
        stop_words="english",
        ngram_range=config.TOPIC_NGRAM_RANGE,
    )


def build_bertopic(
    min_cluster_size: int,
    embedding_model: str | None = config.EMBEDDING_MODEL,
    zero_shot: bool = False,
    seed: int = config.SEED,
):
    """Construct a BERTopic model with the locked UMAP/HDBSCAN/vectorizer config.

    When ``zero_shot`` is True, the seed phrases are passed as
    ``zeroshot_topic_list`` so BERTopic anchors topics to the operational themes.
    BERTopic is imported lazily.
    """
    from bertopic import BERTopic  # lazy

    kwargs = dict(
        embedding_model=embedding_model,
        umap_model=build_umap(seed),
        hdbscan_model=build_hdbscan(min_cluster_size),
        vectorizer_model=build_vectorizer(),
        calculate_probabilities=False,
        verbose=False,
    )
    if zero_shot:
        kwargs["zeroshot_topic_list"] = SEED_TOPIC_PHRASES
        kwargs["zeroshot_min_similarity"] = 0.85
    return BERTopic(**kwargs)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def topic_diversity(topic_words: Sequence[Sequence[str]], top_k: int = config.TOPIC_TOP_K) -> float:
    """Topic diversity: fraction of unique words across all topics' top-k words.

    1.0 means every top word is unique across topics; low values mean topics
    repeat the same words. Outlier/empty topics should be excluded by the caller.
    """
    all_words: list[str] = []
    for words in topic_words:
        all_words.extend(list(words)[:top_k])
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def topic_coherence(
    topic_words: Sequence[Sequence[str]],
    texts: Sequence[str],
    top_k: int = config.TOPIC_TOP_K,
) -> float:
    """Mean c_v coherence over topics via gensim (imported lazily)."""
    from gensim.corpora import Dictionary  # lazy
    from gensim.models.coherencemodel import CoherenceModel

    tokenized = [str(t).lower().split() for t in texts]
    dictionary = Dictionary(tokenized)
    topics = [list(words)[:top_k] for words in topic_words if len(words) > 0]
    if not topics:
        return float("nan")
    cm = CoherenceModel(
        topics=topics,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(cm.get_coherence())


# --------------------------------------------------------------------------- #
# LLM labeling (labels only; Python assembles the table)
# --------------------------------------------------------------------------- #

_LABEL_SYSTEM = (
    "You name customer-complaint topics for restaurant reviews. "
    "Given the top words and a few example reviews, reply with a short, specific "
    "topic label of at most four words. Reply with the label only: no numbers, no "
    "scores, no punctuation beyond spaces, no explanation."
)


def _label_prompt(top_words: Sequence[str], docs: Sequence[str]) -> list[dict]:
    words = ", ".join(list(top_words)[: config.TOPIC_TOP_K])
    examples = "\n".join(
        f"- {str(d).strip()[:300]}" for d in list(docs)[: config.TOPIC_LABEL_N_DOCS]
    )
    user = f"Top words: {words}\n\nExample reviews:\n{examples}\n\nTopic label:"
    return [
        {"role": "system", "content": _LABEL_SYSTEM},
        {"role": "user", "content": user},
    ]


def _clean_label(raw: str) -> str:
    """Keep the first line, strip quotes/bullets, cap length to four words."""
    text = str(raw).strip().splitlines()[0] if raw else ""
    text = text.strip().strip('"').strip("'").lstrip("-").strip()
    words = text.split()
    return " ".join(words[:4]) if words else "Unlabeled"


def label_topics_llm(
    topics_df: pd.DataFrame,
    provider: str | None = None,
    top_words_col: str = "top_words",
    docs_col: str = "rep_docs",
) -> pd.DataFrame:
    """Attach an LLM-generated ``label`` to each topic row.

    Expects ``topics_df`` with a ``topic`` id, a list of top words, and a list of
    representative docs. Calls :func:`lrr.llm.chat` at temperature 0 for each
    non-outlier topic (topic id ``-1`` is the outlier topic and is labeled
    "Outliers"). The model returns a label string only.
    """
    labels: list[str] = []
    for _, row in topics_df.iterrows():
        if int(row["topic"]) == -1:
            labels.append("Outliers")
            continue
        messages = _label_prompt(row[top_words_col], row.get(docs_col, []) or [])
        raw = llm.chat(messages, temperature=0.0, max_tokens=16, provider=provider)
        labels.append(_clean_label(raw))
    out = topics_df.copy()
    out["label"] = labels
    return out


def apply_label_overrides(labels_df: pd.DataFrame, overrides_csv: Path) -> pd.DataFrame:
    """Override LLM labels with any human-provided labels in a CSV.

    The CSV must have ``topic`` and ``label`` columns. A non-empty override wins.
    Missing file is a no-op.
    """
    out = labels_df.copy()
    path = Path(overrides_csv)
    if not path.exists():
        return out
    overrides = pd.read_csv(path)
    if not {"topic", "label"}.issubset(overrides.columns):
        return out
    mapping = {
        int(t): str(lbl).strip()
        for t, lbl in zip(overrides["topic"], overrides["label"])
        if str(lbl).strip()
    }
    out["label"] = [mapping.get(int(t), lbl) for t, lbl in zip(out["topic"], out["label"])]
    return out


# --------------------------------------------------------------------------- #
# Pre-landmark topic shares (survival features)
# --------------------------------------------------------------------------- #


def _to_timestamp(value) -> pd.Timestamp:
    if isinstance(value, (date, datetime)):
        return pd.Timestamp(value)
    return pd.Timestamp(value)


def compute_topic_shares(
    assignments: pd.DataFrame,
    landmark,
    window_days: int | None = config.TOPIC_SHARE_WINDOW_DAYS,
    business_col: str = "business_id",
    date_col: str = "date",
    topic_col: str = "topic",
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """Per-business topic shares over the pre-landmark window.

    Keeps reviews strictly before ``landmark`` (and, if ``window_days`` is set, no
    earlier than ``window_days`` before it), then computes each business's share of
    reviews per topic. Shares for a business sum to 1 when it has any reviews in the
    window, and 0 across all topics otherwise. Returns a wide frame: one row per
    business, one ``topic_{id}`` column per topic.

    This is leakage-safe by construction: no on-or-after-landmark review enters.
    """
    landmark_ts = _to_timestamp(landmark)
    df = assignments.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Strictly before the landmark.
    df = df[df[date_col] < landmark_ts]
    if window_days is not None:
        start = landmark_ts - pd.Timedelta(days=window_days)
        df = df[df[date_col] >= start]
    if drop_outliers:
        df = df[df[topic_col] != -1]

    businesses = sorted(assignments[business_col].astype(str).unique())
    topics = sorted(t for t in assignments[topic_col].unique() if not (drop_outliers and t == -1))
    topic_cols = [f"topic_{int(t)}" for t in topics]

    if df.empty:
        wide = pd.DataFrame(0.0, index=businesses, columns=topic_cols)
        wide.index.name = business_col
        return wide.reset_index()

    counts = (
        df.assign(_b=df[business_col].astype(str))
        .groupby(["_b", topic_col])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(index=businesses, columns=topics, fill_value=0)
    totals = counts.sum(axis=1)
    shares = counts.div(totals.where(totals > 0, other=1), axis=0)
    shares[totals == 0] = 0.0
    shares.columns = topic_cols
    shares.index.name = business_col
    return shares.reset_index()
