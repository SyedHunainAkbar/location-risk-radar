"""Retrieval-augmented evidence assistant (stage 06).

We build a compact per-location index (recent reviews + tips, chunked to <= 120
words, MiniLM float16 embeddings + parquet metadata, under 50 MB), retrieve with
cosine (optional MMR and filters), answer strictly from retrieved snippets with
``[review_id]`` citations, and judge each sentence with a different-family model to
compute a groundedness rate. Answers are precomputed and cached so the app runs with
no API key.

sentence-transformers is imported lazily; embeddings can be injected in tests.
All LLM calls go through :mod:`lrr.llm`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from lrr import config, io, llm

META_COLUMNS = ["chunk_id", "business_id", "chain", "review_id", "date", "stars", "text"]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def chunk_text(text: str, max_words: int = config.RAG_CHUNK_MAX_WORDS) -> list[str]:
    """Split text into chunks of at most ``max_words`` words, preserving order."""
    words = str(text).split()
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# --------------------------------------------------------------------------- #
# Index build / load
# --------------------------------------------------------------------------- #


def _encode(texts: Sequence[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Embed texts with MiniLM, returning float32 (cast to f16 on save).

    We try a CPU-only ONNX backend (``fastembed``) first so the deployed app can
    embed live queries without pulling ``torch`` into a 1 GB environment. If
    ``fastembed`` is unavailable we fall back to ``sentence-transformers``, which
    the offline pipeline already uses to build the index. Both target the same
    ``all-MiniLM-L6-v2`` model, so query vectors stay compatible with the
    precomputed index. The import is lazy and ``encode_fn`` can be injected in
    tests to avoid loading any model.
    """
    items = list(texts)

    # Preferred: ONNX MiniLM via fastembed (no torch, ~150 MB RAM). The encoder is
    # cached on the module so repeated live queries do not reload the model.
    try:
        return _fastembed_encode(items)
    except Exception:
        pass

    # Fallback: sentence-transformers (torch); used offline and in the pipeline.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    return encoder.encode(items, show_progress_bar=False, convert_to_numpy=True)


_ONNX_ENCODER = None


def _fastembed_encode(items: list[str]) -> np.ndarray:
    """Embed with a cached fastembed ONNX MiniLM encoder (384-dim float32)."""
    global _ONNX_ENCODER
    if _ONNX_ENCODER is None:
        # Disable the HuggingFace symlink cache before import. On Windows hosts
        # without symlink privilege the symlinked cache corrupts the ONNX model
        # and crashes onnxruntime; plain-copy caching avoids that and is a no-op
        # on Linux (Streamlit Cloud).
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        _ONNX_ENCODER = TextEmbedding(model_name=config.EMBEDDING_MODEL_ONNX)
    return np.asarray(list(_ONNX_ENCODER.embed(items)), dtype="float32")


def build_index(
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    locations: pd.DataFrame,
    out_dir: Path = config.RAG_INDEX_DIR,
    recent_reviews: int = config.RAG_RECENT_REVIEWS,
    encode_fn=None,
) -> dict:
    """Build the compact per-location RAG index.

    Keeps the ``recent_reviews`` most recent reviews plus all tips per location,
    chunks to <= 120 words, embeds, and writes float16 embeddings + parquet metadata.
    Asserts the total on-disk size is under 50 MB.

    ``encode_fn`` can be injected in tests to avoid loading the model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loc = locations[["business_id", "chain"]].drop_duplicates()

    rows: list[dict] = []
    rev = reviews.copy()
    rev["date"] = pd.to_datetime(rev.get("date"), errors="coerce")
    for bid, grp in rev.groupby("business_id"):
        recent = grp.sort_values("date", ascending=False).head(recent_reviews)
        for _, r in recent.iterrows():
            for ci, chunk in enumerate(chunk_text(r.get("text", ""))):
                rows.append(
                    {
                        "business_id": bid,
                        "review_id": f"{r.get('review_id')}#{ci}",
                        "date": r["date"],
                        "stars": r.get("stars"),
                        "text": chunk,
                    }
                )
    if tips is not None and not tips.empty:
        tp = tips.copy()
        tp["date"] = pd.to_datetime(tp.get("date"), errors="coerce")
        for i, t in tp.iterrows():
            for ci, chunk in enumerate(chunk_text(t.get("text", ""))):
                rows.append(
                    {
                        "business_id": t.get("business_id"),
                        "review_id": f"tip_{i}#{ci}",
                        "date": t["date"],
                        "stars": np.nan,
                        "text": chunk,
                    }
                )

    meta = pd.DataFrame(rows)
    if meta.empty:
        meta = pd.DataFrame(columns=META_COLUMNS)
        emb = np.zeros((0, 384), dtype="float16")
    else:
        meta = meta.merge(loc, on="business_id", how="left")
        meta.insert(0, "chunk_id", np.arange(len(meta)))
        meta = meta[META_COLUMNS]
        enc = encode_fn or _encode
        emb = np.asarray(enc(meta["text"].tolist())).astype("float16")

    emb_path = out_dir / config.RAG_EMBEDDINGS_FILE
    meta_path = out_dir / config.RAG_METADATA_FILE
    np.save(emb_path, emb)
    io.write_parquet(meta, meta_path)

    total_bytes = emb_path.stat().st_size + meta_path.stat().st_size
    assert total_bytes < config.MAX_ARTIFACT_BYTES, (
        f"RAG index is {total_bytes / 1e6:.1f} MB, exceeds the "
        f"{config.MAX_ARTIFACT_BYTES / 1e6:.0f} MB cap."
    )
    return {
        "n_chunks": len(meta),
        "bytes": total_bytes,
        "embeddings_path": emb_path,
        "metadata_path": meta_path,
    }


def load_index(index_dir: Path = config.RAG_INDEX_DIR) -> tuple[np.ndarray, pd.DataFrame]:
    """Load (embeddings float32, metadata) from an index directory."""
    index_dir = Path(index_dir)
    emb = np.load(index_dir / config.RAG_EMBEDDINGS_FILE).astype("float32")
    meta = io.read_parquet(index_dir / config.RAG_METADATA_FILE)
    return emb, meta


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def cosine_top_k(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k rows by cosine similarity to the query."""
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    q = _normalize(np.asarray(query_vec, dtype=float).reshape(1, -1))
    e = _normalize(np.asarray(embeddings, dtype=float))
    sims = (e @ q.T).ravel()
    k = min(k, len(sims))
    return np.argsort(-sims)[:k]


def mmr(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    candidate_idx: np.ndarray,
    k: int,
    lambda_: float = config.RAG_MMR_LAMBDA,
) -> np.ndarray:
    """Maximal Marginal Relevance reranking over candidate indices."""
    if len(candidate_idx) == 0:
        return candidate_idx
    q = _normalize(np.asarray(query_vec, dtype=float).reshape(1, -1))
    e = _normalize(np.asarray(embeddings, dtype=float))
    cand = list(candidate_idx)
    rel = {i: float((e[i] @ q.T).ravel()[0]) for i in cand}
    selected: list[int] = []
    while cand and len(selected) < k:
        if not selected:
            best = max(cand, key=lambda i: rel[i])
        else:

            def score(i: int) -> float:
                div = max(float(e[i] @ e[j].T) for j in selected)
                return lambda_ * rel[i] - (1 - lambda_) * div

            best = max(cand, key=score)
        selected.append(best)
        cand.remove(best)
    return np.array(selected, dtype=int)


def retrieve(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    business_id: str | None = None,
    chain: str | None = None,
    k: int = config.RAG_TOP_K,
    use_mmr: bool = False,
    max_stars: float | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Filter metadata, then cosine top-k (optionally MMR). Returns snippet rows.

    Filters: ``business_id`` or ``chain``, optional ``max_stars`` (<=), optional
    ``date_range`` (inclusive ``(start, end)``). The returned frame carries the
    original ``chunk_id`` so the caller can map back to embeddings.
    """
    mask = pd.Series(True, index=metadata.index)
    if business_id is not None:
        mask &= metadata["business_id"] == business_id
    if chain is not None:
        mask &= metadata["chain"] == chain
    if max_stars is not None:
        mask &= pd.to_numeric(metadata["stars"], errors="coerce") <= max_stars
    if date_range is not None:
        d = pd.to_datetime(metadata["date"], errors="coerce")
        start, end = date_range
        mask &= (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))

    sub = metadata[mask]
    if sub.empty:
        return sub.copy()
    local_emb = embeddings[sub.index.to_numpy()]
    local_top = cosine_top_k(query_vec, local_emb, k if not use_mmr else max(k * 3, k))
    if use_mmr:
        local_top = mmr(query_vec, local_emb, local_top, k)
    return sub.iloc[local_top].copy()


# --------------------------------------------------------------------------- #
# Generation and judging
# --------------------------------------------------------------------------- #


def _format_snippets(snippets: pd.DataFrame) -> str:
    return "\n".join(f"[{row['review_id']}] {row['text']}" for _, row in snippets.iterrows())


def answer_prompt(question: str, snippets: pd.DataFrame) -> list[dict]:
    """Messages for the answerer: answer only from snippets, cite [review_id]."""
    system = (
        "You answer questions about a restaurant location using ONLY the provided "
        "snippets. Cite the snippet id in square brackets, like [abc#0], after every "
        "claim. If the snippets do not support an answer, reply exactly: "
        '"Not enough evidence." Do not use outside knowledge. Do not invent numbers.'
    )
    user = (
        f"Snippets:\n{_format_snippets(snippets)}\n\n"
        f"Question: {question}\n\nAnswer (with [review_id] citations):"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(
    question: str,
    snippets: pd.DataFrame,
    provider: str | None = None,
    model: str = config.RAG_ANSWER_MODEL,
) -> str:
    """Generate a grounded answer. Abstains when there are no snippets."""
    if snippets is None or snippets.empty:
        return "Not enough evidence."
    return llm.chat(
        answer_prompt(question, snippets),
        model=model,
        temperature=0.0,
        max_tokens=512,
        provider=provider,
    )


def split_sentences(answer: str) -> list[str]:
    """Naive sentence split that keeps citations attached."""
    parts = re.split(r"(?<=[.!?])\s+", str(answer).strip())
    return [p.strip() for p in parts if p.strip()]


_CITE_RE = re.compile(r"\[([^\]]+)\]")


def cited_ids(sentence: str) -> list[str]:
    """Return the review ids cited in a sentence."""
    return _CITE_RE.findall(sentence)


def judge_prompt(sentence: str, snippet_text: str) -> list[dict]:
    """Messages for the judge: is the sentence supported by its cited snippet?"""
    system = (
        "You are a strict fact-checker. Decide whether the claim is supported by the "
        'evidence. Reply with one word only: "supported" or "unsupported".'
    )
    user = f"Evidence:\n{snippet_text}\n\nClaim:\n{sentence}\n\nLabel:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_sentences(
    answer: str,
    snippets: pd.DataFrame,
    provider: str | None = None,
    model: str = config.RAG_JUDGE_MODEL,
) -> list[tuple[str, bool]]:
    """Label each answer sentence supported/unsupported against its cited snippet.

    A sentence with no citation is unsupported. The snippet text is the concatenation
    of the cited ids' chunk texts.
    """
    by_id = dict(zip(snippets["review_id"], snippets["text"])) if len(snippets) else {}
    labels: list[tuple[str, bool]] = []
    for sentence in split_sentences(answer):
        ids = cited_ids(sentence)
        if not ids:
            labels.append((sentence, False))
            continue
        evidence = " ".join(by_id.get(i, "") for i in ids).strip()
        if not evidence:
            labels.append((sentence, False))
            continue
        verdict = llm.chat(
            judge_prompt(sentence, evidence),
            model=model,
            temperature=0.0,
            max_tokens=8,
            provider=provider,
        )
        labels.append(
            (
                sentence,
                "unsupported" not in str(verdict).lower() and "supported" in str(verdict).lower(),
            )
        )
    return labels


def groundedness_rate(labels: Sequence[tuple[str, bool]]) -> float:
    """Fraction of sentences judged supported. 1.0 for an empty answer set."""
    if not labels:
        return 1.0
    return sum(1 for _, ok in labels if ok) / len(labels)


def visible_answer(labels: Sequence[tuple[str, bool]]) -> str:
    """Join only the supported sentences for display in the UI."""
    return " ".join(s for s, ok in labels if ok).strip()


# --------------------------------------------------------------------------- #
# Precomputed cache
# --------------------------------------------------------------------------- #


def build_rag_cache(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    high_risk_ids: Sequence[str],
    questions: Sequence[str] = config.RAG_STANDARD_QUESTIONS,
    query_encode_fn=None,
    provider: str | None = None,
) -> pd.DataFrame:
    """Precompute grounded answers for standard questions per high-risk location.

    Returns rows: business_id, question, answer, visible_answer, groundedness_rate,
    n_citations. ``query_encode_fn`` embeds a question string; injected in tests.
    """
    encode = query_encode_fn or (lambda q: _encode([q])[0])
    rows: list[dict] = []
    for bid in high_risk_ids:
        for q in questions:
            qvec = np.asarray(encode(q), dtype=float)
            snippets = retrieve(
                qvec, embeddings, metadata, business_id=bid, k=config.RAG_TOP_K, use_mmr=True
            )
            answer = generate_answer(q, snippets, provider=provider)
            labels = judge_sentences(answer, snippets, provider=provider)
            rows.append(
                {
                    "business_id": bid,
                    "question": q,
                    "answer": answer,
                    "visible_answer": visible_answer(labels),
                    "groundedness_rate": groundedness_rate(labels),
                    "n_citations": int(sum(len(cited_ids(s)) for s, _ in labels)),
                }
            )
    return pd.DataFrame(rows)
