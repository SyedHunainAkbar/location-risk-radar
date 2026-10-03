"""Tests for lrr.text.

The spaCy model is optional in CI, so functional tests skip cleanly when
``en_core_web_sm`` is not installed. The module must still import without spaCy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lrr import text


def test_module_imports_without_spacy() -> None:
    # Importing the module must not require spaCy (lazy imports).
    assert hasattr(text, "normalize_tokens")
    assert hasattr(text, "contrast_lexicons")


def _spacy_available() -> bool:
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    try:
        import en_core_web_sm  # noqa: F401

        return True
    except Exception:
        try:
            import spacy

            spacy.load("en_core_web_sm")
            return True
        except Exception:
            return False


spacy_required = pytest.mark.skipif(not _spacy_available(), reason="en_core_web_sm not installed")


@spacy_required
def test_normalize_tokens_lowercases_and_filters() -> None:
    nlp = text.load_nlp()
    toks = text.normalize_tokens(["The FOOD was Great!!! 123"], nlp)
    assert all(t.islower() for t in toks[0])
    assert all(t.isalpha() for t in toks[0])
    assert "123" not in toks[0]


@spacy_required
def test_pos_lexicon_shape() -> None:
    nlp = text.load_nlp()
    texts = ["great tasty burger", "slow rude service"] * 5
    labels = ["open", "closed"] * 5
    lex = text.pos_lexicon(texts, labels, nlp, "ADJ", top_n=5)
    assert list(lex.columns) == ["group", "pos", "term", "count", "rank"]
    assert set(lex["group"]).issubset({"open", "closed"})
    assert (lex["rank"] <= 5).all()


@spacy_required
def test_contrast_lexicons_keys() -> None:
    nlp = text.load_nlp()
    df = pd.DataFrame(
        {
            "text": ["great tasty burger", "slow rude service"] * 4,
            "closed_open": ["open", "closed"] * 4,
            "cluster": ["Fast Food", "Non-Fast Food"] * 4,
        }
    )
    tables = text.contrast_lexicons(df, nlp)
    assert set(tables) == {
        "lexicon_closed_open_noun",
        "lexicon_closed_open_adj",
        "lexicon_cluster_noun",
        "lexicon_cluster_adj",
    }
