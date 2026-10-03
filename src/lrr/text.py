"""spaCy normalization and descriptive lexicons (LA1).

We normalize review text with spaCy's `nlp.pipe` in batches (lowercase lemma,
stopword and punctuation removal) and keep part-of-speech tags so we can isolate
nouns and adjectives. From these we build the top-N term tables that contrast
closed vs open locations and Fast Food vs Non-Fast Food reviews.

spaCy is imported lazily inside the functions that need it so this module imports
cleanly in environments where the model is not installed (for example CI running
only the sentiment tests).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd

from lrr import config


def load_nlp(
    model: str = config.SPACY_MODEL_APP,
    disable: Sequence[str] = ("parser",),
):
    """Load a spaCy pipeline with unused components disabled.

    We keep the tagger, lemmatizer, attribute ruler, and NER; the dependency
    parser is disabled by default because the lexicon work does not need it and
    disabling it speeds up `nlp.pipe`.

    Raises:
        OSError: If the spaCy model is not installed.
    """
    import spacy  # lazy import

    return spacy.load(model, disable=list(disable))


def normalize_tokens(
    texts: Iterable[str],
    nlp,
    batch_size: int = config.SPACY_BATCH_SIZE,
    keep_pos: set[str] | None = None,
) -> list[list[str]]:
    """Normalize texts to token lists via `nlp.pipe`.

    For each document we lowercase the lemma and drop stopwords, punctuation,
    whitespace, and non-alphabetic tokens. When `keep_pos` is given (for example
    `{"NOUN"}` or `{"ADJ"}`), only tokens with those coarse POS tags survive.

    Args:
        texts: Iterable of raw review strings.
        nlp: A loaded spaCy pipeline (see :func:`load_nlp`).
        batch_size: Batch size passed to `nlp.pipe`.
        keep_pos: Optional set of coarse POS tags to retain.

    Returns:
        A list of token lists, one per input document, in input order.
    """
    out: list[list[str]] = []
    for doc in nlp.pipe((str(t) for t in texts), batch_size=batch_size):
        toks: list[str] = []
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            if not tok.is_alpha:
                continue
            if keep_pos is not None and tok.pos_ not in keep_pos:
                continue
            toks.append(tok.lemma_.lower())
        out.append(toks)
    return out


def pos_lexicon(
    texts: Sequence[str],
    labels: Sequence[str],
    nlp,
    pos: str,
    top_n: int = config.LEXICON_TOP_N,
    batch_size: int = config.SPACY_BATCH_SIZE,
) -> pd.DataFrame:
    """Top-`top_n` lemmas of a POS within each label group.

    Args:
        texts: Review strings.
        labels: Group label per text (same length as `texts`).
        nlp: Loaded spaCy pipeline.
        pos: Coarse POS tag to isolate, for example "NOUN" or "ADJ".
        top_n: Number of terms to keep per group.
        batch_size: Batch size for `nlp.pipe`.

    Returns:
        Tidy DataFrame with columns `group, pos, term, count, rank`.
    """
    token_lists = normalize_tokens(texts, nlp, batch_size, keep_pos={pos})
    counters: dict[str, Counter] = {}
    for toks, label in zip(token_lists, labels):
        counters.setdefault(str(label), Counter()).update(toks)

    rows = []
    for group, counter in counters.items():
        for rank, (term, count) in enumerate(counter.most_common(top_n), start=1):
            rows.append(
                {"group": group, "pos": pos, "term": term, "count": int(count), "rank": rank}
            )
    cols = ["group", "pos", "term", "count", "rank"]
    return pd.DataFrame(rows, columns=cols)


def contrast_lexicons(reviews_df: pd.DataFrame, nlp) -> dict[str, pd.DataFrame]:
    """Build the four descriptive lexicon tables.

    Expects `reviews_df` with columns `text`, `closed_open` (values "closed" or
    "open"), and `cluster` (values "Fast Food" or "Non-Fast Food").

    Returns a dict keyed by artifact filename stem:
    `lexicon_closed_open_noun`, `lexicon_closed_open_adj`,
    `lexicon_cluster_noun`, `lexicon_cluster_adj`.
    """
    texts = reviews_df["text"].astype(str).tolist()
    closed_open = reviews_df["closed_open"].astype(str).tolist()
    cluster = reviews_df["cluster"].astype(str).tolist()

    return {
        "lexicon_closed_open_noun": pos_lexicon(texts, closed_open, nlp, "NOUN"),
        "lexicon_closed_open_adj": pos_lexicon(texts, closed_open, nlp, "ADJ"),
        "lexicon_cluster_noun": pos_lexicon(texts, cluster, nlp, "NOUN"),
        "lexicon_cluster_adj": pos_lexicon(texts, cluster, nlp, "ADJ"),
    }
