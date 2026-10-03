"""Agent 2: Voice of Customer. Grounded issue extraction from RAG snippets.

The model returns verbatim quotes tagged with an operational issue, polarity, and
intensity. Python grounds every quote against the cited snippet (rapidfuzz partial
ratio >= 85, at least two words, matching review_id), drops ungrounded quotes, and
computes per-issue scores and prevalence. No number comes from the model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from lrr import config, gateway
from lrr.agents.schemas import VoCItem, VoCOutput

# --------------------------------------------------------------------------- #
# Grounding (shares the project-wide thresholds)
# --------------------------------------------------------------------------- #


def _partial_ratio(quote: str, source: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(quote.lower(), source.lower()))
    except ImportError:
        q, s = quote.lower(), source.lower()
        if q in s:
            return 100.0
        qt = q.split()
        best = 0
        for n in range(len(qt), 0, -1):
            for i in range(len(qt) - n + 1):
                if " ".join(qt[i : i + n]) in s:
                    best = max(best, n)
                    break
            if best:
                break
        return 100.0 * best / max(len(qt), 1)


def is_grounded(quote: str, review_id: str, snippets: pd.DataFrame) -> bool:
    """True if the quote is grounded in the snippet with the matching review_id."""
    if not quote or len(quote.split()) < config.QUOTE_MIN_WORDS:
        return False
    match = snippets.loc[snippets["review_id"] == review_id]
    if match.empty:
        return False
    source = str(match.iloc[0]["text"])
    return _partial_ratio(quote, source) >= config.QUOTE_MIN_PARTIAL_RATIO


def ground_items(items: Sequence[VoCItem], snippets: pd.DataFrame) -> dict:
    """Keep grounded items; count drops. Returns {grounded, dropped, n_proposed}."""
    grounded, dropped = [], 0
    for it in items:
        if is_grounded(it.quote, it.review_id, snippets):
            grounded.append(it)
        else:
            dropped += 1
    return {"grounded": grounded, "dropped": dropped, "n_proposed": len(items)}


# --------------------------------------------------------------------------- #
# Issue scoring (Python computes every number)
# --------------------------------------------------------------------------- #


def _sign(polarity: str) -> int:
    return {"positive": 1, "negative": -1, "neutral": 0}.get(polarity, 0)


def issue_scores(items: Sequence[VoCItem]) -> dict:
    """Per-issue score = mean(sign * intensity / 3) over grounded items."""
    by_issue: dict[str, list[float]] = {}
    for it in items:
        by_issue.setdefault(it.issue, []).append(_sign(it.polarity) * int(it.intensity) / 3.0)
    return {issue: float(np.mean(vals)) for issue, vals in by_issue.items()}


def issue_prevalence(items: Sequence[VoCItem]) -> dict:
    """Share of grounded quotes attributed to each issue."""
    n = len(items)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for it in items:
        counts[it.issue] = counts.get(it.issue, 0) + 1
    return {issue: c / n for issue, c in counts.items()}


# --------------------------------------------------------------------------- #
# Agent call
# --------------------------------------------------------------------------- #

_SYSTEM = (
    "You extract customer issues from restaurant review snippets. For each clear "
    "opinion, return a VERBATIM quote copied from a snippet, the snippet's review_id, "
    "an issue from this list: " + ", ".join(config.VOC_ISSUES) + ", a polarity "
    "(positive/negative/neutral), and an intensity 1-3. Return JSON "
    '{"items": [{"quote": str, "review_id": str, "issue": str, '
    '"polarity": str, "intensity": int}]}. Quotes must be copied exactly. No prose.'
)


def build_prompt(snippets: pd.DataFrame) -> list[dict]:
    lines = "\n".join(f"[{r['review_id']}] {r['text']}" for _, r in snippets.iterrows())
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Snippets:\n{lines}\n\nItems (JSON):"},
    ]


def run_voc(snippets: pd.DataFrame, completion_fn=None) -> dict:
    """Call the Voice-of-Customer agent, ground quotes, and score issues.

    Returns a dict with the grounded items, drop count, groundedness rate, issue
    scores, and issue prevalence.
    """
    if snippets is None or snippets.empty:
        return {
            "items": [],
            "dropped": 0,
            "n_proposed": 0,
            "groundedness": 1.0,
            "issue_scores": {},
            "issue_prevalence": {},
        }
    result = gateway.chat(
        build_prompt(snippets),
        role="voice_of_customer",
        schema=VoCOutput,
        completion_fn=completion_fn,
    )
    parsed = result.parsed or _coerce(result.text)
    g = ground_items(parsed.items, snippets)
    grounded = g["grounded"]
    groundedness = (len(grounded) / g["n_proposed"]) if g["n_proposed"] else 1.0
    return {
        "items": grounded,
        "dropped": g["dropped"],
        "n_proposed": g["n_proposed"],
        "groundedness": groundedness,
        "issue_scores": issue_scores(grounded),
        "issue_prevalence": issue_prevalence(grounded),
    }


def _coerce(text: str) -> VoCOutput:
    try:
        start, end = text.find("{"), text.rfind("}")
        return VoCOutput.model_validate(json.loads(text[start : end + 1]))
    except Exception:
        return VoCOutput(items=[])
