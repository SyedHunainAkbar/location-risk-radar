"""Aspect agents: grounded quotes with polarity and intensity (stage 06).

Three aspect agents (food, service, ambience) and a lead agent return JSON only;
Python grounds every quote and computes every number. The LLM never emits a numeric
score. Grounding uses rapidfuzz partial ratio >= 85 with at least two words, matching
the project-wide discipline.

rapidfuzz and scipy are imported lazily so this module imports in a plain
environment; grounding falls back to a pure-Python ratio when rapidfuzz is absent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np

from lrr import config, llm

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_POLARITY_SET = set(config.POLARITIES)


class AspectSchemaError(ValueError):
    """Raised when the agent JSON does not match the aspect schema."""


def _extract_json_array(raw: str) -> list:
    """Extract the first top-level JSON array from a model response."""
    if raw is None:
        raise AspectSchemaError("Empty response.")
    text = raw.strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise AspectSchemaError("No JSON array found in response.")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AspectSchemaError(f"Invalid JSON: {exc}") from exc


def parse_aspect_json(raw: str) -> list[dict]:
    """Parse and validate the aspect agent output.

    Each item must have a non-empty string ``quote``, a ``polarity`` in the allowed
    set, and an integer ``intensity`` in [1, 3]. Raises :class:`AspectSchemaError`.
    """
    data = _extract_json_array(raw)
    if not isinstance(data, list):
        raise AspectSchemaError("Top-level JSON must be an array.")
    items: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise AspectSchemaError(f"Item {i} is not an object.")
        quote = item.get("quote")
        polarity = item.get("polarity")
        intensity = item.get("intensity")
        if not isinstance(quote, str) or not quote.strip():
            raise AspectSchemaError(f"Item {i}: 'quote' must be a non-empty string.")
        if polarity not in _POLARITY_SET:
            raise AspectSchemaError(f"Item {i}: 'polarity' must be one of {sorted(_POLARITY_SET)}.")
        try:
            intensity_int = int(intensity)
        except (TypeError, ValueError):
            raise AspectSchemaError(f"Item {i}: 'intensity' must be an integer.")
        if not (config.INTENSITY_MIN <= intensity_int <= config.INTENSITY_MAX):
            raise AspectSchemaError(
                f"Item {i}: 'intensity' must be in "
                f"[{config.INTENSITY_MIN}, {config.INTENSITY_MAX}]."
            )
        items.append({"quote": quote.strip(), "polarity": polarity, "intensity": intensity_int})
    return items


# --------------------------------------------------------------------------- #
# Prompts (few-shot, temperature 0)
# --------------------------------------------------------------------------- #

_ASPECT_GUIDE = {
    "food": "food quality, taste, temperature, portion, freshness, and the menu",
    "service": "staff speed, attentiveness, friendliness, accuracy, and problem handling",
    "ambience": "cleanliness, noise, decor, seating, comfort, and overall atmosphere",
}

_FEWSHOT = (
    'Example positive -> [{"quote": "the fries were hot and crispy", '
    '"polarity": "positive", "intensity": 2}]\n'
    'Example negative -> [{"quote": "waited forty minutes for a cold burger", '
    '"polarity": "negative", "intensity": 3}]'
)


def aspect_prompt(aspect: str, review_text: str) -> list[dict]:
    """Few-shot messages for one aspect agent. JSON array only, temperature 0."""
    guide = _ASPECT_GUIDE[aspect]
    system = (
        f"You extract {aspect} opinions ({guide}) from a restaurant review. "
        "Return a JSON array only. Each element is "
        '{"quote": verbatim span from the review, "polarity": one of '
        '"positive"/"negative"/"neutral", "intensity": integer 1-3}. '
        "Quote spans must be copied verbatim from the review. If the aspect is not "
        "mentioned, return []. No prose, no numbers other than intensity.\n"
        f"{_FEWSHOT}"
    )
    user = f"Review:\n{review_text}\n\n{aspect.capitalize()} opinions (JSON array):"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def lead_prompt(review_text: str) -> list[dict]:
    """Messages for the lead agent: overall polarity and intensity as JSON object."""
    system = (
        "You summarize the overall sentiment of a restaurant review. Return a JSON "
        'object only: {"polarity": "positive"/"negative"/"neutral", "intensity": '
        "integer 1-3}. No prose."
    )
    user = f"Review:\n{review_text}\n\nOverall (JSON object):"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Agents (JSON validation with retry)
# --------------------------------------------------------------------------- #


def run_aspect_agent(
    aspect: str,
    review_text: str,
    provider: str | None = None,
    retries: int = config.ASPECT_JSON_RETRIES,
) -> list[dict]:
    """Call one aspect agent, validating JSON with a bounded retry.

    On a schema error we append the error to the prompt and retry up to ``retries``
    more times. Returns the validated (ungrounded) item list.
    """
    messages = aspect_prompt(aspect, review_text)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        raw = llm.chat(messages, temperature=0.0, max_tokens=512, provider=provider)
        try:
            return parse_aspect_json(raw)
        except AspectSchemaError as exc:
            last_err = exc
            messages = messages + [
                {"role": "assistant", "content": str(raw)},
                {
                    "role": "user",
                    "content": f"That was invalid ({exc}). Return a valid JSON array only.",
                },
            ]
    raise AspectSchemaError(f"Aspect {aspect!r} failed schema after retries: {last_err}")


def run_lead_agent(review_text: str, provider: str | None = None) -> dict:
    """Call the lead agent and return {polarity, intensity}; falls back to neutral."""
    raw = llm.chat(lead_prompt(review_text), temperature=0.0, max_tokens=64, provider=provider)
    try:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
        pol = obj.get("polarity")
        inten = int(obj.get("intensity", 1))
        if pol in _POLARITY_SET and config.INTENSITY_MIN <= inten <= config.INTENSITY_MAX:
            return {"polarity": pol, "intensity": inten}
    except Exception:
        pass
    return {"polarity": "neutral", "intensity": 1}


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def _partial_ratio(quote: str, source: str) -> float:
    """rapidfuzz partial ratio with a pure-Python fallback."""
    try:
        from rapidfuzz import fuzz  # lazy

        return float(fuzz.partial_ratio(quote.lower(), source.lower()))
    except ImportError:
        # Fallback: longest contiguous substring match ratio over the quote length.
        q = quote.lower()
        s = source.lower()
        if q in s:
            return 100.0
        # crude token overlap as a conservative fallback
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


def ground_quote(quote: str, source: str) -> bool:
    """True if the quote is grounded in the source.

    Requires at least ``QUOTE_MIN_WORDS`` words and a rapidfuzz partial ratio of at
    least ``QUOTE_MIN_PARTIAL_RATIO``.
    """
    if not quote or len(quote.split()) < config.QUOTE_MIN_WORDS:
        return False
    return _partial_ratio(quote, source) >= config.QUOTE_MIN_PARTIAL_RATIO


# --------------------------------------------------------------------------- #
# Scoring (Python computes every number)
# --------------------------------------------------------------------------- #


def polarity_sign(polarity: str) -> int:
    """Map polarity to a sign: positive +1, negative -1, neutral 0."""
    return {"positive": 1, "negative": -1, "neutral": 0}.get(polarity, 0)


def aspect_score(items: Sequence[dict]) -> float:
    """Aspect score = mean(sign * intensity / 3) over items. 0.0 when empty."""
    if not items:
        return 0.0
    vals = [polarity_sign(it["polarity"]) * int(it["intensity"]) / 3.0 for it in items]
    return float(np.mean(vals))


def predicted_stars(items: Sequence[dict]) -> float:
    """Predicted stars = 3 + 2 * aspect_score, clipped to [1, 5]."""
    return float(np.clip(3.0 + 2.0 * aspect_score(items), 1.0, 5.0))


def analyze_review(review_text: str, provider: str | None = None) -> dict:
    """Run all aspect agents + lead on one review, ground quotes, and score.

    Returns a dict with per-aspect grounded items and scores, the overall
    predicted stars, the lead summary, and the dropped-quote count.
    """
    result: dict = {"aspects": {}, "dropped_quotes": 0, "schema_failures": 0}
    all_items: list[dict] = []
    for aspect in config.ASPECTS:
        try:
            items = run_aspect_agent(aspect, review_text, provider=provider)
        except AspectSchemaError:
            # A single aspect that will not return valid JSON after retries must not
            # abort the whole corpus run. We treat it as no groundable quotes for
            # this review and count it, preserving the grounding discipline.
            items = []
            result["schema_failures"] += 1
        grounded = []
        for it in items:
            if ground_quote(it["quote"], review_text):
                grounded.append(it)
            else:
                result["dropped_quotes"] += 1
        result["aspects"][aspect] = {
            "items": grounded,
            "score": aspect_score(grounded),
        }
        all_items.extend(grounded)
    result["overall_score"] = aspect_score(all_items)
    result["predicted_stars"] = predicted_stars(all_items)
    result["lead"] = run_lead_agent(review_text, provider=provider)
    return result


# --------------------------------------------------------------------------- #
# Evaluation against actual stars
# --------------------------------------------------------------------------- #


def evaluate_predicted_stars(predicted, actual) -> dict:
    """Accuracy, macro F1, MAE, and Pearson r with bootstrap CIs.

    Accuracy and macro F1 round predicted stars to the nearest integer star; MAE and
    Pearson use the continuous predictions.
    """
    from sklearn.metrics import accuracy_score, f1_score

    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    pred_round = np.clip(np.rint(predicted), 1, 5).astype(int)
    act_round = np.clip(np.rint(actual), 1, 5).astype(int)

    def _pearson(a, b) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _boot(fn, n: int = config.BOOTSTRAP_N) -> tuple[float, float]:
        rng = np.random.default_rng(config.SEED)
        m = len(predicted)
        stats = []
        for _ in range(n):
            idx = rng.integers(0, m, size=m)
            try:
                v = fn(idx)
            except Exception:
                continue
            if v is not None and not np.isnan(v):
                stats.append(v)
        if not stats:
            return float("nan"), float("nan")
        lo = float(np.percentile(stats, 2.5))
        hi = float(np.percentile(stats, 97.5))
        return lo, hi

    acc = float(accuracy_score(act_round, pred_round))
    mf1 = float(
        f1_score(act_round, pred_round, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0)
    )
    mae = float(np.mean(np.abs(predicted - actual)))
    r = _pearson(predicted, actual)

    return {
        "accuracy": acc,
        "accuracy_ci": _boot(lambda i: accuracy_score(act_round[i], pred_round[i])),
        "macro_f1": mf1,
        "macro_f1_ci": _boot(
            lambda i: f1_score(
                act_round[i],
                pred_round[i],
                average="macro",
                labels=[1, 2, 3, 4, 5],
                zero_division=0,
            )
        ),
        "mae": mae,
        "mae_ci": _boot(lambda i: float(np.mean(np.abs(predicted[i] - actual[i])))),
        "pearson_r": r,
        "pearson_ci": _boot(lambda i: _pearson(predicted[i], actual[i])),
        "n": int(len(predicted)),
    }


def risk_contrast(high_scores, low_scores) -> dict:
    """Mann-Whitney U test + rank-biserial effect size for high vs low risk.

    ``high_scores`` and ``low_scores`` are per-location aspect scores. Returns the
    U statistic, p-value, effect size, and group medians.
    """
    from scipy.stats import mannwhitneyu

    high = np.asarray(high_scores, dtype=float)
    low = np.asarray(low_scores, dtype=float)
    if len(high) == 0 or len(low) == 0:
        return {
            "u": float("nan"),
            "p_value": float("nan"),
            "effect_size": float("nan"),
            "n_high": len(high),
            "n_low": len(low),
        }
    u, p = mannwhitneyu(high, low, alternative="two-sided")
    # Rank-biserial effect size from U.
    effect = 1.0 - (2.0 * u) / (len(high) * len(low))
    return {
        "u": float(u),
        "p_value": float(p),
        "effect_size": float(effect),
        "median_high": float(np.median(high)),
        "median_low": float(np.median(low)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
    }
