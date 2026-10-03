"""Documented final-confidence formula for the committee brief.

Python derives the brief's confidence from three auditable inputs: the auditor's
claim groundedness rate, the number of grounded evidence quotes, and the auditor's
A/B/C grade. The formula is intentionally simple and monotonic in each input.
"""

from __future__ import annotations

from lrr import config

#: The documented formula, as text, for the model card and notebook.
CONFIDENCE_FORMULA: str = (
    "confidence = groundedness_rate * min(1, grounded_evidence_count / "
    f"{config.CONFIDENCE_EVIDENCE_TARGET}) * grade_weight[grade], where grade_weight "
    f"= {config.CONFIDENCE_GRADE_WEIGHT}. The result is clamped to [0, 1] and labeled "
    "High >= 0.66, Medium >= 0.33, else Low."
)


def compute_confidence(
    groundedness_rate: float,
    grounded_evidence_count: int,
    grade: str,
) -> float:
    """Return the brief confidence in [0, 1].

    Args:
        groundedness_rate: supported claims / total claims from the auditor (0..1).
        grounded_evidence_count: number of grounded Voice-of-Customer quotes.
        grade: auditor confidence grade, one of A/B/C.
    """
    gr = max(0.0, min(1.0, float(groundedness_rate)))
    evidence_factor = min(
        1.0, max(0, int(grounded_evidence_count)) / config.CONFIDENCE_EVIDENCE_TARGET
    )
    weight = config.CONFIDENCE_GRADE_WEIGHT.get(str(grade), 0.4)
    return round(max(0.0, min(1.0, gr * evidence_factor * weight)), 3)


def confidence_label(confidence: float) -> str:
    """Map a confidence in [0, 1] to a High/Medium/Low label."""
    cuts = config.CONFIDENCE_LABEL_CUTS
    if confidence >= cuts["High"]:
        return "High"
    if confidence >= cuts["Medium"]:
        return "Medium"
    return "Low"
