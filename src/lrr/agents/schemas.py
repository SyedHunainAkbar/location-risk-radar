"""Pydantic output schemas for the three committee agents.

These are passed to the gateway's JSON mode so a repair retry fires on malformed
output. The agents emit language and labels only; Python computes every number.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, conint

from lrr import config

# Literal unions as plain validators to stay compatible across pydantic minor versions.
_ISSUES = set(config.VOC_ISSUES)
_POLARITIES = {"positive", "negative", "neutral"}
_DIRECTIONS = {"up", "down"}
_VERDICTS = {"supported", "unsupported", "insufficient"}
_GRADES = {"A", "B", "C"}


class Driver(BaseModel):
    """One risk driver as a placeholder-only explanation template."""

    driver_key: str
    direction: str = Field(description="up or down")
    explanation_template: str

    def model_post_init(self, _ctx) -> None:  # pydantic v2 hook
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")


class QuantOutput(BaseModel):
    drivers: list[Driver]


class VoCItem(BaseModel):
    quote: str
    review_id: str
    issue: str
    polarity: str
    intensity: conint(ge=1, le=3)  # type: ignore[valid-type]

    def model_post_init(self, _ctx) -> None:
        if self.issue not in _ISSUES:
            raise ValueError(f"issue must be one of {sorted(_ISSUES)}")
        if self.polarity not in _POLARITIES:
            raise ValueError(f"polarity must be one of {sorted(_POLARITIES)}")


class VoCOutput(BaseModel):
    items: list[VoCItem]


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: str
    reason: str

    def model_post_init(self, _ctx) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}")


class Alternative(BaseModel):
    explanation: str
    verdict: str


class AuditOutput(BaseModel):
    claims: list[ClaimVerdict]
    alternatives: list[Alternative] = []
    confidence_grade: str = "C"

    def model_post_init(self, _ctx) -> None:
        if self.confidence_grade not in _GRADES:
            raise ValueError(f"confidence_grade must be one of {sorted(_GRADES)}")
