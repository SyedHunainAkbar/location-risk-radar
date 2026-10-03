"""Agent 3: Risk Auditor. A different-family model checks agents 1 and 2.

Python builds the claim list from the rendered drivers and grounded Voice-of-Customer
issues, each with a stable claim id, and passes it to the auditor along with context
flags. The auditor returns a verdict per claim, alternative explanations considered,
and an A/B/C confidence grade. Python removes unsupported claims downstream.
"""

from __future__ import annotations

import json

from lrr import gateway
from lrr.agents.schemas import AuditOutput


def build_claims(rendered_drivers: list[dict], voc: dict) -> list[dict]:
    """Build the claim list Python asks the auditor to check.

    Driver claims come from the rendered templates; evidence claims from the grounded
    issues. Each gets a stable ``claim_id`` so verdicts map back deterministically.
    """
    claims = []
    for i, d in enumerate(rendered_drivers):
        claims.append(
            {
                "claim_id": f"driver_{i}",
                "kind": "driver",
                "text": d["text"],
                "driver_key": d["driver_key"],
            }
        )
    for issue, score in sorted(voc.get("issue_scores", {}).items()):
        claims.append(
            {
                "claim_id": f"issue_{issue}",
                "kind": "issue",
                "issue": issue,
                "text": f"Customers raise {issue.replace('_', ' ')} concerns at this location.",
            }
        )
    return claims


_SYSTEM = (
    "You are an independent risk auditor from a different team than the analysts. "
    "For each claim, judge whether the provided evidence supports it. Reply JSON "
    '{"claims": [{"claim_id": str, "verdict": "supported"|"unsupported"|'
    '"insufficient", "reason": str}], "alternatives": [{"explanation": str, '
    '"verdict": str}], "confidence_grade": "A"|"B"|"C"}. Consider the context '
    "flags (low review volume, pandemic overlap, chain-wide trend, cluster label "
    "noise) as possible alternative explanations. No prose outside JSON."
)


def build_prompt(claims: list[dict], voc: dict, context_flags: dict) -> list[dict]:
    evidence = [
        {
            "review_id": it.review_id,
            "quote": it.quote,
            "issue": it.issue,
            "polarity": it.polarity,
            "intensity": int(it.intensity),
        }
        for it in voc.get("items", [])
    ]
    user = (
        "Claims to audit:\n"
        + json.dumps(claims, ensure_ascii=False)
        + "\n\nGrounded evidence (quotes with citations):\n"
        + json.dumps(evidence, ensure_ascii=False)
        + "\n\nContext flags:\n"
        + json.dumps(context_flags, ensure_ascii=False)
        + "\n\nReturn the audit JSON."
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def run_auditor(
    claims: list[dict], voc: dict, context_flags: dict, completion_fn=None
) -> AuditOutput:
    """Call the auditor (role risk_auditor, a different family). Falls back safely."""
    if not claims:
        return AuditOutput(claims=[], alternatives=[], confidence_grade="C")
    result = gateway.chat(
        build_prompt(claims, voc, context_flags),
        role="risk_auditor",
        schema=AuditOutput,
        completion_fn=completion_fn,
    )
    return result.parsed or _coerce(result.text, claims)


def groundedness_rate(audit: AuditOutput) -> float:
    """Supported claims / total claims from the auditor. 0.0 when no claims."""
    total = len(audit.claims)
    if total == 0:
        return 0.0
    supported = sum(1 for c in audit.claims if c.verdict == "supported")
    return supported / total


def supported_claim_ids(audit: AuditOutput) -> set[str]:
    return {c.claim_id for c in audit.claims if c.verdict == "supported"}


def _coerce(text: str, claims: list[dict]) -> AuditOutput:
    """Parse the auditor JSON, or mark every claim 'insufficient' as a safe default."""
    try:
        start, end = text.find("{"), text.rfind("}")
        return AuditOutput.model_validate(json.loads(text[start : end + 1]))
    except Exception:
        return AuditOutput(
            claims=[
                {
                    "claim_id": c["claim_id"],
                    "verdict": "insufficient",
                    "reason": "auditor output unavailable offline",
                }
                for c in claims
            ],
            alternatives=[],
            confidence_grade="C",
        )
