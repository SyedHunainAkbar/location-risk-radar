"""Deterministic orchestration of the three-agent risk committee.

Python runs the Quant Analyst and Voice-of-Customer agents concurrently with
``asyncio.gather``, then the Risk Auditor, then assembles the Location Risk Brief:
tier, rendered drivers, grounded evidence with citations, unsupported claims removed,
confidence (from the documented formula), and recommended actions from the playbook.
Only the per-agent calls use the LLM; everything else is deterministic Python.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lrr import config
from lrr.agents import auditor as auditor_mod
from lrr.agents import confidence as conf_mod
from lrr.agents import playbook as playbook_mod
from lrr.agents import quant as quant_mod
from lrr.agents import voc as voc_mod


@dataclass
class Brief:
    business_id: str
    tier: str
    drivers: list[dict] = field(default_factory=list)  # audit-supported driver text
    rendered_drivers: list[dict] = field(default_factory=list)  # all rendered (pre-audit)
    evidence: list[dict] = field(default_factory=list)  # grounded quotes + citations
    confidence: float = 0.0
    confidence_label: str = "Low"
    actions: list[str] = field(default_factory=list)
    groundedness: float = 0.0
    n_evidence: int = 0
    auditor_grade: str = "C"
    rejected_claims: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        """Flatten to a parquet-friendly row (text joined for the app)."""
        return {
            "business_id": self.business_id,
            "tier": self.tier,
            "drivers": " | ".join(d["text"] for d in self.drivers),
            "evidence": " | ".join(f'"{e["quote"]}" [{e["review_id"]}]' for e in self.evidence),
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "actions": " | ".join(self.actions),
            "groundedness": self.groundedness,
            "n_evidence": self.n_evidence,
            "auditor_grade": self.auditor_grade,
            "n_rejected_claims": len(self.rejected_claims),
        }


async def _run_quant_async(factsheet, completion_fn):
    return await asyncio.to_thread(
        quant_mod.run_quant, factsheet, config.QUANT_PLACEHOLDER_RETRIES, completion_fn
    )


async def _run_voc_async(snippets, completion_fn):
    return await asyncio.to_thread(voc_mod.run_voc, snippets, completion_fn)


async def abuild_brief(
    factsheet: dict,
    snippets: pd.DataFrame,
    context_flags: dict | None = None,
    quant_fn=None,
    voc_fn=None,
    auditor_fn=None,
) -> Brief:
    """Build one Location Risk Brief. Agents 1 and 2 run concurrently, then 3."""
    context_flags = context_flags or {}

    # Agents 1 and 2 concurrently.
    quant_parsed, voc_result = await asyncio.gather(
        _run_quant_async(factsheet, quant_fn),
        _run_voc_async(snippets, voc_fn),
    )

    rendered = quant_mod.render_drivers(quant_parsed, factsheet)

    # Agent 3 audits the assembled claims.
    claims = auditor_mod.build_claims(rendered, voc_result)
    audit = auditor_mod.run_auditor(claims, voc_result, context_flags, auditor_fn)
    supported = auditor_mod.supported_claim_ids(audit)

    # Drop unsupported driver claims from the brief.
    kept_drivers, rejected = [], []
    for i, d in enumerate(rendered):
        if f"driver_{i}" in supported or not audit.claims:
            kept_drivers.append(d)
        else:
            rejected.append(f"driver_{i}")

    gr = auditor_mod.groundedness_rate(audit)
    n_evidence = len(voc_result.get("items", []))
    confidence = conf_mod.compute_confidence(gr, n_evidence, audit.confidence_grade)

    evidence = [
        {"quote": it.quote, "review_id": it.review_id, "issue": it.issue}
        for it in voc_result.get("items", [])
    ]
    actions = playbook_mod.recommend(voc_result.get("issue_prevalence", {}))

    return Brief(
        business_id=factsheet.get("business_id", ""),
        tier=factsheet.get("risk_tier", "Unknown"),
        drivers=kept_drivers,
        rendered_drivers=rendered,
        evidence=evidence,
        confidence=confidence,
        confidence_label=conf_mod.confidence_label(confidence),
        actions=actions,
        groundedness=round(gr, 3),
        n_evidence=n_evidence,
        auditor_grade=audit.confidence_grade,
        rejected_claims=rejected,
    )


def build_brief(
    factsheet: dict,
    snippets: pd.DataFrame,
    context_flags: dict | None = None,
    quant_fn=None,
    voc_fn=None,
    auditor_fn=None,
) -> Brief:
    """Synchronous wrapper over :func:`abuild_brief`."""
    return asyncio.run(
        abuild_brief(factsheet, snippets, context_flags, quant_fn, voc_fn, auditor_fn)
    )


def persist_briefs(briefs: list[Brief], path: Path = None) -> Path:
    """Write briefs to artifacts/briefs.parquet (one row each)."""
    path = Path(path or (config.ARTIFACTS_DIR / config.BRIEFS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([b.to_row() for b in briefs])
    df.to_parquet(path, index=False)
    return path
