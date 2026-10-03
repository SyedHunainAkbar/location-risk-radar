"""Risk committee agents: a three-agent Location Risk Brief orchestrated by Python.

Public surface:

- ``build_brief`` / ``abuild_brief`` and the ``Brief`` dataclass
- ``build_factsheet`` to assemble the Quant Analyst fact sheet
- ``persist_briefs`` to write artifacts/briefs.parquet
- ``compute_confidence`` and ``CONFIDENCE_FORMULA`` (documented)

Every model call routes through ``lrr.gateway`` with the roles ``quant_analyst``,
``voice_of_customer``, and ``risk_auditor`` (a different family). The agents produce
language and labels only; Python grounds quotes, renders numbers, and derives the
confidence grade.
"""

from __future__ import annotations

from lrr.agents.confidence import (
    CONFIDENCE_FORMULA,
    compute_confidence,
    confidence_label,
)
from lrr.agents.factsheet import build_factsheet
from lrr.agents.orchestrator import Brief, abuild_brief, build_brief, persist_briefs
from lrr.agents.quant import numeric_fidelity_rate, scan_rendered_fidelity

__all__ = [
    "Brief",
    "build_brief",
    "abuild_brief",
    "persist_briefs",
    "build_factsheet",
    "compute_confidence",
    "confidence_label",
    "CONFIDENCE_FORMULA",
    "numeric_fidelity_rate",
    "scan_rendered_fidelity",
]
