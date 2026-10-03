"""LLM Gateway: one entry point for every model call in the project.

Public facade:

- ``chat(messages, role=..., schema=..., **overrides) -> GatewayResult``
- ``achat(...)`` async variant
- ``get_gateway()`` shared instance
- ``GatewayResult`` dataclass with the answering provider, model, cache/fallback flags
- ``verify_models()`` to confirm model ids and write artifacts/gateway_models.json

Keys are read only from the environment or ``st.secrets`` and are never logged or
cached. ``mock`` is always the terminal fallback so a keyless run still answers.
"""

from __future__ import annotations

from collections.abc import Sequence

from lrr.gateway.core import (
    Gateway,
    GatewayError,
    GatewayResult,
    get_gateway,
    reset_gateway,
)
from lrr.gateway.ledger import Ledger, LedgerRow
from lrr.gateway.redaction import redact
from lrr.gateway.verify import verify_models

Message = dict[str, str]

__all__ = [
    "chat",
    "achat",
    "Gateway",
    "GatewayResult",
    "GatewayError",
    "get_gateway",
    "reset_gateway",
    "Ledger",
    "LedgerRow",
    "redact",
    "verify_models",
]


def chat(
    messages: Sequence[Message],
    role: str = "quant_analyst",
    schema=None,
    **overrides,
) -> GatewayResult:
    """Route a chat request through the role's fallback chain (shared gateway)."""
    return get_gateway().chat(messages, role=role, schema=schema, **overrides)


async def achat(
    messages: Sequence[Message],
    role: str = "quant_analyst",
    schema=None,
    **overrides,
) -> GatewayResult:
    """Async facade for parallel calls."""
    return await get_gateway().achat(messages, role=role, schema=schema, **overrides)
