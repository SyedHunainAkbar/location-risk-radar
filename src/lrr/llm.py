"""Backward-compatible LLM facade over the gateway package.

This module used to hold the whole LLM client. It is now a thin shim over
``lrr.gateway`` so existing callers (topics, aspects, rag, LA5 helpers) keep working
unchanged, while new code can call ``lrr.gateway.chat`` with a role and schema.

The legacy ``chat(messages, model=None, temperature=..., max_tokens=..., provider=None)``
signature is preserved and still returns a plain string. When ``provider``/``model``
are given, we route through a one-entry ad hoc chain (then mock) so the explicit
choice is honored; otherwise we use the gateway's role routing. The ``mock`` provider
remains deterministic and offline.
"""

from __future__ import annotations

from collections.abc import Sequence

from lrr.gateway import GatewayError, GatewayResult, get_gateway  # noqa: F401
from lrr.gateway import providers as _providers
from lrr.gateway.core import Gateway
from lrr.gateway.routing import RouteEntry

Message = dict[str, str]

# Re-exports so `from lrr.llm import LLMError` style imports still resolve.
LLMError = GatewayError

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024


def _adhoc_chain(
    provider: str, model: str | None, temperature: float, max_tokens: int
) -> list[RouteEntry]:
    """A single explicit provider followed by the mock terminal fallback."""
    chain = [RouteEntry(provider, model, temperature, max_tokens, 60.0)]
    if provider != "mock":
        chain.append(RouteEntry("mock", None, temperature, max_tokens, 30.0))
    return chain


def chat(
    messages: Sequence[Message],
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    provider: str | None = None,
    role: str | None = None,
    *,
    _sleep=None,
    _completion_fn=None,
) -> str:
    """Legacy chat entry point. Returns the completion text as a string.

    - If ``provider`` is given, route through that provider then mock.
    - Else if ``role`` is given, use the gateway's role routing.
    - Else default to the ``quant_analyst`` role.

    ``_completion_fn`` and ``_sleep`` are injectable for tests, matching the old API.
    """
    if provider is not None:
        prov = provider.strip().lower()
        chain = _adhoc_chain(prov, model, temperature, max_tokens)
        gw = Gateway(routes={"_adhoc": chain}, sleep=_sleep or __import__("time").sleep)
        result = gw.chat(messages, role="_adhoc", completion_fn=_completion_fn)
        return result.text

    result = get_gateway().chat(
        messages, role=role or "quant_analyst", completion_fn=_completion_fn
    )
    return result.text


# Legacy helpers some callers referenced; kept for compatibility.
def resolve_provider(provider: str | None = None) -> str:
    name = (provider or _providers.get_secret("LLM_PROVIDER") or "nvidia").strip().lower()
    if name not in _providers.PROVIDERS:
        raise LLMError(f"Unknown provider {name!r}.")
    return name
