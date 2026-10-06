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


def _default_model(provider: str) -> str | None:
    """Model to use when a caller names a provider but no model.

    Previously ``model=None`` was sent to the API, which rejected it, and the call
    silently fell back to mock. We now take the first verified model configured for
    that provider in the gateway routes, then the configured NVIDIA answer model.
    """
    if provider == "mock":
        return None
    try:
        from lrr.gateway import routing as _routing

        for chain in _routing.load_routes().values():
            for entry in chain:
                if entry.provider == provider and entry.model:
                    return entry.model
    except Exception:  # noqa: BLE001
        pass
    if provider == "nvidia":
        from lrr import config as _config

        return _config.RAG_ANSWER_MODEL
    return None


def _strict_live() -> bool:
    """When LRR_STRICT_LIVE=1, never fall back to mock (pipeline runs must be live)."""
    import os

    return os.environ.get("LRR_STRICT_LIVE", "").strip() in {"1", "true", "yes"}


def _adhoc_chain(
    provider: str, model: str | None, temperature: float, max_tokens: int
) -> list[RouteEntry]:
    """A single explicit provider followed by the mock terminal fallback."""
    model = model or _default_model(provider)
    chain = [RouteEntry(provider, model, temperature, max_tokens, 60.0)]
    if provider != "mock" and not _strict_live():
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
        if _strict_live() and getattr(result, "provider", prov) == "mock" and prov != "mock":
            raise LLMError(f"Live call to {prov} fell back to mock (strict mode).")
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
