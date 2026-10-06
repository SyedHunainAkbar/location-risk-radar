"""The Gateway: routing, retries, JSON repair, caching, and the usage ledger.

A caller asks for a role; the gateway resolves it to a fallback chain, drops
unhealthy providers (keeping mock), checks the cache, then walks the chain calling
each provider with a rate limiter, timeout, exponential backoff with jitter on
429/5xx, and a larger-token retry on empty content from reasoning models. When a
pydantic schema is given it validates and performs exactly one repair retry.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lrr import config
from lrr.gateway import cache as cache_mod
from lrr.gateway import health, providers, routing
from lrr.gateway.ledger import Ledger, LedgerRow, estimate_cost
from lrr.gateway.ratelimit import RateLimiter
from lrr.gateway.redaction import get_logger

logger = get_logger()
Message = dict[str, str]


class GatewayError(RuntimeError):
    """Raised when every provider in the chain fails."""


class _DailyBudget:
    """Process-wide estimated spend guard, reset each UTC day.

    This is a soft, best-effort guard for a public URL. It bounds spend across all
    sessions in a single app process; it is not a hard billing control.
    """

    def __init__(self) -> None:
        self._day: str | None = None
        self._spent: float = 0.0

    def _roll(self) -> None:
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._spent = 0.0

    def exceeded(self) -> bool:
        budget = config.GATEWAY_DAILY_BUDGET_USD
        if not budget:
            return False
        self._roll()
        return self._spent >= budget

    def add(self, cost: float) -> None:
        self._roll()
        self._spent += max(0.0, float(cost))

    @property
    def spent_today(self) -> float:
        self._roll()
        return self._spent


DAILY_BUDGET = _DailyBudget()


@dataclass
class GatewayResult:
    text: str
    provider: str
    model: str | None
    role: str
    cache_hit: bool = False
    fallback_used: bool = False
    latency_s: float = 0.0
    parsed: Any = None


def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if any(
        t in name
        for t in (
            "ratelimit",
            "timeout",
            "connection",
            "apierror",
            "internalserver",
            "serviceunavailable",
        )
    ):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return isinstance(status, int) and (status == 429 or 500 <= status < 600)


def _extract(response: Any) -> str | None:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    if content is None:
        return None
    text = str(content).strip()
    return text or None


def _usage(response: Any, prompt_text: str, completion_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is not None and ct is not None:
            return int(pt), int(ct)
    return providers.estimate_tokens(prompt_text), providers.estimate_tokens(completion_text)


def _schema_fields(schema) -> list[str]:
    try:
        return list(schema.model_fields.keys())  # pydantic v2
    except Exception:
        try:
            return list(schema.__fields__.keys())  # pydantic v1
        except Exception:
            return []


class Gateway:
    """Central LLM gateway. One instance is typically shared via get_gateway()."""

    def __init__(
        self,
        routes: dict | None = None,
        ledger: Ledger | None = None,
        rate_limiter: RateLimiter | None = None,
        sleep=time.sleep,
        cached_only: bool = False,
        health_ping=None,
    ) -> None:
        self.routes = routes or routing.load_routes()
        self.ledger = ledger or Ledger()
        self.rate_limiter = rate_limiter or RateLimiter(sleep=sleep)
        self._sleep = sleep
        self.cached_only = cached_only
        self._health_ping = health_ping

    # ------------------------------------------------------------------ #
    # Chain resolution
    # ------------------------------------------------------------------ #

    def _resolve_chain(self, role: str) -> list[routing.RouteEntry]:
        chain = routing.get_chain(role, self.routes)
        healthy = health.healthy_providers(ping=self._health_ping)
        resolved = [
            self._clamp_tokens(e)
            for e in chain
            if e.provider == "mock"
            or (e.provider in healthy and providers.provider_available(e.provider))
        ]
        # Strict live mode (LRR_STRICT_LIVE=1): never answer from mock, fail loudly.
        if _strict_live() and not self.cached_only:
            return [e for e in resolved if e.provider != "mock"]
        # Always guarantee a terminal mock so a keyless run answers.
        if not any(e.provider == "mock" for e in resolved):
            resolved.append(routing.RouteEntry("mock", None, 0.0, 256, 30))
        return resolved

    @staticmethod
    def _clamp_tokens(entry: routing.RouteEntry) -> routing.RouteEntry:
        """Clamp a route's max_tokens to the global per-call ceiling."""
        ceiling = config.GATEWAY_MAX_TOKENS_PER_CALL
        if ceiling and entry.max_tokens > ceiling:
            return routing.RouteEntry(
                entry.provider, entry.model, entry.temperature, ceiling, entry.timeout
            )
        return entry

    # ------------------------------------------------------------------ #
    # One provider call with retries and empty-content handling
    # ------------------------------------------------------------------ #

    def _call_once(
        self,
        entry: routing.RouteEntry,
        messages: Sequence[Message],
        completion_fn=None,
    ) -> tuple[str, Any]:
        """Call a single provider with retries. Returns (text, raw_response)."""
        if entry.provider == "mock" and completion_fn is None:
            fields = getattr(self, "_mock_schema_fields", None)
            return providers.mock_response(
                messages, entry.model or "mock-model", schema_fields=fields
            ), None

        if completion_fn is None:
            client = providers.make_client(entry.provider)

            def completion_fn(**kw):  # noqa: ANN001
                return client.chat.completions.create(**kw)

        max_tokens = entry.max_tokens
        empty_retries = 2
        last_exc: Exception | None = None
        for attempt in range(config.GATEWAY_MAX_RETRIES):
            self.rate_limiter.acquire(entry.provider)
            try:
                resp = completion_fn(
                    model=entry.model,
                    messages=list(messages),
                    temperature=entry.temperature,
                    max_tokens=max_tokens,
                    timeout=entry.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_retryable(exc) and attempt < config.GATEWAY_MAX_RETRIES - 1:
                    self._backoff(attempt)
                    continue
                raise
            text = _extract(resp)
            if text is not None:
                return text, resp
            if empty_retries > 0:  # reasoning model spent budget on hidden tokens
                empty_retries -= 1
                ceiling = config.GATEWAY_MAX_TOKENS_PER_CALL or (max_tokens * 2)
                max_tokens = min(int(max_tokens * 2), ceiling)
                continue
            raise GatewayError(f"{entry.provider} returned empty content")
        raise GatewayError(f"{entry.provider} failed: {last_exc}")

    def _backoff(self, attempt: int) -> None:
        delay = config.GATEWAY_BASE_DELAY * (config.GATEWAY_BACKOFF**attempt)
        delay += random.uniform(0, config.GATEWAY_JITTER)
        self._sleep(delay)

    # ------------------------------------------------------------------ #
    # Public chat
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: Sequence[Message],
        role: str = "quant_analyst",
        schema=None,
        completion_fn=None,
        **overrides,
    ) -> GatewayResult:
        """Route a chat request through the role's fallback chain."""
        chain = self._resolve_chain(role)
        self._mock_schema_fields = _schema_fields(schema) if schema else None

        params = {
            "temperature": chain[0].temperature,
            "max_tokens": chain[0].max_tokens,
            "schema": getattr(schema, "__name__", None),
        }

        # Cache check (keyed on the first entry's provider/model + params).
        key_entry = chain[0]
        key = cache_mod.cache_key(key_entry.provider, key_entry.model, messages, params)
        hit = cache_mod.get(key)
        # A cached mock answer is never reused for a live request: it would silently
        # pass offline placeholder output off as a model result.
        if hit is not None and hit.get("provider") == "mock" and not self.cached_only:
            hit = None
        if hit is not None:
            self.ledger.append(
                LedgerRow(
                    role=role,
                    provider=hit["provider"],
                    model=hit.get("model"),
                    latency_s=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    est_cost_usd=0.0,
                    cache_hit=True,
                    fallback_used=hit.get("fallback_used", False),
                )
            )
            return GatewayResult(
                text=hit["text"],
                provider=hit["provider"],
                model=hit.get("model"),
                role=role,
                cache_hit=True,
                fallback_used=hit.get("fallback_used", False),
                parsed=self._maybe_parse(hit["text"], schema),
            )

        if self.cached_only:
            # Cached-only mode: never hit the network; answer from mock.
            chain = [routing.RouteEntry("mock", None, 0.0, 256, 30)]

        # Global daily spend guard: once exceeded, serve only mock for this process.
        if DAILY_BUDGET.exceeded():
            chain = [routing.RouteEntry("mock", None, 0.0, 256, 30)]

        start = time.time()
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        last_exc: Exception | None = None
        for i, entry in enumerate(chain):
            try:
                text, resp = self._call_once(entry, messages, completion_fn)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.info(
                    "provider %s failed, falling back: %s", entry.provider, type(exc).__name__
                )
                continue

            parsed = None
            if schema is not None:
                text, parsed = self._validate_or_repair(
                    entry, messages, text, schema, completion_fn
                )

            latency = time.time() - start
            pt, ct = (0, 0)
            if resp is not None:
                pt, ct = _usage(resp, prompt_text, text)
            else:
                pt, ct = providers.estimate_tokens(prompt_text), providers.estimate_tokens(text)
            fallback_used = i > 0
            call_cost = estimate_cost(entry.model, pt, ct)
            if entry.provider != "mock":
                DAILY_BUDGET.add(call_cost)
            self.ledger.append(
                LedgerRow(
                    role=role,
                    provider=entry.provider,
                    model=entry.model,
                    latency_s=round(latency, 3),
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    est_cost_usd=call_cost,
                    cache_hit=False,
                    fallback_used=fallback_used,
                )
            )
            if entry.provider != "mock":
                cache_mod.put(
                  key,
                  {
                      "text": text,
                      "provider": entry.provider,
                      "model": entry.model,
                      "fallback_used": fallback_used,
                  },
                )
            return GatewayResult(
                text=text,
                provider=entry.provider,
                model=entry.model,
                role=role,
                cache_hit=False,
                fallback_used=fallback_used,
                latency_s=round(latency, 3),
                parsed=parsed,
            )

        if not chain:
            raise GatewayError(f"strict live mode: no live provider available for {role!r}")
        raise GatewayError(f"all providers failed for role {role!r}: {last_exc}")

    async def achat(
        self, messages, role: str = "quant_analyst", schema=None, **overrides
    ) -> GatewayResult:
        """Async wrapper so callers can fan out parallel requests."""
        import anyio

        return await anyio.to_thread.run_sync(
            lambda: self.chat(messages, role=role, schema=schema, **overrides)
        )

    # ------------------------------------------------------------------ #
    # JSON schema validation + one repair
    # ------------------------------------------------------------------ #

    def _maybe_parse(self, text: str, schema):
        if schema is None:
            return None
        try:
            return schema.model_validate_json(_json_slice(text))
        except Exception:
            return None

    def _validate_or_repair(self, entry, messages, text, schema, completion_fn):
        parsed = self._maybe_parse(text, schema)
        if parsed is not None:
            return text, parsed
        repair = list(messages) + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "That was not valid JSON for the required schema. "
                "Return ONLY a valid JSON object, no prose.",
            },
        ]
        try:
            text2, _ = self._call_once(entry, repair, completion_fn)
            parsed2 = self._maybe_parse(text2, schema)
            if parsed2 is not None:
                return text2, parsed2
            return text2, None
        except Exception:
            return text, None


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _json_slice(text: str) -> str:
    m = _JSON_RE.search(str(text))
    return m.group(0) if m else str(text)


# --------------------------------------------------------------------------- #
# Shared instance
# --------------------------------------------------------------------------- #

_GATEWAY: Gateway | None = None


def _strict_live() -> bool:
    import os

    return os.environ.get("LRR_STRICT_LIVE", "").strip() in ("1", "true", "True")


def get_gateway(**kwargs) -> Gateway:
    """Return a shared Gateway, creating it on first use."""
    global _GATEWAY
    if _GATEWAY is None or kwargs:
        _GATEWAY = Gateway(**kwargs)
    return _GATEWAY


def reset_gateway() -> None:
    """Drop the shared Gateway (tests)."""
    global _GATEWAY
    _GATEWAY = None
