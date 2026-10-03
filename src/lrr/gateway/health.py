"""Per-provider health checks with a 10-minute cache.

A provider is healthy when it is mock, or when it has a key and a guarded ``/models``
ping succeeds. Results cache for ``HEALTH_TTL_SECONDS``. ``voyager`` is auto-disabled
(reported unhealthy) when its ping fails, since it needs the ASU VPN.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from lrr import config
from lrr.gateway import providers
from lrr.gateway.redaction import get_logger

logger = get_logger()

# provider -> (healthy_bool, checked_at_epoch)
_CACHE: dict[str, tuple[bool, float]] = {}


def _ping(provider: str) -> bool:
    """Best-effort reachability probe. mock is always up; others need a key + /models."""
    if provider == "mock":
        return True
    if not providers.has_key(provider):
        return False
    try:
        client = providers.make_client(provider)
        client.models.list()
        return True
    except Exception as exc:  # unreachable, auth error, or no network
        logger.info("health check failed for %s: %s", provider, type(exc).__name__)
        return False


def check_provider(
    provider: str,
    now: float | None = None,
    ping: Callable[[str], bool] | None = None,
    ttl: int = config.HEALTH_TTL_SECONDS,
) -> bool:
    """Return provider health, using the cached value within ``ttl`` seconds.

    ``ping`` is injectable for tests. ``voyager`` is auto-disabled on a failed ping.
    """
    if provider == "mock":
        return True  # the mock provider is always available (offline, keyless)
    now = time.time() if now is None else now
    cached = _CACHE.get(provider)
    if cached is not None and (now - cached[1]) < ttl:
        return cached[0]
    probe = ping or _ping
    healthy = bool(probe(provider))
    _CACHE[provider] = (healthy, now)
    if provider == "voyager" and not healthy:
        logger.info("voyager unreachable; auto-disabled for this session.")
    return healthy


def healthy_providers(
    candidates=providers.PROVIDERS,
    ping: Callable[[str], bool] | None = None,
) -> set[str]:
    """Return the set of currently healthy providers (mock always included)."""
    return {p for p in candidates if check_provider(p, ping=ping)}


def reset_cache() -> None:
    """Clear the health cache (used by tests)."""
    _CACHE.clear()
