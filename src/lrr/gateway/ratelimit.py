"""Per-provider token-bucket rate limiter.

A simple monotonic token bucket caps requests per second per provider. ``acquire``
sleeps just enough to stay under the limit. The sleep function is injectable so tests
run without real waiting.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from lrr import config


class TokenBucket:
    """Monotonic token bucket: ``rps`` tokens per second, capacity ``rps``."""

    def __init__(
        self,
        rps: float,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rps = max(rps, 0.0001)
        self.capacity = max(rps, 1.0)
        self.tokens = self.capacity
        self._sleep = sleep
        self._clock = clock
        self._last = clock()

    def acquire(self, amount: float = 1.0) -> None:
        self._refill()
        if self.tokens < amount:
            deficit = amount - self.tokens
            self._sleep(deficit / self.rps)
            self._refill()
        self.tokens = max(0.0, self.tokens - amount)

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rps)


class RateLimiter:
    """Holds one bucket per provider."""

    def __init__(
        self,
        rps: float = config.GATEWAY_RATE_LIMIT_RPS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rps = rps
        self._sleep = sleep
        self._buckets: dict[str, TokenBucket] = {}

    def acquire(self, provider: str) -> None:
        if provider == "mock":
            return  # mock is local and free
        bucket = self._buckets.get(provider)
        if bucket is None:
            bucket = TokenBucket(self.rps, sleep=self._sleep)
            self._buckets[provider] = bucket
        bucket.acquire()
