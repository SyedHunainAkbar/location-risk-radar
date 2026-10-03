"""Disk cache for LLM responses, keyed by a hash of the call.

The key is ``sha256(provider, model, messages, params)``; secrets never enter the
key or the stored value. Entries are small JSON files under ``.cache/llm``
(gitignored).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from lrr import config

Message = dict[str, str]


def cache_key(provider: str, model: str | None, messages: Sequence[Message], params: dict) -> str:
    """Deterministic cache key. ``params`` must not contain secrets."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "messages": list(messages),
            "params": {k: params[k] for k in sorted(params)},
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(key: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{key}.json"


def get(key: str, cache_dir: Path = None) -> dict | None:
    """Return the cached value for ``key`` or None."""
    cache_dir = config.LLM_CACHE_DIR if cache_dir is None else cache_dir
    path = _path(key, cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def put(key: str, value: dict, cache_dir: Path = None) -> None:
    """Write ``value`` (JSON-serializable, no secrets) under ``key``."""
    cache_dir = config.LLM_CACHE_DIR if cache_dir is None else cache_dir
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _path(key, cache_dir).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
