"""Provider registry and clients for the LLM gateway.

Four OpenAI-compatible providers: openai, nvidia, voyager, and a deterministic mock.
Keys come only from environment variables or ``st.secrets`` and are never logged or
cached. The real client is imported lazily so the package imports without the openai
SDK, and the mock path needs no network.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from typing import Any

from lrr.gateway.redaction import get_logger

logger = get_logger()

Message = dict[str, str]

PROVIDERS: tuple[str, ...] = ("openai", "nvidia", "voyager", "mock")

PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,  # SDK default
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "voyager": "https://openai.rc.asu.edu/v1",
    "mock": None,
}

PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "voyager": "VOYAGER_API_KEY",
}


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, then ``st.secrets``. Never cached."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # pragma: no cover - only under Streamlit
        import streamlit as st

        secret = st.secrets.get(name)  # type: ignore[attr-defined]
        if secret:
            return str(secret)
    except Exception:
        pass
    return None


def has_key(provider: str) -> bool:
    """True if the provider is mock or has a usable key available."""
    if provider == "mock":
        return True
    env = PROVIDER_KEY_ENV.get(provider)
    return bool(env and get_secret(env))


def make_client(provider: str) -> Any:
    """Construct an OpenAI-compatible client for a live provider (lazy import)."""
    if provider == "mock":
        raise ValueError("mock provider has no network client")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("The 'openai' package is required for live providers.") from exc
    key_env = PROVIDER_KEY_ENV[provider]
    api_key = get_secret(key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider} ({key_env}).")
    base_url = PROVIDER_BASE_URLS.get(provider)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# --------------------------------------------------------------------------- #
# Mock provider (deterministic, offline)
# --------------------------------------------------------------------------- #


def mock_response(
    messages: Sequence[Message], model: str, schema_fields: Sequence[str] | None = None
) -> str:
    """Deterministic mock completion.

    Returns a stable digest-tagged echo for normal calls, or a minimal valid JSON
    object when ``schema_fields`` is provided so JSON-mode callers get parseable
    output offline.
    """
    payload = json.dumps(
        {"model": model, "messages": list(messages)}, sort_keys=True, ensure_ascii=False
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    if schema_fields:
        obj = {f: (0 if "intensity" in f or "score" in f else f"mock-{f}") for f in schema_fields}
        return json.dumps(obj)
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    return f"[mock:{digest}] {last_user}".strip()


# --------------------------------------------------------------------------- #
# Token estimation and cost
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) when the API omits usage."""
    return max(1, len(str(text)) // 4)
