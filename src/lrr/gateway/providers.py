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


def _secrets_file_exists() -> bool:
    """True if a Streamlit secrets.toml is present in any location Streamlit reads.

    We check this before touching ``st.secrets`` because accessing it with no file
    present makes Streamlit raise and render a visible "No secrets found" error. On
    a keyless public deploy that would spray errors across every page.
    """
    candidates = [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
        "/mount/src/.streamlit/secrets.toml",
    ]
    return any(os.path.isfile(p) for p in candidates)


def _secret_from_streamlit(name: str) -> str | None:
    """Read ``st.secrets[name]`` only when a secrets file exists; silent otherwise."""
    if not _secrets_file_exists():
        return None
    try:  # pragma: no cover - only under Streamlit with a secrets file
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except Exception:
        return None
    return None


def get_secret(name: str) -> str | None:
    """Read a secret from ``st.secrets`` first, then the environment. Never cached."""
    secret = _secret_from_streamlit(name)
    if secret:
        return secret
    value = os.environ.get(name)
    return value or None


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud.

    The platform sets ``HOSTNAME`` to a value that starts with ``streamlit`` and
    exposes ``/mount/src`` as the app root. We also honor an explicit override via
    ``STREAMLIT_CLOUD=1`` so the behavior can be forced in tests or local checks.
    """
    if os.environ.get("STREAMLIT_CLOUD", "").strip() in ("1", "true", "True"):
        return True
    hostname = os.environ.get("HOSTNAME", "")
    return hostname.startswith("streamlit") or os.path.isdir("/mount/src")


def provider_available(provider: str) -> bool:
    """False for VPN-only providers (for example Voyager) on Streamlit Cloud."""
    from lrr import config

    if provider in config.VPN_ONLY_PROVIDERS and is_streamlit_cloud():
        return False
    return True


def has_key(provider: str) -> bool:
    """True if the provider is mock or has a usable key available.

    VPN-only providers (Voyager) report no usable key on Streamlit Cloud so the UI
    and routing both treat them as unavailable there.
    """
    if provider == "mock":
        return True
    if not provider_available(provider):
        return False
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
