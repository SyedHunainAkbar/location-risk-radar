"""Model-id verification against each provider's /models endpoint.

Before finalizing, we confirm every configured model id exists and substitute an
available id when it does not. The verified lists and substitutions are written to
``artifacts/gateway_models.json``. Unreachable providers never raise; we record only
what we could confirm.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lrr import config
from lrr.gateway import providers, routing
from lrr.gateway.redaction import get_logger

logger = get_logger()


def _available_models(provider: str, list_fn=None) -> list[str]:
    """Return available model ids for a provider, or [] if unreachable."""
    if provider == "mock":
        return ["mock-model"]
    if list_fn is not None:
        try:
            return list(list_fn(provider))
        except Exception:
            return []
    if not providers.has_key(provider):
        return []
    try:
        client = providers.make_client(provider)
        resp = client.models.list()
        return [m.id for m in resp.data]
    except Exception as exc:
        logger.info("could not list models for %s: %s", provider, type(exc).__name__)
        return []


def _pick_substitute(wanted: str, available: list[str]) -> str | None:
    """Choose a replacement id: exact match, else a prefix/substring, else first."""
    if wanted in available:
        return wanted
    stem = wanted.split("/")[-1].split("-")[0].lower()
    for mid in available:
        if stem and stem in mid.lower():
            return mid
    return available[0] if available else None


def verify_models(
    routes: dict | None = None,
    write: bool = True,
    list_fn=None,
    out_path: Path = None,
) -> dict:
    """Verify configured model ids per provider; record the result.

    Returns a dict ``{provider: [available ids], "substitutions": {...},
    "checked_at": iso}``. Never raises on an unreachable provider.
    """
    routes = routes or routing.load_routes()
    wanted: dict[str, set] = {}
    for chain in routes.values():
        for entry in chain:
            if entry.model:
                wanted.setdefault(entry.provider, set()).add(entry.model)

    available: dict[str, list[str]] = {}
    substitutions: dict[str, str] = {}
    for provider, models in wanted.items():
        avail = _available_models(provider, list_fn=list_fn)
        available[provider] = avail
        if not avail:
            continue
        for m in sorted(models):
            if m not in avail:
                sub = _pick_substitute(m, avail)
                if sub and sub != m:
                    substitutions[f"{provider}:{m}"] = f"{provider}:{sub}"

    result = {
        **available,
        "substitutions": substitutions,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    if write:
        out = Path(out_path or (config.ARTIFACTS_DIR / config.GATEWAY_MODELS_FILE))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("wrote verified model list to %s", out.name)
    return result
