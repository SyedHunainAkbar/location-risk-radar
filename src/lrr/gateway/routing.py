"""Role routing: map task roles to ordered provider fallback chains.

Reads ``config/gateway.yaml`` when pyyaml is available, else uses an equivalent
in-code default so the gateway works without the file or the dependency. ``mock`` is
always the terminal fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrr import config

_DEFAULTS = {"temperature": 0.2, "max_tokens": 1024, "timeout": 60}


@dataclass(frozen=True)
class RouteEntry:
    """One hop in a fallback chain."""

    provider: str
    model: str | None
    temperature: float
    max_tokens: int
    timeout: float


#: In-code fallback mirroring config/gateway.yaml (used if the file/pyyaml is absent).
_NV_PRIMARY = "meta/llama-3.2-90b-vision-instruct"
_NV_JUDGE = "meta/llama-3.2-11b-vision-instruct"

DEFAULT_ROUTES: dict[str, list[tuple[str, str | None, dict]]] = {
    "quant_analyst": [
        ("nvidia", _NV_PRIMARY, {"temperature": 0.0}),
        ("mock", None, {"temperature": 0.0}),
    ],
    "voice_of_customer": [
        ("nvidia", _NV_PRIMARY, {"temperature": 0.3}),
        ("mock", None, {"temperature": 0.3}),
    ],
    "risk_auditor": [
        ("nvidia", _NV_JUDGE, {"temperature": 0.0}),
        ("nvidia", _NV_PRIMARY, {"temperature": 0.0}),
        ("mock", None, {"temperature": 0.0}),
    ],
    "topic_labeler": [
        ("nvidia", _NV_PRIMARY, {"temperature": 0.0, "max_tokens": 16}),
        ("mock", None, {"temperature": 0.0, "max_tokens": 16}),
    ],
    "rag_answer": [
        ("nvidia", _NV_PRIMARY, {"temperature": 0.0, "max_tokens": 512}),
        ("mock", None, {"temperature": 0.0, "max_tokens": 512}),
    ],
    "rag_judge": [
        ("nvidia", _NV_JUDGE, {"temperature": 0.0, "max_tokens": 8}),
        ("mock", None, {"temperature": 0.0, "max_tokens": 8}),
    ],
}


def _entry(provider, model, params, defaults) -> RouteEntry:
    merged = {**defaults, **params}
    return RouteEntry(
        provider=provider,
        model=model,
        temperature=float(merged["temperature"]),
        max_tokens=int(merged["max_tokens"]),
        timeout=float(merged["timeout"]),
    )


def _routes_from_defaults() -> dict[str, list[RouteEntry]]:
    out = {}
    for role, entries in DEFAULT_ROUTES.items():
        out[role] = [_entry(p, m, params, _DEFAULTS) for p, m, params in entries]
    return out


def load_routes(path: Path = config.GATEWAY_YAML) -> dict[str, list[RouteEntry]]:
    """Load role -> fallback chain. Falls back to DEFAULT_ROUTES on any problem."""
    path = Path(path)
    try:
        import yaml
    except ImportError:
        return _routes_from_defaults()
    if not path.exists():
        return _routes_from_defaults()
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return _routes_from_defaults()

    defaults = {**_DEFAULTS, **(doc.get("defaults") or {})}
    roles = doc.get("roles") or {}
    out: dict[str, list[RouteEntry]] = {}
    for role, spec in roles.items():
        role_params = {k: v for k, v in (spec or {}).items() if k != "chain"}
        role_defaults = {**defaults, **role_params}
        entries = []
        for item in (spec or {}).get("chain", []):
            provider, _, model = str(item).partition(":")
            entries.append(_entry(provider, model or None, {}, role_defaults))
        if entries:
            out[role] = entries
    return out or _routes_from_defaults()


def get_chain(role: str, routes: dict | None = None) -> list[RouteEntry]:
    """Return the fallback chain for a role, defaulting to quant_analyst."""
    routes = routes or load_routes()
    if role in routes:
        return routes[role]
    return routes.get("quant_analyst") or _routes_from_defaults()["quant_analyst"]
