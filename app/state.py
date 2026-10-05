"""Session state: global filters, Live/Cached mode, and the live-call cap.

Every page reads filters through :func:`apply_filters` and makes live model calls
only through :func:`guarded_chat`, which enforces the per-session cap and falls back
to a cached result with a visible notice if a provider fails. No key is ever shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config as _config  # noqa: E402
from lrr.gateway import providers as _providers  # noqa: E402

#: Per-session live-call cap, sourced from config (LIVE_CALL_CAP secret/env).
LIVE_CALL_CAP = _config.LIVE_CALL_CAP
FILTER_KEYS = ("f_cluster", "f_chain", "f_state", "f_tier")


def init_state() -> None:
    """Seed session defaults once.

    First-time visitors always start in Cached mode, even when a key is present, so
    a public URL never spends on a cold open. Users opt into Live from the sidebar.
    """
    ss = st.session_state
    for k in FILTER_KEYS:
        ss.setdefault(k, [])
    ss.setdefault("live_calls", 0)
    ss.setdefault("gateway_mode", "Cached")


def any_key_present() -> bool:
    """True if any live-provider key is available (env or st.secrets)."""
    return any(_providers.has_key(p) for p in ("openai", "nvidia", "voyager"))


def mode() -> str:
    return st.session_state.get("gateway_mode", "Cached")


def is_live() -> bool:
    """Live only when a key exists, the switch is Live, and the cap is not reached."""
    return (
        mode() == "Live"
        and any_key_present()
        and st.session_state.get("live_calls", 0) < LIVE_CALL_CAP
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the active sidebar filters to a frame that has the filter columns."""
    if df is None or df.empty:
        return df
    out = df
    ss = st.session_state
    if ss.get("f_cluster") and "cluster" in out:
        out = out[out["cluster"].isin(ss["f_cluster"])]
    if ss.get("f_chain") and "chain" in out:
        out = out[out["chain"].isin(ss["f_chain"])]
    if ss.get("f_state") and "state" in out:
        out = out[out["state"].isin(ss["f_state"])]
    if ss.get("f_tier") and "risk_tier" in out:
        out = out[out["risk_tier"].isin(ss["f_tier"])]
    return out


def guarded_chat(messages, role: str, schema=None, cached_fallback: str = ""):
    """Make a live gateway call under the session cap, with a safe fallback.

    Returns a dict: {text, provider, model, cache_hit, fallback_used, latency_s,
    parsed, error}. On the cap or any failure we return the cached fallback text and
    a visible provider-failed notice; the app never crashes and never shows a key.
    """
    from lrr import gateway

    result = {
        "text": cached_fallback,
        "provider": "cached",
        "model": None,
        "cache_hit": True,
        "fallback_used": True,
        "latency_s": 0.0,
        "parsed": None,
        "error": None,
    }
    if not is_live():
        return result
    try:
        gw = st.session_state.get("gateway") or gateway.get_gateway()
        gr = gw.chat(messages, role=role, schema=schema)
        st.session_state["live_calls"] = st.session_state.get("live_calls", 0) + 1
        result.update(
            text=gr.text,
            provider=gr.provider,
            model=gr.model,
            cache_hit=gr.cache_hit,
            fallback_used=gr.fallback_used,
            latency_s=gr.latency_s,
            parsed=gr.parsed,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = type(exc).__name__
        st.warning(f"A provider call failed ({result['error']}). Showing cached evidence instead.")
    return result
