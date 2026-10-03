"""Streamlit sidebar helper for the LLM Gateway panel.

Renders provider health dots, a Live/Cached mode switch, and exposes a badge helper
so each response can show which provider and model answered. Streamlit is imported
lazily so the gateway package imports without it.
"""

from __future__ import annotations

from lrr.gateway import health
from lrr.gateway.core import GatewayResult, get_gateway

_HEALTH_DOT = {True: "🟢", False: "🔴"}


def render_sidebar(st, providers=("openai", "nvidia", "voyager", "mock")) -> str:
    """Render the LLM Gateway sidebar panel. Returns the selected mode.

    Shows a health dot per provider, a Live/Cached radio (Cached forces cache-only
    and the mock terminal), and stores the mode and a shared ledger in session state.
    """
    st.sidebar.markdown("### LLM Gateway")
    for p in providers:
        ok = health.check_provider(p)
        st.sidebar.write(f"{_HEALTH_DOT[ok]} {p}")

    mode = st.sidebar.radio(
        "Mode",
        ["Live", "Cached"],
        index=0,
        help="Cached serves precomputed answers with no API key.",
    )
    cached_only = mode == "Cached"

    gw = get_gateway(cached_only=cached_only)
    st.session_state["gateway"] = gw
    st.session_state["gateway_mode"] = mode
    return mode


def response_badge(st, result: GatewayResult) -> None:
    """Show a small badge naming the provider and model that answered."""
    tag = "cached" if result.cache_hit else ("fallback" if result.fallback_used else "live")
    st.caption(
        f"answered by **{result.provider}:{result.model or 'mock'}** "
        f"({tag}, {result.latency_s:.2f}s)"
    )


def ledger_summary(st) -> None:
    """Render a compact ledger table from session state, if present."""
    gw = st.session_state.get("gateway")
    if gw is None or not gw.ledger.rows:
        return
    st.sidebar.markdown("**Usage this session**")
    df = gw.ledger.to_frame()
    cols = [
        c
        for c in [
            "role",
            "provider",
            "model",
            "latency_s",
            "cache_hit",
            "fallback_used",
            "est_cost_usd",
        ]
        if c in df.columns
    ]
    st.sidebar.dataframe(df[cols], hide_index=True)
