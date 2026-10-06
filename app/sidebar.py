"""Shared sidebar: global filters and the LLM Gateway panel.

Rendered on every page. Filters persist in session state; the gateway panel shows
provider health, the Live/Cached switch (Live disabled with no keys), the session
request counter against the hard cap, and token and cost totals from the ledger.
"""

from __future__ import annotations

import streamlit as st

from app import data, state


def _options(df, col):
    if df is None or col not in df:
        return []
    return sorted(x for x in df[col].dropna().unique().tolist())


def render() -> None:
    state.init_state()
    cohort = data.locations()

    st.sidebar.markdown("## Filters")
    st.sidebar.multiselect("Cluster", _options(cohort, "cluster"), key="f_cluster")
    st.sidebar.multiselect("Chain", _options(cohort, "chain"), key="f_chain")
    st.sidebar.multiselect("State", _options(cohort, "state"), key="f_state")
    st.sidebar.multiselect("Risk tier", ["High", "Elevated", "Watch", "Low", "Not scored"], key="f_tier")

    st.sidebar.divider()
    _gateway_panel()


def _gateway_panel() -> None:
    from lrr import config, gateway
    from lrr.gateway import health
    from lrr.gateway import providers as _providers

    st.sidebar.markdown("## LLM Gateway")
    keys = state.any_key_present()
    if not keys:
        st.sidebar.caption("No API key detected. Serving cached results.")

    dot = {True: "🟢", False: "🔴"}
    name = {"openai": "OpenAI", "nvidia": "NVIDIA", "voyager": "Voyager", "mock": "Mock"}
    for p in ("openai", "nvidia", "voyager", "mock"):
        if not _providers.provider_available(p):
            # VPN-only provider on Streamlit Cloud: show it as disabled with a note.
            st.sidebar.write(f"⚪ {name[p]} (disabled)")
            st.sidebar.caption("Voyager requires the ASU VPN and is available in local runs only.")
            continue
        try:
            ok = health.check_provider(p)
        except Exception:
            ok = False
        st.sidebar.write(f"{dot[ok]} {name[p]}")

    # Live disabled when no keys.
    options = ["Live", "Cached"] if keys else ["Cached"]
    default = 0 if (keys and state.mode() == "Live") else (0 if not keys else 1)
    choice = st.sidebar.radio(
        "Mode",
        options,
        index=min(default, len(options) - 1),
        help="Cached serves precomputed results with no key.",
    )
    st.session_state["gateway_mode"] = choice if keys else "Cached"
    st.session_state["gateway"] = gateway.get_gateway(
        cached_only=(st.session_state["gateway_mode"] == "Cached")
    )

    used = st.session_state.get("live_calls", 0)
    st.sidebar.progress(
        min(used / state.LIVE_CALL_CAP, 1.0), text=f"Live calls: {used} / {state.LIVE_CALL_CAP}"
    )
    if used >= state.LIVE_CALL_CAP:
        st.sidebar.caption("Session cap reached. Live calls are paused.")

    # Token and cost totals from the ledger (never a key).
    gw = st.session_state["gateway"]
    if gw.ledger.rows:
        df = gw.ledger.to_frame()
        tok = int(df.get("prompt_tokens", 0).sum() + df.get("completion_tokens", 0).sum())
        cost = float(df.get("est_cost_usd", 0).sum())
        st.sidebar.caption(f"Tokens this session: {tok:,} | Est. cost: ${cost:.4f}")

    # Global daily spend guard (shared across all visitors in this app process).
    budget = config.GATEWAY_DAILY_BUDGET_USD
    if budget:
        spent = gateway.DAILY_BUDGET.spent_today
        st.sidebar.progress(
            min(spent / budget, 1.0),
            text=f"Daily budget: ${spent:.4f} / ${budget:.2f}",
        )
        if gateway.DAILY_BUDGET.exceeded():
            st.sidebar.caption("Daily budget reached. Serving cached results for everyone today.")
