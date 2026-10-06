"""Reusable UI widgets: KPI tile, risk badge, citation card, agent card.

All visual styling comes from :mod:`app.components.theme` so tiers and cards read the
same on every page. No widget ever prints a secret.
"""

from __future__ import annotations

import html

import streamlit as st

from app.components import theme


def kpi_tile(col, label: str, value, help_text: str = "") -> None:
    """Render a single KPI metric in a column."""
    col.metric(label, value, help=help_text or None)


def risk_badge(tier: str) -> str:
    """Return an HTML pill for a risk tier (use with unsafe_allow_html=True)."""
    color = theme.tier_color(tier)
    return f"<span class='lrr-badge' style='background:{color}'>{html.escape(str(tier))}</span>"


def citation_card(
    container, quote: str, review_text: str, stars=None, date=None, review_id: str = ""
) -> None:
    """Render a citation card with the quote highlighted inside the source review."""
    safe_review = html.escape(str(review_text or ""))
    safe_quote = html.escape(str(quote or ""))
    highlighted = safe_review
    if safe_quote and safe_quote in safe_review:
        highlighted = safe_review.replace(
            safe_quote, f"<span class='lrr-quote'>{safe_quote}</span>"
        )
    meta = []
    if stars is not None and str(stars) != "nan":
        meta.append(f"{stars} stars")
    if date:
        meta.append(str(date)[:10])
    if review_id:
        meta.append(f"[{html.escape(str(review_id))}]")
    meta_line = " | ".join(meta)
    container.markdown(
        f"<div class='lrr-card'><div class='lrr-muted'>{meta_line}</div>"
        f"<div>{highlighted}</div></div>",
        unsafe_allow_html=True,
    )


def agent_card(
    container,
    name: str,
    status: str = "waiting",
    provider: str | None = None,
    model: str | None = None,
    latency: float | None = None,
    cache: bool | None = None,
    detail: str | None = None,
) -> None:
    """Render an agent card with status and the provider/model badge that answered."""
    dot = {"waiting": "⚪", "running": "🟡", "done": "🟢", "failed": "🔴", "skipped": "⚫"}.get(
        status, "⚪"
    )
    lines = [f"<b>{dot} {html.escape(name)}</b>", f"<span class='lrr-muted'>{status.capitalize()}</span>"]
    if provider:
        tag = "cached" if cache else "live"
        lines.append(
            f"<span class='lrr-muted'>{html.escape(provider)}:"
            f"{html.escape(str(model or 'mock'))} ({tag})</span>"
        )
    if latency is not None:
        lines.append(f"<span class='lrr-muted'>{latency:.1f}s</span>")
    if detail:
        lines.append(f"<span class='lrr-muted'>{html.escape(detail)}</span>")
    container.markdown(
        "<div class='lrr-card'>" + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    """Inject the shared card CSS once per page."""
    st.markdown(theme.CARD_CSS, unsafe_allow_html=True)
