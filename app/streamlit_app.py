"""Location Risk Radar: production Streamlit entry point.

Reads only precomputed artifacts and calls models only through lrr.gateway. Works
with no API keys (Cached mode) and with keys (Live mode). Every page shows the shared
sidebar (global filters + LLM Gateway panel). Copy is concise, first person plural,
no em dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the app package and src importable whether run from repo root or app/.
_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="Location Risk Radar", page_icon="📍", layout="wide")

from app import sidebar  # noqa: E402
from app.components import widgets  # noqa: E402
from app.pages import (  # noqa: E402
    ask_the_reviews,
    complaint_themes,
    location_deep_dive,
    methodology,
    model_lab,
    portfolio_radar,
    risk_committee,
)

TAGLINE = "Early warning for restaurant locations, explained by customer evidence"


def _page(module, title: str):
    """Wrap a page module's render() with the shared header and sidebar."""

    def _run():
        widgets.inject_css()
        st.title("Location Risk Radar")
        st.caption(TAGLINE)
        sidebar.render()
        module.render()

    _run.__name__ = title.replace(" ", "_").lower()
    return _run


PAGES = [
    (_page(portfolio_radar, "Portfolio Radar"), "Portfolio Radar", "📡"),
    (_page(location_deep_dive, "Location Deep Dive"), "Location Deep Dive", "🔎"),
    (_page(risk_committee, "Risk Committee"), "Risk Committee", "🧑‍⚖️"),
    (_page(complaint_themes, "Complaint Themes"), "Complaint Themes", "🗣️"),
    (_page(ask_the_reviews, "Ask the Reviews"), "Ask the Reviews", "💬"),
    (_page(model_lab, "Model Lab"), "Model Lab", "🧪"),
    (_page(methodology, "Methodology"), "Methodology and Limitations", "📘"),
]


def main() -> None:
    pages = [st.Page(fn, title=title, icon=icon) for fn, title, icon in PAGES]
    nav = st.navigation(pages)
    nav.run()


main()
