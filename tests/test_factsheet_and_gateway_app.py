"""Unit tests for the Quant Analyst fact sheet and the gateway sidebar helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lrr.agents.factsheet import _fmt, build_factsheet
from lrr.gateway import app as gw_app
from lrr.gateway.core import GatewayResult

# --------------------------------------------------------------------------- #
# Fact sheet
# --------------------------------------------------------------------------- #


def _risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2"],
            "chain": ["Subway", "Subway", "Chili's"],
            "cluster": ["Fast Food", "Fast Food", "Casual"],
            "risk_score": [0.9, 0.4, 0.7],
            "risk_percentile": [95.0, 40.0, 70.0],
            "risk_tier": ["High", "Watch", "Elevated"],
            "surv_12m": [0.6, 0.8, 0.7],
            "surv_24m": [0.4, 0.7, 0.5],
            "top_3_drivers": ["velocity, sentiment | stars", "nan", None],
        }
    )


def test_factsheet_core_fields_and_peer_percentiles():
    fs = build_factsheet("b0", _risk())
    assert fs["risk_tier"] == "High"
    assert fs["risk_percentile"] == "95.00"
    assert fs["surv_12m"] == "60%"
    assert fs["chain"] == "Subway"
    # b0 outranks the other Subway location.
    assert fs["percentile_within_chain"] == "50.00"
    assert fs["percentile_within_cluster"] == "50.00"


def test_factsheet_falls_back_to_model_top_drivers():
    fs = build_factsheet("b0", _risk())
    keys = [d["driver_key"] for d in fs["top_drivers"]]
    assert keys == ["velocity", "sentiment", "stars"]
    assert fs["model_top_drivers"] == "velocity, sentiment, stars"


def test_factsheet_ignores_missing_model_drivers():
    fs = build_factsheet("b1", _risk())
    assert fs["top_drivers"] == []
    assert "model_top_drivers" not in fs


def test_factsheet_uses_shap_and_features_when_given():
    shap = pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b1"],
            "driver_key": ["velocity", "stars", "velocity"],
            "contribution": [-0.2, 0.5, 0.1],
        }
    )
    feats = pd.DataFrame(
        {
            "business_id": ["b0"],
            "review_velocity_ratio": [0.7],
            "sentiment_slope": [-0.01234],
            "share_negative": [0.25],
        }
    )
    fs = build_factsheet("b0", _risk(), shap_drivers=shap, features=feats)
    # Ranked by absolute contribution.
    assert [d["driver_key"] for d in fs["top_drivers"]] == ["stars", "velocity"]
    assert fs["driver_stars"] == "0.500"
    assert fs["velocity_ratio"] == "0.70"
    assert fs["sentiment_slope"] == "-0.0123"
    assert fs["share_negative"] == "25%"
    assert "model_top_drivers" not in fs


def test_factsheet_unknown_location_is_minimal():
    fs = build_factsheet("zzz", _risk(), features=pd.DataFrame({"business_id": ["b0"]}))
    assert fs == {"business_id": "zzz", "top_drivers": []}


def test_fmt_edge_cases():
    assert _fmt(None) == "n/a"
    assert _fmt(np.nan) == "n/a"
    assert _fmt("abc") == "abc"
    assert _fmt(0.256, pct=True) == "26%"
    assert _fmt(42.0, pct=True) == "42%"
    assert _fmt(1.23456, places=3) == "1.235"


# --------------------------------------------------------------------------- #
# Gateway sidebar helpers (fake Streamlit, no network)
# --------------------------------------------------------------------------- #


class _FakeSidebar:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls: list = []

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def write(self, text):
        self.calls.append(("write", text))

    def radio(self, label, options, index=0, help=None):
        return self.mode

    def dataframe(self, df, hide_index=True):
        self.calls.append(("dataframe", list(df.columns)))


class _FakeSt:
    def __init__(self, mode: str = "Cached"):
        self.sidebar = _FakeSidebar(mode)
        self.session_state: dict = {}
        self.captions: list = []

    def caption(self, text):
        self.captions.append(text)


def test_render_sidebar_sets_mode_and_gateway(monkeypatch):
    seen = {}
    monkeypatch.setattr(gw_app.health, "check_provider", lambda p: p == "mock")
    monkeypatch.setattr(
        gw_app, "get_gateway", lambda cached_only: seen.setdefault("cached", cached_only) or "gw"
    )
    st = _FakeSt("Cached")
    mode = gw_app.render_sidebar(st, providers=("openai", "mock"))
    assert mode == "Cached"
    assert seen["cached"] is True
    assert st.session_state["gateway_mode"] == "Cached"
    writes = [t for k, t in st.sidebar.calls if k == "write"]
    assert writes == ["🔴 openai", "🟢 mock"]


def test_response_badge_tags():
    st = _FakeSt()
    base = {"text": "x", "provider": "nvidia", "model": "llama", "role": "rag_answer"}
    gw_app.response_badge(st, GatewayResult(**base, cache_hit=True))
    gw_app.response_badge(st, GatewayResult(**base, fallback_used=True))
    gw_app.response_badge(st, GatewayResult(**{**base, "model": None}, latency_s=1.5))
    assert "(cached" in st.captions[0]
    assert "(fallback" in st.captions[1]
    assert "mock" in st.captions[2] and "(live, 1.50s)" in st.captions[2]


def test_ledger_summary_renders_known_columns():
    st = _FakeSt()
    gw_app.ledger_summary(st)  # no gateway: nothing rendered
    assert st.sidebar.calls == []

    frame = pd.DataFrame({"role": ["rag_answer"], "provider": ["mock"], "extra": [1]})
    ledger = SimpleNamespace(rows=[1], to_frame=lambda: frame)
    st.session_state["gateway"] = SimpleNamespace(ledger=ledger)
    gw_app.ledger_summary(st)
    assert ("dataframe", ["role", "provider"]) in st.sidebar.calls


def test_ledger_summary_skips_empty_ledger():
    st = _FakeSt()
    st.session_state["gateway"] = SimpleNamespace(ledger=SimpleNamespace(rows=[]))
    gw_app.ledger_summary(st)
    assert st.sidebar.calls == []
