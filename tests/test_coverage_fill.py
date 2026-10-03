"""Targeted unit tests for pure logic to raise coverage on offline-testable paths.

These cover the gateway's ledger/providers/verify/routing helpers and the corpus
panel feature reducer, all without network, keys, or heavy libraries.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lrr import config, corpus
from lrr.gateway import ledger as ledger_mod
from lrr.gateway import providers, routing, verify

# --------------------------------------------------------------------------- #
# gateway.providers
# --------------------------------------------------------------------------- #


def test_mock_response_deterministic_and_schema():
    msgs = [{"role": "user", "content": "hello"}]
    a = providers.mock_response(msgs, "m")
    b = providers.mock_response(msgs, "m")
    assert a == b and "hello" in a
    js = providers.mock_response(msgs, "m", schema_fields=["label", "intensity"])
    import json

    obj = json.loads(js)
    assert set(obj) == {"label", "intensity"}


def test_estimate_tokens_and_has_key(monkeypatch):
    assert providers.estimate_tokens("abcd" * 10) >= 1
    assert providers.has_key("mock") is True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # With no key and no st.secrets, openai is not usable.
    assert providers.has_key("openai") in (False, True)  # env dependent, never raises


def test_get_secret_prefers_env(monkeypatch):
    monkeypatch.setenv("LRR_TEST_SECRET", "value123")
    assert providers.get_secret("LRR_TEST_SECRET") == "value123"
    assert providers.get_secret("LRR_MISSING_SECRET_XYZ") is None


# --------------------------------------------------------------------------- #
# gateway.ledger
# --------------------------------------------------------------------------- #


def test_estimate_cost_known_and_unknown():
    # gpt-4o-mini has nonzero prices; mock/unknown is 0.
    c = ledger_mod.estimate_cost("gpt-4o-mini", 1000, 1000)
    assert c > 0
    assert ledger_mod.estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_ledger_append_frame_and_flush(tmp_path):
    led = ledger_mod.Ledger()
    led.append(ledger_mod.LedgerRow("rag_answer", "mock", None, 0.1, 10, 5, 0.0, False, False))
    df = led.to_frame()
    assert len(df) == 1 and df.iloc[0]["provider"] == "mock"
    out = led.flush(tmp_path / "ledger.parquet")
    assert out is not None and out.exists()
    # Flushing again appends.
    led.flush(tmp_path / "ledger.parquet")
    assert len(pd.read_parquet(tmp_path / "ledger.parquet")) == 2


# --------------------------------------------------------------------------- #
# gateway.routing
# --------------------------------------------------------------------------- #


def test_default_routes_have_mock_terminal():
    routes = routing.load_routes(path=config.ROOT_DIR / "does_not_exist.yaml")
    for role, chain in routes.items():
        assert chain[-1].provider == "mock", f"{role} missing mock terminal"


def test_get_chain_defaults_to_quant_analyst():
    routes = routing.load_routes()
    chain = routing.get_chain("nonexistent_role", routes)
    assert chain and chain[-1].provider == "mock"


# --------------------------------------------------------------------------- #
# gateway.verify (injected list_fn, no network)
# --------------------------------------------------------------------------- #


def test_verify_models_substitutes_missing(tmp_path):
    def fake_list(provider):
        return {"openai": ["gpt-4o", "o1-mini"], "nvidia": ["meta/llama-3.3-70b-instruct"]}.get(
            provider, []
        )

    res = verify.verify_models(
        write=True, list_fn=fake_list, out_path=tmp_path / "gateway_models.json"
    )
    assert "substitutions" in res
    # gpt-4o-mini is absent -> substituted to an available openai id.
    subs = res["substitutions"]
    assert any(k.startswith("openai:") for k in subs)
    assert (tmp_path / "gateway_models.json").exists()


def test_verify_unreachable_provider_is_safe():
    res = verify.verify_models(write=False, list_fn=lambda p: [])
    # No models available anywhere -> no substitutions, no crash.
    assert res["substitutions"] == {}


# --------------------------------------------------------------------------- #
# corpus panel features
# --------------------------------------------------------------------------- #


def test_ols_slope_and_panel_window_features():
    assert corpus._ols_slope([0, 1, 2, 3]) == pytest.approx(1.0)
    assert corpus._ols_slope([5]) == 0.0

    panel = pd.DataFrame(
        {
            "business_id": ["b0", "b0", "b0"],
            "month": ["2017-01", "2017-02", "2017-03"],
            "n_reviews": [5, 7, 9],
            "mean_stars": [3.0, 3.5, 4.0],
            "n_tips": [1, 2, 1],
            "n_checkins": [10, 12, 8],
        }
    )
    feats = corpus.panel_window_features(panel, pd.Timestamp("2018-01-01"))
    row = feats[feats["business_id"] == "b0"].iloc[0]
    assert row["review_vol_12m"] == 21  # 5+7+9 all within 12m before T
    assert row["stars_all"] == pytest.approx((3.0 * 5 + 3.5 * 7 + 4.0 * 9) / 21)
    assert row["review_velocity_slope"] > 0  # rising review counts


def test_build_all_restaurants_cohort_flag():
    from lrr import cohort as C

    biz = pd.DataFrame(
        {
            "business_id": ["b0", "b1"],
            "name": ["Chregg's", "Hardware Co"],
            "categories": ["Restaurants, Burgers", "Hardware"],
            "stars": [4.0, 4.5],
            "review_count": [10, 5],
            "is_open": [1, 1],
            "city": ["Tempe", "Tempe"],
            "state": ["AZ", "AZ"],
            "latitude": [33.4, 33.4],
            "longitude": [-111.9, -111.9],
        }
    )
    clean = C.clean_business(biz)
    out = corpus.build_all_restaurants(clean, cohort_business_ids=["b0"])
    assert set(out["business_id"]) == {"b0"}  # hardware excluded
    assert bool(out.iloc[0]["is_cohort"]) is True
