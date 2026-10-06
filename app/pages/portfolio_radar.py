"""Portfolio Radar: rank locations by risk with Tier 1 context and Tier 2 scores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data, state
from app.components import charts, widgets


def render() -> None:
    st.header("Portfolio Radar")
    st.caption(
        "We rank locations by closure risk, with Tier 1 corpus context and "
        "the Tier 2 cohort scores."
    )

    risk = data.require("risk_scores")
    if risk is None:
        return
    view = state.apply_filters(risk)

    score_col = next(
        (c for c in ("final_score", "tier2_score", "risk_score") if c in view.columns), None
    )

    # KPI tiles.
    cols = st.columns(5)
    widgets.kpi_tile(
        cols[0],
        "Locations monitored",
        f"{len(view):,}",
        "Tier 2 cohort locations in the current filter.",
    )
    if "risk_tier" in view:
        widgets.kpi_tile(cols[1], "High risk", f"{int((view['risk_tier'] == 'High').sum())}")
        widgets.kpi_tile(
            cols[2], "Elevated risk", f"{int((view['risk_tier'] == 'Elevated').sum())}"
        )
    if score_col and "cluster" in view and len(view):
        med = view.groupby("cluster")[score_col].median()
        widgets.kpi_tile(cols[3], "Median risk (FF)", f"{med.get('Fast Food', float('nan')):.2f}")
    # Tier 1 context: all-restaurant base closure rate within 24 months.
    base_rate = _tier1_base_rate()
    widgets.kpi_tile(
        cols[4],
        "Tier 1 base closure rate",
        base_rate,
        "All-restaurant 24-month closure base rate (Tier 1).",
    )

    _cohort_coverage(risk)

    # Map colored by tier.
    st.subheader("Where the risk is")
    _risk_map(view)

    # Sortable table + CSV.
    st.subheader("Watchlist")
    table = _watchlist_table(view, score_col)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_watchlist.csv",
        mime="text/csv",
    )

    # Distribution + Tier 1 vs Tier 2.
    st.subheader("Risk distribution and the two tiers")
    c1, c2 = st.columns(2)
    if score_col:
        charts.risk_distribution(c1, view, score_col)
    charts.tier_vs_tier(c2, view)


def _tier1_base_rate() -> str:
    metrics = data.load("model_metrics")
    # If a tier1 corpus event rate was recorded, show it; else a documented default.
    if metrics is not None and "tier" in metrics.columns:
        t1 = metrics[metrics["tier"] == "tier1"]
        if "base_rate" in t1.columns and len(t1):
            return f"{float(t1['base_rate'].iloc[0]) * 100:.1f}%"
    return "see Model Lab"


def _watchlist_table(view: pd.DataFrame, score_col) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "business_id",
            "chain",
            "cluster",
            "risk_tier",
            score_col,
            "risk_percentile",
            "top_3_drivers",
            "surv_24m",
        ]
        if c and c in view.columns
    ]
    out = view[cols].copy()
    if score_col:
        out = out.sort_values(score_col, ascending=False)
    rename = {
        "business_id": "location",
        score_col: "final_score",
        "top_3_drivers": "top 3 drivers",
        "surv_24m": "24m survival",
    }
    return out.rename(columns=rename).head(300)


def _risk_map(view: pd.DataFrame) -> None:
    loc = data.load("cohort_locations")
    if loc is None or not {"lat", "lon"}.issubset(loc.columns):
        st.caption("Map needs cohort_locations with lat/lon. Run pipeline/01_ingest_cohort.py.")
        return
    merged = loc.merge(
        view[[c for c in ["business_id", "risk_tier", "risk_percentile"] if c in view.columns]],
        on="business_id",
        how="inner",
    )
    if merged.empty:
        st.caption("No mapped locations in the current filter.")
        return
    from app.components import theme

    def _rgb(tier):
        hexv = theme.tier_color(tier).lstrip("#")
        return [int(hexv[i : i + 2], 16) for i in (0, 2, 4)] + [160]

    merged["color"] = merged.get("risk_tier", "Low").map(_rgb)
    try:
        import pydeck as pdk

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=merged,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=600,
            pickable=True,
        )
        tooltip = {"text": "{chain}\n{city}, {state}\nTier: {risk_tier}"}
        view_state = pdk.ViewState(
            latitude=float(merged["lat"].mean()), longitude=float(merged["lon"].mean()), zoom=3.2
        )
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None)
        )
    except Exception:
        st.map(merged.rename(columns={"lat": "latitude", "lon": "longitude"}))


def _cohort_coverage(risk: pd.DataFrame) -> None:
    """Show that every one of the EDA's 2,054 cohort locations is accounted for.

    Locations with no review, tip, or check-in activity in the 6 months before the
    2018-01-01 landmark are outside the prediction window. Scoring them would leak
    the outcome (a location that has already gone quiet is mostly already closed),
    so they are excluded from training and shown here with their observed status.
    """
    loc = data.load("cohort_locations")
    if loc is None:
        return
    scored = set(risk["business_id"])
    loc = loc.copy()
    loc["eligibility"] = [
        "Scored (active at landmark)" if b in scored else "Inactive before landmark"
        for b in loc["business_id"]
    ]
    n_all, n_scored = len(loc), int(loc["eligibility"].str.startswith("Scored").sum())
    with st.expander(
        f"EDA cohort coverage: all {n_all:,} locations of the 30 chains "
        f"({n_scored:,} scored, {n_all - n_scored:,} inactive before the landmark)"
    ):
        st.caption(
            "The survival model scores locations still active in the 6 months before "
            "2018-01-01. The rest had already gone quiet, so scoring them would leak the "
            "outcome; we list them with their observed status instead."
        )
        inactive = loc[loc["eligibility"] != "Scored (active at landmark)"].copy()
        inactive["observed status"] = inactive["is_open"].map({1: "Open", 0: "Closed"})
        summary = (
            loc.groupby(["cluster", "eligibility"]).size().unstack(fill_value=0).reset_index()
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.dataframe(
            inactive[["business_id", "chain", "cluster", "city", "state", "observed status"]],
            hide_index=True,
            use_container_width=True,
        )
