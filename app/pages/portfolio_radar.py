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
    allv = state.apply_filters(data.locations())
    view = allv[allv["eligibility"].str.startswith("Scored")]

    score_col = next(
        (c for c in ("final_score", "tier2_score", "risk_score") if c in view.columns), None
    )

    # KPI tiles.
    cols = st.columns(5)
    widgets.kpi_tile(
        cols[0],
        "Locations in EDA cohort",
        f"{len(allv):,}",
        f"{len(view):,} scored, {len(allv) - len(view):,} inactive before the 2018 landmark.",
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
    _risk_map(allv)

    # Sortable table + CSV.
    st.subheader("Watchlist")
    table = _watchlist_table(allv, score_col)
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
            "chain",
            "city",
            "state",
            "cluster",
            "risk_tier",
            score_col,
            "risk_percentile",
            "top_3_drivers",
            "surv_24m",
            "observed_status",
            "business_id",
        ]
        if c and c in view.columns
    ]
    out = view[cols].copy()
    if score_col:
        out = out.sort_values(score_col, ascending=False, na_position="last")
    rename = {
        "business_id": "yelp id",
        "observed_status": "observed status",
        score_col: "final_score",
        "top_3_drivers": "top 3 drivers",
        "surv_24m": "24m survival",
    }
    return out.rename(columns=rename)


def _risk_map(view: pd.DataFrame) -> None:
    """One dot per location, colored and sized by risk tier, zoomed to the filter."""
    if not {"lat", "lon"}.issubset(view.columns):
        st.caption("Map needs cohort_locations with lat/lon.")
        return
    merged = view.dropna(subset=["lat", "lon"]).copy()
    if merged.empty:
        st.info(data.empty_filter_message(data.locations()))
        return
    from app.components import theme

    order = {"High": 4, "Elevated": 3, "Watch": 2, "Low": 1, "Not scored": 0}
    size = {"High": 9, "Elevated": 7, "Watch": 5, "Low": 4, "Not scored": 3}

    def _rgb(tier):
        if tier not in theme.TIER_COLORS:
            return [150, 150, 150, 120]
        hexv = theme.tier_color(tier).lstrip("#")
        return [int(hexv[i : i + 2], 16) for i in (0, 2, 4)] + [230]

    merged["color"] = merged["risk_tier"].map(_rgb)
    merged["px"] = merged["risk_tier"].map(size).fillna(3)
    merged["score_txt"] = merged.get("risk_score", pd.Series(index=merged.index)).map(
        lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
    )
    # Draw High last so the riskiest dots sit on top.
    merged = merged.sort_values("risk_tier", key=lambda s: s.map(order))
    st.caption(
        "Each dot is one location, colored by risk tier: "
        "\U0001F534 High, \U0001F7E0 Elevated, \U0001F7E1 Watch, \U0001F535 Low, "
        "\u26AA not scored (inactive before 2018). Larger dots are higher risk. "
        "Hover for chain, city, tier and score; use the sidebar to zoom into a state."
    )
    try:
        import pydeck as pdk

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=merged[["lon", "lat", "color", "px", "chain", "city", "state",
                         "cluster", "risk_tier", "score_txt"]],
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="px",
            radius_units="pixels",
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0.5,
            pickable=True,
        )
        tooltip = {"text": "{chain} ({cluster})\n{city}, {state}\nTier: {risk_tier}\nScore: {score_txt}"}
        lat0, lat1 = merged["lat"].quantile([0.02, 0.98])
        lon0, lon1 = merged["lon"].quantile([0.02, 0.98])
        span = max(lat1 - lat0, (lon1 - lon0) / 1.6, 0.05)
        zoom = float(max(2.5, min(11, 8.5 - __import__("math").log2(span))))
        view_state = pdk.ViewState(
            latitude=float((lat0 + lat1) / 2), longitude=float((lon0 + lon1) / 2), zoom=zoom
        )
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None)
        )
    except Exception:
        st.map(merged.rename(columns={"lat": "latitude", "lon": "longitude"}))
    tiers = [t for t in ["High", "Elevated", "Watch", "Low", "Not scored"]
             if t in set(merged["risk_tier"])]
    by_state = merged.groupby(["state", "risk_tier"]).size().unstack(fill_value=0)[tiers]
    if "High" in by_state:
        by_state = by_state.sort_values("High", ascending=False)
    st.caption("Locations by state and tier (current filter).")
    st.dataframe(by_state, use_container_width=True)


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
