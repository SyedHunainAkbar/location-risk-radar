"""Location Deep Dive: one location's risk, drivers, trajectories, and themes."""

from __future__ import annotations

import streamlit as st

from app import data, state
from app.components import charts, widgets


def render() -> None:
    st.header("Location Deep Dive")
    st.caption(
        "We open one location to show its risk, drivers, trajectories, and complaint themes."
    )

    risk = data.require("risk_scores")
    if risk is None:
        return
    view = state.apply_filters(risk)
    if view.empty:
        st.info("No locations match the current filters.")
        return

    # Picker searchable by chain and city.
    labels = {
        r["business_id"]: f"{r['business_id']} | {r.get('chain', '')} | {r.get('cluster', '')}"
        for _, r in view.iterrows()
    }
    bid = st.selectbox(
        "Location", view["business_id"].tolist(), format_func=lambda b: labels.get(b, b)
    )
    row = view[view["business_id"] == bid].iloc[0]

    # Risk header.
    st.markdown(widgets.risk_badge(row.get("risk_tier", "Unknown")), unsafe_allow_html=True)
    k = st.columns(4)
    score_col = next(
        (c for c in ("final_score", "tier2_score", "risk_score") if c in view.columns), None
    )
    if score_col:
        widgets.kpi_tile(k[0], "Risk score", f"{row[score_col]:.3f}")
    if "risk_percentile" in row:
        widgets.kpi_tile(k[1], "Percentile", f"{row['risk_percentile']:.0f}")
    if "surv_12m" in row:
        widgets.kpi_tile(k[2], "12m survival", f"{row['surv_12m'] * 100:.0f}%")
    if "surv_24m" in row:
        widgets.kpi_tile(k[3], "24m survival", f"{row['surv_24m'] * 100:.0f}%")

    # Survival curve.
    st.subheader("Predicted survival")
    curves = data.load("survival_curves")
    if curves is not None and "business_id" in curves:
        cu = curves[curves["business_id"] == bid].sort_values("t_days")
        if len(cu):
            lo = cu["survival"] * 0.95 if "survival" in cu else None
            hi = (cu["survival"] * 1.05).clip(upper=1.0) if "survival" in cu else None
            charts.survival_curve(st, cu["t_days"] / 30.4, cu["survival"], lo, hi)
        else:
            st.caption("No survival curve stored for this location.")
    else:
        st.info(data.missing_message("survival_curves"))

    # SHAP drivers.
    st.subheader("Top drivers")
    shap = data.load("shap_drivers") if "shap_drivers" in data.ARTIFACTS else None
    try:
        shap = data._cached_read(str(data.config.ARTIFACTS_DIR / "shap_drivers.parquet"))
    except Exception:
        shap = None
    if shap is not None and {"business_id", "driver_key", "contribution"}.issubset(shap.columns):
        d = shap[shap["business_id"] == bid]
        d = d.reindex(d["contribution"].abs().sort_values(ascending=False).index).head(5)
        if len(d):
            charts.shap_bar(st, [_plain(k) for k in d["driver_key"]], d["contribution"].tolist())
        else:
            st.caption("No SHAP drivers stored for this location.")
    elif "top_3_drivers" in row:
        st.write("Top drivers: ", row["top_3_drivers"])
    else:
        st.caption("SHAP drivers appear once pipeline/05_survival.py writes them.")

    # Voice-of-Customer issues and topic shares vs peers.
    st.subheader("Customer issues and complaint topics")
    _issues_and_topics(bid, row)

    # TA-adapted aspect scores.
    st.subheader("Aspect scores (food, service, ambience)")
    st.caption(
        "Adapted from the TA's Yelp Aspect Agents reference, benchmarked "
        "against stars. These are a separate diagnostic, not the risk model."
    )
    aspect = data.load("aspect_eval")
    if aspect is not None:
        st.dataframe(aspect.round(4), hide_index=True, use_container_width=True)
    else:
        st.info(data.missing_message("aspect_eval"))


def _issues_and_topics(bid, row) -> None:
    shares = data.load("topic_shares")
    if shares is not None and "business_id" in shares:
        s = shares[shares["business_id"] == bid]
        topic_cols = [c for c in shares.columns if c.startswith("topic_")]
        if len(s) and topic_cols:
            melted = s[topic_cols].T.reset_index()
            melted.columns = ["topic", "share"]
            charts.topic_bar(
                st, melted["topic"], melted["share"], "This location's complaint-topic shares"
            )
            return
    st.caption("Topic shares appear once pipeline/03_topics_colab.py has run.")


def _plain(driver_key: str) -> str:
    """Map a feature key to a short plain-English label."""
    mapping = {
        "review_velocity_slope": "review velocity trend",
        "review_velocity_ratio": "recent vs prior review volume",
        "sentiment_mean": "average sentiment",
        "sentiment_slope": "sentiment trend",
        "share_negative": "share of negative reviews",
        "checkin_slope": "check-in trend",
        "stars_trend": "star-rating trend",
        "chain_prior": "chain closure prior",
        "location_age_days": "location age",
    }
    return mapping.get(str(driver_key), str(driver_key).replace("_", " "))
