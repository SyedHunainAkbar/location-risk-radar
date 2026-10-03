"""Chart helpers. Every chart title states its takeaway; axes carry units.

Plotly is imported lazily inside each helper. When plotly is unavailable we fall back
to native Streamlit charts so the app still renders. The tier palette is used for any
risk-colored series; no default rainbow.
"""

from __future__ import annotations

import pandas as pd

from app.components import theme


def _px():
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        return px, go
    except ImportError:
        return None, None


def risk_distribution(
    st_container, df: pd.DataFrame, score_col: str, cluster_col: str = "cluster"
) -> None:
    """Risk score distribution by cluster. Title states the takeaway."""
    title = "Risk concentrates differently across the two clusters"
    px, _ = _px()
    if px is None or cluster_col not in df:
        st_container.caption(title)
        st_container.bar_chart(df[[score_col]])
        return
    fig = px.histogram(
        df,
        x=score_col,
        color=cluster_col,
        nbins=30,
        color_discrete_map=theme.CLUSTER_COLORS,
        barmode="overlay",
        opacity=0.75,
        title=title,
    )
    fig.update_layout(
        template=theme.plotly_template(),
        height=360,
        xaxis_title="Risk score",
        yaxis_title="Locations",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def tier_vs_tier(st_container, df: pd.DataFrame) -> None:
    """Tier 1 corpus score vs Tier 2 cohort score, one point per cohort location."""
    title = "Tier 2 refines the Tier 1 corpus signal for the cohort"
    px, _ = _px()
    if px is None or not {"tier1_score", "tier2_score"}.issubset(df.columns):
        st_container.caption(title)
        if {"tier1_score", "tier2_score"}.issubset(df.columns):
            st_container.scatter_chart(df, x="tier1_score", y="tier2_score")
        return
    fig = px.scatter(
        df,
        x="tier1_score",
        y="tier2_score",
        color="cluster" if "cluster" in df else None,
        color_discrete_map=theme.CLUSTER_COLORS,
        title=title,
        hover_data=[c for c in ["business_id", "chain", "risk_tier"] if c in df],
    )
    fig.update_layout(
        template=theme.plotly_template(),
        height=380,
        xaxis_title="Tier 1 corpus score",
        yaxis_title="Tier 2 cohort score",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def survival_curve(
    st_container, times, surv, lo=None, hi=None, landmark_label: str = "landmark"
) -> None:
    """Predicted survival over months with an optional 95% band."""
    title = "Predicted survival declines over the 24-month horizon"
    px, go = _px()
    if go is None:
        st_container.caption(title)
        st_container.line_chart(pd.DataFrame({"survival": surv}, index=times))
        return
    fig = go.Figure()
    if lo is not None and hi is not None:
        fig.add_trace(
            go.Scatter(
                x=list(times) + list(times)[::-1],
                y=list(hi) + list(lo)[::-1],
                fill="toself",
                fillcolor="rgba(33,102,172,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% band",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(times),
            y=list(surv),
            mode="lines",
            line=dict(color=theme.ACCENT, width=2),
            name="survival",
        )
    )
    fig.update_layout(
        template=theme.plotly_template(),
        title=title,
        height=360,
        xaxis_title="Months after landmark",
        yaxis_title="Survival probability",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def shap_bar(st_container, labels, contributions) -> None:
    """Top signed SHAP drivers as a horizontal bar; red raises risk, blue lowers."""
    title = "The strongest drivers of this location's risk, signed"
    px, go = _px()
    colors = [
        theme.TIER_COLORS["High"] if c >= 0 else theme.TIER_COLORS["Low"] for c in contributions
    ]
    if go is None:
        st_container.caption(title)
        st_container.bar_chart(pd.DataFrame({"contribution": contributions}, index=labels))
        return
    fig = go.Figure(
        go.Bar(x=list(contributions), y=list(labels), orientation="h", marker_color=colors)
    )
    fig.update_layout(
        template=theme.plotly_template(),
        title=title,
        height=360,
        xaxis_title="Signed contribution to risk",
        yaxis_title="",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def topic_bar(st_container, labels, sizes, title: str) -> None:
    """Topic sizes with human labels."""
    px, go = _px()
    if go is None:
        st_container.caption(title)
        st_container.bar_chart(pd.DataFrame({"size": sizes}, index=labels))
        return
    fig = go.Figure(
        go.Bar(x=list(sizes), y=list(labels), orientation="h", marker_color=theme.ACCENT)
    )
    fig.update_layout(
        template=theme.plotly_template(),
        title=title,
        height=400,
        xaxis_title="Reviews in topic",
        yaxis_title="",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def ablation_bar(st_container, feature_sets, c_values, lo=None, hi=None) -> None:
    """Ablation: C index rises as engagement and text are added to stars."""
    title = "Adding engagement and text lifts the C index above stars alone"
    px, go = _px()
    if go is None:
        st_container.caption(title)
        st_container.bar_chart(pd.DataFrame({"C index": c_values}, index=feature_sets))
        return
    err = None
    if lo is not None and hi is not None:
        err = dict(
            type="data",
            symmetric=False,
            array=[h - c for h, c in zip(hi, c_values)],
            arrayminus=[c - l for l, c in zip(lo, c_values)],
        )
    fig = go.Figure(
        go.Bar(x=list(feature_sets), y=list(c_values), marker_color=theme.ACCENT, error_y=err)
    )
    fig.update_layout(
        template=theme.plotly_template(),
        title=title,
        height=360,
        xaxis_title="Feature set",
        yaxis_title="Harrell C index",
    )
    st_container.plotly_chart(fig, use_container_width=True)


def calibration_plot(st_container, mean_pred, mean_obs) -> None:
    """Calibration: predicted vs observed; the diagonal is perfect calibration."""
    title = "Predicted risk tracks observed closure, close to the diagonal"
    px, go = _px()
    if go is None:
        st_container.caption(title)
        st_container.line_chart(pd.DataFrame({"observed": mean_obs}, index=mean_pred))
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=theme.MUTED, dash="dash"),
            name="perfect",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(mean_pred),
            y=list(mean_obs),
            mode="markers+lines",
            line=dict(color=theme.ACCENT),
            name="model",
        )
    )
    fig.update_layout(
        template=theme.plotly_template(),
        title=title,
        height=360,
        xaxis_title="Mean predicted",
        yaxis_title="Mean observed",
    )
    st_container.plotly_chart(fig, use_container_width=True)
