"""Python builds the Quant Analyst fact sheet for one location.

The fact sheet is the single source of truth for every number a brief can show. The
Quant Analyst references these values only through placeholders; Python renders them.
No LLM touches these numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_factsheet(
    business_id: str,
    risk_scores: pd.DataFrame,
    shap_drivers: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    """Assemble a flat fact-sheet dict for one location.

    Keys become the placeholders the Quant Analyst may reference, for example
    ``{risk_percentile}`` or ``{velocity_ratio}``. Values are the rendered numbers.

    Args:
        business_id: location id.
        risk_scores: artifacts/risk_scores.parquet rows.
        shap_drivers: optional per-location signed SHAP contributions
            (columns: business_id, driver_key, contribution).
        features: optional feature matrix row for engagement/sentiment values.
    """
    row = risk_scores.loc[risk_scores["business_id"] == business_id]
    fs: dict = {"business_id": business_id}
    if not row.empty:
        r = row.iloc[0]
        fs["risk_tier"] = str(r.get("risk_tier", "Unknown"))
        fs["risk_percentile"] = _fmt(r.get("risk_percentile"))
        fs["surv_12m"] = _fmt(r.get("surv_12m"), pct=True)
        fs["surv_24m"] = _fmt(r.get("surv_24m"), pct=True)
        fs["cluster"] = str(r.get("cluster", ""))
        fs["chain"] = str(r.get("chain", ""))

    # Percentile within chain and cluster (computed by Python).
    if not row.empty and "risk_score" in risk_scores.columns:
        score = (
            float(row.iloc[0]["risk_score"]) if pd.notna(row.iloc[0].get("risk_score")) else np.nan
        )
        for scope_col in ("chain", "cluster"):
            if scope_col in risk_scores.columns and pd.notna(score):
                peers = risk_scores[risk_scores[scope_col] == row.iloc[0][scope_col]]
                pct = (peers["risk_score"] < score).mean() * 100 if len(peers) else np.nan
                fs[f"percentile_within_{scope_col}"] = _fmt(pct)

    if features is not None and not features.empty:
        frow = features.loc[features["business_id"] == business_id]
        if not frow.empty:
            f = frow.iloc[0]
            fs["velocity_ratio"] = _fmt(f.get("review_velocity_ratio"))
            fs["sentiment_slope"] = _fmt(f.get("sentiment_slope"), places=4)
            fs["share_negative"] = _fmt(f.get("share_negative"), pct=True)

    drivers = []
    if shap_drivers is not None and not shap_drivers.empty:
        d = shap_drivers.loc[shap_drivers["business_id"] == business_id]
        d = d.reindex(d["contribution"].abs().sort_values(ascending=False).index)
        for _, dr in d.head(5).iterrows():
            key = str(dr["driver_key"])
            drivers.append({"driver_key": key, "contribution": float(dr["contribution"])})
            fs[f"driver_{key}"] = _fmt(dr["contribution"], places=3)
    # No per-location SHAP table: fall back to the survival model's top-3 drivers
    # (hazard-increasing covariates ranked in stage 05), so the Quant Analyst always
    # explains the model's actual drivers rather than inventing its own.
    if not drivers and not row.empty and "top_3_drivers" in risk_scores.columns:
        raw = row.iloc[0].get("top_3_drivers")
        keys = [k.strip() for k in str(raw).replace("|", ",").split(",") if k.strip()]
        drivers = [{"driver_key": k, "contribution": 1.0} for k in keys[:3] if k != "nan"]
        if drivers:
            fs["model_top_drivers"] = ", ".join(d["driver_key"] for d in drivers)
    fs["top_drivers"] = drivers
    return fs


def _fmt(value, places: int = 2, pct: bool = False) -> str:
    """Format a numeric value as a string for placeholder rendering."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pct:
        return f"{v * 100:.0f}%" if v <= 1.0 else f"{v:.0f}%"
    return f"{v:.{places}f}"
