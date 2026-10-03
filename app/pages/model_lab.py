"""Model Lab: Phase A (Tier 1) vs Phase B (Tier 2), ablations, and agent evaluation."""

from __future__ import annotations

import streamlit as st

from app import data
from app.components import charts


def render() -> None:
    st.header("Model Lab")
    st.caption(
        "We compare the corpus model (Tier 1) and the cohort model (Tier 2), "
        "prove the lift from engagement and text, and report agent quality."
    )

    metrics = data.require("model_metrics")

    # --- Phase A vs Phase B metrics with CIs ---
    if metrics is not None:
        st.subheader("Phase A (corpus) vs Phase B (cohort)")
        st.caption(
            "Concordance and discrimination with 95% confidence intervals; "
            "higher is better except the Brier score."
        )
        show = metrics.copy()
        display_cols = [
            c
            for c in [
                "tier",
                "scope",
                "feature_set",
                "model",
                "harrell_c",
                "harrell_c_lo",
                "harrell_c_hi",
                "uno_c",
                "auc_12m",
                "auc_24m",
                "ibs",
            ]
            if c in show.columns
        ]
        st.dataframe(
            show[display_cols].round(4) if display_cols else show.round(4),
            hide_index=True,
            use_container_width=True,
        )

        # --- Ablation chart ---
        st.subheader("Ablation: the value of engagement and text over stars")
        order = ["stars_only", "engagement", "engagement_text", "engagement_sentiment", "full"]
        abl = show[show["feature_set"].isin(order)] if "feature_set" in show else show
        if "feature_set" in abl and "harrell_c" in abl and len(abl):
            agg = (
                abl.groupby("feature_set")
                .agg(
                    c=("harrell_c", "mean"),
                    lo=("harrell_c_lo", "mean") if "harrell_c_lo" in abl else ("harrell_c", "mean"),
                    hi=("harrell_c_hi", "mean") if "harrell_c_hi" in abl else ("harrell_c", "mean"),
                )
                .reindex([o for o in order if o in set(abl["feature_set"])])
            )
            charts.ablation_bar(
                st, agg.index.tolist(), agg["c"].tolist(), agg["lo"].tolist(), agg["hi"].tolist()
            )
    else:
        st.caption("Run pipeline/05_survival.py and 05b_tier1_survival.py for metrics.")

    # --- Calibration + temporal + COVID notes ---
    st.subheader("Calibration and validation")
    st.caption(
        "We validate with GroupKFold by chain and a temporal check (train on "
        "the 2018 landmark, test on 2019). The 24-month window at the 2018 "
        "landmark stays clear of the COVID shock after early 2020; the 2019 "
        "landmark is our sensitivity check."
    )
    calib = data._cached_read(str(data.config.ARTIFACTS_DIR / "calibration.parquet"))
    if calib is not None and {"mean_predicted", "mean_observed"}.issubset(calib.columns):
        charts.calibration_plot(st, calib["mean_predicted"], calib["mean_observed"])

    # --- Sentiment benchmark + transfer ---
    st.subheader("Sentiment benchmark and transfer test")
    bench = data.load("sentiment_benchmark")
    if bench is not None:
        cols = [
            c
            for c in [
                "model",
                "features",
                "accuracy",
                "macro_f1",
                "macro_f1_lo",
                "macro_f1_hi",
                "roc_auc",
                "n_test",
            ]
            if c in bench.columns
        ]
        st.dataframe(
            bench[cols].sort_values("macro_f1", ascending=False).round(4),
            hide_index=True,
            use_container_width=True,
        )
    transfer = data.load("transfer_eval")
    if transfer is not None:
        st.caption(
            "Trained on non-cohort reviews, evaluated on a held-out non-cohort "
            "set and on all cohort reviews (transfer)."
        )
        st.dataframe(transfer.round(4), hide_index=True, use_container_width=True)

    # --- Agent evaluation ---
    st.subheader("Agent evaluation")
    agent = data.load("agent_eval")
    if agent is not None:
        st.caption(
            "Numeric fidelity is an independent re-scan: every number in a "
            "rendered brief must trace to the Python fact sheet."
        )
        st.dataframe(agent.round(4), hide_index=True, use_container_width=True)
    else:
        st.info(data.missing_message("agent_eval"))
