"""Complaint Themes: BERTopic results per cluster, over time, and by outcome."""

from __future__ import annotations

import streamlit as st

from app import data
from app.components import charts


def render() -> None:
    st.header("Complaint Themes")
    st.caption(
        "We model negative reviews per cluster to show what customers complain "
        "about, how it shifts over time, and how it differs by outcome."
    )

    any_found = False
    for cluster in ("Fast Food", "Non-Fast Food"):
        topics = data.topics_artifact(cluster)
        if topics is None:
            continue
        any_found = True
        st.subheader(cluster)
        if {"label", "size"}.issubset(topics.columns):
            top = topics.sort_values("size", ascending=False).head(12)
            charts.topic_bar(st, top["label"], top["size"], f"{cluster}: largest complaint topics")
        st.dataframe(topics.head(15), hide_index=True, use_container_width=True)

        slug = cluster.lower().replace(" ", "_").replace("-", "_")
        ot = data._cached_read(str(data.config.ARTIFACTS_DIR / f"topics_over_time_{slug}.parquet"))
        if ot is not None and {"quarter", "topic", "count"}.issubset(ot.columns):
            st.caption("Topic volume by quarter: complaint mix shifts over time.")
            pivot = ot.pivot_table(index="quarter", columns="topic", values="count", fill_value=0)
            st.line_chart(pivot)

        co = data._cached_read(
            str(data.config.ARTIFACTS_DIR / f"topics_closed_open_{slug}.parquet")
        )
        if co is not None and {"closed_open", "topic", "count"}.issubset(co.columns):
            st.caption("Topic counts for closed versus open locations.")
            st.dataframe(co.head(20), hide_index=True, use_container_width=True)

    if not any_found:
        st.info(
            data.missing_message("topic_shares").replace(
                "topic_shares.parquet", "topics_{cluster}.parquet"
            )
        )

    st.subheader("High vs low risk, and zero-shot vs unsupervised")
    st.caption(
        "Topic-share differences between high- and low-risk locations, with "
        "the zero-shot seeded model compared against the unsupervised model, "
        "are produced by pipeline/03_topics_colab.py and shown here when "
        "available."
    )
