"""Risk Committee: the three-agent Location Risk Brief with live or cached evidence."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data, state
from app.components import widgets


def render() -> None:
    st.header("Risk Committee")
    st.caption(
        "Three agents write the brief: a Quant Analyst, a Voice of Customer, "
        "and a different-family Risk Auditor. Python orchestrates them."
    )

    risk = data.require("risk_scores")
    briefs = data.load("briefs")
    if risk is None and briefs is None:
        return

    # Picker defaults to the highest-risk location in the current filter.
    scope = state.apply_filters(data.locations(scored_only=True)) if risk is not None else None
    if scope is not None and briefs is not None:
        # Locations with a precomputed brief come first so Cached mode always works.
        scope = scope.assign(_has=scope["business_id"].isin(set(briefs["business_id"])))
    if scope is not None and len(scope):
        score_col = next(
            (c for c in ("final_score", "tier2_score", "risk_score") if c in scope.columns), None
        )
        if score_col:
            by = (["_has"] if "_has" in scope else []) + [score_col]
            scope = scope.sort_values(by, ascending=False)
        ids = scope["business_id"].tolist()
        has = set(briefs["business_id"]) if briefs is not None else set()
        labels = {
            b: ("\u2605 " if b in has else "") + f"{lab} | {t}"
            for b, lab, t in zip(scope["business_id"], scope["location_label"], scope["risk_tier"])
        }
        st.caption("\u2605 = precomputed committee brief available in Cached mode.")
    else:
        ids = briefs["business_id"].tolist() if briefs is not None else []
        labels = data.label_map(data.locations())
    if not ids:
        st.info("No locations match the current filters.")
        return
    bid = st.selectbox("Location", ids, format_func=lambda b: labels.get(b, b))

    # Three agent cards.
    st.subheader("Committee")
    cols = st.columns(3)
    widgets.agent_card(cols[0], "Quant Analyst", "waiting")
    widgets.agent_card(cols[1], "Voice of Customer", "waiting")
    widgets.agent_card(cols[2], "Risk Auditor", "waiting")

    live = state.is_live()
    if live:
        if st.button("Convene committee", type="primary"):
            _run_live(bid, risk)
            return
        st.caption(
            "Live mode: click to convene the committee. Cached briefs show below until then."
        )
    else:
        st.caption("Cached mode: the precomputed brief loads instantly.")

    _render_cached(bid, briefs)


def _render_cached(bid: str, briefs) -> None:
    if briefs is None or bid not in set(briefs["business_id"]):
        st.info(data.missing_message("briefs"))
        return
    row = briefs[briefs["business_id"] == bid].iloc[0]
    st.markdown(widgets.risk_badge(str(row.get("tier", "Unknown"))), unsafe_allow_html=True)
    k = st.columns(3)
    from lrr.agents import CONFIDENCE_FORMULA

    k[0].metric(
        "Confidence",
        f"{row.get('confidence', float('nan')):.2f}",
        row.get("confidence_label", ""),
        help=CONFIDENCE_FORMULA,
    )
    k[1].metric("Groundedness", f"{row.get('groundedness', float('nan')):.2f}")
    k[2].metric("Auditor grade", str(row.get("auditor_grade", "")))

    st.subheader("Why we flagged it")
    for d in str(row.get("drivers", "")).split(" | "):
        if d.strip():
            st.markdown(f"- {d.strip()}")

    st.subheader("Evidence")
    ev = str(row.get("evidence", "")).strip()
    if ev and ev != "nan":
        for e in ev.split(" | "):
            if e.strip():
                st.markdown(f"- {e.strip()}")
    else:
        st.caption("No grounded evidence stored for this cached brief.")

    st.subheader("Recommended actions")
    for a in str(row.get("actions", "")).split(" | "):
        if a.strip():
            st.markdown(f"- {a.strip()}")

    n_rej = int(row.get("n_rejected_claims", 0) or 0)
    with st.expander(f"Audit trail ({n_rej} claim(s) removed as unsupported)"):
        st.caption(f"Confidence formula: {CONFIDENCE_FORMULA}")
        st.write(
            f"The auditor removed {n_rej} unsupported claim(s) before this brief. "
            f"Groundedness rate {row.get('groundedness', float('nan')):.2f}."
        )


def _run_live(bid: str, risk) -> None:
    import asyncio

    from lrr import agents
    from lrr.agents import auditor as auditor_mod
    from lrr.agents import quant as quant_mod
    from lrr.agents import voc as voc_mod

    features = data._cached_read(
        str(data.config.DATA_DIR / data.config.features_file(data.config.LANDMARK_PRIMARY))
    )
    shap = data._cached_read(str(data.config.ARTIFACTS_DIR / "shap_drivers.parquet"))
    fs = agents.build_factsheet(bid, risk, shap_drivers=shap, features=features)
    snippets = _retrieve(bid)

    with st.status("Convening the committee ...", expanded=True) as status:
        st.write("Agents 1 and 2 running in parallel ...")

        async def _parallel():
            return await asyncio.gather(
                asyncio.to_thread(
                    quant_mod.run_quant, fs, data.config.QUANT_PLACEHOLDER_RETRIES, None
                ),
                asyncio.to_thread(voc_mod.run_voc, snippets, None),
            )

        try:
            quant_parsed, voc_result = asyncio.run(_parallel())
            st.write("Quant Analyst and Voice of Customer done. Auditor checking ...")
            rendered = quant_mod.render_drivers(quant_parsed, fs)
            claims = auditor_mod.build_claims(rendered, voc_result)
            audit = auditor_mod.run_auditor(claims, voc_result, {"review_volume": "unknown"})
            status.update(label="Committee complete.", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Committee failed; showing cached brief.", state="error")
            st.warning(f"A provider call failed ({type(exc).__name__}).")
            _render_cached(bid, data.load("briefs"))
            return

    _render_live_brief(fs, rendered, voc_result, claims, audit, snippets)


def _retrieve(bid: str):
    try:
        from lrr import rag

        emb_path = data.config.RAG_INDEX_DIR / data.config.RAG_EMBEDDINGS_FILE
        qvecs = data.encode_query(["Why is this location at risk?"])
        if emb_path.exists() and qvecs is not None:
            emb, meta = rag.load_index(data.config.RAG_INDEX_DIR)
            got = rag.retrieve(
                qvecs[0],
                emb,
                meta,
                business_id=bid,
                k=data.config.RAG_TOP_K,
                use_mmr=True,
                max_stars=2,
            )
            return got
    except Exception:
        pass
    return pd.DataFrame(columns=["review_id", "text", "stars", "date"])


def _render_live_brief(fs, rendered, voc_result, claims, audit, snippets) -> None:
    from lrr import agents
    from lrr.agents import auditor as auditor_mod

    supported = auditor_mod.supported_claim_ids(audit)
    gr = auditor_mod.groundedness_rate(audit)
    confidence = agents.compute_confidence(gr, len(voc_result["items"]), audit.confidence_grade)

    st.markdown(widgets.risk_badge(str(fs.get("risk_tier", "Unknown"))), unsafe_allow_html=True)
    k = st.columns(3)
    k[0].metric(
        "Confidence",
        f"{confidence:.2f}",
        agents.confidence_label(confidence),
        help=agents.CONFIDENCE_FORMULA,
    )
    k[1].metric("Groundedness", f"{gr:.2f}")
    k[2].metric("Auditor grade", audit.confidence_grade)

    st.subheader("Why we flagged it")
    for i, d in enumerate(rendered):
        if f"driver_{i}" in supported or not audit.claims:
            st.markdown(f"- {d['text']}")
        else:
            st.markdown(
                f"- <span class='lrr-strike'>{d['text']}</span> "
                f"<span class='lrr-muted'>(removed: unsupported)</span>",
                unsafe_allow_html=True,
            )

    st.subheader("Evidence")
    by_id = {}
    if snippets is not None and "review_id" in snippets:
        by_id = {r["review_id"]: r for _, r in snippets.iterrows()}
    for it in voc_result["items"]:
        src = by_id.get(it.review_id, {})
        widgets.citation_card(
            st, it.quote, src.get("text", it.quote), src.get("stars"), src.get("date"), it.review_id
        )

    st.subheader("Recommended actions")
    from lrr.agents import playbook

    for a in playbook.recommend(voc_result.get("issue_prevalence", {})):
        st.markdown(f"- {a}")

    # Numeric fidelity check on the rendered drivers.
    stray = []
    for d in rendered:
        stray += agents.scan_rendered_fidelity(d["text"], fs)
    fidelity_ok = not stray

    with st.expander("Audit trail"):
        st.caption(f"Confidence formula: {agents.CONFIDENCE_FORMULA}")
        st.write(
            f"Groundedness rate: {gr:.2f} | Numeric fidelity: {'PASS' if fidelity_ok else 'CHECK'}"
        )
        for c in audit.claims:
            text = next((cl["text"] for cl in claims if cl["claim_id"] == c.claim_id), c.claim_id)
            if c.verdict == "supported":
                st.markdown(f"- ✅ {text}  _(supported)_")
            elif c.verdict == "unsupported":
                st.markdown(
                    f"- ❌ <span class='lrr-strike'>{text}</span>  _(unsupported: {c.reason})_",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"- ⚠️ {text}  _(insufficient)_")
        if audit.alternatives:
            st.write("Alternative explanations considered:")
            for alt in audit.alternatives:
                st.markdown(f"- {alt.explanation} ({alt.verdict})")
