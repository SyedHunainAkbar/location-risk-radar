"""Methodology and Limitations: framework, diagrams, traceability, limits, ethics."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.header("Methodology and Limitations")
    st.caption(
        "How the system is built, what each lab objective maps to, and where "
        "we are careful about what the model can and cannot say."
    )

    st.subheader("Framework")
    st.markdown(
        "We combine a survival model, BERTopic complaint themes, a retrieval "
        "assistant, and a three-agent Risk Committee, all behind one LLM gateway."
    )
    st.markdown(
        """
```mermaid
flowchart LR
  D[Yelp corpus] --> S[Survival model: who is at risk]
  D --> T[BERTopic: what customers complain about]
  S --> C[Risk Committee]
  T --> C
  D --> R[RAG evidence assistant]
  R --> C
  C --> B[Location Risk Brief]
```
        """
    )

    st.subheader("Landmark design")
    st.markdown(
        """
```mermaid
flowchart LR
  A[first review] --> L[T minus 6 months]
  L --> T[T landmark]
  T --> H[T plus 24 months]
  A -. features use only data before T .-> T
  T -. closure in this window counts as an event .-> H
```
We freeze a landmark T (2018-01-01 primary, 2019-01-01 sensitivity), build every
feature from records dated strictly before T, and observe closure over the next 24
months. We proxy the closure date with last activity because the data gives a status
flag, not a closure date.
        """
    )

    st.subheader("LLM gateway")
    st.markdown(
        """
```mermaid
flowchart LR
  CA[Caller: topics, aspects, rag, agents] --> G[Gateway]
  G --> RT[Role routing]
  RT --> P1[openai]
  RT --> P2[nvidia]
  RT --> P3[voyager]
  RT --> M[mock terminal fallback]
  P1 --> Q[retry, JSON repair, cache, ledger]
  P2 --> Q
  P3 --> Q
  M --> Q
```
Every model call routes through the gateway, which checks provider health, retries
transient failures, repairs JSON, caches responses, and records a usage ledger. Keys
are read only from the environment or secrets and are never shown or logged.
        """
    )

    st.subheader("Lab objectives traceability")
    trace = pd.DataFrame(
        [
            {
                "objective": "LA1 spaCy normalization and lexicons",
                "app_page": "Location Deep Dive, Complaint Themes",
                "notebook_section": "3. Text processing",
            },
            {
                "objective": "LA2 classical sentiment (TF-IDF + calibration)",
                "app_page": "Model Lab",
                "notebook_section": "4. Sentiment modeling",
            },
            {
                "objective": "LA3 deep sentiment (GRU/LSTM + GloVe)",
                "app_page": "Model Lab",
                "notebook_section": "4. Sentiment modeling",
            },
            {
                "objective": "LA4 BERTopic complaint themes",
                "app_page": "Complaint Themes",
                "notebook_section": "5. Complaint themes",
            },
            {
                "objective": "Survival / hazard model (two-tier)",
                "app_page": "Portfolio Radar, Location Deep Dive, Model Lab",
                "notebook_section": "6. Survival model",
            },
            {
                "objective": "LA5 LLM layer: gateway, aspects, RAG, Risk Committee",
                "app_page": "Risk Committee, Ask the Reviews",
                "notebook_section": "7. LLM layer",
            },
        ]
    )
    st.dataframe(trace, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown(
        """
- No `chain_id` in the data; chains are grouped by name normalization, which is
  imperfect.
- Fast Food label noise: sit-down brands such as Chili's, Applebee's, and Denny's
  appear in the Fast Food cluster.
- `is_open` is a current status flag, not a closure date.
- Closure date is proxied by last observed activity, not a confirmed close date.
- Right censoring: locations still open at the end of the window have unknown
  eventual outcomes.
- Survivorship in text: reviews from closed locations stop when the location stops,
  biasing late-period language.
        """
    )

    st.subheader("Ethical use")
    st.markdown(
        "This is decision support, not an automated closure decision. We surface "
        "locations that merit a closer look, with the evidence attached, for a human "
        "to review. The tool should never be used to justify adverse employment "
        "action, and every flag is explained by cited customer text."
    )

    st.subheader("Team and disclosure")
    st.markdown(
        """
**Team LRR Analytics:** Andy Lin, Hunain Akbar, Ishani Patel, Jake Ida, Yukti Gandhi.

**Generative AI disclosure.** We used Kiro to scaffold the package and pipelines, and
ChatGPT, Gemini, and Claude for targeted prompt design, debugging, and wording. All
numbers in the app are computed by our Python code from the artifacts; the LLM layer
returns only quotes and labels.

Links: GitHub repository and the final notebook are referenced in the project README.
        """
    )
