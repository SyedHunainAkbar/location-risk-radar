# Requirements: Streamlit App

## Introduction

The production Streamlit app for Location Risk Radar. It reads only precomputed files
in `artifacts/` and calls LLMs only through `lrr.gateway`. It must run within 1 GB RAM
on Streamlit Community Cloud, load every page in under 3 seconds from cache, and work
fully with no API keys (Cached mode) and with keys (Live mode). Copy is concise, first
person plural, no em dashes.

## Requirements

### Requirement 1: Entry, structure, and theme

#### Acceptance Criteria
1. `app/streamlit_app.py` SHALL use `st.navigation` with the seven pages, set a global
   `st.set_page_config` (wide layout, title "Location Risk Radar", icon), and show a
   shared header with the tagline "Early warning for restaurant locations, explained by
   customer evidence".
2. `app/components/` SHALL hold reusable `kpi_tile`, `risk_badge`, `citation_card`,
   `agent_card`, chart helpers, and a single theme module with the palette and a Plotly
   template.
3. `.streamlit/config.toml` SHALL set a clean light theme with one accent color.
4. The risk-tier palette (High, Elevated, Watch, Low) SHALL be colorblind-safe and used
   identically everywhere. No default rainbow colors.

### Requirement 2: Data loading and robustness

#### Acceptance Criteria
1. `app/data.py` SHALL provide `st.cache_data` loaders for every artifact with schema
   validation.
2. A missing file SHALL show a clear message naming the pipeline script that creates
   it, never a stack trace.
3. Gateway calls SHALL be wrapped with friendly fallbacks to cached results and a
   visible notice naming the provider that failed. The app SHALL never crash.
4. The app SHALL never display or log API keys.

### Requirement 3: Sidebar on every page

#### Acceptance Criteria
1. The sidebar SHALL hold global filters (cluster, chain, state, risk tier) kept in
   `st.session_state`.
2. The sidebar SHALL hold an "LLM Gateway" panel showing provider health (OpenAI,
   NVIDIA, Voyager, Mock), a Live/Cached switch (Live disabled when no keys), a session
   request counter with a hard cap (default 25 live calls per session), and the token
   and cost total from the gateway ledger.

### Requirement 4: Pages

#### Acceptance Criteria
1. **Portfolio Radar**: KPI tiles (cohort locations, High count, Elevated count, median
   risk by cluster, Tier 1 base closure rate within 24 months); a pydeck map colored by
   tier with hover; a sortable table (location, chain, cluster, tier, final_score,
   percentile within chain, top-3 plain-English drivers, 24-month survival) with CSV
   download; a risk-distribution-by-cluster chart with Tier 1 vs Tier 2 comparison.
2. **Location Deep Dive**: location picker (search by chain and city); risk score,
   percentile vs chain and cluster, tier badge; survival curve with 95% band; monthly
   review volume, sentiment trajectory, and check-ins with the landmark marked; top-5
   SHAP drivers as a signed bar chart with plain-English labels; Voice-of-Customer issue
   scores and top complaint topics vs chain peers; food/service/ambience aspect scores
   labeled as TA-adapted.
3. **Risk Committee**: location picker defaulting to the highest-risk location in the
   current filter; three agent cards (Quant Analyst, Voice of Customer, Risk Auditor)
   each showing the provider/model badge, latency, and cache status; Live mode has a
   "Convene committee" button running agents 1 and 2 in parallel then the auditor with
   `st.status` progress; Cached mode loads `artifacts/briefs.parquet` instantly; the
   brief shows tier, rendered drivers, grounded evidence as citation cards, a confidence
   grade with the formula in a tooltip, and recommended actions; an "Audit trail"
   expander shows unsupported claims struck through with reasons, alternatives,
   groundedness rate, and the numeric-fidelity check result.
4. **Complaint Themes**: topics per cluster with human labels, topics over time by
   quarter, topic share High vs Low with difference and CI, zero-shot vs unsupervised,
   representative quotes per topic.
5. **Ask the Reviews**: chat scoped to a location or chain with stars and date filters;
   answers cite `[review_id]` for every claim with expandable source snippets; the judge
   verdict appears as a groundedness badge; unsupported sentences hidden behind a "show
   removed" toggle; Cached mode offers the six precomputed questions as buttons;
   out-of-scope questions get a polite "not enough evidence" refusal.
6. **Model Lab**: Phase A (Tier 1) vs Phase B (Tier 2) metrics side by side (Harrell and
   Uno C, time-dependent AUC at 12 and 24 months, integrated Brier), all with 95% CIs;
   an ablation chart (stars only vs engagement vs engagement+text vs full); calibration;
   temporal validation; COVID sensitivity; the sentiment benchmark with transfer test;
   agent evaluation (numeric fidelity, groundedness, committee vs single-agent, auditor
   vs human kappa, latency and cost per brief).
7. **Methodology and Limitations**: framework, landmark, and gateway diagrams; a lab
   traceability table (LA1 to LA5 mapped to page and notebook section); all limitations
   from product.md; an ethical-use statement; team members; GenAI disclosure; links to
   the repo and notebook.

### Requirement 5: Chat input sanitization

#### Acceptance Criteria
1. Chat input SHALL be length-capped.
2. Retrieved review text SHALL be kept in a delimited data block, and the model SHALL
   be instructed to treat it as data, not instructions, to resist prompt injection such
   as "ignore previous instructions".

### Requirement 6: Design and accessibility

#### Acceptance Criteria
1. Every chart title SHALL state its takeaway in one sentence.
2. No em dashes in any UI text.
3. The layout SHALL be mobile tolerant (columns collapse), with accessible contrast and
   units on every axis.

### Requirement 7: Tests and performance

#### Acceptance Criteria
1. `tests/test_app_smoke.py` using `streamlit.testing.v1.AppTest` SHALL verify every
   page loads with no exceptions in Cached mode using the mock provider, the Risk
   Committee page renders a cached brief, filters change table row counts, and a missing
   artifact shows the friendly message.
2. A test SHALL measure peak memory when loading all pages and assert under 800 MB.
3. `pytest` SHALL pass.
