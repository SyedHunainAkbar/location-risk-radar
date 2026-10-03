# Design: Streamlit App

## Overview

A multipage Streamlit app under `app/`, wired with `st.navigation`. It reads only the
small `artifacts/` parquet files through cached loaders and calls models only through
`lrr.gateway`. Everything degrades gracefully: a missing artifact shows a friendly
message naming the pipeline that creates it, and a failed gateway call falls back to
cached results with a visible notice. No key is ever displayed or logged.

Local environment note: `plotly` and `pandera` are not guaranteed present, so charts
import plotly lazily and fall back to native Streamlit charts, and schema validation
uses a lightweight column/dtype contract (pydantic-free) that never raises to the UI.

## Layout

```
app/
  streamlit_app.py        # st.navigation entry, set_page_config, shared header
  data.py                 # cached loaders + schema contracts + friendly messages
  sidebar.py              # global filters + LLM Gateway panel (shared on every page)
  state.py                # session defaults, live-call counter + hard cap
  components/
    theme.py              # palette, tier colors, plotly template, CSS
    widgets.py            # kpi_tile, risk_badge, citation_card, agent_card
    charts.py             # chart helpers (plotly lazy, native fallback)
  pages/
    portfolio_radar.py
    location_deep_dive.py
    risk_committee.py
    complaint_themes.py
    ask_the_reviews.py
    model_lab.py
    methodology.py
```

Pages are plain modules exposing a `render()` function; `streamlit_app.py` wraps each
in `st.Page` via a tiny shim so `AppTest` can run the entry file directly.

## data.py

```python
ARTIFACTS = {name: (filename, base, required_columns)}
@st.cache_data loaders: load(name) -> DataFrame | None
def require(name) -> DataFrame | None      # shows friendly message if missing
def missing_message(name) -> str           # names the pipeline script
def validate(df, name) -> df               # keep known columns, coerce, never raise
```

Each artifact entry records the filename, whether it lives in `artifacts/` or `data/`,
the pipeline script that produces it, and the minimal required columns. `require`
returns the frame or renders `st.info(...)` naming the script (for example
"Run pipeline/05_survival.py to produce risk_scores.parquet") and returns None.

## sidebar.py and state.py

`state.py` seeds `st.session_state` with filter defaults, the mode, a `live_calls`
counter, and `LIVE_CALL_CAP = 25`. `sidebar.render()` draws:

- Global filters: cluster, chain, state, risk tier (multiselects seeded from the
  loaded cohort). Selections persist in session state and every page reads them via
  `state.apply_filters(df)`.
- LLM Gateway panel: provider health dots (OpenAI, NVIDIA, Voyager, Mock); a
  Live/Cached radio with Live disabled when no key is present; the session request
  counter `used / cap` with Live auto-forced to Cached at the cap; token and cost
  totals from the gateway ledger.

`guarded_chat(...)` centralizes every live call: it checks the cap, increments the
counter, calls `lrr.gateway.chat`, and on any exception returns a cached fallback with
a visible provider-failed notice.

## components

- `theme.py`: `ACCENT`, `TIER_COLORS` (colorblind-safe: High `#B2182B`, Elevated
  `#EF8A62`, Watch `#F7C948`, Low `#2166AC`), a Plotly template builder, and a small
  CSS string for cards. One palette, used everywhere.
- `widgets.py`: `kpi_tile(label, value, help)`, `risk_badge(tier)`,
  `citation_card(quote, review, stars, date, review_id)` (quote highlighted inside the
  source review), `agent_card(name, status, provider, model, latency, cache)`.
- `charts.py`: `risk_distribution`, `survival_curve`, `shap_bar`, `topic_bar`,
  `ablation_bar`, `calibration_plot`. Each sets a one-sentence takeaway title, labels
  axes with units, and uses the tier palette. Plotly is imported inside each function;
  if unavailable the helper falls back to `st.bar_chart`/`st.line_chart`.

## Pages

Each page: read filters, pull the artifacts it needs through `data.require`, render.
All numbers come from artifacts; the only live computation is the Risk Committee's
optional "Convene committee" run and Ask the Reviews chat, both via `guarded_chat`.

- **Risk Committee** reuses `lrr.agents` for the live path (agents 1 and 2 via
  `asyncio.gather`, then the auditor) with `st.status` per agent, and
  `artifacts/briefs.parquet` for the cached path. Evidence renders as citation cards;
  unsupported claims are struck through in the audit-trail expander; confidence shows
  the formula in a help tooltip.
- **Ask the Reviews** builds a delimited data block from retrieved snippets and
  instructs the model to treat it as data; input is length-capped; the judge verdict
  drives a groundedness badge and the "show removed" toggle.

## Testing (tests/test_app_smoke.py)

`streamlit.testing.v1.AppTest.from_file("app/streamlit_app.py")`:

- Each page renders with no exception in Cached mode (mock provider, no keys).
- The Risk Committee page shows a cached brief when `briefs.parquet` exists (a tiny
  fixture is written to a temp artifacts dir the loaders point at).
- Changing a filter changes the Portfolio table row count.
- A missing artifact yields the friendly message, not an exception.
- A memory test loads all pages and asserts peak RSS under 800 MB via `psutil` /
  `tracemalloc`.

Tests inject a temp artifacts directory (monkeypatching `config.ARTIFACTS_DIR` and the
cache) with small synthetic parquet fixtures so they run offline and fast.
