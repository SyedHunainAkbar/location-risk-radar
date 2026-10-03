# Tasks: Streamlit App

- [ ] 1. `.streamlit/config.toml` light theme + `app/components/theme.py`
  - Accent color, colorblind-safe tier palette, Plotly template, card CSS.
  - _Requirements: 1.3, 1.4, 6.1_

- [ ] 2. `app/components/widgets.py` + `app/components/charts.py`
  - kpi_tile, risk_badge, citation_card, agent_card; chart helpers with takeaway
    titles and plotly-lazy fallback.
  - _Requirements: 1.2, 6.1, 6.3_

- [ ] 3. `app/data.py`
  - Cached loaders, schema contracts, friendly missing-file messages.
  - _Requirements: 2.1, 2.2_

- [ ] 4. `app/state.py` + `app/sidebar.py`
  - Session filters + live-call counter/cap; LLM Gateway panel; guarded_chat.
  - _Requirements: 2.3, 2.4, 3.1, 3.2, 5.1, 5.2_

- [ ] 5. `app/streamlit_app.py` + Portfolio Radar + Location Deep Dive
  - st.navigation, header, tagline; the first two pages.
  - _Requirements: 1.1, 4.1, 4.2_

- [ ] 6. Risk Committee + Complaint Themes + Ask the Reviews
  - Agent cards, cached brief, audit trail; topics; sanitized chat + judge badge.
  - _Requirements: 4.3, 4.4, 4.5, 5.1, 5.2_

- [ ] 7. Model Lab + Methodology
  - Phase A/B metrics with CIs, ablation, calibration, agent eval; diagrams,
    traceability, limitations, ethics, disclosure.
  - _Requirements: 4.6, 4.7_

- [ ] 8. Tests + performance
  - AppTest per-page no-exception, cached brief, filter row counts, missing message;
    peak-memory < 800 MB.
  - _Requirements: 7.1, 7.2_

- [ ] 9. Run pytest, fix failures, describe each page
  - _Requirements: 7.3_
