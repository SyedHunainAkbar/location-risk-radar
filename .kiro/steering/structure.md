# Project Structure

## Package: `src/lrr/`

Importable library code. One concern per module.

- `config.py` — all paths, constants, `SEED = 42`
- `io.py` — read/write artifacts and raw data
- `cohort.py` — chain name normalization, cohort selection, Fast Food labeling
- `text.py` — text cleaning and preprocessing
- `sentiment.py` — VADER and related sentiment features
- `topics.py` — BERTopic complaint themes
- `features.py` — engagement and feature engineering for the model
- `survival.py` — survival/hazard models
- `llm.py` — OpenAI-compatible client and provider selection
- `aspects.py` — aspect/label extraction with grounding checks
- `rag.py` — retrieval and evidence assembly
- `evaluation.py` — metrics, bootstrap CIs, baselines, ablations

## `pipeline/`

Offline scripts, run in order, each idempotent with CLI args:

- `01_ingest_cohort.py` ... `07_build_app_artifacts.py`

## `notebooks/`

- `Location_Risk_Radar_Final.ipynb`
- `LA5_Akbar_SyedHunain.ipynb`

## `app/`

- `streamlit_app.py`
- `app/pages/`

## Other

- `artifacts/` — committed small precomputed outputs the app reads
- `data/` — raw data, gitignored
- `tests/` — pytest suite
- `.streamlit/config.toml` — Streamlit config

## Rules

- The app reads only from `artifacts/`. It never touches `data/` or runs heavy compute.
- Pipeline scripts write to `artifacts/`. Library code in `src/lrr/` holds the logic the scripts and app share.
