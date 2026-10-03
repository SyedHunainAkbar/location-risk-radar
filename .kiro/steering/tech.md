# Tech Stack

## Runtime

Python 3.11.

## Libraries

- Data: pandas, numpy, pyarrow
- ML: scikit-learn, xgboost, shap
- NLP: spacy (`en_core_web_sm` in the app, `en_core_web_lg` offline only), vaderSentiment, sentence-transformers (`all-MiniLM-L6-v2`)
- Topics: bertopic, umap-learn, hdbscan
- Survival: lifelines, scikit-survival
- Deep learning: tensorflow/keras only in offline notebooks, never in the app
- LLM: openai client
- Matching: rapidfuzz
- Viz and app: plotly, streamlit
- Testing: pytest

## LLM Access

Use an OpenAI-compatible client. Providers:

- `nvidia`: base_url `https://integrate.api.nvidia.com/v1`. Default in the deployed app.
- `voyager`: base_url `https://openai.rc.asu.edu/v1`. Local only, requires ASU VPN.
- `mock`: offline, deterministic. Use for tests and when no network is available.

Keys come only from environment variables or `st.secrets`. Never hard-code keys and never commit them.

## LLM Discipline

- LLMs never output numeric scores. They return quotes and labels only; Python computes every number.
- Every quote must be grounded in the source text: rapidfuzz partial ratio greater than or equal to 85, and at least 2 words.
- Ungrounded quotes are dropped and counted. We report the drop count.

## Reproducibility

- Global `SEED = 42`. Seed all randomness.
- All paths come from `src/lrr/config.py`. No hard-coded paths elsewhere.

## Compute Boundary

- Heavy compute happens offline in `pipeline/` scripts.
- The Streamlit app only reads precomputed artifacts in `artifacts/`.
- Each artifact file is under 50 MB, stored as parquet or npy float16.
- The app must run in 1 GB RAM.
