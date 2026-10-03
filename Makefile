# Location Risk Radar developer tasks.
# Usage: make <target>. On Windows without make, run the commands directly.

.PHONY: setup test lint format pipeline app audit metrics clean

PY ?= python

setup:  ## Install runtime and dev dependencies, and pre-commit hooks
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:  ## Lint and format-check with ruff
	ruff check src app pipeline tests scripts
	ruff format --check src app pipeline tests scripts

format:  ## Auto-format with ruff
	ruff format src app pipeline tests scripts
	ruff check --fix src app pipeline tests scripts

test:  ## Run the test suite with coverage (fails under 80% for src/lrr)
	$(PY) -m pytest --cov=src/lrr --cov-report=term-missing --cov-fail-under=80

app:  ## Run the Streamlit app locally
	streamlit run app/streamlit_app.py

pipeline:  ## Run the offline pipeline in order (requires YELP_DIR and the dataset)
	$(PY) pipeline/01_ingest_cohort.py
	$(PY) pipeline/01b_corpus_ingest.py
	$(PY) pipeline/02_text_sentiment.py
	$(PY) pipeline/02c_corpus_sentiment.py
	$(PY) pipeline/04_features.py
	$(PY) pipeline/04b_tier1_features.py
	$(PY) pipeline/05_survival.py
	$(PY) pipeline/05b_tier1_survival.py
	$(PY) pipeline/05c_tier2_stack.py
	$(PY) pipeline/06_aspects_rag.py
	$(PY) pipeline/08_eval_agents.py
	$(PY) scripts/generate_metrics_table.py

metrics:  ## Regenerate the headline metrics table from artifacts
	$(PY) scripts/generate_metrics_table.py

audit:  ## Secret scan of the working tree (gitleaks if installed, else a note)
	@gitleaks detect --source . --config .gitleaks.toml --no-banner || \
	 echo "gitleaks not installed; see .github/workflows/ci.yml for the CI scan"

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
